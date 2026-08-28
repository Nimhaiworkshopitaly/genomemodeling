#!/usr/bin/env python3
"""Run from the genomemodeling root: make-jobs, run, or plot."""
import argparse
import csv
import shlex
import sys
from pathlib import Path

SEEDS = [42, 1042, 2042, 3042, 4042]
FRACTIONS = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
LEGACY = Path('hpc_multicpu/results_inversion_pairwise_25pct')
RESULTS = Path('hpc_multicpu/results_inversion_pairwise_all')
FIGURES = Path('hpc_multicpu/figures_inversion_pairwise_all')


def existing_result(fraction, seed):
    """Validate saved jobs before reusing them; never overwrite older results."""
    import numpy as np
    import pandas as pd
    name = f'pairs_if_{fraction:g}_seed_{seed}.csv'
    paths = [directory / name for directory in [LEGACY, RESULTS]
             if (directory / name).exists()]
    if len(paths) > 1:
        raise ValueError(f'Duplicate result locations for {name}: {paths}')
    if not paths:
        return None
    path = paths[0]
    data = pd.read_csv(path, dtype={'pair_a': str, 'pair_b': str})
    expected = dict(seed=seed, inversion_fraction=fraction, rf=0.1,
                    total_rearrangement_rate=0.316227766017,
                    n_runs=100, workers=16, core_fraction=0.5,
                    core_protection=0.9, translocation_exp=1e9, gain_loss_exp=1e9)
    if len(data) != 28 or data[['pair_a', 'pair_b']].isna().any().any():
        raise ValueError(f'{path}: expected 28 complete pairs.')
    if data.duplicated(['pair_a', 'pair_b']).any():
        raise ValueError(f'{path}: duplicate pairs.')
    for column, value in expected.items():
        if not np.allclose(data[column], value, rtol=1e-10, atol=1e-12):
            raise ValueError(f'{path}: unexpected {column}.')
    for column, value in [('root_mode', 'median_synthetic'),
                          ('inversion_size_mode', 'uniform_breakpoints')]:
        if not data[column].eq(value).all():
            raise ValueError(f'{path}: unexpected {column}.')
    if not np.isfinite(data[['ks_statistic', 'kuiper_statistic']]).all().all():
        raise ValueError(f'{path}: nonfinite scores.')
    return path


def make_jobs():
    if not Path('hpc_multicpu/multicpu_eval_one_setting_inversion_fraction.py').is_file():
        raise SystemExit('Run from the genomemodeling repository root.')
    RESULTS.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    commands = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            if existing_result(fraction, seed) is not None:
                continue
            commands.append(shlex.join([
                sys.executable, str(script), 'run', '--fraction', str(fraction),
                '--seed', str(seed),
            ]))
    path = Path('hpc_multicpu/jobs_inversion_pairwise_all.swarm')
    path.write_text('\n'.join(commands) + ('\n' if commands else ''))
    print(f'Wrote {len(commands)} jobs: {path}')
    print(f'Reused {30-len(commands)} validated jobs. If zero jobs remain, run plot.')


def run(args):
    if existing_result(args.fraction, args.seed) is not None:
        print('Validated result already exists; skipping simulation.')
        return
    import numpy as np
    from multiprocessing import Pool
    sys.path.insert(0, str(Path.cwd() / 'hpc_multicpu'))
    from multicpu_eval_one_setting_inversion_fraction import (
        build_real_pmfs, make_root_genome, split_counts, worker_simulate,
        merge_counts, score_real_vs_sim_counts,
    )
    from prod_1b_core_composite import _distribution_distances

    tree_path = str(Path('ATGC0070/yuri_gl26/ATGC0070.gl.tre').resolve())
    cc_path = str(Path('ATGC0070/atgc.cc.csv').resolve())
    real, tree, genomes, _ = build_real_pmfs(tree_path, cc_path)
    root = make_root_genome('median_synthetic', tree, cc_path, real_genomes=genomes)
    total = 0.316227766017
    rng = np.random.default_rng(args.seed)
    payloads = []
    for count in split_counts(100, 16):
        seed = int(rng.integers(0, 2**32 - 1))
        payloads.append((tree, root, 0.1, total*(1-args.fraction),
                         total*args.fraction, 1e9, 3.0, 'uniform_breakpoints',
                         1e9, 0.5, 0.9, count, seed))
    with Pool(len(payloads)) as pool:
        counts = merge_counts(pool.map(worker_simulate, payloads))
    aggregate = score_real_vs_sim_counts(real, counts)
    rows = []
    for pair, (rv, rp) in sorted(real.items()):
        counter = counts.get(pair)
        if not counter or sum(counter.values()) <= 0:
            raise ValueError(f'Missing simulated blocks for {pair}')
        sv = np.array(sorted(counter), dtype=float)
        sp = np.array([counter[v] for v in sv], dtype=float)
        sp /= sp.sum()
        distances = _distribution_distances(rv, rp, sv, sp)
        rows.append(dict(pair_a=pair[0], pair_b=pair[1], seed=args.seed,
                         inversion_fraction=args.fraction,
                         ks_statistic=distances['ks_statistic'],
                         kuiper_statistic=distances['kuiper_statistic'],
                         rf=0.1, total_rearrangement_rate=total,
                         inversion_size_mode='uniform_breakpoints',
                         root_mode='median_synthetic', n_runs=100, workers=16,
                         core_fraction=0.5, core_protection=0.9,
                         translocation_exp=1e9, gain_loss_exp=1e9,
                         tree_path=tree_path, cc_path=cc_path))
    if len(rows) != 28:
        raise ValueError(f'Expected 28 ATGC0070 pairs, found {len(rows)}')
    for metric in ['ks_statistic', 'kuiper_statistic']:
        values = np.array([r[metric] for r in rows])
        if not np.isfinite(values).all() or not np.isclose(
            values.mean(), aggregate['avg_' + metric], rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f'Pair scores do not reproduce aggregate {metric}')
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f'pairs_if_{args.fraction:g}_seed_{args.seed}.csv'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    print(f'Saved {len(rows)} pairs: {path}', flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    paths = []
    missing = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            path = existing_result(fraction, seed)
            if path is None:
                missing.append((fraction, seed))
            else:
                paths.append(path)
    if missing:
        raise SystemExit(f'Missing fraction/seed jobs: {missing}')
    data = pd.concat([pd.read_csv(p, dtype={'pair_a': str, 'pair_b': str})
                      for p in paths], ignore_index=True)
    keys = ['pair_a', 'pair_b', 'seed']
    if data.duplicated(keys + ['inversion_fraction']).any():
        raise ValueError('Duplicate pair/seed/fraction rows.')
    metrics = ['ks_statistic', 'kuiper_statistic']
    if not np.isfinite(data[metrics]).all().all():
        raise ValueError('Nonfinite scores.')
    metadata = ['rf', 'total_rearrangement_rate', 'inversion_size_mode',
                'root_mode', 'n_runs', 'workers', 'core_fraction',
                'core_protection', 'translocation_exp', 'gain_loss_exp',
                'tree_path', 'cc_path']
    if any(data[c].nunique(dropna=False) != 1 or data[c].isna().any()
           for c in metadata):
        raise ValueError('Results mix experiment settings or have missing metadata.')
    pairs = data[['pair_a', 'pair_b']].drop_duplicates()
    expected = {(a, b, seed, fraction) for a, b in pairs.itertuples(index=False)
                for seed in SEEDS for fraction in FRACTIONS}
    actual = set(data[keys + ['inversion_fraction']].itertuples(index=False, name=None))
    if len(pairs) != 28 or actual != expected or len(data) != 840:
        raise ValueError('Expected 28 pairs x five seeds x six fractions.')
    baseline = data[data.inversion_fraction == 0].set_index(keys)[metrics]
    differences = []
    for fraction in FRACTIONS[1:]:
        inverted = data[data.inversion_fraction == fraction].set_index(keys)[metrics]
        difference = (baseline - inverted).reset_index()
        difference['inversion_fraction'] = fraction
        differences.append(difference)
    difference = pd.concat(differences, ignore_index=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURES / 'combined_pair_scores.csv', index=False)
    difference.to_csv(FIGURES / 'paired_seed_improvements.csv', index=False)
    summary = difference.groupby(['pair_a', 'pair_b', 'inversion_fraction'])[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(FIGURES / 'per_pair_improvements.csv')
    pair_order = sorted(set(zip(data.pair_a, data.pair_b)))
    names = [f'{a} vs {b}' for a, b in pair_order]
    y = np.arange(len(names))
    report = []
    for metric, label in zip(metrics, ['KS', 'Kuiper']):
        limit = max(float((summary[metric+'_mean'].abs() + summary[metric+'_std']).max())*1.1, 0.001)
        for fraction in FRACTIONS[1:]:
            ordered = summary.xs(fraction, level='inversion_fraction').reindex(pair_order)
            mean = ordered[metric + '_mean'].to_numpy()
            sd = ordered[metric + '_std'].to_numpy()
            pct = int(round(fraction*100))
            fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
            ax.barh(y, mean, xerr=sd, capsize=3, height=0.7,
                    color=['#299d8f' if v >= 0 else '#cf465a' for v in mean],
                    error_kw={'elinewidth': 1, 'ecolor': '#333333'})
            ax.axvline(0, color='black', linewidth=1)
            ax.set_yticks(y, names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(-limit, limit)
            ax.set_xlabel(f'{label} improvement: 0% minus {pct}% inversions\n'
                          'Positive favors inversions; error bars = SD of paired differences')
            ax.set_title(f'Per-genome-pair {label}: {pct}% inversions; five matched seeds\n'
                         'Uniform-breakpoint proposals; total rate = 0.316228; core protection = 0.9')
            ax.grid(axis='x', alpha=0.2)
            ax.set_axisbelow(True)
            for extension in ['png', 'pdf']:
                fig.savefig(FIGURES / f'per_pair_{label.lower()}_improvement_{pct}pct.{extension}', dpi=300)
            plt.close(fig)
            report.append(dict(metric=label, inversion_fraction=fraction,
                               mean_improvement=mean.mean(), positive_pairs=int((mean>0).sum()),
                               negative_pairs=int((mean<0).sum()), zero_pairs=int((mean==0).sum())))
            print(f'{label}, f={fraction:g}: mean improvement={mean.mean():+.6f}; '
                  f'positive pairs={(mean > 0).sum()}/28')
        matrix = summary[metric+'_mean'].unstack('inversion_fraction').reindex(index=pair_order, columns=FRACTIONS[1:])
        matrix.to_csv(FIGURES / f'{label.lower()}_heatmap_values.csv')
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('improvement', ['#cf465a', '#ffffff', '#299d8f'])
        bound = max(float(np.abs(matrix.to_numpy()).max()), 0.001)
        fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
        im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-bound, vmax=bound, aspect='auto')
        ax.set_yticks(y, names, fontsize=9)
        ax.set_xticks(np.arange(5), [f'{f*100:g}%' for f in FRACTIONS[1:]])
        ax.set_xlabel('Inversion fraction (compared with 0% baseline)')
        ax.set_title(f'{label} improvement: mean across five matched seeds\n'
                     'Uniform-breakpoint proposals; total rate = 0.316228; core protection = 0.9')
        for i in range(28):
            for j in range(5):
                value = matrix.iloc[i, j]
                ax.text(j, i, f'{value:+.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(value) > .75*bound else 'black')
        fig.colorbar(im, ax=ax, label='Baseline minus inversion score; positive favors inversions', shrink=.8)
        for extension in ['png', 'pdf']:
            fig.savefig(FIGURES / f'per_pair_{label.lower()}_heatmap.{extension}', dpi=300)
        plt.close(fig)
    pd.DataFrame(report).to_csv(FIGURES / 'fraction_summary.csv', index=False)
    print(f'Saved figures and numerical tables: {FIGURES}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('make-jobs')
    runner = commands.add_parser('run')
    runner.add_argument('--fraction', type=float, choices=FRACTIONS, required=True)
    runner.add_argument('--seed', type=int, choices=SEEDS, required=True)
    commands.add_parser('plot')
    args = parser.parse_args()
    if args.command == 'make-jobs':
        make_jobs()
    elif args.command == 'run':
        run(args)
    else:
        plot()
