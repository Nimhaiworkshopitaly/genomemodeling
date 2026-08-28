#!/usr/bin/env python3
"""Run from the genomemodeling root: make-jobs, run, or plot."""
import argparse
import csv
import shlex
import sys
from pathlib import Path

SEEDS = [42, 1042, 2042, 3042, 4042]
RESULTS = Path('hpc_multicpu/results_inversion_pairwise_25pct')
FIGURES = Path('hpc_multicpu/figures_inversion_pairwise_25pct')


def make_jobs():
    if not Path('hpc_multicpu/multicpu_eval_one_setting_inversion_fraction.py').is_file():
        raise SystemExit('Run from the genomemodeling repository root.')
    RESULTS.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    commands = []
    for fraction in [0, 0.25]:
        for seed in SEEDS:
            commands.append(shlex.join([
                sys.executable, str(script), 'run', '--fraction', str(fraction),
                '--seed', str(seed),
            ]))
    path = Path('hpc_multicpu/jobs_inversion_pairwise_25pct.swarm')
    path.write_text('\n'.join(commands) + '\n')
    print(f'Wrote {len(commands)} jobs: {path}')


def run(args):
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

    paths = sorted(RESULTS.glob('pairs_if_*_seed_*.csv'))
    if len(paths) != 10:
        raise SystemExit(f'Expected 10 completed pair CSVs, found {len(paths)}.')
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
                for seed in SEEDS for fraction in [0.0, 0.25]}
    actual = set(data[keys + ['inversion_fraction']].itertuples(index=False, name=None))
    if len(pairs) != 28 or actual != expected or len(data) != 280:
        raise ValueError('Expected 28 pairs x five seeds x two fractions.')
    baseline = data[data.inversion_fraction == 0].set_index(keys)[metrics]
    inverted = data[data.inversion_fraction == 0.25].set_index(keys)[metrics]
    difference = baseline - inverted
    FIGURES.mkdir(parents=True, exist_ok=True)
    difference.to_csv(FIGURES / 'paired_seed_improvements.csv')
    summary = difference.groupby(level=['pair_a', 'pair_b']).agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(FIGURES / 'per_pair_improvements.csv')
    for metric, label in zip(metrics, ['KS', 'Kuiper']):
        ordered = summary.sort_values(metric + '_mean')
        mean = ordered[metric + '_mean'].to_numpy()
        sd = ordered[metric + '_std'].to_numpy()
        y = np.arange(len(ordered))
        names = [f'{a} vs {b}' for a, b in ordered.index]
        fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
        ax.barh(y, mean, xerr=sd, capsize=3, height=0.7,
                color=['#299d8f' if v >= 0 else '#cf465a' for v in mean],
                error_kw={'elinewidth': 1, 'ecolor': '#333333'})
        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(y, names, fontsize=9)
        ax.set_xlabel(f'{label} improvement: 0% minus 25% inversions\n'
                      'Positive favors 25% inversions; error bars = SD of paired differences')
        ax.set_title(f'Per-genome-pair {label}: mean across five matched seeds\n'
                     'Uniform-breakpoint inversions; fixed total rate = 0.316228')
        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)
        for extension in ['png', 'pdf']:
            fig.savefig(FIGURES / f'per_pair_{label.lower()}_improvement_25pct.{extension}', dpi=300)
        plt.close(fig)
        print(f'{label}: mean improvement = {mean.mean():+.6f}; '
              f'positive pairs = {(mean > 0).sum()}/{len(mean)}')
    print(f'Saved figures and numerical tables: {FIGURES}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('make-jobs')
    runner = commands.add_parser('run')
    runner.add_argument('--fraction', type=float, choices=[0, 0.25], required=True)
    runner.add_argument('--seed', type=int, choices=SEEDS, required=True)
    commands.add_parser('plot')
    args = parser.parse_args()
    if args.command == 'make-jobs':
        make_jobs()
    elif args.command == 'run':
        run(args)
    else:
        plot()
