#!/usr/bin/env python3
"""Neutral rf x rt landscapes using pangenome-sampled synthetic roots."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from multiple_observed_roots_neutral_grid import (
    BIOWULF_TREE,
    FALLBACK_TREE,
    METRICS,
    RF_VALUES,
    RT_VALUES,
    SEEDS,
    format_rate,
    token,
)
from prod_1b_core_composite import build_real_pmfs, convert_to_numeric


N_ROOTS = 5
ROOT_SEEDS = (1701, 2701, 3701, 4701, 5701)
ROOT = Path("hpc_multicpu/pangenome_sampled_roots_neutral")
ROOT_GENOMES = ROOT / "root_genomes"
ROOTS_CSV = ROOT / "synthetic_roots.csv"
RESULTS = ROOT / "results"
PARTS = ROOT / "parts"
FIGURES = ROOT / "figures"
EVALUATOR = Path("hpc_multicpu/multicpu_eval_one_setting_core_composite.py")


def observed_genomes():
    tree_path = BIOWULF_TREE if BIOWULF_TREE.is_file() else FALLBACK_TREE
    if not tree_path.is_file():
        raise FileNotFoundError(f"Neither {BIOWULF_TREE} nor {FALLBACK_TREE} exists")
    _, tree, real_genomes, _ = build_real_pmfs(
        str(tree_path), "ATGC0070/atgc.cc.csv"
    )
    genome_ids = sorted(
        leaf.name for leaf in tree.get_terminals() if leaf.name in real_genomes
    )
    genomes = {
        genome_id: convert_to_numeric(real_genomes[genome_id])
        for genome_id in genome_ids
    }
    return genomes


def pangenome_statistics(genomes):
    prevalence = Counter()
    adjacency = Counter()
    for genome in genomes.values():
        prevalence.update(set(genome))
        if len(genome) > 1:
            for index, gene in enumerate(genome):
                neighbor = genome[(index + 1) % len(genome)]
                if gene != neighbor:
                    adjacency[tuple(sorted((gene, neighbor)))] += 1
    return prevalence, adjacency


def weighted_choice(rng, values, weights):
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    return values[int(rng.choice(len(values), p=probabilities))]


def sample_gene_content(rng, prevalence, target_length):
    genes = np.asarray(sorted(prevalence), dtype=int)
    if target_length > len(genes):
        raise ValueError(
            f"Target length {target_length} exceeds pangenome size {len(genes)}"
        )
    weights = np.asarray([prevalence[int(gene)] for gene in genes], dtype=float)
    probabilities = weights / weights.sum()
    selected = rng.choice(genes, size=target_length, replace=False, p=probabilities)
    return {int(gene) for gene in selected}


def order_by_adjacency(rng, selected, prevalence, adjacency):
    available = set(selected)
    first = weighted_choice(
        rng, sorted(available), [prevalence[gene] for gene in sorted(available)]
    )
    path = [first]
    available.remove(first)
    supported_edges = 0
    disconnected_jumps = 0
    while available:
        options = []
        for side, endpoint in (("left", path[0]), ("right", path[-1])):
            for gene in available:
                weight = adjacency.get(tuple(sorted((endpoint, gene))), 0)
                if weight:
                    options.append((side, gene, weight))
        if options:
            choice = weighted_choice(rng, options, [option[2] for option in options])
            side, gene, _ = choice
            supported_edges += 1
        else:
            genes = sorted(available)
            gene = weighted_choice(rng, genes, [prevalence[item] for item in genes])
            side = "right"
            disconnected_jumps += 1
        if side == "left":
            path.insert(0, gene)
        else:
            path.append(gene)
        available.remove(gene)
    circular_supported = int(
        adjacency.get(tuple(sorted((path[-1], path[0]))), 0) > 0
    )
    supported_edges += circular_supported
    return path, supported_edges, disconnected_jumps, circular_supported


def generate_roots(n_roots):
    genomes = observed_genomes()
    prevalence, adjacency = pangenome_statistics(genomes)
    observed_lengths = np.asarray([len(genome) for genome in genomes.values()])
    ROOT_GENOMES.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n_roots):
        seed = ROOT_SEEDS[index] if index < len(ROOT_SEEDS) else ROOT_SEEDS[-1] + index
        rng = np.random.default_rng(seed)
        source_length = int(rng.choice(observed_lengths))
        content = sample_gene_content(rng, prevalence, source_length)
        ordered, supported, jumps, circular = order_by_adjacency(
            rng, content, prevalence, adjacency
        )
        synthetic_id = f"pangenome_root_{index + 1:02d}"
        path = ROOT_GENOMES / f"{synthetic_id}.json"
        path.write_text(json.dumps(ordered) + "\n")
        rows.append({
            "root_index": index + 1,
            "synthetic_id": synthetic_id,
            "root_seed": seed,
            "genome_length": len(ordered),
            "unique_cogs": len(set(ordered)),
            "observed_adjacency_edges": supported,
            "adjacency_support_fraction": supported / len(ordered),
            "disconnected_jumps": jumps,
            "circular_closure_supported": circular,
            "root_file": str(path),
            "observed_genomes_used": len(genomes),
            "pangenome_size": len(prevalence),
        })
    frame = pd.DataFrame(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOTS_CSV, index=False)
    return frame


def load_roots():
    if not ROOTS_CSV.is_file():
        raise SystemExit(f"Missing {ROOTS_CSV}; run generate-roots first")
    roots = pd.read_csv(ROOTS_CSV, dtype={"synthetic_id": str, "root_file": str})
    for path in roots["root_file"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return roots


def result_path(synthetic_id, rf, rt, seed):
    return RESULTS / (
        f"root_{synthetic_id}_rf_{token(rf)}_rt_{token(rt)}_seed_{seed}.csv"
    )


def write_parts(commands, jobs_per_part):
    PARTS.mkdir(parents=True, exist_ok=True)
    for old in PARTS.glob("part_*.swarm"):
        old.unlink()
    paths = []
    for start in range(0, len(commands), jobs_per_part):
        path = PARTS / f"part_{len(paths) + 1:03d}.swarm"
        path.write_text("\n".join(commands[start:start + jobs_per_part]) + "\n")
        paths.append(path)
    return paths


def make_jobs(jobs_per_part):
    roots = load_roots()
    RESULTS.mkdir(parents=True, exist_ok=True)
    commands, skipped = [], 0
    for root in roots.itertuples(index=False):
        for rf in RF_VALUES:
            for rt in RT_VALUES:
                for seed in SEEDS:
                    output = result_path(root.synthetic_id, rf, rt, seed)
                    if output.is_file() and output.stat().st_size > 0:
                        skipped += 1
                        continue
                    command = [
                        sys.executable, str(EVALUATOR.resolve()),
                        "--atgc-dir", "ATGC0070",
                        "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
                        "--root-mode", "pangenome_sampled",
                        "--root-genome-file", root.root_file,
                        "--rf", f"{rf:.12g}", "--rt", f"{rt:.12g}",
                        "--inv-rate", "0", "--huge-exp", "1e9",
                        "--core-mode", "synthetic_fraction",
                        "--core-fraction", "0", "--core-protection", "0",
                        "--n-runs", "100", "--workers", "16",
                        "--seed", str(seed), "--out-csv", str(output),
                    ]
                    commands.append(shlex.join(command))
    parts = write_parts(commands, jobs_per_part)
    expected = len(roots) * len(RF_VALUES) * len(RT_VALUES) * len(SEEDS)
    print(f"Wrote {len(commands)} jobs in {len(parts)} parts: {PARTS}")
    print(f"Skipped {skipped}; complete experiment = {expected} jobs")


def status():
    roots = load_roots()
    expected = len(RF_VALUES) * len(RT_VALUES) * len(SEEDS)
    total = 0
    for synthetic_id in roots["synthetic_id"]:
        count = sum(
            result_path(synthetic_id, rf, rt, seed).is_file()
            and result_path(synthetic_id, rf, rt, seed).stat().st_size > 0
            for rf in RF_VALUES for rt in RT_VALUES for seed in SEEDS
        )
        total += count
        print(f"{synthetic_id}: {count}/{expected} completed")
    print(f"total: {total}/{expected * len(roots)} completed")


def load_results():
    roots = load_roots()
    frames = []
    for root in roots.itertuples(index=False):
        expected_label = f"synthetic:{root.synthetic_id}"
        for rf in RF_VALUES:
            for rt in RT_VALUES:
                for seed in SEEDS:
                    path = result_path(root.synthetic_id, rf, rt, seed)
                    if not path.is_file() or path.stat().st_size == 0:
                        raise SystemExit(f"Missing result: {path}")
                    frame = pd.read_csv(path)
                    if len(frame) != 1:
                        raise ValueError(f"{path}: expected one row")
                    if str(frame.iloc[0]["selected_root_genome_id"]) != expected_label:
                        raise ValueError(f"{path}: unexpected root identifier")
                    frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if not np.allclose(data["core_protection"], 0):
        raise ValueError("Some results used core protection")
    return roots, data


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    roots, data = load_results()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURES / "combined_results.csv", index=False)
    means = data.groupby(
        ["selected_root_genome_id", "rf", "rt"], as_index=False
    )[[metric for metric, _ in METRICS]].mean()
    means.to_csv(FIGURES / "mean_results.csv", index=False)
    ncols, nrows = min(3, len(roots)), math.ceil(len(roots) / 3)
    x, y = np.log10(RT_VALUES), np.log10(RF_VALUES)
    xx, yy = np.meshgrid(x, y)
    for metric, label in METRICS:
        matrices, optima = [], []
        for root in roots.itertuples(index=False):
            root_label = f"synthetic:{root.synthetic_id}"
            subset = means[means["selected_root_genome_id"].eq(root_label)]
            matrix = subset.pivot(index="rf", columns="rt", values=metric).reindex(
                index=RF_VALUES, columns=RT_VALUES
            )
            matrices.append(matrix.to_numpy())
            optima.append(subset.loc[subset[metric].idxmin()])
        all_values = np.concatenate([matrix.ravel() for matrix in matrices])
        norm = Normalize(float(np.nanmin(all_values)), float(np.nanmax(all_values)))
        levels = np.linspace(norm.vmin, norm.vmax, 18)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 5.8 * nrows),
            sharex=True, sharey=True, constrained_layout=True, squeeze=False,
        )
        contour = None
        for index, (root, matrix, optimum) in enumerate(
            zip(roots.itertuples(index=False), matrices, optima)
        ):
            ax = axes.flat[index]
            contour = ax.contourf(xx, yy, matrix, levels=levels, cmap="viridis_r",
                                  norm=norm, extend="both")
            lines = ax.contour(xx, yy, matrix, levels=6, colors="black",
                               linewidths=0.55, alpha=0.55)
            ax.clabel(lines, inline=True, fontsize=7, fmt="%.3g")
            ax.scatter(math.log10(optimum["rt"]), math.log10(optimum["rf"]),
                       marker="*", s=360, facecolors="none", edgecolors="red",
                       linewidths=2.4, zorder=5)
            ax.set_title(
                f"Synthetic root {root.root_index}; length={root.genome_length}\n"
                f"best={optimum[metric]:.5g}; rf={optimum['rf']:.5g}; "
                f"rt={optimum['rt']:.5g}", fontsize=12,
            )
            ax.grid(color="white", alpha=0.16, linewidth=0.7)
        for index in range(len(roots), nrows * ncols):
            axes.flat[index].set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Single-gene translocation rate (rt)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Gene gain/loss rate (rf)")
        for ax in axes.flat[:len(roots)]:
            ax.set_xticks(x, [format_rate(value) for value in RT_VALUES],
                          rotation=45, ha="right")
            ax.set_yticks(y, [format_rate(value) for value in RF_VALUES])
        colorbar = fig.colorbar(contour, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label(label)
        fig.suptitle(
            f"{label} landscapes across pangenome-sampled starting genomes\n"
            "Neutral model: no core protection; mean across five matched seeds; "
            "lower is better", fontsize=17,
        )
        stem = FIGURES / f"{metric}_pangenome_sampled_roots_neutral"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {stem.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("generate-roots", "make-jobs", "status", "analyze")
    )
    parser.add_argument("--n-roots", type=int, default=N_ROOTS)
    parser.add_argument("--jobs-per-part", type=int, default=500)
    args = parser.parse_args()
    if args.n_roots < 2:
        parser.error("--n-roots must be at least 2")
    if args.action == "generate-roots":
        print(generate_roots(args.n_roots).to_string(index=False))
    elif args.action == "make-jobs":
        make_jobs(args.jobs_per_part)
    elif args.action == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
