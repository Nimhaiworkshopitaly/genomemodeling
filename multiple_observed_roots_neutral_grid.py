#!/usr/bin/env python3
"""Neutral rf x rt landscapes across multiple observed starting genomes."""
from __future__ import annotations

import argparse
import math
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prod_1b_core_composite import (
    _circular_adjacencies,
    _jaccard_distance,
    build_real_pmfs,
    convert_to_numeric,
    observed_medoid_genome_id,
)


RF_VALUES = np.logspace(-2, 1.5, 8).tolist()
RT_VALUES = np.logspace(-3, 1, 9).tolist()
SEEDS = [42, 1042, 2042, 3042, 4042]
EXPECTED_ROOTS = 5

ROOT = Path("hpc_multicpu/multiple_observed_roots_neutral")
RESULTS = ROOT / "results"
PARTS = ROOT / "parts"
FIGURES = ROOT / "figures"
ROOTS_CSV = ROOT / "selected_roots.csv"
EVALUATOR = Path("hpc_multicpu/multicpu_eval_one_setting_core_composite.py")
BIOWULF_TREE = Path("ATGC0070/yuri_gl26/ATGC0070.gl.tre")
FALLBACK_TREE = Path("ATGC0070/atgc.iq.r.tre")

METRICS = (
    ("composite_score", "Composite score"),
    ("avg_ks_statistic", "KS statistic"),
    ("avg_kuiper_statistic", "Kuiper statistic"),
)


def token(value: float) -> str:
    return f"{value:.10g}".replace(".", "p").replace("-", "m")


def combined_distance(left, right, contents, adjacencies) -> float:
    return 0.5 * (
        _jaccard_distance(contents[left], contents[right])
        + _jaccard_distance(adjacencies[left], adjacencies[right])
    )


def select_diverse_roots(n_roots: int) -> pd.DataFrame:
    tree_path = BIOWULF_TREE if BIOWULF_TREE.is_file() else FALLBACK_TREE
    if not tree_path.is_file():
        raise FileNotFoundError(
            f"Neither {BIOWULF_TREE} nor {FALLBACK_TREE} exists"
        )
    _, tree, real_genomes, _ = build_real_pmfs(
        str(tree_path),
        "ATGC0070/atgc.cc.csv",
    )
    genome_ids = sorted(
        leaf.name for leaf in tree.get_terminals() if leaf.name in real_genomes
    )
    if n_roots > len(genome_ids):
        raise ValueError(f"Requested {n_roots} roots from {len(genome_ids)} genomes")
    genomes = {
        genome_id: convert_to_numeric(real_genomes[genome_id])
        for genome_id in genome_ids
    }
    contents = {genome_id: set(genome) for genome_id, genome in genomes.items()}
    adjacencies = {
        genome_id: _circular_adjacencies(genome)
        for genome_id, genome in genomes.items()
    }
    median_length = float(np.median([len(genome) for genome in genomes.values()]))
    medoid = observed_medoid_genome_id(real_genomes, genome_ids)
    selected = [medoid]
    minimum_distance = {medoid: 0.0}
    while len(selected) < n_roots:
        candidates = []
        for genome_id in genome_ids:
            if genome_id in selected:
                continue
            distance = min(
                combined_distance(genome_id, chosen, contents, adjacencies)
                for chosen in selected
            )
            candidates.append((
                -distance,
                abs(len(genomes[genome_id]) - median_length),
                genome_id,
            ))
        _, _, chosen = min(candidates)
        minimum_distance[chosen] = min(
            combined_distance(chosen, earlier, contents, adjacencies)
            for earlier in selected
        )
        selected.append(chosen)
    frame = pd.DataFrame([
        {
            "root_index": index + 1,
            "genome_id": genome_id,
            "role": "medoid" if index == 0 else "diverse",
            "genome_length": len(genomes[genome_id]),
            "distance_to_nearest_earlier_root": minimum_distance[genome_id],
        }
        for index, genome_id in enumerate(selected)
    ])
    ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOTS_CSV, index=False)
    return frame


def load_roots() -> pd.DataFrame:
    if not ROOTS_CSV.is_file():
        raise SystemExit(
            f"Missing {ROOTS_CSV}. Run the select-roots action first."
        )
    roots = pd.read_csv(ROOTS_CSV, dtype={"genome_id": str})
    if roots.empty or roots["genome_id"].duplicated().any():
        raise ValueError("Selected-roots table is empty or contains duplicates")
    return roots


def result_path(genome_id: str, rf: float, rt: float, seed: int) -> Path:
    return RESULTS / (
        f"root_{genome_id}_rf_{token(rf)}_rt_{token(rt)}_seed_{seed}.csv"
    )


def write_parts(commands: list[str], jobs_per_part: int) -> list[Path]:
    PARTS.mkdir(parents=True, exist_ok=True)
    for old in PARTS.glob("part_*.swarm"):
        old.unlink()
    paths = []
    for start in range(0, len(commands), jobs_per_part):
        path = PARTS / f"part_{len(paths) + 1:03d}.swarm"
        path.write_text("\n".join(commands[start:start + jobs_per_part]) + "\n")
        paths.append(path)
    return paths


def make_jobs(jobs_per_part: int) -> None:
    if not EVALUATOR.is_file():
        raise SystemExit("Run this command from the genomemodeling root.")
    roots = load_roots()
    RESULTS.mkdir(parents=True, exist_ok=True)
    commands, skipped = [], 0
    for genome_id in roots["genome_id"]:
        for rf in RF_VALUES:
            for rt in RT_VALUES:
                for seed in SEEDS:
                    output = result_path(genome_id, rf, rt, seed)
                    if output.is_file() and output.stat().st_size > 0:
                        skipped += 1
                        continue
                    command = [
                        sys.executable, str(EVALUATOR.resolve()),
                        "--atgc-dir", "ATGC0070",
                        "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
                        "--root-mode", "observed_selected",
                        "--root-genome-id", genome_id,
                        "--rf", f"{rf:.12g}",
                        "--rt", f"{rt:.12g}",
                        "--inv-rate", "0",
                        "--huge-exp", "1e9",
                        "--core-mode", "synthetic_fraction",
                        "--core-fraction", "0",
                        "--core-protection", "0",
                        "--n-runs", "100",
                        "--workers", "16",
                        "--seed", str(seed),
                        "--out-csv", str(output),
                    ]
                    commands.append(shlex.join(command))
    parts = write_parts(commands, jobs_per_part)
    expected = len(roots) * len(RF_VALUES) * len(RT_VALUES) * len(SEEDS)
    print(f"Roots: {len(roots)}")
    print(f"Grid per root: {len(RF_VALUES)} rf x {len(RT_VALUES)} rt x {len(SEEDS)} seeds")
    print(f"Wrote {len(commands)} jobs in {len(parts)} parts: {PARTS}")
    print(f"Skipped {skipped}; complete experiment = {expected} jobs")


def status() -> None:
    roots = load_roots()
    expected_per_root = len(RF_VALUES) * len(RT_VALUES) * len(SEEDS)
    total = 0
    for genome_id in roots["genome_id"]:
        completed = sum(
            result_path(genome_id, rf, rt, seed).is_file()
            and result_path(genome_id, rf, rt, seed).stat().st_size > 0
            for rf in RF_VALUES for rt in RT_VALUES for seed in SEEDS
        )
        total += completed
        print(f"{genome_id}: {completed}/{expected_per_root} completed")
    print(f"total: {total}/{expected_per_root * len(roots)} completed")


def load_complete_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    roots = load_roots()
    frames = []
    for genome_id in roots["genome_id"]:
        for rf in RF_VALUES:
            for rt in RT_VALUES:
                for seed in SEEDS:
                    path = result_path(genome_id, rf, rt, seed)
                    if not path.is_file() or path.stat().st_size == 0:
                        raise SystemExit(f"Missing result: {path}")
                    frame = pd.read_csv(path)
                    if len(frame) != 1:
                        raise ValueError(f"{path}: expected one row")
                    frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if not np.allclose(data["core_protection"], 0):
        raise ValueError("Some results used core protection")
    if not np.allclose(data["core_fraction"], 0):
        raise ValueError("Some results used a nonzero core fraction")
    return roots, data


def format_rate(value: float) -> str:
    return f"{value:.6g}"


def analyze() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    roots, data = load_complete_results()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURES / "combined_results.csv", index=False)
    means = data.groupby(
        ["selected_root_genome_id", "rf", "rt"], as_index=False
    )[[metric for metric, _ in METRICS]].mean()
    means.to_csv(FIGURES / "mean_results.csv", index=False)

    n_roots = len(roots)
    ncols = min(3, n_roots)
    nrows = math.ceil(n_roots / ncols)
    x = np.log10(np.asarray(RT_VALUES))
    y = np.log10(np.asarray(RF_VALUES))
    xx, yy = np.meshgrid(x, y)
    for metric, label in METRICS:
        matrices, optima = [], []
        for genome_id in roots["genome_id"]:
            subset = means[means["selected_root_genome_id"].astype(str).eq(genome_id)]
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
        for index, (genome_id, matrix, optimum) in enumerate(zip(
            roots["genome_id"], matrices, optima
        )):
            ax = axes.flat[index]
            contour = ax.contourf(xx, yy, matrix, levels=levels, cmap="viridis_r",
                                  norm=norm, extend="both")
            lines = ax.contour(xx, yy, matrix, levels=6, colors="black",
                               linewidths=0.55, alpha=0.55)
            ax.clabel(lines, inline=True, fontsize=7, fmt="%.3g")
            best_x = math.log10(float(optimum["rt"]))
            best_y = math.log10(float(optimum["rf"]))
            ax.scatter(best_x, best_y, marker="*", s=360, facecolors="none",
                       edgecolors="red", linewidths=2.4, zorder=5)
            role = roots.iloc[index]["role"]
            ax.set_title(
                f"Root {index + 1}: {genome_id} ({role})\n"
                f"best={optimum[metric]:.5g}; rf={optimum['rf']:.5g}; "
                f"rt={optimum['rt']:.5g}",
                fontsize=12,
            )
            ax.grid(color="white", alpha=0.16, linewidth=0.7)
        for index in range(n_roots, nrows * ncols):
            axes.flat[index].set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Single-gene translocation rate (rt)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Gene gain/loss rate (rf)")
        for ax in axes.flat[:n_roots]:
            ax.set_xticks(x, [format_rate(value) for value in RT_VALUES],
                          rotation=45, ha="right")
            ax.set_yticks(y, [format_rate(value) for value in RF_VALUES])
        colorbar = fig.colorbar(contour, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label(label)
        fig.suptitle(
            f"{label} landscapes across observed starting genomes\n"
            "Neutral model: no core protection; mean across five matched seeds; "
            "lower is better",
            fontsize=17,
        )
        stem = FIGURES / f"{metric}_multiple_observed_roots_neutral"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {stem.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("select-roots", "make-jobs", "status", "analyze")
    )
    parser.add_argument("--n-roots", type=int, default=EXPECTED_ROOTS)
    parser.add_argument("--jobs-per-part", type=int, default=500)
    args = parser.parse_args()
    if args.n_roots < 2:
        parser.error("--n-roots must be at least 2")
    if args.jobs_per_part < 1:
        parser.error("--jobs-per-part must be positive")
    if args.action == "select-roots":
        print(select_diverse_roots(args.n_roots).to_string(index=False))
    elif args.action == "make-jobs":
        make_jobs(args.jobs_per_part)
    elif args.action == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
