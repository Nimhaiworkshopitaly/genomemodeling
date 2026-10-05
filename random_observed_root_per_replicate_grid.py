#!/usr/bin/env python3
"""Neutral rf x rt grids with an observed root sampled for every replicate."""
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
    METRICS,
    RF_VALUES,
    RT_VALUES,
    SEEDS,
    format_rate,
    token,
)


SAMPLING_MODES = ("uniform", "medoid_proximity", "phylogenetic_root")
ROOT = Path("hpc_multicpu/random_observed_root_per_replicate")
RESULTS = ROOT / "results"
PARTS = ROOT / "parts"
FIGURES = ROOT / "figures"
EVALUATOR = Path("hpc_multicpu/multicpu_eval_one_setting_core_composite.py")


def result_path(mode, rf, rt, seed):
    return RESULTS / f"{mode}_rf_{token(rf)}_rt_{token(rt)}_seed_{seed}.csv"


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
    if not EVALUATOR.is_file():
        raise SystemExit("Run this command from the genomemodeling root")
    RESULTS.mkdir(parents=True, exist_ok=True)
    commands, skipped = [], 0
    for mode in SAMPLING_MODES:
        for rf in RF_VALUES:
            for rt in RT_VALUES:
                for seed in SEEDS:
                    output = result_path(mode, rf, rt, seed)
                    if output.is_file() and output.stat().st_size > 0:
                        skipped += 1
                        continue
                    command = [
                        sys.executable, str(EVALUATOR.resolve()),
                        "--atgc-dir", "ATGC0070",
                        "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
                        "--root-mode", "random_observed_per_replicate",
                        "--root-sampling", mode,
                        "--rf", f"{rf:.12g}", "--rt", f"{rt:.12g}",
                        "--inv-rate", "0", "--huge-exp", "1e9",
                        "--core-mode", "synthetic_fraction",
                        "--core-fraction", "0", "--core-protection", "0",
                        "--n-runs", "100", "--workers", "16",
                        "--seed", str(seed), "--out-csv", str(output),
                    ]
                    commands.append(shlex.join(command))
    parts = write_parts(commands, jobs_per_part)
    expected = len(SAMPLING_MODES) * len(RF_VALUES) * len(RT_VALUES) * len(SEEDS)
    print(f"Sampling modes: {len(SAMPLING_MODES)}")
    print(f"Wrote {len(commands)} jobs in {len(parts)} parts: {PARTS}")
    print(f"Skipped {skipped}; complete experiment = {expected} jobs")


def status():
    expected = len(RF_VALUES) * len(RT_VALUES) * len(SEEDS)
    total = 0
    for mode in SAMPLING_MODES:
        count = sum(
            result_path(mode, rf, rt, seed).is_file()
            and result_path(mode, rf, rt, seed).stat().st_size > 0
            for rf in RF_VALUES for rt in RT_VALUES for seed in SEEDS
        )
        total += count
        print(f"{mode}: {count}/{expected} completed")
    print(f"total: {total}/{expected * len(SAMPLING_MODES)} completed")


def load_results():
    frames = []
    for mode in SAMPLING_MODES:
        for rf in RF_VALUES:
            for rt in RT_VALUES:
                for seed in SEEDS:
                    path = result_path(mode, rf, rt, seed)
                    if not path.is_file() or path.stat().st_size == 0:
                        raise SystemExit(f"Missing result: {path}")
                    frame = pd.read_csv(path)
                    if len(frame) != 1:
                        raise ValueError(f"{path}: expected one row")
                    if str(frame.iloc[0]["root_sampling"]) != mode:
                        raise ValueError(f"{path}: unexpected root-sampling mode")
                    frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if not np.allclose(data["core_protection"], 0):
        raise ValueError("Some results used core protection")
    return data


def write_sampling_audit(data):
    counts = {mode: Counter() for mode in SAMPLING_MODES}
    expected_weights = {}
    for row in data.itertuples(index=False):
        mode = row.root_sampling
        counts[mode].update(json.loads(row.sampled_root_counts))
        expected_weights[mode] = json.loads(row.root_sampling_weights)
    rows = []
    for mode in SAMPLING_MODES:
        total = sum(counts[mode].values())
        for genome_id, weight in expected_weights[mode].items():
            observed = counts[mode][genome_id]
            rows.append({
                "root_sampling": mode,
                "genome_id": genome_id,
                "expected_probability": weight,
                "sampled_replicates": observed,
                "observed_fraction": observed / total,
                "total_replicates": total,
            })
    pd.DataFrame(rows).to_csv(FIGURES / "root_sampling_audit.csv", index=False)


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    data = load_results()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURES / "combined_results.csv", index=False)
    write_sampling_audit(data)
    available_metrics = []
    for metric, label in METRICS:
        if metric not in data.columns:
            print(f"Skipping {metric}: column is absent from the result files")
            continue
        values = pd.to_numeric(data[metric], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if values.notna().any():
            available_metrics.append((metric, label))
        else:
            print(f"Skipping {metric}: column has no finite values")
    metric_names = [metric for metric, _ in available_metrics]
    means = data.groupby(["root_sampling", "rf", "rt"], as_index=False)[
        metric_names
    ].mean()
    means.to_csv(FIGURES / "mean_results.csv", index=False)
    x, y = np.log10(RT_VALUES), np.log10(RF_VALUES)
    xx, yy = np.meshgrid(x, y)
    optimum_rows = []
    for metric, label in available_metrics:
        matrices, optima = [], []
        for mode in SAMPLING_MODES:
            subset = means[means["root_sampling"].eq(mode)]
            finite = pd.to_numeric(subset[metric], errors="coerce").dropna()
            matrix = subset.pivot(index="rf", columns="rt", values=metric)
            matrix = matrix.sort_index().sort_index(axis=1)
            expected_shape = (len(RF_VALUES), len(RT_VALUES))
            if matrix.shape != expected_shape or finite.empty:
                raise ValueError(f"{mode}, {metric}: incomplete landscape")
            matrices.append(matrix.to_numpy())
            optimum = subset.loc[finite.idxmin()]
            optima.append(optimum)
            optimum_rows.append({
                "metric": metric, "root_sampling": mode,
                "rf": optimum["rf"], "rt": optimum["rt"],
                "score": optimum[metric],
            })
        all_values = np.concatenate([matrix.ravel() for matrix in matrices])
        norm = Normalize(float(np.nanmin(all_values)), float(np.nanmax(all_values)))
        levels = np.linspace(norm.vmin, norm.vmax, 18)
        fig, axes = plt.subplots(
            1, 3, figsize=(20, 6.2), sharex=True, sharey=True,
            constrained_layout=True,
        )
        contour = None
        for ax, mode, matrix, optimum in zip(
            axes, SAMPLING_MODES, matrices, optima
        ):
            contour = ax.contourf(
                xx, yy, matrix, levels=levels, cmap="viridis_r",
                norm=norm, extend="both"
            )
            lines = ax.contour(
                xx, yy, matrix, levels=6, colors="black",
                linewidths=0.55, alpha=0.55
            )
            ax.clabel(lines, inline=True, fontsize=7, fmt="%.3g")
            ax.scatter(
                math.log10(optimum["rt"]), math.log10(optimum["rf"]),
                marker="*", s=360, facecolors="none", edgecolors="red",
                linewidths=2.4, zorder=5,
            )
            ax.set_title(
                f"{mode.replace('_', ' ').title()}\n"
                f"best={optimum[metric]:.5g}; rf={optimum['rf']:.5g}; "
                f"rt={optimum['rt']:.5g}"
            )
            ax.set_xlabel("Single-gene translocation rate (rt)")
            ax.set_xticks(x, [format_rate(value) for value in RT_VALUES],
                          rotation=45, ha="right")
            ax.set_yticks(y, [format_rate(value) for value in RF_VALUES])
            ax.grid(color="white", alpha=0.16, linewidth=0.7)
        axes[0].set_ylabel("Gene gain/loss rate (rf)")
        fig.colorbar(contour, ax=axes, shrink=0.84, pad=0.02, label=label)
        fig.suptitle(
            f"{label}: observed base genome sampled independently per replicate\n"
            "Neutral model: no core protection; mean across five matched seeds; "
            "lower is better", fontsize=16,
        )
        stem = FIGURES / f"{metric}_random_observed_root_per_replicate"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {stem.with_suffix('.png')}")
    pd.DataFrame(optimum_rows).to_csv(FIGURES / "best_models.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("make-jobs", "status", "analyze"))
    parser.add_argument("--jobs-per-part", type=int, default=500)
    args = parser.parse_args()
    if args.action == "make-jobs":
        make_jobs(args.jobs_per_part)
    elif args.action == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
