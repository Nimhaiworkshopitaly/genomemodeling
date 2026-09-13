#!/usr/bin/env python3
"""Focused inversion-size experiment using the observed-medoid root."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

TRANSLOCATION_RATES = [0.25, 0.30, 0.35]
INVERSION_RATES = [0.0, 0.005, 0.01, 0.02, 0.05]
SIZE_MODELS = ["geometric", "powerlaw", "lognormal", "uniform_breakpoints"]
SEEDS = [42, 1042, 2042, 3042, 4042]
MODEL_LABELS = {
    "geometric": "Geometric (mean 10)",
    "powerlaw": "Power law (alpha 3)",
    "lognormal": "Log-normal (median 10, sigma 1)",
    "uniform_breakpoints": "Uniform breakpoints",
}

def locations(rf):
    base = "focused_inversion_size_observed_medoid_empirical_core"
    if rf != 0.1:
        rate_token = f"{rf:g}".replace(".", "p")
        base = f"{base}_rf_{rate_token}"
    return (
        Path("hpc_multicpu") / f"results_{base}",
        Path("hpc_multicpu") / f"figures_{base}",
        Path("hpc_multicpu") / f"jobs_{base}.swarm",
    )


def result_path(results, model, translocation_rate, inversion_rate, seed):
    return results / (
        f"result_{model}_t_{translocation_rate:.2f}_"
        f"i_{inversion_rate:.3f}_seed_{seed}.csv"
    )


def make_jobs(rf):
    evaluator = Path(
        "hpc_multicpu/multicpu_eval_one_setting_inversion_fraction.py"
    )
    if not evaluator.is_file():
        raise SystemExit("Run this command from the genomemodeling root.")
    results, _, swarm = locations(rf)
    results.mkdir(parents=True, exist_ok=True)
    commands = []
    skipped = 0
    for model in SIZE_MODELS:
        for translocation_rate in TRANSLOCATION_RATES:
            for inversion_rate in INVERSION_RATES:
                total = translocation_rate + inversion_rate
                fraction = inversion_rate / total
                for seed in SEEDS:
                    output = result_path(
                        results, model, translocation_rate, inversion_rate, seed
                    )
                    if output.exists() and output.stat().st_size > 0:
                        skipped += 1
                        continue
                    commands.append(shlex.join([
                        sys.executable, str(evaluator.resolve()),
                        "--atgc-dir", "ATGC0070",
                        "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
                        "--root-mode", "observed_medoid",
                        "--rf", f"{rf:g}",
                        "--total-rearrangement-rate", f"{total:.12g}",
                        "--inversion-fraction", f"{fraction:.12g}",
                        "--translocation-exp", "1e9",
                        "--inversion-exp", "3",
                        "--inversion-size-mode", model,
                        "--inversion-geometric-mean", "10",
                        "--inversion-lognormal-median", "10",
                        "--inversion-lognormal-sigma", "1",
                        "--inversion-min-size", "2",
                        "--gain-loss-exp", "1e9",
                        "--core-fraction", "0",
                        "--core-protection", "0.9",
                        "--core-mode", "empirical",
                        "--core-prevalence", "1.0",
                        "--n-runs", "100", "--workers", "16",
                        "--seed", str(seed), "--out-csv", str(output),
                    ]))
    swarm.write_text("\n".join(commands) + ("\n" if commands else ""))
    print(f"Wrote {len(commands)} jobs: {swarm}")
    print(f"Skipped {skipped}; complete grid = 300 jobs.")


def load_results(rf):
    import numpy as np
    import pandas as pd

    results, _, _ = locations(rf)
    frames = []
    for model in SIZE_MODELS:
        for translocation_rate in TRANSLOCATION_RATES:
            for inversion_rate in INVERSION_RATES:
                for seed in SEEDS:
                    path = result_path(
                        results, model, translocation_rate, inversion_rate, seed
                    )
                    if not path.is_file() or path.stat().st_size == 0:
                        continue
                    frame = pd.read_csv(path)
                    if len(frame) != 1:
                        raise ValueError(f"{path}: expected one row, found {len(frame)}")
                    frame["requested_size_model"] = model
                    frame["requested_translocation_rate"] = translocation_rate
                    frame["requested_inversion_rate"] = inversion_rate
                    frame["source_file"] = str(path)
                    frames.append(frame)
    if not frames:
        raise SystemExit(f"No results found in {results}")
    data = pd.concat(frames, ignore_index=True)
    keys = [
        "requested_size_model", "requested_translocation_rate",
        "requested_inversion_rate", "seed",
    ]
    expected = {
        (model, translocation_rate, inversion_rate, seed)
        for model in SIZE_MODELS
        for translocation_rate in TRANSLOCATION_RATES
        for inversion_rate in INVERSION_RATES
        for seed in SEEDS
    }
    actual = set(data[keys].itertuples(index=False, name=None))
    if actual != expected or len(data) != 300:
        missing = sorted(expected - actual)
        raise SystemExit(
            f"Incomplete grid: {len(actual)}/300 jobs completed; "
            f"first missing settings: {missing[:10]}"
        )
    if not data["root_mode"].eq("observed_medoid").all():
        raise ValueError("Some results do not use the observed-medoid root.")
    if data["selected_root_genome_id"].nunique() != 1:
        raise ValueError("The selected medoid differs among jobs.")
    if not data["core_mode"].eq("empirical").all():
        raise ValueError("Some results do not use empirical core COGs.")
    if not np.allclose(data["core_protection"], 0.9):
        raise ValueError("Some results do not use core protection 0.9.")
    if not np.allclose(data["rf"], rf):
        raise ValueError(f"Some results do not use gain/loss rate {rf:g}.")
    return data


def analyze(rf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = load_results(rf)
    _, figures, _ = locations(rf)
    figures.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("composite_score", "Composite score"),
        ("avg_ks_statistic", "KS statistic"),
        ("avg_kuiper_statistic", "Kuiper statistic"),
    ]
    group_keys = [
        "requested_size_model", "requested_translocation_rate",
        "requested_inversion_rate",
    ]
    aggregations = {}
    for column, _ in metrics:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    summary = data.groupby(group_keys, as_index=False).agg(**aggregations)
    data.to_csv(figures / "combined_results.csv", index=False)
    summary.to_csv(figures / "mean_sd_results.csv", index=False)

    for column, label in metrics:
        matrices = []
        for model in SIZE_MODELS:
            subset = summary[summary["requested_size_model"] == model]
            matrix = subset.pivot(
                index="requested_inversion_rate",
                columns="requested_translocation_rate",
                values=f"{column}_mean",
            ).reindex(index=INVERSION_RATES, columns=TRANSLOCATION_RATES)
            matrices.append(matrix)
            matrix.to_csv(figures / f"{model}_{column}_values.csv")

        all_values = np.concatenate([matrix.to_numpy().ravel() for matrix in matrices])
        vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
        fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
        image = None
        for ax, model, matrix in zip(axes.flat, SIZE_MODELS, matrices):
            values = matrix.to_numpy(dtype=float)
            image = ax.imshow(
                values, cmap="viridis_r", aspect="auto", vmin=vmin, vmax=vmax
            )
            ax.set_xticks(
                range(len(TRANSLOCATION_RATES)),
                [f"{value:.2f}" for value in TRANSLOCATION_RATES],
            )
            ax.set_yticks(
                range(len(INVERSION_RATES)),
                [f"{value:g}" for value in INVERSION_RATES],
            )
            ax.set_xlabel("Single-gene translocation rate")
            ax.set_ylabel("Inversion rate")
            ax.set_title(MODEL_LABELS[model])
            midpoint = (vmin + vmax) / 2
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    value = values[row, col]
                    ax.text(
                        col, row, f"{value:.4f}", ha="center", va="center",
                        color="white" if value > midpoint else "black",
                    )
            row, col = np.unravel_index(np.nanargmin(values), values.shape)
            ax.scatter(
                col, row, marker="*", s=300, facecolors="none",
                edgecolors="red", linewidths=2.2,
            )
        fig.suptitle(
            f"{label}: observed-medoid root with empirical core protection\n"
            f"Mean across five matched seeds; gain/loss rate {rf:g}; lower is better"
        )
        fig.colorbar(image, ax=axes, label=label, shrink=0.85)
        output = figures / f"{column}_size_model_comparison.png"
        fig.savefig(output, dpi=300)
        plt.close(fig)
        print(f"Saved: {output}")

    print(f"Medoid: {data['selected_root_genome_id'].iloc[0]}")
    print(f"Saved figures and tables: {figures}")


def main():
    parser = argparse.ArgumentParser(
        description="Run or analyze the focused inversion-size grid."
    )
    parser.add_argument("action", choices=("make-jobs", "analyze"))
    parser.add_argument(
        "--rf", type=float, default=0.1,
        help="Per-gene gain and loss rate (default: 0.1).",
    )
    args = parser.parse_args()
    if args.rf <= 0:
        parser.error("--rf must be positive")
    (make_jobs if args.action == "make-jobs" else analyze)(args.rf)


if __name__ == "__main__":
    main()
