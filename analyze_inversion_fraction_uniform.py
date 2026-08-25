#!/usr/bin/env python3
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 1042, 2042, 3042, 4042]
METRICS = [
    ("avg_ks_statistic", "Average KS statistic"),
    ("avg_kuiper_statistic", "Average Kuiper statistic"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize the uniform-breakpoint inversion-fraction experiment."
    )
    parser.add_argument(
        "--results-dir",
        default="hpc_multicpu/results_inversion_fraction_uniform",
    )
    parser.add_argument(
        "--out-dir",
        default="hpc_multicpu/figures_inversion_fraction_uniform",
    )
    parser.add_argument(
        "--combined-csv",
        default="hpc_multicpu/combined_inversion_fraction_uniform_30.csv",
    )
    return parser.parse_args()


def load_results(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSV files found in {results_dir}")

    frames = [pd.read_csv(path).assign(source_file=path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    required = {
        "inversion_fraction",
        "seed",
        "inversion_size_mode",
        "avg_ks_statistic",
        "avg_kuiper_statistic",
    }
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise SystemExit(f"Missing columns: {sorted(missing_columns)}")

    if len(data) != 30:
        raise SystemExit(f"Expected 30 result rows, found {len(data)}")
    if set(data["inversion_size_mode"]) != {"uniform_breakpoints"}:
        raise SystemExit(
            "Results include a non-uniform inversion mode: "
            f"{sorted(set(data['inversion_size_mode']))}"
        )

    observed = {
        (round(float(row.inversion_fraction), 8), int(row.seed))
        for row in data.itertuples()
    }
    expected = {(fraction, seed) for fraction in FRACTIONS for seed in SEEDS}
    missing = sorted(expected.difference(observed))
    unexpected = sorted(observed.difference(expected))
    if missing or unexpected or len(observed) != 30:
        raise SystemExit(
            f"Fraction/seed grid is invalid. Missing={missing}; unexpected={unexpected}"
        )
    return data.sort_values(["inversion_fraction", "seed"])


def summarize(data):
    columns = [name for name, _ in METRICS]
    summary = (
        data.groupby("inversion_fraction", as_index=False)[columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "inversion_fraction"
        if column[0] == "inversion_fraction"
        else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    return summary


def plot_summary(summary, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    color = "#264653"
    best_color = "#d1495b"

    for axis, (metric, label) in zip(axes, METRICS):
        means = summary[f"{metric}_mean"].to_numpy()
        stds = summary[f"{metric}_std"].to_numpy()
        fractions = summary["inversion_fraction"].to_numpy()
        best_index = int(np.nanargmin(means))

        axis.errorbar(
            fractions,
            means,
            yerr=stds,
            marker="o",
            markersize=8,
            linewidth=2.5,
            capsize=4,
            color=color,
            label="Mean +/- SD",
        )
        axis.scatter(
            fractions[best_index],
            means[best_index],
            marker="*",
            s=260,
            color=best_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="Best",
        )
        axis.annotate(
            f"best fraction = {fractions[best_index]:g}\n{means[best_index]:.4f}",
            (fractions[best_index], means[best_index]),
            xytext=(18, 18),
            textcoords="offset points",
            color=best_color,
            fontsize=11,
        )
        axis.set_title(label)
        axis.set_xlabel("Inversion fraction")
        axis.set_ylabel(label)
        axis.set_xticks(FRACTIONS)
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(
        "Fixed total rate inversion experiment\n"
        "uniform-breakpoint inversions; mean +/- SD across five matched seeds",
        fontsize=18,
    )
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.combined_csv) or ".", exist_ok=True)

    data = load_results(args.results_dir)
    data.to_csv(args.combined_csv, index=False)
    summary = summarize(data)
    summary_path = os.path.join(args.out_dir, "inversion_fraction_summary.csv")
    figure_path = os.path.join(args.out_dir, "inversion_fraction_ks_kuiper.png")
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, figure_path)

    print(summary.to_string(index=False))
    print(f"Combined CSV: {args.combined_csv}")
    print(f"Summary CSV:  {summary_path}")
    print(f"Figure:       {figure_path}")


if __name__ == "__main__":
    main()
