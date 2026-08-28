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
COMPONENTS = [
    ("avg_w1", "Wasserstein", 1.0),
    ("avg_singleton_abs_error", "Singleton", 100.0),
    ("avg_short_cdf_abs_error", "Short CDF", 100.0),
    ("avg_long_tail_abs_error", "Long tail", 100.0),
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
        "--comparison-fraction", type=float,
        help="Nonzero fraction for the three-panel figure (default: best mean Kuiper).",
    )
    parser.add_argument(
        "--genome-length", type=int, default=1000,
        help="Illustrative genome length for the uniform-breakpoint size distribution.",
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
    paths = sorted(path for path in glob.glob(os.path.join(results_dir, "*.csv"))
                   if not path.endswith("_pairs.csv"))
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
        "avg_w1", "avg_singleton_abs_error", "avg_short_cdf_abs_error",
        "avg_long_tail_abs_error", "composite_score",
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


def load_pair_results(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "*_pairs.csv")))
    if not paths:
        raise SystemExit(
            "Pair-level CSVs are required for per-pair improvement. Existing global "
            "result CSVs are insufficient; rerun the 30 jobs with the updated evaluator."
        )
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"inversion_fraction", "seed", "genome_a", "genome_b",
                "ks_statistic", "kuiper_statistic"}
    missing = required.difference(data.columns)
    if missing:
        raise SystemExit(f"Pair-level CSVs are missing columns: {sorted(missing)}")
    duplicates = data.duplicated(["inversion_fraction", "seed", "genome_a", "genome_b"])
    if duplicates.any():
        raise SystemExit("Pair-level CSVs contain duplicate fraction/seed/genome-pair rows")
    return data


def choose_comparison_fraction(summary, requested):
    nonzero = summary[summary["inversion_fraction"] > 0]
    if requested is None:
        return float(nonzero.loc[nonzero["avg_kuiper_statistic_mean"].idxmin(),
                                 "inversion_fraction"])
    if requested <= 0 or not np.isclose(summary["inversion_fraction"], requested).any():
        raise SystemExit("--comparison-fraction must be one of the available nonzero fractions")
    return float(requested)


def plot_inversion_comparison(data, pairs, fraction, genome_length, output_path):
    baseline = data[np.isclose(data["inversion_fraction"], 0.0)]
    treatment = data[np.isclose(data["inversion_fraction"], fraction)]
    common_seeds = sorted(set(baseline["seed"]).intersection(treatment["seed"]))
    if len(common_seeds) != 5:
        raise SystemExit(f"Expected five matched seeds for fractions 0 and {fraction:g}")
    baseline = baseline.set_index("seed").loc[common_seeds]
    treatment = treatment.set_index("seed").loc[common_seeds]

    contributions = [weight * (treatment[column].mean() - baseline[column].mean())
                     for column, _, weight in COMPONENTS]

    selected_pairs = pairs[
        np.isclose(pairs["inversion_fraction"], 0.0)
        | np.isclose(pairs["inversion_fraction"], fraction)
    ].copy()
    selected_pairs = selected_pairs[selected_pairs["seed"].isin(common_seeds)]
    seed_counts = selected_pairs.groupby(
        ["inversion_fraction", "genome_a", "genome_b"]
    )["seed"].nunique()
    if (seed_counts != 5).any():
        raise SystemExit("Every genome pair must have five seeds at baseline and comparison fraction")
    baseline_pair_set = set(map(tuple, selected_pairs[
        np.isclose(selected_pairs["inversion_fraction"], 0.0)
    ][["genome_a", "genome_b"]].drop_duplicates().to_numpy()))
    treatment_pair_set = set(map(tuple, selected_pairs[
        np.isclose(selected_pairs["inversion_fraction"], fraction)
    ][["genome_a", "genome_b"]].drop_duplicates().to_numpy()))
    if baseline_pair_set != treatment_pair_set:
        raise SystemExit("Baseline and inversion fractions do not contain the same genome pairs")
    pair_means = selected_pairs.groupby(
        ["inversion_fraction", "genome_a", "genome_b"], as_index=False
    )[["ks_statistic", "kuiper_statistic"]].mean()
    base_pairs = pair_means[np.isclose(pair_means["inversion_fraction"], 0.0)]
    test_pairs = pair_means[np.isclose(pair_means["inversion_fraction"], fraction)]
    merged = base_pairs.merge(test_pairs, on=["genome_a", "genome_b"],
                              suffixes=("_baseline", "_inversion"), validate="one_to_one")
    merged["ks_improvement"] = (merged["ks_statistic_baseline"]
                                - merged["ks_statistic_inversion"])
    merged["kuiper_improvement"] = (merged["kuiper_statistic_baseline"]
                                    - merged["kuiper_statistic_inversion"])
    merged["pair"] = merged["genome_a"].astype(str) + " / " + merged["genome_b"].astype(str)
    merged = merged.sort_values("kuiper_improvement")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    lengths = np.arange(1, genome_length)
    inversion_probability = np.zeros_like(lengths, dtype=float)
    inversion_probability[lengths >= 2] = 1.0 / (genome_length - 2)
    axes[0].plot(lengths, inversion_probability, color="#2a9d8f", linewidth=2,
                 label="Uniform-breakpoint inversion")
    axes[0].vlines(1, 0, 1, color="#457b9d", linewidth=2)
    axes[0].scatter([1], [1], color="#457b9d", s=35,
                    label="Single-gene translocation baseline")
    axes[0].set(xlabel="Genes affected per event", ylabel="Event-size probability",
                title="A. Uniform-breakpoint inversion size")
    axes[0].set_yscale("log")
    axes[0].legend()

    colors = ["#d1495b" if value > 0 else "#2a9d8f" for value in contributions]
    y = np.arange(len(COMPONENTS))
    axes[1].barh(y, contributions, color=colors)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(y, [label for _, label, _ in COMPONENTS])
    axes[1].invert_yaxis()
    axes[1].set(xlabel=f"Component difference: fraction {fraction:g} - fraction 0",
                title="B. Composite-score decomposition")
    for yi, value in zip(y, contributions):
        axes[1].annotate(f"{value:+.3f}", (value, yi),
                         xytext=(5 if value >= 0 else -5, 0), textcoords="offset points",
                         ha="left" if value >= 0 else "right", va="center")

    pair_colors = ["#2a9d8f" if value > 0 else "#d1495b"
                   for value in merged["kuiper_improvement"]]
    pair_y = np.arange(len(merged))
    axes[2].barh(pair_y, merged["kuiper_improvement"], color=pair_colors)
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_yticks(pair_y, merged["pair"], fontsize=7)
    axes[2].set(xlabel="Kuiper improvement: fraction 0 - fraction f",
                title="C. Per-pair Kuiper improvement\n(five-seed means)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f"Uniform-breakpoint inversions: fraction {fraction:g} versus 0", fontsize=17)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return merged


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
    pairs = load_pair_results(args.results_dir)
    comparison_fraction = choose_comparison_fraction(summary, args.comparison_fraction)
    summary_path = os.path.join(args.out_dir, "inversion_fraction_summary.csv")
    figure_path = os.path.join(args.out_dir, "inversion_fraction_ks_kuiper.png")
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, figure_path)
    comparison_path = os.path.join(args.out_dir, "inversion_comparison_three_panel.png")
    pair_improvements = plot_inversion_comparison(
        data, pairs, comparison_fraction, args.genome_length, comparison_path
    )
    pair_path = os.path.join(args.out_dir, "per_pair_improvements.csv")
    pair_improvements.to_csv(pair_path, index=False)

    print(summary.to_string(index=False))
    print(f"Combined CSV: {args.combined_csv}")
    print(f"Summary CSV:  {summary_path}")
    print(f"Figure:       {figure_path}")
    print(f"Comparison:   {comparison_path}")
    print(f"Pair metrics: {pair_path}")


if __name__ == "__main__":
    main()
