#!/usr/bin/env python3
"""Plot a matched-seed composite comparison of single-gene and exponent-3 models."""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPONENTS = [
    ("avg_w1", "Wasserstein", 1.0),
    ("avg_singleton_abs_error", "Singleton", 100.0),
    ("avg_short_cdf_abs_error", "Short CDF", 100.0),
    ("avg_long_tail_abs_error", "Long tail", 100.0),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="hpc_multicpu/results_rearrangement_replicates",
    )
    parser.add_argument("--rf", type=float, default=0.1)
    parser.add_argument("--rt", type=float, default=0.316227766017)
    parser.add_argument("--exp3", type=float, default=3.0)
    parser.add_argument("--single-exp", type=float, default=1e9)
    parser.add_argument(
        "--out",
        default=(
            "hpc_multicpu/figures_rearrangement_replicates/"
            "single_vs_exp3_composite.png"
        ),
    )
    return parser.parse_args()


def read_results(results_dir):
    paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def select_model(frame, rf, rt, exponent):
    selected = frame[
        np.isclose(frame["rf"].astype(float), rf)
        & np.isclose(frame["rt"].astype(float), rt, rtol=1e-5)
        & np.isclose(frame["rearrangement_exp"].astype(float), exponent)
    ].copy()
    if "inversion_multiplier" in selected:
        selected = selected[np.isclose(selected["inversion_multiplier"], 0.0)]
    return selected.sort_values("seed")


def main():
    args = parse_args()
    frame = read_results(args.results_dir)
    single = select_model(frame, args.rf, args.rt, args.single_exp)
    exp3 = select_model(frame, args.rf, args.rt, args.exp3)

    if single.empty or exp3.empty:
        raise ValueError(
            "Could not find both models. Check --results-dir, --rf, and --rt."
        )

    common_seeds = sorted(set(single["seed"]).intersection(exp3["seed"]))
    if not common_seeds:
        raise ValueError("The two models do not have matched seeds.")
    single = single[single["seed"].isin(common_seeds)].set_index("seed").loc[common_seeds]
    exp3 = exp3[exp3["seed"].isin(common_seeds)].set_index("seed").loc[common_seeds]

    colors = ["#457b9d", "#2a9d8f"]
    means = [single["composite_score"].mean(), exp3["composite_score"].mean()]
    sds = [single["composite_score"].std(ddof=1), exp3["composite_score"].std(ddof=1)]

    contributions = []
    component_labels = []
    for column, label, weight in COMPONENTS:
        contributions.append(weight * (exp3[column].mean() - single[column].mean()))
        component_labels.append(label)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.05, 1.35]},
        constrained_layout=True,
    )

    # Panel A: event-size mechanism.
    lengths = np.arange(1, 11)
    exp3_prob = lengths.astype(float) ** -3
    exp3_prob /= exp3_prob.sum()
    axes[0].bar(lengths - 0.18, np.r_[1.0, np.zeros(9)], width=0.36,
                color=colors[0], label="Single gene")
    axes[0].bar(lengths + 0.18, exp3_prob, width=0.36,
                color=colors[1], label="Exponent 3")
    axes[0].set_xlabel("Genes moved per translocation")
    axes[0].set_ylabel("Event-size probability")
    axes[0].set_title("A. Translocation event size")
    axes[0].set_xticks(lengths)
    axes[0].legend()

    # Panel C: exp3 minus single-gene component contributions.
    contribution_colors = ["#d1495b" if value > 0 else "#2a9d8f"
                           for value in contributions]
    y = np.arange(len(contributions))
    contribution_ax = axes[1]
    contribution_ax.barh(y, contributions, color=contribution_colors, height=0.62)
    contribution_ax.axvline(0, color="black", linewidth=1.2)
    contribution_ax.set_yticks(y, component_labels)
    contribution_ax.invert_yaxis()
    contribution_ax.set_xlabel("Weighted contribution: exponent 3 - single gene")
    contribution_ax.set_title("C. Why the composite scores differ")

    value_min = min(0.0, min(contributions))
    value_max = max(0.0, max(contributions))
    value_span = max(value_max - value_min, 0.1)
    contribution_ax.set_xlim(
        value_min - 0.18 * value_span,
        value_max + 0.24 * value_span,
    )
    for yi, value in zip(y, contributions):
        contribution_ax.annotate(
            f"{value:+.3f}",
            xy=(value, yi),
            xytext=(8 if value >= 0 else -8, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=12,
            fontweight="bold",
            clip_on=True,
        )
    contribution_ax.text(
        0.98,
        0.04,
        "Negative values favor exponent 3",
        transform=contribution_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)

    fig.suptitle(
        f"Single-gene versus exponent-3 translocations: rf={args.rf:g}, "
        f"rt={args.rt:.6g}",
        fontsize=17,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)

    print(f"Matched seeds: {common_seeds}")
    print(f"Single-gene composite: {means[0]:.9g} +/- {sds[0]:.9g}")
    print(f"Exponent-3 composite:   {means[1]:.9g} +/- {sds[1]:.9g}")
    print("Component contributions (exponent 3 minus single gene):")
    for label, value in zip(component_labels, contributions):
        print(f"  {label:12s} {value:+.9g}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
