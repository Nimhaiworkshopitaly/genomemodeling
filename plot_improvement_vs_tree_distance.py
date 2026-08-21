#!/usr/bin/env python3
"""Relate event-size model improvement to pairwise tree distance."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
from scipy.stats import linregress, pearsonr, spearmanr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-pair-csv",
        default="per_pair_rearrangement_ks_kuiper/per_pair_comparison.csv",
    )
    parser.add_argument(
        "--tree",
        default="ATGC0070/yuri_gl26/ATGC0070.gl.tre",
    )
    parser.add_argument(
        "--out-dir",
        default="per_pair_rearrangement_ks_kuiper/tree_distance",
    )
    parser.add_argument("--label-top", type=int, default=3)
    return parser.parse_args()


def find_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(
        "None of these required columns were found: " + ", ".join(candidates)
    )


def short_pair(row):
    a = str(row["pair_a"]).replace("GCF_", "")
    b = str(row["pair_b"]).replace("GCF_", "")
    return f"{a} vs {b}"


def plot_metric(frame, metric, output_path, label_top):
    x = frame["tree_distance"].to_numpy(dtype=float)
    y = frame[f"{metric}_improvement_exp3"].to_numpy(dtype=float)

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    regression = linregress(x, y)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(y >= 0, "#2a9d8f", "#d1495b")
    ax.scatter(x, y, c=colors, s=75, edgecolor="black", linewidth=0.5, zorder=3)

    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_line,
        regression.intercept + regression.slope * x_line,
        color="#264653",
        linewidth=2,
        label="Linear fit",
    )
    ax.axhline(0, color="black", linewidth=1)

    n_labels = min(label_top, len(frame))
    for index in np.argsort(np.abs(y))[-n_labels:]:
        row = frame.iloc[index]
        ax.annotate(
            short_pair(row),
            (x[index], y[index]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    display_name = "KS" if metric == "ks" else "Kuiper"
    ax.set_title(
        f"{display_name} improvement versus gain+loss-clock tree distance\n"
        "positive values favor exponent 3"
    )
    ax.set_xlabel("Patristic distance between genomes")
    ax.set_ylabel(
        f"{display_name} improvement\n(single gene - exponent 3)"
    )
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}\n"
        f"Spearman rho = {spearman_rho:.3f}, p = {spearman_p:.3g}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "metric": metric,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "slope": regression.slope,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    frame = pd.read_csv(args.per_pair_csv)
    required = {"pair_a", "pair_b"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    single_ks = find_column(frame, ["single_gene_rt_0.316228_ks", "single_gene_ks"])
    exp3_ks = find_column(frame, ["exponent_3_rt_0.316228_ks", "exponent_3_ks"])
    single_kuiper = find_column(
        frame, ["single_gene_rt_0.316228_kuiper", "single_gene_kuiper"]
    )
    exp3_kuiper = find_column(
        frame, ["exponent_3_rt_0.316228_kuiper", "exponent_3_kuiper"]
    )

    tree = Phylo.read(args.tree, "newick")
    terminal_names = {terminal.name for terminal in tree.get_terminals()}
    names = set(frame["pair_a"]).union(frame["pair_b"])
    absent = sorted(names.difference(terminal_names))
    if absent:
        raise ValueError(f"Genome names absent from tree: {absent}")

    frame["tree_distance"] = [
        tree.distance(str(a), str(b))
        for a, b in zip(frame["pair_a"], frame["pair_b"])
    ]
    frame["ks_improvement_exp3"] = frame[single_ks] - frame[exp3_ks]
    frame["kuiper_improvement_exp3"] = frame[single_kuiper] - frame[exp3_kuiper]

    table_path = os.path.join(args.out_dir, "pair_improvements_with_tree_distance.csv")
    frame.to_csv(table_path, index=False)

    summaries = []
    for metric in ("ks", "kuiper"):
        summaries.append(
            plot_metric(
                frame,
                metric,
                os.path.join(args.out_dir, f"{metric}_improvement_vs_tree_distance.png"),
                args.label_top,
            )
        )

    summary = pd.DataFrame(summaries)
    summary_path = os.path.join(args.out_dir, "tree_distance_correlations.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {table_path}")
    print(f"Wrote figures and correlation summary to {args.out_dir}")


if __name__ == "__main__":
    main()
