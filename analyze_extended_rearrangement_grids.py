#!/usr/bin/env python3
"""Visualize inversion-size and block-translocation parameter sweeps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {
    "composite_score": "Composite score",
    "avg_ks_statistic": "KS statistic",
    "avg_kuiper_statistic": "Kuiper statistic",
}
TRANSLOCATION_RATES = [0.25, 0.30, 0.35]
INVERSION_RATES = [0.0, 0.005, 0.01, 0.02, 0.05]

INVERSION_RESULTS = Path(
    "hpc_multicpu/results_inversion_size_parameter_grid_observed_medoid_"
    "empirical_core"
)
INVERSION_FIGURES = Path(
    "hpc_multicpu/figures_inversion_size_parameter_grid_observed_medoid_"
    "empirical_core"
)
BLOCK_STEM = "block_translocation_mixture_observed_medoid_empirical_core"


def rate_token(rate):
    return f"{rate:g}".replace(".", "p")


def bounded_rf(value):
    try:
        rate = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("rf must be numeric") from error
    if not 0.01 <= rate <= 100:
        raise argparse.ArgumentTypeError("rf must be between 0.01 and 100")
    return rate


def block_results_path(rate):
    return Path(f"hpc_multicpu/results_{BLOCK_STEM}_rf_{rate_token(rate)}")


def block_figures_path(rate):
    return Path(f"hpc_multicpu/figures_{BLOCK_STEM}_rf_{rate_token(rate)}")


def load_results(directory, expected):
    rows = []
    for path in sorted(directory.glob("result_*.csv")):
        if path.stat().st_size == 0 or path.name.endswith("_pairs.csv"):
            continue
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"{path}: expected one row, found {len(frame)}")
        row = frame.iloc[0].copy()
        row["source_file"] = str(path)
        rows.append(row)
    if len(rows) != expected:
        raise SystemExit(
            f"{directory}: found {len(rows)}/{expected} nonempty primary results"
        )
    data = pd.DataFrame(rows)
    missing = [metric for metric in METRICS if metric not in data.columns]
    if missing:
        raise ValueError(f"{directory}: missing columns {missing}")
    for column, expected_values in (
        ("translocation_rate", TRANSLOCATION_RATES),
        ("inversion_rate", INVERSION_RATES),
    ):
        numeric = pd.to_numeric(data[column], errors="raise")
        snapped = numeric.map(
            lambda value: min(expected_values, key=lambda target: abs(value - target))
        )
        if not np.allclose(numeric, snapped, atol=1e-8, rtol=0):
            unexpected = sorted(numeric[~np.isclose(numeric, snapped, atol=1e-8)].unique())
            raise ValueError(f"{directory}: unexpected {column} values {unexpected}")
        data[column] = snapped.astype(float)
    if "root_mode" in data and not data["root_mode"].eq("observed_medoid").all():
        raise ValueError(f"{directory}: results include a non-medoid root")
    if "core_mode" in data and not data["core_mode"].eq("empirical").all():
        raise ValueError(f"{directory}: results include a non-empirical core mode")
    return data


def inversion_label(row):
    mode = row["inversion_size_mode"]
    if mode == "geometric":
        return f"Geometric mean={row['inversion_geometric_mean']:g}"
    if mode == "powerlaw":
        return f"Power law alpha={row['inversion_exp']:g}"
    if mode == "lognormal":
        return (
            f"Log-normal median={row['inversion_lognormal_median']:g}, "
            f"sigma={row['inversion_lognormal_sigma']:g}"
        )
    return "Uniform breakpoints"


def translocation_label(row):
    mode = row["translocation_size_mode"]
    if mode == "single_gene":
        return "Single gene"
    if mode == "geometric":
        return f"Geometric mean={row['translocation_geometric_mean']:g}"
    if mode == "powerlaw":
        return f"Power law alpha={row['translocation_exp']:g}"
    return (
        f"Log-normal median={row['translocation_lognormal_median']:g}, "
        f"sigma={row['translocation_lognormal_sigma']:g}"
    )


def gain_label(row):
    if row["gain_size_mode"] == "single_gene":
        return "Single-gene gain"
    return f"HGT geometric mean={row['gain_geometric_mean']:g}"


def draw_heatmap(table, label, title, output):
    table = table.reindex(index=INVERSION_RATES, columns=TRANSLOCATION_RATES)
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{title}: heatmap contains missing values")
    best_row, best_col = np.unravel_index(np.argmin(values), values.shape)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(
        range(len(TRANSLOCATION_RATES)),
        [f"{value:.2f}" for value in TRANSLOCATION_RATES],
    )
    ax.set_yticks(
        range(len(INVERSION_RATES)),
        [f"{value:g}" for value in INVERSION_RATES],
    )
    ax.set_xlabel("Translocation rate")
    ax.set_ylabel("Inversion rate")
    ax.set_title(title)
    midpoint = (values.min() + values.max()) / 2
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col, row, f"{values[row, col]:.4f}",
                ha="center", va="center",
                color="white" if values[row, col] > midpoint else "black",
            )
    ax.scatter(
        best_col, best_row, marker="*", s=420, facecolors="none",
        edgecolors="red", linewidths=2.5,
    )
    fig.colorbar(image, ax=ax, label=label)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_ranking_plot(ranking, metric, label, title, output):
    ordered = ranking.sort_values(metric, ascending=True)
    height = max(6, 0.35 * len(ordered))
    fig, ax = plt.subplots(figsize=(10, height), constrained_layout=True)
    positions = np.arange(len(ordered))
    values = ordered[metric].to_numpy(dtype=float)
    ax.hlines(positions, values.min(), values, color="#94a3b8", linewidth=1.5)
    ax.scatter(values, positions, color="#007f73", s=55, zorder=3)
    ax.set_yticks(positions, ordered["configuration"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Best mean {label} across rate grid")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    padding = max(np.ptp(values) * 0.02, abs(values.min()) * 0.001)
    ax.set_xlim(values.min() - padding, values.max() + 8 * padding)
    for position, value in zip(positions, values):
        ax.text(value, position, f" {value:.4f}", va="center", fontsize=8)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def matched_improvements(data, configuration_columns, output):
    keys = configuration_columns + ["rf", "translocation_rate", "seed"]
    baseline = data[np.isclose(data["inversion_rate"], 0)].copy()
    baseline = baseline[keys + list(METRICS)].rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )
    nonzero = data[data["inversion_rate"] > 0].copy()
    merged = nonzero.merge(baseline, on=keys, how="left", validate="many_to_one")
    for metric in METRICS:
        merged[f"improvement_{metric}"] = (
            merged[f"baseline_{metric}"] - merged[metric]
        )
    merged.to_csv(output, index=False)
    return merged


def analyze_inversions():
    data = load_results(INVERSION_RESULTS, 5175)
    data["configuration"] = data.apply(inversion_label, axis=1)
    INVERSION_FIGURES.mkdir(parents=True, exist_ok=True)
    data.to_csv(INVERSION_FIGURES / "combined_results.csv", index=False)

    group_keys = [
        "rf", "configuration", "inversion_size_mode",
        "translocation_rate", "inversion_rate",
    ]
    mean_data = data.groupby(group_keys, as_index=False)[list(METRICS)].mean()
    mean_data.to_csv(INVERSION_FIGURES / "mean_results.csv", index=False)
    improvements = matched_improvements(
        data, ["configuration"],
        INVERSION_FIGURES / "matched_seed_improvements.csv",
    )
    improvement_summary = improvements.groupby(
        ["rf", "configuration", "translocation_rate", "inversion_rate"],
        as_index=False,
    ).agg(**{
        f"mean_improvement_{metric}": (f"improvement_{metric}", "mean")
        for metric in METRICS
    })
    improvement_summary.to_csv(
        INVERSION_FIGURES / "mean_matched_improvements.csv", index=False
    )

    best_rows = []
    for rf in sorted(data["rf"].unique()):
        rf_data = mean_data[np.isclose(mean_data["rf"], rf)]
        for metric, label in METRICS.items():
            indices = rf_data.groupby("configuration")[metric].idxmin()
            ranking = rf_data.loc[indices].copy()
            ranking.to_csv(
                INVERSION_FIGURES / f"rf_{str(rf).replace('.', 'p')}_{metric}_ranking.csv",
                index=False,
            )
            save_ranking_plot(
                ranking, metric, label,
                f"Inversion-size ranking; gain/loss rate {rf:g}\nLower is better",
                INVERSION_FIGURES /
                f"rf_{str(rf).replace('.', 'p')}_{metric}_ranking.png",
            )
            best = ranking.loc[ranking[metric].idxmin()]
            best_rows.append(best)
            selected = rf_data[rf_data["configuration"] == best["configuration"]]
            table = selected.pivot(
                index="inversion_rate", columns="translocation_rate", values=metric
            )
            draw_heatmap(
                table, label,
                f"Best inversion-size model: {best['configuration']}\n"
                f"Gain/loss rate {rf:g}; mean across five seeds; lower is better",
                INVERSION_FIGURES /
                f"rf_{str(rf).replace('.', 'p')}_{metric}_best_heatmap.png",
            )
    pd.DataFrame(best_rows).to_csv(
        INVERSION_FIGURES / "best_settings_by_rate_and_metric.csv", index=False
    )
    print(f"Saved inversion-size visualizations: {INVERSION_FIGURES}")


def analyze_blocks(rf):
    results = block_results_path(rf)
    figures = block_figures_path(rf)
    data = load_results(results, 2250)
    if not np.allclose(pd.to_numeric(data["rf"]), rf, atol=1e-8, rtol=0):
        raise ValueError(f"{results}: results do not all have rf={rf:g}")
    data["configuration"] = data.apply(translocation_label, axis=1)
    data["gain_configuration"] = data.apply(gain_label, axis=1)
    figures.mkdir(parents=True, exist_ok=True)
    data.to_csv(figures / "combined_results.csv", index=False)

    group_keys = [
        "rf", "gain_configuration", "configuration",
        "translocation_rate", "inversion_rate",
    ]
    mean_data = data.groupby(group_keys, as_index=False)[list(METRICS)].mean()
    mean_data.to_csv(figures / "mean_results.csv", index=False)
    matched_improvements(
        data, ["gain_configuration", "configuration"],
        figures / "matched_seed_inversion_improvements.csv",
    )

    best_rows = []
    for gain in sorted(data["gain_configuration"].unique()):
        gain_data = mean_data[mean_data["gain_configuration"] == gain]
        gain_token = "hgt_block" if gain.startswith("HGT") else "single_gain"
        for metric, label in METRICS.items():
            indices = gain_data.groupby("configuration")[metric].idxmin()
            ranking = gain_data.loc[indices].copy()
            ranking.to_csv(
                figures / f"{gain_token}_{metric}_ranking.csv", index=False
            )
            save_ranking_plot(
                ranking, metric, label,
                f"Block-translocation ranking; {gain}; gain/loss rate {rf:g}\n"
                "Lower is better",
                figures / f"{gain_token}_{metric}_ranking.png",
            )
            best = ranking.loc[ranking[metric].idxmin()]
            best_rows.append(best)
            selected = gain_data[
                gain_data["configuration"] == best["configuration"]
            ]
            table = selected.pivot(
                index="inversion_rate", columns="translocation_rate", values=metric
            )
            draw_heatmap(
                table, label,
                f"Best translocation model: {best['configuration']}\n"
                f"{gain}; gain/loss rate {rf:g}; mean across five seeds; "
                "lower is better",
                figures / f"{gain_token}_{metric}_best_heatmap.png",
            )
    pd.DataFrame(best_rows).to_csv(
        figures / "best_settings_by_gain_model_and_metric.csv", index=False
    )
    print(f"Saved block-translocation visualizations for rf={rf:g}: {figures}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "analysis", choices=("inversion-sizes", "block-translocations", "all")
    )
    parser.add_argument(
        "--rf", nargs="+", type=bounded_rf,
        help=(
            "Gain/loss rates from 0.01 through 100 for block-translocation "
            "plots (default: 0.1 0.2 0.3)"
        ),
    )
    args = parser.parse_args()
    if args.analysis in ("inversion-sizes", "all"):
        analyze_inversions()
    if args.analysis in ("block-translocations", "all"):
        for rf in args.rf or (0.1, 0.2, 0.3):
            analyze_blocks(rf)


if __name__ == "__main__":
    main()
