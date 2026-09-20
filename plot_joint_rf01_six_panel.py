#!/usr/bin/env python3
"""Plot a six-panel joint translocation/inversion model summary."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


METRICS = [
    ("composite_score", "Composite score"),
    ("avg_ks_statistic", "KS statistic"),
    ("avg_kuiper_statistic", "Kuiper statistic"),
]

GAIN_ORDER = ["Single-gene gain", "HGT geometric mean=3"]

TRANSLOCATION_ORDER = (
    ["Single gene"]
    + [f"Geometric mean={value}" for value in (2, 3, 5, 10)]
    + [f"Power law alpha={value:g}" for value in (1.5, 2, 2.5, 3)]
    + [
        f"Log-normal median={median}, sigma={sigma:g}"
        for median in (2, 5, 10)
        for sigma in (0.5, 1)
    ]
)

INVERSION_ORDER = (
    [f"Geometric mean={value}" for value in (3, 5, 10, 20, 50)]
    + [f"Power law alpha={value:g}" for value in (1.5, 2, 2.5, 3, 4)]
    + [
        f"Log-normal median={median}, sigma={sigma:g}"
        for median in (3, 5, 10, 20)
        for sigma in (0.5, 1, 1.5)
    ]
    + ["Uniform breakpoints"]
)


def short_translocation_label(label: str) -> str:
    return (
        label.replace("Single gene", "Single")
        .replace("Geometric mean=", "Geom ")
        .replace("Power law alpha=", "PL ")
        .replace("Log-normal median=", "LN ")
        .replace(", sigma=", ", σ=")
    )


def short_inversion_label(label: str) -> str:
    return (
        label.replace("Uniform breakpoints", "Uniform breakpoints")
        .replace("Geometric mean=", "Geom ")
        .replace("Power law alpha=", "PL ")
        .replace("Log-normal median=", "LN ")
        .replace(", sigma=", ", σ=")
    )


def validate_models(data: pd.DataFrame) -> None:
    observed_trans = set(data["translocation_model"])
    observed_inv = set(data["inversion_model"])
    if observed_trans != set(TRANSLOCATION_ORDER):
        raise ValueError(
            "Unexpected translocation labels. Missing: "
            f"{sorted(set(TRANSLOCATION_ORDER) - observed_trans)}; extra: "
            f"{sorted(observed_trans - set(TRANSLOCATION_ORDER))}"
        )
    if observed_inv != set(INVERSION_ORDER):
        raise ValueError(
            "Unexpected inversion labels. Missing: "
            f"{sorted(set(INVERSION_ORDER) - observed_inv)}; extra: "
            f"{sorted(observed_inv - set(INVERSION_ORDER))}"
        )


def best_model_matrix(data: pd.DataFrame, gain: str, metric: str):
    subset = data[data["gain_model"].eq(gain)].copy()
    best_indices = subset.groupby(
        ["inversion_model", "translocation_model"]
    )[metric].idxmin()
    best_rates = subset.loc[best_indices].copy()
    matrix = best_rates.pivot(
        index="inversion_model", columns="translocation_model", values=metric
    ).reindex(index=INVERSION_ORDER, columns=TRANSLOCATION_ORDER)
    if matrix.isna().any().any():
        raise ValueError(f"Incomplete model matrix for {gain}, {metric}")
    optimum = best_rates.loc[best_rates[metric].idxmin()]
    return matrix, optimum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf", type=float, default=0.1)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "hpc_multicpu/"
            "figures_joint_size_full_factorial_observed_medoid_empirical_core/"
            "mean_results.csv"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "hpc_multicpu/"
            "figures_joint_size_full_factorial_observed_medoid_empirical_core/"
            "rf_0p1_joint_model_six_panel"
        ),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data = data[np.isclose(data["rf"], args.rf, atol=1e-10, rtol=0)].copy()
    if data.empty:
        raise ValueError(f"No rows found for rf={args.rf:g}")
    validate_models(data)
    missing_gains = set(GAIN_ORDER) - set(data["gain_model"])
    if missing_gains:
        raise ValueError(f"Missing gain models: {sorted(missing_gains)}")

    panels = {}
    norms = {}
    for metric, _ in METRICS:
        values = []
        for gain in GAIN_ORDER:
            matrix, optimum = best_model_matrix(data, gain, metric)
            panels[(gain, metric)] = (matrix, optimum)
            values.extend(matrix.to_numpy().ravel())
        norms[metric] = Normalize(vmin=min(values), vmax=max(values))

    fig, axes = plt.subplots(
        2, 3, figsize=(25, 18), constrained_layout=True, sharex=True, sharey=True
    )
    images = {}
    for row, gain in enumerate(GAIN_ORDER):
        for column, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row, column]
            matrix, optimum = panels[(gain, metric)]
            image = ax.imshow(
                matrix.to_numpy(), aspect="auto", cmap="viridis_r",
                norm=norms[metric], interpolation="nearest",
            )
            images[metric] = image
            x = TRANSLOCATION_ORDER.index(optimum["translocation_model"])
            y = INVERSION_ORDER.index(optimum["inversion_model"])
            ax.plot(
                x, y, marker="*", markersize=20, markerfacecolor="none",
                markeredgecolor="red", markeredgewidth=2.2,
            )
            ax.set_title(
                f"{metric_label}\n"
                f"best={optimum[metric]:.4f}; rt={optimum['translocation_rate']:g}; "
                f"ri={optimum['inversion_rate']:g}",
                fontsize=14,
            )
            if column == 0:
                ax.set_ylabel(
                    f"{gain}\n\nInversion-size model", fontsize=14, labelpad=12
                )
            if row == 1:
                ax.set_xlabel("Translocation-size model", fontsize=14)

    axes[0, 0].set_yticks(
        np.arange(len(INVERSION_ORDER)),
        [short_inversion_label(value) for value in INVERSION_ORDER],
        fontsize=9,
    )
    for ax in axes[1, :]:
        ax.set_xticks(
            np.arange(len(TRANSLOCATION_ORDER)),
            [short_translocation_label(value) for value in TRANSLOCATION_ORDER],
            rotation=58, ha="right", fontsize=9,
        )

    for column, (metric, metric_label) in enumerate(METRICS):
        colorbar = fig.colorbar(
            images[metric], ax=axes[:, column], location="bottom",
            shrink=0.78, pad=0.09, aspect=35,
        )
        colorbar.set_label(
            f"Best mean {metric_label.lower()} across rate combinations; lower is better",
            fontsize=11,
        )

    fig.suptitle(
        f"Joint translocation- and inversion-size model landscapes (gain/loss rate {args.rf:g})\n"
        "Mean across five matched seeds; each cell minimized over translocation and inversion rates",
        fontsize=19,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
