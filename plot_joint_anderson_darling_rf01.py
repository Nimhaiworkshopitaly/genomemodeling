#!/usr/bin/env python3
"""Plot two joint-model Anderson-Darling heatmaps for rf=0.1."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from plot_joint_rf01_six_panel import (
    GAIN_ORDER,
    INVERSION_ORDER,
    TRANSLOCATION_ORDER,
    short_inversion_label,
    short_translocation_label,
)


METRIC = "avg_anderson_darling_distance"


def matrix_and_optimum(data, gain):
    subset = data[data["gain_model"].eq(gain)].copy()
    indices = subset.groupby(
        ["inversion_model", "translocation_model"]
    )[METRIC].idxmin()
    profiled = subset.loc[indices]
    matrix = profiled.pivot(
        index="inversion_model", columns="translocation_model", values=METRIC
    ).reindex(index=INVERSION_ORDER, columns=TRANSLOCATION_ORDER)
    optimum = profiled.loc[profiled[METRIC].idxmin()]
    return matrix, optimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path(
            "hpc_multicpu/"
            "figures_joint_size_cvm_rf_0p1_observed_medoid_empirical_core/"
            "mean_distribution_results.csv"
        ),
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        default=Path(
            "hpc_multicpu/"
            "figures_joint_size_cvm_rf_0p1_observed_medoid_empirical_core/"
            "rf_0p1_anderson_darling_joint_model_two_panel"
        ),
    )
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    panels = {gain: matrix_and_optimum(data, gain) for gain in GAIN_ORDER}
    values = np.concatenate([
        panels[gain][0].to_numpy().ravel() for gain in GAIN_ORDER
    ])
    norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 11), constrained_layout=True, sharex=True, sharey=True
    )
    image = None
    for column, gain in enumerate(GAIN_ORDER):
        ax = axes[column]
        matrix, optimum = panels[gain]
        image = ax.imshow(
            matrix.to_numpy(), aspect="auto", cmap="viridis_r",
            norm=norm, interpolation="nearest",
        )
        x = TRANSLOCATION_ORDER.index(optimum["translocation_model"])
        y = INVERSION_ORDER.index(optimum["inversion_model"])
        ax.plot(
            x, y, marker="*", markersize=20, markerfacecolor="none",
            markeredgecolor="red", markeredgewidth=2.2,
        )
        ax.set_title(
            f"{gain}\nbest={optimum[METRIC]:.6f}; "
            f"rt={optimum['translocation_rate']:g}; "
            f"ri={optimum['inversion_rate']:g}",
            fontsize=14,
        )
        ax.set_xlabel("Translocation-size model", fontsize=13)
        ax.set_xticks(
            np.arange(len(TRANSLOCATION_ORDER)),
            [short_translocation_label(v) for v in TRANSLOCATION_ORDER],
            rotation=58, ha="right", fontsize=8,
        )
    axes[0].set_ylabel("Inversion-size model", fontsize=13)
    axes[0].set_yticks(
        np.arange(len(INVERSION_ORDER)),
        [short_inversion_label(v) for v in INVERSION_ORDER], fontsize=8,
    )
    colorbar = fig.colorbar(image, ax=axes, location="bottom", shrink=0.65, pad=0.10)
    colorbar.set_label(
        "Best mean Anderson-Darling distance across rate combinations; lower is better"
    )
    fig.suptitle(
        "Joint translocation- and inversion-size Anderson-Darling landscapes\n"
        "Gain/loss rate 0.1; mean across five matched seeds",
        fontsize=18,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_prefix.with_suffix('.png')}")
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
