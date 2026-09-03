#!/usr/bin/env python3
"""Generate and analyze a joint translocation/inversion rate grid on ATGC0070."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

TRANSLOCATION_RATES = [0.15, 0.20, 0.25, 0.30, 0.35]
INVERSION_RATES = [0.00, 0.02, 0.05, 0.08, 0.12]
SEEDS = [42, 1042, 2042, 3042, 4042]
RESULTS = Path("hpc_multicpu/results_joint_translocation_inversion_uniform")
FIGURES = Path("hpc_multicpu/figures_joint_translocation_inversion_uniform")
SWARM = Path("hpc_multicpu/jobs_joint_translocation_inversion_uniform.swarm")


def result_path(translocation_rate: float, inversion_rate: float, seed: int) -> Path:
    return RESULTS / (
        f"result_t_{translocation_rate:.2f}_i_{inversion_rate:.2f}_seed_{seed}.csv"
    )


def make_jobs() -> None:
    evaluator = Path("hpc_multicpu/multicpu_eval_one_setting_inversion_fraction.py")
    if not evaluator.is_file():
        raise SystemExit("Run from the genomemodeling root; evaluator was not found.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    commands = []
    skipped = 0
    for translocation_rate in TRANSLOCATION_RATES:
        for inversion_rate in INVERSION_RATES:
            total = translocation_rate + inversion_rate
            fraction = inversion_rate / total
            for seed in SEEDS:
                output = result_path(translocation_rate, inversion_rate, seed)
                if output.exists() and output.stat().st_size > 0:
                    skipped += 1
                    continue
                commands.append(shlex.join([
                    sys.executable,
                    str(evaluator.resolve()),
                    "--atgc-dir", "ATGC0070",
                    "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
                    "--root-mode", "median_synthetic",
                    "--rf", "0.1",
                    "--total-rearrangement-rate", f"{total:.12g}",
                    "--inversion-fraction", f"{fraction:.12g}",
                    "--translocation-exp", "1e9",
                    "--inversion-exp", "3",
                    "--inversion-size-mode", "uniform_breakpoints",
                    "--gain-loss-exp", "1e9",
                    "--core-fraction", "0.5",
                    "--core-protection", "0.9",
                    "--n-runs", "100",
                    "--workers", "16",
                    "--seed", str(seed),
                    "--out-csv", str(output),
                ]))
    SWARM.parent.mkdir(parents=True, exist_ok=True)
    SWARM.write_text("\n".join(commands) + ("\n" if commands else ""))
    print(f"Wrote {len(commands)} jobs: {SWARM}")
    print(f"Skipped {skipped} existing nonempty files; full grid = 125 jobs.")


def load_results():
    import numpy as np
    import pandas as pd

    paths = sorted(
        path for path in RESULTS.glob("result_t_*_i_*_seed_*.csv")
        if not path.name.endswith("_pairs.csv")
    )
    if not paths:
        raise SystemExit(f"No results found in {RESULTS}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"{path}: expected exactly one row, found {len(frame)}")
        parts = path.stem.split("_")
        frame["requested_translocation_rate"] = float(parts[2])
        frame["requested_inversion_rate"] = float(parts[4])
        frame["source_file"] = str(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    keys = ["requested_translocation_rate", "requested_inversion_rate", "seed"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate translocation/inversion/seed result rows.")
    expected = {(t, i, seed) for t in TRANSLOCATION_RATES
                for i in INVERSION_RATES for seed in SEEDS}
    actual = set(data[keys].itertuples(index=False, name=None))
    if actual != expected or len(data) != 125:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SystemExit(f"Incomplete grid. Missing={missing}; unexpected={extra}")
    numeric_checks = {
        "rf": 0.1, "core_fraction": 0.5, "core_protection": 0.9,
        "n_runs": 100, "translocation_exp": 1e9, "gain_loss_exp": 1e9,
    }
    for column, expected_value in numeric_checks.items():
        if not np.allclose(data[column], expected_value, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Unexpected values in {column}.")
    for column, expected_value in {
        "root_mode": "median_synthetic",
        "inversion_size_mode": "uniform_breakpoints",
    }.items():
        if not data[column].eq(expected_value).all():
            raise ValueError(f"Unexpected values in {column}.")
    if not np.allclose(data.translocation_rate,
                       data.requested_translocation_rate, rtol=1e-9, atol=1e-12):
        raise ValueError("Saved translocation rates do not match requested rates.")
    if not np.allclose(data.inversion_rate,
                       data.requested_inversion_rate, rtol=1e-9, atol=1e-12):
        raise ValueError("Saved inversion rates do not match requested rates.")
    return data


def analyze() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    data = load_results()
    metrics = [
        ("avg_ks_statistic", "KS statistic"),
        ("avg_kuiper_statistic", "Kuiper statistic"),
        ("composite_score", "Composite score"),
    ]
    group = ["requested_translocation_rate", "requested_inversion_rate"]
    summary = data.groupby(group)[[metric for metric, _ in metrics]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURES / "combined_joint_rate_results.csv", index=False)
    summary.to_csv(FIGURES / "joint_rate_summary.csv", index=False)

    best_rows = []
    for metric, label in metrics:
        value_column = metric + "_mean"
        best = summary.loc[summary[value_column].idxmin()]
        best_rows.append({
            "metric": label,
            "best_translocation_rate": best.requested_translocation_rate,
            "best_inversion_rate": best.requested_inversion_rate,
            "mean": best[value_column],
            "sd": best[metric + "_std"],
        })
        matrix = summary.pivot(index="requested_inversion_rate",
                               columns="requested_translocation_rate",
                               values=value_column).reindex(
                                   index=INVERSION_RATES,
                                   columns=TRANSLOCATION_RATES)
        matrix.to_csv(FIGURES / f"{metric}_heatmap_values.csv")
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        image = ax.imshow(matrix.to_numpy(), cmap="viridis_r", aspect="auto")
        ax.set_xticks(np.arange(len(TRANSLOCATION_RATES)),
                      [f"{value:.2f}" for value in TRANSLOCATION_RATES])
        ax.set_yticks(np.arange(len(INVERSION_RATES)),
                      [f"{value:.2f}" for value in INVERSION_RATES])
        ax.set_xlabel("Single-gene translocation rate")
        ax.set_ylabel("Uniform-breakpoint inversion rate")
        ax.set_title(f"{label}: mean across five matched seeds\nLower is better; gain/loss rates fixed at 0.1")
        midpoint = (np.nanmin(matrix.to_numpy()) + np.nanmax(matrix.to_numpy())) / 2
        for row in range(len(INVERSION_RATES)):
            for column in range(len(TRANSLOCATION_RATES)):
                value = matrix.iloc[row, column]
                ax.text(column, row, f"{value:.4f}", ha="center", va="center",
                        color="white" if value > midpoint else "black", fontsize=10)
        ax.scatter(TRANSLOCATION_RATES.index(best.requested_translocation_rate),
                   INVERSION_RATES.index(best.requested_inversion_rate),
                   marker="*", s=300, facecolor="none", edgecolor="#e63946", linewidth=2)
        fig.colorbar(image, ax=ax, label=label)
        for extension in ["png", "pdf"]:
            fig.savefig(FIGURES / f"joint_rates_{metric}.{extension}", dpi=300)
        plt.close(fig)

    best_table = pd.DataFrame(best_rows)
    best_table.to_csv(FIGURES / "best_joint_rate_settings.csv", index=False)
    print(best_table.to_string(index=False))
    print(f"Saved figures and tables: {FIGURES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["make-jobs", "analyze"])
    args = parser.parse_args()
    if args.command == "make-jobs":
        make_jobs()
    else:
        analyze()
