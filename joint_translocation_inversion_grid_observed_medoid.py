#!/usr/bin/env python3
"""Matched translocation/inversion grids using one observed-medoid root."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

TRANSLOCATION_RATES = [0.15, 0.20, 0.25, 0.30, 0.35]
INVERSION_RATES = [0.00, 0.02, 0.05, 0.08, 0.12]
SEEDS = [42, 1042, 2042, 3042, 4042]
CONDITIONS = {
    "no_core": {"core_mode": "synthetic_fraction", "protection": 0.0},
    "empirical_core": {"core_mode": "empirical", "protection": 0.9},
}


def locations(condition):
    stem = f"joint_translocation_inversion_observed_medoid_{condition}"
    return (
        Path("hpc_multicpu") / f"results_{stem}",
        Path("hpc_multicpu") / f"figures_{stem}",
        Path("hpc_multicpu") / f"jobs_{stem}.swarm",
    )


def result_path(results, translocation_rate, inversion_rate, seed):
    return results / (
        f"result_t_{translocation_rate:.2f}_i_{inversion_rate:.2f}_seed_{seed}.csv"
    )


def make_jobs(condition):
    config = CONDITIONS[condition]
    results, _, swarm = locations(condition)
    evaluator = Path("hpc_multicpu/multicpu_eval_one_setting_inversion_fraction.py")
    if not evaluator.is_file():
        raise SystemExit("Run this command from the genomemodeling root.")
    results.mkdir(parents=True, exist_ok=True)
    commands = []
    skipped = 0
    for translocation_rate in TRANSLOCATION_RATES:
        for inversion_rate in INVERSION_RATES:
            total = translocation_rate + inversion_rate
            fraction = inversion_rate / total
            for seed in SEEDS:
                output = result_path(results, translocation_rate, inversion_rate, seed)
                if output.exists() and output.stat().st_size > 0:
                    skipped += 1
                    continue
                commands.append(shlex.join([
                    sys.executable, str(evaluator.resolve()),
                    "--atgc-dir", "ATGC0070",
                    "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
                    "--root-mode", "observed_medoid",
                    "--rf", "0.1",
                    "--total-rearrangement-rate", f"{total:.12g}",
                    "--inversion-fraction", f"{fraction:.12g}",
                    "--translocation-exp", "1e9",
                    "--inversion-exp", "3",
                    "--inversion-size-mode", "uniform_breakpoints",
                    "--gain-loss-exp", "1e9",
                    "--core-fraction", "0",
                    "--core-protection", str(config["protection"]),
                    "--core-mode", config["core_mode"],
                    "--core-prevalence", "1.0",
                    "--n-runs", "100", "--workers", "16",
                    "--seed", str(seed), "--out-csv", str(output),
                ]))
    swarm.write_text("\n".join(commands) + ("\n" if commands else ""))
    print(f"Wrote {len(commands)} jobs: {swarm}")
    print(f"Skipped {skipped}; complete grid = 125 jobs.")


def load_results(condition):
    import numpy as np
    import pandas as pd

    results, _, _ = locations(condition)
    frames = []
    for path in sorted(results.glob("result_t_*_i_*_seed_*.csv")):
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"{path}: expected one row, found {len(frame)}")
        parts = path.stem.split("_")
        frame["requested_translocation_rate"] = float(parts[2])
        frame["requested_inversion_rate"] = float(parts[4])
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No results found in {results}")
    data = pd.concat(frames, ignore_index=True)
    keys = ["requested_translocation_rate", "requested_inversion_rate", "seed"]
    expected = {(t, i, seed) for t in TRANSLOCATION_RATES
                for i in INVERSION_RATES for seed in SEEDS}
    actual = set(data[keys].itertuples(index=False, name=None))
    if actual != expected or len(data) != 125:
        raise SystemExit(f"Incomplete grid: {len(actual)}/125 settings completed.")
    if not data["root_mode"].eq("observed_medoid").all():
        raise ValueError("Some rows do not use the observed-medoid root.")
    if data["selected_root_genome_id"].nunique() != 1:
        raise ValueError("The selected medoid differs among matched jobs.")
    expected_protection = CONDITIONS[condition]["protection"]
    if not np.allclose(data["core_protection"], expected_protection):
        raise ValueError("Unexpected core-protection setting.")
    return data


def analyze(condition):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    data = load_results(condition)
    _, figures, _ = locations(condition)
    figures.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("composite_score", "Composite score"),
        ("avg_ks_statistic", "KS statistic"),
        ("avg_kuiper_statistic", "Kuiper statistic"),
    ]
    summary = data.groupby(
        ["requested_translocation_rate", "requested_inversion_rate"],
        as_index=False,
    )[[column for column, _ in metrics]].mean()
    data.to_csv(figures / "combined_results.csv", index=False)
    summary.to_csv(figures / "mean_results.csv", index=False)
    for column, label in metrics:
        matrix = summary.pivot(
            index="requested_inversion_rate",
            columns="requested_translocation_rate",
            values=column,
        ).reindex(index=INVERSION_RATES, columns=TRANSLOCATION_RATES)
        matrix.to_csv(figures / f"{column}_values.csv")
        values = matrix.to_numpy()
        row, col = np.unravel_index(np.nanargmin(values), values.shape)
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        image = ax.imshow(values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(TRANSLOCATION_RATES)),
                      [f"{value:.2f}" for value in TRANSLOCATION_RATES])
        ax.set_yticks(range(len(INVERSION_RATES)),
                      [f"{value:.2f}" for value in INVERSION_RATES])
        ax.set_xlabel("Single-gene translocation rate")
        ax.set_ylabel("Uniform-breakpoint inversion rate")
        ax.set_title(f"{label}: observed-medoid root; {condition.replace('_', ' ')}\n"
                     "Mean across five matched seeds; lower is better")
        midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                ax.text(x, y, f"{values[y, x]:.4f}", ha="center", va="center",
                        color="white" if values[y, x] > midpoint else "black")
        ax.scatter(col, row, marker="*", s=420, facecolors="none",
                   edgecolors="red", linewidths=2.5)
        fig.colorbar(image, ax=ax, label=label)
        fig.savefig(figures / f"{column}.png", dpi=300)
        plt.close(fig)
    print(f"Medoid: {data['selected_root_genome_id'].iloc[0]}")
    print(f"Saved figures and tables: {figures}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("make-jobs", "analyze"))
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    args = parser.parse_args()
    (make_jobs if args.action == "make-jobs" else analyze)(args.condition)


if __name__ == "__main__":
    main()
