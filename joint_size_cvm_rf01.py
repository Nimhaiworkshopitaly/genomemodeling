#!/usr/bin/env python3
"""Run the rf=0.1 joint-size grid with CvM and Anderson-Darling scoring."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import joint_size_full_factorial_observed_medoid as full


RF = 0.1
EXPECTED = 51_750
RESULTS = Path(
    "hpc_multicpu/results_joint_size_cvm_rf_0p1_observed_medoid_empirical_core"
)
JOB_DIR = Path("hpc_multicpu/joint_size_cvm_rf_0p1_parts")
FIGURES = Path(
    "hpc_multicpu/figures_joint_size_cvm_rf_0p1_observed_medoid_empirical_core"
)
METRICS = (
    "avg_cramer_von_mises_distance",
    "avg_anderson_darling_distance",
)


def settings():
    for gain in full.gain_configurations():
        for trans in full.translocation_configurations():
            for inv in full.inversion_configurations():
                for trans_rate in full.TRANSLOCATION_RATES:
                    for inv_rate in full.INVERSION_RATES:
                        for seed in full.SEEDS:
                            yield gain, trans, inv, trans_rate, inv_rate, seed


def result_path(gain, trans, inv, trans_rate, inv_rate, seed):
    return RESULTS / (
        f"result_rf_0p1_{gain['label']}_trans_{trans['label']}_"
        f"inv_{inv['label']}_t_{full.token(trans_rate)}_"
        f"i_{full.token(inv_rate)}_seed_{seed}.csv"
    )


def write_parts(commands, jobs_per_part):
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    for old in JOB_DIR.glob("part_*.swarm"):
        old.unlink()
    parts = []
    for start in range(0, len(commands), jobs_per_part):
        path = JOB_DIR / f"part_{len(parts) + 1:03d}.swarm"
        path.write_text("\n".join(commands[start:start + jobs_per_part]) + "\n")
        parts.append(path)
    return parts


def make_jobs(jobs_per_part):
    evaluator = Path("hpc_multicpu/multicpu_eval_one_setting_block_translocation.py")
    if not evaluator.is_file():
        raise SystemExit("Run this command from the genomemodeling root.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    commands = []
    skipped = 0
    for gain, trans, inv, trans_rate, inv_rate, seed in settings():
        output = result_path(gain, trans, inv, trans_rate, inv_rate, seed)
        if output.is_file() and output.stat().st_size > 0:
            skipped += 1
            continue
        total = trans_rate + inv_rate
        arguments = [
            sys.executable, str(evaluator.resolve()),
            "--atgc-dir", "ATGC0070",
            "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
            "--root-mode", "observed_medoid",
            "--rf", "0.1",
            "--total-rearrangement-rate", f"{total:.12g}",
            "--inversion-fraction", f"{inv_rate / total:.12g}",
            "--translocation-size-mode", trans["mode"],
            "--translocation-min-size", "1",
            "--translocation-exp", "3",
            "--translocation-geometric-mean", "2",
            "--translocation-lognormal-median", "3",
            "--translocation-lognormal-sigma", "1",
            "--inversion-size-mode", inv["mode"],
            "--inversion-min-size", "2",
            "--inversion-exp", "3",
            "--inversion-geometric-mean", "10",
            "--inversion-lognormal-median", "10",
            "--inversion-lognormal-sigma", "1",
            "--gain-size-mode", gain["mode"],
            "--gain-min-size", "1",
            "--gain-loss-exp", "1e9",
            "--core-fraction", "0",
            "--core-protection", "0.9",
            "--core-mode", "empirical",
            "--core-prevalence", "1.0",
            "--n-runs", "100", "--workers", "16",
            "--seed", str(seed), "--out-csv", str(output),
        ]
        arguments.extend(trans["args"])
        arguments.extend(inv["args"])
        arguments.extend(gain["args"])
        commands.append(shlex.join(arguments))
    parts = write_parts(commands, jobs_per_part)
    print(f"Wrote {len(commands)} jobs in {len(parts)} parts: {JOB_DIR}")
    print(f"Skipped {skipped}; complete rf=0.1 CvM grid = {EXPECTED} jobs.")


def status():
    counts = {gain["label"]: 0 for gain in full.gain_configurations()}
    for gain, trans, inv, trans_rate, inv_rate, seed in settings():
        path = result_path(gain, trans, inv, trans_rate, inv_rate, seed)
        if path.is_file() and path.stat().st_size > 0:
            counts[gain["label"]] += 1
    for gain, count in counts.items():
        print(f"{gain}: {count}/25875 completed")
    print(f"total: {sum(counts.values())}/{EXPECTED} completed")


def analyze():
    files = sorted(RESULTS.glob("result_*.csv"))
    if len(files) != EXPECTED:
        raise SystemExit(f"Expected {EXPECTED} files, found {len(files)}")
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    missing = [metric for metric in METRICS if metric not in data]
    if missing:
        raise ValueError(f"Results do not contain: {', '.join(missing)}")
    data["gain_model"] = np.where(
        data["gain_size_mode"].eq("single_gene"),
        "Single-gene gain", "HGT geometric mean=3",
    )
    data["translocation_model"] = data.apply(
        full.model_label, axis=1, prefix="translocation"
    )
    data["inversion_model"] = data.apply(
        full.model_label, axis=1, prefix="inversion"
    )
    keys = [
        "rf", "gain_model", "translocation_model", "inversion_model",
        "translocation_rate", "inversion_rate",
    ]
    means = data.groupby(keys, as_index=False)[list(METRICS)].mean()
    FIGURES.mkdir(parents=True, exist_ok=True)
    means.to_csv(FIGURES / "mean_distribution_results.csv", index=False)
    means.to_csv(FIGURES / "mean_cvm_results.csv", index=False)
    best_rows = []
    for metric in METRICS:
        selected = means.loc[means.groupby("gain_model")[metric].idxmin()].copy()
        selected.insert(2, "metric", metric)
        best_rows.append(selected)
    best = pd.concat(best_rows, ignore_index=True).sort_values(
        ["metric", "gain_model"]
    )
    best.to_csv(FIGURES / "best_distribution_models.csv", index=False)
    best[best["metric"].eq(METRICS[0])].to_csv(
        FIGURES / "best_cvm_models.csv", index=False
    )
    print(best.to_string(index=False))
    print(f"Saved summaries: {FIGURES}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("make-jobs", "status", "analyze"))
    parser.add_argument("--jobs-per-part", type=int, default=1000)
    args = parser.parse_args()
    if args.jobs_per_part < 1:
        parser.error("--jobs-per-part must be positive")
    if args.action == "make-jobs":
        make_jobs(args.jobs_per_part)
    elif args.action == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
