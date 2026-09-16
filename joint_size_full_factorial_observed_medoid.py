#!/usr/bin/env python3
"""Run and summarize the full translocation-size x inversion-size grid."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd


GAIN_LOSS_RATES = [0.1, 0.2, 0.3]
TRANSLOCATION_RATES = [0.25, 0.30, 0.35]
INVERSION_RATES = [0.0, 0.005, 0.01, 0.02, 0.05]
SEEDS = [42, 1042, 2042, 3042, 4042]

TRANSLOCATION_GEOMETRIC_MEANS = [2, 3, 5, 10]
TRANSLOCATION_POWERLAW_ALPHAS = [1.5, 2.0, 2.5, 3.0]
TRANSLOCATION_LOGNORMAL_MEDIANS = [2, 5, 10]
TRANSLOCATION_LOGNORMAL_SIGMAS = [0.5, 1.0]

INVERSION_GEOMETRIC_MEANS = [3, 5, 10, 20, 50]
INVERSION_POWERLAW_ALPHAS = [1.5, 2.0, 2.5, 3.0, 4.0]
INVERSION_LOGNORMAL_MEDIANS = [3, 5, 10, 20]
INVERSION_LOGNORMAL_SIGMAS = [0.5, 1.0, 1.5]

RESULTS = Path(
    "hpc_multicpu/results_joint_size_full_factorial_observed_medoid_empirical_core"
)
JOB_DIR = Path("hpc_multicpu/joint_size_full_factorial_parts")
FIGURES = Path(
    "hpc_multicpu/figures_joint_size_full_factorial_observed_medoid_empirical_core"
)
METRICS = {
    "composite_score": "Composite score",
    "avg_ks_statistic": "KS statistic",
    "avg_kuiper_statistic": "Kuiper statistic",
}
EXPECTED = 155_250


def token(value):
    return f"{value:g}".replace(".", "p")


def translocation_configurations():
    yield {"mode": "single_gene", "label": "single_gene", "args": []}
    for mean in TRANSLOCATION_GEOMETRIC_MEANS:
        yield {
            "mode": "geometric",
            "label": f"geometric_mean_{token(mean)}",
            "args": ["--translocation-geometric-mean", f"{mean:g}"],
        }
    for alpha in TRANSLOCATION_POWERLAW_ALPHAS:
        yield {
            "mode": "powerlaw",
            "label": f"powerlaw_alpha_{token(alpha)}",
            "args": ["--translocation-exp", f"{alpha:g}"],
        }
    for median in TRANSLOCATION_LOGNORMAL_MEDIANS:
        for sigma in TRANSLOCATION_LOGNORMAL_SIGMAS:
            yield {
                "mode": "lognormal",
                "label": f"lognormal_median_{token(median)}_sigma_{token(sigma)}",
                "args": [
                    "--translocation-lognormal-median", f"{median:g}",
                    "--translocation-lognormal-sigma", f"{sigma:g}",
                ],
            }


def inversion_configurations():
    for mean in INVERSION_GEOMETRIC_MEANS:
        yield {
            "mode": "geometric",
            "label": f"geometric_mean_{token(mean)}",
            "args": ["--inversion-geometric-mean", f"{mean:g}"],
        }
    for alpha in INVERSION_POWERLAW_ALPHAS:
        yield {
            "mode": "powerlaw",
            "label": f"powerlaw_alpha_{token(alpha)}",
            "args": ["--inversion-exp", f"{alpha:g}"],
        }
    for median in INVERSION_LOGNORMAL_MEDIANS:
        for sigma in INVERSION_LOGNORMAL_SIGMAS:
            yield {
                "mode": "lognormal",
                "label": f"lognormal_median_{token(median)}_sigma_{token(sigma)}",
                "args": [
                    "--inversion-lognormal-median", f"{median:g}",
                    "--inversion-lognormal-sigma", f"{sigma:g}",
                ],
            }
    yield {"mode": "uniform_breakpoints", "label": "uniform_breakpoints", "args": []}


def gain_configurations():
    yield {"mode": "single_gene", "label": "single_gain", "args": []}
    yield {
        "mode": "geometric",
        "label": "hgt_geometric_mean_3",
        "args": ["--gain-geometric-mean", "3"],
    }


def result_path(rf, gain, trans, inv, trans_rate, inv_rate, seed):
    return RESULTS / (
        f"result_rf_{token(rf)}_{gain['label']}_trans_{trans['label']}_"
        f"inv_{inv['label']}_t_{token(trans_rate)}_i_{token(inv_rate)}_"
        f"seed_{seed}.csv"
    )


def all_settings():
    trans_configs = list(translocation_configurations())
    inv_configs = list(inversion_configurations())
    gains = list(gain_configurations())
    if (len(trans_configs), len(inv_configs), len(gains)) != (15, 23, 2):
        raise RuntimeError("Unexpected model count")
    for rf in GAIN_LOSS_RATES:
        for gain in gains:
            for trans in trans_configs:
                for inv in inv_configs:
                    for trans_rate in TRANSLOCATION_RATES:
                        for inv_rate in INVERSION_RATES:
                            for seed in SEEDS:
                                yield rf, gain, trans, inv, trans_rate, inv_rate, seed


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
    for rf, gain, trans, inv, trans_rate, inv_rate, seed in all_settings():
        output = result_path(rf, gain, trans, inv, trans_rate, inv_rate, seed)
        if output.is_file() and output.stat().st_size > 0:
            skipped += 1
            continue
        total = trans_rate + inv_rate
        arguments = [
            sys.executable, str(evaluator.resolve()),
            "--atgc-dir", "ATGC0070",
            "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
            "--root-mode", "observed_medoid",
            "--rf", f"{rf:g}",
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
    print("Translocation models: 15")
    print("Inversion models: 23")
    print("Gain modes: 2")
    print(f"Wrote {len(commands)} jobs in {len(parts)} parts: {JOB_DIR}")
    print(f"Skipped {skipped}; full factorial grid = {EXPECTED} jobs.")


def status():
    counts = {(rf, gain["label"]): 0 for rf in GAIN_LOSS_RATES
              for gain in gain_configurations()}
    total = 0
    for setting in all_settings():
        rf, gain, trans, inv, trans_rate, inv_rate, seed = setting
        path = result_path(rf, gain, trans, inv, trans_rate, inv_rate, seed)
        if path.is_file() and path.stat().st_size > 0:
            counts[(rf, gain["label"])] += 1
            total += 1
    expected_group = EXPECTED // 6
    for (rf, gain), completed in counts.items():
        print(f"rf={rf:g}, {gain}: {completed}/{expected_group} completed")
    print(f"total: {total}/{EXPECTED} completed")


def analyze():
    files = sorted(RESULTS.glob("result_*.csv"))
    if len(files) != EXPECTED:
        raise SystemExit(
            f"Expected {EXPECTED} result files, found {len(files)}. Run status first."
        )
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    required = list(METRICS) + [
        "rf", "gain_size_mode", "translocation_rate", "inversion_rate",
        "translocation_size_mode", "inversion_size_mode", "seed",
    ]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing result columns: {missing}")
    if len(data) != EXPECTED:
        raise ValueError(f"Expected {EXPECTED} rows, found {len(data)}")

    data["gain_model"] = np.where(
        data["gain_size_mode"].eq("single_gene"),
        "Single-gene gain", "HGT geometric mean=3",
    )
    data["translocation_model"] = data.apply(model_label, axis=1, prefix="translocation")
    data["inversion_model"] = data.apply(model_label, axis=1, prefix="inversion")
    keys = [
        "rf", "gain_model", "translocation_model", "inversion_model",
        "translocation_rate", "inversion_rate",
    ]
    means = data.groupby(keys, as_index=False)[list(METRICS)].mean()
    FIGURES.mkdir(parents=True, exist_ok=True)
    means.to_csv(FIGURES / "mean_results.csv", index=False)

    best_rows = []
    for rf in GAIN_LOSS_RATES:
        for gain in sorted(means["gain_model"].unique()):
            subset = means[(means["rf"].eq(rf)) & (means["gain_model"].eq(gain))]
            for metric in METRICS:
                best = subset.loc[subset[metric].idxmin()].copy()
                best["metric"] = metric
                best_rows.append(best)
    best = pd.DataFrame(best_rows)
    best.to_csv(FIGURES / "best_joint_models_by_condition_and_metric.csv", index=False)
    print(f"Saved joint-model summaries: {FIGURES}")


def model_label(row, prefix):
    mode = row[f"{prefix}_size_mode"]
    if mode == "single_gene":
        return "Single gene"
    if mode == "uniform_breakpoints":
        return "Uniform breakpoints"
    if mode == "geometric":
        return f"Geometric mean={row[f'{prefix}_geometric_mean']:g}"
    if mode == "powerlaw":
        return f"Power law alpha={row[f'{prefix}_exp']:g}"
    if mode == "lognormal":
        return (
            f"Log-normal median={row[f'{prefix}_lognormal_median']:g}, "
            f"sigma={row[f'{prefix}_lognormal_sigma']:g}"
        )
    raise ValueError(f"Unknown {prefix} size mode: {mode}")


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
