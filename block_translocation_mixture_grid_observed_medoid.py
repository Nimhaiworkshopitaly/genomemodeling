#!/usr/bin/env python3
"""Fit block-translocation sizes in a rearrangement/HGT mixture."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

TRANSLOCATION_RATES = [0.25, 0.30, 0.35]
INVERSION_RATES = [0.0, 0.005, 0.01, 0.02, 0.05]
SEEDS = [42, 1042, 2042, 3042, 4042]
GEOMETRIC_MEANS = [2, 3, 5, 10]
POWERLAW_ALPHAS = [1.5, 2.0, 2.5, 3.0]
LOGNORMAL_MEDIANS = [2, 5, 10]
LOGNORMAL_SIGMAS = [0.5, 1.0]


def token(value):
    return f"{value:g}".replace(".", "p")


def translocation_configurations():
    yield {"mode": "single_gene", "label": "single_gene", "args": []}
    for mean in GEOMETRIC_MEANS:
        yield {
            "mode": "geometric",
            "label": f"geometric_mean_{token(mean)}",
            "args": ["--translocation-geometric-mean", f"{mean:g}"],
        }
    for alpha in POWERLAW_ALPHAS:
        yield {
            "mode": "powerlaw",
            "label": f"powerlaw_alpha_{token(alpha)}",
            "args": ["--translocation-exp", f"{alpha:g}"],
        }
    for median in LOGNORMAL_MEDIANS:
        for sigma in LOGNORMAL_SIGMAS:
            yield {
                "mode": "lognormal",
                "label": (
                    f"lognormal_median_{token(median)}_sigma_{token(sigma)}"
                ),
                "args": [
                    "--translocation-lognormal-median", f"{median:g}",
                    "--translocation-lognormal-sigma", f"{sigma:g}",
                ],
            }


def gain_configurations():
    yield {
        "mode": "single_gene",
        "label": "single_gene_gain",
        "args": [],
    }
    yield {
        "mode": "geometric",
        "label": "hgt_geometric_mean_3",
        "args": ["--gain-geometric-mean", "3"],
    }


def locations(rf):
    suffix = f"rf_{token(rf)}"
    stem = f"block_translocation_mixture_observed_medoid_empirical_core_{suffix}"
    return (
        Path("hpc_multicpu") / f"results_{stem}",
        Path("hpc_multicpu") / f"jobs_{stem}.swarm",
    )


def result_path(results, rf, gain, trans, trans_rate, inv_rate, seed):
    return results / (
        f"result_rf_{token(rf)}_{gain['label']}_{trans['label']}_"
        f"t_{token(trans_rate)}_i_{token(inv_rate)}_seed_{seed}.csv"
    )


def make_jobs(rf):
    evaluator = Path(
        "hpc_multicpu/multicpu_eval_one_setting_block_translocation.py"
    )
    if not evaluator.is_file():
        raise SystemExit("Run this command from the genomemodeling root.")
    trans_configs = list(translocation_configurations())
    gain_configs = list(gain_configurations())
    if len(trans_configs) != 15 or len(gain_configs) != 2:
        raise RuntimeError("Unexpected number of size configurations")

    results, swarm = locations(rf)
    results.mkdir(parents=True, exist_ok=True)
    commands = []
    skipped = 0
    for gain in gain_configs:
        for trans in trans_configs:
            for trans_rate in TRANSLOCATION_RATES:
                for inv_rate in INVERSION_RATES:
                    total = trans_rate + inv_rate
                    fraction = inv_rate / total
                    for seed in SEEDS:
                        output = result_path(
                            results, rf, gain, trans, trans_rate, inv_rate, seed
                        )
                        if output.is_file() and output.stat().st_size > 0:
                            skipped += 1
                            continue
                        arguments = [
                            sys.executable, str(evaluator.resolve()),
                            "--atgc-dir", "ATGC0070",
                            "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
                            "--root-mode", "observed_medoid",
                            "--rf", f"{rf:g}",
                            "--total-rearrangement-rate", f"{total:.12g}",
                            "--inversion-fraction", f"{fraction:.12g}",
                            "--translocation-size-mode", trans["mode"],
                            "--translocation-min-size", "1",
                            "--translocation-exp", "3",
                            "--translocation-geometric-mean", "2",
                            "--translocation-lognormal-median", "3",
                            "--translocation-lognormal-sigma", "1",
                            "--inversion-size-mode", "powerlaw",
                            "--inversion-exp", "3",
                            "--inversion-min-size", "2",
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
                        arguments.extend(gain["args"])
                        commands.append(shlex.join(arguments))

    swarm.write_text("\n".join(commands) + ("\n" if commands else ""))
    print(f"Translocation configurations: {len(trans_configs)}")
    print(f"Gain configurations: {len(gain_configs)}")
    print(f"Wrote {len(commands)} jobs: {swarm}")
    print(f"Skipped {skipped}; full grid = 2250 jobs.")


def status(rf):
    results, _ = locations(rf)
    completed = sum(
        1 for path in results.glob("result_*.csv")
        if path.is_file() and path.stat().st_size > 0
    ) if results.is_dir() else 0
    print(f"gain/loss rate {rf:g}: {completed}/2250 completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("make-jobs", "status"))
    parser.add_argument("--rf", type=float, default=0.1)
    args = parser.parse_args()
    if args.rf <= 0:
        parser.error("--rf must be positive")
    (make_jobs if args.action == "make-jobs" else status)(args.rf)


if __name__ == "__main__":
    main()
