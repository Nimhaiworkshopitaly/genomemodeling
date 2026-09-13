#!/usr/bin/env python3
"""Build the combined observed-medoid inversion-size parameter sweep."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

GAIN_LOSS_RATES = [0.1, 0.2, 0.3]
TRANSLOCATION_RATES = [0.25, 0.30, 0.35]
INVERSION_RATES = [0.0, 0.005, 0.01, 0.02, 0.05]
SEEDS = [42, 1042, 2042, 3042, 4042]

GEOMETRIC_MEANS = [3, 5, 10, 20, 50]
POWERLAW_ALPHAS = [1.5, 2.0, 2.5, 3.0, 4.0]
LOGNORMAL_MEDIANS = [3, 5, 10, 20]
LOGNORMAL_SIGMAS = [0.5, 1.0, 1.5]

RESULTS = Path(
    "hpc_multicpu/results_inversion_size_parameter_grid_observed_medoid_"
    "empirical_core"
)
SWARM = Path(
    "hpc_multicpu/jobs_inversion_size_parameter_grid_observed_medoid_"
    "empirical_core.swarm"
)


def number_token(value):
    return f"{value:g}".replace(".", "p")


def configurations():
    for mean in GEOMETRIC_MEANS:
        yield {
            "model": "geometric",
            "label": f"geometric_mean_{number_token(mean)}",
            "arguments": ["--inversion-geometric-mean", f"{mean:g}"],
        }
    for alpha in POWERLAW_ALPHAS:
        yield {
            "model": "powerlaw",
            "label": f"powerlaw_alpha_{number_token(alpha)}",
            "arguments": ["--inversion-exp", f"{alpha:g}"],
        }
    for median in LOGNORMAL_MEDIANS:
        for sigma in LOGNORMAL_SIGMAS:
            yield {
                "model": "lognormal",
                "label": (
                    f"lognormal_median_{number_token(median)}_"
                    f"sigma_{number_token(sigma)}"
                ),
                "arguments": [
                    "--inversion-lognormal-median", f"{median:g}",
                    "--inversion-lognormal-sigma", f"{sigma:g}",
                ],
            }
    yield {
        "model": "uniform_breakpoints",
        "label": "uniform_breakpoints",
        "arguments": [],
    }


def result_path(rf, configuration, translocation_rate, inversion_rate, seed):
    return RESULTS / (
        f"result_rf_{number_token(rf)}_{configuration['label']}_"
        f"t_{number_token(translocation_rate)}_"
        f"i_{number_token(inversion_rate)}_seed_{seed}.csv"
    )


def make_jobs():
    evaluator = Path(
        "hpc_multicpu/multicpu_eval_one_setting_inversion_fraction.py"
    )
    if not evaluator.is_file():
        raise SystemExit("Run this command from the genomemodeling root.")

    configs = list(configurations())
    if len(configs) != 23:
        raise RuntimeError(f"Expected 23 size configurations, found {len(configs)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    commands = []
    skipped = 0
    for rf in GAIN_LOSS_RATES:
        for config in configs:
            for translocation_rate in TRANSLOCATION_RATES:
                for inversion_rate in INVERSION_RATES:
                    total = translocation_rate + inversion_rate
                    fraction = inversion_rate / total
                    for seed in SEEDS:
                        output = result_path(
                            rf, config, translocation_rate, inversion_rate, seed
                        )
                        if output.exists() and output.stat().st_size > 0:
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
                            "--translocation-exp", "1e9",
                            "--inversion-exp", "3",
                            "--inversion-size-mode", config["model"],
                            "--inversion-geometric-mean", "10",
                            "--inversion-lognormal-median", "10",
                            "--inversion-lognormal-sigma", "1",
                            "--inversion-min-size", "2",
                            "--gain-loss-exp", "1e9",
                            "--core-fraction", "0",
                            "--core-protection", "0.9",
                            "--core-mode", "empirical",
                            "--core-prevalence", "1.0",
                            "--n-runs", "100",
                            "--workers", "16",
                            "--seed", str(seed),
                            "--out-csv", str(output),
                        ]
                        arguments.extend(config["arguments"])
                        commands.append(shlex.join(arguments))

    SWARM.write_text("\n".join(commands) + ("\n" if commands else ""))
    print(f"Size configurations: {len(configs)}")
    print(f"Wrote {len(commands)} jobs: {SWARM}")
    print(f"Skipped {skipped} existing results; full grid = 5175 jobs.")


def status():
    configs = list(configurations())
    completed_total = 0
    for rf in GAIN_LOSS_RATES:
        completed = 0
        for config in configs:
            for translocation_rate in TRANSLOCATION_RATES:
                for inversion_rate in INVERSION_RATES:
                    for seed in SEEDS:
                        path = result_path(
                            rf, config, translocation_rate, inversion_rate, seed
                        )
                        if path.is_file() and path.stat().st_size > 0:
                            completed += 1
        completed_total += completed
        print(f"gain/loss rate {rf:g}: {completed}/1725 completed")
    print(f"total: {completed_total}/5175 completed")


def main():
    parser = argparse.ArgumentParser(
        description="Combined inversion-size parameter sweep."
    )
    parser.add_argument("action", choices=("make-jobs", "status"))
    args = parser.parse_args()
    (make_jobs if args.action == "make-jobs" else status)()


if __name__ == "__main__":
    main()
