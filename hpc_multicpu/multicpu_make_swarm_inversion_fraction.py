#!/usr/bin/env python3
"""Generate the 30-job fixed-total-rate inversion experiment."""

from __future__ import annotations

import argparse
import os
import shlex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atgc-dir", default="ATGC0070")
    parser.add_argument("--tree-filename", default="yuri_gl26/ATGC0070.gl.tre")
    parser.add_argument("--rf", type=float, default=0.1)
    parser.add_argument("--total-rearrangement-rate", type=float, default=0.316227766017)
    parser.add_argument("--core-fraction", type=float, default=0.5)
    parser.add_argument("--core-protection", type=float, default=0.9)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--out-dir", default="hpc_multicpu/results_inversion_fraction_uniform"
    )
    parser.add_argument(
        "--swarm-file", default="hpc_multicpu/jobs_inversion_fraction_uniform.swarm"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    seeds = [42, 1042, 2042, 3042, 4042]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.swarm_file)), exist_ok=True)

    commands = []
    index = 0
    for seed in seeds:
        for fraction in fractions:
            index += 1
            out_csv = os.path.join(
                args.out_dir,
                f"result_{index:03d}_if_{fraction:g}_seed_{seed}.csv",
            )
            command = [
                "python",
                "hpc_multicpu/multicpu_eval_one_setting_inversion_fraction.py",
                "--atgc-dir", args.atgc_dir,
                "--tree-filename", args.tree_filename,
                "--rf", f"{args.rf:.12g}",
                "--total-rearrangement-rate", f"{args.total_rearrangement_rate:.12g}",
                "--inversion-fraction", f"{fraction:.12g}",
                "--translocation-exp", "1e9",
                "--inversion-size-mode", "uniform_breakpoints",
                "--core-fraction", f"{args.core_fraction:.12g}",
                "--core-protection", f"{args.core_protection:.12g}",
                "--n-runs", str(args.n_runs),
                "--workers", str(args.workers),
                "--seed", str(seed),
                "--out-csv", out_csv,
            ]
            commands.append(" ".join(shlex.quote(part) for part in command))

    with open(args.swarm_file, "w", newline="") as handle:
        handle.write("\n".join(commands) + "\n")

    print(f"wrote {len(commands)} jobs to {args.swarm_file}")
    print(f"results directory: {args.out_dir}")


if __name__ == "__main__":
    main()
