#!/usr/bin/env python3
"""Create a small Biowulf grid for the rearrangement-size experiment."""

from __future__ import annotations

import argparse
import os
import shlex


def comma_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one numeric value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atgc-dir", default="ATGC0070")
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--rf", type=float, required=True)
    parser.add_argument("--rt", type=float, required=True)
    parser.add_argument(
        "--rearrangement-exps", type=comma_floats, default=comma_floats("1.2,1.5,2,3,1e9")
    )
    parser.add_argument(
        "--inversion-multipliers", type=comma_floats,
        default=comma_floats("0,0.1,0.5,1")
    )
    parser.add_argument("--core-fraction", type=float, default=0.5)
    parser.add_argument("--core-protection", type=float, default=0.9)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", default="hpc_multicpu/results_rearrangement_size"
    )
    parser.add_argument(
        "--swarm-file", default="hpc_multicpu/jobs_rearrangement_size.swarm"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.swarm_file)), exist_ok=True)
    commands = []

    for index, (exponent, inv_multiplier) in enumerate(
        (
            (exponent, multiplier)
            for exponent in args.rearrangement_exps
            for multiplier in args.inversion_multipliers
        ),
        start=1,
    ):
        out_csv = os.path.join(
            args.out_dir,
            f"result_{index:03d}_rexp_{exponent:g}_im_{inv_multiplier:g}.csv",
        )
        command = [
            "python", "hpc_multicpu/multicpu_eval_one_setting_rearrangement_size.py",
            "--atgc-dir", args.atgc_dir,
            "--tree-filename", args.tree_filename,
            "--rf", f"{args.rf:.12g}",
            "--rt", f"{args.rt:.12g}",
            "--rearrangement-exp", f"{exponent:.12g}",
            "--inversion-multiplier", f"{inv_multiplier:.12g}",
            "--core-fraction", f"{args.core_fraction:.12g}",
            "--core-protection", f"{args.core_protection:.12g}",
            "--n-runs", str(args.n_runs),
            "--workers", str(args.workers),
            "--seed", str(args.seed + index * 1000),
            "--out-csv", out_csv,
        ]
        commands.append(" ".join(shlex.quote(part) for part in command))

    with open(args.swarm_file, "w", newline="") as handle:
        handle.write("\n".join(commands) + "\n")

    print(f"wrote {len(commands)} jobs to {args.swarm_file}")
    print(f"results directory: {args.out_dir}")


if __name__ == "__main__":
    main()
