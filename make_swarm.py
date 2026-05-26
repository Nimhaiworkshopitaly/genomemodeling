#!/usr/bin/env python3
"""Generate a Biowulf swarm file for genome-evolution parameter search."""

from __future__ import annotations

import argparse
import math
import os
import shlex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Biowulf swarm file with one rf/rt evaluation per line."
    )
    parser.add_argument("--atgc-dir", default="ATGC0070")
    parser.add_argument("--out-dir", default="hpc/results")
    parser.add_argument("--swarm-file", default="hpc/jobs.swarm")
    parser.add_argument("--rf-min", type=float, default=1e-2)
    parser.add_argument("--rf-max", type=float, default=1e2)
    parser.add_argument("--rt-min", type=float, default=1e-3)
    parser.add_argument("--rt-max", type=float, default=1e1)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1000,
        help="Offset used to derive a deterministic unique seed for each job.",
    )
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def logspace(min_value: float, max_value: float, points: int) -> list[float]:
    if points < 2:
        return [min_value]
    lo = math.log10(min_value)
    hi = math.log10(max_value)
    step = (hi - lo) / (points - 1)
    return [10 ** (lo + step * i) for i in range(points)]


def main() -> None:
    args = parse_args()
    rf_vals = logspace(args.rf_min, args.rf_max, args.points)
    rt_vals = logspace(args.rt_min, args.rt_max, args.points)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.swarm_file)), exist_ok=True)

    lines = []
    job_index = 0
    for rf in rf_vals:
        for rt in rt_vals:
            job_index += 1
            job_seed = args.seed + args.seed_stride * job_index
            out_name = f"result_{job_index:05d}_rf_{rf:.4g}_rt_{rt:.4g}.csv"
            out_csv = os.path.join(args.out_dir, out_name)
            command = [
                args.python,
                "hpc/eval_one_setting.py",
                "--atgc-dir", args.atgc_dir,
                "--rf", f"{rf:.12g}",
                "--rt", f"{rt:.12g}",
                "--n-runs", str(args.n_runs),
                "--seed", str(job_seed),
                "--out-csv", out_csv,
            ]
            lines.append(shell_join(command))

    with open(args.swarm_file, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")

    print(f"wrote {len(lines)} jobs to {args.swarm_file}")
    print(f"results directory: {args.out_dir}")


if __name__ == "__main__":
    main()
