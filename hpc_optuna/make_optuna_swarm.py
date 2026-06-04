#!/usr/bin/env python3
"""Generate a Biowulf swarm file for parallel Optuna workers."""

from __future__ import annotations

import argparse
import os
import shlex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Optuna worker swarm.")
    parser.add_argument("--atgc-dir", default="ATGC0070")
    parser.add_argument("--study-name", default="atgc0070_rf_rt")
    parser.add_argument("--storage", default="sqlite:///hpc_optuna/optuna_atgc0070.db")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--trials-per-worker", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--rf-min", type=float, default=1e-2)
    parser.add_argument("--rf-max", type=float, default=1e2)
    parser.add_argument("--rt-min", type=float, default=1e-3)
    parser.add_argument("--rt-max", type=float, default=1e1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swarm-file", default="hpc_optuna/jobs_optuna_n100.swarm")
    parser.add_argument("--out-dir", default="hpc_optuna/results")
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.swarm_file)), exist_ok=True)

    lines = []
    for worker_idx in range(args.n_workers):
        worker_id = f"worker_{worker_idx:03d}"
        out_csv = os.path.join(args.out_dir, f"{worker_id}.csv")
        cmd = [
            args.python,
            "hpc_optuna/optuna_worker.py",
            "--atgc-dir", args.atgc_dir,
            "--study-name", args.study_name,
            "--storage", args.storage,
            "--rf-min", str(args.rf_min),
            "--rf-max", str(args.rf_max),
            "--rt-min", str(args.rt_min),
            "--rt-max", str(args.rt_max),
            "--n-runs", str(args.n_runs),
            "--n-trials", str(args.trials_per_worker),
            "--seed", str(args.seed + worker_idx),
            "--worker-id", worker_id,
            "--out-csv", out_csv,
        ]
        lines.append(shell_join(cmd))

    with open(args.swarm_file, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"wrote {len(lines)} workers to {args.swarm_file}")
    print(f"total planned trials: {args.n_workers * args.trials_per_worker}")
    print(f"results directory: {args.out_dir}")
    print(f"storage: {args.storage}")


if __name__ == "__main__":
    main()
