#!/usr/bin/env python3
"""Generate Biowulf swarm files for gene-content fitting jobs."""

from __future__ import annotations

import argparse
import math
import os
import shlex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a swarm file for gain/loss-only gene-content fitting."
    )
    parser.add_argument("--atgc-dir", default="ATGC0070")
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--out-dir", default="hpc_multicpu/results_gene_content")
    parser.add_argument("--swarm-file", default="hpc_multicpu/jobs_gene_content.swarm")
    parser.add_argument("--rf-min", type=float, default=1e-2)
    parser.add_argument("--rf-max", type=float, default=1e2)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument(
        "--gain-rates",
        default=None,
        help="Comma-separated gain rates. If set with --loss-rates, runs all pairs.",
    )
    parser.add_argument(
        "--loss-rates",
        default=None,
        help="Comma-separated loss rates. If set with --gain-rates, runs all pairs.",
    )
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--core-fraction", type=float, default=0.0)
    parser.add_argument("--core-protection", type=float, default=0.0)
    parser.add_argument("--terminal-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--internal-rate-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--use-depth-dependent-gain-loss",
        action="store_true",
        help="Apply Y.'s depth-dependent gain/loss multiplier.",
    )
    parser.add_argument("--depth-gain-loss-scale", type=float, default=2700.0)
    parser.add_argument("--depth-gain-loss-decay", type=float, default=3300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=1000)
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


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None or text.strip() == "":
        return None
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Comma-separated value list did not contain any numbers.")
    return values


def safe_label(value: float) -> str:
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    gain_rates = parse_float_list(args.gain_rates)
    loss_rates = parse_float_list(args.loss_rates)

    tied_rf_mode = gain_rates is None and loss_rates is None
    if tied_rf_mode:
        settings = [(rf, rf, rf) for rf in logspace(args.rf_min, args.rf_max, args.points)]
    elif gain_rates is not None and loss_rates is not None:
        settings = [(None, g, l) for g in gain_rates for l in loss_rates]
    else:
        raise SystemExit("Use both --gain-rates and --loss-rates, or neither.")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.swarm_file)), exist_ok=True)

    lines = []
    for job_index, (rf, gain_rate, loss_rate) in enumerate(settings, start=1):
        job_seed = args.seed + args.seed_stride * job_index
        if rf is not None:
            out_name = f"result_{job_index:05d}_rf_{safe_label(rf)}.csv"
        else:
            out_name = (
                f"result_{job_index:05d}_gain_{safe_label(gain_rate)}_"
                f"loss_{safe_label(loss_rate)}.csv"
            )
        out_csv = os.path.join(args.out_dir, out_name)
        command = [
            args.python,
            "hpc_multicpu/multicpu_eval_one_setting_gene_content.py",
            "--atgc-dir", args.atgc_dir,
            "--tree-filename", args.tree_filename,
            "--n-runs", str(args.n_runs),
            "--workers", str(args.workers),
            "--core-fraction", f"{args.core_fraction:.12g}",
            "--core-protection", f"{args.core_protection:.12g}",
            "--terminal-rate-multiplier", f"{args.terminal_rate_multiplier:.12g}",
            "--internal-rate-multiplier", f"{args.internal_rate_multiplier:.12g}",
            "--depth-gain-loss-scale", f"{args.depth_gain_loss_scale:.12g}",
            "--depth-gain-loss-decay", f"{args.depth_gain_loss_decay:.12g}",
            "--seed", str(job_seed),
            "--out-csv", out_csv,
        ]
        if rf is not None:
            command.extend(["--rf", f"{rf:.12g}"])
        else:
            command.extend([
                "--gain-rate", f"{gain_rate:.12g}",
                "--loss-rate", f"{loss_rate:.12g}",
            ])
        if args.use_depth_dependent_gain_loss:
            command.append("--use-depth-dependent-gain-loss")
        lines.append(shell_join(command))

    with open(args.swarm_file, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")

    print(f"wrote {len(lines)} jobs to {args.swarm_file}")
    print(f"results directory: {args.out_dir}")
    print(f"mode: {'tied rf=gain=loss' if tied_rf_mode else 'separate gain/loss'}")
    print(f"workers per job: {args.workers}")


if __name__ == "__main__":
    main()
