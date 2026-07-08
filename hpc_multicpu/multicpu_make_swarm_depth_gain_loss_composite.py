#!/usr/bin/env python3
"""Generate a Biowulf swarm file for multi-CPU parameter jobs."""

from __future__ import annotations

import argparse
import math
import os
import shlex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a swarm file for depth-dependent gain/loss jobs."
    )
    parser.add_argument("--atgc-dir", default="ATGC0070")
    parser.add_argument("--out-dir", default="hpc_multicpu/results_depth_gain_loss_composite")
    parser.add_argument(
        "--swarm-file",
        default="hpc_multicpu/jobs_depth_gain_loss_composite.swarm",
    )
    parser.add_argument("--rf-min", type=float, default=1e-2)
    parser.add_argument("--rf-max", type=float, default=1e2)
    parser.add_argument("--rt-min", type=float, default=1e-3)
    parser.add_argument("--rt-max", type=float, default=1e1)
    parser.add_argument("--points", type=int, default=21)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--core-fraction", type=float, default=0.5)
    parser.add_argument("--core-protection", type=float, default=0.9)
    parser.add_argument("--terminal-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--internal-rate-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--disable-depth-dependent-gain-loss",
        action="store_false",
        dest="use_depth_dependent_gain_loss",
        help="Disable Y.'s depth-dependent gain/loss multiplier.",
    )
    parser.set_defaults(use_depth_dependent_gain_loss=True)
    parser.add_argument("--depth-gain-loss-scale", type=float, default=2700.0)
    parser.add_argument("--depth-gain-loss-decay", type=float, default=3300.0)
    parser.add_argument(
        "--core-fractions",
        default=None,
        help=(
            "Comma-separated core-fraction values, e.g. 0.2,0.4,0.6,0.8. "
            "If set, overrides --core-fraction."
        ),
    )
    parser.add_argument(
        "--core-protections",
        default=None,
        help=(
            "Comma-separated core-protection values, e.g. 0.5,0.7,0.9,0.99. "
            "If set, overrides --core-protection."
        ),
    )
    parser.add_argument(
        "--terminal-rate-multipliers",
        default=None,
        help="Comma-separated terminal-branch multipliers, e.g. 0.5,1,2.",
    )
    parser.add_argument(
        "--internal-rate-multipliers",
        default=None,
        help="Comma-separated internal-branch multipliers, e.g. 0.5,1,2.",
    )
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


def parse_float_list(text: str | None, fallback: float) -> list[float]:
    if text is None or text.strip() == "":
        return [fallback]
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Comma-separated value list did not contain any numbers.")
    return values


def main() -> None:
    args = parse_args()
    rf_vals = logspace(args.rf_min, args.rf_max, args.points)
    rt_vals = logspace(args.rt_min, args.rt_max, args.points)
    core_fraction_vals = parse_float_list(args.core_fractions, args.core_fraction)
    core_protection_vals = parse_float_list(args.core_protections, args.core_protection)
    terminal_multiplier_vals = parse_float_list(
        args.terminal_rate_multipliers,
        args.terminal_rate_multiplier,
    )
    internal_multiplier_vals = parse_float_list(
        args.internal_rate_multipliers,
        args.internal_rate_multiplier,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.swarm_file)), exist_ok=True)

    lines = []
    job_index = 0
    for rf in rf_vals:
        for rt in rt_vals:
            for core_fraction in core_fraction_vals:
                for core_protection in core_protection_vals:
                    for terminal_multiplier in terminal_multiplier_vals:
                        for internal_multiplier in internal_multiplier_vals:
                            job_index += 1
                            job_seed = args.seed + args.seed_stride * job_index
                            out_name = (
                                f"result_{job_index:05d}_rf_{rf:.4g}_rt_{rt:.4g}_"
                                f"cf_{core_fraction:.4g}_cp_{core_protection:.4g}_"
                                f"tm_{terminal_multiplier:.4g}_im_{internal_multiplier:.4g}.csv"
                            )
                            out_csv = os.path.join(args.out_dir, out_name)
                            command = [
                                args.python,
                                "hpc_multicpu/multicpu_eval_one_setting_depth_gain_loss_composite.py",
                                "--atgc-dir", args.atgc_dir,
                                "--rf", f"{rf:.12g}",
                                "--rt", f"{rt:.12g}",
                                "--n-runs", str(args.n_runs),
                                "--workers", str(args.workers),
                                "--core-fraction", f"{core_fraction:.12g}",
                                "--core-protection", f"{core_protection:.12g}",
                                "--terminal-rate-multiplier", f"{terminal_multiplier:.12g}",
                                "--internal-rate-multiplier", f"{internal_multiplier:.12g}",
                                "--depth-gain-loss-scale", f"{args.depth_gain_loss_scale:.12g}",
                                "--depth-gain-loss-decay", f"{args.depth_gain_loss_decay:.12g}",
                                "--seed", str(job_seed),
                                "--out-csv", out_csv,
                            ]
                            if not args.use_depth_dependent_gain_loss:
                                command.append("--disable-depth-dependent-gain-loss")
                            lines.append(shell_join(command))

    with open(args.swarm_file, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")

    print(f"wrote {len(lines)} jobs to {args.swarm_file}")
    print(f"results directory: {args.out_dir}")
    print(f"rf points: {len(rf_vals)}")
    print(f"rt points: {len(rt_vals)}")
    print(f"core_fraction values: {', '.join(f'{x:g}' for x in core_fraction_vals)}")
    print(f"core_protection values: {', '.join(f'{x:g}' for x in core_protection_vals)}")
    print(
        "terminal_rate_multiplier values: "
        f"{', '.join(f'{x:g}' for x in terminal_multiplier_vals)}"
    )
    print(
        "internal_rate_multiplier values: "
        f"{', '.join(f'{x:g}' for x in internal_multiplier_vals)}"
    )
    print(f"use_depth_dependent_gain_loss: {int(args.use_depth_dependent_gain_loss)}")
    print(f"depth_gain_loss_scale: {args.depth_gain_loss_scale:g}")
    print(f"depth_gain_loss_decay: {args.depth_gain_loss_decay:g}")
    print(f"workers per job: {args.workers}")


if __name__ == "__main__":
    main()
