#!/usr/bin/env python3
"""Plot real vs simulated synteny distributions for top rf/rt rows.

Example:
    python plot_top_synteny_fits.py \
      --combined-csv hpc_multicpu/combined_full_21x21_n100_t16_423.csv \
      --atgc-dir ATGC0070 \
      --top 5 \
      --n-runs 100 \
      --workers 16 \
      --out-dir top_synteny_fit_plots
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create best-fit synteny plots for the top parameter sets."
    )
    parser.add_argument("--combined-csv", required=True)
    parser.add_argument("--atgc-dir", required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--out-dir", default="top_synteny_fit_plots")
    parser.add_argument(
        "--plot-script",
        default="plot_real_vs_simulated_synteny_modified.py",
        help="Plot script with --workers support.",
    )
    return parser.parse_args()


def read_top_rows(path: str, top: int) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = sorted(rows, key=lambda row: float(row["avg_w1"]))
    return rows[:top]


def safe_float_label(value: str) -> str:
    return f"{float(value):.10g}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_top_rows(args.combined_csv, args.top)
    if not rows:
        raise SystemExit(f"No rows found in {args.combined_csv}")

    for rank, row in enumerate(rows, start=1):
        rf = row["rf"]
        rt = row["rt"]
        avg_w1 = float(row["avg_w1"])
        out_prefix = os.path.join(
            args.out_dir,
            (
                f"rank_{rank:02d}_rf_{safe_float_label(rf)}"
                f"_rt_{safe_float_label(rt)}"
            ),
        )
        cmd = [
            sys.executable,
            args.plot_script,
            "--atgc-dir",
            args.atgc_dir,
            "--rf",
            rf,
            "--rt",
            rt,
            "--n-runs",
            str(args.n_runs),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed + rank - 1),
            "--max-length",
            str(args.max_length),
            "--out-prefix",
            out_prefix,
        ]
        print(
            f"[rank {rank}] rf={float(rf):.10g}, rt={float(rt):.10g}, "
            f"avg_w1={avg_w1:.6g}"
        )
        subprocess.run(cmd, check=True)

    summary_path = os.path.join(args.out_dir, "top_parameter_sets.csv")
    with open(summary_path, "w", newline="") as fh:
        fieldnames = ["rank"] + list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = {"rank": rank}
            out.update(row)
            writer.writerow(out)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
