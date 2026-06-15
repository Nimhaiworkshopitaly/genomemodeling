#!/usr/bin/env python3
"""Combine per-job CSV files and rank parameter settings by score."""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Biowulf swarm result CSVs into one ranked table."
    )
    parser.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    parser.add_argument("--out-csv", default="hpc/combined_results.csv")
    parser.add_argument(
        "--sort-by",
        default="composite_score",
        choices=["composite_score", "avg_w1", "sum_w1"],
    )
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[str]:
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        files.extend(matches if matches else [pattern])
    return sorted(set(files))


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("No input files found.")

    frames = []
    for path in files:
        if not os.path.exists(path):
            print(f"skipping missing file: {path}")
            continue
        df = pd.read_csv(path)
        df["source_file"] = path
        frames.append(df)

    if not frames:
        raise SystemExit("No readable CSV files found.")

    combined = pd.concat(frames, ignore_index=True)
    if args.sort_by not in combined.columns:
        raise SystemExit(
            f"Column '{args.sort_by}' not found. "
            "Use --sort-by avg_w1 for old result files, or rerun jobs with the composite scorer."
        )
    combined[args.sort_by] = pd.to_numeric(combined[args.sort_by], errors="coerce")
    combined = combined.sort_values(args.sort_by, ascending=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    cols = [
        "rf", "rt", "n_runs", "core_fraction", "core_protection",
        "composite_score", "avg_w1", "sum_w1",
        "avg_singleton_abs_error", "avg_short_cdf_abs_error",
        "avg_long_tail_abs_error", "n_pairs", "seed",
    ]
    visible_cols = [col for col in cols if col in combined.columns]
    print(f"wrote {len(combined)} rows to {args.out_csv}")
    print()
    print(combined[visible_cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
