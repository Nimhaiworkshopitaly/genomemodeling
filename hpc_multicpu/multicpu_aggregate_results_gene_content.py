#!/usr/bin/env python3
"""Combine gene-content fitting CSVs and rank gain/loss settings."""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate gene-content result CSVs.")
    parser.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    parser.add_argument("--out-csv", default="hpc_multicpu/combined_gene_content.csv")
    parser.add_argument(
        "--sort-by",
        default="gene_content_score",
        choices=[
            "gene_content_score",
            "avg_jaccard_distance_abs_error",
            "avg_min_overlap_distance_abs_error",
            "avg_genome_size_relative_error",
        ],
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
    combined[args.sort_by] = pd.to_numeric(combined[args.sort_by], errors="coerce")
    combined = combined.sort_values(args.sort_by, ascending=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    cols = [
        "rf", "gain_rate", "loss_rate", "n_runs", "workers",
        "gene_content_score",
        "avg_jaccard_distance_abs_error",
        "avg_min_overlap_distance_abs_error",
        "avg_jaccard_similarity_abs_error",
        "avg_min_overlap_similarity_abs_error",
        "avg_genome_size_relative_error",
        "n_pairs", "n_genomes", "seed",
    ]
    visible_cols = [col for col in cols if col in combined.columns]
    print(f"wrote {len(combined)} rows to {args.out_csv}")
    print()
    print(combined[visible_cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
