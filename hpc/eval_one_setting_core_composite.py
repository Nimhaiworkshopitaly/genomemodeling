#!/usr/bin/env python3
"""Evaluate one genome-evolution parameter setting.

This script is intended for Biowulf swarm/job-array runs. Each invocation writes
one CSV file, so parallel jobs do not contend for a shared output file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prod_1b_core_composite import simulate_and_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one rf/rt parameter setting and write one CSV row."
    )
    parser.add_argument("--atgc-dir", required=True, help="Dataset directory, e.g. ATGC0070")
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf", type=float, required=True, help="Gain/loss rate")
    parser.add_argument("--rt", type=float, required=True, help="Translocation rate")
    parser.add_argument("--inv-rate", type=float, default=0.0)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--huge-exp", type=float, default=1e9)
    parser.add_argument("--core-fraction", type=float, default=0.5)
    parser.add_argument("--core-protection", type=float, default=0.9)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atgc_dir = os.path.abspath(args.atgc_dir)
    tree_path = os.path.join(atgc_dir, args.tree_filename)
    cc_path = os.path.join(atgc_dir, args.cc_filename)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    res = simulate_and_score(
        atgc_dir=atgc_dir,
        tree_path=tree_path,
        cc_path=cc_path,
        root_mode=args.root_mode,
        per_gene_gain=args.rf,
        per_gene_loss=args.rf,
        per_gene_inv=args.inv_rate,
        per_gene_trans=args.rt,
        exp_gain=args.huge_exp,
        exp_loss=args.huge_exp,
        exp_inv=args.huge_exp,
        exp_trans=args.huge_exp,
        n_runs=args.n_runs,
        seed=args.seed,
        core_fraction=args.core_fraction,
        core_protection=args.core_protection,
    )

    fieldnames = [
        "dataset", "root_mode", "rf", "rt", "inv_rate", "n_runs",
        "core_fraction", "core_protection",
        "sum_w1", "avg_w1", "n_pairs", "skipped_real", "skipped_sim",
        "composite_score", "avg_singleton_abs_error",
        "avg_short_cdf_abs_error", "avg_long_tail_abs_error",
        "short_cdf_length", "long_tail_length",
        "composite_w_w1", "composite_w_singleton",
        "composite_w_short_cdf", "composite_w_long_tail",
        "median_root_to_leaf", "seed", "tree_path", "cc_path", "timestamp",
    ]
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "dataset": res["dataset"],
            "root_mode": args.root_mode,
            "rf": f"{args.rf:.10g}",
            "rt": f"{args.rt:.10g}",
            "inv_rate": f"{args.inv_rate:.10g}",
            "n_runs": args.n_runs,
            "core_fraction": f"{args.core_fraction:.10g}",
            "core_protection": f"{args.core_protection:.10g}",
            "sum_w1": f"{res['sum_w1']:.12g}",
            "avg_w1": f"{res['avg_w1']:.12g}",
            "n_pairs": res["n_pairs"],
            "skipped_real": res["skipped_real"],
            "skipped_sim": res["skipped_sim"],
            "composite_score": f"{res['composite_score']:.12g}",
            "avg_singleton_abs_error": f"{res['avg_singleton_abs_error']:.12g}",
            "avg_short_cdf_abs_error": f"{res['avg_short_cdf_abs_error']:.12g}",
            "avg_long_tail_abs_error": f"{res['avg_long_tail_abs_error']:.12g}",
            "short_cdf_length": res["short_cdf_length"],
            "long_tail_length": res["long_tail_length"],
            "composite_w_w1": f"{res['composite_w_w1']:.8g}",
            "composite_w_singleton": f"{res['composite_w_singleton']:.8g}",
            "composite_w_short_cdf": f"{res['composite_w_short_cdf']:.8g}",
            "composite_w_long_tail": f"{res['composite_w_long_tail']:.8g}",
            "median_root_to_leaf": f"{res['median_root_to_leaf']:.8g}",
            "seed": "" if args.seed is None else args.seed,
            "tree_path": tree_path,
            "cc_path": cc_path,
            "timestamp": res["timestamp"],
        })

    print(
        f"wrote {args.out_csv}: rf={args.rf:.4g}, rt={args.rt:.4g}, "
        f"sum_w1={res['sum_w1']:.4g}, avg_w1={res['avg_w1']:.4g}, "
        f"composite={res['composite_score']:.4g}"
    )


if __name__ == "__main__":
    main()
