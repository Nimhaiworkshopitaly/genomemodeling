#!/usr/bin/env python3
"""Evaluate one fixed-total-rate translocation/inversion mixture."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prod_1b_core_composite import (  # noqa: E402
    build_real_pmfs,
    make_root_genome,
    score_real_vs_sim_counts,
)
from simulation_core_composite import run_simulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition a fixed rearrangement rate between translocations and inversions."
    )
    parser.add_argument("--atgc-dir", required=True)
    parser.add_argument("--tree-filename", default="yuri_gl26/ATGC0070.gl.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf", type=float, default=0.1)
    parser.add_argument("--total-rearrangement-rate", type=float, default=0.316227766017)
    parser.add_argument("--inversion-fraction", type=float, required=True)
    parser.add_argument("--translocation-exp", type=float, default=1e9)
    parser.add_argument("--inversion-exp", type=float, default=3.0)
    parser.add_argument(
        "--inversion-size-mode",
        choices=("powerlaw", "uniform_breakpoints"),
        default="uniform_breakpoints",
        help=(
            "Use two uniform ordered breakpoints (resampling invisible "
            "length-one inversions) or the legacy power-law size sampler."
        ),
    )
    parser.add_argument("--gain-loss-exp", type=float, default=1e9)
    parser.add_argument("--core-fraction", type=float, default=0.5)
    parser.add_argument("--core-protection", type=float, default=0.9)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--pair-out-csv",
        help="Pair-level metric CSV (default: <out-csv stem>_pairs.csv).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.inversion_fraction <= 1.0:
        parser.error("--inversion-fraction must be between 0 and 1")
    if args.rf < 0 or args.total_rearrangement_rate < 0:
        parser.error("rates must be non-negative")
    if min(args.translocation_exp, args.inversion_exp, args.gain_loss_exp) <= 0:
        parser.error("event-size exponents must be positive")
    if args.n_runs < 1 or args.workers < 1:
        parser.error("n-runs and workers must be positive")
    return args


def split_counts(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base, remainder = divmod(total, chunks)
    return [base + int(index < remainder) for index in range(chunks)]


def worker_simulate(payload):
    (
        tree, root_genome, rf, translocation_rate, inversion_rate,
        translocation_exp, inversion_exp, inversion_size_mode, gain_loss_exp,
        core_fraction, core_protection, n_runs, seed,
    ) = payload
    rng = np.random.default_rng(seed)
    counts = defaultdict(Counter)

    for _ in range(n_runs):
        child_seed = int(rng.integers(0, 2**32 - 1))
        random.seed(child_seed)
        np.random.seed(child_seed)
        simulated_pairs = run_simulation(
            tree,
            root_genome,
            per_gene_gain_rate=rf,
            per_gene_loss_rate=rf,
            per_gene_inv_rate=inversion_rate,
            per_gene_trans_rate=translocation_rate,
            gain_exp=gain_loss_exp,
            loss_exp=gain_loss_exp,
            inv_exp=inversion_exp,
            trans_exp=translocation_exp,
            core_fraction=core_fraction,
            core_protection=core_protection,
            inversion_size_mode=inversion_size_mode,
        )
        for (genome_a, genome_b), lengths in simulated_pairs.items():
            pair = tuple(sorted((genome_a, genome_b)))
            counts[pair].update(int(length) for length in lengths)
    return counts


def merge_counts(partials):
    merged = defaultdict(Counter)
    for partial in partials:
        for pair, counter in partial.items():
            merged[pair].update(counter)
    return merged


def main() -> None:
    args = parse_args()
    atgc_dir = os.path.abspath(args.atgc_dir)
    tree_path = os.path.join(atgc_dir, args.tree_filename)
    cc_path = os.path.join(atgc_dir, args.cc_filename)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    inversion_rate = args.total_rearrangement_rate * args.inversion_fraction
    translocation_rate = args.total_rearrangement_rate * (1.0 - args.inversion_fraction)

    real_pmfs, tree, real_genomes, median_path = build_real_pmfs(tree_path, cc_path)
    root_genome = make_root_genome(
        args.root_mode, tree, cc_path, real_genomes=real_genomes
    )

    seed_rng = np.random.default_rng(args.seed)
    payloads = []
    for count in split_counts(args.n_runs, args.workers):
        worker_seed = int(seed_rng.integers(0, 2**32 - 1))
        payloads.append((
            tree, root_genome, args.rf, translocation_rate, inversion_rate,
            args.translocation_exp, args.inversion_exp, args.inversion_size_mode,
            args.gain_loss_exp, args.core_fraction, args.core_protection, count,
            worker_seed,
        ))

    with Pool(processes=len(payloads)) as pool:
        scores = score_real_vs_sim_counts(
            real_pmfs, merge_counts(pool.map(worker_simulate, payloads))
        )

    row = {
        "dataset": os.path.basename(os.path.normpath(atgc_dir)),
        "root_mode": args.root_mode,
        "rf": args.rf,
        "total_rearrangement_rate": args.total_rearrangement_rate,
        "inversion_fraction": args.inversion_fraction,
        "translocation_rate": translocation_rate,
        "inversion_rate": inversion_rate,
        "translocation_exp": args.translocation_exp,
        "inversion_exp": args.inversion_exp,
        "inversion_size_mode": args.inversion_size_mode,
        "gain_loss_exp": args.gain_loss_exp,
        "core_fraction": args.core_fraction,
        "core_protection": args.core_protection,
        "n_runs": args.n_runs,
        "workers": len(payloads),
        "sum_w1": scores["sum_w1"],
        "avg_w1": scores["avg_w1"],
        "composite_score": scores["composite_score"],
        "avg_singleton_abs_error": scores["avg_singleton_abs_error"],
        "avg_short_cdf_abs_error": scores["avg_short_cdf_abs_error"],
        "avg_long_tail_abs_error": scores["avg_long_tail_abs_error"],
        "avg_kl_real_to_sim": scores["avg_kl_real_to_sim"],
        "avg_kl_sim_to_real": scores["avg_kl_sim_to_real"],
        "avg_js_divergence": scores["avg_js_divergence"],
        "avg_bhattacharyya_coefficient": scores["avg_bhattacharyya_coefficient"],
        "avg_bhattacharyya_distance": scores["avg_bhattacharyya_distance"],
        "avg_hellinger_distance": scores["avg_hellinger_distance"],
        "avg_ks_statistic": scores["avg_ks_statistic"],
        "avg_kuiper_statistic": scores["avg_kuiper_statistic"],
        "n_pairs": scores["n_pairs"],
        "skipped_real": scores["skipped_real"],
        "skipped_sim": scores["skipped_sim"],
        "median_root_to_leaf": median_path,
        "seed": args.seed,
        "tree_path": tree_path,
        "cc_path": cc_path,
    }
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    pair_out_csv = args.pair_out_csv or os.path.splitext(args.out_csv)[0] + "_pairs.csv"
    pair_rows = []
    for pair_metric in scores["per_pair_metrics"]:
        pair_rows.append({
            "dataset": row["dataset"],
            "inversion_fraction": args.inversion_fraction,
            "seed": args.seed,
            **pair_metric,
        })
    if not pair_rows:
        raise RuntimeError("No pair-level metrics were produced")
    with open(pair_out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    print(
        f"wrote {args.out_csv}: inversion_fraction={args.inversion_fraction:g}, "
        f"trans_rate={translocation_rate:.6g}, inv_rate={inversion_rate:.6g}, "
        f"inv_size_mode={args.inversion_size_mode}, "
        f"KS={scores['avg_ks_statistic']:.5g}, "
        f"Kuiper={scores['avg_kuiper_statistic']:.5g}; pairs={pair_out_csv}"
    )


if __name__ == "__main__":
    main()
