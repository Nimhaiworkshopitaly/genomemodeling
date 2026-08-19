#!/usr/bin/env python3
"""Evaluate one rearrangement-size setting using multiple CPUs."""

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
        description="Test rearrangement block sizes and inversion rates."
    )
    parser.add_argument("--atgc-dir", required=True)
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf", type=float, required=True)
    parser.add_argument("--rt", type=float, required=True)
    parser.add_argument(
        "--rearrangement-exp",
        type=float,
        required=True,
        help="Power-law exponent shared by translocations and inversions.",
    )
    parser.add_argument(
        "--inversion-multiplier",
        type=float,
        default=0.0,
        help="Inversion rate as a multiple of rt (inv_rate = rt * multiplier).",
    )
    parser.add_argument("--gain-loss-exp", type=float, default=1e9)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--core-fraction", type=float, default=0.5)
    parser.add_argument("--core-protection", type=float, default=0.9)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()
    if args.rf < 0 or args.rt < 0 or args.inversion_multiplier < 0:
        parser.error("rates and inversion multiplier must be non-negative")
    if args.rearrangement_exp <= 0 or args.gain_loss_exp <= 0:
        parser.error("power-law exponents must be positive")
    return args


def split_counts(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base, remainder = divmod(total, chunks)
    return [base + (index < remainder) for index in range(chunks)]


def worker_simulate(payload):
    (
        tree, root_genome, rf, rt, rearrangement_exp, inversion_multiplier,
        gain_loss_exp, n_runs, seed, core_fraction, core_protection,
    ) = payload
    rng = np.random.default_rng(seed) if seed is not None else None
    counts = defaultdict(Counter)
    inversion_rate = rt * inversion_multiplier

    for _ in range(n_runs):
        child_seed = int(rng.integers(0, 2**32 - 1)) if rng is not None else None
        if child_seed is not None:
            random.seed(child_seed)
            np.random.seed(child_seed)

        sim_pairs = run_simulation(
            tree,
            root_genome,
            per_gene_gain_rate=rf,
            per_gene_loss_rate=rf,
            per_gene_inv_rate=inversion_rate,
            per_gene_trans_rate=rt,
            gain_exp=gain_loss_exp,
            loss_exp=gain_loss_exp,
            inv_exp=rearrangement_exp,
            trans_exp=rearrangement_exp,
            core_fraction=core_fraction,
            core_protection=core_protection,
        )
        for (genome_a, genome_b), lengths in sim_pairs.items():
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

    real_pmfs, tree, real_genomes, median_path = build_real_pmfs(tree_path, cc_path)
    root_genome = make_root_genome(
        args.root_mode, tree, cc_path, real_genomes=real_genomes
    )

    run_counts = split_counts(args.n_runs, args.workers)
    seed_rng = np.random.default_rng(args.seed) if args.seed is not None else None
    payloads = []
    for count in run_counts:
        worker_seed = (
            int(seed_rng.integers(0, 2**32 - 1))
            if seed_rng is not None else None
        )
        payloads.append((
            tree, root_genome, args.rf, args.rt, args.rearrangement_exp,
            args.inversion_multiplier, args.gain_loss_exp, count, worker_seed,
            args.core_fraction, args.core_protection,
        ))

    with Pool(processes=len(payloads)) as pool:
        scores = score_real_vs_sim_counts(
            real_pmfs, merge_counts(pool.map(worker_simulate, payloads))
        )

    inversion_rate = args.rt * args.inversion_multiplier
    row = {
        "dataset": os.path.basename(os.path.normpath(atgc_dir)),
        "root_mode": args.root_mode,
        "rf": args.rf,
        "rt": args.rt,
        "inversion_multiplier": args.inversion_multiplier,
        "inv_rate": inversion_rate,
        "rearrangement_exp": args.rearrangement_exp,
        "gain_loss_exp": args.gain_loss_exp,
        "n_runs": args.n_runs,
        "workers": len(payloads),
        "core_fraction": args.core_fraction,
        "core_protection": args.core_protection,
        "sum_w1": scores["sum_w1"],
        "avg_w1": scores["avg_w1"],
        "composite_score": scores["composite_score"],
        "avg_singleton_abs_error": scores["avg_singleton_abs_error"],
        "avg_short_cdf_abs_error": scores["avg_short_cdf_abs_error"],
        "avg_long_tail_abs_error": scores["avg_long_tail_abs_error"],
        "avg_ks_statistic": scores["avg_ks_statistic"],
        "avg_kuiper_statistic": scores["avg_kuiper_statistic"],
        "n_pairs": scores["n_pairs"],
        "skipped_real": scores["skipped_real"],
        "skipped_sim": scores["skipped_sim"],
        "median_root_to_leaf": median_path,
        "seed": "" if args.seed is None else args.seed,
        "tree_path": tree_path,
        "cc_path": cc_path,
    }
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(
        f"wrote {args.out_csv}: rearrangement_exp={args.rearrangement_exp:g}, "
        f"inv_rate={inversion_rate:g}, avg_w1={scores['avg_w1']:.4g}, "
        f"KS={scores['avg_ks_statistic']:.4g}, "
        f"Kuiper={scores['avg_kuiper_statistic']:.4g}"
    )


if __name__ == "__main__":
    main()
