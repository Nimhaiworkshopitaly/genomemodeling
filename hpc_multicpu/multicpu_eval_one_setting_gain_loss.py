#!/usr/bin/env python3
"""Evaluate one gain/loss/translocation setting with multiple CPUs.

This variant separates the original shared rf into two biological parameters:

    --gain-rate  per-gene gain rate
    --loss-rate  per-gene loss rate

The original rf/rt workflow effectively used gain_rate == loss_rate == rf.
"""

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

from prod_1b import build_real_pmfs, make_root_genome, score_real_vs_sim_counts  # noqa: E402
from simulation import run_simulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one gain/loss/rt setting with multiple worker processes."
    )
    parser.add_argument("--atgc-dir", required=True)
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--gain-rate", type=float, required=True)
    parser.add_argument("--loss-rate", type=float, required=True)
    parser.add_argument("--rt", type=float, required=True)
    parser.add_argument("--inv-rate", type=float, default=0.0)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--huge-exp", type=float, default=1e9)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def split_counts(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base = total // chunks
    rem = total % chunks
    return [base + (1 if i < rem else 0) for i in range(chunks)]


def worker_simulate(payload):
    (
        tree,
        root_genome,
        gain_rate,
        loss_rate,
        rt,
        inv_rate,
        huge_exp,
        n_runs,
        seed,
    ) = payload
    rng = np.random.default_rng(seed) if seed is not None else None
    counts = defaultdict(Counter)

    for _ in range(n_runs):
        child_seed = int(rng.integers(0, 2**32 - 1)) if rng is not None else None
        if child_seed is not None:
            random.seed(child_seed)
            np.random.seed(child_seed)

        sim_pairs = run_simulation(
            tree,
            root_genome,
            per_gene_gain_rate=gain_rate,
            per_gene_loss_rate=loss_rate,
            per_gene_inv_rate=inv_rate,
            per_gene_trans_rate=rt,
            gain_exp=huge_exp,
            loss_exp=huge_exp,
            inv_exp=huge_exp,
            trans_exp=huge_exp,
        )
        for (a, b), lens in sim_pairs.items():
            pair = (a, b) if a <= b else (b, a)
            if lens:
                counts[pair].update(int(x) for x in lens)

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

    real_pmfs, tree, real_genomes, med_path = build_real_pmfs(tree_path, cc_path)
    root_genome = make_root_genome(args.root_mode, tree, cc_path, real_genomes=real_genomes)

    run_counts = split_counts(args.n_runs, args.workers)
    seed_rng = np.random.default_rng(args.seed) if args.seed is not None else None
    payloads = []
    for n in run_counts:
        worker_seed = int(seed_rng.integers(0, 2**32 - 1)) if seed_rng is not None else None
        payloads.append((
            tree,
            root_genome,
            args.gain_rate,
            args.loss_rate,
            args.rt,
            args.inv_rate,
            args.huge_exp,
            n,
            worker_seed,
        ))

    with Pool(processes=len(payloads)) as pool:
        partials = pool.map(worker_simulate, payloads)

    scores = score_real_vs_sim_counts(real_pmfs, merge_counts(partials))
    dataset = os.path.basename(os.path.normpath(atgc_dir))

    fieldnames = [
        "dataset",
        "root_mode",
        "gain_rate",
        "loss_rate",
        "rt",
        "inv_rate",
        "n_runs",
        "workers",
        "sum_w1",
        "avg_w1",
        "n_pairs",
        "skipped_real",
        "skipped_sim",
        "composite_score",
        "avg_singleton_abs_error",
        "avg_short_cdf_abs_error",
        "avg_long_tail_abs_error",
        "short_cdf_length",
        "long_tail_length",
        "composite_w_w1",
        "composite_w_singleton",
        "composite_w_short_cdf",
        "composite_w_long_tail",
        "median_root_to_leaf",
        "seed",
        "tree_path",
        "cc_path",
    ]
    row = {
        "dataset": dataset,
        "root_mode": args.root_mode,
        "gain_rate": f"{args.gain_rate:.10g}",
        "loss_rate": f"{args.loss_rate:.10g}",
        "rt": f"{args.rt:.10g}",
        "inv_rate": f"{args.inv_rate:.10g}",
        "n_runs": args.n_runs,
        "workers": len(payloads),
        "sum_w1": f"{scores['sum_w1']:.12g}",
        "avg_w1": f"{scores['avg_w1']:.12g}",
        "n_pairs": scores["n_pairs"],
        "skipped_real": scores["skipped_real"],
        "skipped_sim": scores["skipped_sim"],
        "median_root_to_leaf": f"{med_path:.8g}",
        "seed": "" if args.seed is None else args.seed,
        "tree_path": tree_path,
        "cc_path": cc_path,
    }
    for key in [
        "composite_score",
        "avg_singleton_abs_error",
        "avg_short_cdf_abs_error",
        "avg_long_tail_abs_error",
        "short_cdf_length",
        "long_tail_length",
        "composite_w_w1",
        "composite_w_singleton",
        "composite_w_short_cdf",
        "composite_w_long_tail",
    ]:
        row[key] = scores.get(key, "")
        if isinstance(row[key], float):
            row[key] = f"{row[key]:.12g}"

    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(
        f"wrote {args.out_csv}: gain={args.gain_rate:.4g}, "
        f"loss={args.loss_rate:.4g}, rt={args.rt:.4g}, "
        f"avg_w1={scores['avg_w1']:.4g}"
    )


if __name__ == "__main__":
    main()
