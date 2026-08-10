#!/usr/bin/env python3
"""Evaluate one rf/rt setting with depth-dependent gain/loss rates.

This model keeps the core-composite and terminal/internal branch-rate options,
and optionally applies Y.'s gain/loss depth multiplier on each branch.
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

from prod_1b_core_composite import build_real_pmfs, make_root_genome, score_real_vs_sim_counts  # noqa: E402
from simulation_depth_gain_loss_composite import run_simulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one rf/rt setting with multiple worker processes."
    )
    parser.add_argument("--atgc-dir", required=True)
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf", type=float, required=True)
    parser.add_argument("--rt", type=float, required=True)
    parser.add_argument("--inv-rate", type=float, default=0.0)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--huge-exp", type=float, default=1e9)
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
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def split_counts(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base = total // chunks
    rem = total % chunks
    return [base + (1 if i < rem else 0) for i in range(chunks)]


def worker_simulate(payload):
    (
        tree, root_genome, rf, rt, inv_rate, huge_exp, n_runs, seed,
        core_fraction, core_protection,
        terminal_rate_multiplier, internal_rate_multiplier,
        use_depth_dependent_gain_loss,
        depth_gain_loss_scale, depth_gain_loss_decay,
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
            per_gene_gain_rate=rf,
            per_gene_loss_rate=rf,
            per_gene_inv_rate=inv_rate,
            per_gene_trans_rate=rt,
            gain_exp=huge_exp,
            loss_exp=huge_exp,
            inv_exp=huge_exp,
            trans_exp=huge_exp,
            core_fraction=core_fraction,
            core_protection=core_protection,
            terminal_rate_multiplier=terminal_rate_multiplier,
            internal_rate_multiplier=internal_rate_multiplier,
            use_depth_dependent_gain_loss=use_depth_dependent_gain_loss,
            depth_gain_loss_scale=depth_gain_loss_scale,
            depth_gain_loss_decay=depth_gain_loss_decay,
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
            args.rf,
            args.rt,
            args.inv_rate,
            args.huge_exp,
            n,
            worker_seed,
            args.core_fraction,
            args.core_protection,
            args.terminal_rate_multiplier,
            args.internal_rate_multiplier,
            args.use_depth_dependent_gain_loss,
            args.depth_gain_loss_scale,
            args.depth_gain_loss_decay,
        ))

    with Pool(processes=len(payloads)) as pool:
        partials = pool.map(worker_simulate, payloads)

    scores = score_real_vs_sim_counts(real_pmfs, merge_counts(partials))
    dataset = os.path.basename(os.path.normpath(atgc_dir))

    fieldnames = [
        "dataset", "root_mode", "rf", "rt", "inv_rate", "n_runs", "workers",
        "core_fraction", "core_protection",
        "terminal_rate_multiplier", "internal_rate_multiplier",
        "use_depth_dependent_gain_loss",
        "depth_gain_loss_scale", "depth_gain_loss_decay",
        "sum_w1", "avg_w1", "n_pairs", "skipped_real", "skipped_sim",
        "composite_score", "avg_singleton_abs_error",
        "avg_short_cdf_abs_error", "avg_long_tail_abs_error",
        "avg_kl_real_to_sim", "avg_kl_sim_to_real",
        "avg_js_divergence", "avg_bhattacharyya_coefficient",
        "avg_bhattacharyya_distance", "avg_hellinger_distance",
        "distribution_smoothing_epsilon",
        "short_cdf_length", "long_tail_length",
        "composite_w_w1", "composite_w_singleton",
        "composite_w_short_cdf", "composite_w_long_tail",
        "median_root_to_leaf", "seed", "tree_path", "cc_path",
    ]
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "dataset": dataset,
            "root_mode": args.root_mode,
            "rf": f"{args.rf:.10g}",
            "rt": f"{args.rt:.10g}",
            "inv_rate": f"{args.inv_rate:.10g}",
            "n_runs": args.n_runs,
            "workers": len(payloads),
            "core_fraction": f"{args.core_fraction:.10g}",
            "core_protection": f"{args.core_protection:.10g}",
            "terminal_rate_multiplier": f"{args.terminal_rate_multiplier:.10g}",
            "internal_rate_multiplier": f"{args.internal_rate_multiplier:.10g}",
            "use_depth_dependent_gain_loss": int(args.use_depth_dependent_gain_loss),
            "depth_gain_loss_scale": f"{args.depth_gain_loss_scale:.10g}",
            "depth_gain_loss_decay": f"{args.depth_gain_loss_decay:.10g}",
            "sum_w1": f"{scores['sum_w1']:.12g}",
            "avg_w1": f"{scores['avg_w1']:.12g}",
            "n_pairs": scores["n_pairs"],
            "skipped_real": scores["skipped_real"],
            "skipped_sim": scores["skipped_sim"],
            "composite_score": f"{scores['composite_score']:.12g}",
            "avg_singleton_abs_error": f"{scores['avg_singleton_abs_error']:.12g}",
            "avg_short_cdf_abs_error": f"{scores['avg_short_cdf_abs_error']:.12g}",
            "avg_long_tail_abs_error": f"{scores['avg_long_tail_abs_error']:.12g}",
            "avg_kl_real_to_sim": f"{scores['avg_kl_real_to_sim']:.12g}",
            "avg_kl_sim_to_real": f"{scores['avg_kl_sim_to_real']:.12g}",
            "avg_js_divergence": f"{scores['avg_js_divergence']:.12g}",
            "avg_bhattacharyya_coefficient": f"{scores['avg_bhattacharyya_coefficient']:.12g}",
            "avg_bhattacharyya_distance": f"{scores['avg_bhattacharyya_distance']:.12g}",
            "avg_hellinger_distance": f"{scores['avg_hellinger_distance']:.12g}",
            "distribution_smoothing_epsilon": f"{scores['distribution_smoothing_epsilon']:.3g}",
            "short_cdf_length": scores["short_cdf_length"],
            "long_tail_length": scores["long_tail_length"],
            "composite_w_w1": f"{scores['composite_w_w1']:.8g}",
            "composite_w_singleton": f"{scores['composite_w_singleton']:.8g}",
            "composite_w_short_cdf": f"{scores['composite_w_short_cdf']:.8g}",
            "composite_w_long_tail": f"{scores['composite_w_long_tail']:.8g}",
            "median_root_to_leaf": f"{med_path:.8g}",
            "seed": "" if args.seed is None else args.seed,
            "tree_path": tree_path,
            "cc_path": cc_path,
        })

    print(
        f"wrote {args.out_csv}: rf={args.rf:.4g}, rt={args.rt:.4g}, "
        f"n_runs={args.n_runs}, workers={len(payloads)}, "
        f"depth_gain_loss={int(args.use_depth_dependent_gain_loss)}, "
        f"sum_w1={scores['sum_w1']:.4g}, avg_w1={scores['avg_w1']:.4g}, "
        f"composite={scores['composite_score']:.4g}"
    )


if __name__ == "__main__":
    main()
