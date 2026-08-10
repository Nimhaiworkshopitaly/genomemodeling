#!/usr/bin/env python3
"""Evaluate one gain/loss setting using gene-content distances.

This intentionally ignores gene order: simulations use gain/loss only
with translocation and inversion rates fixed to zero. The output is meant
to constrain gene flux before fitting synteny/rearrangement parameters.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from itertools import combinations
from multiprocessing import Pool

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prod_1b_core_composite import load_real_genomes_from_cc, make_root_genome  # noqa: E402
from Bio import Phylo  # noqa: E402
from simulation_depth_gain_loss_composite import evolve_genome  # noqa: E402


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit gene flux using pairwise gene-content distances."
    )
    parser.add_argument("--atgc-dir", required=True)
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf", type=float, default=None, help="Tied gain/loss rate.")
    parser.add_argument("--gain-rate", type=float, default=None)
    parser.add_argument("--loss-rate", type=float, default=None)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--huge-exp", type=float, default=1e9)
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
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def clean_gene_set(genome) -> set:
    return {gene for gene in genome if gene != -1}


def jaccard_similarity(a: set, b: set) -> float:
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def min_overlap_similarity(a: set, b: set) -> float:
    denom = min(len(a), len(b))
    if denom == 0:
        return 1.0 if len(a) == 0 and len(b) == 0 else 0.0
    return len(a & b) / denom


def neg_log_similarity(similarity: float) -> float:
    return -math.log(max(float(similarity), EPS))


def pairwise_gene_content_metrics(genomes: dict[str, list], leaf_names: list[str]) -> dict:
    metrics = {}
    gene_sets = {name: clean_gene_set(genomes[name]) for name in leaf_names}
    for a, b in combinations(leaf_names, 2):
        s1 = gene_sets[a]
        s2 = gene_sets[b]
        jac = jaccard_similarity(s1, s2)
        minov = min_overlap_similarity(s1, s2)
        pair = (a, b) if a <= b else (b, a)
        metrics[pair] = {
            "jaccard_similarity": jac,
            "jaccard_distance": neg_log_similarity(jac),
            "min_overlap_similarity": minov,
            "min_overlap_distance": neg_log_similarity(minov),
        }
    return metrics


def genome_sizes(genomes: dict[str, list], leaf_names: list[str]) -> dict[str, int]:
    return {name: len(clean_gene_set(genomes[name])) for name in leaf_names}


def split_counts(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base = total // chunks
    rem = total % chunks
    return [base + (1 if i < rem else 0) for i in range(chunks)]


def worker_simulate(payload):
    (
        tree,
        root_genome,
        leaf_names,
        gain_rate,
        loss_rate,
        huge_exp,
        n_runs,
        seed,
        core_fraction,
        core_protection,
        terminal_rate_multiplier,
        internal_rate_multiplier,
        use_depth_dependent_gain_loss,
        depth_gain_loss_scale,
        depth_gain_loss_decay,
    ) = payload
    rng = np.random.default_rng(seed) if seed is not None else None

    pair_sums = {}
    pair_counts = {}
    size_sums = {name: 0.0 for name in leaf_names}
    size_counts = {name: 0 for name in leaf_names}

    for _ in range(n_runs):
        child_seed = int(rng.integers(0, 2**32 - 1)) if rng is not None else None
        if child_seed is not None:
            random.seed(child_seed)
            np.random.seed(child_seed)

        next_gene_id_holder = [len(root_genome) + 1]
        all_genomes = evolve_genome(
            tree,
            root_genome,
            per_gene_gain_rate=gain_rate,
            per_gene_loss_rate=loss_rate,
            per_gene_inv_rate=0.0,
            per_gene_trans_rate=0.0,
            gain_exp=huge_exp,
            loss_exp=huge_exp,
            inv_exp=huge_exp,
            trans_exp=huge_exp,
            next_gene_id_holder=next_gene_id_holder,
            core_fraction=core_fraction,
            core_protection=core_protection,
            terminal_rate_multiplier=terminal_rate_multiplier,
            internal_rate_multiplier=internal_rate_multiplier,
            use_depth_dependent_gain_loss=use_depth_dependent_gain_loss,
            depth_gain_loss_scale=depth_gain_loss_scale,
            depth_gain_loss_decay=depth_gain_loss_decay,
        )
        terminal_genomes = {
            leaf.name: all_genomes[leaf]
            for leaf in tree.get_terminals()
            if leaf.name in leaf_names
        }
        sim_pair_metrics = pairwise_gene_content_metrics(terminal_genomes, leaf_names)
        for pair, vals in sim_pair_metrics.items():
            if pair not in pair_sums:
                pair_sums[pair] = {key: 0.0 for key in vals}
                pair_counts[pair] = 0
            for key, value in vals.items():
                pair_sums[pair][key] += float(value)
            pair_counts[pair] += 1

        sim_sizes = genome_sizes(terminal_genomes, leaf_names)
        for name, size in sim_sizes.items():
            size_sums[name] += float(size)
            size_counts[name] += 1

    return pair_sums, pair_counts, size_sums, size_counts


def merge_partials(partials):
    pair_sums = {}
    pair_counts = {}
    size_sums = {}
    size_counts = {}
    for p_sums, p_counts, s_sums, s_counts in partials:
        for pair, vals in p_sums.items():
            if pair not in pair_sums:
                pair_sums[pair] = {key: 0.0 for key in vals}
                pair_counts[pair] = 0
            for key, value in vals.items():
                pair_sums[pair][key] += value
            pair_counts[pair] += p_counts[pair]
        for name, value in s_sums.items():
            size_sums[name] = size_sums.get(name, 0.0) + value
            size_counts[name] = size_counts.get(name, 0) + s_counts[name]
    return pair_sums, pair_counts, size_sums, size_counts


def mean_sim_pair_metrics(pair_sums, pair_counts):
    return {
        pair: {key: value / pair_counts[pair] for key, value in vals.items()}
        for pair, vals in pair_sums.items()
        if pair_counts[pair] > 0
    }


def mean_sim_sizes(size_sums, size_counts):
    return {
        name: value / size_counts[name]
        for name, value in size_sums.items()
        if size_counts[name] > 0
    }


def summarize_errors(real_pairs, sim_pairs, real_sizes, sim_sizes):
    pair_errors = []
    for pair, real_vals in real_pairs.items():
        sim_vals = sim_pairs.get(pair)
        if sim_vals is None:
            continue
        pair_errors.append({
            "jaccard_distance_abs_error": abs(
                real_vals["jaccard_distance"] - sim_vals["jaccard_distance"]
            ),
            "min_overlap_distance_abs_error": abs(
                real_vals["min_overlap_distance"] - sim_vals["min_overlap_distance"]
            ),
            "jaccard_similarity_abs_error": abs(
                real_vals["jaccard_similarity"] - sim_vals["jaccard_similarity"]
            ),
            "min_overlap_similarity_abs_error": abs(
                real_vals["min_overlap_similarity"] - sim_vals["min_overlap_similarity"]
            ),
        })

    size_errors = []
    for name, real_size in real_sizes.items():
        sim_size = sim_sizes.get(name)
        if sim_size is None or real_size <= 0:
            continue
        size_errors.append(abs(real_size - sim_size) / real_size)

    def avg(key):
        return float(np.mean([row[key] for row in pair_errors])) if pair_errors else float("nan")

    avg_jaccard_distance_abs_error = avg("jaccard_distance_abs_error")
    avg_min_overlap_distance_abs_error = avg("min_overlap_distance_abs_error")
    avg_jaccard_similarity_abs_error = avg("jaccard_similarity_abs_error")
    avg_min_overlap_similarity_abs_error = avg("min_overlap_similarity_abs_error")
    avg_genome_size_relative_error = float(np.mean(size_errors)) if size_errors else float("nan")

    gene_content_score = (
        avg_jaccard_distance_abs_error
        + avg_min_overlap_distance_abs_error
        + avg_genome_size_relative_error
    )

    return {
        "gene_content_score": gene_content_score,
        "avg_jaccard_distance_abs_error": avg_jaccard_distance_abs_error,
        "avg_min_overlap_distance_abs_error": avg_min_overlap_distance_abs_error,
        "avg_jaccard_similarity_abs_error": avg_jaccard_similarity_abs_error,
        "avg_min_overlap_similarity_abs_error": avg_min_overlap_similarity_abs_error,
        "avg_genome_size_relative_error": avg_genome_size_relative_error,
        "n_pairs": len(pair_errors),
        "n_genomes": len(size_errors),
    }


def main() -> None:
    args = parse_args()
    if args.rf is None and (args.gain_rate is None or args.loss_rate is None):
        raise SystemExit("Provide --rf or both --gain-rate and --loss-rate.")
    gain_rate = args.rf if args.rf is not None else args.gain_rate
    loss_rate = args.rf if args.rf is not None else args.loss_rate

    atgc_dir = os.path.abspath(args.atgc_dir)
    tree_path = os.path.join(atgc_dir, args.tree_filename)
    cc_path = os.path.join(atgc_dir, args.cc_filename)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    tree = Phylo.read(tree_path, "newick")
    real_genomes_raw = load_real_genomes_from_cc(cc_path)
    leaf_names = [leaf.name for leaf in tree.get_terminals() if leaf.name in real_genomes_raw]
    if len(leaf_names) < 2:
        raise SystemExit("Fewer than two tree leaves were found in the real genomes.")

    root_genome = make_root_genome(args.root_mode, tree, cc_path, real_genomes=real_genomes_raw)
    real_pairs = pairwise_gene_content_metrics(real_genomes_raw, leaf_names)
    real_sizes = genome_sizes(real_genomes_raw, leaf_names)

    run_counts = split_counts(args.n_runs, args.workers)
    seed_rng = np.random.default_rng(args.seed) if args.seed is not None else None
    payloads = []
    for n in run_counts:
        worker_seed = int(seed_rng.integers(0, 2**32 - 1)) if seed_rng is not None else None
        payloads.append((
            tree,
            root_genome,
            leaf_names,
            float(gain_rate),
            float(loss_rate),
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

    pair_sums, pair_counts, size_sums, size_counts = merge_partials(partials)
    sim_pairs = mean_sim_pair_metrics(pair_sums, pair_counts)
    sim_sizes = mean_sim_sizes(size_sums, size_counts)
    scores = summarize_errors(real_pairs, sim_pairs, real_sizes, sim_sizes)
    dataset = os.path.basename(os.path.normpath(atgc_dir))

    fieldnames = [
        "dataset", "root_mode", "rf", "gain_rate", "loss_rate",
        "n_runs", "workers", "core_fraction", "core_protection",
        "terminal_rate_multiplier", "internal_rate_multiplier",
        "use_depth_dependent_gain_loss", "depth_gain_loss_scale",
        "depth_gain_loss_decay", "gene_content_score",
        "avg_jaccard_distance_abs_error",
        "avg_min_overlap_distance_abs_error",
        "avg_jaccard_similarity_abs_error",
        "avg_min_overlap_similarity_abs_error",
        "avg_genome_size_relative_error",
        "n_pairs", "n_genomes", "seed", "tree_path", "cc_path",
    ]
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "dataset": dataset,
            "root_mode": args.root_mode,
            "rf": "" if args.rf is None else f"{args.rf:.10g}",
            "gain_rate": f"{float(gain_rate):.10g}",
            "loss_rate": f"{float(loss_rate):.10g}",
            "n_runs": args.n_runs,
            "workers": len(payloads),
            "core_fraction": f"{args.core_fraction:.10g}",
            "core_protection": f"{args.core_protection:.10g}",
            "terminal_rate_multiplier": f"{args.terminal_rate_multiplier:.10g}",
            "internal_rate_multiplier": f"{args.internal_rate_multiplier:.10g}",
            "use_depth_dependent_gain_loss": int(args.use_depth_dependent_gain_loss),
            "depth_gain_loss_scale": f"{args.depth_gain_loss_scale:.10g}",
            "depth_gain_loss_decay": f"{args.depth_gain_loss_decay:.10g}",
            "gene_content_score": f"{scores['gene_content_score']:.12g}",
            "avg_jaccard_distance_abs_error": f"{scores['avg_jaccard_distance_abs_error']:.12g}",
            "avg_min_overlap_distance_abs_error": f"{scores['avg_min_overlap_distance_abs_error']:.12g}",
            "avg_jaccard_similarity_abs_error": f"{scores['avg_jaccard_similarity_abs_error']:.12g}",
            "avg_min_overlap_similarity_abs_error": f"{scores['avg_min_overlap_similarity_abs_error']:.12g}",
            "avg_genome_size_relative_error": f"{scores['avg_genome_size_relative_error']:.12g}",
            "n_pairs": scores["n_pairs"],
            "n_genomes": scores["n_genomes"],
            "seed": "" if args.seed is None else args.seed,
            "tree_path": tree_path,
            "cc_path": cc_path,
        })

    print(
        f"wrote {args.out_csv}: gain={float(gain_rate):.4g}, "
        f"loss={float(loss_rate):.4g}, score={scores['gene_content_score']:.4g}, "
        f"jaccard_d_err={scores['avg_jaccard_distance_abs_error']:.4g}, "
        f"min_overlap_d_err={scores['avg_min_overlap_distance_abs_error']:.4g}, "
        f"genome_size_rel_err={scores['avg_genome_size_relative_error']:.4g}"
    )


if __name__ == "__main__":
    main()
