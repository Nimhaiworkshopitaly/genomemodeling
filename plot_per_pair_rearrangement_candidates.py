#!/usr/bin/env python3
"""Compare two rearrangement models against real SBLs for every genome pair."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from prod_1b_core_composite import build_real_pmfs, make_root_genome
from simulation_core_composite import run_simulation


MODELS = (
    {"name": "single_gene_rt_0.1", "label": "Single gene, rt=0.1", "rt": 0.1, "exp": 1e9},
    {"name": "exponent_3_rt_0.316228", "label": "Exponent 3, rt=0.316228", "rt": 0.316227766017, "exp": 3.0},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atgc-dir", required=True)
    parser.add_argument("--tree-filename", default="yuri_gl26/ATGC0070.gl.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf", type=float, default=0.1)
    parser.add_argument("--core-fraction", type=float, default=0.5)
    parser.add_argument("--core-protection", type=float, default=0.9)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", default="per_pair_rearrangement_candidates"
    )
    args = parser.parse_args()
    if args.n_runs < 1 or args.workers < 1:
        parser.error("n-runs and workers must be positive")
    return args


def split_counts(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base, remainder = divmod(total, chunks)
    return [base + int(index < remainder) for index in range(chunks)]


def simulation_worker(payload):
    (
        tree, root_genome, rf, rt, exponent, core_fraction,
        core_protection, n_runs, seed,
    ) = payload
    rng = np.random.default_rng(seed)
    counts = defaultdict(Counter)

    for _ in range(n_runs):
        child_seed = int(rng.integers(0, 2**32 - 1))
        random.seed(child_seed)
        np.random.seed(child_seed)
        pairs = run_simulation(
            tree,
            root_genome,
            per_gene_gain_rate=rf,
            per_gene_loss_rate=rf,
            per_gene_inv_rate=0.0,
            per_gene_trans_rate=rt,
            gain_exp=1e9,
            loss_exp=1e9,
            inv_exp=exponent,
            trans_exp=exponent,
            core_fraction=core_fraction,
            core_protection=core_protection,
        )
        for (genome_a, genome_b), lengths in pairs.items():
            pair = tuple(sorted((genome_a, genome_b)))
            counts[pair].update(int(length) for length in lengths)
    return counts


def merge_counts(partials):
    merged = defaultdict(Counter)
    for partial in partials:
        for pair, counter in partial.items():
            merged[pair].update(counter)
    return merged


def simulate_model(args, tree, root_genome, model):
    run_counts = split_counts(args.n_runs, args.workers)
    seed_rng = np.random.default_rng(args.seed)
    payloads = []
    for count in run_counts:
        worker_seed = int(seed_rng.integers(0, 2**32 - 1))
        payloads.append((
            tree, root_genome, args.rf, model["rt"], model["exp"],
            args.core_fraction, args.core_protection, count, worker_seed,
        ))
    print(f"Simulating {model['label']} with {len(payloads)} workers")
    if len(payloads) == 1:
        return simulation_worker(payloads[0])
    with Pool(processes=len(payloads)) as pool:
        return merge_counts(pool.map(simulation_worker, payloads))


def counter_to_pmf(counter):
    total = sum(counter.values())
    if total == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    values = np.fromiter(sorted(counter), dtype=int)
    probabilities = np.fromiter(
        (counter[int(value)] / total for value in values), dtype=float
    )
    return values, probabilities


def aligned_cdfs(real_vals, real_probs, sim_vals, sim_probs):
    support = np.array(
        sorted(set(real_vals.tolist()) | set(sim_vals.tolist())), dtype=int
    )
    real_map = dict(zip(real_vals.astype(int), real_probs.astype(float)))
    sim_map = dict(zip(sim_vals.astype(int), sim_probs.astype(float)))
    real_cdf = np.cumsum([real_map.get(int(x), 0.0) for x in support])
    sim_cdf = np.cumsum([sim_map.get(int(x), 0.0) for x in support])
    return support, real_cdf, sim_cdf


def pair_statistics(real_vals, real_probs, sim_vals, sim_probs):
    if real_vals.size == 0 or sim_vals.size == 0:
        return {"w1": np.nan, "ks": np.nan, "kuiper": np.nan, "singleton": np.nan}
    support, real_cdf, sim_cdf = aligned_cdfs(
        real_vals, real_probs, sim_vals, sim_probs
    )
    difference = real_cdf - sim_cdf
    return {
        "w1": float(wasserstein_distance(
            real_vals, sim_vals, u_weights=real_probs, v_weights=sim_probs
        )),
        "ks": float(np.max(np.abs(difference))),
        "kuiper": float(np.max(difference) - np.min(difference)),
        "singleton": float(sim_probs[sim_vals == 1].sum()),
    }


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def plot_pair(path, pair, real_vals, real_probs, model_pmfs, statistics):
    colors = {
        "real": "#1f77b4",
        MODELS[0]["name"]: "#e76f51",
        MODELS[1]["name"]: "#2a9d8f",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    axes[0].plot(real_vals, real_probs, color=colors["real"], marker="o", label="Real")
    for model in MODELS:
        vals, probs = model_pmfs[model["name"]]
        axes[0].plot(
            vals, probs, color=colors[model["name"]], marker="s",
            markersize=3, alpha=0.9, label=model["label"],
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Synteny block length")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("SBL frequency")
    axes[0].legend(fontsize=9)

    combined_support = set(real_vals.tolist())
    for vals, _ in model_pmfs.values():
        combined_support.update(vals.tolist())
    support = np.array(sorted(combined_support), dtype=int)

    real_map = dict(zip(real_vals.astype(int), real_probs.astype(float)))
    real_cdf = np.cumsum([real_map.get(int(x), 0.0) for x in support])
    axes[1].step(support, real_cdf, where="post", color=colors["real"], linewidth=2.5, label="Real")
    for model in MODELS:
        vals, probs = model_pmfs[model["name"]]
        sim_map = dict(zip(vals.astype(int), probs.astype(float)))
        sim_cdf = np.cumsum([sim_map.get(int(x), 0.0) for x in support])
        stat = statistics[model["name"]]
        label = (
            f"{model['label']} (W1={stat['w1']:.2f}, "
            f"KS={stat['ks']:.3f}, V={stat['kuiper']:.3f})"
        )
        axes[1].step(
            support, sim_cdf, where="post", color=colors[model["name"]],
            linewidth=2.2, label=label,
        )
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Synteny block length")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("SBL cumulative distribution")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{pair[0]} vs {pair[1]}", fontsize=15)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    atgc_dir = os.path.abspath(args.atgc_dir)
    tree_path = os.path.join(atgc_dir, args.tree_filename)
    cc_path = os.path.join(atgc_dir, args.cc_filename)
    os.makedirs(args.out_dir, exist_ok=True)

    real_pmfs, tree, real_genomes, _ = build_real_pmfs(tree_path, cc_path)
    root_genome = make_root_genome(
        args.root_mode, tree, cc_path, real_genomes=real_genomes
    )
    model_counts = {
        model["name"]: simulate_model(args, tree, root_genome, model)
        for model in MODELS
    }

    rows = []
    for pair, (real_vals, real_probs) in sorted(real_pmfs.items()):
        model_pmfs = {}
        statistics = {}
        real_singleton = float(real_probs[real_vals == 1].sum())
        for model in MODELS:
            vals, probs = counter_to_pmf(model_counts[model["name"]].get(pair, Counter()))
            model_pmfs[model["name"]] = (vals, probs)
            statistics[model["name"]] = pair_statistics(
                real_vals, real_probs, vals, probs
            )

        stem = safe_name(f"{pair[0]}__vs__{pair[1]}")
        plot_pair(
            os.path.join(args.out_dir, f"{stem}.png"), pair,
            real_vals, real_probs, model_pmfs, statistics,
        )
        row = {
            "pair_a": pair[0],
            "pair_b": pair[1],
            "real_singleton_frequency": real_singleton,
        }
        for model in MODELS:
            name = model["name"]
            stat = statistics[name]
            row.update({
                f"{name}_w1": stat["w1"],
                f"{name}_ks": stat["ks"],
                f"{name}_kuiper": stat["kuiper"],
                f"{name}_singleton_frequency": stat["singleton"],
                f"{name}_singleton_abs_error": abs(stat["singleton"] - real_singleton),
            })
        row["singleton_error_improvement_exp3"] = (
            row["single_gene_rt_0.1_singleton_abs_error"]
            - row["exponent_3_rt_0.316228_singleton_abs_error"]
        )
        rows.append(row)

    summary_path = os.path.join(args.out_dir, "per_pair_comparison.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    improved = sum(row["singleton_error_improvement_exp3"] > 0 for row in rows)
    print(f"Wrote {len(rows)} pair figures to {args.out_dir}")
    print(f"Wrote {summary_path}")
    print(f"Exponent 3 improves singleton error in {improved}/{len(rows)} pairs")


if __name__ == "__main__":
    main()
