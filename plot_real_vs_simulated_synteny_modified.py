#!/usr/bin/env python3
"""Plot real vs simulated synteny-block length distributions for one ATGC.

Example:
    python plot_real_vs_simulated_synteny.py \
      --atgc-dir ATGC0070 \
      --rf 79.8 \
      --rt 6.2 \
      --n-runs 100 \
      --out-prefix ATGC0070_bestfit
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from prod_1b import build_real_pmfs, make_root_genome
from simulation import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real and simulated synteny-block distributions."
    )
    parser.add_argument("--atgc-dir", required=True, help="ATGC folder, e.g. ATGC0070.")
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf", type=float, required=True, help="Gain/loss parameter.")
    parser.add_argument("--rt", type=float, required=True, help="Translocation parameter.")
    parser.add_argument("--inv-rate", type=float, default=0.0)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes used to split n-runs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--huge-exp", type=float, default=1e9)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional maximum block length to show on the x-axis.",
    )
    return parser.parse_args()


def split_counts(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base = total // chunks
    rem = total % chunks
    return [base + (1 if i < rem else 0) for i in range(chunks)]


def pmfs_to_counts(real_pmfs: dict) -> Counter:
    """Convert per-pair real PMFs into one pooled distribution."""
    counts = Counter()
    for vals, probs in real_pmfs.values():
        for val, prob in zip(vals, probs):
            counts[int(val)] += float(prob)
    return counts


def simulate_counts_worker(payload) -> Counter:
    tree, root_genome, rf, rt, inv_rate, huge_exp, n_runs, seed = payload
    counts = Counter()
    rng = np.random.default_rng(seed)

    for _ in range(n_runs):
        child_seed = int(rng.integers(0, 2**32 - 1))
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
        )
        for lengths in sim_pairs.values():
            counts.update(int(length) for length in lengths)

    return counts


def simulate_counts(args: argparse.Namespace, tree, root_genome) -> Counter:
    run_counts = split_counts(args.n_runs, args.workers)
    seed_rng = np.random.default_rng(args.seed)
    payloads = []
    for n in run_counts:
        worker_seed = int(seed_rng.integers(0, 2**32 - 1))
        payloads.append((
            tree,
            root_genome,
            args.rf,
            args.rt,
            args.inv_rate,
            args.huge_exp,
            n,
            worker_seed,
        ))

    if len(payloads) == 1:
        return simulate_counts_worker(payloads[0])

    merged = Counter()
    with Pool(processes=len(payloads)) as pool:
        for partial in pool.map(simulate_counts_worker, payloads):
            merged.update(partial)
    return merged


def normalize(counter: Counter) -> dict[int, float]:
    total = float(sum(counter.values()))
    if total == 0:
        return {}
    return {int(k): float(v) / total for k, v in sorted(counter.items())}


def write_distribution_csv(path: str, real_freq: dict[int, float], sim_freq: dict[int, float]) -> None:
    lengths = sorted(set(real_freq) | set(sim_freq))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["block_length", "real_frequency", "simulated_frequency"],
        )
        writer.writeheader()
        for length in lengths:
            writer.writerow({
                "block_length": length,
                "real_frequency": f"{real_freq.get(length, 0.0):.12g}",
                "simulated_frequency": f"{sim_freq.get(length, 0.0):.12g}",
            })


def plot_distribution(
    path: str,
    real_freq: dict[int, float],
    sim_freq: dict[int, float],
    title: str,
    max_length: int | None,
) -> None:
    lengths = sorted(set(real_freq) | set(sim_freq))
    if max_length is not None:
        lengths = [x for x in lengths if x <= max_length]

    real_y = [real_freq.get(x, 0.0) for x in lengths]
    sim_y = [sim_freq.get(x, 0.0) for x in lengths]

    plt.figure(figsize=(10, 5.5))
    plt.plot(lengths, real_y, marker="o", linewidth=2, label="Real")
    plt.plot(lengths, sim_y, marker="s", linewidth=2, label="Simulated")
    plt.xlabel("Synteny block length")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    atgc_dir = os.path.abspath(args.atgc_dir)
    tree_path = os.path.join(atgc_dir, args.tree_filename)
    cc_path = os.path.join(atgc_dir, args.cc_filename)

    out_prefix = args.out_prefix
    if out_prefix is None:
        dataset = os.path.basename(os.path.normpath(atgc_dir))
        out_prefix = f"{dataset}_rf_{args.rf:g}_rt_{args.rt:g}_synteny_fit"

    real_pmfs, tree, real_genomes, _ = build_real_pmfs(tree_path, cc_path)
    root_genome = make_root_genome(
        args.root_mode,
        tree,
        cc_path,
        real_genomes=real_genomes,
    )

    real_counts = pmfs_to_counts(real_pmfs)
    sim_counts = simulate_counts(args, tree, root_genome)
    real_freq = normalize(real_counts)
    sim_freq = normalize(sim_counts)

    csv_path = f"{out_prefix}.csv"
    png_path = f"{out_prefix}.png"
    write_distribution_csv(csv_path, real_freq, sim_freq)
    plot_distribution(
        png_path,
        real_freq,
        sim_freq,
        title=(
            f"Real vs simulated synteny-block distribution "
            f"(rf={args.rf:g}, rt={args.rt:g}, n={args.n_runs}, "
            f"workers={min(args.workers, args.n_runs)})"
        ),
        max_length=args.max_length,
    )

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
