#!/usr/bin/env python3
"""Plot real vs simulated synteny-block length distributions per genome pair.

Example:
    python plot_per_pair_synteny_fit.py \
      --atgc-dir ATGC0070 \
      --rf 0.3981071706 \
      --rt 3.981071706 \
      --n-runs 100 \
      --out-dir per_pair_synteny_fit
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from prod_1b import build_real_pmfs, make_root_genome
from simulation import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot real vs simulated SBL distributions for each genome pair."
    )
    parser.add_argument("--atgc-dir", required=True, help="ATGC folder, e.g. ATGC0070.")
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf", type=float, required=True, help="Gain/loss rate.")
    parser.add_argument("--rt", type=float, required=True, help="Translocation rate.")
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
    parser.add_argument("--out-dir", default="per_pair_synteny_fit")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--plot-cdf",
        action="store_true",
        help="Also write one CDF plot per genome pair.",
    )
    return parser.parse_args()


def split_counts(total: int, chunks: int) -> list[int]:
    chunks = max(1, min(chunks, total))
    base = total // chunks
    rem = total % chunks
    return [base + (1 if i < rem else 0) for i in range(chunks)]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def counts_to_freq(counter: Counter) -> dict[int, float]:
    total = float(sum(counter.values()))
    if total == 0:
        return {}
    return {int(k): float(v) / total for k, v in sorted(counter.items())}


def pmf_to_freq(vals: np.ndarray, probs: np.ndarray) -> dict[int, float]:
    return {int(v): float(p) for v, p in zip(vals, probs)}


def freq_to_arrays(freq: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    if not freq:
        return np.array([], dtype=int), np.array([], dtype=float)
    vals = np.fromiter(sorted(freq.keys()), dtype=int)
    probs = np.fromiter((freq[int(v)] for v in vals), dtype=float)
    return vals, probs


def simulate_pair_counts_worker(payload) -> dict[tuple[str, str], Counter]:
    tree, root_genome, rf, rt, inv_rate, huge_exp, n_runs, seed = payload
    rng = np.random.default_rng(seed)
    pooled_counts = defaultdict(Counter)

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

        for (a, b), lengths in sim_pairs.items():
            pair = (a, b) if a <= b else (b, a)
            pooled_counts[pair].update(int(x) for x in lengths)

    return pooled_counts


def merge_pair_counts(partials) -> dict[tuple[str, str], Counter]:
    merged = defaultdict(Counter)
    for partial in partials:
        for pair, counter in partial.items():
            merged[pair].update(counter)
    return merged


def simulate_pair_counts(args: argparse.Namespace, tree, root_genome) -> dict[tuple[str, str], Counter]:
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

    print(f"running {args.n_runs} simulations across {len(payloads)} worker(s)")
    if len(payloads) == 1:
        return simulate_pair_counts_worker(payloads[0])

    with Pool(processes=len(payloads)) as pool:
        partials = pool.map(simulate_pair_counts_worker, payloads)
    return merge_pair_counts(partials)


def write_pair_csv(path: str, real_freq: dict[int, float], sim_freq: dict[int, float]) -> None:
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


def plot_frequency(
    path: str,
    pair: tuple[str, str],
    real_freq: dict[int, float],
    sim_freq: dict[int, float],
    w1: float,
    args: argparse.Namespace,
) -> None:
    lengths = sorted(set(real_freq) | set(sim_freq))
    if args.max_length is not None:
        lengths = [x for x in lengths if x <= args.max_length]

    real_y = [real_freq.get(x, 0.0) for x in lengths]
    sim_y = [sim_freq.get(x, 0.0) for x in lengths]

    plt.figure(figsize=(10, 5.5))
    plt.plot(lengths, real_y, marker="o", linewidth=2, label="Real")
    plt.plot(lengths, sim_y, marker="s", linewidth=2, label="Simulated")
    plt.xlabel("Synteny block length")
    plt.ylabel("Frequency")
    plt.title(
        f"{pair[0]} vs {pair[1]}\n"
        f"rf={args.rf:g}, rt={args.rt:g}, n={args.n_runs}, W1={w1:.4g}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cdf(
    path: str,
    pair: tuple[str, str],
    real_freq: dict[int, float],
    sim_freq: dict[int, float],
    w1: float,
    args: argparse.Namespace,
) -> None:
    lengths = sorted(set(real_freq) | set(sim_freq))
    if args.max_length is not None:
        lengths = [x for x in lengths if x <= args.max_length]

    real = np.array([real_freq.get(x, 0.0) for x in lengths], dtype=float)
    sim = np.array([sim_freq.get(x, 0.0) for x in lengths], dtype=float)
    if real.sum() > 0:
        real = np.cumsum(real / real.sum())
    if sim.sum() > 0:
        sim = np.cumsum(sim / sim.sum())

    plt.figure(figsize=(8, 5.5))
    plt.step(lengths, real, where="post", linewidth=2.5, label="Real CDF")
    plt.step(lengths, sim, where="post", linewidth=2.5, label="Simulated CDF")
    plt.xscale("log")
    plt.ylim(0, 1.02)
    plt.xlabel("Synteny block length")
    plt.ylabel("CDF")
    plt.title(f"{pair[0]} vs {pair[1]}\nW1={w1:.4g}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    atgc_dir = os.path.abspath(args.atgc_dir)
    tree_path = os.path.join(atgc_dir, args.tree_filename)
    cc_path = os.path.join(atgc_dir, args.cc_filename)
    os.makedirs(args.out_dir, exist_ok=True)

    real_pmfs, tree, real_genomes, _ = build_real_pmfs(tree_path, cc_path)
    root_genome = make_root_genome(
        args.root_mode,
        tree,
        cc_path,
        real_genomes=real_genomes,
    )
    sim_counts = simulate_pair_counts(args, tree, root_genome)

    summary_rows = []
    for pair, (real_vals, real_probs) in sorted(real_pmfs.items()):
        real_freq = pmf_to_freq(real_vals, real_probs)
        sim_freq = counts_to_freq(sim_counts.get(pair, Counter()))
        sim_vals, sim_probs = freq_to_arrays(sim_freq)

        if real_vals.size and sim_vals.size:
            w1 = float(wasserstein_distance(
                real_vals,
                sim_vals,
                u_weights=real_probs,
                v_weights=sim_probs,
            ))
        else:
            w1 = float("nan")

        stem = safe_name(f"{pair[0]}__vs__{pair[1]}")
        csv_path = os.path.join(args.out_dir, f"{stem}.csv")
        png_path = os.path.join(args.out_dir, f"{stem}.png")
        write_pair_csv(csv_path, real_freq, sim_freq)
        plot_frequency(png_path, pair, real_freq, sim_freq, w1, args)

        if args.plot_cdf:
            cdf_path = os.path.join(args.out_dir, f"{stem}_cdf.png")
            plot_cdf(cdf_path, pair, real_freq, sim_freq, w1, args)

        summary_rows.append({
            "pair_a": pair[0],
            "pair_b": pair[1],
            "w1": w1,
            "real_singleton_frequency": real_freq.get(1, 0.0),
            "simulated_singleton_frequency": sim_freq.get(1, 0.0),
            "real_n_lengths": int(len(real_vals)),
            "simulated_n_blocks": int(sum(sim_counts.get(pair, Counter()).values())),
            "csv": csv_path,
            "png": png_path,
        })

    summary_rows.sort(key=lambda row: row["w1"])
    summary_path = os.path.join(args.out_dir, "per_pair_summary_ranked.csv")
    with open(summary_path, "w", newline="") as fh:
        fieldnames = [
            "rank",
            "pair_a",
            "pair_b",
            "w1",
            "real_singleton_frequency",
            "simulated_singleton_frequency",
            "real_n_lengths",
            "simulated_n_blocks",
            "csv",
            "png",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(summary_rows, start=1):
            writer.writerow({
                **row,
                "rank": rank,
                "w1": f"{row['w1']:.12g}",
                "real_singleton_frequency": f"{row['real_singleton_frequency']:.12g}",
                "simulated_singleton_frequency": f"{row['simulated_singleton_frequency']:.12g}",
            })

    print(f"wrote per-pair plots and CSVs to {args.out_dir}")
    print(f"wrote ranked summary to {summary_path}")


if __name__ == "__main__":
    main()
