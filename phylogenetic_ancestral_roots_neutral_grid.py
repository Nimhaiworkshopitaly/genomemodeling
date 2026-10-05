#!/usr/bin/env python3
"""Neutral rf x rt landscapes using phylogenetically reconstructed roots."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from multiple_observed_roots_neutral_grid import (
    BIOWULF_TREE,
    FALLBACK_TREE,
    METRICS,
    RF_VALUES,
    RT_VALUES,
    SEEDS,
    format_rate,
    token,
)
from prod_1b_core_composite import build_real_pmfs, convert_to_numeric


N_ROOTS = 5
ROOT_SEEDS = (1801, 2801, 3801, 4801, 5801)
ROOT = Path("hpc_multicpu/phylogenetic_ancestral_roots_neutral")
ROOT_GENOMES = ROOT / "root_genomes"
ROOTS_CSV = ROOT / "ancestral_roots.csv"
MODEL_CSV = ROOT / "reconstruction_model.csv"
RESULTS = ROOT / "results"
PARTS = ROOT / "parts"
FIGURES = ROOT / "figures"
EVALUATOR = Path("hpc_multicpu/multicpu_eval_one_setting_core_composite.py")


def observed_data():
    tree_path = BIOWULF_TREE if BIOWULF_TREE.is_file() else FALLBACK_TREE
    if not tree_path.is_file():
        raise FileNotFoundError(f"Neither {BIOWULF_TREE} nor {FALLBACK_TREE} exists")
    _, tree, real_genomes, _ = build_real_pmfs(
        str(tree_path), "ATGC0070/atgc.cc.csv"
    )
    leaf_names = [
        leaf.name for leaf in tree.get_terminals() if leaf.name in real_genomes
    ]
    if len(leaf_names) < 2:
        raise ValueError("At least two tree leaves must match observed genomes")
    genomes = {
        name: convert_to_numeric(real_genomes[name]) for name in leaf_names
    }
    return tree, leaf_names, genomes


def circular_edges(genome):
    if len(genome) < 2:
        return set()
    return {
        tuple(sorted((genome[index], genome[(index + 1) % len(genome)])))
        for index in range(len(genome))
        if genome[index] != genome[(index + 1) % len(genome)]
    }


def transition_matrix(gain_rate, loss_rate, branch_length):
    total = gain_rate + loss_rate
    stationary_one = gain_rate / total
    stationary_zero = loss_rate / total
    decay = math.exp(-total * max(float(branch_length or 0.0), 1e-12))
    return np.asarray([
        [stationary_zero + stationary_one * decay,
         stationary_one * (1.0 - decay)],
        [stationary_zero * (1.0 - decay),
         stationary_one + stationary_zero * decay],
    ])


def root_posteriors(tree, leaf_names, states, gain_rate, loss_rate):
    """Return marginal P(root state=1) for binary features using pruning."""
    leaf_index = {name: index for index, name in enumerate(leaf_names)}

    def likelihood(clade):
        if clade.is_terminal():
            observed = states[leaf_index[clade.name]]
            result = np.zeros((states.shape[1], 2), dtype=float)
            result[:, 0] = observed == 0
            result[:, 1] = observed == 1
            return result
        result = np.ones((states.shape[1], 2), dtype=float)
        for child in clade.clades:
            child_likelihood = likelihood(child)
            transition = transition_matrix(
                gain_rate, loss_rate, child.branch_length
            )
            result *= child_likelihood @ transition.T
        scale = result.max(axis=1, keepdims=True)
        result /= np.maximum(scale, np.finfo(float).tiny)
        return result

    root_likelihood = likelihood(tree.root)
    prior = np.asarray([
        loss_rate / (gain_rate + loss_rate),
        gain_rate / (gain_rate + loss_rate),
    ])
    weighted = root_likelihood * prior
    return weighted[:, 1] / np.maximum(weighted.sum(axis=1), np.finfo(float).tiny)


def feature_matrix(leaf_names, feature_sets, features):
    feature_index = {feature: index for index, feature in enumerate(features)}
    states = np.zeros((len(leaf_names), len(features)), dtype=np.uint8)
    for row, name in enumerate(leaf_names):
        for feature in feature_sets[name]:
            states[row, feature_index[feature]] = 1
    return states


def order_genes(selected, sampled_edges, edge_probabilities, rng):
    """Construct a circular order by extending the two ends of a weighted path."""
    selected = set(selected)
    neighbors = defaultdict(list)
    for edge in sampled_edges:
        left, right = edge
        if left in selected and right in selected:
            weight = edge_probabilities[edge]
            neighbors[left].append((right, weight))
            neighbors[right].append((left, weight))
    first = max(selected, key=lambda gene: sum(w for _, w in neighbors[gene]))
    path = [first]
    available = selected - {first}
    supported = 0
    fallback = 0
    while available:
        candidates = []
        for side, endpoint in (("left", path[0]), ("right", path[-1])):
            for gene, weight in neighbors[endpoint]:
                if gene in available:
                    candidates.append((side, gene, weight))
        if candidates:
            weights = np.asarray([item[2] for item in candidates], dtype=float)
            weights /= weights.sum()
            side, gene, _ = candidates[int(rng.choice(len(candidates), p=weights))]
            supported += 1
        else:
            gene = int(rng.choice(sorted(available)))
            side = "right"
            fallback += 1
        if side == "left":
            path.insert(0, gene)
        else:
            path.append(gene)
        available.remove(gene)
    closure = int(tuple(sorted((path[0], path[-1]))) in sampled_edges)
    return path, supported + closure, fallback, closure


def calibrated_rates(tree, states, turnover_depth):
    """Set stationary prevalence empirically and calibrate total turnover depth."""
    median_depth = float(np.median([
        tree.distance(tree.root, leaf) for leaf in tree.get_terminals()
    ]))
    if median_depth <= 0:
        raise ValueError("Tree has a nonpositive median root-to-leaf distance")
    stationary_one = float(states.mean())
    total_rate = turnover_depth / median_depth
    return (
        total_rate * stationary_one,
        total_rate * (1.0 - stationary_one),
        median_depth,
    )


def generate_roots(n_roots, turnover_depth):
    tree, leaf_names, genomes = observed_data()
    cog_sets = {name: set(genomes[name]) for name in leaf_names}
    edge_sets = {name: circular_edges(genomes[name]) for name in leaf_names}
    cogs = sorted(set().union(*cog_sets.values()))
    edges = sorted(set().union(*edge_sets.values()))
    cog_states = feature_matrix(leaf_names, cog_sets, cogs)
    edge_states = feature_matrix(leaf_names, edge_sets, edges)

    cog_gain, cog_loss, median_depth = calibrated_rates(
        tree, cog_states, turnover_depth
    )
    edge_gain, edge_loss, _ = calibrated_rates(
        tree, edge_states, turnover_depth
    )
    cog_probabilities = root_posteriors(
        tree, leaf_names, cog_states, cog_gain, cog_loss
    )
    edge_probability_values = root_posteriors(
        tree, leaf_names, edge_states, edge_gain, edge_loss
    )
    edge_probabilities = dict(zip(edges, edge_probability_values))

    ROOT_GENOMES.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    model = pd.DataFrame([
        {
            "feature_type": "COG presence",
            "model": "empirical stationary frequency; calibrated turnover",
            "turnover_depth": turnover_depth,
            "median_root_to_leaf": median_depth,
            "gain_rate": cog_gain,
            "loss_rate": cog_loss,
            "features": len(cogs),
        },
        {
            "feature_type": "adjacency presence",
            "model": "empirical stationary frequency; calibrated turnover",
            "turnover_depth": turnover_depth,
            "median_root_to_leaf": median_depth,
            "gain_rate": edge_gain,
            "loss_rate": edge_loss,
            "features": len(edges),
        },
    ])
    model.to_csv(MODEL_CSV, index=False)

    rows = []
    observed_lengths = np.asarray([len(genomes[name]) for name in leaf_names])
    for index in range(n_roots):
        seed = ROOT_SEEDS[index] if index < len(ROOT_SEEDS) else ROOT_SEEDS[-1] + index
        rng = np.random.default_rng(seed)
        if index == 0:
            target_length = int(np.median(observed_lengths))
            ranked = np.argsort(cog_probabilities)[::-1][:target_length]
            selected = {cogs[i] for i in ranked}
            reconstruction = "MAP content, posterior-weighted order"
        else:
            target_length = int(rng.choice(observed_lengths))
            weights = np.maximum(cog_probabilities, np.finfo(float).tiny)
            weights /= weights.sum()
            selected_indices = rng.choice(
                len(cogs), size=target_length, replace=False, p=weights
            )
            selected = {cogs[i] for i in selected_indices}
            reconstruction = "posterior content and order sample"
        if len(selected) < 2:
            raise ValueError("Ancestral reconstruction selected fewer than two COGs")
        # Every candidate edge was observed at a leaf; its root posterior controls
        # the stochastic ordering weight instead of imposing an arbitrary cutoff.
        sampled_edges = set(edges)
        ordered, supported, fallback, closure = order_genes(
            selected, sampled_edges, edge_probabilities, rng
        )
        synthetic_id = f"phylo_root_{index + 1:02d}"
        path = ROOT_GENOMES / f"{synthetic_id}.json"
        path.write_text(json.dumps(ordered) + "\n")
        rows.append({
            "root_index": index + 1,
            "synthetic_id": synthetic_id,
            "reconstruction": reconstruction,
            "root_seed": seed,
            "target_length": target_length,
            "genome_length": len(ordered),
            "unique_cogs": len(set(ordered)),
            "reconstructed_adjacency_edges": supported,
            "adjacency_support_fraction": supported / len(ordered),
            "fallback_jumps": fallback,
            "circular_closure_reconstructed": closure,
            "mean_selected_cog_posterior": float(np.mean([
                cog_probabilities[cogs.index(gene)] for gene in selected
            ])),
            "root_file": str(path),
            "observed_genomes_used": len(leaf_names),
            "pangenome_size": len(cogs),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOTS_CSV, index=False)
    return frame, model


def load_roots():
    if not ROOTS_CSV.is_file():
        raise SystemExit(f"Missing {ROOTS_CSV}; run generate-roots first")
    roots = pd.read_csv(ROOTS_CSV, dtype={"synthetic_id": str, "root_file": str})
    for path in roots["root_file"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return roots


def result_path(synthetic_id, rf, rt, seed):
    return RESULTS / (
        f"root_{synthetic_id}_rf_{token(rf)}_rt_{token(rt)}_seed_{seed}.csv"
    )


def write_parts(commands, jobs_per_part):
    PARTS.mkdir(parents=True, exist_ok=True)
    for old in PARTS.glob("part_*.swarm"):
        old.unlink()
    paths = []
    for start in range(0, len(commands), jobs_per_part):
        path = PARTS / f"part_{len(paths) + 1:03d}.swarm"
        path.write_text("\n".join(commands[start:start + jobs_per_part]) + "\n")
        paths.append(path)
    return paths


def make_jobs(jobs_per_part):
    roots = load_roots()
    RESULTS.mkdir(parents=True, exist_ok=True)
    commands, skipped = [], 0
    for root in roots.itertuples(index=False):
        for rf in RF_VALUES:
            for rt in RT_VALUES:
                for seed in SEEDS:
                    output = result_path(root.synthetic_id, rf, rt, seed)
                    if output.is_file() and output.stat().st_size > 0:
                        skipped += 1
                        continue
                    command = [
                        sys.executable, str(EVALUATOR.resolve()),
                        "--atgc-dir", "ATGC0070",
                        "--tree-filename", "yuri_gl26/ATGC0070.gl.tre",
                        "--root-mode", "phylogenetic_reconstruction",
                        "--root-genome-file", root.root_file,
                        "--rf", f"{rf:.12g}", "--rt", f"{rt:.12g}",
                        "--inv-rate", "0", "--huge-exp", "1e9",
                        "--core-mode", "synthetic_fraction",
                        "--core-fraction", "0", "--core-protection", "0",
                        "--n-runs", "100", "--workers", "16",
                        "--seed", str(seed), "--out-csv", str(output),
                    ]
                    commands.append(shlex.join(command))
    parts = write_parts(commands, jobs_per_part)
    expected = len(roots) * len(RF_VALUES) * len(RT_VALUES) * len(SEEDS)
    print(f"Wrote {len(commands)} jobs in {len(parts)} parts: {PARTS}")
    print(f"Skipped {skipped}; complete experiment = {expected} jobs")


def status():
    roots = load_roots()
    expected = len(RF_VALUES) * len(RT_VALUES) * len(SEEDS)
    total = 0
    for synthetic_id in roots["synthetic_id"]:
        count = sum(
            result_path(synthetic_id, rf, rt, seed).is_file()
            and result_path(synthetic_id, rf, rt, seed).stat().st_size > 0
            for rf in RF_VALUES for rt in RT_VALUES for seed in SEEDS
        )
        total += count
        print(f"{synthetic_id}: {count}/{expected} completed")
    print(f"total: {total}/{expected * len(roots)} completed")


def load_results():
    roots = load_roots()
    frames = []
    for root in roots.itertuples(index=False):
        expected_label = f"synthetic:{root.synthetic_id}"
        for rf in RF_VALUES:
            for rt in RT_VALUES:
                for seed in SEEDS:
                    path = result_path(root.synthetic_id, rf, rt, seed)
                    if not path.is_file() or path.stat().st_size == 0:
                        raise SystemExit(f"Missing result: {path}")
                    frame = pd.read_csv(path)
                    if len(frame) != 1:
                        raise ValueError(f"{path}: expected one row")
                    if str(frame.iloc[0]["selected_root_genome_id"]) != expected_label:
                        raise ValueError(f"{path}: unexpected root identifier")
                    frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if not np.allclose(data["core_protection"], 0):
        raise ValueError("Some results used core protection")
    return roots, data


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    roots, data = load_results()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURES / "combined_results.csv", index=False)
    available_metrics = []
    for metric, label in METRICS:
        if metric not in data.columns:
            print(f"Skipping {metric}: column is absent from the result files")
            continue
        values = pd.to_numeric(data[metric], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if not values.notna().any():
            print(f"Skipping {metric}: column has no finite values")
            continue
        available_metrics.append((metric, label))
    if not available_metrics:
        raise SystemExit("No plottable metric columns were found")
    means = data.groupby(
        ["selected_root_genome_id", "rf", "rt"], as_index=False
    )[[metric for metric, _ in available_metrics]].mean()
    means.to_csv(FIGURES / "mean_results.csv", index=False)
    ncols, nrows = min(3, len(roots)), math.ceil(len(roots) / 3)
    x, y = np.log10(RT_VALUES), np.log10(RF_VALUES)
    xx, yy = np.meshgrid(x, y)
    for metric, label in available_metrics:
        matrices, optima = [], []
        for root in roots.itertuples(index=False):
            root_label = f"synthetic:{root.synthetic_id}"
            subset = means[means["selected_root_genome_id"].eq(root_label)]
            finite = pd.to_numeric(subset[metric], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if finite.empty:
                raise ValueError(f"{root.synthetic_id}, {metric}: no finite values")
            matrix = subset.pivot(index="rf", columns="rt", values=metric)
            matrix = matrix.sort_index().sort_index(axis=1)
            expected_shape = (len(RF_VALUES), len(RT_VALUES))
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"{root.synthetic_id}, {metric}: expected landscape shape "
                    f"{expected_shape}, found {matrix.shape}"
                )
            matrices.append(matrix.to_numpy())
            optima.append(subset.loc[finite.idxmin()])
        all_values = np.concatenate([matrix.ravel() for matrix in matrices])
        norm = Normalize(float(np.nanmin(all_values)), float(np.nanmax(all_values)))
        levels = np.linspace(norm.vmin, norm.vmax, 18)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 5.8 * nrows),
            sharex=True, sharey=True, constrained_layout=True, squeeze=False,
        )
        contour = None
        for index, (root, matrix, optimum) in enumerate(
            zip(roots.itertuples(index=False), matrices, optima)
        ):
            ax = axes.flat[index]
            contour = ax.contourf(
                xx, yy, matrix, levels=levels, cmap="viridis_r",
                norm=norm, extend="both"
            )
            lines = ax.contour(
                xx, yy, matrix, levels=6, colors="black",
                linewidths=0.55, alpha=0.55
            )
            ax.clabel(lines, inline=True, fontsize=7, fmt="%.3g")
            ax.scatter(
                math.log10(optimum["rt"]), math.log10(optimum["rf"]),
                marker="*", s=360, facecolors="none", edgecolors="red",
                linewidths=2.4, zorder=5
            )
            ax.set_title(
                f"Ancestral root {root.root_index} ({root.reconstruction}); "
                f"length={root.genome_length}\n"
                f"best={optimum[metric]:.5g}; rf={optimum['rf']:.5g}; "
                f"rt={optimum['rt']:.5g}", fontsize=11,
            )
            ax.grid(color="white", alpha=0.16, linewidth=0.7)
        for index in range(len(roots), nrows * ncols):
            axes.flat[index].set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Single-gene translocation rate (rt)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Gene gain/loss rate (rf)")
        for ax in axes.flat[:len(roots)]:
            ax.set_xticks(x, [format_rate(value) for value in RT_VALUES],
                          rotation=45, ha="right")
            ax.set_yticks(y, [format_rate(value) for value in RF_VALUES])
        colorbar = fig.colorbar(contour, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label(label)
        fig.suptitle(
            f"{label} landscapes across phylogenetically reconstructed roots\n"
            "Neutral model: no core protection; mean across five matched seeds; "
            "lower is better", fontsize=17,
        )
        stem = FIGURES / f"{metric}_phylogenetic_ancestral_roots_neutral"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {stem.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("generate-roots", "make-jobs", "status", "analyze")
    )
    parser.add_argument("--n-roots", type=int, default=N_ROOTS)
    parser.add_argument("--jobs-per-part", type=int, default=500)
    parser.add_argument(
        "--turnover-depth", type=float, default=0.1,
        help=("Expected binary-state turnover across the median root-to-leaf "
              "distance during ancestral reconstruction"),
    )
    args = parser.parse_args()
    if args.n_roots < 2:
        parser.error("--n-roots must be at least 2")
    if args.turnover_depth <= 0:
        parser.error("--turnover-depth must be positive")
    if args.action == "generate-roots":
        roots, model = generate_roots(args.n_roots, args.turnover_depth)
        print("Fitted ancestral models:")
        print(model.to_string(index=False))
        print("\nReconstructed roots:")
        print(roots.to_string(index=False))
    elif args.action == "make-jobs":
        make_jobs(args.jobs_per_part)
    elif args.action == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
