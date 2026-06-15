from __future__ import annotations
import os, csv, math, json, time
from datetime import datetime
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from Bio import Phylo
from scipy.stats import wasserstein_distance

from synteny_tools import findSyntenyReal2

from simulation_core_composite import run_simulation  # expects your signature
import pandas as pd

# Cache: (tree_path, cc_path, finder_name) -> (real_pmfs, tree, genomes_numeric, median_root_to_leaf)
_RPMFS_CACHE: dict[tuple[str, str, str], tuple[dict, object, dict, float]] = {}

import inspect, random

DEFAULT_COMPOSITE_WEIGHTS = {
    "w1": 1.0,
    "singleton": 100.0,
    "short_cdf": 100.0,
    "long_tail": 100.0,
}

# ---------------------------
# Helpers (self-contained)
# ---------------------------
def convert_to_numeric(cls_list):
    # Accepts like ["cls.123", "cls.45"] or ["cog.123", ...] -> ints
    out = []
    for x in cls_list:
        if isinstance(x, int):
            out.append(int(x))
        else:
            s = str(x)
            if "." in s:
                s = s.split(".", 1)[1]
            out.append(int(s))
    return out


def load_real_genomes_from_cc(cc_path):
    """
    Reads ATGC cc file into dict: genome_id -> [cls/cog IDs as strings]
    Works for both 'cls_ID' or 'atgc_cog_ID' column names.
    """
    df = pd.read_csv(
        cc_path,
        names=[
            "gene_ID", "genome_ID", "protein_ID", "protein_length",
            "atgc_cog_footprint", "atgc_cog_footprint_length",
            "atgc_cog_ID", "protein_cluster_ID", "match_class"
        ],
        dtype=str,
    )
    # Back-compat if older schema uses "cls_ID"
    if "atgc_cog_ID" not in df.columns or df["atgc_cog_ID"].isnull().all():
        df = pd.read_csv(
            cc_path,
            names=[
                "gene_ID", "genome_ID", "protein_ID", "protein_length",
                "atgc_cog_footprint", "atgc_cog_footprint_length",
                "cog_ID", "cls_ID", "match_class"
            ],
            dtype=str,
        )
        key_col = "cls_ID"
    else:
        key_col = "atgc_cog_ID"

    df = df.dropna(subset=[key_col])
    return df.groupby("genome_ID")[key_col].apply(list).to_dict()


def lengths_to_pmf(lengths):
    """
    list[int] -> (vals[np.int64], probs[np.float64]) for discrete pmf.
    Returns ([],[]) if empty.
    """
    c = Counter(lengths)
    tot = sum(c.values())
    if tot == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    vals = np.fromiter(sorted(c.keys()), dtype=int)
    probs = np.fromiter((c[v] / tot for v in vals), dtype=float)
    return vals, probs


def median_root_to_leaf(tree):
    return float(np.median([tree.distance(tree.root, lf) for lf in tree.get_terminals()]))


def _pmf_probability_at(vals, probs, length):
    if vals.size == 0:
        return 0.0
    hits = vals == length
    if not np.any(hits):
        return 0.0
    return float(probs[hits].sum())


def _pmf_cdf_at(vals, probs, max_length):
    if vals.size == 0:
        return 0.0
    return float(probs[vals <= max_length].sum())


def _pmf_tail_at(vals, probs, min_length):
    if vals.size == 0:
        return 0.0
    return float(probs[vals >= min_length].sum())


def score_real_vs_sim_counts(
    real_pmfs,
    pooled_sim_counts,
    composite_weights=None,
    short_cdf_length=10,
    long_tail_length=50,
):
    """
    Compare real per-pair SBL PMFs to pooled simulated per-pair SBL counts.

    The composite score keeps avg_w1 but adds interpretable penalties:
    singleton mismatch, short-block CDF mismatch, and long-tail mismatch.
    Smaller values are better.
    """
    weights = dict(DEFAULT_COMPOSITE_WEIGHTS)
    if composite_weights:
        weights.update(composite_weights)

    total_w1 = 0.0
    total_singleton_abs_error = 0.0
    total_short_cdf_abs_error = 0.0
    total_long_tail_abs_error = 0.0
    compared = 0
    skipped_real = 0
    skipped_sim = 0

    for pair, (r_vals, r_probs) in real_pmfs.items():
        if r_vals.size == 0:
            skipped_real += 1
            continue
        sim_counts = pooled_sim_counts.get(pair)
        if not sim_counts:
            skipped_sim += 1
            continue
        s_total = sum(sim_counts.values())
        if s_total == 0:
            skipped_sim += 1
            continue

        s_vals = np.fromiter(sorted(sim_counts.keys()), dtype=int)
        s_probs = np.fromiter((sim_counts[v] / s_total for v in s_vals), dtype=float)

        dist = wasserstein_distance(r_vals, s_vals, u_weights=r_probs, v_weights=s_probs)
        total_w1 += float(dist)
        total_singleton_abs_error += abs(
            _pmf_probability_at(r_vals, r_probs, 1) - _pmf_probability_at(s_vals, s_probs, 1)
        )
        total_short_cdf_abs_error += abs(
            _pmf_cdf_at(r_vals, r_probs, short_cdf_length)
            - _pmf_cdf_at(s_vals, s_probs, short_cdf_length)
        )
        total_long_tail_abs_error += abs(
            _pmf_tail_at(r_vals, r_probs, long_tail_length)
            - _pmf_tail_at(s_vals, s_probs, long_tail_length)
        )
        compared += 1

    avg_w1 = (total_w1 / compared) if compared > 0 else float("nan")
    avg_singleton_abs_error = (
        total_singleton_abs_error / compared if compared > 0 else float("nan")
    )
    avg_short_cdf_abs_error = (
        total_short_cdf_abs_error / compared if compared > 0 else float("nan")
    )
    avg_long_tail_abs_error = (
        total_long_tail_abs_error / compared if compared > 0 else float("nan")
    )
    composite_score = (
        weights["w1"] * avg_w1
        + weights["singleton"] * avg_singleton_abs_error
        + weights["short_cdf"] * avg_short_cdf_abs_error
        + weights["long_tail"] * avg_long_tail_abs_error
    )

    return {
        "sum_w1": total_w1,
        "avg_w1": avg_w1,
        "n_pairs": compared,
        "skipped_real": skipped_real,
        "skipped_sim": skipped_sim,
        "avg_singleton_abs_error": avg_singleton_abs_error,
        "avg_short_cdf_abs_error": avg_short_cdf_abs_error,
        "avg_long_tail_abs_error": avg_long_tail_abs_error,
        "composite_score": composite_score,
        "short_cdf_length": short_cdf_length,
        "long_tail_length": long_tail_length,
        "composite_w_w1": weights["w1"],
        "composite_w_singleton": weights["singleton"],
        "composite_w_short_cdf": weights["short_cdf"],
        "composite_w_long_tail": weights["long_tail"],
    }


def build_real_pmfs(tree_path, cc_path, synteny_finder=findSyntenyReal2, genomes_numeric=None):
    """
    Compute real SBL pmfs for each cognate pair of leaves in the tree.
    Returns (real_pmfs, tree, genomes_numeric, median_root_to_leaf).
    """
    tree = Phylo.read(tree_path, "newick")
    # Load/convert once
    if genomes_numeric is None:
        genomes_str = load_real_genomes_from_cc(cc_path)
        genomes_numeric = {gid: convert_to_numeric(v) for gid, v in genomes_str.items()}

    # Only leaves that actually exist in CC
    leaf_names = [lf.name for lf in tree.get_terminals() if lf.name in genomes_numeric]

    real_pmfs = {}
    for a, b in combinations(leaf_names, 2):
        g1 = genomes_numeric[a]
        g2 = genomes_numeric[b]
        blocks = synteny_finder(g1.copy(), g2.copy())
        lens = [int(abs(bk[4])) for bk in blocks if len(bk) >= 5]
        pair = (a, b) if a <= b else (b, a)
        real_pmfs[pair] = lengths_to_pmf(lens)

    med_path = float(np.median([tree.distance(tree.root, lf) for lf in tree.get_terminals()]))
    return real_pmfs, tree, genomes_numeric, med_path



def make_root_genome(root_mode, tree, cc_path, real_genomes=None):
    """
    root_mode: "median_synthetic" or "from_info_tab"
    Returns list[int] root genome (integer labels, 1..L0 or from real genome IDs).
    """
    if real_genomes is None:
        real_genomes = load_real_genomes_from_cc(cc_path)

    if root_mode == "from_info_tab":
        info_tab = os.path.join(os.path.dirname(cc_path), "atgc.info.tab")
        if not os.path.exists(info_tab):
            raise FileNotFoundError(f"Root mode 'from_info_tab' but {info_tab} not found.")
        with open(info_tab, "r") as f:
            first_line = f.readline().strip()
            gid = first_line.split()[0]
        root_seq = convert_to_numeric(real_genomes[gid])
        return root_seq

    # default: median_synthetic
    med_len = int(np.median([len(convert_to_numeric(v)) for v in real_genomes.values()]))
    return list(range(1, med_len + 1))

def simulate_and_score(
    atgc_dir: str,
    tree_path: str,
    cc_path: str,
    root_mode: str,
    per_gene_gain: float,
    per_gene_loss: float,
    per_gene_inv: float,
    per_gene_trans: float,
    exp_gain: float,
    exp_loss: float,
    exp_inv: float,
    exp_trans: float,
    n_runs: int,
    seed: int | None = None,
    core_fraction: float = 0.5,
    core_protection: float = 0.9,
    synteny_finder=findSyntenyReal2,
):
    """
    Runs n_runs simulations on the given tree/root genome, pools simulated SBL counts per pair,
    builds pmfs once, computes Wasserstein-1 vs real pmfs for each pair, and sums across pairs.
    """
    # --- Real side: cache to avoid rebuilding for every grid point ---
    cache_key = (tree_path, cc_path, getattr(synteny_finder, "__name__", "finder"))
    try:
        _RPMFS_CACHE  # type: ignore
    except NameError:
        globals()["_RPMFS_CACHE"] = {}

    if cache_key in _RPMFS_CACHE:
        real_pmfs, tree, real_genomes, med_path = _RPMFS_CACHE[cache_key]
    else:
        res = build_real_pmfs(tree_path, cc_path, synteny_finder=synteny_finder)
        # Support both old (2-return) and new (4-return) versions
        if isinstance(res, tuple) and len(res) == 4:
            real_pmfs, tree, genomes_numeric, med_path = res
            real_genomes = genomes_numeric  # already numeric
        else:
            real_pmfs, tree = res
            real_genomes = load_real_genomes_from_cc(cc_path)
            med_path = median_root_to_leaf(tree)
        _RPMFS_CACHE[cache_key] = (real_pmfs, tree, real_genomes, med_path)

    # Root genome (uses cached real_genomes)
    root_genome = make_root_genome(root_mode, tree, cc_path, real_genomes=real_genomes)

    # Pool simulated counts across runs
    pooled_sim_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)

    # --- Per-run deterministic seeding (if seed provided) ---
    child_rng = np.random.default_rng(seed) if seed is not None else None
    sim_sig = inspect.signature(run_simulation)

    # Run simulations
    for _ in range(n_runs):
        # derive a child seed for this run
        child_seed = int(child_rng.integers(0, 2**32 - 1)) if child_rng is not None else None

        # pass seed/rng if run_simulation supports it; otherwise reseed globals
        extra = {}
        if child_seed is not None:
            if 'seed' in sim_sig.parameters:
                extra['seed'] = child_seed
            elif 'rng' in sim_sig.parameters:
                extra['rng'] = np.random.default_rng(child_seed)
            else:
                # fallback: reseed global RNGs to keep determinism
                random.seed(child_seed)
                np.random.seed(child_seed)

        sim_pairs = run_simulation(
            tree,
            root_genome,
            per_gene_gain_rate=per_gene_gain,
            per_gene_loss_rate=per_gene_loss,
            per_gene_inv_rate=per_gene_inv,
            per_gene_trans_rate=per_gene_trans,
            gain_exp=exp_gain,
            loss_exp=exp_loss,
            inv_exp=exp_inv,
            trans_exp=exp_trans,
            core_fraction=core_fraction,
            core_protection=core_protection,
            **extra,
        )

        # Pool counts; standardize pair key ordering without tuple(sorted(...)) overhead
        for (a, b), lens in sim_pairs.items():
            pair = (a, b) if a <= b else (b, a)
            if lens:
                pooled_sim_counts[pair].update(int(x) for x in lens)

    scores = score_real_vs_sim_counts(real_pmfs, pooled_sim_counts)
    dataset = os.path.basename(os.path.normpath(atgc_dir))

    return {
        **scores,
        "dataset": dataset,
        "median_root_to_leaf": med_path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "params": {
            "root_mode": root_mode,
            "per_gene_gain": per_gene_gain,
            "per_gene_loss": per_gene_loss,
            "per_gene_inv": per_gene_inv,
            "per_gene_trans": per_gene_trans,
            "exp_gain": exp_gain,
            "exp_loss": exp_loss,
            "exp_inv": exp_inv,
            "exp_trans": exp_trans,
            "n_runs": n_runs,
            "seed": seed,
            "core_fraction": core_fraction,
            "core_protection": core_protection,
            "tree_path": tree_path,
            "cc_path": cc_path,
        },
    }



# ---------------------------
# 2) 21×21 grid for the 2‑parameter (rf, rt) case
# ---------------------------
def run_grid_2d(
    atgc_dir: str,
    tree_filename: str = "atgc.iq.r.tre",
    cc_filename: str = "atgc.cc.csv",
    root_mode: str = "median_synthetic",
    rf_min: float = 1e-2,
    rf_max: float = 1e2,
    rt_min: float = 1e-3,
    rt_max: float = 1e1,
    points: int = 21,           # 21 points -> ~5 per decade over 4 decades
    n_runs: int = 4,            
    seed: int | None = 42,
    out_csv: str | None = None,
    huge_exp: float = 1e9,      
    core_fraction: float = 0.5,
    core_protection: float = 0.9,
    quiet: bool = False,
    synteny_finder=findSyntenyReal2,
):
    """
    Balanced flux: gain=loss=rf; inversion=0; translocation=rt; exponents huge (k=1).
    Writes one CSV row per point with results + params. Returns path to CSV.
    """
    atgc_dir = os.path.abspath(atgc_dir)
    tree_path = os.path.join(atgc_dir, tree_filename)
    cc_path = os.path.join(atgc_dir, cc_filename)
    if out_csv is None:
        out_csv = os.path.join(atgc_dir, f"{os.path.basename(atgc_dir)}_grid2d_results.csv")

    rf_vals = np.logspace(math.log10(rf_min), math.log10(rf_max), points)
    rt_vals = np.logspace(math.log10(rt_min), math.log10(rt_max), points)

    # Prepare CSV
    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow([
                "dataset", "root_mode", "rf", "rt", "n_runs",
                "core_fraction", "core_protection",
                "sum_w1", "avg_w1", "n_pairs", "skipped_real", "skipped_sim",
                "composite_score", "avg_singleton_abs_error",
                "avg_short_cdf_abs_error", "avg_long_tail_abs_error",
                "short_cdf_length", "long_tail_length",
                "composite_w_w1", "composite_w_singleton",
                "composite_w_short_cdf", "composite_w_long_tail",
                "median_root_to_leaf", "seed", "tree_path", "cc_path", "timestamp"
            ])

        total = len(rf_vals) * len(rt_vals)
        done = 0
        t0 = time.time()

        for rf in rf_vals:
            for rt in rt_vals:
                res = simulate_and_score(
                    atgc_dir=atgc_dir,
                    tree_path=tree_path,
                    cc_path=cc_path,
                    root_mode=root_mode,
                    per_gene_gain=float(rf),
                    per_gene_loss=float(rf),
                    per_gene_inv=0.0,
                    per_gene_trans=float(rt),
                    exp_gain=huge_exp,
                    exp_loss=huge_exp,
                    exp_inv=huge_exp,
                    exp_trans=huge_exp,
                    n_runs=n_runs,
                    seed=seed,
                    core_fraction=core_fraction,
                    core_protection=core_protection,
                    synteny_finder=synteny_finder,
                )

                w.writerow([
                    res["dataset"], root_mode, f"{rf:.8g}", f"{rt:.8g}", n_runs,
                    f"{core_fraction:.8g}", f"{core_protection:.8g}",
                    f"{res['sum_w1']:.10g}", f"{res['avg_w1']:.10g}", res["n_pairs"],
                    res["skipped_real"], res["skipped_sim"],
                    f"{res['composite_score']:.10g}",
                    f"{res['avg_singleton_abs_error']:.10g}",
                    f"{res['avg_short_cdf_abs_error']:.10g}",
                    f"{res['avg_long_tail_abs_error']:.10g}",
                    res["short_cdf_length"], res["long_tail_length"],
                    f"{res['composite_w_w1']:.6g}",
                    f"{res['composite_w_singleton']:.6g}",
                    f"{res['composite_w_short_cdf']:.6g}",
                    f"{res['composite_w_long_tail']:.6g}",
                    f"{res['median_root_to_leaf']:.6g}", seed, tree_path, cc_path, res["timestamp"]
                ])
                fh.flush()

                done += 1
                if not quiet:
                    elapsed = time.time() - t0
                    print(f"[grid] rf={rf:.3g}, rt={rt:.3g}  -> sumW1={res['sum_w1']:.3f} "
                          f"composite={res['composite_score']:.3f} "
                          f"({done}/{total}, {elapsed/60:.1f} min elapsed)")

    if not quiet:
        print(f"[grid] wrote results to: {out_csv}")
    return out_csv


# ---------------------------
# Optional CLI entry points
# ---------------------------
if __name__ == "__main__":
    # Example single-point call:
    # r = simulate_and_score(
    #     atgc_dir="ATGC0070",
    #     tree_path="ATGC0070/atgc.iq.r.tre",
    #     cc_path="ATGC0070/atgc.cc.csv",
    #     root_mode="median_synthetic",
    #     per_gene_gain=0.1, per_gene_loss=0.1, per_gene_inv=0.0, per_gene_trans=0.1,
    #     exp_gain=1e9, exp_loss=1e9, exp_inv=1e9, exp_trans=1e9,
    #     n_runs=4, seed=42,
    # )
    # print(json.dumps(r, indent=2))

    # Example 21×21 grid:
    run_grid_2d(atgc_dir="ATGC0070", n_runs=100, seed=None, quiet=False)
