#!/usr/bin/env python3
"""Latin hypercube rf/rt search for the genome evolution model.

This is a standalone alternative to the full grid search in prod_1b.py.
It samples rf and rt in log-space with Latin hypercube sampling, giving broad
coverage of the parameter space with fewer evaluations than a dense grid.
"""

import csv
import math
import os
import random
import time

from prod_1b import findSyntenyReal2, simulate_and_score


def write_header_if_needed(out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    if os.path.exists(out_csv):
        return
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "sample", "dataset", "root_mode", "rf", "rt", "n_runs",
            "sum_w1", "avg_w1", "n_pairs", "skipped_real", "skipped_sim",
            "median_root_to_leaf", "seed", "tree_path", "cc_path", "timestamp",
        ])


def latin_hypercube_log_samples(
    n_samples: int,
    rf_min: float,
    rf_max: float,
    rt_min: float,
    rt_max: float,
    seed: int | None,
) -> list[tuple[float, float]]:
    """Return Latin hypercube samples for rf/rt in log10 space."""
    rng = random.Random(seed)
    rf_bins = list(range(n_samples))
    rt_bins = list(range(n_samples))
    rng.shuffle(rf_bins)
    rng.shuffle(rt_bins)

    rf_lo = math.log10(rf_min)
    rf_hi = math.log10(rf_max)
    rt_lo = math.log10(rt_min)
    rt_hi = math.log10(rt_max)

    samples = []
    for rf_bin, rt_bin in zip(rf_bins, rt_bins):
        rf_u = (rf_bin + rng.random()) / n_samples
        rt_u = (rt_bin + rng.random()) / n_samples
        rf = 10 ** (rf_lo + rf_u * (rf_hi - rf_lo))
        rt = 10 ** (rt_lo + rt_u * (rt_hi - rt_lo))
        samples.append((rf, rt))
    return samples


def run_latin_hypercube_2d(
    atgc_dir: str,
    tree_filename: str = "atgc.iq.r.tre",
    cc_filename: str = "atgc.cc.csv",
    root_mode: str = "median_synthetic",
    rf_min: float = 1e-2,
    rf_max: float = 1e2,
    rt_min: float = 1e-3,
    rt_max: float = 1e1,
    n_samples: int = 100,
    n_runs: int = 100,
    seed: int | None = 42,
    out_csv: str | None = None,
    huge_exp: float = 1e9,
    quiet: bool = False,
) -> dict:
    atgc_dir = os.path.abspath(atgc_dir)
    tree_path = os.path.join(atgc_dir, tree_filename)
    cc_path = os.path.join(atgc_dir, cc_filename)
    if out_csv is None:
        out_csv = os.path.join(
            atgc_dir,
            f"{os.path.basename(atgc_dir)}_latin_hypercube_results.csv",
        )

    write_header_if_needed(out_csv)
    samples = latin_hypercube_log_samples(
        n_samples=n_samples,
        rf_min=rf_min,
        rf_max=rf_max,
        rt_min=rt_min,
        rt_max=rt_max,
        seed=seed,
    )

    best = {"avg_w1": float("inf"), "sum_w1": float("inf"), "rf": None, "rt": None}
    t0 = time.time()

    for sample_idx, (rf, rt) in enumerate(samples, start=1):
        sample_seed = None if seed is None else seed + sample_idx

        if not quiet:
            elapsed = time.time() - t0
            print(
                f"[latin] starting sample={sample_idx}/{n_samples} "
                f"rf={rf:.4g}, rt={rt:.4g} ({elapsed/60:.1f} min elapsed)",
                flush=True,
            )

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
            seed=sample_seed,
            synteny_finder=findSyntenyReal2,
        )

        with open(out_csv, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                sample_idx, res["dataset"], root_mode,
                f"{rf:.10g}", f"{rt:.10g}", n_runs,
                f"{res['sum_w1']:.12g}", f"{res['avg_w1']:.12g}", res["n_pairs"],
                res["skipped_real"], res["skipped_sim"],
                f"{res['median_root_to_leaf']:.8g}", sample_seed,
                tree_path, cc_path, res["timestamp"],
            ])
            fh.flush()

        score = float(res["avg_w1"])
        if score < best["avg_w1"]:
            best = {
                "avg_w1": score,
                "sum_w1": float(res["sum_w1"]),
                "rf": rf,
                "rt": rt,
            }

        if not quiet:
            elapsed = time.time() - t0
            print(
                f"[latin] sample={sample_idx}/{n_samples} "
                f"rf={rf:.4g}, rt={rt:.4g} -> avgW1={res['avg_w1']:.4g}; "
                f"best avgW1={best['avg_w1']:.4g} at "
                f"rf={best['rf']:.4g}, rt={best['rt']:.4g} "
                f"({elapsed/60:.1f} min elapsed)",
                flush=True,
            )

    if not quiet:
        print()
        print(
            f"[done] best rf={best['rf']:.8g}, rt={best['rt']:.8g}, "
            f"sumW1={best['sum_w1']:.10g}, avgW1={best['avg_w1']:.10g}"
        )
        print(f"[done] wrote results to: {out_csv}")

    return {
        "best_rf": best["rf"],
        "best_rt": best["rt"],
        "best_sum_w1": best["sum_w1"],
        "best_avg_w1": best["avg_w1"],
        "out_csv": out_csv,
        "n_evals": n_samples,
    }


if __name__ == "__main__":
    run_latin_hypercube_2d(
        atgc_dir="ATGC0070",
        n_samples=100,
        n_runs=100,
        seed=42,
        quiet=False,
    )
