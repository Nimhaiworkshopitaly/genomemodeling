#!/usr/bin/env python3
"""Optuna rf/rt search for the genome evolution model.

This is a standalone alternative to the full grid search in prod_1b.py.
It uses Optuna's TPE sampler to choose promising rf/rt values in log-space.

Install Optuna before running:
    pip install optuna
"""

import csv
import math
import os
import time

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Install it with `pip install optuna` "
        "or load an environment that contains optuna."
    ) from exc

from prod_1b import findSyntenyReal2, simulate_and_score


def write_header_if_needed(out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    if os.path.exists(out_csv):
        return
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "trial", "dataset", "root_mode", "rf", "rt", "n_runs",
            "sum_w1", "avg_w1", "n_pairs", "skipped_real", "skipped_sim",
            "median_root_to_leaf", "seed", "tree_path", "cc_path", "timestamp",
        ])


def run_optuna_2d(
    atgc_dir: str,
    tree_filename: str = "atgc.iq.r.tre",
    cc_filename: str = "atgc.cc.csv",
    root_mode: str = "median_synthetic",
    rf_min: float = 1e-2,
    rf_max: float = 1e2,
    rt_min: float = 1e-3,
    rt_max: float = 1e1,
    n_trials: int = 50,
    n_runs: int = 100,
    seed: int | None = 42,
    out_csv: str | None = None,
    huge_exp: float = 1e9,
    score_column: str = "avg_w1",
    quiet: bool = False,
):
    """Run Optuna search over rf and rt.

    rf and rt are sampled in log10 space because the useful values span several
    orders of magnitude.
    """
    if score_column not in {"avg_w1", "sum_w1"}:
        raise ValueError("score_column must be 'avg_w1' or 'sum_w1'")

    atgc_dir = os.path.abspath(atgc_dir)
    tree_path = os.path.join(atgc_dir, tree_filename)
    cc_path = os.path.join(atgc_dir, cc_filename)
    if out_csv is None:
        out_csv = os.path.join(
            atgc_dir,
            f"{os.path.basename(atgc_dir)}_optuna_results.csv",
        )

    write_header_if_needed(out_csv)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        log_rf = trial.suggest_float("log10_rf", math.log10(rf_min), math.log10(rf_max))
        log_rt = trial.suggest_float("log10_rt", math.log10(rt_min), math.log10(rt_max))
        rf = 10 ** log_rf
        rt = 10 ** log_rt
        trial_seed = None if seed is None else seed + trial.number + 1

        if not quiet:
            elapsed = time.time() - t0
            print(
                f"[optuna] starting trial={trial.number + 1}/{n_trials} "
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
            seed=trial_seed,
            synteny_finder=findSyntenyReal2,
        )

        score = float(res[score_column])
        if not math.isfinite(score):
            score = float("inf")

        with open(out_csv, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                trial.number + 1, res["dataset"], root_mode,
                f"{rf:.10g}", f"{rt:.10g}", n_runs,
                f"{res['sum_w1']:.12g}", f"{res['avg_w1']:.12g}", res["n_pairs"],
                res["skipped_real"], res["skipped_sim"],
                f"{res['median_root_to_leaf']:.8g}", trial_seed,
                tree_path, cc_path, res["timestamp"],
            ])
            fh.flush()

        trial.set_user_attr("rf", rf)
        trial.set_user_attr("rt", rt)
        trial.set_user_attr("sum_w1", float(res["sum_w1"]))
        trial.set_user_attr("avg_w1", float(res["avg_w1"]))

        if not quiet:
            elapsed = time.time() - t0
            print(
                f"[optuna] trial={trial.number + 1}/{n_trials} "
                f"rf={rf:.4g}, rt={rt:.4g} -> {score_column}={score:.4g} "
                f"({elapsed/60:.1f} min elapsed)",
                flush=True,
            )

        return score

    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_rf = best_trial.user_attrs["rf"]
    best_rt = best_trial.user_attrs["rt"]
    best_sum_w1 = best_trial.user_attrs["sum_w1"]
    best_avg_w1 = best_trial.user_attrs["avg_w1"]

    if not quiet:
        print()
        print(
            f"[done] best trial={best_trial.number + 1}, "
            f"rf={best_rf:.8g}, rt={best_rt:.8g}, "
            f"sumW1={best_sum_w1:.10g}, avgW1={best_avg_w1:.10g}"
        )
        print(f"[done] wrote results to: {out_csv}")

    return {
        "best_rf": best_rf,
        "best_rt": best_rt,
        "best_sum_w1": best_sum_w1,
        "best_avg_w1": best_avg_w1,
        "best_trial": best_trial.number + 1,
        "out_csv": out_csv,
        "study": study,
    }


if __name__ == "__main__":
    run_optuna_2d(
        atgc_dir="ATGC0070",
        n_trials=50,
        n_runs=100,
        seed=42,
        quiet=False,
    )
