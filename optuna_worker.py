#!/usr/bin/env python3
"""Run Optuna trials as one Biowulf swarm worker.

Multiple copies of this script can run at the same time. They share the same
Optuna storage database and collaboratively optimize rf/rt.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Install it with `pip install optuna` "
        "or load an environment that contains optuna."
    ) from exc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from prod_1b import findSyntenyReal2, simulate_and_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna trials for rf/rt search.")
    parser.add_argument("--study-name", default="atgc0070_rf_rt")
    parser.add_argument("--storage", default="sqlite:///hpc_optuna/optuna_atgc0070.db")
    parser.add_argument("--atgc-dir", default="ATGC0070")
    parser.add_argument("--tree-filename", default="atgc.iq.r.tre")
    parser.add_argument("--cc-filename", default="atgc.cc.csv")
    parser.add_argument("--root-mode", default="median_synthetic")
    parser.add_argument("--rf-min", type=float, default=1e-2)
    parser.add_argument("--rf-max", type=float, default=1e2)
    parser.add_argument("--rt-min", type=float, default=1e-3)
    parser.add_argument("--rt-max", type=float, default=1e1)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--score-column", default="avg_w1", choices=["avg_w1", "sum_w1"])
    parser.add_argument("--huge-exp", type=float, default=1e9)
    parser.add_argument("--out-csv", default=None)
    return parser.parse_args()


def append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new_file = not os.path.exists(path)
    fieldnames = [
        "worker_id", "trial", "dataset", "root_mode", "rf", "rt", "n_runs",
        "score_column", "score", "sum_w1", "avg_w1", "n_pairs",
        "skipped_real", "skipped_sim", "median_root_to_leaf", "seed",
        "tree_path", "cc_path", "timestamp",
    ]
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def main() -> None:
    args = parse_args()
    atgc_dir = os.path.abspath(args.atgc_dir)
    tree_path = os.path.join(atgc_dir, args.tree_filename)
    cc_path = os.path.join(atgc_dir, args.cc_filename)
    out_csv = args.out_csv
    if out_csv is None:
        out_csv = f"hpc_optuna/results/{args.worker_id}.csv"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        log_rf = trial.suggest_float("log10_rf", math.log10(args.rf_min), math.log10(args.rf_max))
        log_rt = trial.suggest_float("log10_rt", math.log10(args.rt_min), math.log10(args.rt_max))
        rf = 10 ** log_rf
        rt = 10 ** log_rt
        trial_seed = args.seed + 100000 * trial.number + int(args.worker_id.split("_")[-1] or 0)

        elapsed = time.time() - t0
        print(
            f"[optuna:{args.worker_id}] starting trial={trial.number} "
            f"rf={rf:.4g}, rt={rt:.4g} ({elapsed/60:.1f} min elapsed)",
            flush=True,
        )

        res = simulate_and_score(
            atgc_dir=atgc_dir,
            tree_path=tree_path,
            cc_path=cc_path,
            root_mode=args.root_mode,
            per_gene_gain=float(rf),
            per_gene_loss=float(rf),
            per_gene_inv=0.0,
            per_gene_trans=float(rt),
            exp_gain=args.huge_exp,
            exp_loss=args.huge_exp,
            exp_inv=args.huge_exp,
            exp_trans=args.huge_exp,
            n_runs=args.n_runs,
            seed=trial_seed,
            synteny_finder=findSyntenyReal2,
        )
        score = float(res[args.score_column])
        if not math.isfinite(score):
            score = float("inf")

        trial.set_user_attr("rf", rf)
        trial.set_user_attr("rt", rt)
        trial.set_user_attr("sum_w1", float(res["sum_w1"]))
        trial.set_user_attr("avg_w1", float(res["avg_w1"]))
        trial.set_user_attr("n_runs", args.n_runs)
        trial.set_user_attr("worker_id", args.worker_id)

        append_row(out_csv, {
            "worker_id": args.worker_id,
            "trial": trial.number,
            "dataset": res["dataset"],
            "root_mode": args.root_mode,
            "rf": f"{rf:.10g}",
            "rt": f"{rt:.10g}",
            "n_runs": args.n_runs,
            "score_column": args.score_column,
            "score": f"{score:.12g}",
            "sum_w1": f"{res['sum_w1']:.12g}",
            "avg_w1": f"{res['avg_w1']:.12g}",
            "n_pairs": res["n_pairs"],
            "skipped_real": res["skipped_real"],
            "skipped_sim": res["skipped_sim"],
            "median_root_to_leaf": f"{res['median_root_to_leaf']:.8g}",
            "seed": trial_seed,
            "tree_path": tree_path,
            "cc_path": cc_path,
            "timestamp": res["timestamp"],
        })

        elapsed = time.time() - t0
        print(
            f"[optuna:{args.worker_id}] finished trial={trial.number} "
            f"rf={rf:.4g}, rt={rt:.4g} {args.score_column}={score:.4g} "
            f"({elapsed/60:.1f} min elapsed)",
            flush=True,
        )
        return score

    study.optimize(objective, n_trials=args.n_trials)
    print(f"[optuna:{args.worker_id}] done; best value={study.best_value:.6g}")


if __name__ == "__main__":
    main()
