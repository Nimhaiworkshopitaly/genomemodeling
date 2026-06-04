#!/usr/bin/env python3
"""Export parallel Optuna study results to CSV and print best trials."""

from __future__ import annotations

import argparse
import os

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Install it with `pip install optuna` "
        "or load an environment that contains optuna."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Optuna study results.")
    parser.add_argument("--study-name", default="atgc0070_rf_rt")
    parser.add_argument("--storage", default="sqlite:///hpc_optuna/optuna_atgc0070.db")
    parser.add_argument("--out-csv", default="hpc_optuna/combined_optuna_results.csv")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    rows = []
    for trial in study.trials:
        if trial.value is None:
            continue
        rows.append({
            "trial": trial.number,
            "value": trial.value,
            "rf": trial.user_attrs.get("rf"),
            "rt": trial.user_attrs.get("rt"),
            "sum_w1": trial.user_attrs.get("sum_w1"),
            "avg_w1": trial.user_attrs.get("avg_w1"),
            "n_runs": trial.user_attrs.get("n_runs"),
            "worker_id": trial.user_attrs.get("worker_id"),
            "state": trial.state.name,
        })
    rows.sort(key=lambda row: float(row["value"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w") as fh:
        headers = ["trial", "value", "rf", "rt", "sum_w1", "avg_w1", "n_runs", "worker_id", "state"]
        fh.write(",".join(headers) + "\n")
        for row in rows:
            fh.write(",".join("" if row[h] is None else str(row[h]) for h in headers) + "\n")

    print(f"wrote {len(rows)} completed trials to {args.out_csv}")
    print()
    print("Best trials:")
    for rank, row in enumerate(rows[:args.top], start=1):
        print(
            f"{rank:2d}. trial={row['trial']} rf={float(row['rf']):.8g} "
            f"rt={float(row['rt']):.8g} value={float(row['value']):.8g} "
            f"avg_w1={float(row['avg_w1']):.8g} sum_w1={float(row['sum_w1']):.8g}"
        )


if __name__ == "__main__":
    main()
