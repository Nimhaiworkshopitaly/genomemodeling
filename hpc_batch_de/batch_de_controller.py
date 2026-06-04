#!/usr/bin/env python3
"""Batch differential evolution controller for Biowulf swarm.

This script parallelizes the differential-evolution search by evaluating one
generation of candidates at a time. Each generation is written as a swarm file;
after the swarm finishes, this script reads the result CSVs and updates the DE
population.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
from datetime import datetime


DEFAULT_STATE = "hpc_batch_de/state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch DE controller for rf/rt search.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="Create initial DE population state.")
    add_common(init)
    init.add_argument("--population-size", type=int, default=24)
    init.add_argument("--seed", type=int, default=42)
    init.add_argument("--state", default=DEFAULT_STATE)

    make = sub.add_parser("make-swarm", help="Write swarm file for current batch.")
    make.add_argument("--state", default=DEFAULT_STATE)
    make.add_argument("--python", default="python")
    make.add_argument("--swarm-file", default=None)

    update = sub.add_parser("update", help="Read current batch results and update population.")
    update.add_argument("--state", default=DEFAULT_STATE)
    update.add_argument("--allow-partial", action="store_true")

    best = sub.add_parser("best", help="Print best current population members.")
    best.add_argument("--state", default=DEFAULT_STATE)
    best.add_argument("--top", type=int, default=10)

    export = sub.add_parser("export", help="Export population history to CSV.")
    export.add_argument("--state", default=DEFAULT_STATE)
    export.add_argument("--out-csv", default="hpc_batch_de/batch_de_history.csv")

    return parser.parse_args()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--atgc-dir", default="ATGC0070")
    parser.add_argument("--rf-min", type=float, default=1e-2)
    parser.add_argument("--rf-max", type=float, default=1e2)
    parser.add_argument("--rt-min", type=float, default=1e-3)
    parser.add_argument("--rt-max", type=float, default=1e1)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--mutation", type=float, default=0.8, help="DE mutation factor F.")
    parser.add_argument("--crossover", type=float, default=0.7, help="DE crossover rate CR.")
    parser.add_argument("--score-column", default="avg_w1", choices=["avg_w1", "sum_w1"])
    parser.add_argument("--results-dir", default="hpc_batch_de/results")
    parser.add_argument("--swarm-dir", default="hpc_batch_de/swarms")


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(math.log10(lo), math.log10(hi))


def to_real(log_rf: float, log_rt: float) -> tuple[float, float]:
    return 10 ** log_rf, 10 ** log_rt


def save_state(state: dict) -> None:
    path = state["state_path"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)


def load_state(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def init_state(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    bounds = {
        "log_rf": [math.log10(args.rf_min), math.log10(args.rf_max)],
        "log_rt": [math.log10(args.rt_min), math.log10(args.rt_max)],
    }

    population = []
    for idx in range(args.population_size):
        log_rf = log_uniform(rng, args.rf_min, args.rf_max)
        log_rt = log_uniform(rng, args.rt_min, args.rt_max)
        rf, rt = to_real(log_rf, log_rt)
        population.append({
            "idx": idx,
            "log_rf": log_rf,
            "log_rt": log_rt,
            "rf": rf,
            "rt": rt,
            "score": None,
            "sum_w1": None,
            "avg_w1": None,
        })

    state = {
        "state_path": args.state,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "generation": 0,
        "phase": "initial",
        "seed": args.seed,
        "rng_state": rng.getstate()[1],
        "config": {
            "atgc_dir": args.atgc_dir,
            "n_runs": args.n_runs,
            "mutation": args.mutation,
            "crossover": args.crossover,
            "score_column": args.score_column,
            "results_dir": args.results_dir,
            "swarm_dir": args.swarm_dir,
        },
        "bounds": bounds,
        "population": population,
        "current_batch": make_initial_batch(population),
        "history": [],
    }
    save_state(state)
    print(f"initialized {args.population_size} candidates in {args.state}")
    print("next: python hpc_batch_de/batch_de_controller.py make-swarm")


def make_initial_batch(population: list[dict]) -> list[dict]:
    return [{
        "kind": "initial",
        "target_idx": member["idx"],
        "log_rf": member["log_rf"],
        "log_rt": member["log_rt"],
        "rf": member["rf"],
        "rt": member["rt"],
    } for member in population]


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def make_swarm(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    cfg = state["config"]
    generation = state["generation"]
    batch = state["current_batch"]
    if not batch:
        raise SystemExit("No current batch. Run update first or reinitialize.")

    swarm_file = args.swarm_file
    if swarm_file is None:
        swarm_file = os.path.join(cfg["swarm_dir"], f"generation_{generation:03d}.swarm")
    os.makedirs(os.path.dirname(os.path.abspath(swarm_file)), exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    lines = []
    for item in batch:
        out_csv = result_path(cfg["results_dir"], generation, item["target_idx"])
        seed = state["seed"] + 100000 * generation + 1000 * int(item["target_idx"])
        cmd = [
            args.python,
            "hpc/eval_one_setting.py",
            "--atgc-dir", cfg["atgc_dir"],
            "--rf", f"{item['rf']:.12g}",
            "--rt", f"{item['rt']:.12g}",
            "--n-runs", str(cfg["n_runs"]),
            "--seed", str(seed),
            "--out-csv", out_csv,
        ]
        lines.append(shell_join(cmd))

    with open(swarm_file, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    state["current_swarm_file"] = swarm_file
    save_state(state)
    print(f"wrote {len(lines)} jobs to {swarm_file}")
    print(f"results directory: {cfg['results_dir']}")


def result_path(results_dir: str, generation: int, target_idx: int) -> str:
    return os.path.join(results_dir, f"gen_{generation:03d}_target_{target_idx:04d}.csv")


def update_state(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    cfg = state["config"]
    generation = state["generation"]
    batch = state["current_batch"]
    if not batch:
        raise SystemExit("No current batch to update.")

    result_rows = {}
    missing = []
    for item in batch:
        path = result_path(cfg["results_dir"], generation, item["target_idx"])
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            missing.append(path)
            continue
        result_rows[item["target_idx"]] = rows[0]

    if missing and not args.allow_partial:
        print(f"{len(missing)} result files are missing. First few:")
        for path in missing[:10]:
            print(path)
        raise SystemExit("Wait for the swarm to finish, or rerun with --allow-partial.")

    accepted = 0
    evaluated = 0
    population = state["population"]
    for item in batch:
        row = result_rows.get(item["target_idx"])
        if row is None:
            continue
        evaluated += 1
        score = float(row[cfg["score_column"]])
        target = population[int(item["target_idx"])]
        old_score = target["score"]

        should_accept = old_score is None or score <= float(old_score)
        history_row = {
            "generation": generation,
            "kind": item["kind"],
            "target_idx": item["target_idx"],
            "rf": item["rf"],
            "rt": item["rt"],
            "score": score,
            "sum_w1": float(row["sum_w1"]),
            "avg_w1": float(row["avg_w1"]),
            "accepted": should_accept,
        }
        state["history"].append(history_row)

        if should_accept:
            accepted += 1
            target.update({
                "log_rf": item["log_rf"],
                "log_rt": item["log_rt"],
                "rf": item["rf"],
                "rt": item["rt"],
                "score": score,
                "sum_w1": float(row["sum_w1"]),
                "avg_w1": float(row["avg_w1"]),
            })

    if state["phase"] == "initial":
        state["phase"] = "evolving"

    state["generation"] = generation + 1
    state["current_batch"] = make_trial_batch(state)
    save_state(state)
    print(f"updated generation {generation}: evaluated={evaluated}, accepted={accepted}")
    print_best_members(state, top=5)
    print("next: make-swarm, submit swarm, then update")


def make_trial_batch(state: dict) -> list[dict]:
    rng = random.Random()
    rng.setstate((3, tuple(state["rng_state"]), None))

    pop = state["population"]
    n = len(pop)
    f = state["config"]["mutation"]
    cr = state["config"]["crossover"]
    rf_lo, rf_hi = state["bounds"]["log_rf"]
    rt_lo, rt_hi = state["bounds"]["log_rt"]
    bounds = [(rf_lo, rf_hi), (rt_lo, rt_hi)]

    batch = []
    for idx, target in enumerate(pop):
        choices = [i for i in range(n) if i != idx]
        a_i, b_i, c_i = rng.sample(choices, 3)
        a, b, c = pop[a_i], pop[b_i], pop[c_i]
        mutant = [
            a["log_rf"] + f * (b["log_rf"] - c["log_rf"]),
            a["log_rt"] + f * (b["log_rt"] - c["log_rt"]),
        ]
        mutant = [
            min(max(mutant[d], bounds[d][0]), bounds[d][1])
            for d in range(2)
        ]
        target_vec = [target["log_rf"], target["log_rt"]]
        forced_dim = rng.randrange(2)
        trial = []
        for d in range(2):
            if d == forced_dim or rng.random() < cr:
                trial.append(mutant[d])
            else:
                trial.append(target_vec[d])

        rf, rt = to_real(trial[0], trial[1])
        batch.append({
            "kind": "trial",
            "target_idx": idx,
            "log_rf": trial[0],
            "log_rt": trial[1],
            "rf": rf,
            "rt": rt,
        })

    state["rng_state"] = rng.getstate()[1]
    return batch


def print_best(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    print_best_members(state, args.top)


def print_best_members(state: dict, top: int) -> None:
    members = [m for m in state["population"] if m["score"] is not None]
    members = sorted(members, key=lambda m: float(m["score"]))
    print("Best population members:")
    for rank, member in enumerate(members[:top], start=1):
        print(
            f"{rank:2d}. idx={member['idx']} rf={member['rf']:.8g} "
            f"rt={member['rt']:.8g} score={member['score']:.8g} "
            f"avg_w1={member['avg_w1']:.8g} sum_w1={member['sum_w1']:.8g}"
        )


def export_history(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fieldnames = [
        "generation", "kind", "target_idx", "rf", "rt", "score",
        "sum_w1", "avg_w1", "accepted",
    ]
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(state["history"])
    print(f"wrote {len(state['history'])} rows to {args.out_csv}")


def main() -> None:
    args = parse_args()
    if args.cmd == "init":
        init_state(args)
    elif args.cmd == "make-swarm":
        make_swarm(args)
    elif args.cmd == "update":
        update_state(args)
    elif args.cmd == "best":
        print_best(args)
    elif args.cmd == "export":
        export_history(args)


if __name__ == "__main__":
    main()
