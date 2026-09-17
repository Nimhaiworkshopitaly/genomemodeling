#!/usr/bin/env python3
"""Submit joint-size swarm files in checkpointed batches on Biowulf."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


PARTS_DIR = Path("hpc_multicpu/joint_size_full_factorial_parts")
RESULTS_DIR = Path(
    "hpc_multicpu/results_joint_size_full_factorial_observed_medoid_empirical_core"
)
STATE_FILE = Path("hpc_multicpu/joint_size_auto_submit_state.json")
TOTAL_PARTS = 156
TOTAL_RESULTS = 155_250


def completed_results() -> int:
    if not RESULTS_DIR.is_dir():
        return 0
    return sum(
        1 for path in RESULTS_DIR.glob("result_*.csv")
        if path.is_file() and path.stat().st_size > 0
    )


def active_joint_jobs() -> int:
    result = subprocess.run(
        ["squeue", "-u", subprocess.check_output(["whoami"], text=True).strip(),
         "-h", "-n", "joint_sizes", "-o", "%A"],
        check=True, capture_output=True, text=True,
    )
    return len({line.strip() for line in result.stdout.splitlines() if line.strip()})


def load_state(start_part: int) -> dict:
    if STATE_FILE.is_file():
        state = json.loads(STATE_FILE.read_text())
        if "next_part" not in state:
            raise ValueError(f"Invalid state file: {STATE_FILE}")
        return state
    return {"next_part": start_part, "submitted_parts": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE_FILE)


def submit_part(part_number: int, memory_gb: int, hours: int) -> str:
    swarm_file = PARTS_DIR / f"part_{part_number:03d}.swarm"
    if not swarm_file.is_file():
        raise FileNotFoundError(f"Missing swarm file: {swarm_file}")
    command = [
        "swarm", "-f", str(swarm_file), "-t", "16", "-g", str(memory_gb),
        "--time", f"{hours}:00:00", "--job-name", "joint_sizes",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    print(f"Submitted {swarm_file}: {output}", flush=True)
    return output


def submit_batch(state: dict, batch_size: int, memory_gb: int, hours: int) -> None:
    for _ in range(batch_size):
        part = int(state["next_part"])
        if part > TOTAL_PARTS:
            return
        output = submit_part(part, memory_gb, hours)
        state["submitted_parts"].append({"part": part, "swarm_output": output})
        state["next_part"] = part + 1
        save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-part", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--poll-minutes", type=float, default=10)
    parser.add_argument("--memory-gb", type=int, default=8)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument(
        "--once", action="store_true",
        help="Check once and submit one batch if no joint_sizes jobs are active.",
    )
    args = parser.parse_args()
    if not 1 <= args.start_part <= TOTAL_PARTS:
        parser.error(f"--start-part must be between 1 and {TOTAL_PARTS}")
    if args.batch_size < 1 or args.poll_minutes <= 0:
        parser.error("batch size and poll interval must be positive")

    state = load_state(args.start_part)
    print(
        f"Starting auto-submitter at part {state['next_part']:03d}; "
        f"state file: {STATE_FILE}", flush=True,
    )
    while True:
        completed = completed_results()
        active = active_joint_jobs()
        print(
            f"Completed {completed}/{TOTAL_RESULTS}; active joint_sizes "
            f"allocations: {active}; next part: {state['next_part']:03d}",
            flush=True,
        )
        if int(state["next_part"]) > TOTAL_PARTS:
            print("All swarm parts have been submitted.", flush=True)
            return
        if active == 0:
            submit_batch(state, args.batch_size, args.memory_gb, args.hours)
        if args.once:
            return
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    main()
