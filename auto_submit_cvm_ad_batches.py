#!/usr/bin/env python3
"""Submit the rf=0.1 CvM/Anderson-Darling swarm grid in batches."""
from __future__ import annotations

import argparse
import getpass
import json
import subprocess
import time
from pathlib import Path


PARTS_DIR = Path("hpc_multicpu/joint_size_cvm_rf_0p1_parts")
RESULTS_DIR = Path(
    "hpc_multicpu/results_joint_size_cvm_rf_0p1_observed_medoid_empirical_core"
)
STATE_FILE = Path("hpc_multicpu/cvm_ad_auto_submit_state.json")
JOB_NAME = "joint_cvm_ad"
TOTAL_PARTS = 52
TOTAL_RESULTS = 51_750


def completed_results() -> int:
    if not RESULTS_DIR.is_dir():
        return 0
    return sum(
        1
        for path in RESULTS_DIR.glob("result_*.csv")
        if path.is_file() and path.stat().st_size > 0
    )


def active_jobs() -> int:
    result = subprocess.run(
        [
            "squeue", "-u", getpass.getuser(), "-h", "-n", JOB_NAME,
            "-o", "%A",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return len(
        {line.strip() for line in result.stdout.splitlines() if line.strip()}
    )


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


def validate_parts(start_part: int) -> None:
    missing = [
        str(PARTS_DIR / f"part_{part:03d}.swarm")
        for part in range(start_part, TOTAL_PARTS + 1)
        if not (PARTS_DIR / f"part_{part:03d}.swarm").is_file()
    ]
    if missing:
        preview = "\n".join(missing[:5])
        suffix = "\n..." if len(missing) > 5 else ""
        raise FileNotFoundError(f"Missing swarm files:\n{preview}{suffix}")


def submit_part(part_number: int, memory_gb: int, hours: int) -> str:
    swarm_file = PARTS_DIR / f"part_{part_number:03d}.swarm"
    command = [
        "swarm", "-f", str(swarm_file), "-t", "16", "-g", str(memory_gb),
        "--time", f"{hours}:00:00", "--job-name", JOB_NAME,
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
        state["submitted_parts"].append(
            {"part": part, "swarm_output": output}
        )
        state["next_part"] = part + 1
        save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-part", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--poll-minutes", type=float, default=10)
    parser.add_argument("--memory-gb", type=int, default=8)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and submit one batch if no matching jobs are active.",
    )
    args = parser.parse_args()
    if not 1 <= args.start_part <= TOTAL_PARTS:
        parser.error(f"--start-part must be between 1 and {TOTAL_PARTS}")
    if args.batch_size < 1 or args.poll_minutes <= 0:
        parser.error("batch size and poll interval must be positive")
    if args.memory_gb < 1 or args.hours < 1:
        parser.error("memory and time must be positive")

    state = load_state(args.start_part)
    next_part = int(state["next_part"])
    if next_part <= TOTAL_PARTS:
        validate_parts(next_part)
    print(
        f"Starting CvM/AD auto-submitter at part {next_part:03d}; "
        f"state file: {STATE_FILE}",
        flush=True,
    )

    while True:
        completed = completed_results()
        active = active_jobs()
        next_part = int(state["next_part"])
        print(
            f"Completed {completed}/{TOTAL_RESULTS}; active {JOB_NAME} "
            f"allocations: {active}; next part: {next_part:03d}",
            flush=True,
        )
        if next_part > TOTAL_PARTS:
            if active == 0:
                print("All swarm parts finished or left the queue.", flush=True)
                return
            print("All swarm parts submitted; waiting for active jobs.", flush=True)
        elif active == 0:
            submit_batch(state, args.batch_size, args.memory_gb, args.hours)
        if args.once:
            return
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    main()
