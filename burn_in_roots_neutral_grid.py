#!/usr/bin/env python3
"""Neutral rf x rt landscapes across independently burned-in synthetic roots.

The burn-in parameterization is fixed before the downstream rf x rt grid is
evaluated. This makes the initial-condition sensitivity explicit and avoids
using the fitted grid point itself to manufacture its own root.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import pangenome_sampled_roots_neutral_grid as workflow
from simulation_core_composite import evolve_genome_branch


N_ROOTS = 5
ROOT_SEEDS = (1801, 2801, 3801, 4801, 5801)
DEFAULT_BURN_IN_CYCLES = 25
DEFAULT_BURN_IN_RF = 0.1
DEFAULT_BURN_IN_RT = 0.1
DEFAULT_BURN_IN_RI = 0.0

ROOT = Path("hpc_multicpu/burn_in_roots_neutral")
ROOT_GENOMES = ROOT / "root_genomes"
ROOTS_CSV = ROOT / "burn_in_roots.csv"
RESULTS = ROOT / "results"
PARTS = ROOT / "parts"
FIGURES = ROOT / "figures"


def configure_workflow_paths():
    """Point the shared neutral-grid workflow at this experiment's files."""
    workflow.ROOT = ROOT
    workflow.ROOT_GENOMES = ROOT_GENOMES
    workflow.ROOTS_CSV = ROOTS_CSV
    workflow.RESULTS = RESULTS
    workflow.PARTS = PARTS
    workflow.FIGURES = FIGURES
    workflow.ROOT_MODE = "neutral_burn_in"


def burn_in_root(initial_length, seed, cycles, rf, rt, ri):
    """Evolve an arbitrary genome for repeated neutral burn-in cycles."""
    random.seed(seed)
    np.random.seed(seed)
    genome = list(range(1, initial_length + 1))
    next_gene_id = [initial_length + 1]
    lengths = [len(genome)]

    for cycle in range(cycles):
        if len(genome) < 2:
            raise RuntimeError(
                f"Burn-in root collapsed below two genes after cycle {cycle}"
            )
        current_length = len(genome)
        genome = evolve_genome_branch(
            genome=genome,
            branch_length=1.0,
            gain_rate=current_length * rf,
            loss_rate=current_length * rf,
            inv_rate=current_length * ri,
            trans_rate=current_length * rt,
            gain_exp=1e9,
            loss_exp=1e9,
            inv_exp=1e9,
            trans_exp=1e9,
            next_gene_id_holder=next_gene_id,
            L0_ancestral=current_length,
            core_gene_ids=set(),
            core_protection=0.0,
        )
        # The simulator marks deleted genes as -1 during a branch. Removing
        # those markers between cycles makes each cycle start from the extant
        # genome rather than from deletion placeholders.
        genome = [gene for gene in genome if gene != -1]
        lengths.append(len(genome))

    if len(genome) != len(set(genome)):
        raise RuntimeError("Burn-in produced duplicate extant gene identifiers")
    return genome, lengths, next_gene_id[0]


def generate_roots(n_roots, cycles, rf, rt, ri):
    if n_roots < 2:
        raise ValueError("At least two roots are required for a root-sensitivity panel")
    if cycles < 1:
        raise ValueError("Burn-in cycles must be positive")
    if min(rf, rt, ri) < 0:
        raise ValueError("Burn-in rates cannot be negative")

    observed = workflow.observed_genomes()
    observed_lengths = np.asarray([len(genome) for genome in observed.values()])
    initial_length = int(round(float(np.median(observed_lengths))))
    ROOT.mkdir(parents=True, exist_ok=True)
    ROOT_GENOMES.mkdir(parents=True, exist_ok=True)
    rows = []

    for index in range(n_roots):
        seed = ROOT_SEEDS[index] if index < len(ROOT_SEEDS) else ROOT_SEEDS[-1] + index
        genome, lengths, next_gene_id = burn_in_root(
            initial_length, seed, cycles, rf, rt, ri
        )
        synthetic_id = f"burn_in_root_{index + 1:02d}"
        root_file = ROOT_GENOMES / f"{synthetic_id}.json"
        root_file.write_text(json.dumps(genome) + "\n")
        rows.append({
            "root_index": index + 1,
            "synthetic_id": synthetic_id,
            "root_seed": seed,
            "initial_genome_length": initial_length,
            "genome_length": len(genome),
            "unique_genes": len(set(genome)),
            "burn_in_cycles": cycles,
            "burn_in_rf": rf,
            "burn_in_rt": rt,
            "burn_in_ri": ri,
            "minimum_cycle_length": min(lengths),
            "maximum_cycle_length": max(lengths),
            "next_gene_id_start": next_gene_id,
            "root_file": str(root_file),
            "observed_genomes_used_for_length_only": len(observed),
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOTS_CSV, index=False)
    return frame


def main():
    configure_workflow_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("generate-roots", "make-jobs", "status", "analyze")
    )
    parser.add_argument("--n-roots", type=int, default=N_ROOTS)
    parser.add_argument("--burn-in-cycles", type=int, default=DEFAULT_BURN_IN_CYCLES)
    parser.add_argument("--burn-in-rf", type=float, default=DEFAULT_BURN_IN_RF)
    parser.add_argument("--burn-in-rt", type=float, default=DEFAULT_BURN_IN_RT)
    parser.add_argument("--burn-in-ri", type=float, default=DEFAULT_BURN_IN_RI)
    parser.add_argument("--jobs-per-part", type=int, default=500)
    args = parser.parse_args()

    if args.action == "generate-roots":
        frame = generate_roots(
            args.n_roots,
            args.burn_in_cycles,
            args.burn_in_rf,
            args.burn_in_rt,
            args.burn_in_ri,
        )
        print(frame.to_string(index=False))
    elif args.action == "make-jobs":
        workflow.make_jobs(args.jobs_per_part)
    elif args.action == "status":
        workflow.status()
    else:
        workflow.analyze(
            landscape_description="neutral burn-in starting genomes",
            filename_suffix="burn_in_roots_neutral",
            root_title="Burn-in root",
        )


if __name__ == "__main__":
    main()
