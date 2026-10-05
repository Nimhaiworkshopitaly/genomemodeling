#!/usr/bin/env python3
"""Neutral rate landscapes averaged across posterior ancestral genomes."""
from __future__ import annotations

import argparse
from pathlib import Path

import phylogenetic_ancestral_roots_neutral_grid as workflow


N_ROOTS = 20
ROOT = Path("hpc_multicpu/ensemble_reconstructed_ancestors_neutral")
ROOT_GENOMES = ROOT / "root_genomes"
ROOTS_CSV = ROOT / "ancestral_ensemble.csv"
MODEL_CSV = ROOT / "reconstruction_model.csv"
RESULTS = ROOT / "results"
PARTS = ROOT / "parts"
FIGURES = ROOT / "figures"


def configure_workflow_paths():
    workflow.ROOT = ROOT
    workflow.ROOT_GENOMES = ROOT_GENOMES
    workflow.ROOTS_CSV = ROOTS_CSV
    workflow.MODEL_CSV = MODEL_CSV
    workflow.RESULTS = RESULTS
    workflow.PARTS = PARTS
    workflow.FIGURES = FIGURES


def main():
    configure_workflow_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("generate-roots", "make-jobs", "status", "analyze")
    )
    parser.add_argument("--n-roots", type=int, default=N_ROOTS)
    parser.add_argument("--turnover-depth", type=float, default=0.1)
    parser.add_argument("--jobs-per-part", type=int, default=500)
    args = parser.parse_args()
    if args.n_roots < 2:
        parser.error("--n-roots must be at least 2")
    if args.turnover_depth <= 0:
        parser.error("--turnover-depth must be positive")
    if args.action == "generate-roots":
        roots, model = workflow.generate_roots(
            args.n_roots, args.turnover_depth, include_map=False
        )
        print("Reconstruction model:")
        print(model.to_string(index=False))
        print("\nPosterior ancestral ensemble:")
        print(roots.to_string(index=False))
    elif args.action == "make-jobs":
        workflow.make_jobs(args.jobs_per_part)
    elif args.action == "status":
        workflow.status()
    else:
        workflow.analyze_ensemble()


if __name__ == "__main__":
    main()
