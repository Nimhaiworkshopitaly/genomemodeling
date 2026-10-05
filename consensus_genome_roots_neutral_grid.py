#!/usr/bin/env python3
"""Neutral rf x rt landscapes across prevalence-defined consensus genomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import pangenome_sampled_roots_neutral_grid as workflow


MINIMUM_COUNTS = (4, 5, 6, 7, 8)
ROOT = Path("hpc_multicpu/consensus_genome_roots_neutral")
ROOT_GENOMES = ROOT / "root_genomes"
ROOTS_CSV = ROOT / "consensus_roots.csv"
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
    workflow.ROOT_MODE = "consensus_genome"


def observed_statistics(genomes):
    gene_sets = {name: set(genome) for name, genome in genomes.items()}
    edge_sets = {}
    prevalence = Counter()
    edge_support = Counter()
    for name, genome in genomes.items():
        prevalence.update(gene_sets[name])
        edges = {
            tuple(sorted((genome[index], genome[(index + 1) % len(genome)])))
            for index in range(len(genome))
            if genome[index] != genome[(index + 1) % len(genome)]
        }
        edge_sets[name] = edges
        edge_support.update(edges)
    return gene_sets, edge_sets, prevalence, edge_support


def conditional_adjacency_scores(
    selected, gene_sets, edge_support, minimum_fraction
):
    scores = {}
    conserved = set()
    for edge, support in edge_support.items():
        left, right = edge
        if left not in selected or right not in selected:
            continue
        cooccurrence = sum(
            left in genes and right in genes for genes in gene_sets.values()
        )
        if not cooccurrence:
            continue
        score = support / cooccurrence
        scores[edge] = score
        if score >= minimum_fraction:
            conserved.add(edge)
    return scores, conserved


def assemble_consensus(selected, prevalence, scores, conserved):
    """Build a deterministic path while enforcing at most two neighbors per COG."""
    selected = set(selected)
    neighbors = defaultdict(list)
    all_neighbors = defaultdict(list)
    for edge, score in scores.items():
        left, right = edge
        all_neighbors[left].append((right, score))
        all_neighbors[right].append((left, score))
        if edge in conserved:
            neighbors[left].append((right, score))
            neighbors[right].append((left, score))

    first = max(
        selected,
        key=lambda gene: (
            sum(score for _, score in neighbors[gene]), prevalence[gene], -gene
        ),
    )
    path = [first]
    available = selected - {first}
    conserved_used = 0
    relaxed_used = 0
    unsupported_used = 0
    while available:
        candidates = []
        for side, endpoint in (("left", path[0]), ("right", path[-1])):
            for gene, score in neighbors[endpoint]:
                if gene in available:
                    candidates.append((score, prevalence[gene], -gene, side, gene))
        if candidates:
            _, _, _, side, gene = max(candidates)
            conserved_used += 1
        else:
            relaxed = []
            for side, endpoint in (("left", path[0]), ("right", path[-1])):
                for gene, score in all_neighbors[endpoint]:
                    if gene in available:
                        relaxed.append((score, prevalence[gene], -gene, side, gene))
            if relaxed:
                _, _, _, side, gene = max(relaxed)
                relaxed_used += 1
            else:
                gene = max(available, key=lambda item: (prevalence[item], -item))
                side = "right"
                unsupported_used += 1
        if side == "left":
            path.insert(0, gene)
        else:
            path.append(gene)
        available.remove(gene)

    closure = tuple(sorted((path[0], path[-1])))
    closure_conserved = int(closure in conserved)
    closure_relaxed = int(closure in scores and closure not in conserved)
    conserved_used += closure_conserved
    relaxed_used += closure_relaxed
    unsupported_used += int(closure not in scores)
    return (
        path,
        conserved_used,
        relaxed_used,
        unsupported_used,
        closure_conserved,
    )


def generate_roots():
    genomes = workflow.observed_genomes()
    gene_sets, _, prevalence, edge_support = observed_statistics(genomes)
    n_genomes = len(genomes)
    ROOT_GENOMES.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, minimum_count in enumerate(MINIMUM_COUNTS, start=1):
        threshold = minimum_count / n_genomes
        selected = {
            gene for gene, count in prevalence.items() if count >= minimum_count
        }
        scores, conserved = conditional_adjacency_scores(
            selected, gene_sets, edge_support, threshold
        )
        ordered, supported, relaxed, unsupported, closure = assemble_consensus(
            selected, prevalence, scores, conserved
        )
        synthetic_id = f"consensus_root_{minimum_count:02d}of{n_genomes:02d}"
        path = ROOT_GENOMES / f"{synthetic_id}.json"
        path.write_text(json.dumps(ordered) + "\n")
        rows.append({
            "root_index": index,
            "synthetic_id": synthetic_id,
            "minimum_genomes": minimum_count,
            "prevalence_threshold": threshold,
            "genome_length": len(ordered),
            "unique_cogs": len(set(ordered)),
            "conserved_adjacencies_used": supported,
            "relaxed_observed_adjacencies_used": relaxed,
            "unsupported_jumps": unsupported,
            "conserved_adjacency_fraction": supported / len(ordered),
            "observed_adjacency_fraction": (supported + relaxed) / len(ordered),
            "circular_closure_conserved": closure,
            "root_file": str(path),
            "observed_genomes_used": n_genomes,
            "pangenome_size": len(prevalence),
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
    parser.add_argument("--jobs-per-part", type=int, default=500)
    args = parser.parse_args()
    if args.action == "generate-roots":
        print(generate_roots().to_string(index=False))
    elif args.action == "make-jobs":
        workflow.make_jobs(args.jobs_per_part)
    elif args.action == "status":
        workflow.status()
    else:
        workflow.analyze(
            landscape_description="prevalence-defined consensus genomes",
            filename_suffix="consensus_genome_roots_neutral",
            root_title="Consensus root",
        )


if __name__ == "__main__":
    main()
