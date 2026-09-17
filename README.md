# Genome Modeling of Synteny-Block Distributions

This repository simulates bacterial genome evolution along a phylogenetic tree
and compares simulated synteny block length distributions with observed data.
The current analyses focus on ATGC0070 and examine how gene gain/loss,
single-gene translocation, uniform-breakpoint inversion, and protection of
empirical core COGs affect model fit.

The main workflows are designed for parallel execution on the NIH Biowulf
cluster. They report a composite fit score together with per genome pair
Kolmogorov-Smirnov (KS), Kuiper, and other distributional statistics.

## Main analyses

1. **Gene gain/loss x translocation (`rf x rt`)** searches logarithmically
   spaced gene gain/loss and single-gene translocation rates.
2. **Translocation x inversion** independently varies single-gene
   translocation and uniform-breakpoint inversion rates while holding gene gain
   and loss rates fixed.

Each analysis can compare:

- **No core protection:** no event is rejected because of core-gene status.
- **Empirical core protection:** COGs meeting a prevalence threshold in the
  observed genomes are protected probabilistically.

## Repository structure

```text
ATGC0070/                        ATGC0070 input data
hpc_multicpu/                    Biowulf swarm generators and evaluators
simulation_core_composite.py     Genome-evolution simulation engine
prod_1b_core_composite.py        Data loading, root selection, and scoring
synteny_tools.py                 Synteny-block detection functions
joint_translocation_inversion_grid_observed_medoid.py
                                 Matched translocation/inversion workflow
hpc/                             Earlier single-CPU Biowulf workflow
hpc_optuna/                      Optuna parameter-search workflow
legacy code/                     Historical scripts retained for reference
```

Run commands from the repository root.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- SciPy
- Biopython
- Matplotlib

Create a local environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy biopython matplotlib
```

On Biowulf, use the Python environment appropriate for your account and
allocation.

## Input data

An ATGC directory is expected to contain at least:

```text
ATGC0070/atgc.cc.csv
ATGC0070/atgc.info.tab
ATGC0070/atgc.iq.r.tre
```

The analyses below use Yuri's gain+loss-clock tree at:

```text
ATGC0070/yuri_gl26/ATGC0070.gl.tre
```

That tree is not currently included in this repository and must be supplied
before reproducing those runs. Paths passed with `--tree-filename` are relative
to the selected ATGC directory.

## Root-genome modes

The simulation supports three root modes:

- `median_synthetic` creates an artificial genome of median observed length
  with consecutive numerical gene identifiers.
- `from_info_tab` uses the first genome listed in `atgc.info.tab`.
- `observed_medoid` selects the observed genome with the smallest average
  dissimilarity from the other tree genomes.

The observed medoid distance gives equal weight to Jaccard distance between
sets of observed COGs and Jaccard distance between orientation independent
circular gene adjacencies. Distance from the median genome length and genome ID
are deterministic tie-breakers. For the current ATGC0070 data and Yuri tree,
this procedure selected `GCF_900187235.1`.

The observed medoid retains real COG identities and an observed gene order,
making it more appropriate than a/ synthetic root for empirical-core
protection. It is a representative observed genome, not a reconstructed
ancestor.

## Core protection

### Synthetic-fraction core

```text
--core-mode synthetic_fraction
--core-fraction VALUE
--core-protection VALUE
```

This legacy mode marks a requested fraction of root genes as core. Setting
`--core-fraction 0 --core-protection 0` gives the no-core condition.

### Empirical core COGs

```text
--core-mode empirical
--core-prevalence 1.0
--core-protection 0.9
```

With prevalence `1.0`, empirical core COGs are those present in every observed
tree genome. The active protected set is restricted to empirical core COGs
present in the selected root.

Protection is probabilistic:

- a gain adjacent to a core COG is rejected with probability
  `core_protection`;
- for a loss, translocation, or inversion, rejection probability is
  `core_protection * core_fraction_in_affected_block`.

Rejected events are skipped rather than resampled. The realized event rate can
therefore be lower than the nominal rate when protection is enabled. Newly
gained genes receive identifiers greater than the largest COG identifier in
the observed data, preventing accidental classification as empirical core
COGs.

## Fit statistics

Lower values indicate a better match between observed and simulated
synteny block length distributions.

### Composite score

```text
composite =
    1 * mean Wasserstein distance
  + 100 * mean singleton absolute error
  + 100 * mean short-CDF absolute error
  + 100 * mean long-tail absolute error
```

The default short-CDF cutoff is 10 genes and the default long-tail cutoff is
50 genes.

### KS and Kuiper

The KS statistic is the largest absolute difference between the observed and
simulated CDFs for a genome pair. The Kuiper statistic is the sum of the
largest positive and negative CDF deviations. Because these statistics and the
composite score emphasize different distributional features, their best
parameter locations need not coincide.

## Observed-medoid translocation x inversion experiment

This workflow uses Yuri's tree, the same observed-medoid root in both
conditions, gain and loss rates fixed at `0.1`, five translocation rates, five
inversion rates, five matched seeds, uniform-breakpoint inversions, and 100
simulation replicates per setting.

Generate both 125-job swarm files:

```bash
python joint_translocation_inversion_grid_observed_medoid.py \
  make-jobs --condition no_core

python joint_translocation_inversion_grid_observed_medoid.py \
  make-jobs --condition empirical_core
```

Submit on Biowulf:

```bash
swarm \
  -f hpc_multicpu/jobs_joint_translocation_inversion_observed_medoid_no_core.swarm \
  -t 16 -g 8 --time 5-00:00:00 --job-name medoid_no_core

swarm \
  -f hpc_multicpu/jobs_joint_translocation_inversion_observed_medoid_empirical_core.swarm \
  -t 16 -g 8 --time 5-00:00:00 --job-name medoid_emp_core
```

Analyze the completed grids:

```bash
python joint_translocation_inversion_grid_observed_medoid.py \
  analyze --condition no_core

python joint_translocation_inversion_grid_observed_medoid.py \
  analyze --condition empirical_core
```

Outputs are written beneath:

```text
hpc_multicpu/figures_joint_translocation_inversion_observed_medoid_no_core/
hpc_multicpu/figures_joint_translocation_inversion_observed_medoid_empirical_core/
```

## Observed-medoid rf x rt experiment

Generate the no-core grid:

```bash
python hpc_multicpu/multicpu_make_swarm_core_composite.py \
  --atgc-dir ATGC0070 \
  --tree-filename yuri_gl26/ATGC0070.gl.tre \
  --root-mode observed_medoid \
  --out-dir hpc_multicpu/results_rf_rt_observed_medoid_no_core \
  --swarm-file hpc_multicpu/jobs_rf_rt_observed_medoid_no_core.swarm \
  --rf-min 0.01 --rf-max 100 --rt-min 0.001 --rt-max 10 \
  --points 9 --n-runs 100 --workers 16 \
  --core-mode synthetic_fraction \
  --core-fraction 0 --core-protection 0
```

Generate the matched empirical-core grid:

```bash
python hpc_multicpu/multicpu_make_swarm_core_composite.py \
  --atgc-dir ATGC0070 \
  --tree-filename yuri_gl26/ATGC0070.gl.tre \
  --root-mode observed_medoid \
  --out-dir hpc_multicpu/results_rf_rt_observed_medoid_empirical_core \
  --swarm-file hpc_multicpu/jobs_rf_rt_observed_medoid_empirical_core.swarm \
  --rf-min 0.01 --rf-max 100 --rt-min 0.001 --rt-max 10 \
  --points 9 --n-runs 100 --workers 16 \
  --core-mode empirical --core-prevalence 1.0 \
  --core-fraction 0 --core-protection 0.9
```

Each command creates an 81-job grid. Submit the two swarm files with matching
resources:

```bash
swarm -f hpc_multicpu/jobs_rf_rt_observed_medoid_no_core.swarm \
  -t 16 -g 8 --time 5-00:00:00 --job-name rf_rt_med_no

swarm -f hpc_multicpu/jobs_rf_rt_observed_medoid_empirical_core.swarm \
  -t 16 -g 8 --time 5-00:00:00 --job-name rf_rt_med_emp
```

## Monitoring Biowulf jobs

```bash
squeue -u "$USER"
```

Summarize jobs by name and state:

```bash
squeue -u "$USER" -h -r -o '%j %T' | sort | uniq -c
```

Count nonempty primary result files:

```bash
find RESULTS_DIRECTORY -maxdepth 1 -type f \
  -name 'result_*.csv' ! -name '*_pairs.csv' -size +0c | wc -l
```

Result-file counts are the most reliable completion check because a scheduler
job can terminate without writing its expected output.

## Output files

Single setting evaluators write one summary CSV per job. Fields include rates,
seed, tree path, root mode, selected medoid, root and empirical-core counts,
composite-score components, average KS and Kuiper statistics, and the number of
compared genome pairs. Analysis scripts produce combined CSV tables, metric
matrices, and PNG heatmaps.

Generated result and figure directories can be large and should not normally
be committed to the repository.

## Reproducibility and interpretation

- Use matched seeds, the same tree, and the same root when comparing protection
  conditions.
- Verify `tree_path`, `root_mode`, `selected_root_genome_id`, `core_mode`, and
  `core_protection` in result CSVs before combining runs.
- The observed medoid is not necessarily the ancestral genome.
- Core-event rejection changes the realized event rate, so nominally matched
  rates do not guarantee matched accepted-event counts.
- Uniform-breakpoint inversions choose two positions on a circular genome and
  reject invisible one-gene inversions.
- Objective landscape contours interpolate between evaluated grid points;
  reported optima should refer to sampled settings unless continuous
  optimization was performed.

## Legacy workflows

The `hpc/`, `hpc_optuna/`, and `legacy code/` directories contain earlier or
alternative workflows. They are useful for provenance, but the multicore
scripts and observed-medoid driver documented above are the recommended entry
points for the current analyses.
