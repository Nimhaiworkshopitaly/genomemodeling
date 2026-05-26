# Biowulf Parallel Workflow

This folder contains helper scripts for running the genome-evolution parameter
search on NIH Biowulf with `swarm`.

## Files

- `eval_one_setting.py`: evaluates one `(rf, rt)` parameter setting and writes one CSV.
- `make_swarm.py`: creates a `jobs.swarm` file with one command per setting.
- `aggregate_results.py`: combines per-job CSV files and ranks settings by score.

## 1. Prepare an environment

Load or create a Python environment with:

```bash
numpy
pandas
scipy
biopython
matplotlib
```

Example with a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy biopython matplotlib
```

On Biowulf you can instead use site modules or a conda environment if your
group already has one.

## 2. Generate the swarm file

From the repository root:

```bash
python hpc/make_swarm.py \
  --atgc-dir ATGC0070 \
  --points 21 \
  --n-runs 20 \
  --seed 42 \
  --swarm-file hpc/jobs.swarm \
  --out-dir hpc/results
```

This creates `21 x 21 = 441` jobs. Each job evaluates one pair of rates:

- `rf`: gain/loss rate
- `rt`: translocation rate

Each job writes its own result CSV under `hpc/results/`.

## 3. Submit to Biowulf

Example:

```bash
swarm -f hpc/jobs.swarm -g 8 -t 1 --time=02:00:00 --module python
```

Tune `-g` and `--time` after a few small test jobs. If you use a virtualenv or
conda environment, activate it before running `swarm`, or use the appropriate
Biowulf module/environment flags for your setup.

## 4. Aggregate results

After the swarm completes:

```bash
python hpc/aggregate_results.py 'hpc/results/*.csv' \
  --out-csv hpc/combined_results.csv \
  --sort-by avg_w1
```

The best parameter settings are the rows with the lowest `avg_w1` or `sum_w1`.

## Recommended search strategy

Start cheap:

```bash
python hpc/make_swarm.py --atgc-dir ATGC0070 --points 7 --n-runs 5
```

Then look at the best region and rerun a denser search around it with larger
`n_runs`, for example 20, 50, or 100.
