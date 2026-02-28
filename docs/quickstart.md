# Quickstart (end-to-end)

These commands run the pipeline end-to-end:
1) registration outputs under `${CONVOLT_RESULTS_ROOT}/...` (or the default),
2) UQ outputs under `${CONVOLT_UQ_ROOT}/...` (or `uq_results/` by default),
3) combined tables under `uq_results/_tables/`,
4) figures under `uq_results/_figures_paper/`.

## 0) Configure paths (recommended)

Set these once in your shell (adjust for your machine):
```bash
export CONVOLT_DATA_ROOT=/path/to/Registration
export CONVOLT_RESULTS_ROOT=/path/to/Registration/outputs
export CONVOLT_UQ_ROOT=/path/to/convolt/uq_results
```

If you do not set them, the code defaults to the author’s paths under `/scratch/yc130/...`.

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2) Run registration (all datasets, demons + voxelmorph)

```bash
sh scripts/run_registration_all.sh
```

To run one dataset/method:
```bash
ONLY_DATASET=oasis ONLY_METHOD=demons sh scripts/run_registration_all.sh
```

## 3) Run UQ (all output folders)

```bash
sh scripts/run_uq_all_outputs.sh
```

## 4) Combine tables

```bash
python scripts/combine_uq_tables.py --uq_root "${CONVOLT_UQ_ROOT:-uq_results}" --out_dir uq_results/_tables
```

## 5) Make paper figures

```bash
python scripts/make_paper_figures.py --datasets lungct,nlst,oasis --backends demons,voxelmorph
```

To make only the coefficient-stability figure (fast on second run due to caching):
```bash
python scripts/make_paper_figures.py --no_default_suite --coef_stability --coef_stability_topk 10
```

