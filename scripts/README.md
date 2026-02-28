# Scripts

This folder contains convenience scripts used to reproduce benchmark results and paper figures.

## Registration sweeps

- `scripts/run_registration_all.sh`: runs registration for all datasets (intra-patient + Learn2Reg inter-patient).
  - Supports `ONLY_DATASET=...`, `ONLY_METHOD=...`, `DEVICE=...`, and Learn2Reg atlas/VoxelMorph settings.

Example:
```bash
CUDA_VISIBLE_DEVICES=0 sh scripts/run_registration_all.sh
ONLY_DATASET=oasis ONLY_METHOD=demons sh scripts/run_registration_all.sh
```

## UQ sweeps

- `scripts/run_uq_all_outputs.sh`: runs UQ for every registration output folder under `${OUTPUTS_ROOT}` that contains `summary.csv`.
  - Defaults: `OUTPUTS_ROOT=${CONVOLT_RESULTS_ROOT:-/scratch/yc130/Registration/outputs}`, `UQ_ROOT=${CONVOLT_UQ_ROOT:-uq_results}`.

Example:
```bash
ALPHA=0.1 N_REPEATS=100 sh scripts/run_uq_all_outputs.sh
ONLY_DATASET=oasis ONLY_METHOD=voxelmorph sh scripts/run_uq_all_outputs.sh
```

- `scripts/run_uq_one_output.sh`: runs UQ for a single output folder (useful for debugging).

## Tables and figures

- `scripts/combine_uq_tables.py`: merges per-run `cp_summary.csv` / `region_cp_summary.csv` into paper-ready tables under `uq_results/_tables/`.
- `scripts/make_paper_figures.py`: makes all paper plots from `uq_results/_tables/` and cached feature data.
  - Use `--no_default_suite` to run only selected figures (faster).

## Qualitative figures

- `scripts/make_registration_sample_sets.py`: builds image grids (fixed, atlas/moving, warped, deformation magnitude, label overlays) for a sample of cases.
- `scripts/make_beta_case_visualizations.py`: visualizes “large vs small” learned multiplicative correction cases (volume scaling proxy via Jacobian scaling).

