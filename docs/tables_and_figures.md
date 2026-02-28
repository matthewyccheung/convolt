# Tables and figures

## 1) Run UQ

Tables/figures are generated from UQ outputs under `${CONVOLT_UQ_ROOT:-uq_results}`.
Run UQ first:
```bash
sh scripts/run_uq_all_outputs.sh
```

## 2) Combine tables

```bash
python scripts/combine_uq_tables.py \
  --uq_root "${CONVOLT_UQ_ROOT:-uq_results}" \
  --out_dir uq_results/_tables \
  --datasets lungct,nlst,oasis
```

This produces:
- `uq_results/_tables/main_results_table_main.csv`
- `uq_results/_tables/region_results_table_main.csv`
- `uq_results/_tables/main_results_table_ablations.csv`
- and additional claim/diagnostic tables.

## 3) Make figures

```bash
python scripts/make_paper_figures.py --datasets lungct,nlst,oasis --backends demons,voxelmorph
```

Some figure routines cache feature matrices under:
`uq_results/_figures_paper/_cache/`.

## 4) Registration sample grids (qualitative figures)

```bash
python scripts/make_registration_sample_sets.py -h
```

## 5) Coefficient-stability figure

```bash
python scripts/make_paper_figures.py --no_default_suite --coef_stability --coef_stability_topk 10
```

