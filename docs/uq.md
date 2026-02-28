# Uncertainty quantification (UQ)

Run UQ after registration (requires `summary.csv` and `pairs/*/artifacts.npz`):
```bash
python -m reg.uq.cli --dataset nlst --method demons --alpha 0.1 --n_repeats 100
```

## Targets

- Intra-patient: default target is `delta_volume` (volume change).
- Learn2Reg inter-patient: default “volume suite” is union volume plus a subset of labels.

Explicit target selection:
- `--uq_target delta_volume`
- `--uq_target volume_union`
- `--uq_target volume_label` (Learn2Reg only; requires `label_volumes.csv`)

Selecting which labels to run:
- `--uq_topk_labels K` (top-K by mean GT volume)
- `--uq_label_list all` (all labels present in `label_volumes.csv`, excluding background)
- `--uq_label_list 1,2,6` (explicit list)

## Baselines and ConVOLT

The per-target UQ report includes:
- `SCP(|err|)`: split conformal in output space.
- `LCP`: localized (kNN) split conformal in output space.
- `CQR`: conformalized quantile regression (Romano et al., 2019).
- `ConVOLT(scale-CP)`: ridge regression to predict a multiplicative scale \(k(x)\) from deformation-derived features, with split conformal in k-space.

Regional guarantees for lungs:
- Enable with `--region_defs radial --radial_shells 5`
- Control ConVOLT regional training with `--region_beta_mode {region,global}`

