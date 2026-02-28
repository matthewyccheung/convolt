# Datasets and layouts

The unified CLI supports:

## Intra-patient (paired scans)

- `nlst` (CT): global + radial-shell regional volume change.
- `lungct` (ThoraxCBCT): global + radial-shell regional volume change.
- `acdc` (cardiac MRI): ED/ES volume change + derived metrics.

These datasets are expected to be organized nnUNet-style (e.g. `imagesTr/`, `masksTr/`), with a `*_dataset.json` describing pairs where applicable.

## Inter-patient Learn2Reg (atlas-based segmentation)

- `oasis` (MR T1w → MR T1w; labels missing for the official test set)

These datasets are also expected nnUNet-style with `imagesTr/`, `labelsTr/`, `imagesTs/` and an accompanying `*_dataset.json`.

### Path configuration

Set `CONVOLT_DATA_ROOT` to the directory that contains the dataset folders:
```bash
export CONVOLT_DATA_ROOT=/path/to/Registration
```

Defaults (if unset) match the author’s environment under `/scratch/yc130/Registration`.
