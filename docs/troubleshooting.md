# Troubleshooting

## “Missing summary.csv”

UQ expects registration outputs to exist. Re-run registration for that dataset/method (and the correct atlas tag if applicable).

## Corrupted `pairs/*/artifacts.npz`

If UQ fails with `zipfile.BadZipFile`, one or more artifact files were partially written (common causes: out-of-disk, interrupted job, NFS hiccup).
Delete the affected case folder under `results_dir/pairs/<case_id>/` and rerun registration for that case.

## Elastix “Internal elastix error”

Enable elastix logging to file/console in `reg/external/itk_elastix.py` and inspect the elastix log for the failing pair.
Common causes are bad initialization, non-overlapping images, or invalid spacing/origin metadata.

## SimpleITK spacing / overlap errors

These typically indicate inconsistent NIfTI metadata (zero spacing) or insufficient overlap without initialization.
Workarounds include aligning centers before registration or verifying spacing/origin values in the source data.

