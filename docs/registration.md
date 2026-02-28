# Registration

The main entrypoint is:
```bash
python -m reg register ...
```

If installed via `pip install -e .`, you can also use:
```bash
convolt-register register ...
```

## Common flags

- `--dataset {nlst,lungct,acdc,oasis}`
- `--method {demons,voxelmorph,sitk_diffeomorphic_demons,sitk_bspline,itk_elastix_bspline}`
- `--dataset_dir PATH` override input dataset location
- `--results_dir PATH` override output folder
- `--split ...` dataset-specific split selector

## Learn2Reg atlas-based segmentation

For inter-patient Learn2Reg tasks, the atlas is treated as the **moving** image and the target subject is the **fixed** image.
Each atlas is registered **moving → fixed** and the atlas label map is warped into fixed space. Multi-atlas fusion then produces the predicted labels.

Atlas configuration:
- `--atlas_mode {multi,single,average}` (default `multi`)
- `--atlas_n N` (default 5)
- `--atlas_seed SEED`

## Outputs

Each `results_dir` contains:

- `summary.csv`: per-case metrics and the baseline volume prediction \( \widehat{Y}^0 \).
- `pairs/<case_id>/artifacts.npz`: deformation features (displacement/Jacobian) used by UQ.
- `figures/*.png`: quick visual checks.

Learn2Reg training (labeled) runs also include:
- `label_volumes.csv`: per-label volumes for `--uq_target volume_label`.
