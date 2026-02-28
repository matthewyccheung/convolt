# Migration notes (old → new names)

This repo was originally organized under a folder named `compass_reg/` with a Python package named `nlstreg`.

It has been renamed for publication:

- Repo folder (you choose on disk): `convolt/`
- Python package: `reg/`
- PyPI/project name (in `pyproject.toml`): `convolt`

## Command mapping

Old:
```bash
python -m nlstreg register ...
python -m nlstreg.uq.cli ...
```

New:
```bash
python -m reg register ...
python -m reg.uq.cli ...
```

If installed with `pip install -e .`, you can also use:
```bash
convolt-register register ...
convolt-uq ...
```

## Path configuration

To avoid hard-coded paths, set:
```bash
export CONVOLT_DATA_ROOT=/path/to/Registration
export CONVOLT_RESULTS_ROOT=/path/to/Registration/outputs
export CONVOLT_UQ_ROOT=/path/to/convolt/uq_results
```

