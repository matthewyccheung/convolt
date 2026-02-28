# Installation

## 1) Python environment

This project is pure Python. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the package in editable mode so the `convolt-*` commands are available:
```bash
pip install -e .
```

## 2) Configure data/output paths (recommended)

Defaults fall back to the original author’s environment under `/scratch/yc130/...`. For portability, set:
```bash
export CONVOLT_DATA_ROOT=/path/to/Registration
export CONVOLT_RESULTS_ROOT=/path/to/Registration/outputs
export CONVOLT_UQ_ROOT=/path/to/convolt/uq_results
```

## 3) Optional registration backends

The core code supports multiple backends. Some are optional:

- `demons`: implemented in this repo (SciPy/NumPy). No extra install.
- `voxelmorph`: implemented in this repo (PyTorch required).
  - Install `torch` with CUDA if you want GPU acceleration.
- `sitk_diffeomorphic_demons`, `sitk_bspline`: require `SimpleITK`.
- `itk_elastix_bspline`: requires `itk-elastix` (Python bindings for elastix).

The provided `requirements.txt` includes these as optional dependencies; you can comment them out if you only need the in-repo backends.
