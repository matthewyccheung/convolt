from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """
    Resolve the repository/project root (parent of the reg package directory).
    This makes default output paths stable regardless of the current working directory.
    """
    return Path(__file__).resolve().parent.parent


def _env_path(name: str, default: str) -> Path:
    v = os.environ.get(str(name), "").strip()
    return Path(v) if v else Path(default)


def default_results_dir(dataset: str, method: str) -> Path:
    dataset = str(dataset).lower()
    method = str(method).lower()
    root = _env_path("CONVOLT_RESULTS_ROOT", "/scratch/yc130/Registration/outputs")
    return root / f"{dataset}_{method}"


def default_results_dir_tagged(dataset: str, method: str, tag: str) -> Path:
    tag = str(tag).strip()
    if not tag:
        return default_results_dir(dataset, method)
    dataset = str(dataset).lower()
    method = str(method).lower()
    root = _env_path("CONVOLT_RESULTS_ROOT", "/scratch/yc130/Registration/outputs")
    return root / f"{dataset}_{method}_{tag}"


def default_uq_dir(dataset: str, method: str) -> Path:
    dataset = str(dataset).lower()
    method = str(method).lower()
    uq_root = _env_path("CONVOLT_UQ_ROOT", str(project_root() / "uq_results"))
    return uq_root / f"{dataset}_{method}"


def default_uq_dir_tagged(dataset: str, method: str, tag: str) -> Path:
    tag = str(tag).strip()
    if not tag:
        return default_uq_dir(dataset, method)
    dataset = str(dataset).lower()
    method = str(method).lower()
    uq_root = _env_path("CONVOLT_UQ_ROOT", str(project_root() / "uq_results"))
    return uq_root / f"{dataset}_{method}_{tag}"


def default_dataset_dir(dataset: str) -> Path:
    dataset = str(dataset).lower()
    # Set CONVOLT_DATA_ROOT to point at the directory that contains the dataset folders
    # (e.g., ".../Registration" that contains NLST/, LungCT/, OASIS/, ...).
    data_root = _env_path("CONVOLT_DATA_ROOT", "/scratch/yc130/Registration")
    if dataset == "nlst":
        return data_root / "NLST"
    if dataset == "lungct":
        return data_root / "LungCT"
    if dataset == "acdc":
        return data_root / "ACDC" / "database"
    if dataset == "hippocampusmr":
        return data_root / "HippocampusMR"
    if dataset == "oasis":
        return data_root / "OASIS"
    if dataset == "abdomenctct":
        return data_root / "AbdomenCTCT"
    raise ValueError("dataset must be one of: nlst, lungct, acdc, hippocampusmr, oasis, abdomenctct")
