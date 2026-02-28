from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple


SplitName = Literal["training", "testing"]


@dataclass(frozen=True)
class ACDCPair:
    patient_id: str
    split: SplitName
    ed_frame: int
    es_frame: int
    ed_image: Path
    es_image: Path
    ed_seg: Path
    es_seg: Path
    info_cfg: Path


def _parse_info_cfg(path: Path) -> tuple[int, int]:
    ed = None
    es = None
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().upper()
        v = v.strip()
        if k == "ED":
            ed = int(v)
        elif k == "ES":
            es = int(v)
    if ed is None or es is None:
        raise ValueError(f"Could not parse ED/ES from {path}")
    return ed, es


def _resolve_single_nifti(path: Path) -> Path:
    """
    Kaggle-preprocessed ACDC can store each image under a folder named *.nii
    containing the actual NIfTI file.
    """
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    nii = sorted(list(path.glob("*.nii")) + list(path.glob("*.nii.gz")))
    if len(nii) != 1:
        raise ValueError(f"Expected exactly 1 NIfTI under {path}, found {len(nii)}")
    return nii[0]


def load_acdc_pairs(dataset_root: str | Path, split: SplitName) -> List[ACDCPair]:
    dataset_root = Path(dataset_root)
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    pairs: List[ACDCPair] = []
    for patient_dir in sorted(split_dir.glob("patient*")):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        info = patient_dir / "Info.cfg"
        if not info.exists():
            continue
        ed, es = _parse_info_cfg(info)

        ed_tag = f"{ed:02d}"
        es_tag = f"{es:02d}"

        ed_img = _resolve_single_nifti(patient_dir / f"{pid}_frame{ed_tag}.nii")
        es_img = _resolve_single_nifti(patient_dir / f"{pid}_frame{es_tag}.nii")
        ed_gt = _resolve_single_nifti(patient_dir / f"{pid}_frame{ed_tag}_gt.nii")
        es_gt = _resolve_single_nifti(patient_dir / f"{pid}_frame{es_tag}_gt.nii")

        pairs.append(
            ACDCPair(
                patient_id=pid,
                split=split,
                ed_frame=int(ed),
                es_frame=int(es),
                ed_image=ed_img,
                es_image=es_img,
                ed_seg=ed_gt,
                es_seg=es_gt,
                info_cfg=info,
            )
        )
    if not pairs:
        raise ValueError(f"No ACDC pairs found under {split_dir}")
    return pairs

