from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional


SplitName = Literal["training", "test"]


@dataclass(frozen=True)
class Learn2RegCase:
    """
    Single-subject case in a Learn2Reg nnUNet-like task directory.

    For inter-patient tasks we treat each case as an independent subject ("patient_id").
    """

    patient_id: str
    split: SplitName
    image: Path
    label: Optional[Path] = None
    mask: Optional[Path] = None


def _patient_id_from_filename(name: str) -> str:
    # e.g. OASIS_0415_0000.nii.gz -> OASIS_0415
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    parts = name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return name


def _infer_dataset_json_path(dataset_dir: Path) -> Path:
    cands = sorted(dataset_dir.glob("*_dataset.json"))
    if len(cands) != 1:
        raise FileNotFoundError(f"Expected exactly one *_dataset.json under {dataset_dir}, found {len(cands)}")
    return cands[0]


def load_learn2reg_cases(dataset_dir: str | Path, split: SplitName) -> List[Learn2RegCase]:
    """
    Load Learn2Reg cases for an inter-patient task.

    - training: uses dataset json 'training' list (with optional label/mask fields).
    - test: ignores dataset json 'test' list (can contain .csv placeholders) and instead
      enumerates imagesTs/*.nii.gz and attaches masksTs when present.
    """
    dataset_dir = Path(dataset_dir)
    js_path = _infer_dataset_json_path(dataset_dir)
    js = json.loads(js_path.read_text())

    if split == "training":
        out: List[Learn2RegCase] = []
        for item in js.get("training", []):
            if not isinstance(item, dict) or "image" not in item:
                continue
            img = (dataset_dir / str(item["image"])).resolve()
            if not img.exists():
                continue
            pid = _patient_id_from_filename(img.name)
            lbl = None
            if "label" in item:
                p = (dataset_dir / str(item["label"])).resolve()
                if p.exists():
                    lbl = p
            msk = None
            if "mask" in item:
                p = (dataset_dir / str(item["mask"])).resolve()
                if p.exists():
                    msk = p
            out.append(Learn2RegCase(patient_id=pid, split="training", image=img, label=lbl, mask=msk))
        if not out:
            raise ValueError(f"No training cases found from {js_path}")
        return out

    # split == "test"
    images_ts = sorted((dataset_dir / "imagesTs").glob("*.nii.gz"))
    if not images_ts:
        raise FileNotFoundError(f"No images found under {dataset_dir / 'imagesTs'}")
    masks_ts_dir = dataset_dir / "masksTs"
    out = []
    for img in images_ts:
        pid = _patient_id_from_filename(img.name)
        msk = None
        if masks_ts_dir.exists():
            cand = (masks_ts_dir / img.name).resolve()
            if cand.exists():
                msk = cand
        out.append(Learn2RegCase(patient_id=pid, split="test", image=img.resolve(), label=None, mask=msk))
    return out

