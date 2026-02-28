from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal


SplitName = Literal["training", "val", "test", "all"]


@dataclass(frozen=True)
class Pair:
    patient_id: str
    fixed_image: Path
    moving_image: Path
    fixed_mask: Path
    moving_mask: Path


def _patient_id_from_path(p: Path) -> str:
    # NLST_0001_0000.nii.gz -> NLST_0001
    name = p.name
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    parts = name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return name


def _infer_dataset_json_path(dataset_dir: Path) -> Path:
    """
    Support NLST-style and other nnUNet-like datasets by inferring the dataset json.
    Preference order:
      1) NLST_dataset.json (backwards compatible)
      2) the single *_dataset.json file if exactly one exists
    """
    nlst = dataset_dir / "NLST_dataset.json"
    if nlst.exists():
        return nlst
    cands = sorted(dataset_dir.glob("*_dataset.json"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise FileNotFoundError(f"No dataset json found in {dataset_dir} (expected NLST_dataset.json or *_dataset.json)")
    raise FileNotFoundError(f"Multiple *_dataset.json found in {dataset_dir}; please keep only one. Found: {cands}")


def _derive_mask_from_image_path(img_path: Path) -> Path | None:
    """
    Try to derive mask path from an image path using common nnUNet-like folder names.
    """
    parts = list(img_path.parts)
    for i, part in enumerate(parts):
        if part == "imagesTr":
            parts[i] = "masksTr"
            cand = Path(*parts)
            return cand if cand.exists() else None
        if part == "imagesTs":
            # Some datasets (e.g. NLST) may not provide a separate masksTs folder and
            # instead keep all masks under masksTr even for imagesTs.
            parts_ts = parts.copy()
            parts_ts[i] = "masksTs"
            cand_ts = Path(*parts_ts)
            if cand_ts.exists():
                return cand_ts
            parts_tr = parts.copy()
            parts_tr[i] = "masksTr"
            cand_tr = Path(*parts_tr)
            return cand_tr if cand_tr.exists() else None
    return None


def load_pairs(dataset_dir: str | Path, split: SplitName) -> List[Pair]:
    dataset_dir = Path(dataset_dir)
    dataset_json = _infer_dataset_json_path(dataset_dir)
    js = json.load(open(dataset_json, "r"))

    # Prefer the explicit paired list when available.
    keys = {
        "training": ("training_paired_images",),
        "val": ("registration_val",),
        "test": ("registration_test",),
        "all": ("training_paired_images", "registration_val", "registration_test"),
    }[split]
    paired_all = []
    for k in keys:
        v = js.get(k)
        if v:
            paired_all.extend(list(v))
    if not paired_all:
        raise ValueError(f"No paired list found for split='{split}' in {dataset_json.name} (looked for keys={keys})")

    # De-duplicate pairs (fixed,moving).
    seen = set()
    paired = []
    for entry in paired_all:
        fixed = str(entry["fixed"])
        moving = str(entry["moving"])
        key = (fixed, moving)
        if key in seen:
            continue
        seen.add(key)
        paired.append(entry)

    # Build a quick lookup from image->mask using any lists that explicitly provide it.
    image_to_mask = {}
    for list_key in ("training", "test"):
        for item in js.get(list_key, []):
            if not isinstance(item, dict):
                continue
            if "image" not in item or "mask" not in item:
                continue
            img = (dataset_dir / str(item["image"])).resolve()
            msk = (dataset_dir / str(item["mask"])).resolve()
            # Some datasets have stale/incorrect entries; only record if it exists.
            if msk.exists():
                image_to_mask[str(img)] = msk

    out: List[Pair] = []
    skipped_missing_files = 0
    skipped_missing_masks = 0
    for entry in paired:
        fixed = (dataset_dir / entry["fixed"]).resolve()
        moving = (dataset_dir / entry["moving"]).resolve()
        if not fixed.exists() or not moving.exists():
            skipped_missing_files += 1
            continue
        fixed_mask = image_to_mask.get(str(fixed))
        if fixed_mask is None:
            fixed_mask = _derive_mask_from_image_path(fixed)
        moving_mask = image_to_mask.get(str(moving))
        if moving_mask is None:
            moving_mask = _derive_mask_from_image_path(moving)
        if fixed_mask is None or moving_mask is None or (not fixed_mask.exists()) or (not moving_mask.exists()):
            skipped_missing_masks += 1
            continue
        patient_id = _patient_id_from_path(fixed)
        out.append(
            Pair(
                patient_id=patient_id,
                fixed_image=fixed,
                moving_image=moving,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
            )
        )
    if len(out) == 0:
        raise ValueError(
            f"No usable pairs found for split='{split}' in {dataset_json.name}. "
            f"Skipped missing_files={skipped_missing_files}, missing_masks={skipped_missing_masks}."
        )
    return out
