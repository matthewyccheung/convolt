from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .demons import resample_to_fixed_grid
from .nifti import read_nifti, save_npz, write_nifti_scalar_zyx


AtlasMode = Literal["multi", "single", "average"]


@dataclass(frozen=True)
class AtlasSpec:
    mode: AtlasMode = "multi"
    n: int = 5
    seed: int = 0
    atlas_ids: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class AtlasTemplate:
    """
    Atlas template in its own reference grid (typically the first atlas subject grid).
    """

    atlas_ids: Tuple[str, ...]
    ref_patient_id: str
    image_zyx: np.ndarray
    label_zyx: np.ndarray
    mask_zyx: np.ndarray
    spacing_zyx: Tuple[float, float, float]


def atlas_tag(spec: AtlasSpec) -> str:
    mode = str(spec.mode).lower()
    if mode == "multi":
        return f"atlas-multi{int(spec.n)}"
    if mode == "single":
        return "atlas-single"
    if mode == "average":
        return f"atlas-avg{int(spec.n)}"
    raise ValueError("atlas mode must be one of: multi, single, average")


def select_atlas_ids(
    *,
    candidate_ids: Sequence[str],
    spec: AtlasSpec,
) -> Tuple[str, ...]:
    ids = list(candidate_ids)
    if len(ids) == 0:
        raise ValueError("No atlas candidates available")

    if spec.atlas_ids is not None and len(spec.atlas_ids) > 0:
        want = list(spec.atlas_ids)
        missing = [x for x in want if x not in set(ids)]
        if missing:
            raise ValueError(f"Requested atlas_ids not found in candidates: {missing}")
        chosen = want
    else:
        rng = np.random.default_rng(int(spec.seed))
        n = int(max(1, spec.n))
        if str(spec.mode).lower() == "single":
            n = 1
        n = int(min(n, len(ids)))
        chosen = list(rng.choice(np.array(ids, dtype=object), size=n, replace=False).tolist())

    # Stable order.
    return tuple(sorted(map(str, chosen)))


def write_atlas_meta(
    *,
    out_path: str | Path,
    dataset: str,
    spec: AtlasSpec,
    atlas_ids: Sequence[str],
    vm_train_ids: Sequence[str] | None = None,
    extra: dict | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": str(dataset),
        "atlas_mode": str(spec.mode),
        "atlas_n": int(spec.n),
        "atlas_seed": int(spec.seed),
        "atlas_ids": list(map(str, atlas_ids)),
    }
    if vm_train_ids is not None:
        meta["vm_train_ids"] = list(map(str, vm_train_ids))
    if extra:
        meta.update(dict(extra))
    out_path.write_text(json.dumps(meta, indent=2))


def _fuse_labels_mode(labels_nzyx: np.ndarray, *, chunk_size: int = 512_000) -> np.ndarray:
    """
    Voxel-wise mode fusion with tie-breaking by smallest label id.

    labels_nzyx: (N,Z,Y,X) integer array
    Returns: (Z,Y,X) integer array
    """
    import scipy.stats as stats

    a = np.asarray(labels_nzyx)
    if a.ndim != 4:
        raise ValueError("labels_nzyx must have shape (N,Z,Y,X)")
    n, z, y, x = a.shape
    v = int(z * y * x)
    flat = a.reshape(n, v)
    out = np.zeros(v, dtype=np.int32)
    for s in range(0, v, int(chunk_size)):
        e = int(min(v, s + int(chunk_size)))
        # stats.mode returns smallest mode by default for ties.
        m = stats.mode(flat[:, s:e], axis=0, keepdims=False).mode
        out[s:e] = np.asarray(m, dtype=np.int32).reshape(-1)
    return out.reshape(z, y, x)


def build_average_atlas(
    *,
    atlas_cases: Sequence[object],
    out_dir: str | Path | None = None,
) -> AtlasTemplate:
    """
    Build a simple atlas template by resampling all atlas subjects to the first atlas grid,
    averaging intensities, and majority-vote fusing labels in atlas space.

    atlas_cases must expose:
      - patient_id
      - image (Path)
      - label (Path)
      - mask (Path|None)
    """
    if len(atlas_cases) == 0:
        raise ValueError("Empty atlas_cases")

    ref = atlas_cases[0]
    ref_img = read_nifti(ref.image)
    ref_lbl = read_nifti(ref.label) if getattr(ref, "label", None) is not None else None
    if ref_lbl is None:
        raise ValueError("Average atlas requires labels for all atlas cases")

    fixed_shape = ref_img.data_zyx.shape
    spacing = tuple(map(float, ref_img.spacing_zyx))

    imgs = []
    labels = []
    masks = []

    for c in atlas_cases:
        img = read_nifti(c.image)
        lbl = read_nifti(c.label) if getattr(c, "label", None) is not None else None
        if lbl is None:
            raise ValueError("Average atlas requires labels for all atlas cases")

        img_rs = resample_to_fixed_grid(
            img.data_zyx,
            img.spacing_zyx,
            fixed_shape_zyx=fixed_shape,
            fixed_spacing_zyx=spacing,
            order=1,
            cval=float(np.min(img.data_zyx)),
        ).astype(np.float32, copy=False)
        lbl_rs = resample_to_fixed_grid(
            lbl.data_zyx.astype(np.float32, copy=False),
            lbl.spacing_zyx,
            fixed_shape_zyx=fixed_shape,
            fixed_spacing_zyx=spacing,
            order=0,
            cval=0.0,
        ).astype(np.int32, copy=False)

        m = None
        if getattr(c, "mask", None) is not None:
            m_img = read_nifti(c.mask)
            m = resample_to_fixed_grid(
                (m_img.data_zyx > 0).astype(np.float32),
                m_img.spacing_zyx,
                fixed_shape_zyx=fixed_shape,
                fixed_spacing_zyx=spacing,
                order=0,
                cval=0.0,
            )
        else:
            m = (lbl_rs > 0).astype(np.float32)

        imgs.append(img_rs)
        labels.append(lbl_rs)
        masks.append((m > 0.5).astype(np.uint8))

    mean_img = np.mean(np.stack(imgs, axis=0), axis=0).astype(np.float32, copy=False)
    fused_lbl = _fuse_labels_mode(np.stack(labels, axis=0).astype(np.int32, copy=False)).astype(np.uint16, copy=False)
    fused_mask = (fused_lbl > 0).astype(np.uint8)

    atlas_ids = tuple(sorted(str(getattr(c, "patient_id")) for c in atlas_cases))
    tpl = AtlasTemplate(
        atlas_ids=atlas_ids,
        ref_patient_id=str(getattr(ref, "patient_id")),
        image_zyx=mean_img,
        label_zyx=fused_lbl,
        mask_zyx=fused_mask,
        spacing_zyx=spacing,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_npz(
            out_dir / "average_atlas.npz",
            spacing_zyx=np.asarray(spacing, dtype=np.float32),
            atlas_image_zyx=mean_img.astype(np.float32, copy=False),
            atlas_label_zyx=fused_lbl.astype(np.uint16, copy=False),
            atlas_mask_zyx=fused_mask.astype(np.uint8, copy=False),
            atlas_ids=np.asarray(atlas_ids),
            ref_patient_id=np.asarray([tpl.ref_patient_id]),
        )
        # Optional NIfTI outputs for debugging/visualization.
        try:
            write_nifti_scalar_zyx(out_dir / "atlas_image.nii.gz", mean_img, spacing, dtype=np.float32)
            write_nifti_scalar_zyx(out_dir / "atlas_label.nii.gz", fused_lbl.astype(np.float32), spacing, dtype=np.float32)
        except Exception:
            pass

    return tpl

