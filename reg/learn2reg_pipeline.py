from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .atlas import AtlasMode, AtlasSpec, AtlasTemplate, _fuse_labels_mode, atlas_tag
from .demons import (
    DemonsParams,
    displacement_magnitude_mm,
    disp_vox_to_mm,
    jacobian_determinant_from_disp_mm,
    register_demons_multires,
    resample_to_fixed_grid,
    warp_zyx,
)
from .metrics import dice_coefficient
from .nifti import read_nifti, save_npz, write_nifti_scalar_zyx
from .viz import save_pair_figure

from scipy import ndimage


def _pick_slice(mask_zyx: np.ndarray) -> int:
    idx = np.argwhere(mask_zyx > 0)
    if idx.size == 0:
        return int(mask_zyx.shape[0] // 2)
    zmin, zmax = int(idx[:, 0].min()), int(idx[:, 0].max())
    return int((zmin + zmax) // 2)


def _voxel_volume_ml(spacing_zyx: Tuple[float, float, float]) -> float:
    dz, dy, dx = map(float, spacing_zyx)
    return (dz * dy * dx) / 1000.0


def _as_mask(arr_zyx: np.ndarray) -> np.ndarray:
    return (np.asarray(arr_zyx) > 0).astype(np.uint8, copy=False)


def _label_volumes_ml(label_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float], label_ids: Sequence[int]) -> Dict[int, float]:
    vml = float(_voxel_volume_ml(spacing_zyx))
    lab = np.asarray(label_zyx)
    out: Dict[int, float] = {}
    for lid in label_ids:
        out[int(lid)] = float(np.count_nonzero(lab == int(lid))) * vml
    return out


@dataclass(frozen=True)
class Learn2RegCaseResult:
    row: Dict[str, object]
    fixed_img_zyx: np.ndarray
    moving_warped_zyx: np.ndarray
    fixed_union_mask_zyx: np.ndarray
    pred_union_mask_zyx: np.ndarray
    disp_mag_mm_zyx: np.ndarray
    slice_z: int
    pred_label_zyx: np.ndarray


def _compute_roi_mask(
    *,
    fixed_shape_zyx: Tuple[int, int, int],
    fixed_spacing_zyx: Tuple[float, float, float],
    mask_path: Path | None,
) -> np.ndarray:
    if mask_path is not None and mask_path.exists():
        m = read_nifti(mask_path)
        if m.data_zyx.shape != fixed_shape_zyx or tuple(map(float, m.spacing_zyx)) != tuple(map(float, fixed_spacing_zyx)):
            rs = resample_to_fixed_grid(
                (m.data_zyx > 0).astype(np.float32),
                m.spacing_zyx,
                fixed_shape_zyx=fixed_shape_zyx,
                fixed_spacing_zyx=fixed_spacing_zyx,
                order=0,
                cval=0.0,
            )
            return (rs > 0.5).astype(np.uint8)
        return (m.data_zyx > 0).astype(np.uint8)
    return np.ones(fixed_shape_zyx, dtype=np.uint8)


def _warp_labels_nearest_demons(label_zyx: np.ndarray, disp_vox_3zyx: np.ndarray) -> np.ndarray:
    warped_f = warp_zyx(label_zyx.astype(np.float32, copy=False), disp_vox_3zyx, order=0, cval=0.0)
    return np.asarray(np.rint(warped_f), dtype=np.uint16)


def _foreground_mask_percentile(vol_zyx: np.ndarray) -> np.ndarray:
    """
    Cheap MR foreground mask to support VM pre-alignment when explicit masks are absent.
    """
    v = np.asarray(vol_zyx, dtype=np.float32)
    p5 = float(np.percentile(v, 5.0))
    p95 = float(np.percentile(v, 95.0))
    thr = p5 + 0.10 * (p95 - p5)
    return (v > thr).astype(np.uint8)


def _center_of_mass_zyx(mask_zyx: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask_zyx > 0)
    if idx.size == 0:
        z, y, x = mask_zyx.shape
        return np.array([0.5 * (z - 1), 0.5 * (y - 1), 0.5 * (x - 1)], dtype=np.float32)
    return idx.mean(axis=0).astype(np.float32)


def _prealign_translation(
    *,
    fixed_img_zyx: np.ndarray,
    moving_img_zyx: np.ndarray,
    moving_lbl_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Translation-only prealignment by matching foreground centers-of-mass.
    Applies the shift to moving image (linear) and moving label (nearest).
    """
    fixed_fg = _foreground_mask_percentile(fixed_img_zyx)
    moving_fg = _foreground_mask_percentile(moving_img_zyx)
    cf = _center_of_mass_zyx(fixed_fg)
    cm = _center_of_mass_zyx(moving_fg)
    shift = (cf - cm).astype(np.float32)  # z,y,x
    # ndimage.shift uses positive shift to move content toward higher indices.
    mov_img = ndimage.shift(moving_img_zyx, shift=shift, order=1, mode="nearest")
    mov_lbl = ndimage.shift(moving_lbl_zyx, shift=shift, order=0, mode="constant", cval=0.0)
    return mov_img.astype(np.float32, copy=False), mov_lbl.astype(np.uint16, copy=False)


def _register_one_atlas(
    *,
    backend: str,
    fixed_img_zyx: np.ndarray,
    fixed_spacing_zyx: Tuple[float, float, float],
    fixed_roi_mask_zyx: np.ndarray,
    moving_img_path: Path | None,
    moving_lbl_path: Path | None,
    moving_img_zyx: np.ndarray | None = None,
    moving_lbl_zyx: np.ndarray | None = None,
    moving_spacing_zyx: Tuple[float, float, float] | None = None,
    params: DemonsParams,
    work_dir: Path,
    vxm_model: object | None = None,
    vxm_cfg: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Register moving->fixed and warp moving labels into fixed space.

    Returns:
      moving_warped_zyx, label_warped_zyx, disp_mm_3zyx, jac_det_zyx
    """
    fixed_shape = fixed_img_zyx.shape
    spacing = fixed_spacing_zyx

    if moving_img_zyx is None or moving_lbl_zyx is None or moving_spacing_zyx is None:
        if moving_img_path is None or moving_lbl_path is None:
            raise ValueError("moving image/label must be provided")
        mv_img = read_nifti(moving_img_path)
        mv_lbl = read_nifti(moving_lbl_path)
        moving_img_zyx = mv_img.data_zyx
        moving_lbl_zyx = mv_lbl.data_zyx
        moving_spacing_zyx = mv_img.spacing_zyx

    moving_img_rs = resample_to_fixed_grid(
        moving_img_zyx,
        moving_spacing_zyx,
        fixed_shape_zyx=fixed_shape,
        fixed_spacing_zyx=spacing,
        order=1,
        cval=float(np.min(moving_img_zyx)),
    ).astype(np.float32, copy=False)
    moving_lbl_rs = resample_to_fixed_grid(
        moving_lbl_zyx.astype(np.float32, copy=False),
        moving_spacing_zyx,
        fixed_shape_zyx=fixed_shape,
        fixed_spacing_zyx=spacing,
        order=0,
        cval=0.0,
    ).astype(np.uint16, copy=False)

    backend_n = str(backend).lower()
    if backend_n == "demons":
        disp_vox = register_demons_multires(
            fixed_img_zyx,
            moving_img_rs,
            fixed_mask_zyx=fixed_roi_mask_zyx,
            params=params,
        )
        moving_warped = warp_zyx(moving_img_rs, disp_vox, order=1, cval=float(np.min(moving_img_rs))).astype(np.float32, copy=False)
        label_warped = _warp_labels_nearest_demons(moving_lbl_rs, disp_vox)
        disp_mm = disp_vox_to_mm(disp_vox, spacing)
        jac = jacobian_determinant_from_disp_mm(disp_mm, spacing).astype(np.float32, copy=False)
        return moving_warped, label_warped, disp_mm.astype(np.float32, copy=False), jac

    if backend_n == "voxelmorph":
        if vxm_model is None or vxm_cfg is None:
            raise ValueError("voxelmorph backend requires vxm_model and vxm_cfg")
        from .vxm import infer_flow_and_warp, warp_image_bilinear, warp_labels_nearest

        # VM can fail if atlas and target are translated far apart. Do a lightweight translation-only
        # prealignment to improve initial overlap (no GT leakage).
        moving_img_infer = moving_img_rs
        moving_lbl_infer = moving_lbl_rs
        try:
            moving_img_infer, moving_lbl_infer = _prealign_translation(
                fixed_img_zyx=fixed_img_zyx,
                moving_img_zyx=moving_img_rs,
                moving_lbl_zyx=moving_lbl_rs,
            )
        except Exception:
            pass

        moving_warped, disp_vox = infer_flow_and_warp(
            model=vxm_model,
            inhale_ct_zyx=fixed_img_zyx,
            exhale_ct_zyx=moving_img_infer,
            device=getattr(vxm_cfg, "device", "auto"),
            scale=int(getattr(vxm_cfg, "scale", 1)),
            norm=getattr(vxm_cfg, "norm", "percentile"),
            ct_clip=getattr(vxm_cfg, "ct_clip", (-1000.0, 400.0)),
            pct=getattr(vxm_cfg, "pct", (1.0, 99.0)),
        )

        label_warped = warp_labels_nearest(
            labels_zyx=moving_lbl_infer,
            disp_vox_3zyx=disp_vox,
            device=getattr(vxm_cfg, "device", "auto"),
        )
        # Sanity: for some datasets, a sign convention mismatch or bad flow can warp labels out-of-bounds.
        # Compare against the negated flow and pick the better one using overlap with a fixed foreground proxy
        # (no GT leakage). Only compute the alternative when the primary looks suspicious.
        fixed_fg = _foreground_mask_percentile(fixed_img_zyx)
        nz = int(np.count_nonzero(label_warped))
        ov = int(np.count_nonzero((label_warped > 0) & (fixed_fg > 0)))
        ov_frac = float(ov / max(nz, 1))
        if nz == 0 or ov_frac < 0.01:
            label_warped_alt = warp_labels_nearest(
                labels_zyx=moving_lbl_infer,
                disp_vox_3zyx=-disp_vox,
                device=getattr(vxm_cfg, "device", "auto"),
            )
            nz_alt = int(np.count_nonzero(label_warped_alt))
            ov_alt = int(np.count_nonzero((label_warped_alt > 0) & (fixed_fg > 0)))
            # Prefer higher foreground overlap; break ties by larger nonzero support.
            if (ov_alt > ov) or (ov_alt == ov and nz_alt > nz):
                disp_vox = -disp_vox
                label_warped = label_warped_alt
                moving_warped = warp_image_bilinear(
                    volume_zyx=moving_img_infer,
                    disp_vox_3zyx=disp_vox,
                    device=getattr(vxm_cfg, "device", "auto"),
                    padding_mode="border",
                )
        disp_mm = disp_vox_to_mm(disp_vox, spacing)
        jac = jacobian_determinant_from_disp_mm(disp_mm, spacing).astype(np.float32, copy=False)
        return moving_warped.astype(np.float32, copy=False), label_warped.astype(np.uint16, copy=False), disp_mm.astype(np.float32, copy=False), jac

    if backend_n == "sitk_diffeomorphic_demons":
        from .external.sitk import register_diffeomorphic_demons

        # Use moving_lbl_rs as the "moving_mask" to warp with NN; the backend uses fixed_roi_mask for masking.
        moving_warped, label_warped, disp_mm, jac = register_diffeomorphic_demons(
            fixed_zyx=fixed_img_zyx,
            moving_zyx=moving_img_rs,
            fixed_mask_zyx=fixed_roi_mask_zyx,
            moving_mask_zyx=moving_lbl_rs,
            spacing_zyx=spacing,
            work_dir=work_dir,
        )
        return (
            moving_warped.astype(np.float32, copy=False),
            np.asarray(label_warped, dtype=np.uint16),
            disp_mm.astype(np.float32, copy=False),
            jac.astype(np.float32, copy=False),
        )

    if backend_n == "sitk_bspline":
        from .external.sitk import register_bspline_ffd

        moving_warped, label_warped, disp_mm, jac = register_bspline_ffd(
            fixed_zyx=fixed_img_zyx,
            moving_zyx=moving_img_rs,
            fixed_mask_zyx=fixed_roi_mask_zyx,
            moving_mask_zyx=moving_lbl_rs,
            spacing_zyx=spacing,
            work_dir=work_dir,
        )
        return (
            moving_warped.astype(np.float32, copy=False),
            np.asarray(label_warped, dtype=np.uint16),
            disp_mm.astype(np.float32, copy=False),
            jac.astype(np.float32, copy=False),
        )

    if backend_n == "itk_elastix_bspline":
        from .external.itk_elastix import register_affine_bspline

        moving_warped, label_warped, disp_mm, jac = register_affine_bspline(
            fixed_zyx=fixed_img_zyx,
            moving_zyx=moving_img_rs,
            fixed_mask_zyx=fixed_roi_mask_zyx,
            moving_mask_zyx=moving_lbl_rs,
            spacing_zyx=spacing,
            work_dir=work_dir,
        )
        return (
            moving_warped.astype(np.float32, copy=False),
            np.asarray(label_warped, dtype=np.uint16),
            disp_mm.astype(np.float32, copy=False),
            jac.astype(np.float32, copy=False),
        )

    raise ValueError(f"Unknown backend: {backend}")


def process_learn2reg_case(
    *,
    dataset: str,
    backend: str,
    case: object,
    atlas_spec: AtlasSpec,
    atlas_cases: Sequence[object],
    atlas_template: AtlasTemplate | None,
    out_dir: str | Path,
    params: DemonsParams,
    label_ids: Sequence[int],
    vxm_model: object | None = None,
    vxm_cfg: object | None = None,
    save_nifti_outputs: bool = True,
) -> Learn2RegCaseResult:
    out_dir = Path(out_dir)
    pair_dir = out_dir / "pairs" / str(getattr(case, "patient_id"))
    pair_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = read_nifti(getattr(case, "image"))
    fixed_spacing = tuple(map(float, fixed_img.spacing_zyx))
    fixed_zyx = fixed_img.data_zyx.astype(np.float32, copy=False)
    fixed_shape = fixed_zyx.shape

    gt_label_zyx: Optional[np.ndarray] = None
    if getattr(case, "label", None) is not None:
        gt = read_nifti(getattr(case, "label"))
        # Prefer direct voxel alignment when the label is already on the same grid as the image.
        # Some Learn2Reg tasks include consistent image/label grids; resampling via spacing-only can introduce
        # unintended shifts due to center-crop/pad heuristics.
        if gt.data_zyx.shape == fixed_shape and tuple(map(float, gt.spacing_zyx)) == tuple(map(float, fixed_spacing)):
            gt_label_zyx = gt.data_zyx.astype(np.uint16, copy=False)
        else:
            gt_label_zyx = resample_to_fixed_grid(
                gt.data_zyx.astype(np.float32, copy=False),
                gt.spacing_zyx,
                fixed_shape_zyx=fixed_shape,
                fixed_spacing_zyx=fixed_spacing,
                order=0,
                cval=0.0,
            ).astype(np.uint16, copy=False)

    fixed_roi = _compute_roi_mask(
        fixed_shape_zyx=fixed_shape,
        fixed_spacing_zyx=fixed_spacing,
        mask_path=getattr(case, "mask", None),
    )

    # Build list of atlas sources to register.
    atlas_mode = str(atlas_spec.mode).lower()
    sources: List[tuple[str, Path | None, Path | None, np.ndarray | None, np.ndarray | None, Tuple[float, float, float] | None]] = []
    if atlas_mode == "average":
        if atlas_template is None:
            raise ValueError("atlas_template required for average mode")
        sources.append(("average_atlas", None, None, atlas_template.image_zyx, atlas_template.label_zyx, atlas_template.spacing_zyx))
    else:
        for ac in atlas_cases:
            sources.append((str(getattr(ac, "patient_id")), getattr(ac, "image"), getattr(ac, "label"), None, None, None))
        if atlas_mode == "single":
            sources = sources[:1]

    warped_imgs = []
    warped_labels = []
    disp_list = []
    jac_list = []

    for atlas_id, img_path, lbl_path, img_arr, lbl_arr, spc in sources:
        work = pair_dir / f"{atlas_id}"
        moving_warped, label_warped, disp_mm, jac = _register_one_atlas(
            backend=str(backend),
            fixed_img_zyx=fixed_zyx,
            fixed_spacing_zyx=fixed_spacing,
            fixed_roi_mask_zyx=fixed_roi,
            moving_img_path=img_path,
            moving_lbl_path=lbl_path,
            moving_img_zyx=img_arr,
            moving_lbl_zyx=lbl_arr,
            moving_spacing_zyx=spc,
            params=params,
            work_dir=work,
            vxm_model=vxm_model,
            vxm_cfg=vxm_cfg,
        )
        warped_imgs.append(moving_warped.astype(np.float32, copy=False))
        warped_labels.append(label_warped.astype(np.uint16, copy=False))
        disp_list.append(disp_mm.astype(np.float32, copy=False))
        jac_list.append(jac.astype(np.float32, copy=False))

    if len(warped_labels) == 0:
        raise RuntimeError("No atlas predictions produced")

    labels_stack = np.stack(warped_labels, axis=0)
    pred_label = _fuse_labels_mode(labels_stack) if labels_stack.shape[0] > 1 else labels_stack[0].astype(np.int32, copy=False)
    pred_label = pred_label.astype(np.uint16, copy=False)
    pred_union = (pred_label > 0).astype(np.uint8)

    fixed_union = (gt_label_zyx > 0).astype(np.uint8) if gt_label_zyx is not None else np.zeros_like(pred_union, dtype=np.uint8)

    # ROI mask for feature extraction must be available at test-time.
    # Prefer provided masks; otherwise use the predicted union mask (no GT leakage).
    if getattr(case, "mask", None) is not None:
        roi_mask = fixed_roi
    else:
        # If prediction is empty (all background), fall back to a non-empty ROI to avoid all-NaN features.
        roi_mask = pred_union if np.any(pred_union) else fixed_roi

    union_dice = float(dice_coefficient(fixed_union, pred_union)) if gt_label_zyx is not None else float("nan")

    vml = float(_voxel_volume_ml(fixed_spacing))
    pred_union_vol = float(np.count_nonzero(pred_union)) * vml
    gt_union_vol = float(np.count_nonzero(fixed_union)) * vml if gt_label_zyx is not None else float("nan")

    # Aggregate deformation fields for features/viz.
    disp_mm_mean = np.mean(np.stack(disp_list, axis=0), axis=0).astype(np.float32, copy=False)
    jac_mean = np.mean(np.stack(jac_list, axis=0), axis=0).astype(np.float32, copy=False)
    disp_mag = displacement_magnitude_mm(disp_mm_mean)

    moving_warped_mean = np.mean(np.stack(warped_imgs, axis=0), axis=0).astype(np.float32, copy=False)

    # Pick a slice that actually intersects a meaningful segmentation region.
    # For labeled (training) cases, prefer the fixed GT union; otherwise fall back to prediction/ROI.
    if gt_label_zyx is not None and np.any(fixed_union):
        slice_z = _pick_slice(fixed_union)
    else:
        slice_z = _pick_slice(pred_union if np.any(pred_union) else fixed_roi)

    # Save artifacts (align keys with existing UQ feature extractor).
    save_npz(
        pair_dir / "artifacts.npz",
        spacing_zyx=np.asarray(fixed_spacing, dtype=np.float32),
        fixed_image_zyx=fixed_zyx.astype(np.float32, copy=False),
        moving_warped_zyx=moving_warped_mean.astype(np.float32, copy=False),
        fixed_mask_zyx=fixed_roi.astype(np.uint8, copy=False),
        roi_mask_zyx=roi_mask.astype(np.uint8, copy=False),
        pred_label_zyx=pred_label.astype(np.uint16, copy=False),
        pred_union_mask_zyx=pred_union.astype(np.uint8, copy=False),
        jac_det_zyx=jac_mean.astype(np.float32, copy=False),
        disp_mm_3zyx=disp_mm_mean.astype(np.float32, copy=False),
        disp_mag_mm_zyx=disp_mag.astype(np.float32, copy=False),
    )

    if save_nifti_outputs:
        try:
            write_nifti_scalar_zyx(pair_dir / "jac_det.nii.gz", jac_mean, fixed_spacing, dtype=np.float32)
            write_nifti_scalar_zyx(pair_dir / "disp_mag_mm.nii.gz", disp_mag, fixed_spacing, dtype=np.float32)
            write_nifti_scalar_zyx(pair_dir / "pred_union_mask.nii.gz", pred_union.astype(np.float32), fixed_spacing, dtype=np.float32)
        except Exception:
            pass

    # Figure: fixed, warped atlas mean, |disp|
    stats = (
        f"Dice(union): {union_dice:.3f}\n"
        f"V_union GT: {gt_union_vol:.1f} mL\n"
        f"V_union Pred: {pred_union_vol:.1f} mL"
    )
    save_pair_figure(
        out_path=out_dir / "figures" / f"{getattr(case, 'patient_id')}.png",
        fixed_img_zyx=fixed_zyx,
        moving_warped_zyx=moving_warped_mean,
        fixed_mask_zyx=(fixed_union if gt_label_zyx is not None else pred_union),
        moving_mask_warped_zyx=pred_union,
        disp_mag_mm_zyx=disp_mag,
        slice_z=int(slice_z),
        title_fixed=f"{getattr(case, 'patient_id')} | fixed",
        title_moving="atlas (warped → fixed)",
        stats_text=stats,
    )

    row: Dict[str, object] = {
        "patient_id": str(getattr(case, "patient_id")),
        "split": str(getattr(case, "split")),
        "backend": str(backend),
        "atlas_mode": str(atlas_spec.mode),
        "atlas_n": int(atlas_spec.n),
        "atlas_tag": atlas_tag(atlas_spec),
        "image": str(getattr(case, "image")),
        "label": str(getattr(case, "label")) if getattr(case, "label", None) is not None else "",
        "mask": str(getattr(case, "mask")) if getattr(case, "mask", None) is not None else "",
        "spacing_z_mm": float(fixed_spacing[0]),
        "spacing_y_mm": float(fixed_spacing[1]),
        "spacing_x_mm": float(fixed_spacing[2]),
        "union_dice": float(union_dice),
        "union_vol_ml_gt": float(gt_union_vol),
        "union_vol_ml_pred": float(pred_union_vol),
        "pair_dir": str(pair_dir),
    }

    # Per-label volumes are saved separately by the caller (label_volumes.csv),
    # but we keep union stats in summary.csv.
    return Learn2RegCaseResult(
        row=row,
        fixed_img_zyx=fixed_zyx,
        moving_warped_zyx=moving_warped_mean,
        fixed_union_mask_zyx=fixed_union,
        pred_union_mask_zyx=pred_union,
        disp_mag_mm_zyx=disp_mag,
        slice_z=int(slice_z),
        pred_label_zyx=pred_label,
    )


def run_learn2reg_registration(
    *,
    dataset: str,
    backend: str,
    cases: Sequence[object],
    atlas_spec: AtlasSpec,
    atlas_cases: Sequence[object],
    atlas_template: AtlasTemplate | None,
    out_dir: str | Path,
    params: DemonsParams,
    label_ids: Sequence[int],
    vxm_model: object | None = None,
    vxm_cfg: object | None = None,
    save_nifti_outputs: bool = True,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Run atlas-based segmentation by registration for a list of target cases.

    Returns:
      - summary_rows: list of dicts for summary.csv (union volume + dice)
      - label_rows: list of dicts for label_volumes.csv (per-label volumes + dice where available)
    """
    out_dir = Path(out_dir)
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    label_rows: List[Dict[str, object]] = []

    for case in tqdm(cases, desc=f"Learn2Reg {dataset} ({backend}, {atlas_tag(atlas_spec)})"):
        res = process_learn2reg_case(
            dataset=dataset,
            backend=str(backend),
            case=case,
            atlas_spec=atlas_spec,
            atlas_cases=atlas_cases,
            atlas_template=atlas_template,
            out_dir=out_dir,
            params=params,
            label_ids=label_ids,
            vxm_model=vxm_model,
            vxm_cfg=vxm_cfg,
            save_nifti_outputs=save_nifti_outputs,
        )
        summary_rows.append(res.row)

        # Per-label volumes (GT if available).
        spacing = (float(res.row["spacing_z_mm"]), float(res.row["spacing_y_mm"]), float(res.row["spacing_x_mm"]))
        # Note: res.row spacing is Z,Y,X; _label_volumes_ml expects Z,Y,X spacing too.
        vol_pred = _label_volumes_ml(res.pred_label_zyx, spacing, label_ids)
        gt_path = getattr(case, "label", None)
        vol_gt: Dict[int, float] = {int(l): float("nan") for l in label_ids}
        dice_gt: Dict[int, float] = {int(l): float("nan") for l in label_ids}
        if gt_path is not None and Path(gt_path).exists():
            gt = read_nifti(Path(gt_path))
            if gt.data_zyx.shape == res.pred_label_zyx.shape and tuple(map(float, gt.spacing_zyx)) == tuple(map(float, spacing)):
                gt_lbl = gt.data_zyx.astype(np.uint16, copy=False)
            else:
                gt_lbl = resample_to_fixed_grid(
                    gt.data_zyx.astype(np.float32, copy=False),
                    gt.spacing_zyx,
                    fixed_shape_zyx=res.pred_label_zyx.shape,
                    fixed_spacing_zyx=spacing,
                    order=0,
                    cval=0.0,
                ).astype(np.uint16, copy=False)
            vol_gt = _label_volumes_ml(gt_lbl, spacing, label_ids)
            for lid in label_ids:
                a = (gt_lbl == int(lid)).astype(np.uint8)
                b = (res.pred_label_zyx == int(lid)).astype(np.uint8)
                dice_gt[int(lid)] = float(dice_coefficient(a, b))

        for lid in label_ids:
            label_rows.append(
                {
                    "patient_id": str(getattr(case, "patient_id")),
                    "split": str(getattr(case, "split")),
                    "backend": str(backend),
                    "atlas_tag": atlas_tag(atlas_spec),
                    "label_id": int(lid),
                    "vol_ml_pred": float(vol_pred[int(lid)]),
                    "vol_ml_gt": float(vol_gt[int(lid)]),
                    "dice": float(dice_gt[int(lid)]),
                    "pair_dir": str(res.row["pair_dir"]),
                }
            )

    return summary_rows, label_rows
