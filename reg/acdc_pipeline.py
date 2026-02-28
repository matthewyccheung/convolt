from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .acdc_dataset import ACDCPair
from .demons import (
    DemonsParams,
    displacement_magnitude_mm,
    disp_vox_to_mm,
    jacobian_determinant_from_disp_mm,
    register_demons_multires,
    resample_to_fixed_grid,
    warp_zyx,
)
from .metrics import dice_coefficient, mask_volume_ml, predicted_moving_volume_ml_from_jacobian
from .nifti import read_nifti, save_npz, write_nifti_scalar_zyx


@dataclass(frozen=True)
class ACDCResult:
    row: Dict[str, object]
    fixed_img_zyx: np.ndarray
    moving_warped_zyx: np.ndarray
    fixed_mask_zyx: np.ndarray
    moving_mask_warped_zyx: np.ndarray
    disp_mag_mm_zyx: np.ndarray
    slice_z: int


def _pick_slice_from_mask(mask_zyx: np.ndarray) -> int:
    idx = np.argwhere(mask_zyx > 0)
    if idx.size == 0:
        return mask_zyx.shape[0] // 2
    zmin, zmax = int(idx[:, 0].min()), int(idx[:, 0].max())
    return (zmin + zmax) // 2


def _robust_clip_range(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
    x = x.astype(np.float32, copy=False)
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def process_acdc_pair(
    pair: ACDCPair,
    *,
    out_dir: str | Path,
    params: DemonsParams,
    label_lv: int = 3,
    backend: str = "demons",
    vxm_model: object | None = None,
    vxm_cfg: object | None = None,
    save_nifti_outputs: bool = True,
) -> ACDCResult:
    out_dir = Path(out_dir)
    pair_dir = out_dir / "pairs" / pair.patient_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    ed_img = read_nifti(pair.ed_image)
    es_img = read_nifti(pair.es_image)
    ed_seg = read_nifti(pair.ed_seg)
    es_seg = read_nifti(pair.es_seg)

    fixed_img = ed_img.data_zyx
    fixed_spacing = ed_img.spacing_zyx
    fixed_mask = (ed_seg.data_zyx == int(label_lv)).astype(np.uint8)

    moving_img = resample_to_fixed_grid(
        es_img.data_zyx,
        es_img.spacing_zyx,
        fixed_shape_zyx=fixed_img.shape,
        fixed_spacing_zyx=fixed_spacing,
        order=1,
        cval=float(np.min(es_img.data_zyx)),
    )
    moving_mask_native = (es_seg.data_zyx == int(label_lv)).astype(np.uint8)
    moving_mask_rs = resample_to_fixed_grid(
        moving_mask_native,
        es_seg.spacing_zyx,
        fixed_shape_zyx=fixed_img.shape,
        fixed_spacing_zyx=fixed_spacing,
        order=0,
        cval=0.0,
    ).astype(np.uint8)

    # For MRI, set Demons intensity clipping based on robust percentiles per-pair.
    lo, hi = _robust_clip_range(np.concatenate([fixed_img.reshape(-1), moving_img.reshape(-1)]))
    params_pair = replace(params, intensity_clip_hu=(lo, hi))

    backend_n = str(backend).lower()
    if backend_n not in {"demons", "voxelmorph", "itk_elastix_bspline", "sitk_diffeomorphic_demons", "sitk_bspline"}:
        raise ValueError(
            "backend must be one of: demons, voxelmorph, itk_elastix_bspline, sitk_diffeomorphic_demons, sitk_bspline"
        )

    if backend_n == "demons":
        disp_vox = register_demons_multires(fixed_img, moving_img, fixed_mask_zyx=fixed_mask, params=params_pair)
        moving_warped = warp_zyx(moving_img, disp_vox, order=1, cval=float(np.min(moving_img)))
        moving_mask_warped = (warp_zyx(moving_mask_rs.astype(np.float32), disp_vox, order=0, cval=0.0) > 0.5).astype(np.uint8)
    elif backend_n == "voxelmorph":
        if vxm_model is None or vxm_cfg is None:
            raise ValueError("voxelmorph backend requires vxm_model and vxm_cfg")
        from .vxm import infer_flow_and_warp, warp_mask_nearest

        moving_warped, disp_vox = infer_flow_and_warp(
            model=vxm_model,
            inhale_ct_zyx=fixed_img,
            exhale_ct_zyx=moving_img,
            device=getattr(vxm_cfg, "device", "auto"),
            scale=int(getattr(vxm_cfg, "scale", 1)),
            norm=getattr(vxm_cfg, "norm", "percentile"),
            ct_clip=getattr(vxm_cfg, "ct_clip", (-1000.0, 400.0)),
            pct=getattr(vxm_cfg, "pct", (1.0, 99.0)),
        )
        moving_mask_warped = warp_mask_nearest(
            mask_zyx=moving_mask_rs,
            disp_vox_3zyx=disp_vox,
            device=getattr(vxm_cfg, "device", "auto"),
        )
    elif backend_n == "itk_elastix_bspline":
        from .external.itk_elastix import register_affine_bspline

        moving_warped, moving_mask_warped, disp_mm_3zyx, jac_ext = register_affine_bspline(
            fixed_zyx=fixed_img,
            moving_zyx=moving_img,
            fixed_mask_zyx=fixed_mask,
            moving_mask_zyx=moving_mask_rs,
            spacing_zyx=fixed_spacing,
            work_dir=pair_dir / "itk_elastix",
        )
        dz, dy, dx = map(float, fixed_spacing)
        disp_vox = disp_mm_3zyx.astype(np.float32, copy=True)
        disp_vox[0] /= np.float32(dz)
        disp_vox[1] /= np.float32(dy)
        disp_vox[2] /= np.float32(dx)
        jac = jac_ext.astype(np.float32, copy=False)
    elif backend_n == "sitk_diffeomorphic_demons":
        from .external.sitk import register_diffeomorphic_demons

        moving_warped, moving_mask_warped, disp_mm_3zyx, jac_ext = register_diffeomorphic_demons(
            fixed_zyx=fixed_img,
            moving_zyx=moving_img,
            fixed_mask_zyx=fixed_mask,
            moving_mask_zyx=moving_mask_rs,
            spacing_zyx=fixed_spacing,
            work_dir=pair_dir / "sitk_ddemons",
        )
        dz, dy, dx = map(float, fixed_spacing)
        disp_vox = disp_mm_3zyx.astype(np.float32, copy=True)
        disp_vox[0] /= np.float32(dz)
        disp_vox[1] /= np.float32(dy)
        disp_vox[2] /= np.float32(dx)
        jac = jac_ext.astype(np.float32, copy=False)
    else:  # sitk_bspline
        from .external.sitk import register_bspline_ffd

        moving_warped, moving_mask_warped, disp_mm_3zyx, jac_ext = register_bspline_ffd(
            fixed_zyx=fixed_img,
            moving_zyx=moving_img,
            fixed_mask_zyx=fixed_mask,
            moving_mask_zyx=moving_mask_rs,
            spacing_zyx=fixed_spacing,
            work_dir=pair_dir / "sitk_bspline",
        )
        dz, dy, dx = map(float, fixed_spacing)
        disp_vox = disp_mm_3zyx.astype(np.float32, copy=True)
        disp_vox[0] /= np.float32(dz)
        disp_vox[1] /= np.float32(dy)
        disp_vox[2] /= np.float32(dx)
        jac = jac_ext.astype(np.float32, copy=False)

    disp_mm = disp_vox_to_mm(disp_vox, fixed_spacing)
    if "jac" not in locals():
        jac = jacobian_determinant_from_disp_mm(disp_mm, fixed_spacing)
    disp_mag = displacement_magnitude_mm(disp_mm)

    dice_lv = dice_coefficient(fixed_mask, moving_mask_warped)

    edv_ml = mask_volume_ml(fixed_mask, fixed_spacing)
    esv_ml_gt_native = mask_volume_ml(moving_mask_native, es_seg.spacing_zyx)
    esv_ml_gt_fixedgrid = mask_volume_ml(moving_mask_rs, fixed_spacing)
    esv_ml_pred_jac = predicted_moving_volume_ml_from_jacobian(jac, fixed_mask, fixed_spacing)
    mean_jac_in_mask = float(np.mean(jac[fixed_mask > 0])) if np.count_nonzero(fixed_mask) else float("nan")

    # Use EDV from ED segmentation (GT) and ESV predicted to compute LVEF.
    lvef_gt = (edv_ml - esv_ml_gt_native) / max(edv_ml, 1e-6)
    lvef_pred = (edv_ml - esv_ml_pred_jac) / max(edv_ml, 1e-6)

    slice_z = _pick_slice_from_mask(fixed_mask)

    save_npz(
        pair_dir / "artifacts.npz",
        fixed_spacing_zyx=np.asarray(fixed_spacing, dtype=np.float32),
        ed_image_zyx=fixed_img.astype(np.int16, copy=False),
        es_image_resampled_zyx=moving_img.astype(np.int16, copy=False),
        es_image_warped_zyx=moving_warped.astype(np.float32, copy=False),
        ed_lv_mask_zyx=fixed_mask.astype(np.uint8, copy=False),
        es_lv_mask_resampled_zyx=moving_mask_rs.astype(np.uint8, copy=False),
        es_lv_mask_warped_zyx=moving_mask_warped.astype(np.uint8, copy=False),
        disp_vox_3zyx=disp_vox.astype(np.float32, copy=False),
        disp_mm_3zyx=disp_mm.astype(np.float32, copy=False),
        jac_det_zyx=jac.astype(np.float32, copy=False),
        disp_mag_mm_zyx=disp_mag.astype(np.float32, copy=False),
        backend=np.asarray(backend_n),
        label_lv=np.asarray(int(label_lv)),
    )

    if save_nifti_outputs:
        write_nifti_scalar_zyx(pair_dir / "jac_det.nii.gz", jac, fixed_spacing, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "disp_mag_mm.nii.gz", disp_mag, fixed_spacing, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "disp_z_mm.nii.gz", disp_mm[0], fixed_spacing, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "disp_y_mm.nii.gz", disp_mm[1], fixed_spacing, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "disp_x_mm.nii.gz", disp_mm[2], fixed_spacing, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "es_warped.nii.gz", moving_warped, fixed_spacing, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "es_lv_mask_warped.nii.gz", moving_mask_warped, fixed_spacing, dtype=np.uint8)

    row: Dict[str, object] = {
        "patient_id": pair.patient_id,
        "split": pair.split,
        "backend": backend_n,
        "ed_frame": int(pair.ed_frame),
        "es_frame": int(pair.es_frame),
        "ed_image": str(pair.ed_image),
        "es_image": str(pair.es_image),
        "ed_seg": str(pair.ed_seg),
        "es_seg": str(pair.es_seg),
        "spacing_z_mm": float(fixed_spacing[0]),
        "spacing_y_mm": float(fixed_spacing[1]),
        "spacing_x_mm": float(fixed_spacing[2]),
        "label_lv": int(label_lv),
        "dice_lv": float(dice_lv),
        "edv_ml_gt": float(edv_ml),
        "esv_ml_gt_native": float(esv_ml_gt_native),
        "esv_ml_gt_fixedgrid": float(esv_ml_gt_fixedgrid),
        "esv_ml_pred_jac": float(esv_ml_pred_jac),
        "mean_jac_in_ed_mask": float(mean_jac_in_mask),
        "esv_ml_error": float(esv_ml_pred_jac - esv_ml_gt_native),
        "delta_ml_gt_es_minus_ed": float(esv_ml_gt_native - edv_ml),
        "delta_ml_pred_es_minus_ed": float(esv_ml_pred_jac - edv_ml),
        "delta_ml_error": float((esv_ml_pred_jac - edv_ml) - (esv_ml_gt_native - edv_ml)),
        "lvef_gt": float(lvef_gt),
        "lvef_pred": float(lvef_pred),
        "lvef_error": float(lvef_pred - lvef_gt),
        "pair_dir": str(pair_dir),
    }

    return ACDCResult(
        row=row,
        fixed_img_zyx=fixed_img,
        moving_warped_zyx=moving_warped,
        fixed_mask_zyx=fixed_mask,
        moving_mask_warped_zyx=moving_mask_warped,
        disp_mag_mm_zyx=disp_mag,
        slice_z=int(slice_z),
    )
