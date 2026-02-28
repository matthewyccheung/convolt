from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .dataset import Pair
from .demons import (
    DemonsParams,
    displacement_magnitude_mm,
    disp_vox_to_mm,
    jacobian_determinant_from_disp_mm,
    register_demons_multires,
    resample_to_fixed_grid,
    warp_zyx,
)
from .metrics import compute_volume_change, dice_coefficient, jaccard_index
from .metrics import mask_volume_ml
from .nifti import read_nifti, save_npz, write_nifti_scalar_zyx


@dataclass(frozen=True)
class PairResult:
    row: Dict[str, object]
    fixed_ct_zyx: np.ndarray
    moving_ct_warped_zyx: np.ndarray
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


def process_pair(
    pair: Pair,
    *,
    out_dir: str | Path,
    params: DemonsParams,
    save_nifti_outputs: bool = True,
    phase_policy: str = "mask_volume",
    backend: str = "demons",
    vxm_model: object | None = None,
    vxm_cfg: object | None = None,
) -> PairResult:
    out_dir = Path(out_dir)
    pair_dir = out_dir / "pairs" / f"{pair.patient_id}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    a_img = read_nifti(pair.fixed_image)
    b_img = read_nifti(pair.moving_image)
    a_mask_img = read_nifti(pair.fixed_mask)
    b_mask_img = read_nifti(pair.moving_mask)

    a_mask = (a_mask_img.data_zyx > 0).astype(np.uint8)
    b_mask = (b_mask_img.data_zyx > 0).astype(np.uint8)

    inhale_img = a_img
    exhale_img = b_img
    inhale_mask_img = a_mask_img
    exhale_mask_img = b_mask_img
    inhale_mask = a_mask
    exhale_mask = b_mask
    inhale_path = pair.fixed_image
    exhale_path = pair.moving_image
    inhale_mask_path = pair.fixed_mask
    exhale_mask_path = pair.moving_mask
    swapped = False

    policy = str(phase_policy).lower()
    if policy not in {"json", "suffix", "mask_volume"}:
        raise ValueError("phase_policy must be one of: json, suffix, mask_volume")

    if policy == "suffix":
        # User-provided convention: 0000=inhale, 0001=exhale (can be wrong; use with care).
        a_is_0000 = str(pair.fixed_image).endswith("_0000.nii.gz")
        b_is_0000 = str(pair.moving_image).endswith("_0000.nii.gz")
        if b_is_0000 and not a_is_0000:
            swapped = True
    elif policy == "mask_volume":
        # More robust: inhale tends to have larger lung mask volume.
        va = mask_volume_ml(a_mask, a_mask_img.spacing_zyx)
        vb = mask_volume_ml(b_mask, b_mask_img.spacing_zyx)
        if vb > va:
            swapped = True

    if swapped:
        inhale_img, exhale_img = b_img, a_img
        inhale_mask_img, exhale_mask_img = b_mask_img, a_mask_img
        inhale_mask, exhale_mask = b_mask, a_mask
        inhale_path, exhale_path = pair.moving_image, pair.fixed_image
        inhale_mask_path, exhale_mask_path = pair.moving_mask, pair.fixed_mask

    inhale_ct = inhale_img.data_zyx
    inhale_mask = inhale_mask.astype(np.uint8, copy=False)
    spacing_zyx = inhale_img.spacing_zyx

    exhale_ct = resample_to_fixed_grid(
        exhale_img.data_zyx,
        exhale_img.spacing_zyx,
        fixed_shape_zyx=inhale_ct.shape,
        fixed_spacing_zyx=spacing_zyx,
        order=1,
        cval=float(np.min(exhale_img.data_zyx)),
    )
    exhale_mask_rs = resample_to_fixed_grid(
        exhale_mask.astype(np.uint8, copy=False),
        exhale_mask_img.spacing_zyx,
        fixed_shape_zyx=inhale_ct.shape,
        fixed_spacing_zyx=spacing_zyx,
        order=0,
        cval=0.0,
    ).astype(np.uint8)

    backend_n = str(backend).lower()
    if backend_n not in {"demons", "voxelmorph", "itk_elastix_bspline", "sitk_diffeomorphic_demons", "sitk_bspline"}:
        raise ValueError(
            "backend must be one of: demons, voxelmorph, itk_elastix_bspline, sitk_diffeomorphic_demons, sitk_bspline"
        )

    if backend_n == "demons":
        disp_vox = register_demons_multires(inhale_ct, exhale_ct, fixed_mask_zyx=inhale_mask, params=params)
        exhale_ct_warped = warp_zyx(exhale_ct, disp_vox, order=1, cval=float(np.min(exhale_ct)))
        exhale_mask_warped = (warp_zyx(exhale_mask_rs.astype(np.float32), disp_vox, order=0, cval=0.0) > 0.5).astype(np.uint8)
    else:
        if backend_n == "voxelmorph":
            if vxm_model is None or vxm_cfg is None:
                raise ValueError("voxelmorph backend requires vxm_model and vxm_cfg")
            from .vxm import infer_flow_and_warp, warp_mask_nearest

            exhale_ct_warped, disp_vox = infer_flow_and_warp(
                model=vxm_model,
                inhale_ct_zyx=inhale_ct,
                exhale_ct_zyx=exhale_ct,
                device=getattr(vxm_cfg, "device", "auto"),
                scale=int(getattr(vxm_cfg, "scale", 1)),
                norm=getattr(vxm_cfg, "norm", "ct_hu"),
                ct_clip=getattr(vxm_cfg, "ct_clip", (-1000.0, 400.0)),
                pct=getattr(vxm_cfg, "pct", (1.0, 99.0)),
            )
            exhale_mask_warped = warp_mask_nearest(
                mask_zyx=exhale_mask_rs,
                disp_vox_3zyx=disp_vox,
                device=getattr(vxm_cfg, "device", "auto"),
            )
        elif backend_n == "itk_elastix_bspline":
            from .external.itk_elastix import register_affine_bspline

            moving_warped, moving_mask_warped, disp_mm_3zyx, jac_ext = register_affine_bspline(
                fixed_zyx=inhale_ct,
                moving_zyx=exhale_ct,
                fixed_mask_zyx=inhale_mask,
                moving_mask_zyx=exhale_mask_rs,
                spacing_zyx=spacing_zyx,
                work_dir=pair_dir / "itk_elastix",
            )
            exhale_ct_warped = moving_warped
            exhale_mask_warped = moving_mask_warped
            dz, dy, dx = map(float, spacing_zyx)
            disp_vox = disp_mm_3zyx.copy()
            disp_vox[0] /= np.float32(dz)
            disp_vox[1] /= np.float32(dy)
            disp_vox[2] /= np.float32(dx)
            jac = jac_ext.astype(np.float32, copy=False)
        elif backend_n == "sitk_diffeomorphic_demons":
            from .external.sitk import register_diffeomorphic_demons

            moving_warped, moving_mask_warped, disp_mm_3zyx, jac_ext = register_diffeomorphic_demons(
                fixed_zyx=inhale_ct,
                moving_zyx=exhale_ct,
                fixed_mask_zyx=inhale_mask,
                moving_mask_zyx=exhale_mask_rs,
                spacing_zyx=spacing_zyx,
                work_dir=pair_dir / "sitk_ddemons",
            )
            exhale_ct_warped = moving_warped
            exhale_mask_warped = moving_mask_warped
            dz, dy, dx = map(float, spacing_zyx)
            disp_vox = disp_mm_3zyx.copy()
            disp_vox[0] /= np.float32(dz)
            disp_vox[1] /= np.float32(dy)
            disp_vox[2] /= np.float32(dx)
            jac = jac_ext.astype(np.float32, copy=False)
        else:  # sitk_bspline
            from .external.sitk import register_bspline_ffd

            moving_warped, moving_mask_warped, disp_mm_3zyx, jac_ext = register_bspline_ffd(
                fixed_zyx=inhale_ct,
                moving_zyx=exhale_ct,
                fixed_mask_zyx=inhale_mask,
                moving_mask_zyx=exhale_mask_rs,
                spacing_zyx=spacing_zyx,
                work_dir=pair_dir / "sitk_bspline",
            )
            exhale_ct_warped = moving_warped
            exhale_mask_warped = moving_mask_warped
            dz, dy, dx = map(float, spacing_zyx)
            disp_vox = disp_mm_3zyx.copy()
            disp_vox[0] /= np.float32(dz)
            disp_vox[1] /= np.float32(dy)
            disp_vox[2] /= np.float32(dx)
            jac = jac_ext.astype(np.float32, copy=False)

    disp_mm = disp_vox_to_mm(disp_vox, spacing_zyx)
    if "jac" not in locals():
        jac = jacobian_determinant_from_disp_mm(disp_mm, spacing_zyx)
    disp_mag = displacement_magnitude_mm(disp_mm)

    dice = dice_coefficient(inhale_mask, exhale_mask_warped)
    jac_idx = jaccard_index(inhale_mask, exhale_mask_warped)

    vol = compute_volume_change(
        fixed_mask_zyx=inhale_mask,
        moving_mask_zyx=exhale_mask_rs,
        jac_det=jac,
        spacing_zyx=spacing_zyx,
    )

    slice_z = _pick_slice_from_mask(inhale_mask)

    save_npz(
        pair_dir / "artifacts.npz",
        spacing_zyx=np.asarray(spacing_zyx, dtype=np.float32),
        inhale_ct_zyx=inhale_ct.astype(np.int16, copy=False),
        exhale_ct_resampled_zyx=exhale_ct.astype(np.int16, copy=False),
        exhale_ct_warped_zyx=exhale_ct_warped.astype(np.float32, copy=False),
        inhale_mask_zyx=inhale_mask.astype(np.uint8, copy=False),
        exhale_mask_resampled_zyx=exhale_mask_rs.astype(np.uint8, copy=False),
        exhale_mask_warped_zyx=exhale_mask_warped.astype(np.uint8, copy=False),
        disp_vox_3zyx=disp_vox.astype(np.float32, copy=False),
        disp_mm_3zyx=disp_mm.astype(np.float32, copy=False),
        jac_det_zyx=jac.astype(np.float32, copy=False),
        disp_mag_mm_zyx=disp_mag.astype(np.float32, copy=False),
        backend=np.asarray(backend_n),
    )

    if save_nifti_outputs:
        write_nifti_scalar_zyx(pair_dir / "jac_det.nii.gz", jac, spacing_zyx, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "disp_mag_mm.nii.gz", disp_mag, spacing_zyx, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "disp_z_mm.nii.gz", disp_mm[0], spacing_zyx, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "disp_y_mm.nii.gz", disp_mm[1], spacing_zyx, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "disp_x_mm.nii.gz", disp_mm[2], spacing_zyx, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "exhale_warped.nii.gz", exhale_ct_warped, spacing_zyx, dtype=np.float32)
        write_nifti_scalar_zyx(pair_dir / "exhale_mask_warped.nii.gz", exhale_mask_warped, spacing_zyx, dtype=np.uint8)

    row: Dict[str, object] = {
        "patient_id": pair.patient_id,
        "backend": backend_n,
        "phase_policy": policy,
        "swapped_vs_json_pair": bool(swapped),
        "inhale_image": str(inhale_path),
        "exhale_image": str(exhale_path),
        "inhale_mask": str(inhale_mask_path),
        "exhale_mask": str(exhale_mask_path),
        "spacing_z_mm": float(spacing_zyx[0]),
        "spacing_y_mm": float(spacing_zyx[1]),
        "spacing_x_mm": float(spacing_zyx[2]),
        "dice_mask": float(dice),
        "jaccard_mask": float(jac_idx),
        "inhale_vol_ml_gt": float(vol.fixed_vol_ml),
        "exhale_vol_ml_gt": float(vol.moving_vol_ml),
        "exhale_vol_ml_pred_jac": float(vol.pred_moving_vol_ml),
        "delta_vol_ml_gt_exhale_minus_inhale": float(vol.gt_delta_ml),
        "delta_vol_ml_pred_exhale_minus_inhale": float(vol.pred_delta_ml),
        "delta_vol_ml_error": float(vol.delta_error_ml),
        "exhale_vol_ml_error": float(vol.moving_vol_error_ml),
        "pair_dir": str(pair_dir),
    }

    return PairResult(
        row=row,
        fixed_ct_zyx=inhale_ct,
        moving_ct_warped_zyx=exhale_ct_warped,
        fixed_mask_zyx=inhale_mask,
        moving_mask_warped_zyx=exhale_mask_warped,
        disp_mag_mm_zyx=disp_mag,
        slice_z=int(slice_z),
    )
