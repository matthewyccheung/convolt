from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def dice_coefficient(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)
    inter = np.count_nonzero(a & b)
    sa = np.count_nonzero(a)
    sb = np.count_nonzero(b)
    return float((2.0 * inter + eps) / (sa + sb + eps))


def jaccard_index(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float((inter + eps) / (union + eps))


def voxel_volume_mm3(spacing_zyx: Tuple[float, float, float]) -> float:
    dz, dy, dx = map(float, spacing_zyx)
    return dz * dy * dx


def mask_volume_ml(mask_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> float:
    mm3 = float(np.count_nonzero(mask_zyx > 0)) * voxel_volume_mm3(spacing_zyx)
    return mm3 / 1000.0


def predicted_moving_volume_ml_from_jacobian(
    jac_det: np.ndarray,
    fixed_mask_zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    *,
    jac_clip: Tuple[float, float] = (0.0, 5.0),
) -> float:
    """
    Predict moving (exhale) volume by integrating the Jacobian determinant over the fixed (inhale) lung mask:
      V_moving ≈ ∑_{x in lung_fixed} J(x) * dV_fixed
    """
    j = np.clip(jac_det.astype(np.float32, copy=False), float(jac_clip[0]), float(jac_clip[1]))
    m = (fixed_mask_zyx > 0).astype(np.float32)
    mm3 = float(np.sum(j * m)) * voxel_volume_mm3(spacing_zyx)
    return mm3 / 1000.0


@dataclass(frozen=True)
class VolumeChangeResult:
    fixed_vol_ml: float
    moving_vol_ml: float
    pred_moving_vol_ml: float
    gt_delta_ml: float
    pred_delta_ml: float
    delta_error_ml: float
    moving_vol_error_ml: float


def compute_volume_change(
    *,
    fixed_mask_zyx: np.ndarray,
    moving_mask_zyx: np.ndarray,
    jac_det: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
) -> VolumeChangeResult:
    fixed_vol_ml = mask_volume_ml(fixed_mask_zyx, spacing_zyx)
    moving_vol_ml = mask_volume_ml(moving_mask_zyx, spacing_zyx)
    pred_moving_vol_ml = predicted_moving_volume_ml_from_jacobian(jac_det, fixed_mask_zyx, spacing_zyx)

    gt_delta_ml = moving_vol_ml - fixed_vol_ml
    pred_delta_ml = pred_moving_vol_ml - fixed_vol_ml

    return VolumeChangeResult(
        fixed_vol_ml=fixed_vol_ml,
        moving_vol_ml=moving_vol_ml,
        pred_moving_vol_ml=pred_moving_vol_ml,
        gt_delta_ml=gt_delta_ml,
        pred_delta_ml=pred_delta_ml,
        delta_error_ml=pred_delta_ml - gt_delta_ml,
        moving_vol_error_ml=pred_moving_vol_ml - moving_vol_ml,
    )

