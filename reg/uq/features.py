from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SpatialFeatures:
    values: Dict[str, float]


def _safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def extract_spatial_features(
    *,
    jac_det_zyx: np.ndarray,
    disp_mag_mm_zyx: np.ndarray | None,
    inhale_mask_zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float] | None = None,
    disp_mm_3zyx: np.ndarray | None = None,
    fixed_image_zyx: np.ndarray | None = None,
    moving_warped_zyx: np.ndarray | None = None,
    eps: float = 1e-6,
) -> SpatialFeatures:
    """
    Feature extractor intended for ConVOLD beta modeling / difficulty scoring.

    Uses only information available from the deformation field (Jacobian, displacement)
    restricted to the inhale lung mask.
    """
    m = inhale_mask_zyx > 0
    jac = jac_det_zyx.astype(np.float32, copy=False)
    jac_m = jac[m]
    jac_m = jac_m[np.isfinite(jac_m)]
    jac_m = np.clip(jac_m, eps, 10.0)
    logj = np.log(jac_m) if jac_m.size else np.array([], dtype=np.float32)

    feats: Dict[str, float] = {}
    feats["n_mask_vox"] = float(np.count_nonzero(m))

    feats["jac_mean"] = float(np.mean(jac_m)) if jac_m.size else float("nan")
    feats["jac_std"] = float(np.std(jac_m)) if jac_m.size else float("nan")
    feats["jac_min"] = float(np.min(jac_m)) if jac_m.size else float("nan")
    feats["jac_p01"] = _safe_quantile(jac_m, 0.01)
    feats["jac_p10"] = _safe_quantile(jac_m, 0.10)
    feats["jac_p50"] = _safe_quantile(jac_m, 0.50)
    feats["jac_p90"] = _safe_quantile(jac_m, 0.90)
    feats["jac_p99"] = _safe_quantile(jac_m, 0.99)
    feats["frac_jac_lt_1"] = float(np.mean(jac_m < 1.0)) if jac_m.size else float("nan")
    feats["frac_jac_gt_1"] = float(np.mean(jac_m > 1.0)) if jac_m.size else float("nan")
    feats["frac_jac_lt_01"] = float(np.mean(jac_m < 0.1)) if jac_m.size else float("nan")
    feats["frac_jac_lt_001"] = float(np.mean(jac_m < 0.01)) if jac_m.size else float("nan")

    feats["logj_mean"] = float(np.mean(logj)) if logj.size else float("nan")
    feats["logj_std"] = float(np.std(logj)) if logj.size else float("nan")
    feats["logj_p01"] = _safe_quantile(logj, 0.01)
    feats["logj_p10"] = _safe_quantile(logj, 0.10)
    feats["logj_p50"] = _safe_quantile(logj, 0.50)
    feats["logj_p90"] = _safe_quantile(logj, 0.90)
    feats["logj_p99"] = _safe_quantile(logj, 0.99)
    feats["mean_abs_logj"] = float(np.mean(np.abs(logj))) if logj.size else float("nan")

    if disp_mag_mm_zyx is not None:
        disp = disp_mag_mm_zyx.astype(np.float32, copy=False)
        disp_m = disp[m]
        disp_m = disp_m[np.isfinite(disp_m)]
        feats["disp_mean_mm"] = float(np.mean(disp_m)) if disp_m.size else float("nan")
        feats["disp_std_mm"] = float(np.std(disp_m)) if disp_m.size else float("nan")
        feats["disp_p90_mm"] = _safe_quantile(disp_m, 0.90)
        feats["disp_max_mm"] = float(np.max(disp_m)) if disp_m.size else float("nan")
    else:
        feats["disp_mean_mm"] = float("nan")
        feats["disp_std_mm"] = float("nan")
        feats["disp_p90_mm"] = float("nan")
        feats["disp_max_mm"] = float("nan")

    # Gradient magnitude of logJ: captures spatial roughness/heterogeneity.
    try:
        if feats["n_mask_vox"] > 0:
            dz, dy, dx = (1.0, 1.0, 1.0) if spacing_zyx is None else tuple(map(float, spacing_zyx))
            logj_field = np.log(np.clip(jac.astype(np.float64, copy=False), eps, 10.0))
            gz, gy, gx = np.gradient(logj_field, dz, dy, dx, edge_order=1)
            grad_mag = np.sqrt(gz * gz + gy * gy + gx * gx)
            gm = grad_mag[m]
            gm = gm[np.isfinite(gm)]
            feats["gradlogj_mean"] = float(np.mean(gm)) if gm.size else float("nan")
            feats["gradlogj_p90"] = _safe_quantile(gm, 0.90)
            feats["gradlogj_max"] = float(np.max(gm)) if gm.size else float("nan")
        else:
            feats["gradlogj_mean"] = float("nan")
            feats["gradlogj_p90"] = float("nan")
            feats["gradlogj_max"] = float("nan")
    except Exception:
        feats["gradlogj_mean"] = float("nan")
        feats["gradlogj_p90"] = float("nan")
        feats["gradlogj_max"] = float("nan")

    # Divergence/curl of displacement (if vector displacement is available).
    try:
        if disp_mm_3zyx is not None and feats["n_mask_vox"] > 0:
            u = np.asarray(disp_mm_3zyx, dtype=np.float64)
            if u.ndim == 4 and u.shape[0] == 3:
                dz, dy, dx = (1.0, 1.0, 1.0) if spacing_zyx is None else tuple(map(float, spacing_zyx))
                duz_dz = np.gradient(u[0], dz, axis=0, edge_order=1)
                duy_dy = np.gradient(u[1], dy, axis=1, edge_order=1)
                dux_dx = np.gradient(u[2], dx, axis=2, edge_order=1)
                div = duz_dz + duy_dy + dux_dx
                dv = div[m]
                dv = dv[np.isfinite(dv)]
                feats["div_mean"] = float(np.mean(dv)) if dv.size else float("nan")
                feats["div_std"] = float(np.std(dv)) if dv.size else float("nan")
                feats["div_p90_abs"] = _safe_quantile(np.abs(dv), 0.90) if dv.size else float("nan")

                dux_dy = np.gradient(u[2], dy, axis=1, edge_order=1)
                dux_dz = np.gradient(u[2], dz, axis=0, edge_order=1)
                duy_dx = np.gradient(u[1], dx, axis=2, edge_order=1)
                duy_dz = np.gradient(u[1], dz, axis=0, edge_order=1)
                duz_dx = np.gradient(u[0], dx, axis=2, edge_order=1)
                duz_dy = np.gradient(u[0], dy, axis=1, edge_order=1)
                curl_z = dux_dy - duy_dx
                curl_y = duz_dx - dux_dz
                curl_x = duy_dz - duz_dy
                curl_mag = np.sqrt(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z)
                cv = curl_mag[m]
                cv = cv[np.isfinite(cv)]
                feats["curl_mean"] = float(np.mean(cv)) if cv.size else float("nan")
                feats["curl_p90"] = _safe_quantile(cv, 0.90)
                feats["curl_max"] = float(np.max(cv)) if cv.size else float("nan")
            else:
                feats["div_mean"] = float("nan")
                feats["div_std"] = float("nan")
                feats["div_p90_abs"] = float("nan")
                feats["curl_mean"] = float("nan")
                feats["curl_p90"] = float("nan")
                feats["curl_max"] = float("nan")
        else:
            feats["div_mean"] = float("nan")
            feats["div_std"] = float("nan")
            feats["div_p90_abs"] = float("nan")
            feats["curl_mean"] = float("nan")
            feats["curl_p90"] = float("nan")
            feats["curl_max"] = float("nan")
    except Exception:
        feats["div_mean"] = float("nan")
        feats["div_std"] = float("nan")
        feats["div_p90_abs"] = float("nan")
        feats["curl_mean"] = float("nan")
        feats["curl_p90"] = float("nan")
        feats["curl_max"] = float("nan")

    # Similarity residuals in the fixed mask (fixed vs warped moving).
    try:
        if fixed_image_zyx is not None and moving_warped_zyx is not None and feats["n_mask_vox"] > 0:
            a = np.asarray(fixed_image_zyx, dtype=np.float64)[m]
            b = np.asarray(moving_warped_zyx, dtype=np.float64)[m]
            ok = np.isfinite(a) & np.isfinite(b)
            a = a[ok]
            b = b[ok]
            if a.size:
                d = a - b
                feats["sim_mae"] = float(np.mean(np.abs(d)))
                feats["sim_mse"] = float(np.mean(d * d))
                a0 = a - float(np.mean(a))
                b0 = b - float(np.mean(b))
                denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
                feats["sim_corr"] = float(np.sum(a0 * b0) / denom) if denom > 0 else float("nan")
            else:
                feats["sim_mae"] = float("nan")
                feats["sim_mse"] = float("nan")
                feats["sim_corr"] = float("nan")
        else:
            feats["sim_mae"] = float("nan")
            feats["sim_mse"] = float("nan")
            feats["sim_corr"] = float("nan")
    except Exception:
        feats["sim_mae"] = float("nan")
        feats["sim_mse"] = float("nan")
        feats["sim_corr"] = float("nan")

    # A single scalar "difficulty" score (spatial heterogeneity of deformation).
    feats["s"] = feats["logj_std"]

    return SpatialFeatures(values=feats)


def features_to_vector(feats: SpatialFeatures, keys: Tuple[str, ...]) -> np.ndarray:
    return np.array([float(feats.values.get(k, float("nan"))) for k in keys], dtype=np.float32)
