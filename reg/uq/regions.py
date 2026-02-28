from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class RegionData:
    """
    Per-region values for one case, computed on the fixed grid.

    Regions are defined on the fixed mask (inhale for NLST/LungCT; ED mask for ACDC).

    moving_vol_ml: Eulerian occupancy in fixed regions (moving mask resampled to fixed grid).
    pred0_vol_ml: Lagrangian material volume proxy via Jacobian integral over fixed regions.
    X_region: region-restricted deformation features for regionbeta / other models.
    depth: normalized region depth in [0,1] (0=near boundary, 1=deep interior).
    """

    region_names: Tuple[str, ...]
    moving_vol_ml: np.ndarray  # (K,)
    pred0_vol_ml: np.ndarray  # (K,)
    X_region: np.ndarray  # (K, d)
    depth: np.ndarray  # (K,)

    @property
    def K(self) -> int:
        return int(len(self.region_names))


def _voxel_volume_ml(spacing_zyx: Tuple[float, float, float]) -> float:
    dz, dy, dx = map(float, spacing_zyx)
    return (dz * dy * dx) / 1000.0


def _disp_mag_mm(disp_mm_3zyx: np.ndarray | None) -> np.ndarray | None:
    if disp_mm_3zyx is None:
        return None
    u = np.asarray(disp_mm_3zyx, dtype=np.float32)
    if u.ndim != 4 or u.shape[0] != 3:
        return None
    return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).astype(np.float32, copy=False)


def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _region_features(
    *,
    jac_det_zyx: np.ndarray,
    disp_mag_mm_zyx: np.ndarray | None,
    region_mask: np.ndarray,
    feature_keys: Tuple[str, ...],
    eps: float = 1e-6,
) -> np.ndarray:
    m = np.asarray(region_mask) > 0
    jac = np.asarray(jac_det_zyx, dtype=np.float64)
    jac_m = jac[m]
    jac_m = jac_m[np.isfinite(jac_m)]
    jac_m = np.clip(jac_m, eps, 10.0)
    logj = np.log(jac_m) if jac_m.size else np.array([], dtype=np.float64)

    feat_map: Dict[str, float] = {}
    feat_map["logj_mean"] = float(np.mean(logj)) if logj.size else float("nan")
    feat_map["logj_std"] = float(np.std(logj)) if logj.size else float("nan")
    feat_map["mean_abs_logj"] = float(np.mean(np.abs(logj))) if logj.size else float("nan")
    feat_map["jac_p10"] = _safe_quantile(jac_m, 0.10)
    feat_map["jac_p50"] = _safe_quantile(jac_m, 0.50)
    feat_map["jac_p90"] = _safe_quantile(jac_m, 0.90)

    if disp_mag_mm_zyx is not None:
        disp = np.asarray(disp_mag_mm_zyx, dtype=np.float64)
        d = disp[m]
        d = d[np.isfinite(d)]
        feat_map["disp_mean_mm"] = float(np.mean(d)) if d.size else float("nan")
        feat_map["disp_p90_mm"] = _safe_quantile(d, 0.90)
        feat_map["disp_max_mm"] = float(np.max(d)) if d.size else float("nan")
    else:
        feat_map["disp_mean_mm"] = float("nan")
        feat_map["disp_p90_mm"] = float("nan")
        feat_map["disp_max_mm"] = float("nan")

    return np.array([float(feat_map.get(k, float("nan"))) for k in feature_keys], dtype=np.float32)


def radial_shell_regions(
    *,
    fixed_mask_zyx: np.ndarray,
    moving_mask_resampled_zyx: np.ndarray,
    jac_det_zyx: np.ndarray,
    disp_mm_3zyx: np.ndarray | None,
    spacing_zyx: Tuple[float, float, float],
    n_shells: int = 5,
    feature_keys: Tuple[str, ...] = (),
) -> RegionData:
    """
    Radial shells from distance transform inside the fixed mask.

    Shell edges are uniform in normalized distance-to-boundary (0..1), which aligns with
    the intuition of "expansion depth" for lungs.
    """
    import scipy.ndimage as ndi

    n_shells = int(max(1, n_shells))
    m = np.asarray(fixed_mask_zyx) > 0
    names = tuple(f"shell_{i}" for i in range(n_shells))
    if not np.any(m):
        return RegionData(
            region_names=names,
            moving_vol_ml=np.zeros(n_shells, np.float32),
            pred0_vol_ml=np.zeros(n_shells, np.float32),
            X_region=np.full((n_shells, len(feature_keys)), np.nan, dtype=np.float32),
            depth=np.linspace(0.0, 1.0, n_shells, dtype=np.float32),
        )

    dist = ndi.distance_transform_edt(m, sampling=tuple(map(float, spacing_zyx))).astype(np.float64)
    dmax = float(np.max(dist[m]))
    if not np.isfinite(dmax) or dmax <= 0:
        dmax = 1.0
    dn = np.zeros_like(dist, dtype=np.float64)
    dn[m] = dist[m] / dmax

    edges = np.linspace(0.0, 1.0 + 1e-6, n_shells + 1, dtype=np.float64)

    vml = float(_voxel_volume_ml(spacing_zyx))
    moving = np.asarray(moving_mask_resampled_zyx) > 0
    jac = np.asarray(jac_det_zyx, dtype=np.float64)
    disp_mag = _disp_mag_mm(disp_mm_3zyx)

    moving_vol = np.zeros(n_shells, dtype=np.float64)
    pred0_vol = np.zeros(n_shells, dtype=np.float64)
    X = np.full((n_shells, len(feature_keys)), np.nan, dtype=np.float32)
    depth = np.zeros(n_shells, dtype=np.float64)

    for k in range(n_shells):
        a, b = float(edges[k]), float(edges[k + 1])
        region = m & (dn >= a) & (dn < b)
        if not np.any(region):
            depth[k] = float((k + 0.5) / n_shells)
            continue
        moving_vol[k] = float(np.sum(region & moving)) * vml
        pred0_vol[k] = float(np.sum(jac[region])) * vml
        depth[k] = float((k + 0.5) / n_shells)
        if len(feature_keys) > 0:
            X[k] = _region_features(jac_det_zyx=jac_det_zyx, disp_mag_mm_zyx=disp_mag, region_mask=region, feature_keys=feature_keys)

    return RegionData(
        region_names=names,
        moving_vol_ml=moving_vol.astype(np.float32),
        pred0_vol_ml=pred0_vol.astype(np.float32),
        X_region=X,
        depth=depth.astype(np.float32),
    )


def compute_regions(
    *,
    region_def: str,
    fixed_mask_zyx: np.ndarray,
    moving_mask_resampled_zyx: np.ndarray,
    jac_det_zyx: np.ndarray,
    disp_mm_3zyx: np.ndarray | None,
    spacing_zyx: Tuple[float, float, float],
    n_shells: int,
    feature_keys: Tuple[str, ...] = (),
) -> RegionData:
    rd = str(region_def).lower().strip()
    if rd not in {"radial", "shell", "shells"}:
        raise ValueError("region_def must be 'radial' (radial shells).")
    return radial_shell_regions(
        fixed_mask_zyx=fixed_mask_zyx,
        moving_mask_resampled_zyx=moving_mask_resampled_zyx,
        jac_det_zyx=jac_det_zyx,
        disp_mm_3zyx=disp_mm_3zyx,
        spacing_zyx=spacing_zyx,
        n_shells=int(n_shells),
        feature_keys=feature_keys,
    )
