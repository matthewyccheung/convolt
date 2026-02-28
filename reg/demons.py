from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DemonsParams:
    levels: Tuple[int, ...] = (4, 2, 1)
    iterations: Tuple[int, ...] = (80, 40, 20)
    update_step: float = 1.0
    diffusion_sigma: float = 1.5
    # Intensity normalization for Demons similarity term.
    # - ct_hu: clamp to intensity_clip_hu then scale to [0,1]
    # - percentile: clamp to per-volume percentiles intensity_pct then scale
    # - minmax: scale by per-volume min/max
    intensity_norm: str = "ct_hu"  # ct_hu|percentile|minmax
    intensity_clip_hu: Tuple[float, float] = (-1000.0, 400.0)
    intensity_pct: Tuple[float, float] = (1.0, 99.0)
    mask_dilate_radius: int = 3
    eps: float = 1e-6


def normalize_ct_hu_to_0_1(ct_zyx: np.ndarray, clip_hu: Tuple[float, float]) -> np.ndarray:
    lo, hi = map(float, clip_hu)
    x = ct_zyx.astype(np.float32, copy=False)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x


def normalize_to_0_1(
    vol_zyx: np.ndarray,
    *,
    mode: str,
    ct_clip: Tuple[float, float],
    pct: Tuple[float, float],
) -> np.ndarray:
    """
    Normalize a 3D volume to [0,1] for Demons.
    """
    mode = str(mode).lower()
    x = vol_zyx.astype(np.float32, copy=False)
    if mode == "ct_hu":
        return normalize_ct_hu_to_0_1(x, ct_clip)
    if mode == "percentile":
        p_lo, p_hi = map(float, pct)
        lo = float(np.percentile(x, p_lo))
        hi = float(np.percentile(x, p_hi))
        if hi <= lo:
            hi = lo + 1.0
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo + 1e-6)
    if mode == "minmax":
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            hi = lo + 1.0
        return (x - lo) / (hi - lo + 1e-6)
    raise ValueError("intensity_norm must be one of: ct_hu, percentile, minmax")


def _center_crop_or_pad_zyx(arr: np.ndarray, target_shape_zyx: Tuple[int, int, int], pad_value: float) -> np.ndarray:
    tz, ty, tx = target_shape_zyx
    z, y, x = arr.shape
    out = arr

    # Crop if too large.
    if z > tz:
        start = (z - tz) // 2
        out = out[start : start + tz, :, :]
    if y > ty:
        start = (y - ty) // 2
        out = out[:, start : start + ty, :]
    if x > tx:
        start = (x - tx) // 2
        out = out[:, :, start : start + tx]

    # Pad if too small.
    z, y, x = out.shape
    pad_z0 = max((tz - z) // 2, 0)
    pad_y0 = max((ty - y) // 2, 0)
    pad_x0 = max((tx - x) // 2, 0)
    pad_z1 = max(tz - z - pad_z0, 0)
    pad_y1 = max(ty - y - pad_y0, 0)
    pad_x1 = max(tx - x - pad_x0, 0)
    if any(v > 0 for v in (pad_z0, pad_z1, pad_y0, pad_y1, pad_x0, pad_x1)):
        out = np.pad(
            out,
            ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode="constant",
            constant_values=pad_value,
        )
    return out


def resample_to_fixed_grid(
    moving_zyx: np.ndarray,
    moving_spacing_zyx: Tuple[float, float, float],
    fixed_shape_zyx: Tuple[int, int, int],
    fixed_spacing_zyx: Tuple[float, float, float],
    *,
    order: int,
    cval: float,
) -> np.ndarray:
    """
    Best-effort resampling of moving volume into the fixed grid using spacing ratios only.

    This assumes the dataset is already roughly aligned in orientation and origin (common in curated challenges).
    """
    zoom = tuple(float(s_in) / float(s_out) for s_in, s_out in zip(moving_spacing_zyx, fixed_spacing_zyx))
    rs = ndimage.zoom(moving_zyx, zoom=zoom, order=order, mode="constant", cval=cval)
    rs = _center_crop_or_pad_zyx(rs, fixed_shape_zyx, pad_value=cval)
    return rs


def warp_zyx(
    moving_zyx: np.ndarray,
    disp_vox_3zyx: np.ndarray,
    *,
    order: int,
    cval: float,
) -> np.ndarray:
    """
    Warp moving into the *fixed* grid using a displacement field defined on the fixed grid.

    disp_vox_3zyx is in voxel units, in Z,Y,X component order:
      z_moving = z_fixed + disp[0]
      y_moving = y_fixed + disp[1]
      x_moving = x_fixed + disp[2]
    """
    if disp_vox_3zyx.shape[0] != 3:
        raise ValueError("disp_vox_3zyx must have shape (3, Z, Y, X)")
    z, y, x = moving_zyx.shape
    zz, yy, xx = np.meshgrid(
        np.arange(z, dtype=np.float32),
        np.arange(y, dtype=np.float32),
        np.arange(x, dtype=np.float32),
        indexing="ij",
        sparse=False,
    )
    return warp_zyx_with_base(moving_zyx, disp_vox_3zyx, base_coords_zyx=(zz, yy, xx), order=order, cval=cval)


def warp_zyx_with_base(
    moving_zyx: np.ndarray,
    disp_vox_3zyx: np.ndarray,
    *,
    base_coords_zyx: tuple[np.ndarray, np.ndarray, np.ndarray],
    order: int,
    cval: float,
) -> np.ndarray:
    """
    Same as warp_zyx, but reuses a precomputed base coordinate grid for speed.
    """
    zz, yy, xx = base_coords_zyx
    coords = (zz + disp_vox_3zyx[0], yy + disp_vox_3zyx[1], xx + disp_vox_3zyx[2])
    return ndimage.map_coordinates(moving_zyx, coords, order=order, mode="constant", cval=cval)


def _build_mask_weight(fixed_mask_zyx: np.ndarray, radius: int) -> np.ndarray:
    if fixed_mask_zyx is None:
        return None
    m = (fixed_mask_zyx > 0).astype(np.float32)
    if radius > 0:
        m = ndimage.binary_dilation(m > 0, iterations=radius).astype(np.float32)
    return m


def _demons_single_level(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    disp_vox_3zyx: np.ndarray,
    *,
    n_iter: int,
    update_step: float,
    diffusion_sigma: float,
    fixed_mask_zyx: Optional[np.ndarray],
    mask_dilate_radius: int,
    eps: float,
) -> np.ndarray:
    fixed = fixed_zyx.astype(np.float32, copy=False)
    moving = moving_zyx.astype(np.float32, copy=False)

    fixed_grad = np.stack(np.gradient(fixed), axis=0).astype(np.float32)  # (3, z,y,x) in z,y,x order
    grad_sq = np.sum(fixed_grad * fixed_grad, axis=0) + eps

    weight = _build_mask_weight(fixed_mask_zyx, mask_dilate_radius) if fixed_mask_zyx is not None else None

    disp = disp_vox_3zyx.astype(np.float32, copy=True)

    z, y, x = fixed.shape
    zz, yy, xx = np.meshgrid(
        np.arange(z, dtype=np.float32),
        np.arange(y, dtype=np.float32),
        np.arange(x, dtype=np.float32),
        indexing="ij",
        sparse=False,
    )

    for _ in range(int(n_iter)):
        warped = warp_zyx_with_base(moving, disp, base_coords_zyx=(zz, yy, xx), order=1, cval=0.0)
        diff = fixed - warped

        denom = grad_sq + diff * diff + eps
        upd = (diff[None, ...] * fixed_grad) / denom[None, ...]

        if weight is not None:
            upd *= weight[None, ...]

        disp = disp + np.float32(update_step) * upd

        if diffusion_sigma and diffusion_sigma > 0:
            for c in range(3):
                disp[c] = ndimage.gaussian_filter(disp[c], sigma=float(diffusion_sigma), mode="nearest")

    return disp


def register_demons_multires(
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    *,
    fixed_mask_zyx: Optional[np.ndarray] = None,
    params: DemonsParams = DemonsParams(),
) -> np.ndarray:
    """
    Multi-resolution Demons registration.

    Returns disp_vox_3zyx defined on the fixed grid (voxel units).
    """
    if len(params.levels) != len(params.iterations):
        raise ValueError("params.levels and params.iterations must have the same length")

    fixed_n = normalize_to_0_1(
        fixed_zyx,
        mode=str(params.intensity_norm),
        ct_clip=params.intensity_clip_hu,
        pct=params.intensity_pct,
    )
    moving_n = normalize_to_0_1(
        moving_zyx,
        mode=str(params.intensity_norm),
        ct_clip=params.intensity_clip_hu,
        pct=params.intensity_pct,
    )

    disp: Optional[np.ndarray] = None
    for level, n_iter in zip(params.levels, params.iterations):
        shrink = int(level)
        if shrink <= 0:
            raise ValueError("levels must be positive integers")

        if shrink == 1:
            fx = fixed_n
            mv = moving_n
            msk = fixed_mask_zyx
        else:
            zoom = (1.0 / shrink, 1.0 / shrink, 1.0 / shrink)
            fx = ndimage.zoom(fixed_n, zoom=zoom, order=1, mode="nearest")
            mv = ndimage.zoom(moving_n, zoom=zoom, order=1, mode="nearest")
            msk = None
            if fixed_mask_zyx is not None:
                msk = ndimage.zoom((fixed_mask_zyx > 0).astype(np.float32), zoom=zoom, order=0, mode="nearest")

        if disp is None:
            disp_lvl = np.zeros((3,) + fx.shape, dtype=np.float32)
        else:
            # Upsample displacement from previous level and scale to the new resolution.
            scale = float(shrink_prev) / float(shrink)
            disp_lvl = np.stack(
                [
                    ndimage.zoom(disp[c], zoom=tuple(np.array(fx.shape) / np.array(disp.shape[1:])), order=1, mode="nearest")
                    for c in range(3)
                ],
                axis=0,
            ).astype(np.float32)
            disp_lvl *= np.float32(scale)

        disp = _demons_single_level(
            fx,
            mv,
            disp_lvl,
            n_iter=int(n_iter),
            update_step=float(params.update_step),
            diffusion_sigma=float(params.diffusion_sigma),
            fixed_mask_zyx=msk,
            mask_dilate_radius=int(params.mask_dilate_radius),
            eps=float(params.eps),
        )
        shrink_prev = shrink

    if disp is None:
        raise RuntimeError("Registration produced no displacement field")
    return disp


def disp_vox_to_mm(disp_vox_3zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> np.ndarray:
    dz, dy, dx = map(float, spacing_zyx)
    disp_mm = disp_vox_3zyx.astype(np.float32, copy=True)
    disp_mm[0] *= np.float32(dz)
    disp_mm[1] *= np.float32(dy)
    disp_mm[2] *= np.float32(dx)
    return disp_mm


def jacobian_determinant_from_disp_mm(
    disp_mm_3zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
) -> np.ndarray:
    """
    Jacobian determinant of the forward transform x_moving = x_fixed + u(x_fixed),
    where u is disp_mm_3zyx in mm, defined on the fixed grid.
    """
    if disp_mm_3zyx.shape[0] != 3:
        raise ValueError("disp_mm_3zyx must have shape (3, Z, Y, X)")

    dz, dy, dx = map(float, spacing_zyx)
    duz_dz, duz_dy, duz_dx = np.gradient(disp_mm_3zyx[0], dz, dy, dx)
    duy_dz, duy_dy, duy_dx = np.gradient(disp_mm_3zyx[1], dz, dy, dx)
    dux_dz, dux_dy, dux_dx = np.gradient(disp_mm_3zyx[2], dz, dy, dx)

    a11 = 1.0 + duz_dz
    a12 = duz_dy
    a13 = duz_dx
    a21 = duy_dz
    a22 = 1.0 + duy_dy
    a23 = duy_dx
    a31 = dux_dz
    a32 = dux_dy
    a33 = 1.0 + dux_dx

    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )
    return det.astype(np.float32)


def displacement_magnitude_mm(disp_mm_3zyx: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(disp_mm_3zyx * disp_mm_3zyx, axis=0)).astype(np.float32)
