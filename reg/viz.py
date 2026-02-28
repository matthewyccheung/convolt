from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


def _imshow_ct(ax, ct_yx: np.ndarray, *, vmin: float = -1000.0, vmax: float = 400.0, title: str):
    ax.imshow(ct_yx, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def _overlay_mask_contour(ax, mask_yx: np.ndarray, *, color: str, linewidth: float = 1.0):
    m = (mask_yx > 0).astype(np.uint8)
    if m.max() == 0:
        return
    for contour in measure.find_contours(m, level=0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def _overlay_mask_alpha(ax, mask_yx: np.ndarray, *, color: Tuple[float, float, float], alpha: float):
    """
    Transparent mask overlay (so the image remains visible).
    color: RGB tuple in [0,1]
    """
    m = (np.asarray(mask_yx) > 0).astype(np.float32)
    if m.max() == 0:
        return
    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = float(color[0])
    rgba[..., 1] = float(color[1])
    rgba[..., 2] = float(color[2])
    rgba[..., 3] = m * float(alpha)
    ax.imshow(rgba, interpolation="nearest")


def _imshow_heat(ax, heat_yx: np.ndarray, *, title: str, cmap: str = "magma"):
    im = ax.imshow(heat_yx, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    return im


def save_patient_figure(
    *,
    out_path: str | Path,
    inhale_ct_zyx: np.ndarray,
    exhale_ct_warped_zyx: np.ndarray,
    inhale_mask_zyx: np.ndarray,
    exhale_mask_warped_zyx: np.ndarray,
    disp_mag_mm_zyx: np.ndarray,
    slice_z: int,
    patient_id: str,
    dice: float,
    vol_inhale_ml: float,
    vol_exhale_ml: float,
    pred_exhale_ml: float,
    delta_gt_exhale_minus_inhale_ml: float,
    delta_pred_exhale_minus_inhale_ml: float,
    delta_err_ml: float,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z = int(np.clip(slice_z, 0, inhale_ct_zyx.shape[0] - 1))

    fixed_yx = inhale_ct_zyx[z]
    warped_yx = exhale_ct_warped_zyx[z]
    fixed_m_yx = inhale_mask_zyx[z]
    warped_m_yx = exhale_mask_warped_zyx[z]
    disp_yx = disp_mag_mm_zyx[z]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    _imshow_ct(axes[0], fixed_yx, title=f"{patient_id} | Inhale (fixed)")
    _overlay_mask_alpha(axes[0], fixed_m_yx, color=(0.0, 1.0, 0.0), alpha=0.25)
    _overlay_mask_contour(axes[0], fixed_m_yx, color="lime")

    _imshow_ct(axes[1], warped_yx, title="Exhale (warped → inhale space)")
    _overlay_mask_alpha(axes[1], fixed_m_yx, color=(0.0, 1.0, 0.0), alpha=0.20)
    _overlay_mask_alpha(axes[1], warped_m_yx, color=(0.0, 1.0, 1.0), alpha=0.20)
    _overlay_mask_contour(axes[1], fixed_m_yx, color="lime")
    _overlay_mask_contour(axes[1], warped_m_yx, color="cyan")

    im = _imshow_heat(axes[2], disp_yx, title="|displacement| (mm)")
    _overlay_mask_alpha(axes[2], fixed_m_yx, color=(1.0, 1.0, 1.0), alpha=0.12)
    _overlay_mask_contour(axes[2], fixed_m_yx, color="white")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("mm")

    txt = (
        f"Dice(mask): {dice:.3f}\n"
        f"V_inhale GT: {vol_inhale_ml:.1f} mL\n"
        f"V_exhale GT: {vol_exhale_ml:.1f} mL\n"
        f"V_exhale Pred(J): {pred_exhale_ml:.1f} mL\n"
        f"ΔV GT (ex-in): {delta_gt_exhale_minus_inhale_ml:.1f} mL\n"
        f"ΔV Pred (ex-in): {delta_pred_exhale_minus_inhale_ml:.1f} mL\n"
        f"ΔV Error: {delta_err_ml:.1f} mL"
    )
    axes[2].text(
        0.02,
        0.98,
        txt,
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_pair_figure(
    *,
    out_path: str | Path,
    fixed_img_zyx: np.ndarray,
    moving_warped_zyx: np.ndarray,
    fixed_mask_zyx: np.ndarray,
    moving_mask_warped_zyx: np.ndarray,
    disp_mag_mm_zyx: np.ndarray,
    slice_z: int,
    title_fixed: str,
    title_moving: str,
    title_disp: str = "|displacement| (mm)",
    stats_text: str = "",
    ct_vmin: float | None = None,
    ct_vmax: float | None = None,
):
    """
    Generic 3-column figure:
      (1) fixed image + mask contour
      (2) moving warped into fixed space + contours
      (3) displacement magnitude heatmap + mask contour + optional stats text
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z = int(np.clip(slice_z, 0, fixed_img_zyx.shape[0] - 1))
    fixed_yx = fixed_img_zyx[z]
    warped_yx = moving_warped_zyx[z]
    fixed_m_yx = fixed_mask_zyx[z]
    warped_m_yx = moving_mask_warped_zyx[z]
    disp_yx = disp_mag_mm_zyx[z]

    vmin = ct_vmin
    vmax = ct_vmax
    if vmin is None:
        vmin = float(np.percentile(fixed_yx, 1.0))
    if vmax is None:
        vmax = float(np.percentile(fixed_yx, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    _imshow_ct(axes[0], fixed_yx, vmin=vmin, vmax=vmax, title=title_fixed)
    _overlay_mask_alpha(axes[0], fixed_m_yx, color=(0.0, 1.0, 0.0), alpha=0.25)
    _overlay_mask_contour(axes[0], fixed_m_yx, color="lime")

    _imshow_ct(axes[1], warped_yx, vmin=vmin, vmax=vmax, title=title_moving)
    _overlay_mask_alpha(axes[1], fixed_m_yx, color=(0.0, 1.0, 0.0), alpha=0.18)
    _overlay_mask_alpha(axes[1], warped_m_yx, color=(0.0, 1.0, 1.0), alpha=0.18)
    _overlay_mask_contour(axes[1], fixed_m_yx, color="lime")
    _overlay_mask_contour(axes[1], warped_m_yx, color="cyan")

    im = _imshow_heat(axes[2], disp_yx, title=title_disp)
    _overlay_mask_alpha(axes[2], fixed_m_yx, color=(1.0, 1.0, 1.0), alpha=0.12)
    _overlay_mask_contour(axes[2], fixed_m_yx, color="white")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("mm")

    if stats_text:
        axes[2].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
        )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
