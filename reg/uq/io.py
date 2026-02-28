from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import zipfile

from .features import SpatialFeatures, extract_spatial_features


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing required columns. Looked for: {candidates}")


@dataclass(frozen=True)
class LoadedResults:
    df: pd.DataFrame
    feature_keys: Tuple[str, ...]
    diagnostic_feature_keys: Tuple[str, ...]


def _standardize_summary_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize different dataset summary schemas to a common set of columns used by UQ:
      - inhale_vol_ml_gt
      - exhale_vol_ml_gt
      - exhale_vol_ml_pred0
      - delta_vol_ml_gt
      - delta_vol_ml_pred0
      - vol_union_ml_gt
      - vol_union_ml_pred0
      - patient_id
      - pair_dir

    Supports:
      - NLST-style outputs (inhale/exhale)
      - ACDC-style outputs (ED/ES)
      - Learn2Reg inter-patient atlas segmentation outputs (union volume)
    """
    # Learn2Reg schema detection (absolute volumes).
    # Some unlabeled/test summaries may omit GT columns; standardize what we can.
    if "union_vol_ml_pred" in df.columns or "union_vol_ml_gt" in df.columns:
        out = df.copy()
        if "union_vol_ml_gt" in out.columns:
            out["vol_union_ml_gt"] = out["union_vol_ml_gt"]
        elif "vol_union_ml_gt" not in out.columns:
            out["vol_union_ml_gt"] = np.nan
        if "union_vol_ml_pred" in out.columns:
            out["vol_union_ml_pred0"] = out["union_vol_ml_pred"]
        elif "vol_union_ml_pred0" not in out.columns:
            out["vol_union_ml_pred0"] = np.nan
        return out

    # ACDC schema detection
    if "edv_ml_gt" in df.columns and "esv_ml_pred_jac" in df.columns:
        exhale_gt_col = "esv_ml_gt_native" if "esv_ml_gt_native" in df.columns else "esv_ml_gt_fixedgrid"
        out = df.copy()
        out["inhale_vol_ml_gt"] = out["edv_ml_gt"]
        out["exhale_vol_ml_gt"] = out[exhale_gt_col]
        out["exhale_vol_ml_pred0"] = out["esv_ml_pred_jac"]
        if "delta_ml_gt_es_minus_ed" in out.columns:
            out["delta_vol_ml_gt"] = out["delta_ml_gt_es_minus_ed"]
        else:
            out["delta_vol_ml_gt"] = out["exhale_vol_ml_gt"] - out["inhale_vol_ml_gt"]
        if "delta_ml_pred_es_minus_ed" in out.columns:
            out["delta_vol_ml_pred0"] = out["delta_ml_pred_es_minus_ed"]
        else:
            out["delta_vol_ml_pred0"] = out["exhale_vol_ml_pred0"] - out["inhale_vol_ml_gt"]
        return out

    # NLST schema (current + legacy)
    inhale_vol_col = _find_col(df, ("inhale_vol_ml_gt", "fixed_vol_ml_gt"))
    exhale_vol_col = _find_col(df, ("exhale_vol_ml_gt", "moving_vol_ml_gt"))
    exhale_pred0_col = _find_col(df, ("exhale_vol_ml_pred_jac", "moving_vol_ml_pred_jac"))
    delta_gt_col = _find_col(df, ("delta_vol_ml_gt_exhale_minus_inhale", "delta_vol_ml_gt"))
    delta_pred0_col = _find_col(df, ("delta_vol_ml_pred_exhale_minus_inhale", "delta_vol_ml_pred"))

    out = df.rename(
        columns={
            inhale_vol_col: "inhale_vol_ml_gt",
            exhale_vol_col: "exhale_vol_ml_gt",
            exhale_pred0_col: "exhale_vol_ml_pred0",
            delta_gt_col: "delta_vol_ml_gt",
            delta_pred0_col: "delta_vol_ml_pred0",
        }
    )
    return out


def load_registration_results_with_features(
    *,
    results_dir: str | Path,
    require_artifacts: bool = True,
) -> LoadedResults:
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    df = pd.read_csv(summary_path)

    patient_col = _find_col(df, ("patient_id",))
    pair_dir_col = _find_col(df, ("pair_dir",))

    df = _standardize_summary_schema(df)

    # Extract spatial features from artifacts for each patient.
    feats_list: list[Dict[str, float]] = []
    missing = 0
    bad = 0
    for _, row in df.iterrows():
        pair_dir = Path(row[pair_dir_col])
        artifacts_path = pair_dir / "artifacts.npz"
        if not artifacts_path.exists():
            if require_artifacts:
                raise FileNotFoundError(f"Missing {artifacts_path}")
            missing += 1
            feats_list.append({})
            continue

        try:
            npz = np.load(artifacts_path, allow_pickle=True)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            bad += 1
            # Treat corrupted artifacts like missing features: keep the row but fill NaNs.
            # This commonly happens if registration was interrupted mid-write.
            print(f"WARNING: could not read artifacts {artifacts_path} ({type(e).__name__}: {e}); skipping features for this case.")
            feats_list.append({})
            continue
        # Handle naming differences across datasets/pipelines.
        # spacing
        try:
            if "spacing_zyx" in npz:
                spacing_zyx = tuple(map(float, npz["spacing_zyx"].tolist()))
            elif "fixed_spacing_zyx" in npz:
                spacing_zyx = tuple(map(float, npz["fixed_spacing_zyx"].tolist()))
            else:
                spacing_zyx = None

            if "roi_mask_zyx" in npz:
                inhale_mask = npz["roi_mask_zyx"]
            elif "inhale_mask_zyx" in npz:
                inhale_mask = npz["inhale_mask_zyx"]
            elif "fixed_mask_zyx" in npz:
                inhale_mask = npz["fixed_mask_zyx"]
            elif "ed_lv_mask_zyx" in npz:
                inhale_mask = npz["ed_lv_mask_zyx"]
            else:
                raise KeyError(f"{artifacts_path} missing a fixed mask key (roi_mask_zyx/inhale_mask_zyx/fixed_mask_zyx/ed_lv_mask_zyx)")
            jac = npz["jac_det_zyx"]
            disp_mag = npz["disp_mag_mm_zyx"] if "disp_mag_mm_zyx" in npz else None
            disp_mm = npz["disp_mm_3zyx"] if "disp_mm_3zyx" in npz else None

            # Fixed/warped moving images (for similarity residual features).
            fixed_img = None
            moving_warped = None
            if "inhale_ct_zyx" in npz:
                fixed_img = npz["inhale_ct_zyx"]
                moving_warped = npz["exhale_ct_warped_zyx"] if "exhale_ct_warped_zyx" in npz else None
            elif "ed_image_zyx" in npz:
                fixed_img = npz["ed_image_zyx"]
                moving_warped = npz["es_image_warped_zyx"] if "es_image_warped_zyx" in npz else None
            elif "fixed_image_zyx" in npz:
                fixed_img = npz["fixed_image_zyx"]
                moving_warped = npz["moving_warped_zyx"] if "moving_warped_zyx" in npz else None
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            bad += 1
            print(f"WARNING: could not extract arrays from {artifacts_path} ({type(e).__name__}: {e}); skipping features for this case.")
            feats_list.append({})
            continue

        feats = extract_spatial_features(
            jac_det_zyx=jac,
            disp_mag_mm_zyx=disp_mag,
            inhale_mask_zyx=inhale_mask,
            spacing_zyx=spacing_zyx,
            disp_mm_3zyx=disp_mm,
            fixed_image_zyx=fixed_img,
            moving_warped_zyx=moving_warped,
        )
        feats_list.append(feats.values)

    feats_df = pd.DataFrame(feats_list)
    out = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
    if bad > 0:
        print(f"UQ: WARNING: {bad} corrupted artifacts.npz files; features were skipped (set to NaN) for those cases.")

    # Choose a stable feature set for beta modeling (keep small by default).
    feature_keys = tuple(
        k
        for k in (
            "logj_mean",
            "logj_std",
            "mean_abs_logj",
            "jac_p10",
            "jac_p50",
            "jac_p90",
            "disp_mean_mm",
            "disp_p90_mm",
            "disp_max_mm",
        )
        if k in out.columns
    )

    # A richer feature pack for diagnostics (and optional future modeling).
    diagnostic_feature_keys = tuple(
        k
        for k in (
            # beta-model defaults
            *feature_keys,
            # folding / extremes
            "frac_jac_lt_01",
            "frac_jac_lt_001",
            "jac_min",
            "jac_p01",
            "jac_p99",
            "logj_p01",
            "logj_p99",
            # roughness
            "gradlogj_mean",
            "gradlogj_p90",
            "gradlogj_max",
            # displacement structure
            "div_mean",
            "div_std",
            "div_p90_abs",
            "curl_mean",
            "curl_p90",
            "curl_max",
            # similarity residuals
            "sim_mae",
            "sim_mse",
            "sim_corr",
        )
        if k in out.columns
    )

    return LoadedResults(df=out, feature_keys=feature_keys, diagnostic_feature_keys=diagnostic_feature_keys)
