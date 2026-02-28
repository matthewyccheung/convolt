from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from .conformal import (
    Interval,
    conformal_quantile,
    split_cp_symmetric,
    weighted_split_cp_symmetric,
)
from .io import LoadedResults, load_registration_results_with_features
from .models import QuantileRegressor, RidgeRegressor, fit_quantile_ridge, fit_ridge, fit_ridge_weighted
from .regions import RegionData, compute_regions
from .viz import (
    save_feature_diagnostics,
    save_jac_det_histograms_in_mask,
    save_nonconformity_histogram,
    save_region_diagnostics,
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conformal prediction UQ for volume targets + ConVOLT.")
    p.add_argument(
        "--dataset",
        choices=["nlst", "lungct", "acdc", "oasis"],
        default=None,
        help="If set with --method, uses standardized defaults for results/out dirs.",
    )
    p.add_argument(
        "--method",
        choices=["demons", "voxelmorph", "itk_elastix_bspline", "sitk_diffeomorphic_demons", "sitk_bspline"],
        default=None,
        help="If set with --dataset, uses standardized defaults for results/out dirs.",
    )
    p.add_argument(
        "--results_dir",
        default=None,
        type=Path,
        help="Registration output dir (contains summary.csv and pairs/*/artifacts.npz). If omitted, requires --dataset and --method.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        type=Path,
        help="Output dir for UQ tables/plots. If omitted, uses uq_results/{dataset}_{method} when --dataset/--method are set.",
    )
    # Learn2Reg atlas tag options (used for default results/out dirs).
    p.add_argument("--atlas_mode", choices=["multi", "single", "average"], default="multi")
    p.add_argument("--atlas_n", type=int, default=5)
    p.add_argument("--atlas_seed", type=int, default=0)

    p.add_argument(
        "--uq_target",
        choices=["delta_volume", "volume_union", "volume_label"],
        default=None,
        help="UQ target. Default: delta_volume for intra-patient datasets; union+topK labels for Learn2Reg inter-patient datasets.",
    )
    p.add_argument("--uq_topk_labels", type=int, default=3, help="For Learn2Reg: choose top-K labels by mean GT volume (exclude background).")
    p.add_argument("--uq_label_list", type=str, default="", help="Override label subset (comma-separated label IDs, e.g. 1,2,6).")
    p.add_argument(
        "--oasis_label_fullregion",
        action="store_true",
        help="OASIS-only: also compute ConVOLT label-local/label-hier variants using features pooled over the full predicted label region (no narrow band).",
    )
    p.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (e.g., 0.1 => 90pct intervals).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_repeats", type=int, default=1, help="Repeat split+calibration experiment N times (seed is offset by repeat index).")
    p.add_argument(
        "--save_intervals_each_repeat",
        action="store_true",
        help="If set (and n_repeats>1), also save per-patient test intervals for each repeat.",
    )

    p.add_argument("--split_mode", choices=["random"], default="random")
    p.add_argument("--frac_train", type=float, default=0.4)
    p.add_argument("--frac_calib", type=float, default=0.4)
    p.add_argument("--n_train", type=int, default=None, help="If set, override frac_train with an absolute count.")
    p.add_argument("--n_calib", type=int, default=None, help="If set, override frac_calib with an absolute count.")
    p.add_argument("--n_test", type=int, default=None, help="If set, use an absolute test count (remaining samples are dropped).")
    p.add_argument("--min_calib", type=int, default=5)
    p.add_argument("--min_test", type=int, default=5)

    # Historical flag name: --beta_model. We keep it for backwards compatibility, but it now controls the
    # *scale-CP* ConVOLT model that predicts k_hat(x) (a multiplicative scale in k-space).
    p.add_argument("--beta_model", choices=["none", "ridge"], default="ridge", help="Model for k_hat(x) in ConVOLT (scale-CP).")
    p.add_argument("--ridge_l2", type=float, default=1e-2)

    p.add_argument("--wcp_weight", choices=["abs_pred", "inhale_vol", "constant"], default="abs_pred")
    p.add_argument("--scp_local", action="store_true", help="Add LocalSCP baseline: kNN local quantile of |y-ŷ| in output space.")
    p.add_argument("--scp_knn_k", type=int, default=50, help="k for --scp_local.")
    p.add_argument(
        "--scp_local_s",
        choices=["abs_pred", "pred", "inhale_vol"],
        default="abs_pred",
        help="1D similarity score s(x) for LocalSCP kNN (computed from predictions/metadata).",
    )

    # Region-based guarantees (lung datasets).
    p.add_argument(
        "--region_defs",
        type=str,
        default="",
        help="Comma-separated region definitions for region-based guarantees. Supported: radial. Empty disables.",
    )
    p.add_argument(
        "--region_scores",
        type=str,
        default="q90",
        help="Comma-separated patient-level region score aggregations: q90 (recommended), max, mean.",
    )
    p.add_argument("--region_q", type=float, default=0.9, help="Within-patient quantile for q90 region score (default 0.9).")
    p.add_argument(
        "--patient_region_frac",
        type=float,
        default=0.9,
        help="For reporting: fraction of regions covered per patient (e.g. 0.9 means '>=90% regions covered').",
    )
    p.add_argument("--radial_shells", type=int, default=5, help="Number of radial shells for --region_defs radial (recommend 4-6).")
    p.add_argument(
        "--region_beta_mode",
        choices=["global", "region"],
        default="region",
        help="For RegionCP(ConVOLT-point): use global k_hat or region-wise k_hat learned from region features (recommended).",
    )

    p.add_argument(
        "--jac_hist_patients",
        type=int,
        default=6,
        help="Save jac_det-in-mask histograms for a few patients (top by frac(J<0.01)); set 0 to disable.",
    )
    return p.parse_args(argv)


def _make_splits(
    patient_ids: List[str],
    seed: int,
    frac_train: float,
    frac_calib: float,
    *,
    n_train: int | None = None,
    n_calib: int | None = None,
    n_test: int | None = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    ids = np.array(sorted(patient_ids))
    rng.shuffle(ids)
    n = ids.size
    if n_train is None:
        n_train_i = int(round(float(frac_train) * n))
    else:
        n_train_i = int(n_train)
    if n_calib is None:
        n_cal_i = int(round(float(frac_calib) * n))
    else:
        n_cal_i = int(n_calib)
    n_train_i = int(np.clip(n_train_i, 0, n))
    n_cal_i = int(np.clip(n_cal_i, 0, n - n_train_i))

    train = ids[:n_train_i]
    calib = ids[n_train_i : n_train_i + n_cal_i]

    rest = ids[n_train_i + n_cal_i :]
    if n_test is None:
        test = rest
    else:
        n_test_i = int(np.clip(int(n_test), 0, rest.size))
        test = rest[:n_test_i]
    return {"train": train, "calib": calib, "test": test}


def _weight_from_choice(
    df: pd.DataFrame,
    choice: str,
    *,
    pred_col: str,
    inhale_col: str | None = None,
) -> np.ndarray:
    choice = str(choice).lower()
    if choice == "abs_pred":
        return np.abs(df[pred_col].to_numpy(dtype=np.float32)) + 1.0
    if choice == "inhale_vol":
        if inhale_col is not None and inhale_col in df.columns:
            return df[inhale_col].to_numpy(dtype=np.float32) + 1.0
        # Fallback: behave like abs_pred when inhale volume isn't available.
        return np.abs(df[pred_col].to_numpy(dtype=np.float32)) + 1.0
    if choice == "constant":
        return np.ones(len(df), dtype=np.float32)
    raise ValueError(f"Unknown wcp_weight: {choice}")


def _scalar_s_from_choice(
    df: pd.DataFrame,
    choice: str,
    *,
    pred_col: str,
    inhale_col: str | None = None,
) -> np.ndarray:
    """
    1D score used to define neighborhoods for LocalSCP (output-space local CP).
    """
    choice = str(choice).lower()
    if choice == "abs_pred":
        return np.abs(df[pred_col].to_numpy(dtype=np.float32))
    if choice == "pred":
        return df[pred_col].to_numpy(dtype=np.float32)
    if choice == "inhale_vol":
        if inhale_col is not None and inhale_col in df.columns:
            return df[inhale_col].to_numpy(dtype=np.float32)
        return np.abs(df[pred_col].to_numpy(dtype=np.float32))
    raise ValueError(f"Unknown scp_local_s: {choice}")


def _region_feature_keys() -> Tuple[str, ...]:
    # Backwards-compatible feature set (same keys as global extract_spatial_features, but computed per region).
    return (
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


def _impute_feature_nans(X: np.ndarray) -> np.ndarray:
    """
    Replace non-finite entries with per-column means; if a column has no finite values, impute 0.
    Avoids RuntimeWarnings from np.nanmean on all-NaN slices.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.size == 0:
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    finite = np.isfinite(X)
    # Column means computed over finite entries; columns with no finite entries get mean 0.
    col_sum = np.where(finite, X, 0.0).sum(axis=0, keepdims=True)
    col_cnt = finite.sum(axis=0, keepdims=True).astype(np.float32)
    col_mean = np.divide(col_sum, np.maximum(col_cnt, 1.0), out=np.zeros_like(col_sum, dtype=np.float32), where=(col_cnt > 0))
    X = np.where(finite, X, col_mean)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _build_k_model(
    df_train: pd.DataFrame,
    feature_keys: Tuple[str, ...],
    *,
    ridge_l2: float,
    k_col: str = "k_true",
) -> RidgeRegressor:
    X = df_train.loc[:, list(feature_keys)].to_numpy(dtype=np.float32)
    y = df_train[k_col].to_numpy(dtype=np.float32)
    X = _impute_feature_nans(X)
    coef, intercept = fit_ridge(X, y, l2=float(ridge_l2))
    if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        intercept = float(1.0 if not np.isfinite(intercept) else intercept)
    return RidgeRegressor(feature_keys=feature_keys, coef_=coef, intercept_=intercept)


def _predict_k(model: RidgeRegressor | None, df: pd.DataFrame, feature_keys: Tuple[str, ...]) -> np.ndarray:
    if model is None:
        return np.ones(len(df), dtype=np.float32)
    X = df.loc[:, list(feature_keys)].to_numpy(dtype=np.float32)
    X = _impute_feature_nans(X)
    return model.predict(X).astype(np.float32)


def _build_additive_center_model(
    df_train: pd.DataFrame,
    feature_keys: Tuple[str, ...],
    *,
    ridge_l2: float,
    delta_col: str = "delta_true",
) -> RidgeRegressor:
    """
    Additive learned-center baseline:

      y = y0 + δ(x)

    Fit ridge to predict δ from features, then conformalize residuals in δ-space.
    """
    X = df_train.loc[:, list(feature_keys)].to_numpy(dtype=np.float32)
    y = df_train[delta_col].to_numpy(dtype=np.float32)
    X = _impute_feature_nans(X)
    coef, intercept = fit_ridge(X, y, l2=float(ridge_l2))
    if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        intercept = float(0.0 if not np.isfinite(intercept) else intercept)
    return RidgeRegressor(feature_keys=feature_keys, coef_=coef, intercept_=intercept)


def _predict_add_delta(model: RidgeRegressor | None, df: pd.DataFrame, feature_keys: Tuple[str, ...]) -> np.ndarray:
    if model is None:
        return np.zeros(len(df), dtype=np.float32)
    X = df.loc[:, list(feature_keys)].to_numpy(dtype=np.float32)
    X = _impute_feature_nans(X)
    return model.predict(X).astype(np.float32)


def _cqr_X(df: pd.DataFrame, *, pred_col: str, feature_keys: Tuple[str, ...]) -> np.ndarray:
    """
    CQR design matrix: always include the baseline prediction as a feature, then append spatial features if available.
    """
    x0 = df[pred_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    if len(feature_keys) == 0:
        return np.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    Xf = df.reindex(columns=list(feature_keys)).to_numpy(dtype=np.float32)
    Xf = _impute_feature_nans(Xf)
    X = np.concatenate([x0, Xf], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, float(q)))


def _oasis_label_local_feature_keys(*, include_fusion: bool) -> Tuple[str, ...]:
    # Keep this reasonably small/stable.
    keys = (
        "logj_mean",
        "logj_std",
        "mean_abs_logj",
        "jac_p10",
        "jac_p50",
        "jac_p90",
        "frac_jac_lt_01",
        "frac_jac_lt_001",
        "disp_mean_mm",
        "disp_p90_mm",
        "disp_max_mm",
        "gradlogj_mean",
        "gradlogj_p90",
        "gradlogj_max",
        "div_mean",
        "div_p90_abs",
        "curl_mean",
        "curl_p90",
        "curl_max",
        "sim_mae",
        "sim_mse",
        "sim_corr",
    )
    if include_fusion:
        keys = keys + ("vote_entropy_mean", "vote_maxfrac_mean")
    return keys


def _oasis_build_label_local_feature_cache(
    *,
    df: pd.DataFrame,
    label_ids: List[int],
    keys: Tuple[str, ...],
    dilate_iters: int,
    band: bool,
    include_fusion: bool,
    cache_path: Path,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute label-conditioned deformation-field feature vectors for OASIS:
      X[i,j,:] = features for patient i, label label_ids[j], computed inside a dilated/boundary band of pred_label.

    Returns:
      X: (N, L, D) float32 with NaNs for missing/empty masks
      patient_ids: list of patient IDs in the same order as X axis 0
    """
    label_ids = [int(x) for x in label_ids]
    patient_ids = df["patient_id"].astype(str).tolist()
    pair_dirs = df["pair_dir"].astype(str).tolist() if "pair_dir" in df.columns else ["" for _ in patient_ids]

    if cache_path.exists():
        try:
            npz = np.load(cache_path, allow_pickle=True)
            p = [str(x) for x in npz["patient_id"].tolist()]
            lids = [int(x) for x in npz["label_id"].tolist()]
            kk = tuple(str(x) for x in npz["feature_key"].tolist())
            X = npz["X"].astype(np.float32, copy=False)
            if p == patient_ids and lids == label_ids and kk == tuple(keys):
                return X, patient_ids
        except Exception:
            pass

    N = len(patient_ids)
    L = len(label_ids)
    D = len(keys)
    X = np.full((N, L, D), np.nan, dtype=np.float32)

    # Structure for dilation/erosion.
    structure = np.ones((3, 3, 3), dtype=bool)

    for i, (pid, pdir) in enumerate(zip(patient_ids, pair_dirs)):
        if not pdir:
            continue
        artifacts_path = Path(pdir) / "artifacts.npz"
        if not artifacts_path.exists():
            continue
        try:
            npz = np.load(artifacts_path, allow_pickle=True)
        except Exception:
            continue

        try:
            pred_label = np.asarray(npz["pred_label_zyx"])
            jac = np.asarray(npz["jac_det_zyx"], dtype=np.float64)
            disp_mag = np.asarray(npz["disp_mag_mm_zyx"], dtype=np.float64) if "disp_mag_mm_zyx" in npz else None
            disp = np.asarray(npz["disp_mm_3zyx"], dtype=np.float64) if "disp_mm_3zyx" in npz else None
            fixed = np.asarray(npz["fixed_image_zyx"], dtype=np.float64) if "fixed_image_zyx" in npz else None
            moved = np.asarray(npz["moving_warped_zyx"], dtype=np.float64) if "moving_warped_zyx" in npz else None
            spacing_zyx = tuple(map(float, npz["spacing_zyx"].tolist())) if "spacing_zyx" in npz else (1.0, 1.0, 1.0)
        except Exception:
            continue

        # Precompute fields used by multiple labels.
        jac_c = np.clip(jac, eps, 10.0)
        logj_field = np.log(jac_c)
        try:
            dz, dy, dx = tuple(map(float, spacing_zyx))
            gz, gy, gx = np.gradient(logj_field, dz, dy, dx, edge_order=1)
            grad_mag = np.sqrt(gz * gz + gy * gy + gx * gx)
        except Exception:
            grad_mag = None

        div = None
        curl_mag = None
        if disp is not None and disp.ndim == 4 and disp.shape[0] == 3:
            try:
                dz, dy, dx = tuple(map(float, spacing_zyx))
                duz_dz = np.gradient(disp[0], dz, axis=0, edge_order=1)
                duy_dy = np.gradient(disp[1], dy, axis=1, edge_order=1)
                dux_dx = np.gradient(disp[2], dx, axis=2, edge_order=1)
                div = duz_dz + duy_dy + dux_dx

                dux_dy = np.gradient(disp[2], dy, axis=1, edge_order=1)
                dux_dz = np.gradient(disp[2], dz, axis=0, edge_order=1)
                duy_dx = np.gradient(disp[1], dx, axis=2, edge_order=1)
                duy_dz = np.gradient(disp[1], dz, axis=0, edge_order=1)
                duz_dx = np.gradient(disp[0], dx, axis=2, edge_order=1)
                duz_dy = np.gradient(disp[0], dy, axis=1, edge_order=1)
                curl_z = dux_dy - duy_dx
                curl_y = duz_dx - dux_dz
                curl_x = duy_dz - duz_dy
                curl_mag = np.sqrt(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z)
            except Exception:
                div = None
                curl_mag = None

        vote_entropy = np.asarray(npz["vote_entropy_zyx"], dtype=np.float64) if include_fusion and "vote_entropy_zyx" in npz else None
        vote_maxfrac = np.asarray(npz["vote_maxfrac_zyx"], dtype=np.float64) if include_fusion and "vote_maxfrac_zyx" in npz else None

        for j, lid in enumerate(label_ids):
            m0 = pred_label == int(lid)
            if not np.any(m0):
                continue
            m = m0
            if int(dilate_iters) > 0:
                m = ndi.binary_dilation(m, structure=structure, iterations=int(dilate_iters))
                if band:
                    try:
                        er = ndi.binary_erosion(m0, structure=structure, iterations=int(dilate_iters))
                        if np.any(er):
                            m = np.logical_and(m, np.logical_not(er))
                    except Exception:
                        pass
            m = m.astype(bool)
            if not np.any(m):
                continue

            jac_m = jac_c[m]
            jac_m = jac_m[np.isfinite(jac_m)]
            logj_m = logj_field[m]
            logj_m = logj_m[np.isfinite(logj_m)]

            disp_m = None
            if disp_mag is not None:
                disp_m = disp_mag[m]
                disp_m = disp_m[np.isfinite(disp_m)]

            grad_m = None
            if grad_mag is not None:
                grad_m = grad_mag[m]
                grad_m = grad_m[np.isfinite(grad_m)]

            div_m = None
            if div is not None:
                div_m = div[m]
                div_m = div_m[np.isfinite(div_m)]

            curl_m = None
            if curl_mag is not None:
                curl_m = curl_mag[m]
                curl_m = curl_m[np.isfinite(curl_m)]

            # Similarity residuals.
            sim_mae = sim_mse = sim_corr = float("nan")
            if fixed is not None and moved is not None:
                a = fixed[m]
                b = moved[m]
                ok = np.isfinite(a) & np.isfinite(b)
                a = a[ok]
                b = b[ok]
                if a.size:
                    d_ = a - b
                    sim_mae = float(np.mean(np.abs(d_)))
                    sim_mse = float(np.mean(d_ * d_))
                    a0 = a - float(np.mean(a))
                    b0 = b - float(np.mean(b))
                    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
                    sim_corr = float(np.sum(a0 * b0) / denom) if denom > 0 else float("nan")

            feats: Dict[str, float] = {}
            feats["logj_mean"] = float(np.mean(logj_m)) if logj_m.size else float("nan")
            feats["logj_std"] = float(np.std(logj_m)) if logj_m.size else float("nan")
            feats["mean_abs_logj"] = float(np.mean(np.abs(logj_m))) if logj_m.size else float("nan")
            feats["jac_p10"] = _safe_quantile(jac_m, 0.10)
            feats["jac_p50"] = _safe_quantile(jac_m, 0.50)
            feats["jac_p90"] = _safe_quantile(jac_m, 0.90)
            feats["frac_jac_lt_01"] = float(np.mean(jac_m < 0.1)) if jac_m.size else float("nan")
            feats["frac_jac_lt_001"] = float(np.mean(jac_m < 0.01)) if jac_m.size else float("nan")

            feats["disp_mean_mm"] = float(np.mean(disp_m)) if disp_m is not None and disp_m.size else float("nan")
            feats["disp_p90_mm"] = _safe_quantile(disp_m, 0.90) if disp_m is not None and disp_m.size else float("nan")
            feats["disp_max_mm"] = float(np.max(disp_m)) if disp_m is not None and disp_m.size else float("nan")

            feats["gradlogj_mean"] = float(np.mean(grad_m)) if grad_m is not None and grad_m.size else float("nan")
            feats["gradlogj_p90"] = _safe_quantile(grad_m, 0.90) if grad_m is not None and grad_m.size else float("nan")
            feats["gradlogj_max"] = float(np.max(grad_m)) if grad_m is not None and grad_m.size else float("nan")

            feats["div_mean"] = float(np.mean(div_m)) if div_m is not None and div_m.size else float("nan")
            feats["div_p90_abs"] = _safe_quantile(np.abs(div_m), 0.90) if div_m is not None and div_m.size else float("nan")
            feats["curl_mean"] = float(np.mean(curl_m)) if curl_m is not None and curl_m.size else float("nan")
            feats["curl_p90"] = _safe_quantile(curl_m, 0.90) if curl_m is not None and curl_m.size else float("nan")
            feats["curl_max"] = float(np.max(curl_m)) if curl_m is not None and curl_m.size else float("nan")

            feats["sim_mae"] = float(sim_mae)
            feats["sim_mse"] = float(sim_mse)
            feats["sim_corr"] = float(sim_corr)

            if include_fusion:
                if vote_entropy is not None:
                    ve = vote_entropy[m]
                    ve = ve[np.isfinite(ve)]
                    feats["vote_entropy_mean"] = float(np.mean(ve)) if ve.size else float("nan")
                else:
                    feats["vote_entropy_mean"] = float("nan")
                if vote_maxfrac is not None:
                    vm = vote_maxfrac[m]
                    vm = vm[np.isfinite(vm)]
                    feats["vote_maxfrac_mean"] = float(np.mean(vm)) if vm.size else float("nan")
                else:
                    feats["vote_maxfrac_mean"] = float("nan")

            vec = np.array([float(feats.get(k, float("nan"))) for k in keys], dtype=np.float32)
            X[i, j, :] = vec

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, patient_id=np.asarray(patient_ids), label_id=np.asarray(label_ids), feature_key=np.asarray(keys), X=X)
    return X, patient_ids

def _scale_interval_from_k_volume(
    *,
    k_hat: np.ndarray,
    k_err_cal: np.ndarray,
    alpha: float,
    y_pred0_test: np.ndarray,
    min_k: float = 0.0,
) -> Interval:
    """
    Scale-CP mapping for absolute volume targets:

      V(k) = k * V0

    where V0 is the baseline prediction (e.g., Jacobian-integral volume).
    """
    k_hat = np.asarray(k_hat, dtype=np.float64).reshape(-1)
    k_err_cal = np.asarray(k_err_cal, dtype=np.float64).reshape(-1)
    y0 = np.asarray(y_pred0_test, dtype=np.float64).reshape(-1)
    if k_hat.shape != y0.shape:
        raise ValueError("k_hat and y_pred0_test must have the same shape")

    e = k_err_cal[np.isfinite(k_err_cal)]
    if e.size == 0:
        raise ValueError("No finite calibration k errors")
    q = conformal_quantile(np.abs(e), float(alpha))
    k_lo = np.clip(k_hat - q, float(min_k), np.inf)
    k_hi = np.clip(k_hat + q, float(min_k), np.inf)
    lo = (k_lo * y0).astype(np.float32)
    hi = (k_hi * y0).astype(np.float32)
    return Interval(lo=np.minimum(lo, hi), hi=np.maximum(lo, hi))


def _scale_interval_from_k_delta(
    *,
    k_hat: np.ndarray,
    k_err_cal: np.ndarray,
    alpha: float,
    exhale_pred0_ml: np.ndarray,
    inhale_vol_ml: np.ndarray,
    min_k: float = 0.0,
) -> Interval:
    """
    Scale-CP mapping for delta-volume targets:

      ΔV(k) = k * V_ex0 - V_inh

    where V_ex0 is baseline predicted exhale volume (Jacobian integral in fixed mask),
    and V_inh is GT inhale volume.
    """
    k_hat = np.asarray(k_hat, dtype=np.float64).reshape(-1)
    ex0 = np.asarray(exhale_pred0_ml, dtype=np.float64).reshape(-1)
    inh = np.asarray(inhale_vol_ml, dtype=np.float64).reshape(-1)
    if k_hat.shape != ex0.shape or k_hat.shape != inh.shape:
        raise ValueError("k_hat, exhale_pred0_ml, inhale_vol_ml must have the same shape")

    e = np.asarray(k_err_cal, dtype=np.float64).reshape(-1)
    e = e[np.isfinite(e)]
    if e.size == 0:
        raise ValueError("No finite calibration k errors")
    q = conformal_quantile(np.abs(e), float(alpha))
    k_lo = np.clip(k_hat - q, float(min_k), np.inf)
    k_hi = np.clip(k_hat + q, float(min_k), np.inf)
    lo = (k_lo * ex0 - inh).astype(np.float32)
    hi = (k_hi * ex0 - inh).astype(np.float32)
    return Interval(lo=np.minimum(lo, hi), hi=np.maximum(lo, hi))


def _run_once_volume(
    *,
    df: pd.DataFrame,
    loaded: LoadedResults,
    args: argparse.Namespace,
    seed: int,
    out_dir: Path,
    target: str,
    label_id: int | None,
    y_gt_col: str,
    y_pred0_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run split CP for an absolute-volume target.
    """
    splits = _make_splits(
        df["patient_id"].tolist(),
        seed,
        args.frac_train,
        args.frac_calib,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_test=args.n_test,
    )
    df_train = df[df["patient_id"].isin(splits["train"])].reset_index(drop=True)
    df_cal = df[df["patient_id"].isin(splits["calib"])].reset_index(drop=True)
    df_test = df[df["patient_id"].isin(splits["test"])].reset_index(drop=True)
    if len(df_cal) < int(args.min_calib) or len(df_test) < int(args.min_test):
        raise ValueError(f"Not enough data after splitting: train={len(df_train)} calib={len(df_cal)} test={len(df_test)}")

    y_cal = df_cal[y_gt_col].to_numpy(dtype=np.float32)
    yhat_cal = df_cal[y_pred0_col].to_numpy(dtype=np.float32)
    y_test = df_test[y_gt_col].to_numpy(dtype=np.float32)
    yhat_test = df_test[y_pred0_col].to_numpy(dtype=np.float32)

    # Filter non-finite samples (should be none for labeled Learn2Reg, but keep safe).
    ok_cal = np.isfinite(y_cal) & np.isfinite(yhat_cal)
    ok_test = np.isfinite(y_test) & np.isfinite(yhat_test)
    df_cal = df_cal.loc[ok_cal].reset_index(drop=True)
    df_test = df_test.loc[ok_test].reset_index(drop=True)
    y_cal = y_cal[ok_cal]
    yhat_cal = yhat_cal[ok_cal]
    y_test = y_test[ok_test]
    yhat_test = yhat_test[ok_test]
    if len(df_cal) < int(args.min_calib) or len(df_test) < int(args.min_test):
        raise ValueError(f"Not enough finite data after filtering: calib={len(df_cal)} test={len(df_test)}")

    scp_int, _scp_meta = split_cp_symmetric(yhat_cal=yhat_cal, y_cal=y_cal, yhat_test=yhat_test, alpha=float(args.alpha))
    local_scp_int = None
    local_scp_q_mean = None
    if bool(getattr(args, "scp_local", False)):
        from .conformal import local_quantile_1d

        s_cal = _scalar_s_from_choice(df_cal, str(args.scp_local_s), pred_col=y_pred0_col, inhale_col=None)
        s_test = _scalar_s_from_choice(df_test, str(args.scp_local_s), pred_col=y_pred0_col, inhale_col=None)
        r_cal = np.abs(y_cal - yhat_cal).astype(np.float32)
        q_test = local_quantile_1d(s_cal=s_cal, r_cal=r_cal, s_test=s_test, alpha=float(args.alpha), k=int(args.scp_knn_k))
        local_scp_int = Interval(lo=yhat_test - q_test, hi=yhat_test + q_test)
        local_scp_q_mean = float(np.mean(q_test)) if q_test.size else float("nan")
    w_cal = _weight_from_choice(df_cal, args.wcp_weight, pred_col=y_pred0_col, inhale_col=None)
    w_test = _weight_from_choice(df_test, args.wcp_weight, pred_col=y_pred0_col, inhale_col=None)
    wcp_int, _wcp_meta = weighted_split_cp_symmetric(
        yhat_cal=yhat_cal,
        y_cal=y_cal,
        yhat_test=yhat_test,
        w_cal=w_cal,
        w_test=w_test,
        alpha=float(args.alpha),
    )

    # CQR (Romano et al., 2019): conformalized quantile regression.
    # We report three variants:
    #  - CQR: learns quantiles of y using [y_pred0, spatial features]
    #  - CQR(volonly): learns quantiles of y using [y_pred0] only (feature ablation baseline)
    #  - CQR(k): learns quantiles of the multiplicative factor k = y/y0 using [y_pred0, spatial features],
    #            then maps bounds back via y = k*y0 (scale-CP style).
    cqr_int: Interval | None = None
    cqr_volonly_int: Interval | None = None
    cqr_k_int: Interval | None = None
    if len(df_train) > 0:
        tau_lo = float(args.alpha) / 2.0
        tau_hi = 1.0 - float(args.alpha) / 2.0
        y_tr = df_train[y_gt_col].to_numpy(dtype=np.float32)

        # (A) CQR with volumes + features.
        X_tr = _cqr_X(df_train, pred_col=y_pred0_col, feature_keys=loaded.feature_keys)
        ok_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
        if int(np.count_nonzero(ok_tr)) >= max(10, int(0.2 * len(df_train))):
            q_lo_model = fit_quantile_ridge(X_tr[ok_tr], y_tr[ok_tr], tau=tau_lo, l2=float(args.ridge_l2))
            q_hi_model = fit_quantile_ridge(X_tr[ok_tr], y_tr[ok_tr], tau=tau_hi, l2=float(args.ridge_l2))

            X_cal = _cqr_X(df_cal, pred_col=y_pred0_col, feature_keys=loaded.feature_keys)
            X_test = _cqr_X(df_test, pred_col=y_pred0_col, feature_keys=loaded.feature_keys)
            q_lo_cal = q_lo_model.predict(X_cal).astype(np.float32)
            q_hi_cal = q_hi_model.predict(X_cal).astype(np.float32)
            q_lo_test = q_lo_model.predict(X_test).astype(np.float32)
            q_hi_test = q_hi_model.predict(X_test).astype(np.float32)

            s_cal = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal).astype(np.float64)
            try:
                q = conformal_quantile(s_cal, float(args.alpha))
                cqr_int = Interval(lo=(q_lo_test - q).astype(np.float32), hi=(q_hi_test + q).astype(np.float32))
            except ValueError:
                cqr_int = None

        # (B) CQR with output volumes only (no spatial features).
        X_tr0 = _cqr_X(df_train, pred_col=y_pred0_col, feature_keys=tuple())
        ok_tr0 = np.isfinite(y_tr) & np.all(np.isfinite(X_tr0), axis=1)
        if int(np.count_nonzero(ok_tr0)) >= max(10, int(0.2 * len(df_train))):
            q_lo_model0 = fit_quantile_ridge(X_tr0[ok_tr0], y_tr[ok_tr0], tau=tau_lo, l2=float(args.ridge_l2))
            q_hi_model0 = fit_quantile_ridge(X_tr0[ok_tr0], y_tr[ok_tr0], tau=tau_hi, l2=float(args.ridge_l2))
            X_cal0 = _cqr_X(df_cal, pred_col=y_pred0_col, feature_keys=tuple())
            X_test0 = _cqr_X(df_test, pred_col=y_pred0_col, feature_keys=tuple())
            q_lo_cal0 = q_lo_model0.predict(X_cal0).astype(np.float32)
            q_hi_cal0 = q_hi_model0.predict(X_cal0).astype(np.float32)
            q_lo_test0 = q_lo_model0.predict(X_test0).astype(np.float32)
            q_hi_test0 = q_hi_model0.predict(X_test0).astype(np.float32)
            s_cal0 = np.maximum(q_lo_cal0 - y_cal, y_cal - q_hi_cal0).astype(np.float64)
            try:
                q0 = conformal_quantile(s_cal0, float(args.alpha))
                cqr_volonly_int = Interval(lo=(q_lo_test0 - q0).astype(np.float32), hi=(q_hi_test0 + q0).astype(np.float32))
            except ValueError:
                cqr_volonly_int = None

        # (C) CQR in k-space (scale-CP): fit quantiles of k_true = y/y0, then map back.
        eps = 1e-6
        y0_tr = df_train[y_pred0_col].to_numpy(dtype=np.float64)
        y0_cal = df_cal[y_pred0_col].to_numpy(dtype=np.float64)
        y0_test = df_test[y_pred0_col].to_numpy(dtype=np.float64)
        k_tr = ((df_train[y_gt_col].to_numpy(dtype=np.float64) + eps) / (y0_tr + eps)).astype(np.float32)
        k_cal = ((df_cal[y_gt_col].to_numpy(dtype=np.float64) + eps) / (y0_cal + eps)).astype(np.float32)

        X_trk = _cqr_X(df_train, pred_col=y_pred0_col, feature_keys=loaded.feature_keys)
        ok_trk = np.isfinite(k_tr) & np.all(np.isfinite(X_trk), axis=1)
        if int(np.count_nonzero(ok_trk)) >= max(10, int(0.2 * len(df_train))):
            k_lo_model = fit_quantile_ridge(X_trk[ok_trk], k_tr[ok_trk], tau=tau_lo, l2=float(args.ridge_l2))
            k_hi_model = fit_quantile_ridge(X_trk[ok_trk], k_tr[ok_trk], tau=tau_hi, l2=float(args.ridge_l2))
            X_calk = _cqr_X(df_cal, pred_col=y_pred0_col, feature_keys=loaded.feature_keys)
            X_testk = _cqr_X(df_test, pred_col=y_pred0_col, feature_keys=loaded.feature_keys)
            k_lo_cal = k_lo_model.predict(X_calk).astype(np.float32)
            k_hi_cal = k_hi_model.predict(X_calk).astype(np.float32)
            k_lo_test = k_lo_model.predict(X_testk).astype(np.float32)
            k_hi_test = k_hi_model.predict(X_testk).astype(np.float32)
            s_calk = np.maximum(k_lo_cal - k_cal, k_cal - k_hi_cal).astype(np.float64)
            try:
                qk = conformal_quantile(s_calk, float(args.alpha))
                k_lo = np.clip((k_lo_test.astype(np.float64) - qk), 0.0, np.inf)
                k_hi = np.clip((k_hi_test.astype(np.float64) + qk), 0.0, np.inf)
                lo = (k_lo * y0_test).astype(np.float32)
                hi = (k_hi * y0_test).astype(np.float32)
                cqr_k_int = Interval(lo=np.minimum(lo, hi), hi=np.maximum(lo, hi))
            except ValueError:
                cqr_k_int = None

    # Additive learned-center baseline: y = y0 + δ(x), conformal on δ residuals.
    add_int: Interval | None = None
    add_point_test = np.full_like(yhat_test, np.nan, dtype=np.float32)
    if args.beta_model == "ridge" and len(loaded.feature_keys) > 0 and len(df_train) > 0:
        df_train_a = df_train.copy()
        df_cal_a = df_cal.copy()
        df_test_a = df_test.copy()
        df_train_a["delta_true"] = (df_train[y_gt_col].to_numpy(dtype=np.float64) - df_train[y_pred0_col].to_numpy(dtype=np.float64)).astype(np.float32)
        df_cal_a["delta_true"] = (df_cal[y_gt_col].to_numpy(dtype=np.float64) - df_cal[y_pred0_col].to_numpy(dtype=np.float64)).astype(np.float32)

        add_model = _build_additive_center_model(df_train_a, loaded.feature_keys, ridge_l2=float(args.ridge_l2), delta_col="delta_true")
        delta_hat_cal = _predict_add_delta(add_model, df_cal_a, loaded.feature_keys)
        delta_hat_test = _predict_add_delta(add_model, df_test_a, loaded.feature_keys)
        delta_err_cal = (df_cal_a["delta_true"].to_numpy(dtype=np.float32) - delta_hat_cal).astype(np.float32)
        q_add = conformal_quantile(np.abs(delta_err_cal), float(args.alpha))
        add_point_test = (yhat_test + delta_hat_test).astype(np.float32)
        add_int = Interval(lo=add_point_test - float(q_add), hi=add_point_test + float(q_add))

    # Baseline prediction y0.
    y0_cal = df_cal[y_pred0_col].to_numpy(dtype=np.float64)
    y0_test = df_test[y_pred0_col].to_numpy(dtype=np.float64)

    # Scale-CP ablations in k-space: V = k * V0 (no exp()).
    eps = 1e-6
    k_true_train = ((df_train[y_gt_col].to_numpy(dtype=np.float64) + eps) / (df_train[y_pred0_col].to_numpy(dtype=np.float64) + eps)).astype(np.float32)
    k_true_cal = ((df_cal[y_gt_col].to_numpy(dtype=np.float64) + eps) / (df_cal[y_pred0_col].to_numpy(dtype=np.float64) + eps)).astype(np.float32)

    # global1: k_hat ≡ 1
    k_hat_one_test = np.ones(len(df_test), dtype=np.float32)
    k_err_one_cal = (k_true_cal - 1.0).astype(np.float32)
    scale_global_int = _scale_interval_from_k_volume(
        k_hat=k_hat_one_test,
        k_err_cal=k_err_one_cal,
        alpha=float(args.alpha),
        y_pred0_test=y0_test,
        min_k=0.0,
    )

    # constk: k_hat ≡ mean(k_true) on train split
    if k_true_train.size > 0:
        k_const = float(np.nanmean(k_true_train.astype(np.float64)))
        if not np.isfinite(k_const):
            k_const = 1.0
    else:
        k_const = 1.0
    k_hat_const_test = np.full(len(df_test), k_const, dtype=np.float32)
    k_err_const_cal = (k_true_cal - float(k_const)).astype(np.float32)
    scale_const_int = _scale_interval_from_k_volume(
        k_hat=k_hat_const_test,
        k_err_cal=k_err_const_cal,
        alpha=float(args.alpha),
        y_pred0_test=y0_test,
        min_k=0.0,
    )

    # ConVOLT(scale-CP): learn k_hat(x) directly (ridge), conformalize residuals in k-space.
    scale_ridge_int: Interval | None = None
    k_hat_ridge_test = np.full(len(df_test), np.nan, dtype=np.float32)
    if args.beta_model == "ridge" and len(loaded.feature_keys) > 0 and len(df_train) > 0:
        df_train_k = df_train.copy()
        df_train_k["k_true"] = k_true_train
        k_model = _build_k_model(df_train_k, loaded.feature_keys, ridge_l2=float(args.ridge_l2), k_col="k_true")
        k_hat_cal = _predict_k(k_model, df_cal, loaded.feature_keys)
        k_hat_ridge_test = _predict_k(k_model, df_test, loaded.feature_keys)
        k_err_cal = (k_true_cal - k_hat_cal).astype(np.float32)
        scale_ridge_int = _scale_interval_from_k_volume(
            k_hat=k_hat_ridge_test,
            k_err_cal=k_err_cal,
            alpha=float(args.alpha),
            y_pred0_test=y0_test,
            min_k=0.0,
        )

    # Optional diagnostic: compare nonconformity distributions (SCP vs ConVOLT point predictor in k-space).
    if bool(args.beta_model == "ridge") and len(df_train) > 0 and len(loaded.feature_keys) > 0:
        try:
            (out_dir / "histograms").mkdir(parents=True, exist_ok=True)
            fname = f"nonconformity_hist_{target}"
            if label_id is not None:
                fname += f"_label{int(label_id)}"
            fname += ".png"
            if "k_hat_cal" in locals():
                y_convot_cal = (k_hat_cal.astype(np.float64) * y0_cal).astype(np.float32)
                save_nonconformity_histogram(
                    out_path=out_dir / "histograms" / fname,
                    scp_scores=np.abs(y_cal - yhat_cal),
                    compass_scores=np.abs(y_cal - y_convot_cal),
                    title=f"Calibration nonconformity: SCP vs ConVOLT ({target}{'' if label_id is None else f', label {label_id}'})",
                    label_scp="SCP score |y - ŷ0| (mL)",
                    label_compass="ConVOLT score |y - ŷk| (mL)",
                )
        except Exception:
            pass

    def _summ(method: str, interval: Interval, y_test_: np.ndarray) -> Dict[str, object]:
        cov = interval.coverage(y_test_)
        size = interval.size()
        return {
            "target": str(target),
            "label_id": int(label_id) if label_id is not None else -1,
            "method": method,
            "alpha": float(args.alpha),
            "n_train": int(len(df_train)),
            "n_calib": int(len(df_cal)),
            "n_test": int(len(df_test)),
            "coverage": float(np.mean(cov)) if cov.size else float("nan"),
            "mean_interval_size_ml": float(np.mean(size)) if size.size else float("nan"),
            "std_interval_size_ml": float(np.std(size)) if size.size else float("nan"),
        }

    rows = [
        _summ("SCP(|err|)", scp_int, y_test),
        _summ(f"LocalSCP(|err|; s={args.scp_local_s}, k={int(args.scp_knn_k)})", local_scp_int, y_test) if local_scp_int is not None else None,
        _summ(f"wCP(|err|/w), w={args.wcp_weight}", wcp_int, y_test),
        _summ("CQR", cqr_int, y_test) if cqr_int is not None else None,
        _summ("CQR(volonly)", cqr_volonly_int, y_test) if cqr_volonly_int is not None else None,
        _summ("CQR(k)", cqr_k_int, y_test) if cqr_k_int is not None else None,
        _summ("ConVOLT(add-CP)", add_int, y_test) if add_int is not None else None,
        _summ("ConVOLT(scale-CP, global1)", scale_global_int, y_test),
        _summ("ConVOLT(scale-CP, constk)", scale_const_int, y_test),
        _summ("ConVOLT(scale-CP)", scale_ridge_int, y_test) if scale_ridge_int is not None else None,
    ]
    rows = [r for r in rows if r is not None]
    method_table = pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)

    intervals_df = pd.DataFrame(
        {
            "patient_id": df_test["patient_id"].to_numpy(),
            "target": str(target),
            "label_id": int(label_id) if label_id is not None else -1,
            "y_gt_ml": y_test,
            "y_pred0_ml": yhat_test,
            "scp_lo": scp_int.lo,
            "scp_hi": scp_int.hi,
            "lscp_lo": local_scp_int.lo if local_scp_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "lscp_hi": local_scp_int.hi if local_scp_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "wcp_lo": wcp_int.lo,
            "wcp_hi": wcp_int.hi,
            "cqr_lo": cqr_int.lo if cqr_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_hi": cqr_int.hi if cqr_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_volonly_lo": cqr_volonly_int.lo if cqr_volonly_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_volonly_hi": cqr_volonly_int.hi if cqr_volonly_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_k_lo": cqr_k_int.lo if cqr_k_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_k_hi": cqr_k_int.hi if cqr_k_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "add_lo": add_int.lo if add_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "add_hi": add_int.hi if add_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "add_point_ml": add_point_test,
            "scale_global_lo": scale_global_int.lo,
            "scale_global_hi": scale_global_int.hi,
            "scale_global_k_hat": k_hat_one_test,
            "scale_global_point_ml": (k_hat_one_test.astype(np.float64) * y0_test).astype(np.float32),
            "scale_const_lo": scale_const_int.lo,
            "scale_const_hi": scale_const_int.hi,
            "scale_const_k_hat": k_hat_const_test,
            "scale_const_point_ml": (k_hat_const_test.astype(np.float64) * y0_test).astype(np.float32),
            "scale_ridge_lo": scale_ridge_int.lo if scale_ridge_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "scale_ridge_hi": scale_ridge_int.hi if scale_ridge_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "scale_ridge_k_hat": k_hat_ridge_test,
            "scale_ridge_point_ml": (k_hat_ridge_test.astype(np.float64) * y0_test).astype(np.float32) if scale_ridge_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
        }
    )
    return method_table, intervals_df


@dataclass(frozen=True)
class RunOutputs:
    method_table: pd.DataFrame
    intervals_df: pd.DataFrame
    region_method_table: pd.DataFrame | None = None


def _oasis_label_convot_extra_methods(
    *,
    df_feat: pd.DataFrame,
    lv: pd.DataFrame,
    label_ids: List[int],
    args: argparse.Namespace,
    seed: int,
    out_dir_root: Path,
) -> pd.DataFrame:
    """
    Extra ConVOLT variants for OASIS per-label volumes:
      1) ConVOLT(scale-CP,label-local): per-label ridge on label-local features.
      2) ConVOLT(scale-CP,label-hier): shared ridge + label-specific (shrunk) intercepts (partial pooling).
      3) ConVOLT(scale-CP,label+fusion): same as (2) but with optional fusion disagreement features if present.

    These are appended to the per-label method table (CQR/SCP etc. still come from _run_once_volume).
    """
    if len(label_ids) == 0:
        return pd.DataFrame()
    if "patient_id" not in df_feat.columns or "pair_dir" not in df_feat.columns:
        return pd.DataFrame()

    # Patient-level splits (exchangeability unit = patient).
    pids = df_feat["patient_id"].dropna().astype(str).unique().tolist()
    splits = _make_splits(
        pids,
        int(seed),
        float(args.frac_train),
        float(args.frac_calib),
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_test=args.n_test,
    )
    train_set = set(map(str, splits["train"]))
    calib_set = set(map(str, splits["calib"]))
    test_set = set(map(str, splits["test"]))
    if len(calib_set) < int(args.min_calib) or len(test_set) < int(args.min_test):
        return pd.DataFrame()

    # Align patient order and build y/y0 matrices.
    dfp = df_feat.drop_duplicates(subset=["patient_id"]).copy()
    dfp["patient_id"] = dfp["patient_id"].astype(str)
    dfp = dfp[dfp["patient_id"].isin(pids)].reset_index(drop=True)
    patient_ids = dfp["patient_id"].astype(str).tolist()
    N = len(patient_ids)

    label_ids = [int(x) for x in label_ids]
    lv2 = lv.copy()
    lv2["patient_id"] = lv2["patient_id"].astype(str)
    lv2["label_id"] = lv2["label_id"].astype(int)
    lv2 = lv2[(lv2["patient_id"].isin(patient_ids)) & (lv2["label_id"].isin(label_ids))].copy()

    gt = (
        lv2.pivot_table(index="patient_id", columns="label_id", values="vol_ml_gt", aggfunc="first")
        .reindex(index=patient_ids, columns=label_ids)
        .to_numpy(dtype=np.float64)
    )
    y0 = (
        lv2.pivot_table(index="patient_id", columns="label_id", values="vol_ml_pred", aggfunc="first")
        .reindex(index=patient_ids, columns=label_ids)
        .to_numpy(dtype=np.float64)
    )
    if gt.size == 0 or y0.size == 0:
        return pd.DataFrame()

    eps = 1e-6
    k_true = ((gt + eps) / (y0 + eps)).astype(np.float32)

    # Feature caches (label-local; with/without fusion disagreement features).
    cache_dir = out_dir_root / "diagnostics"
    want_full = bool(getattr(args, "oasis_label_fullregion", False))
    keys_local = _oasis_label_local_feature_keys(include_fusion=False)
    X_local, p_order = _oasis_build_label_local_feature_cache(
        df=dfp,
        label_ids=label_ids,
        keys=keys_local,
        dilate_iters=2,
        band=True,
        include_fusion=False,
        cache_path=cache_dir / "oasis_label_local_feats_d2_band1.npz",
    )
    if p_order != patient_ids:
        # Should not happen; keep safe.
        return pd.DataFrame()

    keys_fusion = _oasis_label_local_feature_keys(include_fusion=True)
    X_fusion, p_order2 = _oasis_build_label_local_feature_cache(
        df=dfp,
        label_ids=label_ids,
        keys=keys_fusion,
        dilate_iters=2,
        band=True,
        include_fusion=True,
        cache_path=cache_dir / "oasis_label_localfusion_feats_d2_band1.npz",
    )
    if p_order2 != patient_ids:
        return pd.DataFrame()

    # Index masks.
    pid_arr = np.asarray(patient_ids, dtype=object)
    tr = np.isin(pid_arr, list(train_set))
    ca = np.isin(pid_arr, list(calib_set))
    te = np.isin(pid_arr, list(test_set))

    if int(np.count_nonzero(ca)) < int(args.min_calib) or int(np.count_nonzero(te)) < int(args.min_test):
        return pd.DataFrame()

    def _impute(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        if X.size == 0:
            return X, np.zeros((1, X.shape[-1]), dtype=np.float32)
        finite = np.isfinite(X)
        col_sum = np.where(finite, X, 0.0).sum(axis=0, keepdims=True)
        col_cnt = finite.sum(axis=0, keepdims=True).astype(np.float32)
        col_mean = np.divide(col_sum, np.maximum(col_cnt, 1.0), out=np.zeros_like(col_sum, dtype=np.float32), where=(col_cnt > 0))
        X2 = np.where(finite, X, col_mean)
        X2 = np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return X2, col_mean.astype(np.float32)

    def _rows_from_k_hat(
        *,
        method_name: str,
        k_hat_cal: np.ndarray,
        k_hat_test: np.ndarray,
    ) -> list[dict]:
        rows: list[dict] = []
        for j, lid in enumerate(label_ids):
            ktrue_c = k_true[ca, j].astype(np.float64)
            ktrue_t = k_true[te, j].astype(np.float64)
            y0_t = y0[te, j].astype(np.float64)
            y_t = gt[te, j].astype(np.float64)

            k_hat_c = k_hat_cal[ca, j].astype(np.float64)
            okc = np.isfinite(k_hat_c) & np.isfinite(ktrue_c)
            if int(np.count_nonzero(okc)) < int(args.min_calib):
                continue
            k_err_cal = (ktrue_c[okc] - k_hat_c[okc]).astype(np.float64)
            try:
                qk = conformal_quantile(np.abs(k_err_cal), float(args.alpha))
            except ValueError:
                continue

            k_hat_t = k_hat_test[te, j].astype(np.float64)
            okt = np.isfinite(k_hat_t) & np.isfinite(y0_t) & np.isfinite(y_t)
            if int(np.count_nonzero(okt)) < int(args.min_test):
                continue
            k_lo = np.clip(k_hat_t[okt] - float(qk), 0.0, np.inf)
            k_hi = np.clip(k_hat_t[okt] + float(qk), 0.0, np.inf)
            lo = k_lo * y0_t[okt]
            hi = k_hi * y0_t[okt]
            lo2 = np.minimum(lo, hi)
            hi2 = np.maximum(lo, hi)
            cov = float(np.mean((y_t[okt] >= lo2) & (y_t[okt] <= hi2)))
            widths = (hi2 - lo2).astype(np.float64)
            rows.append(
                {
                    "target": "volume_label",
                    "label_id": int(lid),
                    "method": method_name,
                    "alpha": float(args.alpha),
                    "n_train": int(np.count_nonzero(tr)),
                    "n_calib": int(np.count_nonzero(ca)),
                    "n_test": int(np.count_nonzero(te)),
                    "coverage": cov,
                    "mean_interval_size_ml": float(np.mean(widths)) if widths.size else float("nan"),
                    "std_interval_size_ml": float(np.std(widths)) if widths.size else float("nan"),
                }
            )
        return rows

    rows_all: list[dict] = []

    # (1) Per-label ridge on label-local features.
    k_hat_local = np.full((N, len(label_ids)), np.nan, dtype=np.float32)
    for j in range(len(label_ids)):
        Xtr = X_local[tr, j, :]
        ytr = k_true[tr, j]
        ok_y = np.isfinite(ytr)
        if int(np.count_nonzero(ok_y)) < 15:
            continue
        Xtr_i, col_mean = _impute(Xtr[ok_y])
        coef, intercept = fit_ridge(Xtr_i, ytr[ok_y], l2=float(args.ridge_l2))
        # Predict for all patients (then slice by calib/test masks inside _rows_from_k_hat).
        Xa = np.where(np.isfinite(X_local[:, j, :]), X_local[:, j, :], col_mean).astype(np.float32)
        k_hat_local[:, j] = (Xa @ coef.astype(np.float32) + float(intercept)).astype(np.float32)
    rows_all.extend(_rows_from_k_hat(method_name="ConVOLT(scale-CP,label-local)", k_hat_cal=k_hat_local, k_hat_test=k_hat_local))

    def _fit_hier(*, X: np.ndarray, include_fusion_tag: str) -> None:
        # Shared ridge across labels + shrunk per-label intercepts.
        Xtr = X[tr, :, :].reshape(-1, X.shape[-1])
        ytr = k_true[tr, :].reshape(-1)
        ntr = int(np.count_nonzero(tr))
        lbl = np.tile(np.arange(len(label_ids), dtype=int), ntr)
        # weights: each patient contributes 1/|labels| total weight.
        w = np.tile(np.full(len(label_ids), 1.0 / float(len(label_ids)), dtype=np.float64), ntr)
        ok = np.isfinite(ytr) & np.isfinite(w)
        if int(np.count_nonzero(ok)) < max(50, 10 * len(label_ids)):
            return
        Xtr_i, col_mean = _impute(Xtr[ok])
        coef, intercept = fit_ridge_weighted(Xtr_i, ytr[ok], w[ok], l2=float(args.ridge_l2))
        # Residual means per label (train), with simple shrinkage.
        pred_tr = (Xtr_i @ coef.astype(np.float32) + float(intercept)).astype(np.float64)
        resid = (ytr[ok].astype(np.float64) - pred_tr).astype(np.float64)
        lbl_ok = lbl[ok]
        deltas = np.zeros(len(label_ids), dtype=np.float64)
        counts = np.zeros(len(label_ids), dtype=np.float64)
        for jj in range(len(label_ids)):
            m = lbl_ok == jj
            if np.any(m):
                deltas[jj] = float(np.mean(resid[m]))
                counts[jj] = float(np.count_nonzero(m))
        lam = 50.0  # shrinkage strength (simple default)
        shrink = counts / (counts + lam)
        deltas = deltas * shrink

        def _pred_all() -> np.ndarray:
            Xb = X.reshape(-1, X.shape[-1]).astype(np.float32)  # (N*L, d)
            Xb = np.where(np.isfinite(Xb), Xb, col_mean).astype(np.float32)
            kb = (Xb @ coef.astype(np.float32) + float(intercept)).astype(np.float64)
            kb = kb + np.tile(deltas, N)
            kb = np.clip(kb, 0.0, np.inf)
            return kb.reshape(N, len(label_ids)).astype(np.float32)

        k_all = _pred_all()
        rows_all.extend(_rows_from_k_hat(method_name=f"ConVOLT(scale-CP,{include_fusion_tag})", k_hat_cal=k_all, k_hat_test=k_all))

    _fit_hier(X=X_local, include_fusion_tag="label-hier")
    _fit_hier(X=X_fusion, include_fusion_tag="label+fusion")

    if want_full:
        # Full-region pooling (no narrow band): features pooled over the dilated predicted label region itself.

        def _fit_hier_full(*, X: np.ndarray, tag: str) -> None:
            Xtr = X[tr, :, :].reshape(-1, X.shape[-1])
            ytr = k_true[tr, :].reshape(-1)
            ntr = int(np.count_nonzero(tr))
            lbl = np.tile(np.arange(len(label_ids), dtype=int), ntr)
            w = np.tile(np.full(len(label_ids), 1.0 / float(len(label_ids)), dtype=np.float64), ntr)
            ok = np.isfinite(ytr) & np.isfinite(w)
            if int(np.count_nonzero(ok)) < max(50, 10 * len(label_ids)):
                return
            Xtr_i, col_mean = _impute(Xtr[ok])
            coef, intercept = fit_ridge_weighted(Xtr_i, ytr[ok], w[ok], l2=float(args.ridge_l2))
            pred_tr = (Xtr_i @ coef.astype(np.float32) + float(intercept)).astype(np.float64)
            resid = (ytr[ok].astype(np.float64) - pred_tr).astype(np.float64)
            lbl_ok = lbl[ok]
            deltas = np.zeros(len(label_ids), dtype=np.float64)
            counts = np.zeros(len(label_ids), dtype=np.float64)
            for jj in range(len(label_ids)):
                m = lbl_ok == jj
                if np.any(m):
                    deltas[jj] = float(np.mean(resid[m]))
                    counts[jj] = float(np.count_nonzero(m))
            lam = 50.0
            shrink = counts / (counts + lam)
            deltas = deltas * shrink

            Xb = X.reshape(-1, X.shape[-1]).astype(np.float32)
            Xb = np.where(np.isfinite(Xb), Xb, col_mean).astype(np.float32)
            kb = (Xb @ coef.astype(np.float32) + float(intercept)).astype(np.float64)
            kb = kb + np.tile(deltas, N)
            kb = np.clip(kb, 0.0, np.inf)
            k_all = kb.reshape(N, len(label_ids)).astype(np.float32)
            rows_all.extend(_rows_from_k_hat(method_name=f"ConVOLT(scale-CP,{tag},full)", k_hat_cal=k_all, k_hat_test=k_all))

        X_local_full, p3 = _oasis_build_label_local_feature_cache(
            df=dfp,
            label_ids=label_ids,
            keys=keys_local,
            dilate_iters=2,
            band=False,
            include_fusion=False,
            cache_path=cache_dir / "oasis_label_local_feats_d2_full0.npz",
        )
        if p3 == patient_ids:
            k_hat_local_full = np.full((N, len(label_ids)), np.nan, dtype=np.float32)
            for j in range(len(label_ids)):
                Xtr = X_local_full[tr, j, :]
                ytr = k_true[tr, j]
                ok_y = np.isfinite(ytr)
                if int(np.count_nonzero(ok_y)) < 15:
                    continue
                Xtr_i, col_mean = _impute(Xtr[ok_y])
                coef, intercept = fit_ridge(Xtr_i, ytr[ok_y], l2=float(args.ridge_l2))
                Xa = np.where(np.isfinite(X_local_full[:, j, :]), X_local_full[:, j, :], col_mean).astype(np.float32)
                k_hat_local_full[:, j] = (Xa @ coef.astype(np.float32) + float(intercept)).astype(np.float32)
            rows_all.extend(_rows_from_k_hat(method_name="ConVOLT(scale-CP,label-local,full)", k_hat_cal=k_hat_local_full, k_hat_test=k_hat_local_full))
            _fit_hier_full(X=X_local_full, tag="label-hier")

        X_fusion_full, p4 = _oasis_build_label_local_feature_cache(
            df=dfp,
            label_ids=label_ids,
            keys=keys_fusion,
            dilate_iters=2,
            band=False,
            include_fusion=True,
            cache_path=cache_dir / "oasis_label_localfusion_feats_d2_full0.npz",
        )
        if p4 == patient_ids:
            _fit_hier_full(X=X_fusion_full, tag="label+fusion")

    return pd.DataFrame(rows_all).dropna(how="all").reset_index(drop=True)


def _run_once(
    *,
    df: pd.DataFrame,
    loaded: LoadedResults,
    args: argparse.Namespace,
    seed: int,
    out_dir: Path,
    region_cache: dict[str, dict[str, RegionData]] | None = None,
) -> RunOutputs:
    splits = _make_splits(
        df["patient_id"].tolist(),
        seed,
        args.frac_train,
        args.frac_calib,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_test=args.n_test,
    )
    df_train = df[df["patient_id"].isin(splits["train"])].reset_index(drop=True)
    df_cal = df[df["patient_id"].isin(splits["calib"])].reset_index(drop=True)
    df_test = df[df["patient_id"].isin(splits["test"])].reset_index(drop=True)
    if len(df_cal) < int(args.min_calib) or len(df_test) < int(args.min_test):
        raise ValueError(f"Not enough data after splitting: train={len(df_train)} calib={len(df_cal)} test={len(df_test)}")

    # Baseline SCP (scalar delta).
    y_cal = df_cal["delta_vol_ml_gt"].to_numpy(dtype=np.float32)
    yhat_cal = df_cal["delta_vol_ml_pred0"].to_numpy(dtype=np.float32)
    y_test = df_test["delta_vol_ml_gt"].to_numpy(dtype=np.float32)
    yhat_test = df_test["delta_vol_ml_pred0"].to_numpy(dtype=np.float32)

    scp_int, _scp_meta = split_cp_symmetric(yhat_cal=yhat_cal, y_cal=y_cal, yhat_test=yhat_test, alpha=float(args.alpha))

    # Output-space LocalSCP baseline (kNN local quantiles on a 1D scalar s(x)).
    local_scp_int = None
    if bool(getattr(args, "scp_local", False)):
        from .conformal import local_quantile_1d

        s_cal = _scalar_s_from_choice(df_cal, str(args.scp_local_s), pred_col="delta_vol_ml_pred0", inhale_col="inhale_vol_ml_gt")
        s_test = _scalar_s_from_choice(df_test, str(args.scp_local_s), pred_col="delta_vol_ml_pred0", inhale_col="inhale_vol_ml_gt")
        r_cal = np.abs(y_cal - yhat_cal).astype(np.float32)
        q_test = local_quantile_1d(s_cal=s_cal, r_cal=r_cal, s_test=s_test, alpha=float(args.alpha), k=int(args.scp_knn_k))
        local_scp_int = Interval(lo=yhat_test - q_test, hi=yhat_test + q_test)

    # Weighted CP in output space.
    w_cal = _weight_from_choice(df_cal, args.wcp_weight, pred_col="delta_vol_ml_pred0", inhale_col="inhale_vol_ml_gt")
    w_test = _weight_from_choice(df_test, args.wcp_weight, pred_col="delta_vol_ml_pred0", inhale_col="inhale_vol_ml_gt")
    wcp_int, _wcp_meta = weighted_split_cp_symmetric(
        yhat_cal=yhat_cal,
        y_cal=y_cal,
        yhat_test=yhat_test,
        w_cal=w_cal,
        w_test=w_test,
        alpha=float(args.alpha),
    )

    # CQR (Romano et al., 2019): conformalized quantile regression.
    # We report three variants:
    #  - CQR: learns quantiles of ΔV using [delta_pred0, spatial features]
    #  - CQR(volonly): learns quantiles of ΔV using [delta_pred0] only
    #  - CQR(k): learns quantiles of the multiplicative factor k = V_ex_gt/V_ex0 using [exhale_pred0, spatial features],
    #           then maps bounds back via ΔV(k)=k*V_ex0 - V_inh (scale-CP style).
    cqr_int: Interval | None = None
    cqr_volonly_int: Interval | None = None
    cqr_k_int: Interval | None = None
    if len(df_train) > 0:
        tau_lo = float(args.alpha) / 2.0
        tau_hi = 1.0 - float(args.alpha) / 2.0
        y_tr = df_train["delta_vol_ml_gt"].to_numpy(dtype=np.float32)

        # (A) CQR with delta + features.
        X_tr = _cqr_X(df_train, pred_col="delta_vol_ml_pred0", feature_keys=loaded.feature_keys)
        ok_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
        if int(np.count_nonzero(ok_tr)) >= max(10, int(0.2 * len(df_train))):
            q_lo_model = fit_quantile_ridge(X_tr[ok_tr], y_tr[ok_tr], tau=tau_lo, l2=float(args.ridge_l2))
            q_hi_model = fit_quantile_ridge(X_tr[ok_tr], y_tr[ok_tr], tau=tau_hi, l2=float(args.ridge_l2))
            X_cal = _cqr_X(df_cal, pred_col="delta_vol_ml_pred0", feature_keys=loaded.feature_keys)
            X_test = _cqr_X(df_test, pred_col="delta_vol_ml_pred0", feature_keys=loaded.feature_keys)
            q_lo_cal = q_lo_model.predict(X_cal).astype(np.float32)
            q_hi_cal = q_hi_model.predict(X_cal).astype(np.float32)
            q_lo_test = q_lo_model.predict(X_test).astype(np.float32)
            q_hi_test = q_hi_model.predict(X_test).astype(np.float32)
            s_cal = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal).astype(np.float64)
            try:
                q = conformal_quantile(s_cal, float(args.alpha))
                cqr_int = Interval(lo=(q_lo_test - q).astype(np.float32), hi=(q_hi_test + q).astype(np.float32))
            except ValueError:
                cqr_int = None

        # (B) CQR with delta only.
        X_tr0 = _cqr_X(df_train, pred_col="delta_vol_ml_pred0", feature_keys=tuple())
        ok_tr0 = np.isfinite(y_tr) & np.all(np.isfinite(X_tr0), axis=1)
        if int(np.count_nonzero(ok_tr0)) >= max(10, int(0.2 * len(df_train))):
            q_lo_model0 = fit_quantile_ridge(X_tr0[ok_tr0], y_tr[ok_tr0], tau=tau_lo, l2=float(args.ridge_l2))
            q_hi_model0 = fit_quantile_ridge(X_tr0[ok_tr0], y_tr[ok_tr0], tau=tau_hi, l2=float(args.ridge_l2))
            X_cal0 = _cqr_X(df_cal, pred_col="delta_vol_ml_pred0", feature_keys=tuple())
            X_test0 = _cqr_X(df_test, pred_col="delta_vol_ml_pred0", feature_keys=tuple())
            q_lo_cal0 = q_lo_model0.predict(X_cal0).astype(np.float32)
            q_hi_cal0 = q_hi_model0.predict(X_cal0).astype(np.float32)
            q_lo_test0 = q_lo_model0.predict(X_test0).astype(np.float32)
            q_hi_test0 = q_hi_model0.predict(X_test0).astype(np.float32)
            s_cal0 = np.maximum(q_lo_cal0 - y_cal, y_cal - q_hi_cal0).astype(np.float64)
            try:
                q0 = conformal_quantile(s_cal0, float(args.alpha))
                cqr_volonly_int = Interval(lo=(q_lo_test0 - q0).astype(np.float32), hi=(q_hi_test0 + q0).astype(np.float32))
            except ValueError:
                cqr_volonly_int = None

        # (C) CQR in k-space (scale-CP): fit quantiles of k_true = V_ex_gt/V_ex0, then map to ΔV.
        eps = 1e-6
        ex0_tr = df_train["exhale_vol_ml_pred0"].to_numpy(dtype=np.float64)
        ex0_cal = df_cal["exhale_vol_ml_pred0"].to_numpy(dtype=np.float64)
        ex0_test = df_test["exhale_vol_ml_pred0"].to_numpy(dtype=np.float64)
        inh_test = df_test["inhale_vol_ml_gt"].to_numpy(dtype=np.float64)
        k_tr = ((df_train["exhale_vol_ml_gt"].to_numpy(dtype=np.float64) + eps) / (ex0_tr + eps)).astype(np.float32)
        k_cal = ((df_cal["exhale_vol_ml_gt"].to_numpy(dtype=np.float64) + eps) / (ex0_cal + eps)).astype(np.float32)

        X_trk = _cqr_X(df_train, pred_col="exhale_vol_ml_pred0", feature_keys=loaded.feature_keys)
        ok_trk = np.isfinite(k_tr) & np.all(np.isfinite(X_trk), axis=1)
        if int(np.count_nonzero(ok_trk)) >= max(10, int(0.2 * len(df_train))):
            k_lo_model = fit_quantile_ridge(X_trk[ok_trk], k_tr[ok_trk], tau=tau_lo, l2=float(args.ridge_l2))
            k_hi_model = fit_quantile_ridge(X_trk[ok_trk], k_tr[ok_trk], tau=tau_hi, l2=float(args.ridge_l2))
            X_calk = _cqr_X(df_cal, pred_col="exhale_vol_ml_pred0", feature_keys=loaded.feature_keys)
            X_testk = _cqr_X(df_test, pred_col="exhale_vol_ml_pred0", feature_keys=loaded.feature_keys)
            k_lo_cal = k_lo_model.predict(X_calk).astype(np.float32)
            k_hi_cal = k_hi_model.predict(X_calk).astype(np.float32)
            k_lo_test = k_lo_model.predict(X_testk).astype(np.float32)
            k_hi_test = k_hi_model.predict(X_testk).astype(np.float32)
            s_calk = np.maximum(k_lo_cal - k_cal, k_cal - k_hi_cal).astype(np.float64)
            try:
                qk = conformal_quantile(s_calk, float(args.alpha))
                k_lo = np.clip((k_lo_test.astype(np.float64) - qk), 0.0, np.inf)
                k_hi = np.clip((k_hi_test.astype(np.float64) + qk), 0.0, np.inf)
                lo = (k_lo * ex0_test - inh_test).astype(np.float32)
                hi = (k_hi * ex0_test - inh_test).astype(np.float32)
                cqr_k_int = Interval(lo=np.minimum(lo, hi), hi=np.maximum(lo, hi))
            except ValueError:
                cqr_k_int = None

    # Additive learned-center baseline: y = y0 + δ(x), conformal on δ residuals.
    add_int: Interval | None = None
    add_point_test = np.full_like(yhat_test, np.nan, dtype=np.float32)
    if args.beta_model == "ridge" and len(loaded.feature_keys) > 0 and len(df_train) > 0:
        df_train_a = df_train.copy()
        df_cal_a = df_cal.copy()
        df_test_a = df_test.copy()
        df_train_a["delta_true"] = (df_train_a["delta_vol_ml_gt"].to_numpy(dtype=np.float64) - df_train_a["delta_vol_ml_pred0"].to_numpy(dtype=np.float64)).astype(
            np.float32
        )
        df_cal_a["delta_true"] = (df_cal_a["delta_vol_ml_gt"].to_numpy(dtype=np.float64) - df_cal_a["delta_vol_ml_pred0"].to_numpy(dtype=np.float64)).astype(np.float32)

        add_model = _build_additive_center_model(df_train_a, loaded.feature_keys, ridge_l2=float(args.ridge_l2), delta_col="delta_true")
        delta_hat_cal = _predict_add_delta(add_model, df_cal_a, loaded.feature_keys)
        delta_hat_test = _predict_add_delta(add_model, df_test_a, loaded.feature_keys)
        delta_err_cal = (df_cal_a["delta_true"].to_numpy(dtype=np.float32) - delta_hat_cal).astype(np.float32)
        q_add = conformal_quantile(np.abs(delta_err_cal), float(args.alpha))
        add_point_test = (yhat_test + delta_hat_test).astype(np.float32)
        add_int = Interval(lo=add_point_test - float(q_add), hi=add_point_test + float(q_add))

    # NOTE: ConVOLT beta-CP (exp(beta) scaling) has been removed; we use scale-CP (k-space) everywhere.

    # Scale-CP ablations in k-space: ΔV = k * V_ex0 - V_inh (no exp()).
    eps = 1e-6
    k_true_train = ((df_train["exhale_vol_ml_gt"].to_numpy(dtype=np.float64) + eps) / (df_train["exhale_vol_ml_pred0"].to_numpy(dtype=np.float64) + eps)).astype(
        np.float32
    )
    k_true_cal = ((df_cal["exhale_vol_ml_gt"].to_numpy(dtype=np.float64) + eps) / (df_cal["exhale_vol_ml_pred0"].to_numpy(dtype=np.float64) + eps)).astype(np.float32)

    ex0_test = df_test["exhale_vol_ml_pred0"].to_numpy(dtype=np.float64)
    inh_test = df_test["inhale_vol_ml_gt"].to_numpy(dtype=np.float64)

    # global1: k_hat ≡ 1
    k_hat_one_test = np.ones(len(df_test), dtype=np.float32)
    k_err_one_cal = (k_true_cal - 1.0).astype(np.float32)
    scale_global_int = _scale_interval_from_k_delta(
        k_hat=k_hat_one_test,
        k_err_cal=k_err_one_cal,
        alpha=float(args.alpha),
        exhale_pred0_ml=ex0_test,
        inhale_vol_ml=inh_test,
        min_k=0.0,
    )
    delta_scale_global_point = (k_hat_one_test.astype(np.float64) * ex0_test - inh_test).astype(np.float32)

    # constk: k_hat ≡ mean(k_true) on train split
    if k_true_train.size > 0:
        k_const = float(np.nanmean(k_true_train.astype(np.float64)))
        if not np.isfinite(k_const):
            k_const = 1.0
    else:
        k_const = 1.0
    k_hat_const_test = np.full(len(df_test), k_const, dtype=np.float32)
    k_err_const_cal = (k_true_cal - float(k_const)).astype(np.float32)
    scale_const_int = _scale_interval_from_k_delta(
        k_hat=k_hat_const_test,
        k_err_cal=k_err_const_cal,
        alpha=float(args.alpha),
        exhale_pred0_ml=ex0_test,
        inhale_vol_ml=inh_test,
        min_k=0.0,
    )
    delta_scale_const_point = (k_hat_const_test.astype(np.float64) * ex0_test - inh_test).astype(np.float32)

    # ConVOLT(scale-CP): learn k_hat(x) directly (ridge), conformalize residuals in k-space.
    scale_ridge_int: Interval | None = None
    k_hat_ridge_test = np.full(len(df_test), np.nan, dtype=np.float32)
    k_hat_ridge_cal = np.full(len(df_cal), np.nan, dtype=np.float32)
    k_hat_ridge_train = np.full(len(df_train), np.nan, dtype=np.float32)
    delta_scale_ridge_point = np.full(len(df_test), np.nan, dtype=np.float32)
    if args.beta_model == "ridge" and len(loaded.feature_keys) > 0 and len(df_train) > 0:
        df_train_k = df_train.copy()
        df_train_k["k_true"] = k_true_train
        k_model = _build_k_model(df_train_k, loaded.feature_keys, ridge_l2=float(args.ridge_l2), k_col="k_true")
        k_hat_ridge_train = _predict_k(k_model, df_train, loaded.feature_keys).astype(np.float32, copy=False)
        k_hat_cal = _predict_k(k_model, df_cal, loaded.feature_keys)
        k_hat_ridge_cal = k_hat_cal.astype(np.float32, copy=False)
        k_hat_ridge_test = _predict_k(k_model, df_test, loaded.feature_keys)
        k_err_cal = (k_true_cal - k_hat_cal).astype(np.float32)
        scale_ridge_int = _scale_interval_from_k_delta(
            k_hat=k_hat_ridge_test,
            k_err_cal=k_err_cal,
            alpha=float(args.alpha),
            exhale_pred0_ml=ex0_test,
            inhale_vol_ml=inh_test,
            min_k=0.0,
        )
        delta_scale_ridge_point = (k_hat_ridge_test.astype(np.float64) * ex0_test - inh_test).astype(np.float32)

    # NOTE: constant-beta (exp(beta)) ablation removed with beta-CP.

    def _summ(method: str, interval: Interval, y_test_: np.ndarray) -> Dict[str, object]:
        cov = interval.coverage(y_test_)
        size = interval.size()
        return {
            "method": method,
            "alpha": float(args.alpha),
            "n_train": int(len(df_train)),
            "n_calib": int(len(df_cal)),
            "n_test": int(len(df_test)),
            "coverage": float(np.mean(cov)) if cov.size else float("nan"),
            "mean_interval_size_ml": float(np.mean(size)) if size.size else float("nan"),
            "std_interval_size_ml": float(np.std(size)) if size.size else float("nan"),
        }

    rows = [
        _summ("SCP(|err|)", scp_int, y_test),
        _summ(f"LocalSCP(|err|; s={args.scp_local_s}, k={int(args.scp_knn_k)})", local_scp_int, y_test) if local_scp_int is not None else None,
        _summ(f"wCP(|err|/w), w={args.wcp_weight}", wcp_int, y_test),
        _summ("CQR", cqr_int, y_test) if cqr_int is not None else None,
        _summ("CQR(volonly)", cqr_volonly_int, y_test) if cqr_volonly_int is not None else None,
        _summ("CQR(k)", cqr_k_int, y_test) if cqr_k_int is not None else None,
        _summ("ConVOLT(add-CP)", add_int, y_test) if add_int is not None else None,
        _summ("ConVOLT(scale-CP, global1)", scale_global_int, y_test),
        _summ("ConVOLT(scale-CP, constk)", scale_const_int, y_test),
        _summ("ConVOLT(scale-CP)", scale_ridge_int, y_test) if scale_ridge_int is not None else None,
    ]

    rows = [r for r in rows if r is not None]
    method_table = pd.DataFrame(rows).dropna(how="all").reset_index(drop=True)

    intervals_df = pd.DataFrame(
        {
            "patient_id": df_test["patient_id"].to_numpy(),
            "delta_gt_ml": y_test,
            "delta_pred0_ml": yhat_test,
            "scp_lo": scp_int.lo,
            "scp_hi": scp_int.hi,
            "lscp_lo": local_scp_int.lo if local_scp_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "lscp_hi": local_scp_int.hi if local_scp_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "wcp_lo": wcp_int.lo,
            "wcp_hi": wcp_int.hi,
            "cqr_lo": cqr_int.lo if cqr_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_hi": cqr_int.hi if cqr_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_volonly_lo": cqr_volonly_int.lo if cqr_volonly_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_volonly_hi": cqr_volonly_int.hi if cqr_volonly_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_k_lo": cqr_k_int.lo if cqr_k_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "cqr_k_hi": cqr_k_int.hi if cqr_k_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "add_lo": add_int.lo if add_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "add_hi": add_int.hi if add_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "add_delta_point_ml": add_point_test,
            "scale_global_lo": scale_global_int.lo,
            "scale_global_hi": scale_global_int.hi,
            "scale_global_k_hat": k_hat_one_test,
            "scale_global_delta_point_ml": delta_scale_global_point,
            "scale_const_lo": scale_const_int.lo,
            "scale_const_hi": scale_const_int.hi,
            "scale_const_k_hat": k_hat_const_test,
            "scale_const_delta_point_ml": delta_scale_const_point,
            "scale_ridge_lo": scale_ridge_int.lo if scale_ridge_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "scale_ridge_hi": scale_ridge_int.hi if scale_ridge_int is not None else np.full_like(yhat_test, np.nan, dtype=np.float32),
            "scale_ridge_k_hat": k_hat_ridge_test,
            "scale_ridge_delta_point_ml": delta_scale_ridge_point,
        }
    )

    # NOTE: beta-grid sweep diagnostics removed with beta-CP.

    region_table = None
    region_defs = [s.strip().lower() for s in str(args.region_defs).split(",") if s.strip()]
    if region_cache is not None and len(region_defs) > 0:
        region_feature_keys = loaded.feature_keys
        score_fns = [s.strip().lower() for s in str(args.region_scores).split(",") if s.strip()]
        q_within = float(args.region_q)
        patient_frac = float(args.patient_region_frac)

        rows_r = []
        for rd in region_defs:
            # Only keep patients that have region data.
            cal_ids = df_cal["patient_id"].tolist()
            test_ids = df_test["patient_id"].tolist()
            if any(pid not in region_cache for pid in cal_ids + test_ids):
                continue

            rv_cal = [region_cache[pid][rd] for pid in cal_ids]
            rv_test = [region_cache[pid][rd] for pid in test_ids]
            K = int(rv_cal[0].K) if rv_cal else 0
            if K <= 0:
                continue

            mov_cal = np.stack([rv.moving_vol_ml for rv in rv_cal], axis=0).astype(np.float64)  # (Nc,K)
            pred0_cal = np.stack([rv.pred0_vol_ml for rv in rv_cal], axis=0).astype(np.float64)
            mov_test = np.stack([rv.moving_vol_ml for rv in rv_test], axis=0).astype(np.float64)
            pred0_test = np.stack([rv.pred0_vol_ml for rv in rv_test], axis=0).astype(np.float64)

            # Two point predictors: baseline (pred0) and ConVOLT-point.
            # Global k: k_hat_i * pred0_{i,k}
            # Region k: k_hat_{i,k} * pred0_{i,k}
            region_beta_mode = str(getattr(args, "region_beta_mode", "global")).lower()
            if region_beta_mode not in {"global", "region"}:
                raise ValueError("--region_beta_mode must be one of: global, region")

            k_hat_global_cal = np.nan_to_num(k_hat_ridge_cal.astype(np.float64), nan=1.0, posinf=1.0, neginf=1.0)
            k_hat_global_test = np.nan_to_num(k_hat_ridge_test.astype(np.float64), nan=1.0, posinf=1.0, neginf=1.0)

            if (
                region_beta_mode == "global"
                or args.beta_model != "ridge"
                or len(df_train) == 0
                or len(loaded.feature_keys) == 0
                or any(pid not in region_cache for pid in df_train["patient_id"].tolist())
            ):
                pred_comp_cal = k_hat_global_cal[:, None] * pred0_cal
                pred_comp_test = k_hat_global_test[:, None] * pred0_test
                beta_mode_tag = "globalk" if args.beta_model == "ridge" else "global1"
            else:
                # Fit a single ridge model per region_def using regions as samples (train patients only).
                train_ids = df_train["patient_id"].tolist()
                X_train = np.concatenate([region_cache[pid][rd].X_region for pid in train_ids], axis=0).astype(np.float32)
                y_train = []
                eps = 1e-6
                for pid in train_ids:
                    rvd = region_cache[pid][rd]
                    # Region-wise scale target: k_true = V_gt/V0. Filter non-finite targets below and fall back if needed.
                    y_train.append((rvd.moving_vol_ml.astype(np.float64) + eps) / (rvd.pred0_vol_ml.astype(np.float64) + eps))
                y_train = np.concatenate(y_train, axis=0).astype(np.float32)

                # Impute non-finite features by train column means; all-nonfinite columns become 0.
                finite = np.isfinite(X_train)
                col_sum = np.where(finite, X_train, 0.0).sum(axis=0, keepdims=True)
                col_cnt = finite.sum(axis=0, keepdims=True).astype(np.float32)
                col_mean = np.divide(col_sum, np.maximum(col_cnt, 1.0), out=np.zeros_like(col_sum, dtype=np.float32), where=(col_cnt > 0))
                X_train = np.where(finite, X_train, col_mean)
                # Weight each patient equally: each patient contributes weight 1/K per region.
                w_train = np.concatenate([np.full((int(region_cache[pid][rd].K),), 1.0 / float(region_cache[pid][rd].K), dtype=np.float64) for pid in train_ids], axis=0)
                # Filter any non-finite targets (e.g., degenerate V0).
                ok = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1) & np.isfinite(w_train)
                if int(np.count_nonzero(ok)) < max(10, int(0.1 * len(y_train))):
                    # Too few valid region targets -> fall back to global k for this region_def.
                    pred_comp_cal = k_hat_global_cal[:, None] * pred0_cal
                    pred_comp_test = k_hat_global_test[:, None] * pred0_test
                    beta_mode_tag = "globalk"
                else:
                    coef, intercept = fit_ridge_weighted(X_train[ok], y_train[ok], w_train[ok], l2=float(args.ridge_l2))
                    region_model = RidgeRegressor(feature_keys=region_feature_keys, coef_=coef, intercept_=intercept)

                    def _pred_region(ids: list[str]) -> np.ndarray:
                        X = np.stack([region_cache[pid][rd].X_region for pid in ids], axis=0).astype(np.float32)  # (N,K,d)
                        X2 = X.reshape(-1, X.shape[-1])
                        X2 = np.where(np.isfinite(X2), X2, col_mean)
                        k = region_model.predict(X2).astype(np.float64).reshape(X.shape[0], X.shape[1])
                        return np.clip(k, 0.0, np.inf)

                    k_hat_reg_cal = _pred_region(cal_ids)
                    k_hat_reg_test = _pred_region(test_ids)
                    pred_comp_cal = k_hat_reg_cal * pred0_cal
                    pred_comp_test = k_hat_reg_test * pred0_test
                    beta_mode_tag = "regionk"

            # Region-wise absolute errors in moving volume (mL); fixed cancels out.
            r_scp_cal = np.abs(mov_cal - pred0_cal)
            r_scp_test = np.abs(mov_test - pred0_test)
            r_cmp_cal = np.abs(mov_cal - pred_comp_cal)
            r_cmp_test = np.abs(mov_test - pred_comp_test)
            # Ablation: constant-k multiplicative correction in region space.
            pred_cst_cal = float(k_const) * pred0_cal
            pred_cst_test = float(k_const) * pred0_test
            r_cst_cal = np.abs(mov_cal - pred_cst_cal)
            r_cst_test = np.abs(mov_test - pred_cst_test)

            # Region-CQR(volonly): fit per-region conditional quantiles using [pred0_vol_ml] only, then
            # conformalize the aggregated per-patient violation scores (same aggregation as RegionCP).
            r_cqr_cal = None
            r_cqr_test = None
            qlo_test = None
            qhi_test = None
            if len(df_train) > 0:
                train_ids = df_train["patient_id"].tolist()
                if all(pid in region_cache and rd in region_cache[pid] for pid in train_ids):
                    y_tr = np.concatenate([region_cache[pid][rd].moving_vol_ml for pid in train_ids], axis=0).astype(np.float32)
                    x_tr = np.concatenate([region_cache[pid][rd].pred0_vol_ml for pid in train_ids], axis=0).astype(np.float32)
                    X_tr = x_tr.reshape(-1, 1)
                    ok_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
                    if int(np.count_nonzero(ok_tr)) >= max(10, int(0.2 * len(y_tr))):
                        tau_lo = float(args.alpha) / 2.0
                        tau_hi = 1.0 - float(args.alpha) / 2.0
                        q_lo_model = fit_quantile_ridge(X_tr[ok_tr], y_tr[ok_tr], tau=tau_lo, l2=float(args.ridge_l2))
                        q_hi_model = fit_quantile_ridge(X_tr[ok_tr], y_tr[ok_tr], tau=tau_hi, l2=float(args.ridge_l2))

                        X_cal = pred0_cal.reshape(-1, 1).astype(np.float32)
                        X_test = pred0_test.reshape(-1, 1).astype(np.float32)
                        qlo_cal = q_lo_model.predict(X_cal).astype(np.float32).reshape(pred0_cal.shape)
                        qhi_cal = q_hi_model.predict(X_cal).astype(np.float32).reshape(pred0_cal.shape)
                        qlo_test = q_lo_model.predict(X_test).astype(np.float32).reshape(pred0_test.shape)
                        qhi_test = q_hi_model.predict(X_test).astype(np.float32).reshape(pred0_test.shape)

                        r_cqr_cal = np.maximum(qlo_cal - mov_cal, mov_cal - qhi_cal).astype(np.float32)
                        r_cqr_test = np.maximum(qlo_test - mov_test, mov_test - qhi_test).astype(np.float32)

            # Diagnostics (repeat 0 only): pooled region residual vs depth.
            if seed == int(args.seed) and len(rv_test) > 0:
                depth_test = np.concatenate([rv.depth for rv in rv_test], axis=0)
                save_region_diagnostics(
                    out_dir=out_dir,
                    region_def=rd,
                    depth=depth_test,
                    r_scp=r_scp_test.reshape(-1),
                    r_cmp=r_cmp_test.reshape(-1),
                    title_prefix="Test ",
                )
            # Normalized-residual RegionCP was removed (not used in the paper/benchmark).

            def _agg(r: np.ndarray, mode: str) -> np.ndarray:
                mode = str(mode).lower()
                if mode == "max":
                    return np.max(r, axis=1)
                if mode == "q90" or mode == "q":
                    return np.quantile(r, q_within, axis=1)
                if mode == "mean" or mode == "avg":
                    return np.mean(r, axis=1)
                raise ValueError("region score must be one of: max, q90, mean")

            for score_mode in score_fns:
                # SCP-point region guarantee
                s_cal = _agg(r_scp_cal, score_mode)
                try:
                    q = conformal_quantile(s_cal, float(args.alpha))
                except ValueError:
                    # No finite calibration scores (can happen if all volumes are NaN); skip this configuration.
                    continue
                s_test = _agg(r_scp_test, score_mode)

                cov_patient = float(np.mean(s_test <= q))
                cov_region = float(np.mean((r_scp_test <= q).reshape(-1)))
                frac_per_patient = np.mean(r_scp_test <= q, axis=1)
                cov_patient_frac = float(np.mean(frac_per_patient >= patient_frac))

                rows_r.append(
                    {
                        "method": f"RegionCP(SCP-point, {score_mode})",
                        "alpha": float(args.alpha),
                        "region_def": rd,
                        "region_score": score_mode,
                        "K": int(K),
                        "q_ml": float(q),
                        "coverage_patient": cov_patient,
                        "coverage_region_pooled": cov_region,
                        "coverage_patient_frac_regions": cov_patient_frac,
                        "mean_interval_size_ml": float(2.0 * q),
                    }
                )

                # ConVOLT-point region guarantee (beta_model affects beta_hat)
                s_cal = _agg(r_cmp_cal, score_mode)
                try:
                    q = conformal_quantile(s_cal, float(args.alpha))
                except ValueError:
                    continue
                s_test = _agg(r_cmp_test, score_mode)

                cov_patient = float(np.mean(s_test <= q))
                cov_region = float(np.mean((r_cmp_test <= q).reshape(-1)))
                frac_per_patient = np.mean(r_cmp_test <= q, axis=1)
                cov_patient_frac = float(np.mean(frac_per_patient >= patient_frac))

                rows_r.append(
                    {
                        "method": f"RegionCP(ConVOLT-point-{beta_mode_tag}, {score_mode})",
                        "alpha": float(args.alpha),
                        "region_def": rd,
                        "region_score": score_mode,
                        "K": int(K),
                        "q_ml": float(q),
                        "coverage_patient": cov_patient,
                        "coverage_region_pooled": cov_region,
                        "coverage_patient_frac_regions": cov_patient_frac,
                        "mean_interval_size_ml": float(2.0 * q),
                    }
                )

                # Constant-beta ablation (no spatial learning, just one global multiplicative correction).
                s_cal_cst = _agg(r_cst_cal, score_mode)
                try:
                    q_cst = conformal_quantile(s_cal_cst, float(args.alpha))
                except ValueError:
                    q_cst = None
                if q_cst is not None:
                    s_test_cst = _agg(r_cst_test, score_mode)
                    cov_patient = float(np.mean(s_test_cst <= q_cst))
                    cov_region = float(np.mean((r_cst_test <= q_cst).reshape(-1)))
                    frac_per_patient = np.mean(r_cst_test <= q_cst, axis=1)
                    cov_patient_frac = float(np.mean(frac_per_patient >= patient_frac))
                    rows_r.append(
                        {
                            "method": f"RegionCP(ConVOLT-point-constk, {score_mode})",
                            "alpha": float(args.alpha),
                            "region_def": rd,
                            "region_score": score_mode,
                            "K": int(K),
                            "q_ml": float(q_cst),
                            "coverage_patient": cov_patient,
                            "coverage_region_pooled": cov_region,
                            "coverage_patient_frac_regions": cov_patient_frac,
                            "mean_interval_size_ml": float(2.0 * q_cst),
                        }
                    )

                # CQR(volonly) region guarantee (output-only quantiles + conformalized violations).
                if r_cqr_cal is not None and r_cqr_test is not None and qlo_test is not None and qhi_test is not None:
                    s_cal_cqr = _agg(r_cqr_cal, score_mode)
                    try:
                        q_cqr = conformal_quantile(s_cal_cqr, float(args.alpha))
                    except ValueError:
                        q_cqr = None
                    if q_cqr is not None:
                        s_test_cqr = _agg(r_cqr_test, score_mode)
                        cov_patient = float(np.mean(s_test_cqr <= q_cqr))
                        cov_region = float(np.mean((r_cqr_test <= q_cqr).reshape(-1)))
                        frac_per_patient = np.mean(r_cqr_test <= q_cqr, axis=1)
                        cov_patient_frac = float(np.mean(frac_per_patient >= patient_frac))
                        width_test = (qhi_test - qlo_test).astype(np.float64) + 2.0 * float(q_cqr)
                        rows_r.append(
                            {
                                "method": f"RegionCP(CQR(volonly), {score_mode})",
                                "alpha": float(args.alpha),
                                "region_def": rd,
                                "region_score": score_mode,
                                "K": int(K),
                                "q_ml": float(q_cqr),
                                "coverage_patient": cov_patient,
                                "coverage_region_pooled": cov_region,
                                "coverage_patient_frac_regions": cov_patient_frac,
                                "mean_interval_size_ml": float(np.nanmean(width_test)),
                            }
                        )

        if len(rows_r) > 0:
            region_table = pd.DataFrame(rows_r).sort_values(["region_def", "region_score", "method"])

    return RunOutputs(method_table=method_table, intervals_df=intervals_df, region_method_table=region_table)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    # Standardized defaults (dataset/method) while preserving explicit overrides.
    dataset = str(args.dataset).lower() if args.dataset is not None else None
    method = str(args.method).lower() if args.method is not None else None
    is_learn2reg = dataset in {"oasis"} if dataset is not None else False

    # Dataset-specific split defaults (only if user did not provide any explicit sizes).
    # LungCT is very small (n=30), so fixed sizes avoid accidental empty splits.
    if dataset == "lungct" and args.n_train is None and args.n_calib is None and args.n_test is None:
        args.n_train = 10
        args.n_calib = 15
        args.n_test = 5

    if args.results_dir is None:
        if dataset is None or method is None:
            raise ValueError("Provide either --results_dir or both --dataset and --method.")
        if is_learn2reg:
            from ..paths import default_results_dir_tagged
            from ..atlas import AtlasSpec, atlas_tag

            tag = atlas_tag(AtlasSpec(mode=str(args.atlas_mode), n=int(args.atlas_n), seed=int(args.atlas_seed)))
            args.results_dir = default_results_dir_tagged(dataset, method, tag)
        else:
            from ..paths import default_results_dir

            args.results_dir = default_results_dir(dataset, method)
    if args.out_dir is None:
        if dataset is None or method is None:
            raise ValueError("Provide either --out_dir or both --dataset and --method.")
        if is_learn2reg:
            from ..paths import default_uq_dir_tagged
            from ..atlas import AtlasSpec, atlas_tag

            tag = atlas_tag(AtlasSpec(mode=str(args.atlas_mode), n=int(args.atlas_n), seed=int(args.atlas_seed)))
            args.out_dir = default_uq_dir_tagged(dataset, method, tag)
        else:
            from ..paths import default_uq_dir

            args.out_dir = default_uq_dir(dataset, method)

    args.results_dir = Path(args.results_dir).resolve()
    args.out_dir = Path(args.out_dir).resolve()
    print(f"UQ: results_dir={args.results_dir}")
    print(f"UQ: out_dir={args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded: LoadedResults = load_registration_results_with_features(results_dir=args.results_dir, require_artifacts=True)
    df = loaded.df.copy()

    def _print_split_capacity(*, df_: pd.DataFrame, note: str) -> None:
        if "patient_id" not in df_.columns:
            return
        pids = df_["patient_id"].dropna().astype(str).unique().tolist()
        n = len(pids)
        print(f"UQ: available patients for split ({note}) = {n}")
        if n == 0:
            return
        splits = _make_splits(
            pids,
            int(args.seed),
            float(args.frac_train),
            float(args.frac_calib),
            n_train=args.n_train,
            n_calib=args.n_calib,
            n_test=args.n_test,
        )
        print(f"UQ: split sizes example ({note}): train={len(splits['train'])} calib={len(splits['calib'])} test={len(splits['test'])}")

    uq_target = str(args.uq_target).lower() if args.uq_target is not None else None
    run_volume_suite = False
    if uq_target in {"volume_union", "volume_label"}:
        if not is_learn2reg:
            raise ValueError(
                f"--uq_target {uq_target} is only supported for Learn2Reg inter-patient datasets "
                f"(oasis). For dataset={dataset}, use --uq_target delta_volume "
                "or omit --uq_target to use the dataset default."
            )
        run_volume_suite = True
    elif uq_target is None and is_learn2reg:
        run_volume_suite = True

    if run_volume_suite:
        # Learn2Reg (or other absolute-volume outputs): run CP on union volume and (optionally) per-label volumes.
        meta_path = Path(args.results_dir) / "atlas_meta.json"
        exclude_ids: set[str] = set()
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                exclude_ids.update(map(str, meta.get("atlas_ids", [])))
                exclude_ids.update(map(str, meta.get("vm_train_ids", [])))
            except Exception:
                pass
        if exclude_ids:
            df = df[~df["patient_id"].isin(sorted(exclude_ids))].reset_index(drop=True)
        _print_split_capacity(df_=df, note="volume suite (after exclusions)")

        # Default target set for Learn2Reg: union + topK labels (unless user overrides).
        targets: list[tuple[str, int | None]] = []
        if uq_target is None:
            targets.append(("volume_union", None))
            want_labels = True
        elif uq_target == "volume_union":
            targets.append(("volume_union", None))
            want_labels = False
        else:
            want_labels = True

        label_ids: list[int] = []
        if want_labels:
            lv_path = Path(args.results_dir) / "label_volumes.csv"
            if not lv_path.exists():
                raise FileNotFoundError(f"Missing {lv_path} (run registration on split=training to create it)")
            lv = pd.read_csv(lv_path)
            lv = lv[lv["patient_id"].isin(df["patient_id"])].reset_index(drop=True)
            ulist = str(args.uq_label_list).strip()
            if ulist:
                if ulist.lower() in {"all", "*"}:
                    label_ids = sorted({int(x) for x in lv["label_id"].unique().tolist() if int(x) > 0})
                else:
                    parts = [p for p in re.split(r"[,\s]+", ulist) if p.strip()]
                    label_ids = [int(x) for x in parts]
            else:
                k = int(max(0, args.uq_topk_labels))
                if k > 0 and len(lv) > 0 and "vol_ml_gt" in lv.columns:
                    m = lv[np.isfinite(lv["vol_ml_gt"].to_numpy(dtype=np.float64))].groupby("label_id", as_index=False)["vol_ml_gt"].mean()
                    m = m.sort_values("vol_ml_gt", ascending=False)
                    label_ids = [int(x) for x in m["label_id"].head(k).tolist()]
            if uq_target == "volume_label" and not label_ids:
                raise ValueError("No labels selected for --uq_target volume_label (use --uq_topk_labels or --uq_label_list).")
            for lid in label_ids:
                targets.append(("volume_label", int(lid)))

        # Feature diagnostics: union volume error vs features (best default).
        if "vol_union_ml_gt" in df.columns and "vol_union_ml_pred0" in df.columns:
            signed_err0_all = df["vol_union_ml_pred0"].to_numpy(dtype=np.float64) - df["vol_union_ml_gt"].to_numpy(dtype=np.float64)
            abs_err0_all = np.abs(signed_err0_all)
            save_feature_diagnostics(
                out_dir=args.out_dir,
                df=df,
                feature_keys=loaded.diagnostic_feature_keys,
                abs_err=abs_err0_all,
                err=signed_err0_all,
                title_prefix="All data (union volume): ",
            )

        n_repeats = int(max(1, args.n_repeats))
        run_rows: list[pd.DataFrame] = []

        for rep in range(n_repeats):
            rep_seed = int(args.seed) + int(rep)
            if n_repeats == 1:
                rep_out_dir = args.out_dir
            else:
                rep_out_dir = args.out_dir / "repeats" / f"rep_{rep:03d}"
                rep_out_dir.mkdir(parents=True, exist_ok=True)

            method_tabs: list[pd.DataFrame] = []
            interval_tabs: list[pd.DataFrame] = []

            for tname, lid in targets:
                if tname == "volume_union":
                    if "vol_union_ml_gt" not in df.columns or "vol_union_ml_pred0" not in df.columns:
                        cols = ", ".join(sorted(map(str, df.columns.tolist())))
                        raise KeyError(
                            "Missing vol_union_ml_gt/vol_union_ml_pred0 in summary.csv. "
                            "This usually means (a) you pointed --results_dir to a non-Learn2Reg run, "
                            "or (b) you are using an older Learn2Reg summary schema. "
                            f"Available columns: {cols}"
                        )
                    df_t = df.copy()
                    y_gt_col = "vol_union_ml_gt"
                    y_pred_col = "vol_union_ml_pred0"
                else:
                    # Join per-label volumes with per-patient spatial features.
                    sub = lv[lv["label_id"] == int(lid)][["patient_id", "vol_ml_gt", "vol_ml_pred"]].copy()
                    df_t = df.merge(sub, on="patient_id", how="inner")
                    y_gt_col = "vol_ml_gt"
                    y_pred_col = "vol_ml_pred"

                if rep == 0:
                    _print_split_capacity(
                        df_=df_t,
                        note=f"{tname}{'' if lid is None else f' label={int(lid)}'} (rep0)",
                    )

                mt, it = _run_once_volume(
                    df=df_t,
                    loaded=loaded,
                    args=args,
                    seed=rep_seed,
                    out_dir=rep_out_dir,
                    target=tname,
                    label_id=lid,
                    y_gt_col=y_gt_col,
                    y_pred0_col=y_pred_col,
                )
                method_tabs.append(mt)
                interval_tabs.append(it)

            # OASIS label-specific ConVOLT variants (label-local features + partial pooling).
            # Do NOT affect other datasets/targets.
            if dataset == "oasis" and want_labels and label_ids:
                extra = _oasis_label_convot_extra_methods(
                    df_feat=df,
                    lv=lv,
                    label_ids=label_ids,
                    args=args,
                    seed=rep_seed,
                    out_dir_root=args.out_dir,
                )
                if len(extra) > 0:
                    method_tabs.append(extra)

            method_table = pd.concat(method_tabs, axis=0, ignore_index=True)
            intervals_df = pd.concat(interval_tabs, axis=0, ignore_index=True)
            method_table.to_csv(rep_out_dir / "method_table.csv", index=False)
            if n_repeats == 1 or bool(args.save_intervals_each_repeat):
                intervals_df.to_csv(rep_out_dir / "intervals_test.csv", index=False)

            rep_rows = method_table.copy()
            rep_rows.insert(0, "repeat", rep)
            rep_rows.insert(1, "seed", rep_seed)
            run_rows.append(rep_rows)

        runs_df = pd.concat(run_rows, axis=0, ignore_index=True)
        runs_df.to_csv(args.out_dir / "cp_runs.csv", index=False)

        summary = (
            runs_df.groupby(["target", "label_id", "method"], as_index=False)
            .agg(
                alpha=("alpha", "first"),
                n_repeats=("repeat", "nunique"),
                coverage_mean=("coverage", "mean"),
                coverage_std=("coverage", "std"),
                interval_size_mean=("mean_interval_size_ml", "mean"),
                interval_size_std=("mean_interval_size_ml", "std"),
            )
            .sort_values(["target", "label_id", "method"])
        )
        summary["coverage_mean_pm_std"] = summary.apply(
            lambda r: f"{r['coverage_mean']:.3f}±{(0.0 if pd.isna(r['coverage_std']) else r['coverage_std']):.3f}",
            axis=1,
        )
        summary["interval_size_mean_pm_std_ml"] = summary.apply(
            lambda r: f"{r['interval_size_mean']:.1f}±{(0.0 if pd.isna(r['interval_size_std']) else r['interval_size_std']):.1f}",
            axis=1,
        )
        summary.to_csv(args.out_dir / "cp_summary.csv", index=False)
        return

    # Delta-volume suite (intra-patient).
    _print_split_capacity(df_=df, note="delta-volume suite")

    # Feature diagnostics (split-agnostic): do spatial features predict |baseline error|?
    signed_err0_all = df["delta_vol_ml_pred0"].to_numpy(dtype=np.float64) - df["delta_vol_ml_gt"].to_numpy(dtype=np.float64)
    abs_err0_all = np.abs(signed_err0_all)
    save_feature_diagnostics(
        out_dir=args.out_dir,
        df=df,
        feature_keys=loaded.diagnostic_feature_keys,
        abs_err=abs_err0_all,
        err=signed_err0_all,
        title_prefix="All data: ",
    )

    # Precompute region volumes for lung datasets (optional).
    region_defs = [s.strip().lower() for s in str(args.region_defs).split(",") if s.strip()]
    region_cache: dict[str, dict[str, RegionData]] | None = None
    if len(region_defs) > 0:
        region_cache = {}
        for _, row in df.iterrows():
            pid = str(row["patient_id"])
            pair_dir = Path(str(row["pair_dir"]))
            artifacts = pair_dir / "artifacts.npz"
            if not artifacts.exists():
                continue
            npz = np.load(artifacts, allow_pickle=True)
            if "jac_det_zyx" not in npz:
                continue
            if "spacing_zyx" in npz:
                spacing = tuple(map(float, npz["spacing_zyx"].tolist()))
            elif "fixed_spacing_zyx" in npz:
                spacing = tuple(map(float, npz["fixed_spacing_zyx"].tolist()))
            else:
                continue

            # Support NLST/LungCT naming as well as ACDC naming.
            if "inhale_mask_zyx" in npz:
                fixed_mask = npz["inhale_mask_zyx"]
            elif "fixed_mask_zyx" in npz:
                fixed_mask = npz["fixed_mask_zyx"]
            elif "ed_lv_mask_zyx" in npz:
                fixed_mask = npz["ed_lv_mask_zyx"]
            else:
                continue

            if "exhale_mask_resampled_zyx" in npz:
                moving_mask_rs = npz["exhale_mask_resampled_zyx"]
            elif "moving_mask_zyx" in npz:
                moving_mask_rs = npz["moving_mask_zyx"]
            elif "es_lv_mask_resampled_zyx" in npz:
                moving_mask_rs = npz["es_lv_mask_resampled_zyx"]
            else:
                continue

            jac = npz["jac_det_zyx"]
            disp_mm = npz["disp_mm_3zyx"] if "disp_mm_3zyx" in npz else None
            region_cache[pid] = {}
            for rd in region_defs:
                region_cache[pid][rd] = compute_regions(
                    region_def=rd,
                    fixed_mask_zyx=fixed_mask,
                    moving_mask_resampled_zyx=moving_mask_rs,
                    jac_det_zyx=jac,
                    disp_mm_3zyx=disp_mm,
                    spacing_zyx=spacing,
                    n_shells=int(args.radial_shells),
                    feature_keys=loaded.feature_keys,
                )
        if len(region_cache) == 0:
            print("WARNING: region_defs requested but no region_cache entries could be computed (check artifacts keys).")
        else:
            print(f"RegionCP: computed regions for {len(region_cache)}/{len(df)} cases.")
    else:
        print("RegionCP: disabled (pass --region_defs radial to enable region-based guarantees).")

    # Jacobian determinant histogram within fixed mask for a few illustrative patients.
    n_hist = int(max(0, args.jac_hist_patients))
    if n_hist > 0:
        jac_rows = []
        for _, row in df.iterrows():
            pair_dir = Path(str(row["pair_dir"]))
            artifacts = pair_dir / "artifacts.npz"
            if not artifacts.exists():
                continue
            npz = np.load(artifacts, allow_pickle=True)
            if "jac_det_zyx" not in npz:
                continue
            if "inhale_mask_zyx" in npz:
                m = npz["inhale_mask_zyx"]
            elif "fixed_mask_zyx" in npz:
                m = npz["fixed_mask_zyx"]
            elif "ed_lv_mask_zyx" in npz:
                m = npz["ed_lv_mask_zyx"]
            else:
                continue
            j = np.asarray(npz["jac_det_zyx"], dtype=np.float64)
            mv = j[np.asarray(m) > 0]
            mv = mv[np.isfinite(mv)]
            if mv.size == 0:
                continue
            mv = np.clip(mv, 1e-6, np.inf)
            jac_rows.append(
                {
                    "patient_id": str(row["patient_id"]),
                    "pair_dir": str(pair_dir),
                    "frac_j_lt_001": float(np.mean(mv < 0.01)),
                    "frac_j_lt_01": float(np.mean(mv < 0.1)),
                }
            )
        if len(jac_rows) > 0:
            jac_df = pd.DataFrame(jac_rows).sort_values(["frac_j_lt_001", "frac_j_lt_01"], ascending=False)
            jac_df.to_csv(Path(args.out_dir) / "diagnostics" / "jac_mask_summary.csv", index=False)

            pick = jac_df.head(int(min(n_hist, len(jac_df))))
            pids = pick["patient_id"].tolist()
            jac_list = []
            mask_list = []
            for _, r in pick.iterrows():
                artifacts = Path(r["pair_dir"]) / "artifacts.npz"
                npz = np.load(artifacts, allow_pickle=True)
                if "inhale_mask_zyx" in npz:
                    m = npz["inhale_mask_zyx"]
                elif "fixed_mask_zyx" in npz:
                    m = npz["fixed_mask_zyx"]
                else:
                    m = npz["ed_lv_mask_zyx"]
                jac_list.append(npz["jac_det_zyx"])
                mask_list.append(m)

            save_jac_det_histograms_in_mask(
                out_path=Path(args.out_dir) / "diagnostics" / "jac_det_hist_in_mask.png",
                patient_ids=pids,
                jac_list=jac_list,
                mask_list=mask_list,
                title="Jacobian determinant in fixed mask (selected worst frac(J<0.01))",
            )

    n_repeats = int(max(1, args.n_repeats))
    run_rows: list[pd.DataFrame] = []
    region_run_rows: list[pd.DataFrame] = []

    for rep in range(n_repeats):
        rep_seed = int(args.seed) + int(rep)
        if n_repeats == 1:
            rep_out_dir = args.out_dir
        else:
            rep_out_dir = args.out_dir / "repeats" / f"rep_{rep:03d}"
            rep_out_dir.mkdir(parents=True, exist_ok=True)

        out = _run_once(df=df, loaded=loaded, args=args, seed=rep_seed, out_dir=rep_out_dir, region_cache=region_cache)

        out.method_table.to_csv(rep_out_dir / "method_table.csv", index=False)
        if n_repeats == 1 or bool(args.save_intervals_each_repeat):
            out.intervals_df.to_csv(rep_out_dir / "intervals_test.csv", index=False)
        if out.region_method_table is not None:
            out.region_method_table.to_csv(rep_out_dir / "region_method_table.csv", index=False)

        rep_rows = out.method_table.copy()
        rep_rows.insert(0, "repeat", rep)
        rep_rows.insert(1, "seed", rep_seed)
        run_rows.append(rep_rows)

        if out.region_method_table is not None:
            rep_r = out.region_method_table.copy()
            rep_r.insert(0, "repeat", rep)
            rep_r.insert(1, "seed", rep_seed)
            region_run_rows.append(rep_r)

    runs_df = pd.concat(run_rows, axis=0, ignore_index=True)
    runs_df.to_csv(args.out_dir / "cp_runs.csv", index=False)

    # Summary (mean±std across repeats).
    summary = (
        runs_df.groupby("method", as_index=False)
        .agg(
            alpha=("alpha", "first"),
            n_repeats=("repeat", "nunique"),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            interval_size_mean=("mean_interval_size_ml", "mean"),
            interval_size_std=("mean_interval_size_ml", "std"),
        )
        .sort_values("method")
    )
    summary["coverage_mean_pm_std"] = summary.apply(
        lambda r: f"{r['coverage_mean']:.3f}±{(0.0 if pd.isna(r['coverage_std']) else r['coverage_std']):.3f}",
        axis=1,
    )
    summary["interval_size_mean_pm_std_ml"] = summary.apply(
        lambda r: f"{r['interval_size_mean']:.1f}±{(0.0 if pd.isna(r['interval_size_std']) else r['interval_size_std']):.1f}",
        axis=1,
    )
    summary.to_csv(args.out_dir / "cp_summary.csv", index=False)

    if len(region_run_rows) > 0:
        rr = pd.concat(region_run_rows, axis=0, ignore_index=True)
        rr.to_csv(args.out_dir / "region_cp_runs.csv", index=False)
        region_summary = (
            rr.groupby(["method", "region_def", "region_score", "K"], as_index=False)
            .agg(
                alpha=("alpha", "first"),
                n_repeats=("repeat", "nunique"),
                coverage_patient_mean=("coverage_patient", "mean"),
                coverage_patient_std=("coverage_patient", "std"),
                coverage_region_pooled_mean=("coverage_region_pooled", "mean"),
                coverage_region_pooled_std=("coverage_region_pooled", "std"),
                coverage_patient_frac_regions_mean=("coverage_patient_frac_regions", "mean"),
                coverage_patient_frac_regions_std=("coverage_patient_frac_regions", "std"),
                interval_size_mean=("mean_interval_size_ml", "mean"),
                interval_size_std=("mean_interval_size_ml", "std"),
            )
            .sort_values(["region_def", "region_score", "method"])
        )
        region_summary.to_csv(args.out_dir / "region_cp_summary.csv", index=False)


if __name__ == "__main__":
    main()
