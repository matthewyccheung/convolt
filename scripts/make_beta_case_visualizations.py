#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
# Ensure repo root is importable when running as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _safe_name(s: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_").lower()


def _pick_slice_from_mask(mask_zyx: np.ndarray) -> int:
    m = np.asarray(mask_zyx) > 0
    if m.ndim != 3 or not np.any(m):
        return int(mask_zyx.shape[0] // 2)
    areas = m.reshape(m.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def _pct_window(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    a = float(np.percentile(x, lo))
    b = float(np.percentile(x, hi))
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        return float(np.min(x)), float(np.max(x))
    return a, b


def _overlay_contour(ax, mask_yx: np.ndarray, *, color: str, lw: float = 1.6, alpha: float = 0.95) -> None:
    m = (np.asarray(mask_yx) > 0).astype(np.uint8)
    if not np.any(m):
        return
    try:
        ax.contour(m, levels=[0.5], colors=[color], linewidths=[lw], alpha=alpha)
    except Exception:
        pass


def _impute_feature_nans(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    finite = np.isfinite(X)
    if X.size == 0:
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    col_sum = np.where(finite, X, 0.0).sum(axis=0, keepdims=True)
    col_cnt = finite.sum(axis=0, keepdims=True).astype(np.float32)
    col_mean = np.divide(col_sum, np.maximum(col_cnt, 1.0), out=np.zeros_like(col_sum, dtype=np.float32), where=(col_cnt > 0))
    X = np.where(finite, X, col_mean)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _make_split(patient_ids: list[str], *, seed: int, n_train: int, n_calib: int, n_test: int | None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    ids = np.array(sorted(set(patient_ids)), dtype=object)
    rng.shuffle(ids)
    n = int(ids.size)
    n_train_i = int(np.clip(int(n_train), 0, n))
    n_cal_i = int(np.clip(int(n_calib), 0, n - n_train_i))
    train = ids[:n_train_i]
    calib = ids[n_train_i : n_train_i + n_cal_i]
    rest = ids[n_train_i + n_cal_i :]
    if n_test is None:
        test = rest
    else:
        test = rest[: int(np.clip(int(n_test), 0, int(rest.size)))]
    return {"train": train, "calib": calib, "test": test}


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile for miscoverage alpha (Lei et al., 2018 style).
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("empty scores")
    n = int(s.size)
    k = int(math.ceil((n + 1) * (1.0 - float(alpha)))) - 1
    k = int(np.clip(k, 0, n - 1))
    return float(np.sort(s)[k])


@dataclass(frozen=True)
class CaseViz:
    patient_id: str
    row_idx: int
    k_hat: float
    k_lo: float
    k_hi: float
    width_ml: float
    base_pred_ml: float
    point_pred_ml: float
    artifacts_path: Path
    kind: str  # "large" or "near1"


def _results_dir_from_uq_run(run: str, *, results_root: Path) -> Path:
    name = str(run).strip()
    for suf in ("_globalfeat",):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return results_root / name


def _short_method_name(name: str) -> str:
    n = str(name).strip()
    if n.startswith("ConVOLT"):
        return "ConVOLT"
    if n.startswith("CQR"):
        return "CQR"
    if n.startswith("SCP"):
        return "SCP"
    if n.startswith("LocalSCP"):
        return "LCP"
    return n


def _plot_dataset_backend(
    *,
    dataset: str,
    backend: str,
    run: str,
    results_root: Path,
    uq_root: Path,
    out_dir: Path,
    alpha: float,
    ridge_l2: float,
    beta_large_thresh: float,
    beta_near_tol: float,
    seed: int,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from reg.uq.io import load_registration_results_with_features
    from reg.uq.models import fit_ridge

    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = _results_dir_from_uq_run(run, results_root=results_root)
    print(f"[beta-case] loading features: dataset={dataset} backend={backend} results_dir={results_dir}")
    loaded = load_registration_results_with_features(results_dir=results_dir, require_artifacts=True)
    df = loaded.df.copy()
    if "patient_id" not in df.columns or "pair_dir" not in df.columns:
        return

    # Split sizes from cp_runs.csv (if available).
    n_train, n_calib, n_test = 10, 10, None
    cp_runs = uq_root / run / "cp_runs.csv"
    if cp_runs.exists():
        try:
            cr = pd.read_csv(cp_runs, skipinitialspace=True)
            cr["method"] = cr["method"].astype(str)
            cand = cr[cr["method"] == "ConVOLT(scale-CP)"]
            if len(cand) == 0:
                cand = cr
            if len(cand) > 0:
                r0 = cand.iloc[0]
                n_train = int(r0.get("n_train", n_train))
                n_calib = int(r0.get("n_calib", n_calib))
                n_test = int(r0.get("n_test", 0))
                if n_test <= 0:
                    n_test = None
        except Exception:
            pass

    # Determine which target we’re using (absolute union volume vs exhale volume ratio for delta-V tasks).
    eps = 1e-6
    is_union = bool(
        ("vol_union_ml_gt" in df.columns)
        and ("vol_union_ml_pred0" in df.columns)
        and np.any(np.isfinite(df["vol_union_ml_gt"].to_numpy(dtype=np.float64)))
    )

    if is_union:
        y_gt = df["vol_union_ml_gt"].to_numpy(dtype=np.float64)
        y0 = df["vol_union_ml_pred0"].to_numpy(dtype=np.float64)
        base_pred_col = "vol_union_ml_pred0"
        inhale_col = None
        title_target = "V_union"
    else:
        y_gt = df["exhale_vol_ml_gt"].to_numpy(dtype=np.float64)
        y0 = df["exhale_vol_ml_pred0"].to_numpy(dtype=np.float64)
        base_pred_col = "delta_vol_ml_pred0"
        inhale_col = "inhale_vol_ml_gt"
        title_target = "ΔV"

    k_true = ((y_gt + eps) / (y0 + eps)).astype(np.float32)

    feats = [k for k in loaded.feature_keys if k in df.columns]
    if len(feats) == 0:
        return

    splits = _make_split(df["patient_id"].astype(str).tolist(), seed=int(seed), n_train=int(n_train), n_calib=int(n_calib), n_test=n_test)
    df_tr = df[df["patient_id"].isin(splits["train"])].copy()
    df_ca = df[df["patient_id"].isin(splits["calib"])].copy()

    # Fit ridge on train.
    X_tr = _impute_feature_nans(df_tr.loc[:, feats].to_numpy(dtype=np.float32))
    y_tr = k_true[df_tr.index.to_numpy()]
    ok_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
    if int(np.count_nonzero(ok_tr)) < max(5, int(0.2 * len(df_tr))):
        return
    coef, intercept = fit_ridge(X_tr[ok_tr], y_tr[ok_tr], l2=float(ridge_l2))

    # Predict k_hat for all cases.
    X_all = _impute_feature_nans(df.loc[:, feats].to_numpy(dtype=np.float32))
    k_hat_all = (X_all.astype(np.float64) @ coef.astype(np.float64) + float(intercept)).astype(np.float64)
    k_hat_all = np.clip(k_hat_all, 0.0, np.inf)

    # Calibrate q on calib residuals.
    X_ca = _impute_feature_nans(df_ca.loc[:, feats].to_numpy(dtype=np.float32))
    k_hat_ca = (X_ca.astype(np.float64) @ coef.astype(np.float64) + float(intercept)).astype(np.float64)
    k_hat_ca = np.clip(k_hat_ca, 0.0, np.inf)
    y_ca = k_true[df_ca.index.to_numpy()].astype(np.float64)
    q = _conformal_quantile(np.abs(y_ca - k_hat_ca), float(alpha))

    # Pick one "large beta" and one "near 1" case.
    pids = df["patient_id"].astype(str).tolist()
    large_idx = [i for i, v in enumerate(k_hat_all) if np.isfinite(v) and float(v) > float(beta_large_thresh)]
    if not large_idx:
        # Fallback: just take the max.
        large_idx = [int(np.nanargmax(k_hat_all))]
    i_large = int(max(large_idx, key=lambda i: float(k_hat_all[i])))

    near_idx = int(np.nanargmin(np.abs(k_hat_all - 1.0)))
    if float(np.abs(k_hat_all[near_idx] - 1.0)) > float(beta_near_tol):
        # Still pick the closest; tolerance is just for user expectation.
        pass

    def _case(i: int, kind: str) -> CaseViz:
        pid = str(pids[i])
        k_hat = float(k_hat_all[i])
        k_lo = float(max(0.0, k_hat - q))
        k_hi = float(max(k_lo, k_hat + q))
        v0 = float(y0[i])
        if is_union:
            base_pred = float(df.iloc[i][base_pred_col])
            point_pred = float(k_hat * v0)
            width = float((k_hi - k_lo) * v0)
        else:
            inhale = float(df.iloc[i][inhale_col]) if inhale_col is not None else 0.0
            base_pred = float(df.iloc[i][base_pred_col])
            point_pred = float(k_hat * v0 - inhale)
            width = float((k_hi - k_lo) * v0)
        artifacts = Path(str(df.iloc[i]["pair_dir"])) / "artifacts.npz"
        return CaseViz(
            patient_id=pid,
            row_idx=int(i),
            k_hat=k_hat,
            k_lo=k_lo,
            k_hi=k_hi,
            width_ml=width,
            base_pred_ml=base_pred,
            point_pred_ml=point_pred,
            artifacts_path=artifacts,
            kind=kind,
        )

    cases = [_case(i_large, "large"), _case(near_idx, "near1")]

    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    # 2 rows (large β, near-1 β) × 6 panels:
    # fixed+contour, logJ, |u|, k_lo·J, k_hi·J, text
    fig, axes = plt.subplots(2, 6, figsize=(12.6, 4.2))
    if axes.ndim != 2:
        return

    for r, cv in enumerate(cases):
        try:
            npz = np.load(cv.artifacts_path, allow_pickle=True)
        except Exception:
            continue

        if is_union:
            fixed_img = npz.get("fixed_image_zyx")
            base_mask = npz.get("pred_union_mask_zyx")
            jac = npz.get("jac_det_zyx")
            disp_mag = npz.get("disp_mag_mm_zyx")
            spacing = tuple(map(float, npz.get("spacing_zyx").tolist()))
        else:
            fixed_img = npz.get("inhale_ct_zyx")
            base_mask = npz.get("exhale_mask_warped_zyx")
            jac = npz.get("jac_det_zyx")
            disp_mag = npz.get("disp_mag_mm_zyx")
            spacing = tuple(map(float, npz.get("spacing_zyx").tolist()))

        if fixed_img is None or base_mask is None or jac is None or disp_mag is None:
            continue

        sl = _pick_slice_from_mask(base_mask)
        img2 = fixed_img[sl].astype(np.float32, copy=False)

        # Col 0: fixed + baseline warped contour.
        ax = axes[r, 0]
        lo, hi = _pct_window(img2)
        ax.imshow(img2, cmap="gray", vmin=lo, vmax=hi)
        _overlay_contour(ax, base_mask[sl], color="#F58518", lw=1.8)
        ax.set_title(f"{cv.patient_id} | {cv.kind} β")

        # Col 1: logJ heatmap.
        ax = axes[r, 1]
        logj = np.log(np.clip(jac[sl].astype(np.float32, copy=False), 1e-6, np.inf))
        vmin, vmax = np.percentile(logj[np.isfinite(logj)], [2, 98]) if np.any(np.isfinite(logj)) else (-1.0, 1.0)
        ax.imshow(logj, cmap="coolwarm", vmin=float(vmin), vmax=float(vmax))
        _overlay_contour(ax, base_mask[sl], color="white", lw=1.2, alpha=0.9)
        ax.set_title("log J")

        # Col 2: displacement magnitude.
        ax = axes[r, 2]
        dm = disp_mag[sl].astype(np.float32, copy=False)
        vmin, vmax = _pct_window(dm, 1.0, 99.0)
        ax.imshow(dm, cmap="magma", vmin=vmin, vmax=vmax)
        _overlay_contour(ax, base_mask[sl], color="white", lw=1.2, alpha=0.9)
        ax.set_title("|u| (mm)")

        # Col 3: k_lo · J heatmap (volume scaling proxy).
        ax = axes[r, 3]
        j = np.clip(jac[sl].astype(np.float32, copy=False), 0.0, np.inf)
        j_lo = float(cv.k_lo) * j
        j_hi = float(cv.k_hi) * j
        vv = np.concatenate([j_lo.reshape(-1), j_hi.reshape(-1)])
        vv = vv[np.isfinite(vv)]
        vmin, vmax = (0.0, 1.0)
        if vv.size > 0:
            vmin, vmax = np.percentile(vv, [2, 98])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
                vmin, vmax = float(np.min(vv)), float(np.max(vv))
        ax.imshow(j_lo, cmap="Blues", vmin=float(vmin), vmax=float(vmax))
        _overlay_contour(ax, base_mask[sl], color="black", lw=1.0, alpha=0.85)
        ax.set_title(r"$k_{\mathrm{lo}}\cdot J$")

        # Col 4: k_hi · J heatmap.
        ax = axes[r, 4]
        ax.imshow(j_hi, cmap="Blues", vmin=float(vmin), vmax=float(vmax))
        _overlay_contour(ax, base_mask[sl], color="black", lw=1.0, alpha=0.85)
        ax.set_title(r"$k_{\mathrm{hi}}\cdot J$")

        # Col 5: text panel.
        ax = axes[r, 5]
        ax.axis("off")
        ax.text(
            0.0,
            1.0,
            "\n".join(
                [
                    f"Target: {title_target}",
                    f"β̂ (k): {cv.k_hat:.3f}",
                    f"k_lo/k_hi: {cv.k_lo:.3f} / {cv.k_hi:.3f}",
                    f"Base pred: {cv.base_pred_ml:.1f} mL",
                    f"ConVOLT point: {cv.point_pred_ml:.1f} mL",
                    f"Interval width: {cv.width_ml:.1f} mL",
                    "lo/hi shown via k·J scaling",
                ]
            ),
            ha="left",
            va="top",
            fontsize=8,
        )

    for ax in axes.reshape(-1):
        if ax is not None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / dataset / backend / f"fig_beta_cases_{_safe_name(run)}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make large-β vs small-β case visualizations for ConVOLT(scale-CP).")
    ap.add_argument("--tables_dir", type=Path, default=Path("uq_results") / "_tables")
    ap.add_argument("--uq_root", type=Path, default=Path("uq_results"))
    ap.add_argument("--results_root", type=Path, default=Path(os.environ.get("CONVOLT_RESULTS_ROOT", "/scratch/yc130/Registration/outputs")))
    ap.add_argument("--datasets", type=str, default="lungct,nlst,oasis")
    ap.add_argument("--backends", type=str, default="demons,voxelmorph")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--ridge_l2", type=float, default=0.01)
    ap.add_argument("--beta_large_thresh", type=float, default=1.2)
    ap.add_argument("--beta_near_tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=Path("uq_results") / "_figures_paper" / "beta_cases")
    args = ap.parse_args()

    datasets = [d.strip().lower() for d in str(args.datasets).split(",") if d.strip()]
    backends = [b.strip().lower() for b in str(args.backends).split(",") if b.strip()]
    backends = [b for b in backends if b in {"demons", "voxelmorph"}]
    if not backends:
        backends = ["demons", "voxelmorph"]

    # Use main-table ConVOLT runs to select which output folder to visualize.
    df_main = pd.read_csv(Path(args.tables_dir) / "main_results_long_main.csv", skipinitialspace=True)
    df_main["dataset"] = df_main["dataset"].astype(str).str.strip().str.lower()
    df_main["backend"] = df_main["backend"].astype(str).str.strip().str.lower()
    df_main["uq_method"] = df_main["uq_method"].astype(str).str.strip()

    df_conv = df_main[df_main["uq_method"] == "ConVOLT"].copy()

    for ds in datasets:
        for b in backends:
            rows = df_conv[(df_conv["dataset"] == ds) & (df_conv["backend"] == b)]
            if len(rows) == 0:
                continue
            run = str(rows.iloc[0]["run"]).strip()
            _plot_dataset_backend(
                dataset=ds,
                backend=b,
                run=run,
                results_root=Path(args.results_root),
                uq_root=Path(args.uq_root),
                out_dir=Path(args.out_dir),
                alpha=float(args.alpha),
                ridge_l2=float(args.ridge_l2),
                beta_large_thresh=float(args.beta_large_thresh),
                beta_near_tol=float(args.beta_near_tol),
                seed=int(args.seed),
            )

    print(f"Wrote beta-case figures under: {args.out_dir}")


if __name__ == "__main__":
    main()
