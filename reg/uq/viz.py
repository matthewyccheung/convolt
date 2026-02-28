from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .compass import delta_from_beta


def save_gridsweep_plot(
    *,
    out_path: str | Path,
    patient_id: str,
    beta_grid: np.ndarray,
    exhale_pred0_ml: float,
    inhale_vol_ml: float,
    delta_gt_ml: float,
    delta_point_ml: float,
    interval_lo_ml: float | None = None,
    interval_hi_ml: float | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    beta_grid = np.asarray(beta_grid, dtype=np.float64)
    delta_grid = delta_from_beta(
        beta=beta_grid,
        exhale_pred0_ml=np.array(exhale_pred0_ml, dtype=np.float64),
        inhale_vol_ml=np.array(inhale_vol_ml, dtype=np.float64),
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.plot(beta_grid, delta_grid, label="ΔV(β) = exp(β)*V0 - Vinh")
    ax.axhline(delta_gt_ml, color="black", linestyle="--", linewidth=1.5, label="GT ΔV")
    ax.axhline(delta_point_ml, color="tab:blue", linestyle=":", linewidth=1.5, label="Point ΔV")
    if interval_lo_ml is not None and interval_hi_ml is not None:
        ax.axhspan(interval_lo_ml, interval_hi_ml, color="tab:orange", alpha=0.2, label="Prediction interval")

    ax.set_title(f"{patient_id} monotonicity sweep")
    ax.set_xlabel("β")
    ax.set_ylabel("ΔV (mL)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_nonconformity_histogram(
    *,
    out_path: str | Path,
    scp_scores: np.ndarray,
    compass_scores: np.ndarray,
    title: str = "Calibration nonconformity distributions",
    label_scp: str = "SCP score |y - ŷ0| (mL)",
    label_compass: str = "ConVOLT score |y - ŷβ| (mL)",
    eps: float = 1e-6,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    a = np.asarray(scp_scores, dtype=np.float64).reshape(-1)
    b = np.asarray(compass_scores, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return

    a = np.clip(a, eps, np.inf)
    b = np.clip(b, eps, np.inf)
    allv = np.concatenate([a, b], axis=0)
    lo = float(np.min(allv))
    hi = float(np.quantile(allv, 0.995))
    hi = max(hi, lo * 10.0)

    bins = np.logspace(np.log10(lo), np.log10(hi), 60)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.hist(a, bins=bins, alpha=0.55, label=label_scp)
    ax.hist(b, bins=bins, alpha=0.55, label=label_compass)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("score (mL)")
    ax.set_ylabel("count")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_feature_diagnostics(
    *,
    out_dir: str | Path,
    df,
    feature_keys: Tuple[str, ...],
    abs_err: np.ndarray,
    err: np.ndarray | None = None,
    title_prefix: str = "",
) -> None:
    """
    Writes:
      diagnostics/feature_stats.csv
      diagnostics/feature_corr.csv
      diagnostics/feature_corr_bar.png
      diagnostics/scatter/<feature>.png
    """
    import pandas as pd

    out_dir = Path(out_dir)
    diag_dir = out_dir / "diagnostics"
    (diag_dir / "scatter").mkdir(parents=True, exist_ok=True)

    if len(feature_keys) == 0:
        return

    aerr = np.asarray(abs_err, dtype=np.float64).reshape(-1)
    serr = np.asarray(err, dtype=np.float64).reshape(-1) if err is not None else None

    rows_stats = []
    rows_corr = []
    rows_corr_signed = []
    for k in feature_keys:
        x = np.asarray(df[k].to_numpy(dtype=np.float64), dtype=np.float64).reshape(-1)
        m = np.isfinite(x) & np.isfinite(aerr)
        if int(np.count_nonzero(m)) < 3:
            continue
        xv = x[m]
        yv = aerr[m]

        # Stats
        rows_stats.append(
            {
                "feature": k,
                "mean": float(np.mean(xv)),
                "std": float(np.std(xv)),
                "min": float(np.min(xv)),
                "p10": float(np.quantile(xv, 0.10)),
                "p50": float(np.quantile(xv, 0.50)),
                "p90": float(np.quantile(xv, 0.90)),
                "max": float(np.max(xv)),
                "n": int(xv.size),
            }
        )

        # Correlations with |error|
        x0 = xv - float(np.mean(xv))
        y0 = yv - float(np.mean(yv))
        denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
        pear = float(np.sum(x0 * y0) / denom) if denom > 0 else float("nan")

        try:
            from scipy.stats import spearmanr

            spear = float(spearmanr(xv, yv).correlation)
        except Exception:
            spear = float("nan")

        rows_corr.append({"feature": k, "pearson_r": pear, "spearman_r": spear, "n": int(xv.size)})

        # Scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), constrained_layout=True)
        ax.scatter(xv, yv, s=12, alpha=0.55, edgecolors="none")
        ax.set_xlabel(k)
        ax.set_ylabel("|error| (mL)")
        title = f"{title_prefix}{k} vs |error|"
        ax.set_title(title.strip())
        ax.grid(True, alpha=0.25)
        fig.savefig(diag_dir / "scatter" / f"{k}.png", dpi=150)
        plt.close(fig)

        # Signed error diagnostics (optional).
        if serr is not None:
            ms = np.isfinite(x) & np.isfinite(serr)
            if int(np.count_nonzero(ms)) >= 3:
                xs = x[ms]
                ys = serr[ms]
                x0s = xs - float(np.mean(xs))
                y0s = ys - float(np.mean(ys))
                denoms = float(np.sqrt(np.sum(x0s * x0s) * np.sum(y0s * y0s)))
                pears = float(np.sum(x0s * y0s) / denoms) if denoms > 0 else float("nan")
                try:
                    from scipy.stats import spearmanr

                    spears = float(spearmanr(xs, ys).correlation)
                except Exception:
                    spears = float("nan")
                rows_corr_signed.append({"feature": k, "pearson_r": pears, "spearman_r": spears, "n": int(xs.size)})

                (diag_dir / "scatter_signed").mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), constrained_layout=True)
                ax.scatter(xs, ys, s=12, alpha=0.55, edgecolors="none")
                ax.set_xlabel(k)
                ax.set_ylabel("error (mL)")
                title = f"{title_prefix}{k} vs error"
                ax.set_title(title.strip())
                ax.grid(True, alpha=0.25)
                fig.savefig(diag_dir / "scatter_signed" / f"{k}.png", dpi=150)
                plt.close(fig)

    if rows_stats:
        pd.DataFrame(rows_stats).sort_values("feature").to_csv(diag_dir / "feature_stats.csv", index=False)
    if rows_corr:
        corr_df = pd.DataFrame(rows_corr).sort_values("feature")
        corr_df.to_csv(diag_dir / "feature_corr.csv", index=False)

        # Bar plot of absolute correlations (Pearson).
        order = np.argsort(-np.abs(corr_df["pearson_r"].to_numpy(dtype=np.float64)))
        corr_df_ord = corr_df.iloc[order].reset_index(drop=True)

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4), constrained_layout=True)
        ax.bar(np.arange(len(corr_df_ord)), np.abs(corr_df_ord["pearson_r"].to_numpy(dtype=np.float64)))
        ax.set_xticks(np.arange(len(corr_df_ord)))
        ax.set_xticklabels(corr_df_ord["feature"].tolist(), rotation=45, ha="right")
        ax.set_ylabel("|Pearson r| with |error|")
        ax.set_title((title_prefix + "Feature predictiveness (|r|)").strip())
        ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(diag_dir / "feature_corr_bar.png", dpi=150)
        plt.close(fig)

    if rows_corr_signed:
        corr_df = pd.DataFrame(rows_corr_signed).sort_values("feature")
        corr_df.to_csv(diag_dir / "feature_corr_signed.csv", index=False)

        order = np.argsort(-np.abs(corr_df["pearson_r"].to_numpy(dtype=np.float64)))
        corr_df_ord = corr_df.iloc[order].reset_index(drop=True)

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4), constrained_layout=True)
        ax.bar(np.arange(len(corr_df_ord)), np.abs(corr_df_ord["pearson_r"].to_numpy(dtype=np.float64)))
        ax.set_xticks(np.arange(len(corr_df_ord)))
        ax.set_xticklabels(corr_df_ord["feature"].tolist(), rotation=45, ha="right")
        ax.set_ylabel("|Pearson r| with signed error")
        ax.set_title((title_prefix + "Feature predictiveness (signed error)").strip())
        ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(diag_dir / "feature_corr_signed_bar.png", dpi=150)
        plt.close(fig)


def save_jac_det_histograms_in_mask(
    *,
    out_path: str | Path,
    patient_ids: list[str],
    jac_list: list[np.ndarray],
    mask_list: list[np.ndarray],
    title: str = "Jacobian determinant in fixed mask",
    eps: float = 1e-6,
) -> None:
    """
    Save a grid of histograms (one per patient) for jac_det values within a binary mask.
    Uses log x-scale to reveal mass near 0.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(len(patient_ids))
    if n == 0:
        return

    vals = []
    fracs01 = []
    fracs001 = []
    for jac, m in zip(jac_list, mask_list, strict=True):
        jac = np.asarray(jac, dtype=np.float64)
        m = np.asarray(m) > 0
        v = jac[m]
        v = v[np.isfinite(v)]
        v = np.clip(v, eps, np.inf)
        vals.append(v)
        fracs01.append(float(np.mean(v < 0.1)) if v.size else float("nan"))
        fracs001.append(float(np.mean(v < 0.01)) if v.size else float("nan"))

    allv = np.concatenate([v for v in vals if v.size], axis=0) if any(v.size for v in vals) else np.array([1.0])
    lo = float(max(eps, np.min(allv)))
    hi = float(np.quantile(allv, 0.995))
    hi = max(hi, 2.0)
    bins = np.logspace(np.log10(lo), np.log10(hi), 70)

    cols = int(min(3, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.8 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i >= n:
            ax.axis("off")
            continue
        v = vals[i]
        pid = patient_ids[i]
        ax.hist(v, bins=bins, alpha=0.85, color="tab:blue")
        ax.set_xscale("log")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_title(f"{pid}\nfrac(J<0.1)={fracs01[i]:.3f}, frac(J<0.01)={fracs001[i]:.3f}", fontsize=10)
        ax.set_xlabel("Jacobian det (log x)")
        ax.set_ylabel("count")
        ax.grid(True, which="both", alpha=0.2)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_region_diagnostics(
    *,
    out_dir: str | Path,
    region_def: str,
    depth: np.ndarray,
    r_scp: np.ndarray,
    r_cmp: np.ndarray,
    title_prefix: str = "",
) -> None:
    """
    Minimal diagnostics to verify improvements:
      - |error| vs region depth (0..1) scatter, SCP vs ConVOLT
    Inputs are region-level pooled arrays over patients (same length).
    """
    out_dir = Path(out_dir) / "diagnostics" / "region"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    a = np.asarray(r_scp, dtype=np.float64).reshape(-1)
    b = np.asarray(r_cmp, dtype=np.float64).reshape(-1)
    m = np.isfinite(d) & np.isfinite(a) & np.isfinite(b)
    d, a, b = d[m], a[m], b[m]
    if d.size == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.scatter(d, a, s=10, alpha=0.35, label="SCP-point |err|", edgecolors="none")
    ax.scatter(d, b, s=10, alpha=0.35, label="ConVOLT-point |err|", edgecolors="none")
    ax.set_xlabel("region depth (0=boundary, 1=interior / slab base)")
    ax.set_ylabel("|region volume error| (mL)")
    ax.set_title(f"{title_prefix}{region_def}: |error| vs depth".strip())
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(out_dir / f"{region_def}_err_vs_depth.png", dpi=150)
    plt.close(fig)
