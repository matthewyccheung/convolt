#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing the local `reg` package when running as a script:
#   python scripts/make_paper_figures.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_pm_std(s: str) -> float:
    """
    Parse a 'mean±std' string and return mean as float.
    """
    if s is None:
        return float("nan")
    s = str(s).strip()
    if not s:
        return float("nan")
    # Accept variants: "123.4±56.7", "123.4 ± 56.7"
    s = s.replace(" ", "")
    if "±" in s:
        s = s.split("±", 1)[0]
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _load_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, skipinitialspace=True)


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_").lower()


def _inflation(ref: float, val: float) -> float:
    """
    Relative inflation: (val/ref - 1). Returns NaN if ref<=0 or non-finite.
    """
    if not np.isfinite(ref) or not np.isfinite(val) or ref <= 0:
        return float("nan")
    return float(val / ref - 1.0)


def _dataset_title(ds: str) -> str:
    return {
        "nlst": "NLST",
        "lungct": "ThoraxCBCT",
        "acdc": "ACDC",
        "oasis": "OASIS",
    }.get(ds, ds)


def _method_pretty(method: str) -> str:
    m = str(method)
    # Global experiments (claim 1/2).
    if m == "ConVOLT":
        return "ConVOLT (proposed)"
    if m == "ConVOLT(add-CP)":
        return "Additive scalar"
    if m == "ConVOLT(global1)":
        return "No learning (k=1)"
    if m == "ConVOLT(constk)":
        return "No features (const k)"

    # Region experiments (claim 3).
    if m == "ConVOLT(localfeat)":
        return "Local features"
    if m == "ConVOLT(globalfeat)":
        return "Global features"
    if m == "SCP":
        return "SCP"
    return m


def _barplot_inflations(
    *,
    tables_dir: Path,
    out_dir: Path,
    datasets: list[str],
    backends: list[str],
    region_score: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    c1 = _load_long(tables_dir / "claim1_additive_vs_multiplicative_long.csv")
    c2 = _load_long(tables_dir / "claim2_learning_ablations_long.csv")
    # Use full region long table because claim3 table is ConVOLT-only by design (SCP needed for this figure).
    rlong = _load_long(tables_dir / "region_results_long.csv")

    def _pick(df: pd.DataFrame, ds: str, backend: str, method: str) -> float:
        d = df[(df["dataset"] == ds) & (df["backend"] == backend) & (df["uq_method"] == method)]
        if len(d) == 0:
            return float("nan")
        return _parse_pm_std(d.iloc[0]["interval_size_mean_pm_std_ml"])

    def _pick_region(ds: str, backend: str, method: str) -> float:
        d = rlong[
            (rlong["dataset"] == ds)
            & (rlong["backend"] == backend)
            & (rlong["region_score"].astype(str).str.lower() == str(region_score).lower())
            & (rlong["uq_method"] == method)
        ]
        if len(d) == 0:
            return float("nan")
        return _parse_pm_std(d.iloc[0]["interval_size_mean_pm_std_ml"])

    panels = [{"dataset": ds} for ds in datasets]

    # Matplotlib style: conference-friendly.
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    ncols = len(panels)
    nrows = len(backends)
    if ncols == 0 or nrows == 0:
        return

    # 1-inch margins on US letter => ~6.5" usable width. Use a bit wider for 3 panels.
    fig_w = 6.8 if ncols <= 3 else 7.2
    fig_h = 2.5 * float(nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=False)
    if nrows == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    # Common x layout: 3 groups (Claim1, Claim2, Claim3)
    # Claim1: [Additive]
    # Claim2: [No learning, No features]
    # Claim3: [Global feat, SCP] (relative to localfeat within region, but shown on same axis)
    x = np.array([0, 2, 3, 5, 6], dtype=float)
    labels = ["Additive", "k=1", "Const k", "Global", "SCP"]
    colors = ["#4C78A8", "#F58518", "#F58518", "#54A24B", "#9D755D"]

    df_main = _load_long(tables_dir / "main_results_long_main.csv")
    for r, backend in enumerate(backends):
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ds = p["dataset"]

            ref = _pick(df_main, ds, backend, "ConVOLT")
            add = _pick(c1, ds, backend, "ConVOLT(add-CP)")
            g1 = _pick(c2, ds, backend, "ConVOLT(global1)")
            ck = _pick(c2, ds, backend, "ConVOLT(constk)")
            ref_reg = _pick_region(ds, backend, "ConVOLT(localfeat)")
            glb_reg = _pick_region(ds, backend, "ConVOLT(globalfeat)")
            scp_reg = _pick_region(ds, backend, "SCP")

            # Use a log y-scale => plot multiplicative ratios (always positive if defined).
            has_region = np.isfinite(ref_reg) and ref_reg > 0 and (np.isfinite(glb_reg) or np.isfinite(scp_reg))
            if has_region:
                x_use = x
                labels_use = labels
                vals_ratio = [
                    (add / ref) if np.isfinite(add) and np.isfinite(ref) and ref > 0 else float("nan"),
                    (g1 / ref) if np.isfinite(g1) and np.isfinite(ref) and ref > 0 else float("nan"),
                    (ck / ref) if np.isfinite(ck) and np.isfinite(ref) and ref > 0 else float("nan"),
                    (glb_reg / ref_reg) if np.isfinite(glb_reg) and ref_reg > 0 else float("nan"),
                    (scp_reg / ref_reg) if np.isfinite(scp_reg) and ref_reg > 0 else float("nan"),
                ]
                colors_use = colors
            else:
                # No region locality results (e.g., OASIS): drop Claim 3 section.
                x_use = np.array([0, 2, 3], dtype=float)
                labels_use = ["Additive", "k=1", "Const k"]
                vals_ratio = [
                    (add / ref) if np.isfinite(add) and np.isfinite(ref) and ref > 0 else float("nan"),
                    (g1 / ref) if np.isfinite(g1) and np.isfinite(ref) and ref > 0 else float("nan"),
                    (ck / ref) if np.isfinite(ck) and np.isfinite(ref) and ref > 0 else float("nan"),
                ]
                colors_use = ["#4C78A8", "#F58518", "#F58518"]

            # Keep gridlines behind bars for readability.
            ax.set_axisbelow(True)
            ax.bar(x_use, vals_ratio, width=0.8, color=colors_use, edgecolor="black", linewidth=0.5, zorder=2)

            ax.axvline(1.0, color="0.85", lw=1.0)
            if has_region:
                ax.axvline(4.0, color="0.85", lw=1.0)
            if r == 0:
                ax.set_title(_dataset_title(ds))
            ax.set_xticks(x_use, labels=labels_use, rotation=0)
            ax.set_xlim(-0.8, (6.8 if has_region else 3.8))
            ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
            ax.set_yscale("log")
            ax.set_ylim(0.6, 30.0)

            if r == (nrows - 1):
                ax.text(0.0, -0.18, "Claim 1", ha="center", va="top", transform=ax.get_xaxis_transform())
                ax.text(2.5, -0.18, "Claim 2", ha="center", va="top", transform=ax.get_xaxis_transform())
                if has_region:
                    ax.text(5.5, -0.18, "Claim 3", ha="center", va="top", transform=ax.get_xaxis_transform())

            if c == 0:
                ax.set_ylabel(f"{backend}\nInterval size ratio")

    fig.suptitle("Ablation interval size ratio (log scale)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ablation_inflation_2row.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_ablation_inflation_2row.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _corr_vs_efficiency(
    *,
    tables_dir: Path,
    uq_root: Path,
    out_dir: Path,
    datasets: list[str] | None,
    topk: int,
    baseline_uq_method: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    df_main = _load_long(tables_dir / "main_results_long_main.csv")
    baseline_uq_method = str(baseline_uq_method).strip()
    # Need ConVOLT + baseline per run.
    df_main = df_main[df_main["uq_method"].isin(["ConVOLT", baseline_uq_method])].copy()
    # Normalize join keys.
    df_main["dataset"] = df_main["dataset"].astype(str).str.strip().str.lower()
    df_main["backend"] = df_main["backend"].astype(str).str.strip().str.lower()
    df_main["run"] = df_main["run"].astype(str).str.strip()

    if datasets:
        want = {str(d).strip().lower() for d in datasets}
        df_main = df_main[df_main["dataset"].isin(sorted(want))].copy()

    # Keep only the primary backends for correlation plots:
    # - exact match on the 2nd token in the run name (e.g. nlst_demons, oasis_voxelmorph_supervised_...).
    # This drops runs like nlst_sitk_diffeomorphic_demons (which also contains "demons") and avoids double-counting.
    def _is_primary_run(run: str, backend: str) -> bool:
        parts = str(run).split("_")
        if len(parts) < 2:
            return False
        return parts[1] == str(backend)

    df_main = df_main[df_main.apply(lambda r: _is_primary_run(r["run"], r["backend"]), axis=1)].copy()

    # If there are multiple runs per dataset/backend (e.g. *_globalfeat), pick a single canonical run to avoid
    # double counting.
    def _run_rank(run: str) -> tuple[int, int]:
        run = str(run)
        # Prefer runs without globalfeat suffix, then prefer shorter names.
        return (1 if "globalfeat" in run else 0, len(run))

    keep = []
    for (ds, backend), g in df_main.groupby(["dataset", "backend"], as_index=False):
        runs = sorted(g["run"].unique().tolist(), key=_run_rank)
        keep.append(runs[0])
    df_main = df_main[df_main["run"].isin(keep)].copy()

    # Pivot to get interval sizes.
    rows = []
    for (ds, backend, run), g in df_main.groupby(["dataset", "backend", "run"], as_index=False):
        s = g[g["uq_method"] == baseline_uq_method]
        c = g[g["uq_method"] == "ConVOLT"]
        if len(s) == 0 or len(c) == 0:
            continue
        scp = _parse_pm_std(s.iloc[0]["interval_size_mean_pm_std_ml"])
        conv = _parse_pm_std(c.iloc[0]["interval_size_mean_pm_std_ml"])
        if not np.isfinite(scp) or not np.isfinite(conv) or scp <= 0:
            continue
        eff = float((scp - conv) / scp)

        corr_path = uq_root / str(run) / "diagnostics" / "feature_corr.csv"
        if not corr_path.exists():
            continue
        try:
            fc = pd.read_csv(corr_path)
        except Exception:
            continue
        if "pearson_r" not in fc.columns:
            continue
        top = fc["pearson_r"].abs().sort_values(ascending=False).head(int(topk)).to_numpy(dtype=float)
        top = top[np.isfinite(top)]
        if top.size == 0:
            continue
        corr_mean = float(np.mean(top))
        rows.append({"dataset": ds, "backend": backend, "run": run, "corr_topk_mean": corr_mean, "eff_gain": eff})

    if len(rows) == 0:
        return
    pts = pd.DataFrame(rows)
    # Save the underlying points for debugging/reproducibility.
    safe_base = re.sub(r"[^a-zA-Z0-9]+", "_", str(baseline_uq_method)).strip("_").lower()
    pts.to_csv(out_dir / f"corrmean_vs_efficiency_vs_{safe_base}_points_top{int(topk)}.csv", index=False)

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.8))
    # Color by backend.
    colors = {"demons": "#4C78A8", "voxelmorph": "#F58518"}

    def _stable_jitter(key: str, amp: float) -> float:
        """
        Deterministic jitter in [-amp, +amp] from an arbitrary string key.
        """
        s = str(key)
        # Lightweight, stable hash without importing hashlib.
        h = 0
        for ch in s:
            h = (h * 131 + ord(ch)) % 104729
        u = (h / 104728.0)  # in [0,1]
        return float((u - 0.5) * 2.0 * amp)

    for b, gb in pts.groupby("backend"):
        # Per-point jitter to avoid exact overlaps (e.g., identical eff_gain across datasets) and keep all
        # points visible. Jitter is deterministic (based on run name).
        x = []
        y = []
        for run, xv, yv in zip(gb["run"].astype(str).tolist(), gb["corr_topk_mean"].to_numpy(dtype=float), gb["eff_gain"].to_numpy(dtype=float)):
            x.append(float(xv) + _stable_jitter(f"x|{topk}|{b}|{run}", amp=0.012))
            y.append(float(yv) + _stable_jitter(f"y|{topk}|{b}|{run}", amp=0.012))
        ax.scatter(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            s=40,
            alpha=0.92,
            label=str(b),
            color=colors.get(str(b), None),
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )

    # Optional trend line (simple least squares).
    x = pts["corr_topk_mean"].to_numpy(dtype=float)
    y = pts["eff_gain"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(ok)) >= 2:
        A = np.vstack([x[ok], np.ones(np.count_nonzero(ok))]).T
        slope, intercept = np.linalg.lstsq(A, y[ok], rcond=None)[0]
        xs = np.linspace(float(np.min(x[ok])), float(np.max(x[ok])), 100)
        ax.plot(xs, slope * xs + intercept, color="0.25", lw=1.0, alpha=0.8)

    ax.set_xlabel(f"Mean of top-{int(topk)} |feature-error| correlations")
    ax.set_ylabel(f"Relative efficiency gain vs {baseline_uq_method}")
    ax.set_title(f"Feature signal vs ConVOLT efficiency (baseline={baseline_uq_method})")
    ax.set_axisbelow(True)
    ax.grid(color="0.9", lw=0.8, zorder=0)
    ax.margins(x=0.08, y=0.12)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    safe_base = re.sub(r"[^a-zA-Z0-9]+", "_", str(baseline_uq_method)).strip("_").lower()
    fig.savefig(out_dir / f"fig_corrmean_vs_efficiency_vs_{safe_base}_top{int(topk)}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_corrmean_vs_efficiency_vs_{safe_base}_top{int(topk)}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _feature_scatter_grid_overlay(
    *,
    tables_dir: Path,
    uq_root: Path,
    results_root: Path,
    out_dir: Path,
    datasets: list[str],
    backends: list[str],
    topk: int,
) -> None:
    """
    Overlay demons + voxelmorph in the same scatter-grid (per dataset, per feature).

    For each dataset, select a common top-k feature set by mean |Pearson r| with |error| across (dataset, backend),
    then plot a grid of scatter plots:
      rows = datasets, cols = features

    Each subplot shows:
      - scatter points for each backend in a different color
      - per-backend linear fit line
      - per-backend Pearson r and slope annotation
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which results folder (run name) corresponds to each (dataset, backend) in the main table.
    main = _load_long(tables_dir / "main_results_long_main.csv")
    main = main[(main["uq_method"] == "ConVOLT") & (main["backend"].isin(backends))].copy()
    main["dataset"] = main["dataset"].astype(str).str.strip().str.lower()
    main["backend"] = main["backend"].astype(str).str.strip().str.lower()
    main["run"] = main["run"].astype(str).str.strip()

    # Keep only the primary run names for each backend (avoid sitk_*_demons etc).
    def _is_primary_run(run: str, backend: str) -> bool:
        parts = str(run).split("_")
        if len(parts) < 2:
            return False
        return parts[1] == str(backend)

    main = main[main.apply(lambda r: _is_primary_run(r["run"], r["backend"]), axis=1)].copy()

    def _run_rank(run: str) -> tuple[int, int]:
        run = str(run)
        # Prefer runs without globalfeat suffix, then prefer shorter names.
        return (1 if "globalfeat" in run else 0, len(run))

    # For scatter plots we need raw per-patient feature values. These are computed from artifacts via the same
    # UQ feature extractor, but cached here to avoid repeatedly re-reading artifacts.
    from reg.uq.io import load_registration_results_with_features

    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    per_ds_backend: dict[tuple[str, str], dict] = {}
    r_table: dict[tuple[str, str], dict[str, float]] = {}

    # Resolve canonical runs for each (dataset, backend).
    run_map: dict[tuple[str, str], str] = {}
    for (ds, backend), g in main.groupby(["dataset", "backend"], as_index=False):
        if ds not in datasets:
            continue
        runs = sorted(g["run"].unique().tolist(), key=_run_rank)
        if runs:
            run_map[(ds, backend)] = runs[0]

    for ds in datasets:
        for backend in backends:
            run = run_map.get((ds, backend))
            if not run:
                continue
            results_dir = results_root / run
            if not results_dir.exists():
                continue

            cache_path = cache_dir / f"feature_scatter_{run}.csv"
            if cache_path.exists():
                df = pd.read_csv(cache_path)
            else:
                loaded = load_registration_results_with_features(results_dir=results_dir, require_artifacts=True)
                df = loaded.df.copy()
                # Store only what we need for scatter.
                keep_cols = ["patient_id"]
                if "vol_union_ml_gt" in df.columns and "vol_union_ml_pred0" in df.columns:
                    keep_cols += ["vol_union_ml_gt", "vol_union_ml_pred0"]
                if "delta_vol_ml_gt" in df.columns and "delta_vol_ml_pred0" in df.columns:
                    keep_cols += ["delta_vol_ml_gt", "delta_vol_ml_pred0"]
                keep_cols += [c for c in loaded.diagnostic_feature_keys if c in df.columns]
                df = df.loc[:, list(dict.fromkeys(keep_cols))].copy()
                df.to_csv(cache_path, index=False)

            # Choose a dataset-appropriate baseline error.
            if "vol_union_ml_gt" in df.columns and "vol_union_ml_pred0" in df.columns:
                abs_err = np.abs(df["vol_union_ml_pred0"].to_numpy(dtype=np.float64) - df["vol_union_ml_gt"].to_numpy(dtype=np.float64))
            elif "delta_vol_ml_gt" in df.columns and "delta_vol_ml_pred0" in df.columns:
                abs_err = np.abs(df["delta_vol_ml_pred0"].to_numpy(dtype=np.float64) - df["delta_vol_ml_gt"].to_numpy(dtype=np.float64))
            else:
                continue

            feature_cols = [c for c in df.columns if c not in {"patient_id", "vol_union_ml_gt", "vol_union_ml_pred0", "delta_vol_ml_gt", "delta_vol_ml_pred0"}]
            feats = {}
            r_table[(ds, backend)] = {}
            for f in feature_cols:
                x = df[f].to_numpy(dtype=np.float64)
                ok = np.isfinite(x) & np.isfinite(abs_err)
                if int(np.count_nonzero(ok)) < 10:
                    continue
                xr = x[ok]
                er = abs_err[ok]
                sx = float(np.std(xr))
                se = float(np.std(er))
                if sx <= 0 or se <= 0 or not np.isfinite(sx) or not np.isfinite(se):
                    r = 0.0
                else:
                    r = float(np.corrcoef(xr, er)[0, 1])
                    if not np.isfinite(r):
                        r = 0.0
                feats[f] = {"r": r, "x": xr, "e": er}
                r_table[(ds, backend)][f] = float(abs(r))

            if len(feats) == 0:
                continue
            per_ds_backend[(ds, backend)] = {"run": run, "data": feats}

    if len(per_ds_backend) == 0:
        return

    # Choose a common set of top-k features by mean |r| across (dataset, backend) pairs.
    all_feats = sorted({f for k in r_table for f in r_table[k].keys()})
    mean_abs = {}
    for f in all_feats:
        vals = [r_table[k].get(f, np.nan) for k in r_table.keys()]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 0:
            continue
        mean_abs[f] = float(np.mean(vals))
    feat_cols = [f for f, _ in sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)[: int(topk)]]
    if len(feat_cols) == 0:
        return

    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})

    row_ds = [ds for ds in datasets if any((ds, b) in per_ds_backend for b in backends)]
    nrows = len(row_ds)
    ncols = len(feat_cols)
    fig_w = min(7.2, 1.6 * ncols + 1.6)
    fig_h = 1.6 * nrows + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    colors = {"demons": "#4C78A8", "voxelmorph": "#F58518"}
    legend_done = False
    for i, ds in enumerate(row_ds):
        for j, f in enumerate(feat_cols):
            ax = axes[i, j]
            any_plotted = False
            ann_lines = []
            for backend in backends:
                entry = per_ds_backend.get((ds, backend))
                if entry is None:
                    continue
                d = entry["data"].get(f)
                if d is None:
                    continue
                x = d["x"]
                e = d["e"]
                if x.size == 0:
                    continue
                any_plotted = True
                col = colors.get(backend, None)
                ax.scatter(x, e, s=7, alpha=0.32, color=col, edgecolor="none", label=backend if not legend_done else None)

                # Fit y = a*x + b
                if x.size >= 2:
                    a, b = np.polyfit(x, e, deg=1)
                    xs = np.linspace(float(np.min(x)), float(np.max(x)), 50)
                    ax.plot(xs, a * xs + b, color=col, lw=1.1, alpha=0.85)
                    slope = float(a)
                else:
                    slope = float("nan")

                r = float(d["r"])
                prefix = "D" if backend == "demons" else ("V" if backend == "voxelmorph" else backend[:1].upper())
                ann_lines.append(f"{prefix} r={r:.2f} m={slope:.2g}")

            if not any_plotted:
                ax.axis("off")
                continue

            ax.text(
                0.02,
                0.98,
                "\n".join(ann_lines),
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
            )

            if i == nrows - 1:
                ax.set_xlabel(f)
            if j == 0:
                ax.set_ylabel(_dataset_title(ds))
            ax.grid(color="0.92", lw=0.6)
            if not legend_done:
                ax.legend(frameon=False, loc="lower right", fontsize=7, markerscale=1.2)
                legend_done = True

    fig.suptitle(f"Top-{int(topk)} features vs |error| (demons vs voxelmorph)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_feature_scatter_grid_overlay.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_feature_scatter_grid_overlay.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _oasis_label_inflation(
    *,
    uq_root: Path,
    out_dir: Path,
    topk: int,
    backend: str,
) -> None:
    """
    For OASIS, plot per-label interval-size ratio (log scale) vs ConVOLT for main baselines.
    Requires that UQ was run with per-label targets (volume_label).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick a run folder for this backend.
    candidates = sorted([p for p in uq_root.glob("oasis_*") if p.is_dir() and re.search(rf"(?:^|_)({re.escape(backend)})(?:_|$)", p.name)])
    if len(candidates) == 0:
        return
    # Prefer atlas-multi5 demons/voxelmorph unsupervised/supervised if present.
    run_dir = candidates[0]
    cp = run_dir / "cp_summary.csv"
    if not cp.exists():
        return

    df = pd.read_csv(cp, skipinitialspace=True)
    if "target" not in df.columns:
        return
    d = df[(df["target"] == "volume_label") & (df["label_id"].astype(int) >= 0)].copy()
    if len(d) == 0:
        return
    # Methods to compare (interval sizes vs ConVOLT).
    methods = ["SCP(|err|)", "CQR(volonly)", "LocalSCP(|err|; s=abs_pred, k=50)"]
    conv = d[d["method"] == "ConVOLT(scale-CP)"].copy()
    if len(conv) == 0:
        return

    conv = conv.set_index("label_id")["interval_size_mean"].to_dict()
    labels = sorted(list(conv.keys()))
    # Choose top-k labels by ConVOLT interval size (stable and available).
    labels = sorted(labels, key=lambda lid: float(conv.get(lid, np.inf)))[: int(topk)]

    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.4))

    x0 = np.arange(len(labels), dtype=float)
    w = 0.25
    offsets = [-w, 0.0, w]
    colors = ["#9D755D", "#4C78A8", "#54A24B"]
    names = ["SCP", "CQR(volonly)", "LCP"]

    for off, meth, col, name in zip(offsets, methods, colors, names):
        dd = d[d["method"] == meth].set_index("label_id")["interval_size_mean"].to_dict()
        ratios = []
        for lid in labels:
            base = float(conv.get(lid, np.nan))
            v = float(dd.get(lid, np.nan))
            ratios.append((v / base) if np.isfinite(v) and np.isfinite(base) and base > 0 else np.nan)
        ax.bar(x0 + off, ratios, width=w * 0.9, color=col, edgecolor="black", linewidth=0.4, label=name)

    ax.axhline(1.0, color="0.25", lw=1.0, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Interval size ratio vs ConVOLT (log)")
    ax.set_xticks(x0, labels=[str(int(l)) for l in labels])
    ax.set_xlabel("Label ID (top-k)")
    ax.set_title(f"OASIS per-label interval inflation vs ConVOLT ({backend})")
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_oasis_label_inflation_{backend}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_oasis_label_inflation_{backend}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _oasis_label_cqr_interval_inflation(
    *,
    tables_dir: Path,
    uq_root: Path,
    out_dir: Path,
    backend: str,
    cqr_method: str,
) -> None:
    """
    Bar plot (per-label) of interval inflation (%) for CQR vs ConVOLT on OASIS.

    Requires that UQ was run with per-label targets (volume_label) for the same run that appears in the main table.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the run from the main table to keep the comparison consistent with the paper tables.
    main = _load_long(tables_dir / "main_results_long_main.csv")
    main["dataset"] = main["dataset"].astype(str).str.strip().str.lower()
    main["backend"] = main["backend"].astype(str).str.strip().str.lower()
    main = main[(main["dataset"] == "oasis") & (main["backend"] == backend) & (main["uq_method"] == "ConVOLT")].copy()
    if len(main) == 0:
        print(f"[oasis-label] skip: no main-table run for backend={backend}")
        return
    run = str(main.iloc[0]["run"]).strip()

    cp_path = uq_root / run / "cp_summary.csv"
    if not cp_path.exists():
        print(f"[oasis-label] skip: missing {cp_path}")
        return

    try:
        df = pd.read_csv(cp_path, skipinitialspace=True)
    except Exception as e:
        print(f"[oasis-label] skip: failed to read {cp_path}: {e}")
        return

    if "target" not in df.columns or "label_id" not in df.columns or "method" not in df.columns:
        print(f"[oasis-label] skip: {cp_path} missing required columns")
        return

    d = df[(df["target"] == "volume_label") & (df["label_id"].astype(int) >= 0)].copy()
    if len(d) == 0:
        print(f"[oasis-label] skip: {run} has no volume_label rows; rerun UQ with --uq_target volume_label")
        return

    conv_name = "ConVOLT(scale-CP)"
    if conv_name not in set(d["method"].astype(str)):
        print(f"[oasis-label] skip: {run} missing {conv_name} rows for volume_label")
        return

    # Resolve which CQR method name is present.
    meths = set(d["method"].astype(str))
    cqr_pick = None
    if str(cqr_method) in meths:
        cqr_pick = str(cqr_method)
    else:
        # Prefer volonly if available (output-space baseline).
        for cand in ("CQR(volonly)", "CQR", "CQR(k)"):
            if cand in meths:
                cqr_pick = cand
                break
    if cqr_pick is None:
        print(f"[oasis-label] skip: {run} has no CQR rows for volume_label (available: {sorted(meths)})")
        return

    conv = d[d["method"].astype(str) == conv_name].set_index("label_id")["interval_size_mean"].to_dict()
    cqr = d[d["method"].astype(str) == cqr_pick].set_index("label_id")["interval_size_mean"].to_dict()

    label_ids = sorted({int(x) for x in conv.keys() if int(x) >= 0})
    infl = []
    keep = []
    for lid in label_ids:
        base = float(conv.get(lid, np.nan))
        val = float(cqr.get(lid, np.nan))
        if not np.isfinite(base) or base <= 0 or not np.isfinite(val):
            continue
        keep.append(lid)
        infl.append(100.0 * (val / base - 1.0))
    if len(keep) == 0:
        print(f"[oasis-label] skip: no finite label inflations for run={run} backend={backend}")
        return

    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig_w = min(7.2, 0.35 * len(keep) + 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 2.4))
    ax.set_axisbelow(True)

    x = np.arange(len(keep), dtype=float)
    ax.bar(x, infl, color="#4C78A8", edgecolor="black", linewidth=0.4, zorder=2)
    ax.axhline(0.0, color="0.25", lw=1.0, alpha=0.7)
    ax.set_xticks(x, labels=[str(int(l)) for l in keep], rotation=0)
    ax.set_xlabel("Label ID")
    ax.set_ylabel("Interval inflation vs ConVOLT (%)")
    ax.set_title(f"OASIS: CQR vs ConVOLT interval inflation ({backend}, {cqr_pick})")
    ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)

    fig.tight_layout()
    fig.savefig(out_dir / f"fig_oasis_label_cqr_inflation_{backend}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_oasis_label_cqr_inflation_{backend}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _oasis_label_interval_inflation(
    *,
    tables_dir: Path,
    uq_root: Path,
    out_dir: Path,
    backend: str,
    method: str,
    baseline: str,
) -> None:
    """
    Bar plot (per-label) of interval inflation (%) for `method` vs `baseline` on OASIS:
      inflation[%] = 100 * (interval_method / interval_baseline - 1)

    Requires that UQ was run with per-label targets (volume_label) for the same run that appears in the main table.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def _pick_run_for_backend(*, backend: str, method: str, baseline: str) -> str | None:
        """
        Prefer the run from the main table (if present). Otherwise, search uq_root for any OASIS run for this backend
        that contains volume_label rows and can resolve both method and baseline.
        """
        # (A) Try main-table run first (keeps comparisons aligned with main paper tables) IF it contains the requested methods.
        try:
            main = _load_long(tables_dir / "main_results_long_main.csv")
            main["dataset"] = main["dataset"].astype(str).str.strip().str.lower()
            main["backend"] = main["backend"].astype(str).str.strip().str.lower()
            mm = main[(main["dataset"] == "oasis") & (main["backend"] == backend) & (main["uq_method"] == "ConVOLT")].copy()
            if len(mm) > 0:
                run0 = str(mm.iloc[0]["run"]).strip()
                cp0 = uq_root / run0 / "cp_summary.csv"
                if cp0.exists():
                    try:
                        df0 = pd.read_csv(cp0, skipinitialspace=True)
                        if "target" in df0.columns and "label_id" in df0.columns and "method" in df0.columns:
                            d0 = df0[(df0["target"].astype(str) == "volume_label") & (df0["label_id"].astype(int) >= 0)].copy()
                            if len(d0) > 0:
                                meths0 = set(d0["method"].astype(str).tolist())
                                if str(method) in meths0 and str(baseline) in meths0:
                                    return run0
                    except Exception:
                        pass
        except Exception:
            pass

        # (B) Fallback: find any run with the requested methods in cp_summary volume_label.
        # Prefer runs with the most label coverage (e.g. "*_alllabels") to avoid accidentally selecting a top-k label run.
        candidates = [p for p in uq_root.glob("oasis_*") if p.is_dir() and re.search(rf"(?:^|_)({re.escape(backend)})(?:_|$)", p.name)]
        scored: list[tuple[tuple[int, int, int, int], str]] = []
        for run_dir in candidates:
            cp_path = run_dir / "cp_summary.csv"
            if not cp_path.exists():
                continue
            try:
                df = pd.read_csv(cp_path, skipinitialspace=True)
            except Exception:
                continue
            if "target" not in df.columns or "label_id" not in df.columns or "method" not in df.columns:
                continue
            d = df[(df["target"].astype(str) == "volume_label") & (df["label_id"].astype(int) >= 0)].copy()
            if len(d) == 0:
                continue
            meths = set(d["method"].astype(str))

            def _resolve(name: str) -> str | None:
                q = str(name).strip()
                if not q:
                    return None
                if q in meths:
                    return q
                ql = q.lower()
                aliases = {
                    "scp": "SCP(|err|)",
                    "lcp": "LocalSCP(|err|; s=abs_pred, k=50)",
                    "localscp": "LocalSCP(|err|; s=abs_pred, k=50)",
                    "convot": "ConVOLT(scale-CP)",
                    "convold": "ConVOLT(scale-CP)",
                    "convolt": "ConVOLT(scale-CP)",
                    "convolt(scale-cp)": "ConVOLT(scale-CP)",
                    "cqr": "CQR(volonly)",
                    "cqr(volonly)": "CQR(volonly)",
                    "cqr(k)": "CQR(k)",
                }
                if ql in aliases and aliases[ql] in meths:
                    return aliases[ql]
                for m in meths:
                    if str(m).strip().lower() == ql:
                        return str(m)
                pref = [str(m) for m in meths if str(m).strip().lower().startswith(ql)]
                if len(pref) == 1:
                    return pref[0]
                return None

            if _resolve(method) is not None and _resolve(baseline) is not None:
                n_labels = int(d["label_id"].astype(int).nunique())
                has_all = 1 if "alllabels" in run_dir.name.lower() else 0
                has_globalfeat = 1 if "globalfeat" in run_dir.name.lower() else 0
                # Prefer: more labels, explicit alllabels tag, non-globalfeat (keeps locality variants), then shorter name.
                score = (n_labels, has_all, -has_globalfeat, -len(run_dir.name))
                scored.append((score, run_dir.name))
        if scored:
            scored.sort(reverse=True, key=lambda x: x[0])
            return scored[0][1]
        return None

    run = _pick_run_for_backend(backend=backend, method=method, baseline=baseline)
    if run is None:
        print(f"[oasis-label] skip: could not find an OASIS run for backend={backend} with volume_label methods {method} and {baseline}")
        return

    cp_path = uq_root / run / "cp_summary.csv"
    try:
        df = pd.read_csv(cp_path, skipinitialspace=True)
    except Exception as e:
        print(f"[oasis-label] skip: failed to read {cp_path}: {e}")
        return

    if "target" not in df.columns or "label_id" not in df.columns or "method" not in df.columns:
        print(f"[oasis-label] skip: {cp_path} missing required columns")
        return

    d = df[(df["target"] == "volume_label") & (df["label_id"].astype(int) >= 0)].copy()
    if len(d) == 0:
        print(f"[oasis-label] skip: {run} has no volume_label rows; rerun Learn2Reg UQ volume suite (omit --uq_target volume_union)")
        return

    meths = set(d["method"].astype(str))

    def _resolve(name: str) -> str | None:
        """
        Resolve a user-provided method name to an exact cp_summary.csv `method` entry.
        Accepts case-insensitive matches and common aliases (e.g., 'scp' -> 'SCP(|err|)').
        """
        q = str(name).strip()
        if not q:
            return None
        if q in meths:
            return q
        ql = q.lower()
        aliases = {
            "scp": "SCP(|err|)",
            "lcp": "LocalSCP(|err|; s=abs_pred, k=50)",
            "localscp": "LocalSCP(|err|; s=abs_pred, k=50)",
            "convot": "ConVOLT(scale-CP)",
            "convold": "ConVOLT(scale-CP)",
            "convolt": "ConVOLT(scale-CP)",
            "convolt(scale-cp)": "ConVOLT(scale-CP)",
            "cqr": "CQR(volonly)",
            "cqr(volonly)": "CQR(volonly)",
            "cqr(k)": "CQR(k)",
        }
        if ql in aliases and aliases[ql] in meths:
            return aliases[ql]
        # Case-insensitive exact match.
        for m in meths:
            if str(m).strip().lower() == ql:
                return str(m)
        # Unique prefix match (e.g., "scp" matching "scp(|err|)").
        pref = [str(m) for m in meths if str(m).strip().lower().startswith(ql)]
        if len(pref) == 1:
            return pref[0]
        return None

    method_r = _resolve(method)
    baseline_r = _resolve(baseline)
    if method_r is None:
        print(f"[oasis-label] skip: {run} could not resolve method={method} (available: {sorted(meths)})")
        return
    if baseline_r is None:
        print(f"[oasis-label] skip: {run} could not resolve baseline={baseline} (available: {sorted(meths)})")
        return

    base = d[d["method"].astype(str) == str(baseline_r)].set_index("label_id")["interval_size_mean"].to_dict()
    val = d[d["method"].astype(str) == str(method_r)].set_index("label_id")["interval_size_mean"].to_dict()

    label_ids = sorted({int(x) for x in base.keys() if int(x) >= 0})
    infl = []
    keep = []
    for lid in label_ids:
        b = float(base.get(lid, np.nan))
        v = float(val.get(lid, np.nan))
        if not np.isfinite(b) or b <= 0 or not np.isfinite(v):
            continue
        keep.append(lid)
        infl.append(100.0 * (v / b - 1.0))
    if len(keep) == 0:
        print(f"[oasis-label] skip: no finite label inflations for run={run} backend={backend}")
        return

    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig_w = min(7.2, 0.35 * len(keep) + 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 2.4))
    ax.set_axisbelow(True)

    x = np.arange(len(keep), dtype=float)
    ax.bar(x, infl, color="#4C78A8", edgecolor="black", linewidth=0.4, zorder=2)
    ax.axhline(0.0, color="0.25", lw=1.0, alpha=0.7)
    ax.set_xticks(x, labels=[str(int(l)) for l in keep], rotation=0)
    ax.set_xlabel("Label ID")

    def _short(name: str) -> str:
        n = str(name).strip()
        if n.startswith("ConVOLT"):
            return "ConVOLT"
        if n.startswith("CQR"):
            return "CQR"
        if n.startswith("SCP"):
            return "SCP"
        if n.startswith("LocalSCP"):
            return "LCP"
        if n.startswith("wCP") or "wcp" in n.lower():
            return "RsCP"
        return n

    meth_s = re.sub(r"[^A-Za-z0-9]+", "", _short(method_r))
    base_s = re.sub(r"[^A-Za-z0-9]+", "", _short(baseline_r))
    ax.set_ylabel(rf"$100\times\left(\frac{{\mathrm{{{meth_s}\ Interval\ Size}}}}{{\mathrm{{{base_s}\ Interval\ Size}}}}-1\right)$ (%)")
    ax.set_title(f"OASIS: {method_r} vs {baseline_r} interval inflation ({backend})")
    ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    safe_method = re.sub(r"[^a-zA-Z0-9]+", "_", str(method_r)).strip("_").lower()
    safe_base = re.sub(r"[^a-zA-Z0-9]+", "_", str(baseline_r)).strip("_").lower()
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_oasis_label_inflation_{safe_method}_vs_{safe_base}_{backend}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_oasis_label_inflation_{safe_method}_vs_{safe_base}_{backend}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _oasis_label_interval_inflation_grid(
    *,
    tables_dir: Path,
    uq_root: Path,
    out_dir: Path,
    method: str,
    baseline: str,
    fig_w: float = 7.2,
    fig_h: float = 2.6,
) -> None:
    """
    One figure with 2 subplots (demons | voxelmorph) showing per-label interval inflation (%) for
    `method` vs `baseline` on OASIS.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def _compute(backend: str) -> tuple[list[int], list[float]] | None:
        # Prefer the main-table run if it contains the requested volume_label methods; otherwise pick any run that does.
        run = None
        try:
            main = _load_long(tables_dir / "main_results_long_main.csv")
            main["dataset"] = main["dataset"].astype(str).str.strip().str.lower()
            main["backend"] = main["backend"].astype(str).str.strip().str.lower()
            mm = main[(main["dataset"] == "oasis") & (main["backend"] == backend) & (main["uq_method"] == "ConVOLT")].copy()
            if len(mm) > 0:
                run0 = str(mm.iloc[0]["run"]).strip()
                cp0 = uq_root / run0 / "cp_summary.csv"
                if cp0.exists():
                    try:
                        df0 = pd.read_csv(cp0, skipinitialspace=True)
                        if "target" in df0.columns and "label_id" in df0.columns and "method" in df0.columns:
                            d0 = df0[(df0["target"].astype(str) == "volume_label") & (df0["label_id"].astype(int) >= 0)].copy()
                            if len(d0) > 0:
                                meths0 = set(d0["method"].astype(str).tolist())
                                if str(method) in meths0 and str(baseline) in meths0:
                                    run = run0
                    except Exception:
                        run = None
        except Exception:
            run = None
        if not run:
            candidates = [p for p in uq_root.glob("oasis_*") if p.is_dir() and re.search(rf"(?:^|_)({re.escape(backend)})(?:_|$)", p.name)]
            scored: list[tuple[tuple[int, int, int, int], str]] = []
            for run_dir in candidates:
                cp0 = run_dir / "cp_summary.csv"
                if not cp0.exists():
                    continue
                try:
                    df0 = pd.read_csv(cp0, skipinitialspace=True)
                except Exception:
                    continue
                if "target" not in df0.columns or "label_id" not in df0.columns or "method" not in df0.columns:
                    continue
                d0 = df0[(df0["target"].astype(str) == "volume_label") & (df0["label_id"].astype(int) >= 0)].copy()
                if len(d0) == 0:
                    continue
                meths0 = set(d0["method"].astype(str))

                def _resolve0(name: str) -> str | None:
                    q = str(name).strip()
                    if not q:
                        return None
                    if q in meths0:
                        return q
                    ql = q.lower()
                    aliases = {
                        "scp": "SCP(|err|)",
                        "lcp": "LocalSCP(|err|; s=abs_pred, k=50)",
                        "localscp": "LocalSCP(|err|; s=abs_pred, k=50)",
                        "convot": "ConVOLT(scale-CP)",
                        "convold": "ConVOLT(scale-CP)",
                        "convolt": "ConVOLT(scale-CP)",
                        "convolt(scale-cp)": "ConVOLT(scale-CP)",
                        "cqr": "CQR(volonly)",
                        "cqr(volonly)": "CQR(volonly)",
                        "cqr(k)": "CQR(k)",
                    }
                    if ql in aliases and aliases[ql] in meths0:
                        return aliases[ql]
                    for m in meths0:
                        if str(m).strip().lower() == ql:
                            return str(m)
                    pref = [str(m) for m in meths0 if str(m).strip().lower().startswith(ql)]
                    if len(pref) == 1:
                        return pref[0]
                    return None

                if _resolve0(method) is not None and _resolve0(baseline) is not None:
                    n_labels = int(d0["label_id"].astype(int).nunique())
                    has_all = 1 if "alllabels" in run_dir.name.lower() else 0
                    has_globalfeat = 1 if "globalfeat" in run_dir.name.lower() else 0
                    score = (n_labels, has_all, -has_globalfeat, -len(run_dir.name))
                    scored.append((score, run_dir.name))
            if scored:
                scored.sort(reverse=True, key=lambda x: x[0])
                run = scored[0][1]

        if not run:
            return None

        cp_path = uq_root / run / "cp_summary.csv"
        try:
            df = pd.read_csv(cp_path, skipinitialspace=True)
        except Exception:
            return None
        if "target" not in df.columns or "label_id" not in df.columns or "method" not in df.columns:
            return None
        d = df[(df["target"] == "volume_label") & (df["label_id"].astype(int) >= 0)].copy()
        if len(d) == 0:
            return None
        meths = set(d["method"].astype(str))

        def _resolve(name: str) -> str | None:
            q = str(name).strip()
            if not q:
                return None
            if q in meths:
                return q
            ql = q.lower()
            aliases = {
                "scp": "SCP(|err|)",
                "lcp": "LocalSCP(|err|; s=abs_pred, k=50)",
                "localscp": "LocalSCP(|err|; s=abs_pred, k=50)",
                "convot": "ConVOLT(scale-CP)",
                "convold": "ConVOLT(scale-CP)",
                "convolt": "ConVOLT(scale-CP)",
                "convolt(scale-cp)": "ConVOLT(scale-CP)",
                "cqr": "CQR(volonly)",
                "cqr(volonly)": "CQR(volonly)",
                "cqr(k)": "CQR(k)",
            }
            if ql in aliases and aliases[ql] in meths:
                return aliases[ql]
            for m in meths:
                if str(m).strip().lower() == ql:
                    return str(m)
            pref = [str(m) for m in meths if str(m).strip().lower().startswith(ql)]
            if len(pref) == 1:
                return pref[0]
            return None

        method_r = _resolve(method)
        baseline_r = _resolve(baseline)
        if method_r is None or baseline_r is None:
            return None

        base = d[d["method"].astype(str) == str(baseline_r)].set_index("label_id")["interval_size_mean"].to_dict()
        val = d[d["method"].astype(str) == str(method_r)].set_index("label_id")["interval_size_mean"].to_dict()
        label_ids = sorted({int(x) for x in base.keys() if int(x) >= 0})
        infl = []
        keep = []
        for lid in label_ids:
            b = float(base.get(lid, np.nan))
            v = float(val.get(lid, np.nan))
            if not np.isfinite(b) or b <= 0 or not np.isfinite(v):
                continue
            keep.append(lid)
            infl.append(100.0 * (v / b - 1.0))
        if len(keep) == 0:
            return None
        return keep, infl

    d_res = _compute("demons")
    v_res = _compute("voxelmorph")
    if d_res is None and v_res is None:
        print("[oasis-label] skip: no per-label inflation data for grid plot (need volume_label UQ).")
        return

    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(float(fig_w), float(fig_h)), sharey=True)

    def _short(name: str) -> str:
        n = str(name).strip()
        if n.startswith("ConVOLT"):
            return "ConVOLT"
        if n.startswith("CQR"):
            return "CQR"
        return n

    for ax, backend, res in zip(axes, ["demons", "voxelmorph"], [d_res, v_res]):
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if res is None:
            ax.axis("off")
            continue
        keep, infl = res
        # Summary stats: inflation[%] = 100*(width_method/width_baseline - 1).
        # Negative inflation means method is more efficient (smaller intervals) than baseline.
        infl_arr = np.asarray(infl, dtype=np.float64)
        ok = np.isfinite(infl_arr)
        if int(np.count_nonzero(ok)) > 0:
            mean_infl = float(np.mean(infl_arr[ok]))
            n_better = int(np.count_nonzero(infl_arr[ok] < 0.0))
            print(f"[oasis-label] {backend}: mean inflation (method vs baseline) = {mean_infl:.1f}% ; method better on {n_better}/{int(np.count_nonzero(ok))} labels")

        x = np.arange(len(keep), dtype=float)
        ax.bar(x, infl, color="#4C78A8", edgecolor="black", linewidth=0.4, zorder=2)
        ax.axhline(0.0, color="0.25", lw=1.0, alpha=0.7)
        # Paper-ready ticks: include first+last with evenly spaced intervals.
        if len(keep) <= 8:
            tick_idx = np.arange(len(keep), dtype=int)
        else:
            tick_idx = np.unique(np.round(np.linspace(0, len(keep) - 1, 8)).astype(int))
        ax.set_xticks(x[tick_idx], labels=[str(int(keep[ii])) for ii in tick_idx], rotation=0)
        ax.set_xlabel("Label ID")
        ax.set_title("Demons" if backend == "demons" else ("VoxelMorph" if backend == "voxelmorph" else backend))
        ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)

    meth_s = re.sub(r"[^A-Za-z0-9]+", "", _short(method))
    base_s = re.sub(r"[^A-Za-z0-9]+", "", _short(baseline))
    axes[0].set_ylabel(rf"$100\times\left(\frac{{\mathrm{{{meth_s}\ Interval\ Size}}}}{{\mathrm{{{base_s}\ Interval\ Size}}}}-1\right)$ (%)")
    fig.tight_layout()

    safe_method = re.sub(r"[^a-zA-Z0-9]+", "_", str(method)).strip("_").lower()
    safe_base = re.sub(r"[^a-zA-Z0-9]+", "_", str(baseline)).strip("_").lower()
    fig.savefig(out_dir / f"fig_oasis_label_inflation_{safe_method}_vs_{safe_base}_grid.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_oasis_label_inflation_{safe_method}_vs_{safe_base}_grid.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _oasis_label_error_scatter_with_bands(
    *,
    tables_dir: Path,
    uq_root: Path,
    results_root: Path,
    out_dir: Path,
    backend: str,
    method: str,
    baseline: str,
    topk: int,
) -> None:
    """
    Scatter: x = GT label volume per case, y = |error| per case (from label_volumes.csv),
    with horizontal bands showing typical interval half-width (width/2) for `method` and `baseline`
    (from cp_summary.csv volume_label rows; mean±std).
    """
    import json
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    backend = str(backend).strip().lower()

    main = _load_long(tables_dir / "main_results_long_main.csv")
    main["dataset"] = main["dataset"].astype(str).str.strip().str.lower()
    main["backend"] = main["backend"].astype(str).str.strip().str.lower()
    main = main[(main["dataset"] == "oasis") & (main["backend"] == backend) & (main["uq_method"] == "ConVOLT")].copy()
    if len(main) == 0:
        print(f"[oasis-label-scatter] skip: no main-table run for backend={backend}")
        return
    run = str(main.iloc[0]["run"]).strip()

    cp_path = uq_root / run / "cp_summary.csv"
    if not cp_path.exists():
        print(f"[oasis-label-scatter] skip: missing {cp_path}")
        return
    try:
        cp = pd.read_csv(cp_path, skipinitialspace=True)
    except Exception as e:
        print(f"[oasis-label-scatter] skip: failed to read {cp_path}: {e}")
        return
    if "target" not in cp.columns or "label_id" not in cp.columns or "method" not in cp.columns:
        print(f"[oasis-label-scatter] skip: {cp_path} missing required columns")
        return
    d = cp[(cp["target"].astype(str) == "volume_label") & (cp["label_id"].astype(int) >= 0)].copy()
    if len(d) == 0:
        print(f"[oasis-label-scatter] skip: {run} has no volume_label rows; rerun UQ with volume_label suite")
        return

    meths = set(d["method"].astype(str))

    def _resolve(name: str) -> str | None:
        q = str(name).strip()
        if not q:
            return None
        if q in meths:
            return q
        ql = q.lower()
        aliases = {
            "scp": "SCP(|err|)",
            "lcp": "LocalSCP(|err|; s=abs_pred, k=50)",
            "localscp": "LocalSCP(|err|; s=abs_pred, k=50)",
            "convot": "ConVOLT(scale-CP)",
            "convold": "ConVOLT(scale-CP)",
            "convolt": "ConVOLT(scale-CP)",
            "convolt(scale-cp)": "ConVOLT(scale-CP)",
            "cqr": "CQR(volonly)",
            "cqr(volonly)": "CQR(volonly)",
            "cqr(k)": "CQR(k)",
        }
        if ql in aliases and aliases[ql] in meths:
            return aliases[ql]
        for m in meths:
            if str(m).strip().lower() == ql:
                return str(m)
        pref = [str(m) for m in meths if str(m).strip().lower().startswith(ql)]
        if len(pref) == 1:
            return pref[0]
        return None

    method_r = _resolve(method)
    baseline_r = _resolve(baseline)
    if method_r is None:
        print(f"[oasis-label-scatter] skip: could not resolve method={method} (available: {sorted(meths)})")
        return
    if baseline_r is None:
        print(f"[oasis-label-scatter] skip: could not resolve baseline={baseline} (available: {sorted(meths)})")
        return

    def _width_map(name: str) -> dict[int, tuple[float, float]]:
        dd = d[d["method"].astype(str) == str(name)].copy()
        out: dict[int, tuple[float, float]] = {}
        for _, r in dd.iterrows():
            lid = int(r["label_id"])
            mu = float(r.get("interval_size_mean", np.nan))
            sig = float(r.get("interval_size_std", np.nan))
            if not np.isfinite(mu):
                continue
            if not np.isfinite(sig):
                sig = 0.0
            out[lid] = (mu / 2.0, sig / 2.0)
        return out

    w_method = _width_map(method_r)
    w_base = _width_map(baseline_r)
    if len(w_method) == 0 or len(w_base) == 0:
        print(f"[oasis-label-scatter] skip: missing interval widths for method/baseline in {cp_path}")
        return

    lv_path = results_root / run / "label_volumes.csv"
    if not lv_path.exists():
        print(f"[oasis-label-scatter] skip: missing {lv_path}")
        return
    lv = pd.read_csv(lv_path)
    need_cols = {"patient_id", "split", "backend", "label_id", "vol_ml_gt", "vol_ml_pred"}
    if not need_cols.issubset(set(lv.columns)):
        print(f"[oasis-label-scatter] skip: {lv_path} missing required columns")
        return
    lv = lv[(lv["split"].astype(str).str.lower() == "training") & (lv["backend"].astype(str).str.lower() == backend)].copy()
    lv["label_id"] = lv["label_id"].astype(int)
    lv = lv[lv["label_id"] > 0].copy()
    lv["abs_err_ml"] = np.abs(lv["vol_ml_gt"].to_numpy(dtype=float) - lv["vol_ml_pred"].to_numpy(dtype=float))

    ex_ids: set[str] = set()
    atlas_meta = results_root / run / "atlas_meta.json"
    if atlas_meta.exists():
        try:
            ex_ids.update([str(x) for x in json.loads(atlas_meta.read_text()).get("atlas_ids", [])])
        except Exception:
            pass
    vm_ids = results_root / run / "vm_train_ids.json"
    if vm_ids.exists():
        try:
            ex_ids.update([str(x) for x in json.loads(vm_ids.read_text()).get("vm_train_ids", [])])
        except Exception:
            pass
    if ex_ids:
        lv = lv[~lv["patient_id"].astype(str).isin(ex_ids)].copy()

    g = lv.groupby("label_id", as_index=False)["vol_ml_gt"].mean().rename(columns={"vol_ml_gt": "gt_mean"})
    g = g.sort_values("gt_mean", ascending=False).reset_index(drop=True)
    label_ids = g["label_id"].astype(int).tolist()
    if int(topk) > 0:
        label_ids = label_ids[: int(topk)]
    if len(label_ids) == 0:
        print(f"[oasis-label-scatter] skip: no labels found in {lv_path}")
        return

    n = len(label_ids)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig_w = 7.0
    fig_h = 2.2 + 1.7 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=200, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    lv_sel = lv[lv["label_id"].isin(label_ids)].copy()
    y_max = float(np.nanpercentile(lv_sel["abs_err_ml"].to_numpy(dtype=float), 99.0)) if len(lv_sel) else 1.0
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0

    col_method = "#4C78A8"
    col_base = "#54A24B"

    for ax_i, (ax, lid) in enumerate(zip(axes, label_ids)):
        dd = lv[lv["label_id"] == int(lid)].copy()
        x = dd["vol_ml_gt"].to_numpy(dtype=float)
        y = dd["abs_err_ml"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x = x[ok]
        y = y[ok]
        ax.scatter(x, y, s=8, alpha=0.55, color="0.15", edgecolor="none", zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.15 * y_max)
        ax.grid(color="0.92", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

        if int(lid) in w_base:
            mu, sig = w_base[int(lid)]
            lo = max(0.0, float(mu - sig))
            hi = max(0.0, float(mu + sig))
            ax.axhspan(lo, hi, color=col_base, alpha=0.18, zorder=1)
            ax.axhline(float(mu), color=col_base, lw=1.2, alpha=0.9, zorder=2)
        if int(lid) in w_method:
            mu, sig = w_method[int(lid)]
            lo = max(0.0, float(mu - sig))
            hi = max(0.0, float(mu + sig))
            ax.axhspan(lo, hi, color=col_method, alpha=0.16, zorder=1)
            ax.axhline(float(mu), color=col_method, lw=1.2, alpha=0.9, zorder=2)

        ax.set_title(f"Label {int(lid)}")
        if ax_i % ncols == 0:
            ax.set_ylabel("|error| (mL)")
        if ax_i >= (nrows - 1) * ncols:
            ax.set_xlabel("GT volume (mL, log)")

    for ax in axes[len(label_ids) :]:
        ax.axis("off")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=col_method, lw=2, label=f"{method_r} half-width"),
        Line2D([0], [0], color=col_base, lw=2, label=f"{baseline_r} half-width"),
        Line2D([0], [0], marker="o", color="0.15", lw=0, markersize=4, label="Cases (|err|)"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"OASIS label errors vs GT volume with UQ width bands ({backend})\\nrun={run}", y=1.06)

    safe_m = re.sub(r"[^a-zA-Z0-9]+", "_", str(method_r)).strip("_").lower()
    safe_b = re.sub(r"[^a-zA-Z0-9]+", "_", str(baseline_r)).strip("_").lower()
    fig.savefig(out_dir / f"fig_oasis_label_err_scatter_{safe_m}_vs_{safe_b}_{backend}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_oasis_label_err_scatter_{safe_m}_vs_{safe_b}_{backend}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _oasis_label_error_scatter_method_colored(
    *,
    tables_dir: Path,
    uq_root: Path,
    results_root: Path,
    out_dir: Path,
    backend: str,
    method: str,
    baseline: str,
    topk: int,
    alpha: float = 0.1,
    ridge_l2: float = 1e-3,
) -> None:
    """
    Method-colored scatter plots for OASIS labels:
      x = GT label volume (per case), y = |y - center_method| (per case),
    where center_method is method-specific:
      - CQR(volonly): center is midpoint of the conformalized interval
      - ConVOLT(scale-CP): center is k_hat(x) * y_pred0

    This requires refitting the UQ models to obtain per-case centers (not available in cp_summary.csv).
    """
    import json
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from reg.uq.conformal import conformal_quantile
    from reg.uq.models import fit_quantile_ridge, fit_ridge

    out_dir.mkdir(parents=True, exist_ok=True)
    backend = str(backend).strip().lower()
    alpha = float(alpha)

    main = _load_long(tables_dir / "main_results_long_main.csv")
    main["dataset"] = main["dataset"].astype(str).str.strip().str.lower()
    main["backend"] = main["backend"].astype(str).str.strip().str.lower()
    main = main[(main["dataset"] == "oasis") & (main["backend"] == backend) & (main["uq_method"] == "ConVOLT")].copy()
    if len(main) == 0:
        print(f"[oasis-label-colored] skip: no main-table run for backend={backend}")
        return
    run = str(main.iloc[0]["run"]).strip()

    # Load label targets and per-patient deformation features.
    results_dir = results_root / run
    cache_path = out_dir / f"cache_features_{_safe_name(run)}_{backend}.csv"
    if cache_path.exists():
        df_feat = pd.read_csv(cache_path)
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
            if k in df_feat.columns
        )
    else:
        from reg.uq.io import load_registration_results_with_features

        loaded = load_registration_results_with_features(results_dir=results_dir, require_artifacts=True)
        df_feat = loaded.df.copy()
        feature_keys = tuple(loaded.feature_keys)
        keep_cols = ["patient_id", *list(feature_keys)]
        keep_cols = [c for c in keep_cols if c in df_feat.columns]
        df_feat[keep_cols].to_csv(cache_path, index=False)
    if "patient_id" not in df_feat.columns:
        print(f"[oasis-label-colored] skip: missing patient_id in {results_dir}/summary.csv")
        return

    lv_path = results_dir / "label_volumes.csv"
    if not lv_path.exists():
        print(f"[oasis-label-colored] skip: missing {lv_path}")
        return
    lv = pd.read_csv(lv_path)
    need_cols = {"patient_id", "split", "backend", "label_id", "vol_ml_gt", "vol_ml_pred"}
    if not need_cols.issubset(set(lv.columns)):
        print(f"[oasis-label-colored] skip: {lv_path} missing required columns")
        return
    lv = lv[(lv["split"].astype(str).str.lower() == "training") & (lv["backend"].astype(str).str.lower() == backend)].copy()
    lv["label_id"] = lv["label_id"].astype(int)
    lv = lv[lv["label_id"] > 0].copy()

    # Exclude atlas/vm_train IDs (align with UQ pool).
    ex_ids: set[str] = set()
    atlas_meta = results_dir / "atlas_meta.json"
    if atlas_meta.exists():
        try:
            ex_ids.update([str(x) for x in json.loads(atlas_meta.read_text()).get("atlas_ids", [])])
        except Exception:
            pass
    vm_ids = results_dir / "vm_train_ids.json"
    if vm_ids.exists():
        try:
            ex_ids.update([str(x) for x in json.loads(vm_ids.read_text()).get("vm_train_ids", [])])
        except Exception:
            pass
    if ex_ids:
        df_feat = df_feat[~df_feat["patient_id"].astype(str).isin(ex_ids)].copy()
        lv = lv[~lv["patient_id"].astype(str).isin(ex_ids)].copy()

    # Determine split sizes from cp_runs (repeat 0) if available; otherwise default to 0.4/0.4/0.2.
    n_train = n_calib = n_test = None
    cp_runs = uq_root / run / "cp_runs.csv"
    if cp_runs.exists():
        try:
            cr = pd.read_csv(cp_runs, skipinitialspace=True)
            cr = cr[(cr["repeat"].astype(int) == 0) & (cr["target"].astype(str) == "volume_union")].copy()
            if len(cr) > 0:
                n_train = int(cr.iloc[0]["n_train"])
                n_calib = int(cr.iloc[0]["n_calib"])
                n_test = int(cr.iloc[0]["n_test"])
        except Exception:
            n_train = n_calib = n_test = None

    pids = df_feat["patient_id"].astype(str).tolist()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pids))
    if n_train is None or n_calib is None or n_test is None:
        n = len(pids)
        n_train = int(np.floor(0.4 * n))
        n_calib = int(np.floor(0.4 * n))
        n_test = max(1, n - n_train - n_calib)
    if n_train + n_calib + n_test > len(pids):
        n_test = max(1, len(pids) - n_train - n_calib)
    train_ids = {pids[i] for i in perm[:n_train]}
    calib_ids = {pids[i] for i in perm[n_train : n_train + n_calib]}
    test_ids = {pids[i] for i in perm[n_train + n_calib : n_train + n_calib + n_test]}

    # Pick top-k labels by mean GT volume.
    gl = lv.groupby("label_id", as_index=False)["vol_ml_gt"].mean().rename(columns={"vol_ml_gt": "gt_mean"})
    label_ids = gl.sort_values("gt_mean", ascending=False)["label_id"].astype(int).tolist()
    if int(topk) > 0:
        label_ids = label_ids[: int(topk)]
    if len(label_ids) == 0:
        return

    # Helper: impute non-finite features by train col means; all-nonfinite columns become 0.
    def _impute(X: np.ndarray, col_mean: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        finite = np.isfinite(X)
        return np.where(finite, X, col_mean)

    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    n = len(label_ids)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.2 + 1.7 * nrows), dpi=200, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    col_method = "#4C78A8"
    col_base = "#54A24B"

    for ax, lid in zip(axes, label_ids):
        sub = lv[lv["label_id"] == int(lid)][["patient_id", "vol_ml_gt", "vol_ml_pred"]].copy()
        df_t = df_feat.merge(sub, on="patient_id", how="inner")
        if len(df_t) < max(30, n_train + n_calib + 5):
            ax.axis("off")
            continue

        df_tr = df_t[df_t["patient_id"].astype(str).isin(train_ids)].copy()
        df_cal = df_t[df_t["patient_id"].astype(str).isin(calib_ids)].copy()
        df_te = df_t[df_t["patient_id"].astype(str).isin(test_ids)].copy()
        if len(df_tr) < 10 or len(df_cal) < 10 or len(df_te) < 5:
            ax.axis("off")
            continue

        y_tr = df_tr["vol_ml_gt"].to_numpy(dtype=np.float32)
        y0_tr = df_tr["vol_ml_pred"].to_numpy(dtype=np.float32)
        y_cal = df_cal["vol_ml_gt"].to_numpy(dtype=np.float32)
        y0_cal = df_cal["vol_ml_pred"].to_numpy(dtype=np.float32)
        y_te = df_te["vol_ml_gt"].to_numpy(dtype=np.float32)
        y0_te = df_te["vol_ml_pred"].to_numpy(dtype=np.float32)

        # CQR(volonly): fit quantiles on [pred0] only.
        tau_lo = alpha / 2.0
        tau_hi = 1.0 - alpha / 2.0
        X_tr0 = y0_tr.reshape(-1, 1)
        ok_tr0 = np.isfinite(y_tr) & np.all(np.isfinite(X_tr0), axis=1)
        if int(np.count_nonzero(ok_tr0)) < 10:
            ax.axis("off")
            continue
        # Faster settings for plotting (visual diagnostic only).
        q_lo_model0 = fit_quantile_ridge(X_tr0[ok_tr0], y_tr[ok_tr0], tau=tau_lo, l2=float(ridge_l2), n_iter=1200)
        q_hi_model0 = fit_quantile_ridge(X_tr0[ok_tr0], y_tr[ok_tr0], tau=tau_hi, l2=float(ridge_l2), n_iter=1200)
        q_lo_cal0 = q_lo_model0.predict(y0_cal.reshape(-1, 1))
        q_hi_cal0 = q_hi_model0.predict(y0_cal.reshape(-1, 1))
        s_cal0 = np.maximum(q_lo_cal0 - y_cal, y_cal - q_hi_cal0).astype(np.float64)
        q0 = conformal_quantile(s_cal0, alpha)
        q_lo_te0 = q_lo_model0.predict(y0_te.reshape(-1, 1))
        q_hi_te0 = q_hi_model0.predict(y0_te.reshape(-1, 1))
        lo = (q_lo_te0 - q0).astype(np.float32)
        hi = (q_hi_te0 + q0).astype(np.float32)
        cqr_center = ((lo + hi) / 2.0).astype(np.float32)

        # ConVOLT(scale-CP): ridge fit for k_hat(x) from spatial features.
        eps = 1e-6
        k_tr = ((y_tr.astype(np.float64) + eps) / (y0_tr.astype(np.float64) + eps)).astype(np.float32)
        k_cal = ((y_cal.astype(np.float64) + eps) / (y0_cal.astype(np.float64) + eps)).astype(np.float32)
        fks = list(feature_keys)
        if len(fks) == 0:
            ax.axis("off")
            continue
        X_tr = df_tr[fks].to_numpy(dtype=np.float32)
        finite = np.isfinite(X_tr)
        col_sum = np.where(finite, X_tr, 0.0).sum(axis=0, keepdims=True)
        col_cnt = finite.sum(axis=0, keepdims=True).astype(np.float32)
        col_mean = np.divide(col_sum, np.maximum(col_cnt, 1.0), out=np.zeros_like(col_sum, dtype=np.float32), where=(col_cnt > 0))
        X_tr_i = _impute(X_tr, col_mean)
        ok_k = np.isfinite(k_tr) & np.all(np.isfinite(X_tr_i), axis=1)
        if int(np.count_nonzero(ok_k)) < 10:
            ax.axis("off")
            continue
        coef, intercept = fit_ridge(X_tr_i[ok_k], k_tr[ok_k], l2=float(ridge_l2))
        X_cal = _impute(df_cal[fks].to_numpy(dtype=np.float32), col_mean)
        X_te = _impute(df_te[fks].to_numpy(dtype=np.float32), col_mean)
        k_hat_cal = (X_cal @ coef.astype(np.float32) + float(intercept)).astype(np.float32)
        k_hat_te = (X_te @ coef.astype(np.float32) + float(intercept)).astype(np.float32)
        k_err_cal = (k_cal - k_hat_cal).astype(np.float32)
        qk = conformal_quantile(np.abs(k_err_cal).astype(np.float64), alpha)
        # Center is point prediction k_hat * y0 (ignore interval mapping for center visualization).
        k_hat_te = np.clip(k_hat_te.astype(np.float64), 0.0, np.inf).astype(np.float32)
        conv_center = (k_hat_te.astype(np.float64) * y0_te.astype(np.float64)).astype(np.float32)

        # Method-specific errors.
        x = y_te.astype(np.float64)
        e_cqr = np.abs(y_te.astype(np.float64) - cqr_center.astype(np.float64))
        e_conv = np.abs(y_te.astype(np.float64) - conv_center.astype(np.float64))
        ok = np.isfinite(x) & (x > 0) & np.isfinite(e_cqr) & np.isfinite(e_conv)
        x = x[ok]
        e_cqr = e_cqr[ok]
        e_conv = e_conv[ok]

        ax.scatter(x, e_conv, s=9, alpha=0.45, color=col_base, edgecolor="none", label="ConVOLT center", zorder=3)
        ax.scatter(x, e_cqr, s=9, alpha=0.45, color=col_method, edgecolor="none", label="CQR center", zorder=3)
        ax.set_xscale("log")
        ax.grid(color="0.92", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"Label {int(lid)}")
        ax.set_xlabel("GT volume (mL, log)")
        ax.set_ylabel("|center error| (mL)")

    for ax in axes[len(label_ids) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"OASIS method-colored center errors vs GT ({backend})\\nrun={run}", y=1.06)
    fig.tight_layout()

    safe_m = re.sub(r"[^a-zA-Z0-9]+", "_", str(method)).strip("_").lower()
    safe_b = re.sub(r"[^a-zA-Z0-9]+", "_", str(baseline)).strip("_").lower()
    fig.savefig(out_dir / f"fig_oasis_label_centererr_scatter_{safe_m}_vs_{safe_b}_{backend}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_oasis_label_centererr_scatter_{safe_m}_vs_{safe_b}_{backend}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _oasis_label_cqr_interval_inflation_grid(
    *,
    tables_dir: Path,
    uq_root: Path,
    out_dir: Path,
    cqr_method: str,
    fig_w: float = 7.2,
    fig_h: float = 2.6,
) -> None:
    """
    One figure with 2 subplots (demons | voxelmorph) showing per-label interval inflation (%) for CQR vs ConVOLT on OASIS.
    Each backend picks the best available CQR variant if the requested one is missing.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    def _compute(backend: str) -> tuple[list[int], list[float], str] | None:
        main = _load_long(tables_dir / "main_results_long_main.csv")
        main["dataset"] = main["dataset"].astype(str).str.strip().str.lower()
        main["backend"] = main["backend"].astype(str).str.strip().str.lower()
        main = main[(main["dataset"] == "oasis") & (main["backend"] == backend) & (main["uq_method"] == "ConVOLT")].copy()
        if len(main) == 0:
            return None
        run = str(main.iloc[0]["run"]).strip()
        cp_path = uq_root / run / "cp_summary.csv"
        if not cp_path.exists():
            return None
        try:
            df = pd.read_csv(cp_path, skipinitialspace=True)
        except Exception:
            return None
        if "target" not in df.columns or "label_id" not in df.columns or "method" not in df.columns:
            return None
        d = df[(df["target"] == "volume_label") & (df["label_id"].astype(int) >= 0)].copy()
        if len(d) == 0:
            return None

        conv_name = "ConVOLT(scale-CP)"
        meths = set(d["method"].astype(str))
        if conv_name not in meths:
            return None

        cqr_pick = None
        if str(cqr_method) in meths:
            cqr_pick = str(cqr_method)
        else:
            for cand in ("CQR(volonly)", "CQR", "CQR(k)"):
                if cand in meths:
                    cqr_pick = cand
                    break
        if cqr_pick is None:
            return None

        conv = d[d["method"].astype(str) == conv_name].set_index("label_id")["interval_size_mean"].to_dict()
        cqr = d[d["method"].astype(str) == cqr_pick].set_index("label_id")["interval_size_mean"].to_dict()

        label_ids = sorted({int(x) for x in conv.keys() if int(x) >= 0})
        infl = []
        keep = []
        for lid in label_ids:
            base = float(conv.get(lid, np.nan))
            val = float(cqr.get(lid, np.nan))
            if not np.isfinite(base) or base <= 0 or not np.isfinite(val):
                continue
            keep.append(lid)
            infl.append(100.0 * (val / base - 1.0))
        if len(keep) == 0:
            return None
        return keep, infl, cqr_pick

    d_res = _compute("demons")
    v_res = _compute("voxelmorph")
    if d_res is None and v_res is None:
        print("[oasis-label] skip: no per-label CQR inflation data for grid plot (need volume_label UQ).")
        return

    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(float(fig_w), float(fig_h)), sharey=True)

    for ax, backend, res in zip(axes, ["demons", "voxelmorph"], [d_res, v_res]):
        ax.set_axisbelow(True)
        if res is None:
            ax.axis("off")
            continue
        keep, infl, pick = res
        x = np.arange(len(keep), dtype=float)
        ax.bar(x, infl, color="#4C78A8", edgecolor="black", linewidth=0.4, zorder=2)
        ax.axhline(0.0, color="0.25", lw=1.0, alpha=0.7)
        ax.set_xticks(x, labels=[str(int(l)) for l in keep], rotation=0)
        ax.set_xlabel("Label ID")
        ax.set_title(f"{backend}\n({pick})")
        ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)

    axes[0].set_ylabel("Interval inflation vs ConVOLT (%)")
    fig.suptitle("OASIS per-label interval inflation: CQR vs ConVOLT")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_oasis_label_cqr_inflation_grid.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_oasis_label_cqr_inflation_grid.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

def _plot_topcorr_heatmap(
    *,
    tables_dir: Path,
    uq_root: Path,
    out_dir: Path,
    topn: int,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove legacy topcorr figures.
    for p in out_dir.glob("fig_topcorr_*.*"):
        try:
            p.unlink()
        except Exception:
            pass
    for p in out_dir.glob("fig_topcorr_heatmap_grid.*"):
        try:
            p.unlink()
        except Exception:
            pass

    # Use runs that appear in the main table (ConVOLT entries), but only keep the primary backends.
    main = _load_long(tables_dir / "main_results_long_main.csv")
    main = main[main["uq_method"] == "ConVOLT"].copy()
    main["dataset"] = main["dataset"].astype(str).str.strip().str.lower()
    main["backend"] = main["backend"].astype(str).str.strip().str.lower()
    main["run"] = main["run"].astype(str).str.strip()

    # Prefer a compact, paper-friendly heatmap:
    #   y-axis = datasets, x-axis = features
    # We annotate each cell with demons/voxelmorph correlations (so we still show both methods).
    backends = ["demons", "voxelmorph"]
    datasets = ["lungct", "nlst", "oasis"]

    def _is_primary_run(run: str, backend: str) -> bool:
        parts = str(run).split("_")
        if len(parts) < 2:
            return False
        return parts[1] == str(backend)

    main = main[main.apply(lambda r: _is_primary_run(r["run"], r["backend"]), axis=1)].copy()

    def _run_rank(run: str) -> tuple[int, int]:
        run = str(run)
        return (1 if "globalfeat" in run else 0, len(run))

    run_map: dict[tuple[str, str], str] = {}
    for (ds, backend), g in main.groupby(["dataset", "backend"], as_index=False):
        if ds not in datasets or backend not in backends:
            continue
        runs = sorted(g["run"].unique().tolist(), key=_run_rank)
        if runs:
            run_map[(ds, backend)] = runs[0]

    # Load all available corr tables and determine a global, comparable feature set.
    corr_tables: dict[tuple[str, str], pd.DataFrame] = {}
    feature_scores: dict[str, list[float]] = {}
    for ds in datasets:
        for backend in backends:
            run = run_map.get((ds, backend))
            if not run:
                continue
            p = uq_root / run / "diagnostics" / "feature_corr.csv"
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if "feature" not in df.columns or "pearson_r" not in df.columns:
                continue
            corr_tables[(ds, backend)] = df
            for _, row in df.iterrows():
                f = str(row["feature"])
                v = float(abs(row["pearson_r"])) if np.isfinite(row["pearson_r"]) else float("nan")
                if not np.isfinite(v):
                    continue
                feature_scores.setdefault(f, []).append(v)

    if len(corr_tables) == 0:
        return

    mean_abs = {f: float(np.mean(vs)) for f, vs in feature_scores.items() if len(vs) > 0}
    top_feats = [f for f, _ in sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)[: int(topn)]]

    def _feat_group(f: str) -> int:
        f = str(f)
        if f.startswith("disp_"):
            return 0
        if f.startswith("jac_"):
            return 1
        if f.startswith("logj_") or f == "mean_abs_logj":
            return 2
        if f.startswith("gradlogj_"):
            return 3
        if f.startswith("div_"):
            return 4
        if f.startswith("curl_"):
            return 5
        if f.startswith("sim_"):
            return 6
        return 99

    top_feats = sorted(top_feats, key=lambda f: (_feat_group(f), -mean_abs.get(f, 0.0), str(f)))

    # Compute group boundaries for visual separators.
    groups = [_feat_group(f) for f in top_feats]
    boundaries = []
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            boundaries.append(i - 0.5)

    # Build arrays and shared color scale (color encodes mean abs correlation across backends).
    mean_mat = np.full((len(datasets), len(top_feats)), np.nan, dtype=float)
    text_d = np.full((len(datasets), len(top_feats)), np.nan, dtype=float)
    text_v = np.full((len(datasets), len(top_feats)), np.nan, dtype=float)

    vmax = 0.0
    for ds in datasets:
        i = datasets.index(ds)
        vals = {}
        for backend in backends:
            df = corr_tables.get((ds, backend))
            if df is None:
                continue
            vals[backend] = df.set_index("feature")["pearson_r"].to_dict()

        for j, f in enumerate(top_feats):
            vd = float(abs(vals.get("demons", {}).get(f, np.nan)))
            vv = float(abs(vals.get("voxelmorph", {}).get(f, np.nan)))
            if np.isfinite(vd):
                text_d[i, j] = vd
            if np.isfinite(vv):
                text_v[i, j] = vv
            vs = [v for v in (vd, vv) if np.isfinite(v)]
            if len(vs) == 0:
                continue
            m = float(np.mean(vs))
            mean_mat[i, j] = m
            vmax = max(vmax, m)

    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0

    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 8})
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.4))

    im = ax.imshow(mean_mat, aspect="auto", vmin=0.0, vmax=vmax, cmap="viridis")
    for x in boundaries:
        ax.axvline(x, color="white", lw=1.0, alpha=0.9)

    ax.set_yticks(np.arange(len(datasets)), labels=[_dataset_title(d) for d in datasets])
    ax.set_xticks(np.arange(len(top_feats)), labels=top_feats, rotation=60, ha="right")
    ax.set_xlabel("Deformation-field features (grouped)")
    ax.set_ylabel("Dataset")
    ax.set_title(f"Top-{int(topn)} feature–error correlations (cells show demons/voxelmorph)")

    # Annotate each cell with "d/v" (2 decimals). Keep it light to avoid clutter.
    for i in range(len(datasets)):
        for j in range(len(top_feats)):
            vd = text_d[i, j]
            vv = text_v[i, j]
            if not np.isfinite(vd) and not np.isfinite(vv):
                continue
            sd = f"{vd:.2f}" if np.isfinite(vd) else "--"
            sv = f"{vv:.2f}" if np.isfinite(vv) else "--"
            ax.text(j, i, f"{sd}/{sv}", ha="center", va="center", fontsize=6, color="white")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Mean |Pearson r| with |error| (color)")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_topcorr_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_topcorr_heatmap.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def _feature_importance_stability(
    *,
    tables_dir: Path,
    uq_root: Path,
    results_root: Path,
    out_dir: Path,
    datasets: list[str],
    backends: list[str],
    n_splits: int = 100,
    seed0: int = 0,
    ridge_l2: float = 0.01,
    topk: int = 10,
) -> None:
    """
    Feature importance stability: fit ConVOLT(scale-CP) ridge models across random patient-level splits,
    and plot mean±std of |coef| per feature (grouped by backend) for each dataset.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from reg.uq.io import load_registration_results_with_features
    from reg.uq.models import fit_ridge

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache" / "coef_stability"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _feat_tex(k: str) -> str:
        k = str(k)
        m = {
            # log-Jacobian stats
            "logj_mean": r"$\mu(\log J)$",
            "logj_std": r"$\sigma(\log J)$",
            "mean_abs_logj": r"$\mu(|\log J|)$",
            "logj_p01": r"$Q_{1}(\log J)$",
            "logj_p99": r"$Q_{99}(\log J)$",
            # Jacobian quantiles / extremes
            "jac_min": r"$\min(J)$",
            "jac_p01": r"$Q_{1}(J)$",
            "jac_p10": r"$Q_{10}(J)$",
            "jac_p50": r"$Q_{50}(J)$",
            "jac_p90": r"$Q_{90}(J)$",
            "jac_p99": r"$Q_{99}(J)$",
            # folding fractions
            "frac_jac_lt_1": r"$\Pr[J<1]$",
            "frac_jac_lt_01": r"$\Pr[J<0.1]$",
            "frac_jac_lt_001": r"$\Pr[J<0.01]$",
            "frac_jac_gt_1": r"$\Pr[J>1]$",
            # displacement magnitude stats
            "disp_mean_mm": r"$\mu(\|u\|)$",
            "disp_std_mm": r"$\sigma(\|u\|)$",
            "disp_p90_mm": r"$Q_{90}(\|u\|)$",
            "disp_max_mm": r"$\max(\|u\|)$",
            # grad logJ
            "gradlogj_mean": r"$\mu(\|\nabla\log J\|)$",
            "gradlogj_p90": r"$Q_{90}(\|\nabla\log J\|)$",
            "gradlogj_max": r"$\max(\|\nabla\log J\|)$",
            # divergence/curl summaries
            "div_mean": r"$\mu(\nabla\!\cdot u)$",
            "div_std": r"$\sigma(\nabla\!\cdot u)$",
            "div_p90_abs": r"$Q_{90}(|\nabla\!\cdot u|)$",
            "curl_mean": r"$\mu(\|\nabla\times u\|)$",
            "curl_p90": r"$Q_{90}(\|\nabla\times u\|)$",
            "curl_max": r"$\max(\|\nabla\times u\|)$",
            # fixed/warped similarity residuals
            "sim_mae": r"$\mathrm{MAE}(I^F,I^M_\phi)$",
            "sim_mse": r"$\mathrm{MSE}(I^F,I^M_\phi)$",
            "sim_corr": r"$\mathrm{corr}(I^F,I^M_\phi)$",
            # fusion disagreement (Learn2Reg multi-atlas)
            "vote_entropy_mean": r"$\mu(H_{\mathrm{vote}})$",
            "vote_maxfrac_mean": r"$\mu(\max p_{\mathrm{vote}})$",
        }
        if k in m:
            return m[k]
        return k.replace("_", r"\_")

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

    def _results_dir_from_run(run: str) -> Path:
        # UQ run names append suffixes like "_globalfeat" that do not exist in registration outputs.
        name = str(run).strip()
        for suf in ("_globalfeat",):
            if name.endswith(suf):
                name = name[: -len(suf)]
        return results_root / name

    def _cache_path(*, results_dir: Path, backend: str) -> Path:
        # Cache key depends on the registration output folder, which encodes dataset/method/atlas settings.
        return cache_dir / f"{_safe_name(str(results_dir))}_{_safe_name(str(backend))}.npz"

    def _load_or_build_feature_cache(*, results_dir: Path, backend: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Returns (patient_id[str], y_gt[float64], y0[float64], X[float32], feats[list[str]]).
        """
        p = _cache_path(results_dir=results_dir, backend=backend)
        if p.exists():
            try:
                npz = np.load(p, allow_pickle=True)
                pids = npz["patient_id"].astype(str)
                y_gt = npz["y_gt"].astype(np.float64)
                y0 = npz["y0"].astype(np.float64)
                X = npz["X"].astype(np.float32)
                feats = [str(x) for x in npz["feature_keys"].astype(str).tolist()]
                if pids.shape[0] == X.shape[0] == y_gt.shape[0] == y0.shape[0] and X.shape[1] == len(feats):
                    return pids, y_gt, y0, X, feats
            except Exception:
                pass

        loaded = load_registration_results_with_features(results_dir=results_dir, require_artifacts=True)
        df = loaded.df.copy()
        if "patient_id" not in df.columns:
            raise KeyError("Missing patient_id")

        # Target definition (absolute union volume vs exhale volume).
        if "vol_union_ml_gt" in df.columns and "vol_union_ml_pred0" in df.columns and np.any(np.isfinite(df["vol_union_ml_gt"].to_numpy(dtype=np.float64))):
            y_gt = df["vol_union_ml_gt"].to_numpy(dtype=np.float64)
            y0 = df["vol_union_ml_pred0"].to_numpy(dtype=np.float64)
        else:
            y_gt = df["exhale_vol_ml_gt"].to_numpy(dtype=np.float64)
            y0 = df["exhale_vol_ml_pred0"].to_numpy(dtype=np.float64)

        feats = [k for k in loaded.diagnostic_feature_keys if k in df.columns]
        if len(feats) == 0:
            raise ValueError("No diagnostic features available")
        X = _impute_feature_nans(df.loc[:, feats].to_numpy(dtype=np.float32))
        pids = df["patient_id"].astype(str).to_numpy()

        np.savez_compressed(
            p,
            patient_id=pids.astype("U"),
            y_gt=y_gt.astype(np.float64),
            y0=y0.astype(np.float64),
            X=X.astype(np.float32),
            feature_keys=np.asarray(feats, dtype="U"),
        )
        return pids.astype(str), y_gt, y0, X, feats

    # Use main-table ConVOLT runs as the source of truth for which output folder to read.
    df_main = _load_long(tables_dir / "main_results_long_main.csv")
    df_main["dataset"] = df_main["dataset"].astype(str).str.strip().str.lower()
    df_main["backend"] = df_main["backend"].astype(str).str.strip().str.lower()
    df_main["uq_method"] = df_main["uq_method"].astype(str).str.strip()
    df_conv = df_main[df_main["uq_method"] == "ConVOLT"].copy()

    datasets = [d for d in datasets if d in set(df_conv["dataset"].tolist())]
    backends = [b for b in backends if b in {"demons", "voxelmorph"}]
    if not datasets or not backends:
        return

    # Un-italicize mathtext for the x tick labels (LaTeX feature names).
    with mpl.rc_context({"mathtext.default": "regular"}):
        mpl.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 8})
        topk = int(max(1, topk))
        fig, axes = plt.subplots(1, len(datasets), figsize=(max(7.2, 2.6 * len(datasets)), 2.8), sharey=False)
        if len(datasets) == 1:
            axes = [axes]

        colors = {"demons": "#4C78A8", "voxelmorph": "#F58518"}

        ylabel_done = False
        for ax, ds in zip(axes, datasets):
            rows_ds = df_conv[df_conv["dataset"] == ds].copy()
            if len(rows_ds) == 0:
                ax.axis("off")
                continue

            # Collect (backend -> mean/std per feature) for this dataset.
            stats: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
            for backend in backends:
                rows_b = rows_ds[rows_ds["backend"] == backend]
                if len(rows_b) == 0:
                    continue
                run = str(rows_b.iloc[0]["run"]).strip()

                # Split sizes: read from cp_runs.csv if available, else use sane defaults.
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

                results_dir = _results_dir_from_run(run)
                try:
                    pids, y_gt, y0, X_all, feats = _load_or_build_feature_cache(results_dir=results_dir, backend=backend)
                except Exception:
                    continue

                coef_abs: list[np.ndarray] = []
                pids_list = [str(x) for x in pids.tolist()]
                eps = 1e-6
                k_true = ((y_gt + eps) / (y0 + eps)).astype(np.float32)
                for s in range(int(n_splits)):
                    split = _make_split(pids_list, seed=int(seed0) + s, n_train=int(n_train), n_calib=int(n_calib), n_test=n_test)
                    tr_mask = np.isin(pids, split["train"])
                    n_tr = int(np.count_nonzero(tr_mask))
                    if n_tr < 5:
                        continue
                    y = k_true[tr_mask]
                    X = X_all[tr_mask]
                    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                    if int(np.count_nonzero(ok)) < max(5, int(0.2 * n_tr)):
                        continue
                    coef, _intercept = fit_ridge(X[ok], y[ok], l2=float(ridge_l2))
                    coef_abs.append(np.abs(coef.astype(np.float64)))

                if len(coef_abs) == 0:
                    continue
                print(f"[coef-stab] fitted {len(coef_abs)}/{int(n_splits)} splits: dataset={ds} backend={backend}")
                A = np.stack(coef_abs, axis=0)
                mu = np.nanmean(A, axis=0)
                sd = np.nanstd(A, axis=0)
                stats[backend] = (feats, mu, sd)

            if not stats:
                ax.axis("off")
                continue

            # Use a stable feature order (prefer the first available backend).
            base_backend = sorted(stats.keys())[0]
            feats_all = list(stats[base_backend][0])

            # Pick top-k features by coefficient magnitude for this dataset (aggregate across backends).
            agg = np.zeros(len(feats_all), dtype=np.float64)
            cnt = np.zeros(len(feats_all), dtype=np.float64)
            for backend in ("demons", "voxelmorph"):
                if backend not in stats:
                    continue
                feats_b, mu, _sd = stats[backend]
                pos = {str(f): i for i, f in enumerate(feats_b)}
                for i, f in enumerate(feats_all):
                    j = pos.get(str(f))
                    if j is None:
                        continue
                    v = float(mu[j])
                    if np.isfinite(v):
                        agg[i] += v
                        cnt[i] += 1.0
            agg = np.divide(agg, np.maximum(cnt, 1.0), out=np.zeros_like(agg), where=(cnt > 0))
            idx = np.argsort(-np.nan_to_num(agg, nan=-np.inf))
            keep_idx = [int(i) for i in idx[: int(min(topk, len(feats_all)))] if np.isfinite(agg[int(i)])]
            feats = [feats_all[i] for i in keep_idx]
            x = np.arange(len(feats), dtype=float)
            width = 0.38
            offs = {"demons": -width / 2.0, "voxelmorph": width / 2.0}

            for backend in ("demons", "voxelmorph"):
                if backend not in stats:
                    continue
                feats_b, mu, sd = stats[backend]
                # Align feature arrays to the chosen feature list.
                mu_al = np.full(len(feats), np.nan, dtype=np.float64)
                sd_al = np.full(len(feats), np.nan, dtype=np.float64)
                pos = {str(f): i for i, f in enumerate(feats_b)}
                for i, f in enumerate(feats):
                    j = pos.get(str(f))
                    if j is not None:
                        mu_al[i] = float(mu[j])
                        sd_al[i] = float(sd[j])
                ax.bar(
                    x + offs[backend],
                    mu_al,
                    yerr=sd_al,
                    width=width * 0.95,
                    color=colors[backend],
                    edgecolor="black",
                    linewidth=0.4,
                    capsize=2.5,
                    label=("Demons" if backend == "demons" else "VoxelMorph"),
                )

            ax.set_title(_dataset_title(ds))
            ax.set_xticks(x, labels=[_feat_tex(f) for f in feats], rotation=60, ha="center")
            ax.grid(axis="y", color="0.9", lw=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if not ylabel_done:
                ax.set_ylabel("Coefficient Magnitude")
                ylabel_done = True
            else:
                ax.set_ylabel("")

        # Legend outside, centered below the (rotated) x labels.
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        if handles and labels:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.25), fontsize=8)

        # Reserve space for rotated tick labels + legend.
        fig.tight_layout(rect=(0.0, 0.28, 1.0, 1.0))
        fig.savefig(out_dir / "fig_coef_stability.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig_coef_stability.png", dpi=250, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make conference-paper figures from uq_results/_tables outputs.")
    uq_root_default = Path(os.environ.get("CONVOLT_UQ_ROOT", "uq_results"))
    ap.add_argument("--tables_dir", type=Path, default=uq_root_default / "_tables")
    ap.add_argument("--uq_root", type=Path, default=uq_root_default)
    ap.add_argument("--out_dir", type=Path, default=uq_root_default / "_figures_paper")
    ap.add_argument("--datasets", type=str, default="lungct,nlst,oasis", help="Comma-separated dataset list for the 3-panel plots.")
    ap.add_argument(
        "--backends",
        type=str,
        default="demons,voxelmorph",
        help="Comma-separated backend list for ablation inflation rows (default: demons,voxelmorph).",
    )
    ap.add_argument("--region_score", choices=["q90", "max", "mean"], default="q90", help="Which region_score to use in Claim 3 bars.")
    ap.add_argument("--topn_corr_features", type=int, default=12, help="Top-N features to show per-run correlation bar plots.")
    ap.add_argument(
        "--corr_eff_baseline",
        choices=["SCP", "CQR"],
        default="SCP",
        help="Baseline method for the correlation-vs-efficiency plots (relative efficiency gain vs baseline).",
    )
    ap.add_argument(
        "--scatter_topk",
        type=int,
        default=0,
        help="Top-k features per dataset to include in scatter grid (0 disables; this can be slow on first run).",
    )
    ap.add_argument(
        "--results_root",
        type=Path,
        default=Path(os.environ.get("CONVOLT_RESULTS_ROOT", "/scratch/yc130/Registration/outputs")),
        help="Registration outputs root for raw feature/value scatter plots (default: ${CONVOLT_RESULTS_ROOT}).",
    )
    ap.add_argument("--oasis_label_topk", type=int, default=0, help="Top-k label IDs to include in OASIS per-label inflation plot (0 disables).")
    ap.add_argument(
        "--oasis_label_cqr_inflation",
        action="store_true",
        help="Plot OASIS per-label CQR interval inflation (%%) vs ConVOLT (requires volume_label UQ).",
    )
    ap.add_argument("--oasis_label_cqr_method", type=str, default="CQR(volonly)", help="Which CQR method name to use for OASIS per-label inflation (default: CQR(volonly)).")
    ap.add_argument(
        "--oasis_label_inflation",
        action="store_true",
        help="Plot OASIS per-label interval inflation (%%) for an arbitrary method vs baseline (requires volume_label UQ).",
    )
    ap.add_argument("--oasis_label_inflation_method", type=str, default="ConVOLT(scale-CP)", help="Method name for --oasis_label_inflation.")
    ap.add_argument("--oasis_label_inflation_baseline", type=str, default="SCP(|err|)", help="Baseline method name for --oasis_label_inflation.")
    ap.add_argument(
        "--oasis_label_error_scatter",
        action="store_true",
        help="Plot OASIS label-wise |error| vs GT volume scatter with UQ interval half-width bands (requires label_volumes.csv + volume_label UQ).",
    )
    ap.add_argument("--oasis_label_error_scatter_topk", type=int, default=6, help="Top-k labels by mean GT volume to include in OASIS error scatter plots.")
    ap.add_argument("--oasis_label_error_scatter_method", type=str, default="CQR(volonly)", help="UQ method to show width bands for in OASIS error scatter plots.")
    ap.add_argument("--oasis_label_error_scatter_baseline", type=str, default="ConVOLT(scale-CP)", help="Baseline UQ method for OASIS error scatter plots.")
    ap.add_argument(
        "--oasis_label_error_scatter_colored",
        action="store_true",
        help="Method-colored scatters: plot |y-center_method| vs GT volume for OASIS labels (refits UQ models; can be slow).",
    )
    ap.add_argument("--oasis_label_fig_w", type=float, default=7.2, help="Figure width for OASIS label inflation grids.")
    ap.add_argument("--oasis_label_fig_h", type=float, default=2.6, help="Figure height for OASIS label inflation grids.")
    ap.add_argument(
        "--no_default_suite",
        action="store_true",
        help="Skip the default figure suite (main/region bars, corr-vs-efficiency, topcorr heatmap). Run only explicitly requested plots.",
    )
    ap.add_argument(
        "--coef_stability",
        action="store_true",
        help="Plot feature-importance stability: mean±std of |ridge coef| across random splits for ConVOLT(scale-CP).",
    )
    ap.add_argument("--coef_stability_splits", type=int, default=100, help="Number of random splits for --coef_stability.")
    ap.add_argument("--coef_stability_seed", type=int, default=0, help="Base RNG seed for --coef_stability.")
    ap.add_argument("--coef_stability_l2", type=float, default=0.01, help="Ridge L2 for --coef_stability.")
    ap.add_argument("--coef_stability_topk", type=int, default=10, help="Show the top-k features per dataset in the coef-stability plot.")
    args = ap.parse_args()

    datasets = [s.strip().lower() for s in str(args.datasets).split(",") if s.strip()]
    backends = [s.strip().lower() for s in str(args.backends).split(",") if s.strip()]
    backends = [b for b in backends if b in {"demons", "voxelmorph"}]
    if len(backends) == 0:
        backends = ["demons", "voxelmorph"]

    if not bool(args.no_default_suite):
        _barplot_inflations(
            tables_dir=args.tables_dir,
            out_dir=args.out_dir,
            datasets=datasets,
            backends=backends,
            region_score=str(args.region_score),
        )
        for k in (1, 2, 3):
            _corr_vs_efficiency(
                tables_dir=args.tables_dir,
                uq_root=args.uq_root,
                out_dir=args.out_dir,
                datasets=datasets if len(datasets) > 0 else None,
                topk=k,
                baseline_uq_method=str(args.corr_eff_baseline),
            )
        _plot_topcorr_heatmap(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            out_dir=args.out_dir / "feature_corr",
            topn=int(args.topn_corr_features),
        )

    # New: scatter-grid feature interpretability.
    if int(args.scatter_topk) > 0:
        _feature_scatter_grid_overlay(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            results_root=args.results_root,
            out_dir=args.out_dir / "feature_scatter",
            datasets=datasets,
            backends=backends,
            topk=int(args.scatter_topk),
        )

    # New: OASIS per-label inflation (requires volume_label UQ outputs).
    if int(args.oasis_label_topk) > 0:
        _oasis_label_inflation(
            uq_root=args.uq_root,
            out_dir=args.out_dir,
            topk=int(args.oasis_label_topk),
            backend="demons",
        )
        _oasis_label_inflation(
            uq_root=args.uq_root,
            out_dir=args.out_dir,
            topk=int(args.oasis_label_topk),
            backend="voxelmorph",
        )

    if bool(args.oasis_label_cqr_inflation):
        _oasis_label_cqr_interval_inflation_grid(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            out_dir=args.out_dir,
            cqr_method=str(args.oasis_label_cqr_method),
            fig_w=float(args.oasis_label_fig_w),
            fig_h=float(args.oasis_label_fig_h),
        )

    if bool(args.oasis_label_inflation):
        _oasis_label_interval_inflation_grid(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            out_dir=args.out_dir,
            method=str(args.oasis_label_inflation_method),
            baseline=str(args.oasis_label_inflation_baseline),
            fig_w=float(args.oasis_label_fig_w),
            fig_h=float(args.oasis_label_fig_h),
        )

    if bool(args.oasis_label_error_scatter):
        _oasis_label_error_scatter_with_bands(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            results_root=args.results_root,
            out_dir=args.out_dir,
            backend="demons",
            method=str(args.oasis_label_error_scatter_method),
            baseline=str(args.oasis_label_error_scatter_baseline),
            topk=int(args.oasis_label_error_scatter_topk),
        )
        _oasis_label_error_scatter_with_bands(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            results_root=args.results_root,
            out_dir=args.out_dir,
            backend="voxelmorph",
            method=str(args.oasis_label_error_scatter_method),
            baseline=str(args.oasis_label_error_scatter_baseline),
            topk=int(args.oasis_label_error_scatter_topk),
        )

    if bool(args.oasis_label_error_scatter_colored):
        _oasis_label_error_scatter_method_colored(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            results_root=args.results_root,
            out_dir=args.out_dir,
            backend="demons",
            method=str(args.oasis_label_error_scatter_method),
            baseline=str(args.oasis_label_error_scatter_baseline),
            topk=int(args.oasis_label_error_scatter_topk),
            alpha=0.1,
        )
        _oasis_label_error_scatter_method_colored(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            results_root=args.results_root,
            out_dir=args.out_dir,
            backend="voxelmorph",
            method=str(args.oasis_label_error_scatter_method),
            baseline=str(args.oasis_label_error_scatter_baseline),
            topk=int(args.oasis_label_error_scatter_topk),
            alpha=0.1,
        )

    if bool(args.coef_stability):
        _feature_importance_stability(
            tables_dir=args.tables_dir,
            uq_root=args.uq_root,
            results_root=args.results_root,
            out_dir=args.out_dir,
            datasets=datasets,
            backends=backends,
            n_splits=int(args.coef_stability_splits),
            seed0=int(args.coef_stability_seed),
            ridge_l2=float(args.coef_stability_l2),
            topk=int(args.coef_stability_topk),
        )


if __name__ == "__main__":
    main()
