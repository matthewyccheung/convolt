#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BackendRun:
    backend: str  # demons | voxelmorph
    run_name: str
    results_dir: Path
    uq_dir: Path


def _read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_").lower()


def _excluded_ids(results_dir: Path) -> set[str]:
    ex: set[str] = set()
    atlas_meta = results_dir / "atlas_meta.json"
    if atlas_meta.exists():
        try:
            ex.update([str(x) for x in _read_json(atlas_meta).get("atlas_ids", [])])
        except Exception:
            pass
    vm_ids = results_dir / "vm_train_ids.json"
    if vm_ids.exists():
        try:
            ex.update([str(x) for x in _read_json(vm_ids).get("vm_train_ids", [])])
        except Exception:
            pass
    return ex


def _default_results_dir(*, results_root: Path, dataset: str, backend: str, atlas_tag: str, voxelmorph_mode: str) -> Path:
    dataset = str(dataset).lower()
    backend = str(backend).lower()
    atlas_tag = str(atlas_tag)
    voxelmorph_mode = str(voxelmorph_mode).lower()
    if backend == "demons":
        name = f"{dataset}_demons_{atlas_tag}"
    elif backend == "voxelmorph":
        name = f"{dataset}_voxelmorph_{voxelmorph_mode}_{atlas_tag}"
    else:
        raise ValueError("backend must be demons or voxelmorph")
    return results_root / name


def _candidate_uq_runs(*, uq_root: Path, dataset: str, backend: str, atlas_tag: str, voxelmorph_mode: str) -> list[Path]:
    dataset = str(dataset).lower()
    backend = str(backend).lower()
    atlas_tag = str(atlas_tag)
    voxelmorph_mode = str(voxelmorph_mode).lower()
    if backend == "demons":
        prefix = f"{dataset}_demons_{atlas_tag}"
        pats = [prefix, f"{prefix}_globalfeat"]
    elif backend == "voxelmorph":
        prefix = f"{dataset}_voxelmorph_{voxelmorph_mode}_{atlas_tag}"
        pats = [prefix, f"{prefix}_globalfeat"]
    else:
        raise ValueError("backend must be demons or voxelmorph")
    out: list[Path] = []
    for p in pats:
        out.extend(sorted(uq_root.glob(p)))
    # De-dup while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _pick_uq_run(
    *,
    uq_root: Path,
    dataset: str,
    backend: str,
    atlas_tag: str,
    voxelmorph_mode: str,
    method: str,
    baseline: str,
) -> Path:
    for d in _candidate_uq_runs(
        uq_root=uq_root, dataset=dataset, backend=backend, atlas_tag=atlas_tag, voxelmorph_mode=voxelmorph_mode
    ):
        cp = d / "cp_summary.csv"
        if not cp.exists():
            continue
        df = pd.read_csv(cp, skipinitialspace=True)
        if "target" not in df.columns or "label_id" not in df.columns or "method" not in df.columns:
            continue
        df = df[df["target"].astype(str) == "volume_label"].copy()
        if len(df) == 0:
            continue
        ms = set(df["method"].astype(str).tolist())
        if method in ms and baseline in ms:
            return d
    raise FileNotFoundError(
        f"Could not find a UQ run under {uq_root} for dataset={dataset} backend={backend} that contains "
        f"target=volume_label with methods {method!r} and {baseline!r}."
    )


def _load_interval_widths(cp_summary_path: Path, *, method: str, baseline: str) -> pd.DataFrame:
    df = pd.read_csv(cp_summary_path, skipinitialspace=True)
    df = df[df["target"].astype(str) == "volume_label"].copy()
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"] > 0].copy()

    keep = df[df["method"].astype(str).isin([method, baseline])].copy()
    piv = keep.pivot_table(index="label_id", columns="method", values="interval_size_mean", aggfunc="first").reset_index()
    if method not in piv.columns or baseline not in piv.columns:
        raise ValueError(f"Missing required methods in {cp_summary_path}: need {method!r},{baseline!r}")
    piv = piv.rename(columns={method: "width_method_ml", baseline: "width_baseline_ml"})
    piv["inflation_pct"] = 100.0 * (piv["width_method_ml"].astype(float) / piv["width_baseline_ml"].astype(float) - 1.0)
    return piv[["label_id", "width_method_ml", "width_baseline_ml", "inflation_pct"]].copy()


def _load_label_cases(results_dir: Path, *, backend: str, vol_thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    lv_path = results_dir / "label_volumes.csv"
    if not lv_path.exists():
        raise FileNotFoundError(lv_path)
    df = pd.read_csv(lv_path)
    need = {"patient_id", "split", "backend", "label_id", "vol_ml_gt", "vol_ml_pred", "dice"}
    if not need.issubset(set(df.columns)):
        raise KeyError(f"{lv_path} missing required columns, have={sorted(df.columns)}")
    df = df[(df["split"].astype(str).str.lower() == "training") & (df["backend"].astype(str).str.lower() == str(backend).lower())].copy()
    ex = _excluded_ids(results_dir)
    if ex:
        df = df[~df["patient_id"].astype(str).isin(ex)].copy()
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"] > 0].copy()
    df["abs_err_ml"] = np.abs(df["vol_ml_gt"].to_numpy(dtype=float) - df["vol_ml_pred"].to_numpy(dtype=float))
    df["present"] = (df["vol_ml_gt"].to_numpy(dtype=float) > float(vol_thr)).astype(int)

    # Label-level summaries.
    g = df.groupby("label_id", as_index=False).agg(
        vol_gt_mean_ml=("vol_ml_gt", "mean"),
        vol_gt_std_ml=("vol_ml_gt", "std"),
        prevalence=("present", "mean"),
        dice_mean=("dice", "mean"),
        dice_p10=("dice", lambda x: float(np.nanpercentile(np.asarray(x, dtype=float), 10.0)) if len(x) else np.nan),
        err_p50=("abs_err_ml", lambda x: float(np.nanpercentile(np.asarray(x, dtype=float), 50.0)) if len(x) else np.nan),
        err_p90=("abs_err_ml", lambda x: float(np.nanpercentile(np.asarray(x, dtype=float), 90.0)) if len(x) else np.nan),
        err_p95=("abs_err_ml", lambda x: float(np.nanpercentile(np.asarray(x, dtype=float), 95.0)) if len(x) else np.nan),
        err_p99=("abs_err_ml", lambda x: float(np.nanpercentile(np.asarray(x, dtype=float), 99.0)) if len(x) else np.nan),
        n_cases=("patient_id", "count"),
    )
    g["gt_cv"] = g["vol_gt_std_ml"].to_numpy(dtype=float) / np.clip(g["vol_gt_mean_ml"].to_numpy(dtype=float), 1e-6, np.inf)
    g["tail_p95_p50"] = g["err_p95"].to_numpy(dtype=float) / np.clip(g["err_p50"].to_numpy(dtype=float), 1e-6, np.inf)
    return df, g


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(ok)) < 3:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _scatter_with_fit(ax, x: np.ndarray, y: np.ndarray, *, color: str, label: str, xscale: str | None = None) -> None:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    ax.scatter(x, y, s=22, alpha=0.85, color=color, edgecolor="black", linewidth=0.3, label=label, zorder=3)
    if x.size >= 2:
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, slope * xs + intercept, color=color, lw=1.2, alpha=0.9, zorder=2)
    if xscale:
        ax.set_xscale(xscale)


def _plot_inflation_scatter(
    *,
    df: pd.DataFrame,
    out_path: Path,
    x_col: str,
    x_label: str,
    x_transform: str | None,
    title: str,
) -> None:
    import matplotlib as mpl

    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.1), dpi=200, constrained_layout=True)
    colors = {"demons": "#4C78A8", "voxelmorph": "#F58518"}

    for backend, g in df.groupby("backend"):
        x = g[x_col].to_numpy(dtype=float)
        y = g["inflation_pct"].to_numpy(dtype=float)
        if x_transform == "log10":
            x = np.log10(np.clip(x, 1e-6, np.inf))
        _scatter_with_fit(ax, x, y, color=colors.get(str(backend), "0.5"), label=str(backend))
        r = _pearsonr(x, y)
        ax.text(
            0.02,
            0.98 - (0.12 if str(backend) == "voxelmorph" else 0.0),
            f"{backend}: r={r:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.85),
        )

    ax.axhline(0.0, color="0.25", lw=1.0, alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Interval inflation vs baseline (%)")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(color="0.9", lw=0.8, zorder=0)
    ax.legend(frameon=False, loc="best")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_width_vs_error(
    *,
    df: pd.DataFrame,
    out_path: Path,
    method_name: str,
    baseline_name: str,
    x_col: str,
    x_label: str,
    x_transform: str | None,
) -> None:
    import matplotlib as mpl

    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.1), dpi=200, constrained_layout=True)
    colors = {"demons": "#4C78A8", "voxelmorph": "#F58518"}
    markers = {"baseline": "o", "method": "s"}

    for backend, g in df.groupby("backend"):
        x = g[x_col].to_numpy(dtype=float)
        if x_transform == "log10":
            x = np.log10(np.clip(x, 1e-6, np.inf))
        y_m = g["width_method_ml"].to_numpy(dtype=float)
        y_b = g["width_baseline_ml"].to_numpy(dtype=float)
        okm = np.isfinite(x) & np.isfinite(y_m)
        okb = np.isfinite(x) & np.isfinite(y_b)
        ax.scatter(x[okb], y_b[okb], s=28, alpha=0.85, color=colors.get(str(backend), "0.5"), marker=markers["baseline"], edgecolor="black", linewidth=0.3, label=f"{backend} {baseline_name}")
        ax.scatter(x[okm], y_m[okm], s=28, alpha=0.85, color=colors.get(str(backend), "0.5"), marker=markers["method"], edgecolor="black", linewidth=0.3, label=f"{backend} {method_name}")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Interval width (mL)")
    ax.set_title("Interval width vs error summary (per label)")
    ax.set_axisbelow(True)
    ax.grid(color="0.9", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_heteroscedasticity_curves(
    *,
    cases: pd.DataFrame,
    widths: pd.DataFrame,
    out_path: Path,
    backend: str,
    topk_labels: int,
    n_bins: int,
) -> None:
    """
    For a single backend, pick top-k labels by mean GT volume and plot median(|err|) vs V_pred bin.
    """
    import matplotlib as mpl

    mpl.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    df = cases.copy()
    df = df[np.isfinite(df["vol_ml_pred"].to_numpy(dtype=float)) & np.isfinite(df["abs_err_ml"].to_numpy(dtype=float))].copy()

    g = df.groupby("label_id", as_index=False)["vol_ml_gt"].mean().rename(columns={"vol_ml_gt": "gt_mean"})
    lids = g.sort_values("gt_mean", ascending=False)["label_id"].astype(int).tolist()[: int(topk_labels)]
    if len(lids) == 0:
        return

    n = len(lids)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.2 + 1.6 * nrows), dpi=200, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, lid in zip(axes, lids):
        dd = df[df["label_id"].astype(int) == int(lid)].copy()
        v = dd["vol_ml_pred"].to_numpy(dtype=float)
        e = dd["abs_err_ml"].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(e)
        v = v[ok]
        e = e[ok]
        if v.size < max(10, int(n_bins) * 3):
            ax.axis("off")
            continue
        qs = np.quantile(v, np.linspace(0, 1, int(n_bins) + 1))
        xs = []
        ys = []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (v >= a) & (v <= b)
            if int(np.count_nonzero(m)) < 3:
                continue
            xs.append(float(np.median(v[m])))
            ys.append(float(np.median(e[m])))
        if len(xs) == 0:
            ax.axis("off")
            continue
        ax.plot(xs, ys, marker="o", lw=1.2, ms=3.5, color="#4C78A8")
        ax.set_xscale("log")
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"Label {int(lid)}")
        ax.set_xlabel("V_pred (mL, log)")
        ax.set_ylabel("median |err| (mL)")

        # Optionally overlay horizontal half-widths as reference (method and baseline).
        w = widths[widths["label_id"].astype(int) == int(lid)]
        if len(w) == 1:
            hm = float(w.iloc[0]["width_method_ml"]) / 2.0
            hb = float(w.iloc[0]["width_baseline_ml"]) / 2.0
            if np.isfinite(hm):
                ax.axhline(hm, color="#F58518", lw=1.0, alpha=0.8)
            if np.isfinite(hb):
                ax.axhline(hb, color="#54A24B", lw=1.0, alpha=0.8)

    for ax in axes[len(lids) :]:
        ax.axis("off")

    fig.suptitle(f"Heteroscedasticity: median(|err|) vs V_pred bins ({backend})", y=1.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Label-wise diagnostics for interval inflation (CQR vs ConVOLT baselines).")
    ap.add_argument("--dataset", type=str, default="oasis")
    ap.add_argument("--atlas_tag", type=str, default="atlas-multi5")
    ap.add_argument("--voxelmorph_mode", type=str, default="unsupervised", choices=["unsupervised", "supervised", "hybrid"])
    ap.add_argument("--results_root", type=Path, default=Path("/scratch/yc130/Registration/outputs"))
    ap.add_argument("--uq_root", type=Path, default=Path("uq_results"))
    ap.add_argument("--method", type=str, default="CQR(volonly)")
    ap.add_argument("--baseline", type=str, default="ConVOLT(scale-CP)")
    ap.add_argument("--vol_thr", type=float, default=1e-3, help="Volume threshold (mL) for prevalence V_gt > vol_thr.")
    ap.add_argument("--out_dir", type=Path, default=Path("uq_results") / "_figures_paper")
    ap.add_argument("--topk_labels", type=int, default=9, help="Top-k labels (by mean GT volume) for heteroscedasticity curve plots.")
    ap.add_argument("--hetero_bins", type=int, default=6, help="Number of quantile bins for heteroscedasticity curves.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[BackendRun] = []
    for backend in ("demons", "voxelmorph"):
        res_dir = _default_results_dir(
            results_root=args.results_root,
            dataset=args.dataset,
            backend=backend,
            atlas_tag=args.atlas_tag,
            voxelmorph_mode=args.voxelmorph_mode,
        )
        uq_dir = _pick_uq_run(
            uq_root=args.uq_root,
            dataset=args.dataset,
            backend=backend,
            atlas_tag=args.atlas_tag,
            voxelmorph_mode=args.voxelmorph_mode,
            method=str(args.method),
            baseline=str(args.baseline),
        )
        runs.append(BackendRun(backend=backend, run_name=uq_dir.name, results_dir=res_dir, uq_dir=uq_dir))

    rows_all: list[pd.DataFrame] = []
    cases_by_backend: dict[str, pd.DataFrame] = {}
    widths_by_backend: dict[str, pd.DataFrame] = {}

    for r in runs:
        widths = _load_interval_widths(r.uq_dir / "cp_summary.csv", method=str(args.method), baseline=str(args.baseline))
        cases, label_stats = _load_label_cases(r.results_dir, backend=r.backend, vol_thr=float(args.vol_thr))
        m = widths.merge(label_stats, on="label_id", how="inner")
        m["backend"] = r.backend
        m["run"] = r.run_name
        rows_all.append(m)
        cases_by_backend[r.backend] = cases
        widths_by_backend[r.backend] = widths

    df = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
    if len(df) == 0:
        raise RuntimeError("No data to plot.")

    stem = f"{_safe(args.dataset)}_labels_{_safe(args.method)}_vs_{_safe(args.baseline)}"
    df.to_csv(args.out_dir / f"{stem}_diagnostics.csv", index=False)

    # 1) Inflation vs tail-heaviness (p95/p50)
    _plot_inflation_scatter(
        df=df,
        out_path=args.out_dir / f"fig_{stem}_infl_vs_tailp95p50",
        x_col="tail_p95_p50",
        x_label="Tail heaviness: p95(|err|) / p50(|err|)",
        x_transform=None,
        title="Interval inflation vs error tail-heaviness",
    )
    # (Optional alternative tail metric: p99)
    _plot_inflation_scatter(
        df=df,
        out_path=args.out_dir / f"fig_{stem}_infl_vs_errp99",
        x_col="err_p99",
        x_label="p99(|err|) (mL)",
        x_transform="log10",
        title="Interval inflation vs extreme error (p99)",
    )
    # 2) Inflation vs prevalence
    _plot_inflation_scatter(
        df=df,
        out_path=args.out_dir / f"fig_{stem}_infl_vs_prevalence",
        x_col="prevalence",
        x_label=f"Label prevalence: P(V_gt > {float(args.vol_thr):g} mL)",
        x_transform=None,
        title="Interval inflation vs label prevalence",
    )
    # 3) Inflation vs segmentation quality
    _plot_inflation_scatter(
        df=df,
        out_path=args.out_dir / f"fig_{stem}_infl_vs_dicemean",
        x_col="dice_mean",
        x_label="Mean Dice (label)",
        x_transform=None,
        title="Interval inflation vs segmentation quality (mean Dice)",
    )
    _plot_inflation_scatter(
        df=df,
        out_path=args.out_dir / f"fig_{stem}_infl_vs_dicep10",
        x_col="dice_p10",
        x_label="10th percentile Dice (label)",
        x_transform=None,
        title="Interval inflation vs worst-case quality (Dice p10)",
    )
    # 4) Direct width-vs-error plot
    _plot_width_vs_error(
        df=df,
        out_path=args.out_dir / f"fig_{stem}_width_vs_errp50",
        method_name=str(args.method),
        baseline_name=str(args.baseline),
        x_col="err_p50",
        x_label="median(|err|) per label (mL, log10)",
        x_transform="log10",
    )
    _plot_width_vs_error(
        df=df,
        out_path=args.out_dir / f"fig_{stem}_width_vs_errp90",
        method_name=str(args.method),
        baseline_name=str(args.baseline),
        x_col="err_p90",
        x_label="p90(|err|) per label (mL, log10)",
        x_transform="log10",
    )
    # 5) Heteroscedasticity curves (median(|err|) vs V_pred bins), per backend.
    for backend, cases in cases_by_backend.items():
        _plot_heteroscedasticity_curves(
            cases=cases,
            widths=widths_by_backend.get(backend, pd.DataFrame()),
            out_path=args.out_dir / f"fig_{stem}_hetero_curves_{backend}",
            backend=backend,
            topk_labels=int(args.topk_labels),
            n_bins=int(args.hetero_bins),
        )

    print(f"Wrote diagnostics CSV: {args.out_dir / f'{stem}_diagnostics.csv'}")
    print(f"Wrote figures with stem: fig_{stem}_*.pdf/png")


if __name__ == "__main__":
    main()

