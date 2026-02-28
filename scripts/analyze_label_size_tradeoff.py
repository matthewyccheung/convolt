#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    backend: str
    uq_run: str
    results_dir: Path


def _read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _find_first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


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
    baseline_method: str,
    compare_method: str,
) -> str:
    for d in _candidate_uq_runs(
        uq_root=uq_root, dataset=dataset, backend=backend, atlas_tag=atlas_tag, voxelmorph_mode=voxelmorph_mode
    ):
        cp = d / "cp_summary.csv"
        if not cp.exists():
            continue
        df = pd.read_csv(cp, skipinitialspace=True)
        if "target" not in df.columns or "label_id" not in df.columns:
            continue
        df = df[df["target"].astype(str) == "volume_label"].copy()
        if len(df) == 0:
            continue
        ms = set(df["method"].astype(str).tolist())
        if baseline_method in ms and compare_method in ms:
            return d.name
    raise FileNotFoundError(
        f"Could not find a UQ run under {uq_root} for dataset={dataset} backend={backend} that contains "
        f"target=volume_label with methods {compare_method!r} and {baseline_method!r}."
    )


def _excluded_ids(results_dir: Path) -> set[str]:
    ex: set[str] = set()
    atlas_meta = results_dir / "atlas_meta.json"
    if atlas_meta.exists():
        meta = _read_json(atlas_meta)
        for pid in meta.get("atlas_ids", []):
            ex.add(str(pid))
    vm_ids = results_dir / "vm_train_ids.json"
    if vm_ids.exists():
        meta = _read_json(vm_ids)
        for pid in meta.get("vm_train_ids", []):
            ex.add(str(pid))
    return ex


def _label_sizes(results_dir: Path, *, backend: str) -> pd.DataFrame:
    lv_path = results_dir / "label_volumes.csv"
    if not lv_path.exists():
        raise FileNotFoundError(f"Missing label_volumes.csv at: {lv_path}")
    df = pd.read_csv(lv_path)
    if "vol_ml_gt" not in df.columns:
        raise KeyError(f"label_volumes.csv missing vol_ml_gt: {lv_path}")
    df = df[df["backend"].astype(str).str.lower() == str(backend).lower()].copy()
    df = df[df["split"].astype(str).str.lower() == "training"].copy()
    ex = _excluded_ids(results_dir)
    if ex:
        df = df[~df["patient_id"].astype(str).isin(ex)].copy()
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"] > 0].copy()
    g = df.groupby("label_id", as_index=False)["vol_ml_gt"].agg(["mean", "median", "std", "count"]).reset_index()
    g = g.rename(
        columns={"mean": "vol_gt_mean_ml", "median": "vol_gt_median_ml", "std": "vol_gt_std_ml", "count": "n_cases"}
    )
    return g


def _per_label_intervals(cp_summary_path: Path, *, baseline_method: str, compare_method: str) -> pd.DataFrame:
    df = pd.read_csv(cp_summary_path, skipinitialspace=True)
    df = df[df["target"].astype(str) == "volume_label"].copy()
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"] > 0].copy()

    keep = df[df["method"].astype(str).isin([baseline_method, compare_method])].copy()
    if len(keep) == 0:
        raise ValueError(f"No rows found for methods {baseline_method!r} and {compare_method!r} in {cp_summary_path}")

    piv = keep.pivot_table(index="label_id", columns="method", values="interval_size_mean", aggfunc="first").reset_index()
    if baseline_method not in piv.columns or compare_method not in piv.columns:
        raise ValueError(
            f"Missing one of the required methods in pivot: have={list(piv.columns)} need={baseline_method!r},{compare_method!r}"
        )
    piv = piv.rename(columns={baseline_method: "interval_baseline_mean_ml", compare_method: "interval_compare_mean_ml"})
    piv["ratio"] = piv["interval_compare_mean_ml"].astype(float) / piv["interval_baseline_mean_ml"].astype(float)
    piv["inflation_pct"] = 100.0 * (piv["ratio"].astype(float) - 1.0)
    return piv


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(ok)) < 3:
        return float("nan")
    x = x[ok]
    y = y[ok]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(ok)) < 3:
        return float("nan")
    x = x[ok]
    y = y[ok]
    vx = float(np.var(x))
    if vx <= 0:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / vx)


def main() -> None:
    ap = argparse.ArgumentParser(description="Test whether per-label interval inflation depends on label size (OASIS hypothesis test).")
    ap.add_argument("--dataset", type=str, default="oasis")
    ap.add_argument("--atlas_tag", type=str, default="atlas-multi5")
    ap.add_argument("--voxelmorph_mode", type=str, default="unsupervised", choices=["unsupervised", "supervised", "hybrid"])
    ap.add_argument("--results_root", type=Path, default=Path(os.environ.get("CONVOLT_RESULTS_ROOT", "/scratch/yc130/Registration/outputs")))
    ap.add_argument("--uq_root", type=Path, default=Path("uq_results"))
    ap.add_argument("--baseline_method", type=str, default="ConVOLT(scale-CP)", help="Baseline method name in cp_summary.csv.")
    ap.add_argument("--compare_method", type=str, default="CQR", help="Comparison method name in cp_summary.csv.")
    ap.add_argument(
        "--x_var",
        type=str,
        default="cv",
        choices=["cv", "mean"],
        help="X-axis variable: 'cv' for GT volume coefficient of variation across subjects, or 'mean' for mean GT volume.",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("uq_results") / "_figures_paper")
    ap.add_argument("--save_prefix", type=str, default="", help="Prefix for outputs (PNG/PDF/CSV). If empty, auto-name based on x_var.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not str(args.save_prefix).strip():
        args.save_prefix = f"oasis_label_inflation_vs_{str(args.x_var).lower()}"

    run_specs: list[RunSpec] = []
    for backend in ["demons", "voxelmorph"]:
        results_dir = _default_results_dir(
            results_root=args.results_root,
            dataset=args.dataset,
            backend=backend,
            atlas_tag=args.atlas_tag,
            voxelmorph_mode=args.voxelmorph_mode,
        )
        uq_run = _pick_uq_run(
            uq_root=args.uq_root,
            dataset=args.dataset,
            backend=backend,
            atlas_tag=args.atlas_tag,
            voxelmorph_mode=args.voxelmorph_mode,
            baseline_method=str(args.baseline_method),
            compare_method=str(args.compare_method),
        )
        run_specs.append(RunSpec(backend=backend, uq_run=uq_run, results_dir=results_dir))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=200, constrained_layout=True)
    rows_all: list[pd.DataFrame] = []
    for ax, spec in zip(axes, run_specs):
        cp = Path(args.uq_root) / spec.uq_run / "cp_summary.csv"
        per_label = _per_label_intervals(cp, baseline_method=str(args.baseline_method), compare_method=str(args.compare_method))
        sizes = _label_sizes(spec.results_dir, backend=spec.backend)
        df = per_label.merge(sizes, on="label_id", how="inner")
        df["backend"] = spec.backend
        df["uq_run"] = spec.uq_run
        rows_all.append(df)

        if str(args.x_var).lower() == "mean":
            x = df["vol_gt_mean_ml"].to_numpy(dtype=np.float64)
        else:
            x = (df["vol_gt_std_ml"].to_numpy(dtype=np.float64) / np.clip(df["vol_gt_mean_ml"].to_numpy(dtype=np.float64), 1e-6, np.inf)).astype(np.float64)
        y = df["inflation_pct"].to_numpy(dtype=np.float64)
        ax.scatter(x, y, s=18, alpha=0.85)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
        ax.set_title(spec.backend)
        if str(args.x_var).lower() == "mean":
            ax.set_xscale("log")
            ax.set_xlabel("Mean GT label volume (mL, log)")
            zx = np.log10(np.clip(x, 1e-6, np.inf))
            r = _pearsonr(zx, y)
            slope = _fit_slope(zx, y)
            ax.text(0.02, 0.98, f"r(log10 V, infl)={r:.2f}\\nslope={slope:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=9)
        else:
            ax.set_xlabel("GT label volume CV across subjects")
            r = _pearsonr(x, y)
            slope = _fit_slope(x, y)
            ax.text(0.02, 0.98, f"r(CV, infl)={r:.2f}\\nslope={slope:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=9)
        ax.set_ylabel(f"Interval inflation vs {args.baseline_method} (%)")

    out = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
    csv_path = args.out_dir / f"{args.save_prefix}_points.csv"
    out.to_csv(csv_path, index=False)

    png_path = args.out_dir / f"{args.save_prefix}.png"
    pdf_path = args.out_dir / f"{args.save_prefix}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
