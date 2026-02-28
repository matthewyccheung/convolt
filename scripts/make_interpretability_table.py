#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_pm_mean(s: str) -> float:
    if s is None:
        return float("nan")
    t = str(s).strip().replace(" ", "")
    if "±" in t:
        t = t.split("±", 1)[0]
    try:
        return float(t)
    except Exception:
        return float("nan")


def _is_primary_run(run: str, backend: str) -> bool:
    parts = str(run).split("_")
    if len(parts) < 2:
        return False
    return parts[1] == str(backend)


def _pick_best_run(runs: list[str], *, backend: str) -> str:
    """
    Choose one run per (dataset, backend) for paper figures/tables.

    Policy (in order):
      1) Prefer runs tagged with 'globalfeat' if present.
      2) For VoxelMorph, prefer 'supervised' if present among remaining.
      3) Deterministic tie-breaker: lexicographic.
    """
    cand = list(dict.fromkeys([str(r) for r in runs if str(r).strip()]))
    if not cand:
        raise ValueError("No runs to pick from")

    gf = [r for r in cand if "globalfeat" in r.lower()]
    if gf:
        cand = gf

    if str(backend).lower() == "voxelmorph":
        sup = [r for r in cand if "supervised" in r.lower()]
        if sup:
            cand = sup

    return sorted(cand)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an interpretability table from feature correlations and SCP→ConVOLT efficiency gains.")
    ap.add_argument("--tables_dir", type=Path, default=Path("uq_results") / "_tables")
    ap.add_argument("--uq_root", type=Path, default=Path("uq_results"))
    ap.add_argument("--datasets", type=str, default="", help="Optional comma-separated allowlist (e.g., nlst,lungct,oasis).")
    ap.add_argument("--backends", type=str, default="demons,voxelmorph")
    ap.add_argument("--topk_features", type=int, default=3)
    ap.add_argument("--out_csv", type=Path, default=Path("uq_results") / "_tables" / "interpretability_table.csv")
    args = ap.parse_args()

    allow = {d.strip().lower() for d in str(args.datasets).split(",") if d.strip()}
    backends = [b.strip().lower() for b in str(args.backends).split(",") if b.strip()]
    backends = [b for b in backends if b in {"demons", "voxelmorph"}]
    if not backends:
        backends = ["demons", "voxelmorph"]

    main_long_path = args.tables_dir / "main_results_long_main.csv"
    if not main_long_path.exists():
        raise FileNotFoundError(main_long_path)
    df = pd.read_csv(main_long_path, skipinitialspace=True)
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["backend"] = df["backend"].astype(str).str.strip().str.lower()
    df["run"] = df["run"].astype(str).str.strip()
    df["uq_method"] = df["uq_method"].astype(str).str.strip()
    if allow:
        df = df[df["dataset"].isin(sorted(allow))].copy()

    # Need ConVOLT + SCP per (dataset, backend, run).
    df = df[df["uq_method"].isin(["ConVOLT", "SCP"])].copy()
    df = df[df.apply(lambda r: _is_primary_run(r["run"], r["backend"]), axis=1)].copy()

    # Reduce to a single run per (dataset, backend) using the paper policy:
    # prefer globalfeat runs, and prefer supervised VoxelMorph when applicable.
    keep = []
    for (ds, backend), g0 in df.groupby(["dataset", "backend"], as_index=False):
        run = _pick_best_run(sorted(set(g0["run"].tolist())), backend=str(backend))
        keep.append({"dataset": ds, "backend": backend, "run": run})
    keep_df = pd.DataFrame(keep)
    df = df.merge(keep_df, on=["dataset", "backend", "run"], how="inner")

    rows: list[dict] = []
    for (ds, backend, run), g in df.groupby(["dataset", "backend", "run"], as_index=False):
        conv = g[g["uq_method"] == "ConVOLT"]
        scp = g[g["uq_method"] == "SCP"]
        if len(conv) == 0 or len(scp) == 0:
            continue
        w_conv = _parse_pm_mean(conv.iloc[0]["interval_size_mean_pm_std_ml"])
        w_scp = _parse_pm_mean(scp.iloc[0]["interval_size_mean_pm_std_ml"])
        if not np.isfinite(w_conv) or not np.isfinite(w_scp) or w_scp <= 0:
            continue
        improvement_pct = 100.0 * float((w_scp - w_conv) / w_scp)

        corr_path = args.uq_root / str(run) / "diagnostics" / "feature_corr_signed.csv"
        if not corr_path.exists():
            corr_path = args.uq_root / str(run) / "diagnostics" / "feature_corr.csv"
        if not corr_path.exists():
            continue
        fc = pd.read_csv(corr_path)
        if "feature" not in fc.columns:
            continue
        rcol = "pearson_r" if "pearson_r" in fc.columns else None
        if rcol is None:
            continue
        fc = fc[np.isfinite(fc[rcol].to_numpy(dtype=float))].copy()
        if len(fc) == 0:
            continue
        fc["abs_r"] = fc[rcol].abs()
        top = fc.sort_values("abs_r", ascending=False).head(int(args.topk_features)).copy()

        row: dict = {
            "dataset": ds,
            "backend": backend,
            "run": run,
            "improvement_over_scp_pct": improvement_pct,
        }
        for i in range(int(args.topk_features)):
            if i < len(top):
                row[f"feature_{i+1}"] = str(top.iloc[i]["feature"])
                row[f"r_{i+1}"] = float(top.iloc[i][rcol])
            else:
                row[f"feature_{i+1}"] = ""
                row[f"r_{i+1}"] = float("nan")
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows produced (missing diagnostics/feature_corr*.csv or main table entries).")

    out = pd.DataFrame(rows).sort_values(["dataset", "backend"]).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")

    # Also report the overall correlation between "mean of top-k |r|" and efficiency gain vs SCP.
    rk = [f"r_{i+1}" for i in range(int(args.topk_features))]
    for c in rk + ["improvement_over_scp_pct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if all(c in out.columns for c in rk) and "improvement_over_scp_pct" in out.columns:
        corrmean = out[rk].abs().mean(axis=1)
        ok = np.isfinite(corrmean.to_numpy()) & np.isfinite(out["improvement_over_scp_pct"].to_numpy())
        if int(np.count_nonzero(ok)) >= 2:
            r = float(np.corrcoef(corrmean.to_numpy()[ok], out["improvement_over_scp_pct"].to_numpy()[ok])[0, 1])
            print(f"Interpretability: N={int(np.count_nonzero(ok))} Pearson_r(mean|r_top{int(args.topk_features)}|, eff_gain_vs_SCP)={r:.4f}")


if __name__ == "__main__":
    main()
