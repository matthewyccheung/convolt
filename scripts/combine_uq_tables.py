#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


_BACKENDS = ("demons", "voxelmorph")
_VM_MODES = ("unsupervised", "supervised", "hybrid")


def _detect_dataset(run_name: str) -> str:
    return str(run_name).split("_", 1)[0]


def _detect_backend(run_name: str) -> str | None:
    for b in _BACKENDS:
        if re.search(rf"(?:^|_)({re.escape(b)})(?:_|$)", run_name):
            return b
    return None


def _is_primary_run(run_name: str, backend: str) -> bool:
    """
    Keep only runs where the second token matches the backend, e.g.:
      - lungct_demons
      - lungct_demons_globalfeat
      - oasis_voxelmorph_supervised_atlas-multi5
    This excludes other registration algorithms that happen to contain "demons"/"voxelmorph"
    in their run name (e.g., lungct_sitk_diffeomorphic_demons).
    """
    parts = str(run_name).split("_")
    return len(parts) >= 2 and parts[1] == str(backend)


def _pick_best_main_run(runs: list[str]) -> str:
    """
    Choose one run per (dataset, backend) for main/ablation tables.
    Heuristics:
      1) Avoid label-only runs (e.g., '*alllabels*') when a non-alllabels run exists.
      2) Prefer '*globalfeat*' when present (keeps tables consistent with globalfeat variants).
      3) Deterministic tie-breaker: lexicographic.
    """
    cand = sorted({str(r).strip() for r in runs if str(r).strip()})
    if not cand:
        raise ValueError("No runs to pick from")
    non_all = [r for r in cand if "alllabels" not in r.lower()]
    if non_all:
        cand = non_all
    gf = [r for r in cand if "globalfeat" in r.lower()]
    if gf:
        cand = gf
    return sorted(cand)[0]


def _detect_vm_mode(run_name: str) -> str | None:
    # Naming convention for Learn2Reg VM runs:
    #   {dataset}_voxelmorph_{train_mode}_{atlas_tag}
    parts = str(run_name).split("_")
    if len(parts) < 3:
        return None
    if parts[1] != "voxelmorph":
        return None
    mode = parts[2]
    return mode if mode in _VM_MODES else None


def _pm_std(mean: float, std: float, fmt: str) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:{fmt}}±{std:{fmt}}"


def _parse_pm_std_mean(s: str) -> float:
    """
    Parse a 'mean±std' string and return the mean as float (NaN on failure).
    """
    if s is None:
        return float("nan")
    t = str(s).strip().replace(" ", "")
    if not t:
        return float("nan")
    if "±" in t:
        t = t.split("±", 1)[0]
    try:
        return float(t)
    except ValueError:
        return float("nan")


def _parse_pm_std(s: str) -> tuple[float, float]:
    """
    Parse a 'mean±std' string into (mean, std). If std is missing, returns std=0.
    Returns (NaN, NaN) on failure.
    """
    if s is None:
        return float("nan"), float("nan")
    t = str(s).strip().replace(" ", "")
    if not t:
        return float("nan"), float("nan")
    if "±" in t:
        a, b = t.split("±", 1)
    else:
        a, b = t, "0"
    try:
        return float(a), float(b)
    except ValueError:
        return float("nan"), float("nan")


def _format_cov_pm_std(s: str) -> str:
    """
    Coverage formatting for paper tables: 2 decimal places for both mean and std.
    """
    m, sd = _parse_pm_std(s)
    return _pm_std(m, sd, ".2f") if np.isfinite(m) else ""


def _format_interval_pm_std(s: str) -> str:
    """
    Interval-size formatting for paper tables: nearest integer (0 decimals) for both mean and std.
    """
    m, sd = _parse_pm_std(s)
    return _pm_std(m, sd, ".0f") if np.isfinite(m) else ""


def _method_rename(s: str) -> str:
    s = str(s)
    s = s.replace("COMPASS", "ConVOLT")
    s = s.replace("ConVOLD", "ConVOLT")
    return s


def _method_group(method: str) -> str:
    """
    Collapse minor string differences and map to paper-facing method names.
    """
    m = _method_rename(method).strip()
    # Drop deprecated exp(beta) methods (beta-CP family).
    if "beta-CP" in m or "constbeta" in m or "global0" in m:
        return ""
    if m.startswith("LocalSCP("):
        return "LCP"
    if m.startswith("wCP("):
        return "RsCP"
    if m == "SCP(|err|)":
        return "SCP"
    if m == "ConVOLT(add-CP)":
        return "ConVOLT(add-CP)"
    if m == "CQR":
        # Feature-based CQR (uses [pred0, deformation features]); keep as an ablation since it also leverages
        # deformation-field features, like ConVOLT.
        return "CQR(feat)"
    if m == "CQR(volonly)":
        # Output-only CQR baseline (uses [pred0] only).
        return "CQR"
    if m == "CQR(k)":
        return "CQR(k)"
    if m == "CQR(beta)":
        # Deprecated; kept only for backward compatibility with old result folders.
        return ""
    if m == "ConVOLT(scale-CP)":
        return "ConVOLT"
    if m == "ConVOLT(scale-CP, constk)":
        return "ConVOLT(constk)"
    if m == "ConVOLT(scale-CP, global1)":
        return "ConVOLT(global1)"
    if m == "ConVOLT(scale-CP,label-local)":
        return "ConVOLT(label-local)"
    if m == "ConVOLT(scale-CP,label-hier)":
        return "ConVOLT(label-hier)"
    if m == "ConVOLT(scale-CP,label+fusion)":
        return "ConVOLT(label+fusion)"
    if m.startswith("RegionCP("):
        # Region-CP methods are named like:
        #   RegionCP(SCP-point, q90)
        #   RegionCP(ConVOLT-point-regionk, q90)
        #   RegionCP(ConVOLT-point-constk, q90)
        if "SCP-point" in m:
            return "SCP"
        if "CQR(volonly)" in m:
            return "CQR"
        if "ConVOLT-point-globalk" in m:
            return "ConVOLT(globalfeat)"
        if "ConVOLT-point-regionk" in m:
            return "ConVOLT(localfeat)"
        if "ConVOLT-point-constk" in m:
            return "ConVOLT(constk)"
        if "LCP" in m or "LocalSCP" in m:
            return "LCP"
        if "RsCP" in m or "wCP" in m:
            return "RsCP"
        return m
    return m


def _write_oasis_label_cqr_convot_tables(*, df_main_long: pd.DataFrame, uq_root: Path, out_dir: Path) -> None:
    """
    OASIS label-only ConVOLT vs CQR comparison tables.

    Uses the run names selected for the main table (dataset=oasis, uq_method=ConVOLT) to keep results consistent.
    Writes:
      - oasis_label_cqr_vs_convot_long.csv (per label_id)
      - oasis_label_cqr_vs_convot_summary.csv (aggregated across labels)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df_main_long.copy()
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["backend"] = df["backend"].astype(str).str.strip().str.lower()
    df = df[(df["dataset"] == "oasis") & (df["uq_method"] == "ConVOLT")].copy()
    if len(df) == 0:
        return

    runs = {str(r["backend"]).strip().lower(): str(r["run"]).strip() for _, r in df.iterrows() if str(r.get("run", "")).strip()}
    rows: list[dict] = []
    for backend, run in runs.items():
        cp_path = uq_root / run / "cp_summary.csv"
        if not cp_path.exists():
            continue
        try:
            cpdf = pd.read_csv(cp_path, skipinitialspace=True)
        except Exception:
            continue
        if not {"target", "label_id", "method", "coverage_mean", "coverage_std", "interval_size_mean", "interval_size_std"}.issubset(set(cpdf.columns)):
            continue
        d = cpdf[(cpdf["target"].astype(str) == "volume_label") & (cpdf["label_id"].astype(int) > 0)].copy()
        if len(d) == 0:
            continue
        d["method"] = d["method"].astype(str).str.strip()
        d["method"] = d["method"].map(_method_rename)
        d["uq_method"] = d["method"].map(_method_group)
        # Keep only the label-focused comparison set.
        keep = {
            "SCP",
            "CQR",
            "ConVOLT",
            "ConVOLT(label-local)",
            "ConVOLT(label-hier)",
            "ConVOLT(label+fusion)",
        }
        d = d[d["uq_method"].isin(sorted(keep))].copy()
        if len(d) == 0:
            continue

        for _, r in d.iterrows():
            lid = int(r["label_id"])
            rows.append(
                {
                    "dataset": "oasis",
                    "backend": backend,
                    "run": run,
                    "label_id": lid,
                    "uq_method": str(r["uq_method"]),
                    "coverage_mean": float(r["coverage_mean"]),
                    "coverage_std": float(0.0 if pd.isna(r["coverage_std"]) else r["coverage_std"]),
                    "interval_size_mean": float(r["interval_size_mean"]),
                    "interval_size_std": float(0.0 if pd.isna(r["interval_size_std"]) else r["interval_size_std"]),
                }
            )

    if not rows:
        return

    long_df = pd.DataFrame(rows)
    long_df["coverage_mean_pm_std"] = [
        _pm_std(float(m), float(s), ".3f") for m, s in zip(long_df["coverage_mean"], long_df["coverage_std"])
    ]
    long_df["interval_size_mean_pm_std_ml"] = [
        _pm_std(float(m), float(s), ".1f") for m, s in zip(long_df["interval_size_mean"], long_df["interval_size_std"])
    ]
    long_df.to_csv(out_dir / "oasis_label_cqr_vs_convot_long.csv", index=False)

    # Aggregated summary across labels (median + IQR is robust).
    out_rows: list[dict] = []
    for (backend, uq_method), g in long_df.groupby(["backend", "uq_method"], as_index=False):
        w = g["interval_size_mean"].to_numpy(dtype=float)
        c = g["coverage_mean"].to_numpy(dtype=float)
        w = w[np.isfinite(w)]
        c = c[np.isfinite(c)]
        if w.size == 0 or c.size == 0:
            continue
        w_med = float(np.median(w))
        w_iqr = float(np.quantile(w, 0.75) - np.quantile(w, 0.25))
        c_med = float(np.median(c))
        c_iqr = float(np.quantile(c, 0.75) - np.quantile(c, 0.25))
        out_rows.append(
            {
                "dataset": "oasis",
                "backend": str(backend),
                "method": str(uq_method),
                "interval_size_median_ml": w_med,
                "interval_size_iqr_ml": w_iqr,
                "coverage_median": c_med,
                "coverage_iqr": c_iqr,
            }
        )
    if out_rows:
        pd.DataFrame(out_rows).sort_values(["backend", "method"]).to_csv(out_dir / "oasis_label_cqr_vs_convot_summary.csv", index=False)


def _load_main_rows(cp_summary_path: Path, *, voxelmorph_mode: str) -> list[dict]:
    run_name = cp_summary_path.parent.name
    dataset = _detect_dataset(run_name)
    backend = _detect_backend(run_name)
    if backend is None:
        return []
    if not _is_primary_run(run_name, backend):
        return []
    if backend == "voxelmorph":
        mode = _detect_vm_mode(run_name)
        if voxelmorph_mode in _VM_MODES:
            # Only filter Learn2Reg VM runs where we can parse a mode token.
            if mode is not None and mode != voxelmorph_mode:
                return []
        # voxelmorph_mode == "any" => no filtering.

    # Many of our CSVs are written in a "pretty" comma+space padded format, so quoted fields may have a leading
    # space before the quote. `skipinitialspace=True` ensures commas inside quoted method names are parsed
    # correctly (otherwise pandas' C engine can mis-tokenize).
    df = pd.read_csv(cp_summary_path, skipinitialspace=True)
    # Some outputs have padded column names (e.g., "target      "); normalize.
    df.columns = [str(c).strip() for c in df.columns]
    if "method" not in df.columns:
        return []

    # Learn2Reg cp_summary has (target,label_id,method,...). Keep union only.
    if "target" in df.columns:
        df["target"] = df["target"].astype(str).str.strip()
        if "label_id" in df.columns:
            df = df[(df["target"] == "volume_union") & (df["label_id"].astype(int) == -1)].copy()
        else:
            df = df[df["target"] == "volume_union"].copy()

    df["method"] = df["method"].astype(str).str.strip()
    df["method"] = df["method"].map(_method_rename)
    df["method_group"] = df["method"].map(_method_group)

    if "coverage_mean_pm_std" not in df.columns:
        if "coverage_mean" in df.columns and "coverage_std" in df.columns:
            df["coverage_mean_pm_std"] = [
                _pm_std(float(m), float(s), ".3f") for m, s in zip(df["coverage_mean"], df["coverage_std"])
            ]
        else:
            df["coverage_mean_pm_std"] = ""

    if "interval_size_mean_pm_std_ml" not in df.columns:
        if "interval_size_mean" in df.columns and "interval_size_std" in df.columns:
            df["interval_size_mean_pm_std_ml"] = [
                _pm_std(float(m), float(s), ".1f") for m, s in zip(df["interval_size_mean"], df["interval_size_std"])
            ]
        else:
            df["interval_size_mean_pm_std_ml"] = ""

    rows: list[dict] = []
    for _, r in df.iterrows():
        mg = str(r.get("method_group", "")).strip()
        if mg == "" or mg.lower() == "nan":
            continue
        cov_s = _format_cov_pm_std(str(r.get("coverage_mean_pm_std", "")))
        int_s = _format_interval_pm_std(str(r.get("interval_size_mean_pm_std_ml", "")))
        rows.append(
            {
                "dataset": dataset,
                "uq_method": mg,
                "backend": backend,
                "coverage_mean_pm_std": cov_s,
                "interval_size_mean_pm_std_ml": int_s,
                "run": run_name,
            }
        )
    return rows


def _load_region_rows(region_summary_path: Path, *, voxelmorph_mode: str) -> list[dict]:
    run_name = region_summary_path.parent.name
    dataset = _detect_dataset(run_name)
    backend = _detect_backend(run_name)
    if backend is None:
        return []
    if not _is_primary_run(run_name, backend):
        return []
    if backend == "voxelmorph":
        mode = _detect_vm_mode(run_name)
        if voxelmorph_mode in _VM_MODES:
            if mode is not None and mode != voxelmorph_mode:
                return []

    df = pd.read_csv(region_summary_path, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    if "method" not in df.columns:
        return []

    # Keep radial shells only (current paper scope).
    if "region_def" in df.columns:
        df = df[df["region_def"].astype(str).str.lower() == "radial"].copy()

    df["method"] = df["method"].astype(str).str.strip().map(_method_rename)
    df["method_group"] = df["method"].map(_method_group)

    def cov_pm(row) -> str:
        m = float(row.get("coverage_patient_mean", np.nan))
        s = float(row.get("coverage_patient_std", np.nan))
        return _pm_std(m, s, ".2f")

    def int_pm(row) -> str:
        m = float(row.get("interval_size_mean", np.nan))
        s = float(row.get("interval_size_std", np.nan))
        return _pm_std(m, s, ".0f")

    rows: list[dict] = []
    for _, r in df.iterrows():
        mg = str(r.get("method_group", "")).strip()
        if mg == "" or mg.lower() == "nan":
            continue
        rows.append(
            {
                "dataset": dataset,
                "region_score": str(r.get("region_score", "")),
                "uq_method": mg,
                "backend": backend,
                "coverage_mean_pm_std": cov_pm(r),
                "interval_size_mean_pm_std_ml": int_pm(r),
                "run": run_name,
            }
        )
    return rows


def _pivot_main(df_long: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict] = []
    keys = ["dataset", "uq_method"]
    for (dataset, uq_method), g in df_long.groupby(keys, as_index=False):
        row = {"dataset": dataset, "method": uq_method}
        for b in _BACKENDS:
            gb = g[g["backend"] == b]
            if len(gb) > 0:
                row[f"coverage_{b}"] = gb.iloc[0]["coverage_mean_pm_std"]
                row[f"interval_size_{b}_ml"] = gb.iloc[0]["interval_size_mean_pm_std_ml"]
            else:
                row[f"coverage_{b}"] = ""
                row[f"interval_size_{b}_ml"] = ""
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def _pivot_region(df_long: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict] = []
    keys = ["dataset", "region_score", "uq_method"]
    for (dataset, region_score, uq_method), g in df_long.groupby(keys, as_index=False):
        row = {"dataset": dataset, "region_score": region_score, "method": uq_method}
        for b in _BACKENDS:
            gb = g[g["backend"] == b]
            if len(gb) > 0:
                row[f"coverage_{b}"] = gb.iloc[0]["coverage_mean_pm_std"]
                row[f"interval_size_{b}_ml"] = gb.iloc[0]["interval_size_mean_pm_std_ml"]
            else:
                row[f"coverage_{b}"] = ""
                row[f"interval_size_{b}_ml"] = ""
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["dataset", "region_score", "method"]).reset_index(drop=True)


def _write_claim_tables(*, df_main_long: pd.DataFrame, df_region_long: pd.DataFrame, out_dir: Path) -> None:
    """
    Extra focused tables for the paper claims.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Claim 2: learning ablation (scale-CP): no learning vs global constant vs input-dependent.
    #   no learning: ConVOLT(global1)
    #   global constant: ConVOLT(constk)
    #   input-dependent: ConVOLT
    df_c2 = df_main_long[df_main_long["uq_method"].isin(["ConVOLT(global1)", "ConVOLT(constk)", "ConVOLT"])].copy()
    if len(df_c2) > 0:
        _pivot_main(df_c2).to_csv(out_dir / "claim2_learning_ablations_table.csv", index=False)
        df_c2.to_csv(out_dir / "claim2_learning_ablations_long.csv", index=False)

    # Claim 1: additive vs multiplicative learned centers (output-space additive vs scale-CP multiplicative).
    # Paper focus: compare ConVOLT variants only (no SCP here).
    df_c1 = df_main_long[df_main_long["uq_method"].isin(["ConVOLT(add-CP)", "ConVOLT"])].copy()
    if len(df_c1) > 0:
        _pivot_main(df_c1).to_csv(out_dir / "claim1_additive_vs_multiplicative_table.csv", index=False)
        df_c1.to_csv(out_dir / "claim1_additive_vs_multiplicative_long.csv", index=False)

    # Claim 3: region guarantees (feature locality): region SCP vs global features vs local features.
    # Paper focus: compare ConVOLT locality ablation only (local vs global features).
    df_c3 = df_region_long[df_region_long["uq_method"].isin(["ConVOLT(globalfeat)", "ConVOLT(localfeat)"])].copy()
    if len(df_c3) > 0:
        _pivot_region(df_c3).to_csv(out_dir / "claim3_region_locality_table.csv", index=False)
        df_c3.to_csv(out_dir / "claim3_region_locality_long.csv", index=False)

    # CQR comparison table: CQR variants vs ConVOLT ridge (scale-CP).
    df_cqr = df_main_long[df_main_long["uq_method"].isin(["ConVOLT", "CQR", "CQR(feat)", "CQR(k)"])].copy()
    if len(df_cqr) > 0:
        _pivot_main(df_cqr).to_csv(out_dir / "cqr_vs_convot_table.csv", index=False)
        df_cqr.to_csv(out_dir / "cqr_vs_convot_long.csv", index=False)


def _write_all_tests_table(*, df_main_long: pd.DataFrame, df_region_long: pd.DataFrame, out_dir: Path, claim3_region_score: str) -> None:
    """
    One combined table that summarizes the 3 ablation tests:
      (1) multiplicative vs additive scalar (ConVOLT vs ConVOLT(add-CP))
      (2) no learning vs no features (ConVOLT(global1) vs ConVOLT(constk))
      (3) local vs global deformation features for regional guarantees (ConVOLT(localfeat) vs ConVOLT(globalfeat))

    For each variant, report:
      - coverage (mean±std) for demons/voxelmorph
      - interval size inflation (%) relative to the corresponding proposed baseline
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _is_primary_run(run: str, backend: str) -> bool:
        parts = str(run).split("_")
        if len(parts) < 2:
            return False
        return parts[1] == str(backend)

    def _run_rank(run: str) -> tuple[int, int]:
        run = str(run)
        # Prefer runs without globalfeat suffix, then prefer shorter names.
        return (1 if "globalfeat" in run else 0, len(run))

    def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep a single canonical run per (dataset, backend) to avoid double-counting:
        - drop non-primary run names (e.g. sitk_*_demons)
        - if multiple runs remain (e.g. *_globalfeat), keep the non-globalfeat shortest run name
        """
        d = df.copy()
        d = d[d.apply(lambda r: _is_primary_run(r["run"], r["backend"]), axis=1)].copy()
        keep_runs: list[str] = []
        for (ds, backend), g in d.groupby(["dataset", "backend"], as_index=False):
            runs = sorted(g["run"].astype(str).unique().tolist(), key=_run_rank)
            if runs:
                keep_runs.append(runs[0])
        return d[d["run"].isin(keep_runs)].reset_index(drop=True)

    df_main_long = _canonicalize(df_main_long)
    # For region results we need BOTH:
    # - localfeat baseline (usually from the non-globalfeat run folder)
    # - globalfeat ablation (usually from the *_globalfeat run folder)
    # So we do not canonicalize df_region_long globally; we pick canonical runs per method below.

    # Baselines for inflation.
    base_global = df_main_long[df_main_long["uq_method"] == "ConVOLT"].copy()
    df_region_sc = df_region_long[df_region_long["region_score"].astype(str).str.lower() == str(claim3_region_score).lower()].copy()
    # Keep only primary runs.
    df_region_sc = df_region_sc[df_region_sc.apply(lambda r: _is_primary_run(r["run"], r["backend"]), axis=1)].copy()

    def _pick_region_run(ds: str, backend: str, *, want_globalfeat: bool) -> str | None:
        g = df_region_sc[(df_region_sc["dataset"] == ds) & (df_region_sc["backend"] == backend)]
        if len(g) == 0:
            return None
        runs = sorted(g["run"].astype(str).unique().tolist(), key=_run_rank)
        if want_globalfeat:
            runs = [r for r in runs if "globalfeat" in r]
        else:
            runs = [r for r in runs if "globalfeat" not in r]
        return runs[0] if runs else None

    # Localfeat baseline rows (canonical non-globalfeat run).
    base_region_rows = []
    for (ds, backend), _ in df_region_sc.groupby(["dataset", "backend"], as_index=False):
        run = _pick_region_run(str(ds), str(backend), want_globalfeat=False)
        if run is None:
            continue
        base_region_rows.append(
            df_region_sc[
                (df_region_sc["dataset"] == str(ds))
                & (df_region_sc["backend"] == str(backend))
                & (df_region_sc["run"] == str(run))
                & (df_region_sc["uq_method"] == "ConVOLT(localfeat)")
            ]
        )
    base_region = pd.concat(base_region_rows, axis=0, ignore_index=True) if base_region_rows else df_region_sc.iloc[0:0].copy()

    def _base_interval_pm(df: pd.DataFrame, dataset: str, backend: str) -> tuple[float, float]:
        d = df[(df["dataset"] == dataset) & (df["backend"] == backend)]
        if len(d) == 0:
            return float("nan"), float("nan")
        return _parse_pm_std(d.iloc[0]["interval_size_mean_pm_std_ml"])

    def _ratio_pm(a_pm: tuple[float, float], b_pm: tuple[float, float]) -> tuple[float, float]:
        """
        Approximate mean±std for ratio R = A/B given (muA,sigA), (muB,sigB),
        assuming independence (cov=0). Used only for presentation.
        """
        mu_a, sig_a = a_pm
        mu_b, sig_b = b_pm
        if not np.isfinite(mu_a) or not np.isfinite(mu_b) or mu_b <= 0:
            return float("nan"), float("nan")
        if not np.isfinite(sig_a):
            sig_a = 0.0
        if not np.isfinite(sig_b):
            sig_b = 0.0
        mu_r = float(mu_a / mu_b)
        var_r = float((sig_a / mu_b) ** 2 + ((mu_a * sig_b) / (mu_b**2)) ** 2)
        sig_r = float(np.sqrt(max(var_r, 0.0)))
        return mu_r, sig_r

    def _pick_rows(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
        return df[df["uq_method"].isin(methods)].copy()

    rows: list[dict] = []

    # Tests 1 & 2 (global).
    tests_global = [
        ("claim1_scalar", {"ConVOLT(add-CP)": "additive"}),
        ("claim2_learning", {"ConVOLT(global1)": "no_learning", "ConVOLT(constk)": "no_features"}),
    ]
    for test_name, mapping in tests_global:
        df_t = _pick_rows(df_main_long, list(mapping.keys()))
        for _, r in df_t.iterrows():
            ds = str(r["dataset"])
            backend = str(r["backend"])
            base_pm = _base_interval_pm(base_global, ds, backend)
            val_pm = _parse_pm_std(r["interval_size_mean_pm_std_ml"])
            ratio_pm = _ratio_pm(val_pm, base_pm)
            infl_mu = float("nan")
            infl_sig = float("nan")
            if np.isfinite(ratio_pm[0]):
                infl_mu = 100.0 * (ratio_pm[0] - 1.0)
                infl_sig = 100.0 * ratio_pm[1]
            rows.append(
                {
                    "dataset": ds,
                    "test": test_name,
                    "variant": mapping[str(r["uq_method"])],
                    "backend": backend,
                    "coverage_mean_pm_std": str(r["coverage_mean_pm_std"]),
                    "interval_inflation_mean_pct": infl_mu,
                    "interval_inflation_std_pct": infl_sig,
                }
            )

    # Test 3 (regional).
    # Pull ConVOLT(globalfeat) from canonical *_globalfeat run folder when available.
    globalfeat_rows = []
    for (ds, backend), _ in df_region_sc.groupby(["dataset", "backend"], as_index=False):
        run = _pick_region_run(str(ds), str(backend), want_globalfeat=True)
        if run is None:
            continue
        globalfeat_rows.append(
            df_region_sc[
                (df_region_sc["dataset"] == str(ds))
                & (df_region_sc["backend"] == str(backend))
                & (df_region_sc["run"] == str(run))
                & (df_region_sc["uq_method"] == "ConVOLT(globalfeat)")
            ]
        )
    df_r = pd.concat(globalfeat_rows, axis=0, ignore_index=True) if globalfeat_rows else df_region_sc.iloc[0:0].copy()

    for _, r in df_r.iterrows():
        ds = str(r["dataset"])
        backend = str(r["backend"])
        base_pm = _base_interval_pm(base_region, ds, backend)
        val_pm = _parse_pm_std(r["interval_size_mean_pm_std_ml"])
        ratio_pm = _ratio_pm(val_pm, base_pm)
        infl_mu = float("nan")
        infl_sig = float("nan")
        if np.isfinite(ratio_pm[0]):
            infl_mu = 100.0 * (ratio_pm[0] - 1.0)
            infl_sig = 100.0 * ratio_pm[1]
        rows.append(
            {
                "dataset": ds,
                "test": "claim3_locality",
                "variant": "global_features",
                "backend": backend,
                "coverage_mean_pm_std": str(r["coverage_mean_pm_std"]),
                "interval_inflation_mean_pct": infl_mu,
                "interval_inflation_std_pct": infl_sig,
            }
        )

    if not rows:
        return

    df_long = pd.DataFrame(rows)
    df_long.to_csv(out_dir / "all_tests_long.csv", index=False)

    # Pivot to paper-friendly wide format.
    out_rows: list[dict] = []
    for (ds, test, variant), g in df_long.groupby(["dataset", "test", "variant"], as_index=False):
        row = {"dataset": ds, "test": test, "variant": variant}
        for b in _BACKENDS:
            gb = g[g["backend"] == b]
            if len(gb) == 0:
                row[f"coverage_{b}"] = ""
                row[f"inflation_{b}_pct"] = ""
                continue
            row[f"coverage_{b}"] = str(gb.iloc[0]["coverage_mean_pm_std"])
            mu = float(gb.iloc[0]["interval_inflation_mean_pct"])
            sig = float(gb.iloc[0]["interval_inflation_std_pct"])
            if np.isfinite(mu):
                if not np.isfinite(sig):
                    sig = 0.0
                row[f"inflation_{b}_pct"] = f"{mu:.1f}±{sig:.1f}"
            else:
                row[f"inflation_{b}_pct"] = ""
        out_rows.append(row)
    df_out = pd.DataFrame(out_rows).sort_values(["dataset", "test", "variant"]).reset_index(drop=True)
    df_out.to_csv(out_dir / "all_tests_table.csv", index=False)


def _write_split_tables(*, df_long: pd.DataFrame, out_dir: Path, kind: str) -> None:
    """
    Write main vs ablation tables.
    """
    if kind == "region":
        # Region tables (claim 3): localized vs global deformation features.
        # Main: proposed (local features) vs output-space baselines.
        # Ablation: global features (whole-mask deformation features).
        main_methods = ["ConVOLT(localfeat)", "CQR", "SCP"]
        ablation_methods = ["ConVOLT(globalfeat)"]
    else:
        # Main results: requested baselines only.
        main_methods = ["ConVOLT", "LCP", "SCP", "CQR"]
        # Main ablations aligned to paper tests:
        #  (1) multiplicative vs additive scalar: ConVOLT vs ConVOLT(add-CP)
        #  (2) no learning vs no features: ConVOLT(global1) vs ConVOLT(constk)
        ablation_methods = ["ConVOLT(add-CP)", "ConVOLT(global1)", "ConVOLT(constk)"]

    df_main = df_long[df_long["uq_method"].isin(main_methods)].copy()
    df_ab = df_long[df_long["uq_method"].isin(ablation_methods)].copy()

    if kind == "main":
        if len(df_main) > 0:
            _pivot_main(df_main).to_csv(out_dir / "main_results_table_main.csv", index=False)
            df_main.to_csv(out_dir / "main_results_long_main.csv", index=False)
        if len(df_ab) > 0:
            # For readability, rename ablation method labels to match the paper tests.
            ab_ren = {
                "ConVOLT(add-CP)": "additive",
                "ConVOLT(constk)": "no_features",
                "ConVOLT(global1)": "no_learning",
            }
            df_ab2 = df_ab.copy()
            df_ab2["uq_method"] = df_ab2["uq_method"].map(lambda x: ab_ren.get(str(x), str(x)))
            _pivot_main(df_ab2).to_csv(out_dir / "main_results_table_ablations.csv", index=False)
            df_ab2.to_csv(out_dir / "main_results_long_ablations.csv", index=False)
    else:
        if len(df_main) > 0:
            _pivot_region(df_main).to_csv(out_dir / "region_results_table_main.csv", index=False)
            df_main.to_csv(out_dir / "region_results_long_main.csv", index=False)
        if len(df_ab) > 0:
            _pivot_region(df_ab).to_csv(out_dir / "region_results_table_ablations.csv", index=False)
            df_ab.to_csv(out_dir / "region_results_long_ablations.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine per-run UQ summaries into paper-ready tables.")
    import os

    uq_root_default = Path(os.environ.get("CONVOLT_UQ_ROOT", "uq_results"))
    ap.add_argument("--uq_root", type=Path, default=uq_root_default, help="Root directory containing per-run UQ outputs.")
    ap.add_argument("--out_dir", type=Path, default=uq_root_default / "_tables", help="Output directory for combined tables.")
    ap.add_argument(
        "--voxelmorph_mode",
        choices=["unsupervised", "supervised", "hybrid", "any"],
        default="unsupervised",
        help="Which Learn2Reg VoxelMorph runs to include: unsupervised/supervised/hybrid, or any.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset allowlist (e.g. nlst,lungct,acdc). If empty, include all datasets found.",
    )
    ap.add_argument("--claim3_region_score", type=str, default="q90", help="Region score to use for claim 3 in all_tests_table.csv (default: q90).")
    args = ap.parse_args()

    uq_root = args.uq_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_datasets = {d.strip().lower() for d in str(args.datasets).split(",") if d.strip()}

    main_rows: list[dict] = []
    region_rows: list[dict] = []

    for cp_path in sorted(uq_root.glob("*/cp_summary.csv")):
        if allow_datasets and _detect_dataset(cp_path.parent.name).lower() not in allow_datasets:
            continue
        main_rows.extend(_load_main_rows(cp_path, voxelmorph_mode=str(args.voxelmorph_mode)))
    for rg_path in sorted(uq_root.glob("*/region_cp_summary.csv")):
        if allow_datasets and _detect_dataset(rg_path.parent.name).lower() not in allow_datasets:
            continue
        region_rows.extend(_load_region_rows(rg_path, voxelmorph_mode=str(args.voxelmorph_mode)))

    if main_rows:
        df_main_long = pd.DataFrame(main_rows)

        # Choose a single (best) run per (dataset, backend) for the main/ablation tables.
        chosen = []
        for (ds, be), g in df_main_long.groupby(["dataset", "backend"], as_index=False):
            run = _pick_best_main_run(g["run"].tolist())
            chosen.append({"dataset": ds, "backend": be, "run": run})
        chosen_df = pd.DataFrame(chosen)
        df_main_long = df_main_long.merge(chosen_df, on=["dataset", "backend", "run"], how="inner")

        df_main = _pivot_main(df_main_long)
        df_main.to_csv(out_dir / "main_results_table.csv", index=False)
        df_main_long.to_csv(out_dir / "main_results_long.csv", index=False)
        _write_split_tables(df_long=df_main_long, out_dir=out_dir, kind="main")
    else:
        print(f"No cp_summary.csv files found under {uq_root}")

    if region_rows:
        df_region_long = pd.DataFrame(region_rows)
        df_region = _pivot_region(df_region_long)
        df_region.to_csv(out_dir / "region_results_table.csv", index=False)
        df_region_long.to_csv(out_dir / "region_results_long.csv", index=False)
        _write_split_tables(df_long=df_region_long, out_dir=out_dir, kind="region")
    else:
        print(f"No region_cp_summary.csv files found under {uq_root}")

    if main_rows and region_rows:
        _write_claim_tables(df_main_long=df_main_long, df_region_long=df_region_long, out_dir=out_dir)
        _write_all_tests_table(
            df_main_long=df_main_long,
            df_region_long=df_region_long,
            out_dir=out_dir,
            claim3_region_score=str(args.claim3_region_score),
        )
    if main_rows:
        _write_oasis_label_cqr_convot_tables(df_main_long=df_main_long, uq_root=uq_root, out_dir=out_dir)

    print(f"Wrote combined tables under: {out_dir}")


if __name__ == "__main__":
    main()
