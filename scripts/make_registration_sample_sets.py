#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Ensure repo root is importable when running as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def _pick_slice_from_mask(mask_zyx: np.ndarray) -> int:
    m = np.asarray(mask_zyx) > 0
    if m.ndim != 3 or not np.any(m):
        return int(mask_zyx.shape[0] // 2)
    areas = m.reshape(m.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def _overlay_contour(ax, mask_yx: np.ndarray, *, color: str = "#00FF66", lw: float = 1.2, alpha: float = 0.9) -> None:
    import matplotlib.pyplot as plt

    m = (np.asarray(mask_yx) > 0).astype(np.uint8)
    if not np.any(m):
        return
    try:
        ax.contour(m, levels=[0.5], colors=[color], linewidths=[lw], alpha=alpha)
    except Exception:
        # Matplotlib can throw on degenerate contours; ignore quietly.
        pass


def _overlay_multilabel_contours(ax, lab_yx: np.ndarray, *, lw: float = 0.8, alpha: float = 0.9) -> None:
    """
    Overlay contours for each non-background integer label with a deterministic per-label color.
    Intended for small label sets (e.g., OASIS ~35 labels) on a single 2D slice.
    """
    import matplotlib.pyplot as plt

    lab = np.asarray(lab_yx)
    if lab.ndim != 2:
        return
    lids = [int(x) for x in np.unique(lab) if int(x) > 0]
    if not lids:
        return
    # Stable color assignment: map label id -> color via a cyclic colormap.
    cmap = plt.get_cmap("tab20")
    for lid in lids:
        m = (lab == lid).astype(np.uint8)
        if not np.any(m):
            continue
        col = cmap((lid - 1) % 20)
        try:
            ax.contour(m, levels=[0.5], colors=[col], linewidths=[lw], alpha=alpha)
        except Exception:
            pass


@dataclass(frozen=True)
class SamplePaths:
    case_id: str
    artifacts: Path


def _list_cases(results_dir: Path) -> list[SamplePaths]:
    pairs = results_dir / "pairs"
    if not pairs.exists():
        return []
    out: list[SamplePaths] = []
    for c in sorted([p for p in pairs.iterdir() if p.is_dir()]):
        ap = c / "artifacts.npz"
        if ap.exists():
            out.append(SamplePaths(case_id=c.name, artifacts=ap))
    return out


def _resolve_results_dir(*, results_root: Path, dataset: str, backend: str, oasis_vm_mode: str) -> Path:
    ds = str(dataset).strip().lower()
    b = str(backend).strip().lower()
    if ds in {"nlst", "lungct", "acdc"}:
        return (results_root / f"{ds}_{b}").resolve()
    if ds == "oasis":
        if b == "demons":
            return (results_root / "oasis_demons_atlas-multi5").resolve()
        if b == "voxelmorph":
            mode = str(oasis_vm_mode).strip().lower()
            if mode not in {"unsupervised", "supervised", "hybrid"}:
                mode = "supervised"
            return (results_root / f"oasis_voxelmorph_{mode}_atlas-multi5").resolve()
    raise ValueError(f"Unsupported dataset/backend: {dataset}/{backend}")


def _load_oasis_label_map(*, learn2reg_root: Path) -> dict[str, Path]:
    from reg.learn2reg_dataset import load_learn2reg_cases

    cases = load_learn2reg_cases(learn2reg_root / "OASIS", "training")
    out: dict[str, Path] = {}
    for c in cases:
        if c.label is not None:
            out[str(c.patient_id)] = Path(c.label)
    return out


def _load_oasis_atlas_image_path(*, results_dir: Path, learn2reg_root: Path) -> tuple[str, Path] | None:
    """
    Resolve a representative atlas image path for an OASIS Learn2Reg run.

    For multi-atlas runs, we pick the first atlas id listed in atlas_meta.json.
    This is only for visualization (to show "atlas before warping").
    """
    import json

    meta_path = results_dir / "atlas_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None

    atlas_ids = meta.get("atlas_ids")
    if not isinstance(atlas_ids, list) or not atlas_ids:
        return None
    atlas_id = str(atlas_ids[0])
    img_path = (learn2reg_root / "OASIS" / "imagesTr" / f"{atlas_id}_0000.nii.gz").resolve()
    if not img_path.exists():
        return None
    return atlas_id, img_path


def _plot_case(
    *,
    dataset: str,
    backend: str,
    case_id: str,
    artifacts_path: Path,
    fixed_label_path: Path | None,
    oasis_atlas_image: tuple[str, Path] | None = None,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    try:
        z = np.load(artifacts_path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load {artifacts_path}: {e}")

    ds = str(dataset).strip().lower()

    # Load arrays (dataset-specific).
    if ds in {"nlst", "lungct"}:
        fixed_img = np.asarray(z["inhale_ct_zyx"], dtype=np.float32)
        moving_img = np.asarray(z["exhale_ct_resampled_zyx"], dtype=np.float32)
        warped_img = np.asarray(z["exhale_ct_warped_zyx"], dtype=np.float32) if "exhale_ct_warped_zyx" in z.files else moving_img
        fixed_label = np.asarray(z["inhale_mask_zyx"], dtype=np.uint8)
        moving_label = np.asarray(z["exhale_mask_resampled_zyx"], dtype=np.uint8) if "exhale_mask_resampled_zyx" in z.files else None
        warped_label = np.asarray(z["exhale_mask_warped_zyx"], dtype=np.uint8)
    elif ds == "oasis":
        fixed_img = np.asarray(z["fixed_image_zyx"], dtype=np.float32)
        # We don't store the raw moving atlas image in artifacts; `moving_warped_zyx` is the (mean) atlas warped-to-fixed image.
        atlas_warped_img = np.asarray(z["moving_warped_zyx"], dtype=np.float32) if "moving_warped_zyx" in z.files else fixed_img
        warped_img = atlas_warped_img
        atlas_raw_img = None
        atlas_raw_id = None
        atlas_raw_lab = None
        if oasis_atlas_image is not None:
            atlas_raw_id, atlas_raw_path = oasis_atlas_image
            try:
                from reg.nifti import read_nifti, resample_zyx_to_shape

                atlas_raw_img = read_nifti(atlas_raw_path).data_zyx.astype(np.float32, copy=False)
                if atlas_raw_img.shape != fixed_img.shape:
                    atlas_raw_img = resample_zyx_to_shape(atlas_raw_img, tuple(int(x) for x in fixed_img.shape), order=1).astype(np.float32, copy=False)
                # Also load this atlas's labels (for visualization only).
                atlas_lab_path = (Path(atlas_raw_path).parents[1] / "labelsTr" / f"{atlas_raw_id}_0000.nii.gz").resolve()
                if atlas_lab_path.exists():
                    atlas_raw_lab = read_nifti(atlas_lab_path).data_zyx
                    if atlas_raw_lab.shape != fixed_img.shape:
                        atlas_raw_lab = resample_zyx_to_shape(atlas_raw_lab, tuple(int(x) for x in fixed_img.shape), order=0)
            except Exception:
                atlas_raw_img = None
        warped_label = np.asarray(z["pred_label_zyx"], dtype=np.uint16)
        fixed_label = None
        if fixed_label_path is not None and fixed_label_path.exists():
            from reg.nifti import read_nifti

            gt = read_nifti(fixed_label_path).data_zyx
            if gt.shape == fixed_img.shape:
                fixed_label = gt
        moving_img = atlas_raw_img if atlas_raw_img is not None else fixed_img
        moving_label = atlas_raw_lab
    else:
        raise ValueError(f"Unsupported dataset for plotting: {dataset}")

    disp_mag = np.asarray(z["disp_mag_mm_zyx"], dtype=np.float32) if "disp_mag_mm_zyx" in z.files else None
    if disp_mag is None and "disp_mm_3zyx" in z.files:
        d = np.asarray(z["disp_mm_3zyx"], dtype=np.float32)
        disp_mag = np.sqrt(np.sum(d * d, axis=0))
    if disp_mag is None:
        disp_mag = np.zeros_like(fixed_img, dtype=np.float32)

    # Choose slice from label union (prefer GT if available).
    if fixed_label is not None:
        sl = _pick_slice_from_mask(fixed_label)
    else:
        sl = _pick_slice_from_mask(warped_label)

    fx = fixed_img[sl]
    mx = moving_img[sl]
    wx = warped_img[sl]
    dm = disp_mag[sl]

    vmin_f, vmax_f = _pct_window(fx, 1.0, 99.0)
    vmin_m, vmax_m = _pct_window(mx, 1.0, 99.0)
    vmin_w, vmax_w = _pct_window(wx, 1.0, 99.0)
    vmin_d, vmax_d = 0.0, float(np.percentile(dm[np.isfinite(dm)], 99.0)) if np.any(np.isfinite(dm)) else 1.0
    if not np.isfinite(vmax_d) or vmax_d <= 0:
        vmax_d = 1.0

    # Requested grid:
    # fixed image, fixed label, moving/atlas image, moving/atlas label,
    # warped image, warped label, displacement field.
    fig, axes = plt.subplots(1, 7, figsize=(19.0, 3.2), constrained_layout=True)
    axes = list(axes)

    # Panel 1: fixed image.
    axes[0].imshow(fx, cmap="gray", vmin=vmin_f, vmax=vmax_f)
    axes[0].set_title("Fixed image")

    # Panel 2: fixed label.
    if fixed_label is not None:
        fl = fixed_label[sl]
        if fl.dtype.kind in {"u", "i"} and int(np.max(fl)) > 1:
            axes[1].imshow(fl, cmap="nipy_spectral", interpolation="nearest")
            axes[1].set_title("Fixed GT labels")
        else:
            axes[1].imshow((fl > 0).astype(np.uint8), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[1].set_title("Fixed GT label")
    else:
        axes[1].imshow(np.zeros_like(fx, dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Fixed label (missing)")

    # Panel 3: moving/atlas image.
    axes[2].imshow(mx, cmap="gray", vmin=vmin_m, vmax=vmax_m)
    if ds == "oasis" and oasis_atlas_image is not None:
        axes[2].set_title(f"Atlas image [{oasis_atlas_image[0]}]")
    else:
        axes[2].set_title("Moving image")

    # Panel 4: moving/atlas label.
    if moving_label is not None:
        ml = moving_label[sl]
        if ml.dtype.kind in {"u", "i"} and int(np.max(ml)) > 1:
            axes[3].imshow(ml, cmap="nipy_spectral", interpolation="nearest")
            axes[3].set_title("Atlas labels" if ds == "oasis" else "Moving label")
        else:
            axes[3].imshow((ml > 0).astype(np.uint8), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[3].set_title("Atlas label" if ds == "oasis" else "Moving label")
    else:
        axes[3].imshow(np.zeros_like(fx, dtype=np.float32), cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Atlas label (missing)" if ds == "oasis" else "Moving label (missing)")

    # Panel 5: warped image (moving image after warping into fixed space).
    axes[4].imshow(wx, cmap="gray", vmin=vmin_w, vmax=vmax_w)
    axes[4].set_title("Warped image")

    # Panel 6: warped label.
    wl = warped_label[sl]
    if wl.dtype.kind in {"u", "i"} and int(np.max(wl)) > 1:
        axes[5].imshow(wl, cmap="nipy_spectral", interpolation="nearest")
        axes[5].set_title("Warped labels")
    else:
        axes[5].imshow((wl > 0).astype(np.uint8), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[5].set_title("Warped label")

    # Panel 7: displacement magnitude.
    im = axes[6].imshow(dm, cmap="magma", vmin=vmin_d, vmax=vmax_d)
    axes[6].set_title("|u| (mm)")
    fig.colorbar(im, ax=axes[6], fraction=0.046, pad=0.02)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{dataset} | {backend} | {case_id} | z={sl}", y=1.02, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def _intersection_case_ids(groups: list[list[SamplePaths]]) -> list[str]:
    if not groups:
        return []
    s = set(p.case_id for p in groups[0])
    for g in groups[1:]:
        s &= set(p.case_id for p in g)
    return sorted(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make registration sample visualizations (fixed/moving/disp/labels).")
    ap.add_argument("--results_root", type=Path, default=Path(os.environ.get("CONVOLT_RESULTS_ROOT", "/scratch/yc130/Registration/outputs")))
    ap.add_argument("--learn2reg_root", type=Path, default=Path(os.environ.get("CONVOLT_DATA_ROOT", "/scratch/yc130/Registration")))
    ap.add_argument("--datasets", type=str, default="lungct,nlst,oasis")
    ap.add_argument("--backends", type=str, default="demons,voxelmorph")
    ap.add_argument("--oasis_voxelmorph_mode", type=str, default="supervised", choices=["unsupervised", "supervised", "hybrid"])
    ap.add_argument("--n", type=int, default=3, help="Number of cases per dataset (shared across backends).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=Path("uq_results") / "_figures_paper" / "registration_samples")
    args = ap.parse_args()

    datasets = [d.strip().lower() for d in str(args.datasets).split(",") if d.strip()]
    backends = [b.strip().lower() for b in str(args.backends).split(",") if b.strip()]
    if not backends:
        backends = ["demons", "voxelmorph"]

    rng = random.Random(int(args.seed))

    oasis_label_map: dict[str, Path] = {}
    if "oasis" in datasets:
        try:
            oasis_label_map = _load_oasis_label_map(learn2reg_root=args.learn2reg_root)
        except Exception as e:
            print(f"[warn] failed to load OASIS label map from learn2reg_root={args.learn2reg_root}: {e}")
            oasis_label_map = {}

    for ds in datasets:
        # Resolve runs for each backend and list cases.
        runs: dict[str, Path] = {}
        cases_by_backend: dict[str, list[SamplePaths]] = {}
        for b in backends:
            try:
                rd = _resolve_results_dir(results_root=args.results_root, dataset=ds, backend=b, oasis_vm_mode=str(args.oasis_voxelmorph_mode))
            except Exception:
                continue
            runs[b] = rd
            cases_by_backend[b] = _list_cases(rd)

        if not cases_by_backend:
            continue

        common_ids = _intersection_case_ids(list(cases_by_backend.values()))
        if not common_ids:
            # Fall back to the first backend's list.
            first_b = sorted(cases_by_backend.keys())[0]
            common_ids = [p.case_id for p in cases_by_backend[first_b]]

        if not common_ids:
            continue

        # For OASIS, prefer cases that have GT labels available (training IDs).
        if ds == "oasis" and oasis_label_map:
            common_ids = [cid for cid in common_ids if cid in oasis_label_map]
            if not common_ids:
                continue

        n = int(max(1, args.n))
        pick = common_ids if len(common_ids) <= n else rng.sample(common_ids, n)
        pick = sorted(pick)

        for b, rd in runs.items():
            oasis_atlas_image = None
            if ds == "oasis":
                oasis_atlas_image = _load_oasis_atlas_image_path(results_dir=rd, learn2reg_root=args.learn2reg_root)
            case_map = {p.case_id: p for p in cases_by_backend.get(b, [])}
            for cid in pick:
                sp = case_map.get(cid)
                if sp is None:
                    continue
                gt_path = None
                if ds == "oasis":
                    gt_path = oasis_label_map.get(str(cid))
                    if gt_path is None:
                        # Fallback for OASIS training labels: labelsTr/<patient_id>_0000.nii.gz
                        cand = (args.learn2reg_root / "OASIS" / "labelsTr" / f"{cid}_0000.nii.gz").resolve()
                        if cand.exists():
                            gt_path = cand
                out_path = args.out_dir / ds / b / f"{cid}.png"
                _plot_case(
                    dataset=ds,
                    backend=b,
                    case_id=cid,
                    artifacts_path=sp.artifacts,
                    fixed_label_path=gt_path,
                    oasis_atlas_image=oasis_atlas_image,
                    out_path=out_path,
                )

    print(f"Wrote samples under: {args.out_dir}")


if __name__ == "__main__":
    main()
