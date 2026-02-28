from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .acdc_dataset import load_acdc_pairs
from .acdc_pipeline import process_acdc_pair
from .dataset import load_pairs
from .demons import DemonsParams
from .paths import default_dataset_dir, default_results_dir, default_results_dir_tagged
from .pipeline import process_pair
from .viz import save_pair_figure, save_patient_figure


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified registration runner for NLST/LungCT/ACDC + Learn2Reg tasks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="Run registration and save figures/artifacts/summary.csv")
    reg.add_argument(
        "--dataset",
        choices=["nlst", "lungct", "acdc", "hippocampusmr", "oasis", "abdomenctct"],
        required=True,
    )
    reg.add_argument(
        "--method",
        choices=["demons", "voxelmorph", "itk_elastix_bspline", "sitk_diffeomorphic_demons", "sitk_bspline"],
        required=True,
        help="Registration method/backend (itk_elastix_bspline requires the pip package itk-elastix).",
    )
    reg.add_argument("--dataset_dir", type=Path, default=None, help="Override dataset root directory.")
    reg.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Override results directory (default: /scratch/yc130/Registration/outputs/{dataset}_{method}[_{atlas_tag}]).",
    )
    reg.add_argument("--max_pairs", type=int, default=0)
    reg.add_argument("--patient_id", action="append", default=[])
    reg.add_argument("--split", type=str, default=None, help="NLST: training/val/test. ACDC: training/testing.")

    # Demons params (also used for masks/fig pipeline in general).
    reg.add_argument("--iters", type=str, default="80,40,20")
    reg.add_argument("--levels", type=str, default="4,2,1")
    reg.add_argument("--step", type=float, default=1.0)
    reg.add_argument("--diffusion_sigma", type=float, default=1.5)
    reg.add_argument("--mask_dilate", type=int, default=3)
    reg.add_argument("--phase_policy", default="mask_volume", choices=["mask_volume", "suffix", "json"])
    reg.add_argument("--intensity_norm", choices=["ct_hu", "percentile", "minmax"], default=None)
    reg.add_argument("--intensity_ct_clip", type=str, default="-1000,400")
    reg.add_argument("--intensity_pct", type=str, default="1,99")

    # VoxelMorph options (if method=voxelmorph).
    reg.add_argument("--vm_device", choices=["auto", "cpu", "cuda"], default="auto")
    reg.add_argument("--vm_scale", type=int, default=1)
    reg.add_argument("--vm_weights", type=Path, default=None)
    reg.add_argument("--vm_train_epochs", type=int, default=0)
    reg.add_argument("--vm_train_mode", choices=["none", "unsupervised", "supervised", "hybrid"], default="unsupervised")
    reg.add_argument("--vm_train_cases", type=int, default=None, help="Learn2Reg-only: number of labeled subjects used for VM training.")
    reg.add_argument("--vm_steps_per_epoch", type=int, default=100)
    reg.add_argument("--vm_lr", type=float, default=1e-4)
    reg.add_argument("--vm_smooth", type=float, default=0.05)
    reg.add_argument("--vm_train_max_pairs", type=int, default=20)
    reg.add_argument("--vm_norm", choices=["ct_hu", "percentile", "minmax"], default=None)
    reg.add_argument("--vm_ct_clip", type=str, default="-1000,400")
    reg.add_argument("--vm_pct", type=str, default="1,99")
    reg.add_argument("--vm_sim", choices=["mse", "ncc"], default=None)
    reg.add_argument("--vm_ncc_win", type=int, default=9)
    reg.add_argument("--vm_mask_labels", type=str, default="1,2,3", help="ACDC-only: similarity mask labels during VM training.")

    # Learn2Reg atlas options (inter-patient datasets only).
    reg.add_argument("--atlas_mode", choices=["multi", "single", "average"], default="multi")
    reg.add_argument("--atlas_n", type=int, default=5)
    reg.add_argument("--atlas_seed", type=int, default=0)
    reg.add_argument("--atlas_ids", type=str, default="", help="Optional comma-separated atlas subject IDs to use (e.g. OASIS_0001,OASIS_0002).")
    reg.add_argument("--atlas_keep_intermediates", action="store_true", help="Keep per-atlas intermediate outputs (debugging).")

    reg.add_argument("--no_nifti_outputs", action="store_true")
    return p.parse_args(argv)


def _build_demons_params(args: argparse.Namespace, *, dataset: str) -> DemonsParams:
    levels = tuple(int(x) for x in str(args.levels).split(",") if x.strip())
    iters = tuple(int(x) for x in str(args.iters).split(",") if x.strip())
    ct_lo, ct_hi = (float(x) for x in str(args.intensity_ct_clip).split(","))
    p_lo, p_hi = (float(x) for x in str(args.intensity_pct).split(","))
    if args.intensity_norm is None:
        intensity_norm = "ct_hu" if dataset in {"nlst", "lungct", "abdomenctct"} else "percentile"
    else:
        intensity_norm = str(args.intensity_norm)
    return DemonsParams(
        levels=levels,
        iterations=iters,
        update_step=float(args.step),
        diffusion_sigma=float(args.diffusion_sigma),
        mask_dilate_radius=int(args.mask_dilate),
        intensity_norm=intensity_norm,
        intensity_clip_hu=(ct_lo, ct_hi),
        intensity_pct=(p_lo, p_hi),
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    if args.cmd != "register":
        raise ValueError("Only 'register' is supported in this CLI (use reg.uq.cli for UQ).")

    dataset = str(args.dataset).lower()
    method = str(args.method).lower()

    is_learn2reg = dataset in {"hippocampusmr", "oasis", "abdomenctct"}
    atlas_spec = None
    atlas_ids_cli = tuple(s.strip() for s in str(args.atlas_ids).split(",") if s.strip()) if is_learn2reg else tuple()
    atlas_tag_str = ""
    if is_learn2reg:
        from .atlas import AtlasSpec, atlas_tag

        atlas_spec = AtlasSpec(
            mode=str(args.atlas_mode),
            n=int(args.atlas_n),
            seed=int(args.atlas_seed),
            atlas_ids=atlas_ids_cli if len(atlas_ids_cli) > 0 else None,
        )
        atlas_tag_str = atlas_tag(atlas_spec)

    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else (default_results_dir_tagged(dataset, method, atlas_tag_str) if is_learn2reg else default_results_dir(dataset, method))
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_dir if args.dataset_dir is not None else default_dataset_dir(dataset)
    params = _build_demons_params(args, dataset=dataset)

    # Learn2Reg atlas setup (inter-patient).
    learn2reg_train_cases = None
    learn2reg_atlas_ids: tuple[str, ...] | None = None
    learn2reg_atlas_cases = None
    learn2reg_label_ids: list[int] | None = None
    learn2reg_atlas_template = None
    learn2reg_meta_path = None
    learn2reg_vm_train_ids: list[str] | None = None
    if is_learn2reg:
        if atlas_spec is None:
            raise RuntimeError("atlas_spec unexpectedly None for Learn2Reg dataset")
        from .atlas import build_average_atlas, select_atlas_ids, write_atlas_meta
        from .learn2reg_dataset import load_learn2reg_cases

        learn2reg_meta_path = results_dir / "atlas_meta.json"
        learn2reg_train_cases = load_learn2reg_cases(dataset_dir, "training")
        train_ids = [str(c.patient_id) for c in learn2reg_train_cases]

        # Reuse atlas ids from meta if present (for reproducibility).
        meta = None
        if learn2reg_meta_path.exists():
            try:
                meta = json.loads(learn2reg_meta_path.read_text())
                meta_atlas = tuple(map(str, meta.get("atlas_ids", []))) or None
                if meta_atlas is not None:
                    learn2reg_atlas_ids = meta_atlas
                if "vm_train_ids" in meta:
                    learn2reg_vm_train_ids = list(map(str, meta.get("vm_train_ids", [])))
            except Exception:
                meta = None

        if learn2reg_atlas_ids is None:
            learn2reg_atlas_ids = select_atlas_ids(candidate_ids=train_ids, spec=atlas_spec)

        atlas_id_set = set(learn2reg_atlas_ids)
        learn2reg_atlas_cases = [c for c in learn2reg_train_cases if str(c.patient_id) in atlas_id_set]
        if len(learn2reg_atlas_cases) == 0:
            raise ValueError("No atlas cases selected (check --atlas_ids/--atlas_seed)")

        # Derive label IDs (exclude background 0).
        label_ids: list[int] = []
        try:
            js_path = sorted(Path(dataset_dir).glob("*_dataset.json"))[0]
            js = json.loads(Path(js_path).read_text())
            lbls = js.get("labels", None)
            if isinstance(lbls, dict):
                if "0" in lbls and isinstance(lbls["0"], dict) and len(lbls["0"]) > 0:
                    label_ids = sorted(int(k) for k in lbls["0"].keys())
                elif len(lbls) > 0:
                    # Some Learn2Reg tasks encode labels as {"0": {}} (empty) or only include background.
                    # Treat that as "unknown labels" and fall back to scanning a label map.
                    cand = sorted(int(k) for k in lbls.keys())
                    label_ids = [x for x in cand if int(x) != 0]
        except Exception:
            label_ids = []
        label_ids = [int(x) for x in label_ids if int(x) != 0]
        if not label_ids:
            # Fallback: scan atlas labels and union their unique values (robust if one label happens to be empty).
            from .nifti import read_nifti

            uniq: set[int] = set()
            for c in learn2reg_atlas_cases[: min(10, len(learn2reg_atlas_cases))]:
                if getattr(c, "label", None) is None:
                    continue
                li = read_nifti(c.label)
                vals = np.unique(li.data_zyx)
                for v in vals.tolist():
                    iv = int(v)
                    if iv != 0:
                        uniq.add(iv)
                if uniq:
                    break
            label_ids = sorted(uniq)
        if not label_ids:
            raise ValueError("Could not determine non-background label IDs for this dataset")
        learn2reg_label_ids = label_ids

        # Optional: build average atlas template.
        if str(atlas_spec.mode).lower() == "average":
            learn2reg_atlas_template = build_average_atlas(atlas_cases=learn2reg_atlas_cases, out_dir=results_dir / "atlas")

        # Persist atlas meta early (vm_train_ids may be filled later for voxelmorph).
        write_atlas_meta(
            out_path=learn2reg_meta_path,
            dataset=dataset,
            spec=atlas_spec,
            atlas_ids=learn2reg_atlas_ids,
            vm_train_ids=learn2reg_vm_train_ids if learn2reg_vm_train_ids else None,
        )

    # VoxelMorph model/cfg (only if requested).
    vxm_model = None
    vxm_cfg = None
    if method == "voxelmorph":
        from .vxm import VoxelMorphDense, VxmConfig, load_weights, save_weights, train_hybrid, train_supervised_seg, train_unsupervised

        ct_lo, ct_hi = (float(x) for x in str(args.vm_ct_clip).split(","))
        p_lo, p_hi = (float(x) for x in str(args.vm_pct).split(","))
        # Dataset-specific defaults.
        if args.vm_norm is None:
            vm_norm = "ct_hu" if dataset in {"nlst", "lungct", "abdomenctct"} else "percentile"
        else:
            vm_norm = str(args.vm_norm)
        if args.vm_sim is None:
            vm_sim = "mse" if dataset in {"nlst", "lungct", "abdomenctct"} else "ncc"
        else:
            vm_sim = str(args.vm_sim)

        vxm_cfg = VxmConfig(
            device=args.vm_device,
            scale=int(args.vm_scale),
            smooth_weight=float(args.vm_smooth),
            lr=float(args.vm_lr),
            epochs=int(args.vm_train_epochs),
            steps_per_epoch=int(args.vm_steps_per_epoch),
            weights_path=str(args.vm_weights) if args.vm_weights else None,
            norm=vm_norm,
            ct_clip=(ct_lo, ct_hi),
            pct=(p_lo, p_hi),
            sim=vm_sim,
            ncc_win=int(args.vm_ncc_win),
        )
        vxm_model = VoxelMorphDense()

        default_weight_path = results_dir / "voxelmorph.pt"
        weight_path = None
        if args.vm_weights is not None:
            weight_path = Path(args.vm_weights)
        elif default_weight_path.exists() and int(args.vm_train_epochs) <= 0:
            weight_path = default_weight_path

        if weight_path is not None and Path(weight_path).exists():
            load_weights(vxm_model, Path(weight_path), device=args.vm_device)
        else:
            train_mode = str(args.vm_train_mode).lower()
            if train_mode == "none":
                raise ValueError("voxelmorph requires --vm_weights (or an existing results_dir/voxelmorph.pt) when --vm_train_mode none")
            if int(args.vm_train_epochs) <= 0:
                raise ValueError("voxelmorph requires --vm_train_epochs > 0 when no weights are provided")

            if dataset in {"nlst", "lungct"}:
                split = args.split or ("all" if dataset == "lungct" else "training")
                pairs = load_pairs(dataset_dir, split)
                if args.patient_id:
                    keep = set(args.patient_id)
                    pairs = [p for p in pairs if p.patient_id in keep]
                n_train = int(min(int(args.vm_train_max_pairs), len(pairs)))
                inhale_list = []
                exhale_list = []
                mask_list = []
                from .cli import _prepare_inhale_exhale

                for pair in tqdm(pairs[:n_train], desc="Preloading for VoxelMorph training (NLST/LungCT)"):
                    inh, exh, inh_m, _, _ = _prepare_inhale_exhale(pair, args.phase_policy)
                    inhale_list.append(inh)
                    exhale_list.append(exh)
                    mask_list.append(inh_m)
                train_unsupervised(
                    model=vxm_model,
                    inhale_ct_zyx_list=inhale_list,
                    exhale_ct_zyx_list=exhale_list,
                    inhale_mask_zyx_list=mask_list,
                    cfg=vxm_cfg,
                )
            elif dataset == "acdc":
                pairs = load_acdc_pairs(dataset_dir, args.split or "training")
                if args.patient_id:
                    keep = set(args.patient_id)
                    pairs = [p for p in pairs if p.patient_id in keep]
                n_train = int(min(int(args.vm_train_max_pairs), len(pairs)))
                inhale_list = []
                exhale_list = []
                mask_list = []
                mask_labels = tuple(int(x) for x in str(args.vm_mask_labels).split(",") if x.strip())

                from .nifti import read_nifti
                from .demons import resample_to_fixed_grid

                for pair in tqdm(pairs[:n_train], desc="Preloading for VoxelMorph training (ACDC)"):
                    ed = read_nifti(pair.ed_image)
                    es = read_nifti(pair.es_image)
                    ed_seg = read_nifti(pair.ed_seg)
                    fixed_img = ed.data_zyx
                    moving_img = resample_to_fixed_grid(
                        es.data_zyx,
                        es.spacing_zyx,
                        fixed_shape_zyx=fixed_img.shape,
                        fixed_spacing_zyx=ed.spacing_zyx,
                        order=1,
                        cval=float(es.data_zyx.min()),
                    )
                    seg = ed_seg.data_zyx
                    fixed_mask = np.isin(seg, mask_labels).astype("uint8") if mask_labels else None
                    inhale_list.append(fixed_img)
                    exhale_list.append(moving_img)
                    if fixed_mask is not None:
                        mask_list.append(fixed_mask)
                train_unsupervised(
                    model=vxm_model,
                    inhale_ct_zyx_list=inhale_list,
                    exhale_ct_zyx_list=exhale_list,
                    inhale_mask_zyx_list=mask_list if len(mask_list) else None,
                    cfg=vxm_cfg,
                )
            elif is_learn2reg:
                from .nifti import read_nifti

                if learn2reg_train_cases is None or learn2reg_atlas_ids is None:
                    raise RuntimeError("Learn2Reg atlas initialization missing for voxelmorph training")
                atlas_id_set = set(learn2reg_atlas_ids)
                cand = [c for c in learn2reg_train_cases if str(c.patient_id) not in atlas_id_set]
                if args.patient_id:
                    keep = set(args.patient_id)
                    cand = [c for c in cand if c.patient_id in keep]
                if len(cand) < 2:
                    raise ValueError("Not enough Learn2Reg training cases for VoxelMorph training")

                n_train_cases = int(args.vm_train_cases) if args.vm_train_cases is not None else int(min(200, len(cand)))
                n_train_cases = int(max(2, min(n_train_cases, len(cand))))
                rng = np.random.default_rng(int(args.atlas_seed) + 1337)
                pick = rng.choice(np.array([c.patient_id for c in cand], dtype=object), size=n_train_cases, replace=False).tolist()
                learn2reg_vm_train_ids = list(map(str, pick))
                pick_set = set(map(str, pick))
                vm_cases = [c for c in cand if c.patient_id in pick_set]

                imgs = []
                labs = []
                for c in tqdm(vm_cases, desc="Preloading for VoxelMorph training (Learn2Reg)"):
                    ni = read_nifti(c.image)
                    imgs.append(ni.data_zyx.astype(np.float32, copy=False))
                    if c.label is not None and Path(c.label).exists():
                        li = read_nifti(c.label)
                        labs.append(li.data_zyx.astype(np.uint16, copy=False))
                if train_mode in {"supervised", "hybrid"}:
                    if len(labs) != len(imgs):
                        raise ValueError("Supervised/hybrid VM training requires labels for all selected training cases")
                    if train_mode == "supervised":
                        train_supervised_seg(model=vxm_model, images_zyx_list=imgs, labels_zyx_list=labs, cfg=vxm_cfg)
                    else:
                        train_hybrid(model=vxm_model, images_zyx_list=imgs, labels_zyx_list=labs, cfg=vxm_cfg)
                else:
                    # Unsupervised: create paired lists by shuffling.
                    rng2 = np.random.default_rng(int(args.atlas_seed) + 2024)
                    idx = np.arange(len(imgs))
                    rng2.shuffle(idx)
                    # Ensure no identical pairing by a simple rotation if needed.
                    if np.all(idx == np.arange(len(imgs))):
                        idx = np.roll(idx, 1)
                    fixed_list = [imgs[i] for i in range(len(imgs))]
                    moving_list = [imgs[i] for i in idx.tolist()]
                    # IMPORTANT: for inter-patient MR/CT, background dominates MSE/NCC.
                    # Use provided masks if available; otherwise use a cheap nonzero-intensity mask to reduce collapse.
                    mask_list = []
                    all_have_masks = all(getattr(c, "mask", None) is not None and Path(getattr(c, "mask")).exists() for c in vm_cases)
                    if all_have_masks:
                        for c in vm_cases:
                            mi = read_nifti(Path(getattr(c, "mask")))
                            mask_list.append((mi.data_zyx > 0).astype(np.uint8))
                    else:
                        for im in fixed_list:
                            arr = np.asarray(im, dtype=np.float32)
                            # For some MR tasks, background can be small non-zero noise, so (im!=0) becomes all-ones
                            # and provides no useful masking. Use a simple percentile-based foreground mask instead.
                            if dataset in {"hippocampusmr"}:
                                p5 = float(np.percentile(arr, 5.0))
                                p95 = float(np.percentile(arr, 95.0))
                                thr = p5 + 0.10 * (p95 - p5)
                                mask_list.append((arr > thr).astype(np.uint8))
                            else:
                                mask_list.append((arr != 0).astype(np.uint8))
                    train_unsupervised(
                        model=vxm_model,
                        inhale_ct_zyx_list=fixed_list,
                        exhale_ct_zyx_list=moving_list,
                        inhale_mask_zyx_list=mask_list if len(mask_list) else None,
                        cfg=vxm_cfg,
                    )
            else:
                raise ValueError(f"Unhandled dataset for voxelmorph training: {dataset}")

            save_weights(vxm_model, default_weight_path)
            if is_learn2reg and atlas_spec is not None and learn2reg_meta_path is not None and learn2reg_atlas_ids is not None:
                from .atlas import write_atlas_meta

                write_atlas_meta(
                    out_path=learn2reg_meta_path,
                    dataset=dataset,
                    spec=atlas_spec,
                    atlas_ids=learn2reg_atlas_ids,
                    vm_train_ids=learn2reg_vm_train_ids if learn2reg_vm_train_ids else None,
                )

    if dataset in {"nlst", "lungct"}:
        if dataset == "lungct":
            split = args.split or "all"
        else:
            split = args.split or "training"
        pairs = load_pairs(dataset_dir, split)
        if args.patient_id:
            keep = set(args.patient_id)
            pairs = [p for p in pairs if p.patient_id in keep]
        if args.max_pairs and args.max_pairs > 0:
            pairs = pairs[: int(args.max_pairs)]

        rows = []
        for pair in tqdm(pairs, desc=f"Registering NLST ({method})"):
            res = process_pair(
                pair,
                out_dir=results_dir,
                params=params,
                save_nifti_outputs=not args.no_nifti_outputs,
                phase_policy=args.phase_policy,
                backend=method,
                vxm_model=vxm_model,
                vxm_cfg=vxm_cfg,
            )
            rows.append(res.row)
            fig_path = results_dir / "figures" / f"{pair.patient_id}.png"
            save_patient_figure(
                out_path=fig_path,
                inhale_ct_zyx=res.fixed_ct_zyx,
                exhale_ct_warped_zyx=res.moving_ct_warped_zyx,
                inhale_mask_zyx=res.fixed_mask_zyx,
                exhale_mask_warped_zyx=res.moving_mask_warped_zyx,
                disp_mag_mm_zyx=res.disp_mag_mm_zyx,
                slice_z=res.slice_z,
                patient_id=pair.patient_id,
                dice=float(res.row["dice_mask"]),
                vol_inhale_ml=float(res.row["inhale_vol_ml_gt"]),
                vol_exhale_ml=float(res.row["exhale_vol_ml_gt"]),
                pred_exhale_ml=float(res.row["exhale_vol_ml_pred_jac"]),
                delta_gt_exhale_minus_inhale_ml=float(res.row["delta_vol_ml_gt_exhale_minus_inhale"]),
                delta_pred_exhale_minus_inhale_ml=float(res.row["delta_vol_ml_pred_exhale_minus_inhale"]),
                delta_err_ml=float(res.row["delta_vol_ml_error"]),
            )
        pd.DataFrame(rows).sort_values(["patient_id"]).to_csv(results_dir / "summary.csv", index=False)
    elif dataset == "acdc":
        split = args.split or "training"
        pairs = load_acdc_pairs(dataset_dir, split)  # training/testing
        if args.patient_id:
            keep = set(args.patient_id)
            pairs = [p for p in pairs if p.patient_id in keep]
        if args.max_pairs and args.max_pairs > 0:
            pairs = pairs[: int(args.max_pairs)]

        rows = []
        for pair in tqdm(pairs, desc=f"Registering ACDC ({method})"):
            res = process_acdc_pair(
                pair,
                out_dir=results_dir,
                params=params,
                label_lv=3,
                backend=method,
                vxm_model=vxm_model,
                vxm_cfg=vxm_cfg,
                save_nifti_outputs=not args.no_nifti_outputs,
            )
            rows.append(res.row)
            txt = (
                f"Dice(LV): {float(res.row['dice_lv']):.3f}\n"
                f"EDV GT: {float(res.row['edv_ml_gt']):.1f} mL\n"
                f"ESV GT: {float(res.row['esv_ml_gt_native']):.1f} mL\n"
                f"ESV Pred(J): {float(res.row['esv_ml_pred_jac']):.1f} mL\n"
                f"LVEF GT: {float(res.row['lvef_gt']):.3f}\n"
                f"LVEF Pred: {float(res.row['lvef_pred']):.3f}\n"
                f"LVEF Err: {float(res.row['lvef_error']):+.3f}"
            )
            fig_path = results_dir / "figures" / f"{pair.patient_id}.png"
            save_pair_figure(
                out_path=fig_path,
                fixed_img_zyx=res.fixed_img_zyx,
                moving_warped_zyx=res.moving_warped_zyx,
                fixed_mask_zyx=res.fixed_mask_zyx,
                moving_mask_warped_zyx=res.moving_mask_warped_zyx,
                disp_mag_mm_zyx=res.disp_mag_mm_zyx,
                slice_z=res.slice_z,
                title_fixed=f"{pair.patient_id} | ED (fixed)",
                title_moving="ES (warped → ED space)",
                stats_text=txt,
            )
        pd.DataFrame(rows).sort_values(["patient_id"]).to_csv(results_dir / "summary.csv", index=False)
    else:
        # Learn2Reg inter-patient atlas-based segmentation via registration.
        from .learn2reg_pipeline import run_learn2reg_registration

        split = str(args.split or "training").lower()
        if split not in {"training", "test"}:
            raise ValueError("Learn2Reg split must be 'training' or 'test'")

        if atlas_spec is None or learn2reg_train_cases is None or learn2reg_atlas_ids is None or learn2reg_atlas_cases is None or learn2reg_label_ids is None:
            raise RuntimeError("Learn2Reg atlas initialization missing")
        atlas_id_set = set(learn2reg_atlas_ids)

        if split == "training":
            target_cases = [c for c in learn2reg_train_cases if str(c.patient_id) not in atlas_id_set]
        else:
            from .learn2reg_dataset import load_learn2reg_cases

            target_cases = load_learn2reg_cases(dataset_dir, "test")

        if args.patient_id:
            keep = set(args.patient_id)
            target_cases = [c for c in target_cases if str(c.patient_id) in keep]
        if args.max_pairs and args.max_pairs > 0:
            target_cases = target_cases[: int(args.max_pairs)]

        summary_rows, label_rows = run_learn2reg_registration(
            dataset=dataset,
            backend=method,
            cases=target_cases,
            atlas_spec=atlas_spec,
            atlas_cases=learn2reg_atlas_cases,
            atlas_template=learn2reg_atlas_template,
            out_dir=results_dir,
            params=params,
            label_ids=learn2reg_label_ids,
            vxm_model=vxm_model,
            vxm_cfg=vxm_cfg,
            save_nifti_outputs=not args.no_nifti_outputs,
        )

        if split == "training":
            pd.DataFrame(summary_rows).sort_values(["patient_id"]).to_csv(results_dir / "summary.csv", index=False)
            pd.DataFrame(label_rows).sort_values(["patient_id", "label_id"]).to_csv(results_dir / "label_volumes.csv", index=False)
        else:
            pd.DataFrame(summary_rows).sort_values(["patient_id"]).to_csv(results_dir / "summary_unlabeled.csv", index=False)
