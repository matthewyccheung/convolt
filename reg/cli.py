from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import load_pairs
from .demons import DemonsParams
from .demons import resample_to_fixed_grid
from .metrics import mask_volume_ml
from .nifti import read_nifti
from .pipeline import process_pair
from .viz import save_patient_figure


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsupervised NLST inhale/exhale registration + lung volume change.")
    p.add_argument("--dataset_dir", required=True, type=Path, help="Path to NLST dataset root (contains NLST_dataset.json).")
    p.add_argument("--split", default="training", choices=["training", "val", "test"], help="Which paired list to use.")
    p.add_argument("--out_dir", required=True, type=Path, help="Output directory (will be created).")
    p.add_argument("--max_pairs", type=int, default=0, help="If >0, limit number of pairs processed.")
    p.add_argument("--patient_id", action="append", default=[], help="Process only these patient IDs (repeatable).")
    p.add_argument(
        "--phase_policy",
        default="mask_volume",
        choices=["mask_volume", "suffix", "json"],
        help="How to decide which scan is inhale vs exhale.",
    )

    p.add_argument("--backend", choices=["demons", "voxelmorph"], default="demons", help="Registration backend.")

    p.add_argument("--iters", type=str, default="80,40,20", help="Demons iters per level (comma-separated).")
    p.add_argument("--levels", type=str, default="4,2,1", help="Demons shrink levels (comma-separated).")
    p.add_argument("--step", type=float, default=1.0, help="Update step size.")
    p.add_argument("--diffusion_sigma", type=float, default=1.5, help="Gaussian smoothing sigma for disp (vox).")
    p.add_argument("--mask_dilate", type=int, default=3, help="Dilate fixed mask for ROI weighting.")
    p.add_argument("--no_nifti_outputs", action="store_true", help="Do not write NIfTI outputs (only .npz + figures + CSV).")

    # VoxelMorph (GPU) backend options.
    p.add_argument("--vm_device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--vm_scale", type=int, default=1, help="Downsample factor for VoxelMorph inference/training (>=1).")
    p.add_argument("--vm_weights", type=Path, default=None, help="Path to VoxelMorph weights (.pt).")
    p.add_argument("--vm_train_epochs", type=int, default=0, help="If >0 and no --vm_weights, train VoxelMorph unsupervised first.")
    p.add_argument("--vm_steps_per_epoch", type=int, default=100)
    p.add_argument("--vm_lr", type=float, default=1e-4)
    p.add_argument("--vm_smooth", type=float, default=0.05, help="Flow smoothness weight.")
    p.add_argument("--vm_train_max_pairs", type=int, default=20, help="Max pairs to preload for VoxelMorph training (memory bound).")
    p.add_argument("--vm_norm", choices=["ct_hu", "percentile", "minmax"], default="ct_hu", help="Normalization for VoxelMorph network input.")
    p.add_argument("--vm_ct_clip", type=str, default="-1000,400", help="CT clip lo,hi for --vm_norm ct_hu.")
    p.add_argument("--vm_pct", type=str, default="1,99", help="Percentiles lo,hi for --vm_norm percentile.")
    p.add_argument("--vm_sim", choices=["mse", "ncc"], default="mse", help="Similarity loss for VoxelMorph training.")
    p.add_argument("--vm_ncc_win", type=int, default=9, help="Local NCC window size (odd int).")
    return p.parse_args(argv)


def _prepare_inhale_exhale(pair, phase_policy: str):
    a_img = read_nifti(pair.fixed_image)
    b_img = read_nifti(pair.moving_image)
    a_mask_img = read_nifti(pair.fixed_mask)
    b_mask_img = read_nifti(pair.moving_mask)

    a_mask = (a_mask_img.data_zyx > 0).astype("uint8")
    b_mask = (b_mask_img.data_zyx > 0).astype("uint8")

    inhale_img, exhale_img = a_img, b_img
    inhale_mask_img, exhale_mask_img = a_mask_img, b_mask_img
    inhale_mask, exhale_mask = a_mask, b_mask

    swapped = False
    policy = str(phase_policy).lower()
    if policy == "suffix":
        a_is_0000 = str(pair.fixed_image).endswith("_0000.nii.gz")
        b_is_0000 = str(pair.moving_image).endswith("_0000.nii.gz")
        if b_is_0000 and not a_is_0000:
            swapped = True
    elif policy == "mask_volume":
        va = mask_volume_ml(a_mask, a_mask_img.spacing_zyx)
        vb = mask_volume_ml(b_mask, b_mask_img.spacing_zyx)
        if vb > va:
            swapped = True

    if swapped:
        inhale_img, exhale_img = b_img, a_img
        inhale_mask_img, exhale_mask_img = b_mask_img, a_mask_img
        inhale_mask, exhale_mask = b_mask, a_mask

    inhale_ct = inhale_img.data_zyx
    inhale_mask = inhale_mask.astype("uint8", copy=False)
    spacing_zyx = inhale_img.spacing_zyx

    exhale_ct = resample_to_fixed_grid(
        exhale_img.data_zyx,
        exhale_img.spacing_zyx,
        fixed_shape_zyx=inhale_ct.shape,
        fixed_spacing_zyx=spacing_zyx,
        order=1,
        cval=float(np.min(exhale_img.data_zyx)),
    )
    exhale_mask_rs = resample_to_fixed_grid(
        exhale_mask.astype("uint8", copy=False),
        exhale_mask_img.spacing_zyx,
        fixed_shape_zyx=inhale_ct.shape,
        fixed_spacing_zyx=spacing_zyx,
        order=0,
        cval=0.0,
    ).astype("uint8")
    return inhale_ct, exhale_ct, inhale_mask, exhale_mask_rs, spacing_zyx


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    iters = tuple(int(x) for x in args.iters.split(",") if x.strip())
    params = DemonsParams(
        levels=levels,
        iterations=iters,
        update_step=float(args.step),
        diffusion_sigma=float(args.diffusion_sigma),
        mask_dilate_radius=int(args.mask_dilate),
    )

    pairs = load_pairs(args.dataset_dir, args.split)
    if args.patient_id:
        keep = set(args.patient_id)
        pairs = [p for p in pairs if p.patient_id in keep]
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: int(args.max_pairs)]

    vxm_model = None
    vxm_cfg = None
    if args.backend == "voxelmorph":
        from .vxm import VoxelMorphDense, VxmConfig, load_weights, save_weights, train_unsupervised

        ct_lo, ct_hi = (float(x) for x in str(args.vm_ct_clip).split(","))
        p_lo, p_hi = (float(x) for x in str(args.vm_pct).split(","))
        vxm_cfg = VxmConfig(
            device=args.vm_device,
            scale=int(args.vm_scale),
            smooth_weight=float(args.vm_smooth),
            lr=float(args.vm_lr),
            epochs=int(args.vm_train_epochs),
            steps_per_epoch=int(args.vm_steps_per_epoch),
            weights_path=str(args.vm_weights) if args.vm_weights else None,
            norm=str(args.vm_norm),
            ct_clip=(ct_lo, ct_hi),
            pct=(p_lo, p_hi),
            sim=str(args.vm_sim),
            ncc_win=int(args.vm_ncc_win),
        )
        vxm_model = VoxelMorphDense()

        if args.vm_weights and args.vm_weights.exists():
            load_weights(vxm_model, args.vm_weights, device=args.vm_device)
        else:
            if int(args.vm_train_epochs) <= 0:
                raise ValueError("voxelmorph backend requires --vm_weights or --vm_train_epochs > 0")

            n_train = int(min(int(args.vm_train_max_pairs), len(pairs)))
            if n_train <= 0:
                raise ValueError("--vm_train_max_pairs must be >0 for training")

            inhale_list = []
            exhale_list = []
            mask_list = []
            for pair in tqdm(pairs[:n_train], desc="Preloading for VoxelMorph training"):
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

            weights_out = args.out_dir / "voxelmorph.pt"
            save_weights(vxm_model, weights_out)

    rows = []
    for pair in tqdm(pairs, desc="Registering pairs"):
        res = process_pair(
            pair,
            out_dir=args.out_dir,
            params=params,
            save_nifti_outputs=not args.no_nifti_outputs,
            phase_policy=args.phase_policy,
            backend=args.backend,
            vxm_model=vxm_model,
            vxm_cfg=vxm_cfg,
        )
        rows.append(res.row)

        fig_path = args.out_dir / "figures" / f"{pair.patient_id}.png"
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

    df = pd.DataFrame(rows).sort_values(["patient_id"])
    df.to_csv(args.out_dir / "summary.csv", index=False)
