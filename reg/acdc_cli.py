from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .acdc_dataset import load_acdc_pairs
from .acdc_pipeline import process_acdc_pair
from .demons import DemonsParams
from .viz import save_pair_figure


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACDC ED/ES registration + LV volume/Ejection Fraction evaluation.")
    p.add_argument("--dataset_dir", required=True, type=Path, help="Path to ACDC database root (contains training/ testing).")
    p.add_argument("--split", default="training", choices=["training", "testing"])
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--max_pairs", type=int, default=0)
    p.add_argument("--patient_id", action="append", default=[], help="Process only these patient IDs (repeatable).")

    p.add_argument(
        "--backend",
        choices=["demons", "voxelmorph", "itk_elastix_bspline", "sitk_diffeomorphic_demons", "sitk_bspline"],
        default="demons",
    )
    p.add_argument("--label_lv", type=int, default=3, help="Label id for LV cavity (default 3).")

    p.add_argument("--iters", type=str, default="80,40,20")
    p.add_argument("--levels", type=str, default="4,2,1")
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--diffusion_sigma", type=float, default=1.5)
    p.add_argument("--mask_dilate", type=int, default=3)
    p.add_argument("--no_nifti_outputs", action="store_true")

    # VoxelMorph options (same as reg/cli.py).
    p.add_argument("--vm_device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--vm_scale", type=int, default=1)
    p.add_argument("--vm_weights", type=Path, default=None)
    p.add_argument("--vm_train_epochs", type=int, default=0)
    p.add_argument("--vm_steps_per_epoch", type=int, default=100)
    p.add_argument("--vm_lr", type=float, default=1e-4)
    p.add_argument("--vm_smooth", type=float, default=0.005, help="Flow smoothness weight (ACDC often needs smaller than CT).")
    p.add_argument("--vm_train_max_pairs", type=int, default=20)
    p.add_argument("--vm_norm", choices=["ct_hu", "percentile", "minmax"], default="percentile", help="Normalization for VoxelMorph network input.")
    p.add_argument("--vm_ct_clip", type=str, default="-1000,400", help="CT clip lo,hi for --vm_norm ct_hu.")
    p.add_argument("--vm_pct", type=str, default="1,99", help="Percentiles lo,hi for --vm_norm percentile.")
    p.add_argument("--vm_sim", choices=["mse", "ncc"], default="ncc", help="Similarity loss for VoxelMorph training.")
    p.add_argument("--vm_ncc_win", type=int, default=9, help="Local NCC window size (odd int).")
    p.add_argument(
        "--vm_mask_labels",
        type=str,
        default="1,2,3",
        help="Comma-separated labels to include in the similarity-loss mask during VoxelMorph training (default: 1,2,3 = whole heart).",
    )
    return p.parse_args(argv)


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

    pairs = load_acdc_pairs(args.dataset_dir, args.split)
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
            mask_labels = tuple(int(x) for x in str(args.vm_mask_labels).split(",") if x.strip())
            for pair in tqdm(pairs[:n_train], desc="Preloading for VoxelMorph training"):
                # ED is fixed, ES is moving.
                from .nifti import read_nifti
                from .demons import resample_to_fixed_grid
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
                if len(mask_labels) == 0:
                    fixed_mask = None
                else:
                    seg = ed_seg.data_zyx
                    fixed_mask = np.isin(seg, mask_labels).astype("uint8")

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
            save_weights(vxm_model, args.out_dir / "voxelmorph.pt")

    rows = []
    for pair in tqdm(pairs, desc="Registering ACDC pairs"):
        res = process_acdc_pair(
            pair,
            out_dir=args.out_dir,
            params=params,
            label_lv=int(args.label_lv),
            backend=args.backend,
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

        fig_path = args.out_dir / "figures" / f"{pair.patient_id}.png"
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

    df = pd.DataFrame(rows).sort_values(["patient_id"])
    df.to_csv(args.out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
