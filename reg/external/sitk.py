from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..demons import jacobian_determinant_from_disp_mm


def _require_simpleitk() -> "object":
    try:
        import SimpleITK as sitk  # type: ignore

        return sitk
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SimpleITK is required for this backend. Install it with `pip install SimpleITK` "
            "and rerun with --method sitk_diffeomorphic_demons or sitk_bspline."
        ) from e


def _img_from_zyx(arr_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float], *, is_vector: bool = False) -> "object":
    sitk = _require_simpleitk()
    img = sitk.GetImageFromArray(arr_zyx, isVector=is_vector)
    dz, dy, dx = map(float, spacing_zyx)
    # Guard against invalid spacings (ITK refuses zero/negative spacing).
    def _pos(v: float) -> float:
        v = float(v)
        if not np.isfinite(v) or v <= 0:
            return 1.0
        return v

    dz, dy, dx = _pos(dz), _pos(dy), _pos(dx)
    dim = int(img.GetDimension())
    if dim == 3:
        img.SetSpacing((dx, dy, dz))
    elif dim == 2:
        img.SetSpacing((dx, dy))
    else:
        # Fallback: best-effort set spacing prefix.
        sp = (dx, dy, dz)[:dim]
        img.SetSpacing(tuple(sp))
    return img


def _disp_mm_3zyx_from_sitk(df_img: "object") -> np.ndarray:
    """
    SimpleITK displacement field -> disp_mm_3zyx with (Z,Y,X) component order.

    sitk.GetArrayFromImage(df) yields shape (Z,Y,X,3) with components (x,y,z) in physical units (mm).
    """
    a = np.asarray(_require_simpleitk().GetArrayFromImage(df_img), dtype=np.float32)
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"Unexpected SimpleITK displacement field array shape: {a.shape}")
    # (Z,Y,X,3[x,y,z]) -> (3[z,y,x],Z,Y,X)
    return np.moveaxis(a[..., [2, 1, 0]], -1, 0).astype(np.float32, copy=False)


def register_diffeomorphic_demons(
    *,
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    fixed_mask_zyx: np.ndarray,
    moving_mask_zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    work_dir: str | Path,
    iterations: int = 80,
    smoothing_std_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SimpleITK diffeomorphic demons registration.

    Returns:
      moving_warped_zyx, moving_mask_warped_zyx, disp_mm_3zyx, jac_det_zyx
    """
    sitk = _require_simpleitk()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fixed = _img_from_zyx(np.asarray(fixed_zyx, dtype=np.float32), spacing_zyx)
    moving = _img_from_zyx(np.asarray(moving_zyx, dtype=np.float32), spacing_zyx)
    fixed_mask = _img_from_zyx((np.asarray(fixed_mask_zyx) > 0).astype(np.uint8), spacing_zyx)
    # moving_mask_zyx can be either binary masks or integer label maps to warp with NN.
    moving_mask_arr = np.asarray(moving_mask_zyx)
    if moving_mask_arr.dtype.kind in {"f"}:
        moving_mask_arr = np.rint(moving_mask_arr).astype(np.int32)
    moving_mask = _img_from_zyx(moving_mask_arr.astype(np.uint16, copy=False), spacing_zyx)

    # Restrict to fixed mask to stabilize.
    fixed_m = sitk.Mask(fixed, fixed_mask)
    moving_m = sitk.Mask(moving, fixed_mask)

    # Pre-smooth (mm units).
    if float(smoothing_std_mm) > 0:
        fixed_m = sitk.SmoothingRecursiveGaussian(fixed_m, float(smoothing_std_mm))
        moving_m = sitk.SmoothingRecursiveGaussian(moving_m, float(smoothing_std_mm))

    f = sitk.DiffeomorphicDemonsRegistrationFilter()
    f.SetNumberOfIterations(int(iterations))
    f.SetSmoothDisplacementField(True)
    f.SetStandardDeviations(float(smoothing_std_mm))

    df = f.Execute(fixed_m, moving_m)  # displacement field image (vector, physical units)
    disp_mm_3zyx = _disp_mm_3zyx_from_sitk(df)

    # Warp moving image/mask to fixed grid.
    # Prefer Resample + DisplacementFieldTransform (stable across SimpleITK versions).
    tx = sitk.DisplacementFieldTransform(df)
    warped = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, float(np.min(moving_zyx)), sitk.sitkFloat32)
    warped_mask = sitk.Resample(moving_mask, fixed, tx, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt16)

    moving_warped_zyx = np.asarray(sitk.GetArrayFromImage(warped), dtype=np.float32)
    moving_mask_warped_zyx = np.asarray(sitk.GetArrayFromImage(warped_mask), dtype=np.uint16)

    jac = jacobian_determinant_from_disp_mm(disp_mm_3zyx, spacing_zyx).astype(np.float32, copy=False)
    return moving_warped_zyx, moving_mask_warped_zyx, disp_mm_3zyx, jac


def register_bspline_ffd(
    *,
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    fixed_mask_zyx: np.ndarray,
    moving_mask_zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    work_dir: str | Path,
    mesh_size: Tuple[int, int, int] = (8, 8, 8),
    mi_bins: int = 50,
    sampling_pct: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SimpleITK B-spline free-form deformation registration (single-stage).

    Returns:
      moving_warped_zyx, moving_mask_warped_zyx, disp_mm_3zyx, jac_det_zyx
    """
    sitk = _require_simpleitk()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fixed = _img_from_zyx(np.asarray(fixed_zyx, dtype=np.float32), spacing_zyx)
    moving = _img_from_zyx(np.asarray(moving_zyx, dtype=np.float32), spacing_zyx)
    fixed_mask = _img_from_zyx((np.asarray(fixed_mask_zyx) > 0).astype(np.uint8), spacing_zyx)
    moving_mask_arr = np.asarray(moving_mask_zyx)
    if moving_mask_arr.dtype.kind in {"f"}:
        moving_mask_arr = np.rint(moving_mask_arr).astype(np.int32)
    moving_mask = _img_from_zyx(moving_mask_arr.astype(np.uint16, copy=False), spacing_zyx)

    # Ensure consistent physical metadata. (Some SimpleITK metrics are sensitive to origin/direction.)
    try:
        # Clamp any invalid spacings.
        sp = tuple(float(x) for x in fixed.GetSpacing())
        sp = tuple((1.0 if (not np.isfinite(v) or v <= 0) else v) for v in sp)
        fixed.SetSpacing(sp)
        moving.SetOrigin(fixed.GetOrigin())
        moving.SetDirection(fixed.GetDirection())
        moving.SetSpacing(sp)
        fixed_mask.SetOrigin(fixed.GetOrigin())
        fixed_mask.SetDirection(fixed.GetDirection())
        fixed_mask.SetSpacing(sp)
        moving_mask.SetOrigin(fixed.GetOrigin())
        moving_mask.SetDirection(fixed.GetDirection())
        moving_mask.SetSpacing(sp)
    except Exception:
        pass

    # Initialize transform.
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize=list(map(int, mesh_size)))
    # Defensive: force identity parameters (some builds can leave BSpline params non-zero).
    try:
        tx.SetParameters([0.0] * int(tx.GetNumberOfParameters()))
    except Exception:
        pass

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(mi_bins))
    R.SetMetricFixedMask(fixed_mask)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(float(np.clip(float(sampling_pct), 0.01, 1.0)))
    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=200,
        costFunctionConvergenceFactor=1e7,
    )

    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Center initialization to avoid "all samples map outside moving image buffer" failures.
    try:
        init = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.TranslationTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        if hasattr(R, "SetMovingInitialTransform"):
            R.SetMovingInitialTransform(init)
        else:
            # Fall back: pre-align moving images by the init transform.
            moving = sitk.Resample(moving, fixed, init, sitk.sitkLinear, float(np.min(moving_zyx)), sitk.sitkFloat32)
            moving_mask = sitk.Resample(moving_mask, fixed, init, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)
    except Exception:
        pass

    R.SetInitialTransform(tx, inPlace=False)
    try:
        final_tx = R.Execute(fixed, moving)
    except RuntimeError as e:
        msg = str(e)
        # Retry with a simpler metric if MI can't be evaluated due to overlap issues.
        if "All samples map outside moving image buffer" in msg:
            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMeanSquares()
            R.SetMetricFixedMask(fixed_mask)
            R.SetMetricSamplingStrategy(R.RANDOM)
            R.SetMetricSamplingPercentage(float(np.clip(float(sampling_pct), 0.01, 1.0)))
            R.SetInterpolator(sitk.sitkLinear)
            R.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5,
                numberOfIterations=100,
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=200,
                costFunctionConvergenceFactor=1e7,
            )
            R.SetShrinkFactorsPerLevel([4, 2, 1])
            R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
            R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            try:
                if hasattr(R, "SetMovingInitialTransform") and "init" in locals():
                    R.SetMovingInitialTransform(init)
            except Exception:
                pass
            R.SetInitialTransform(tx, inPlace=False)
            final_tx = R.Execute(fixed, moving)
        else:
            raise

    # Convert transform to a displacement field on the fixed grid.
    df = sitk.TransformToDisplacementField(
        final_tx,
        sitk.sitkVectorFloat32,
        fixed.GetSize(),
        fixed.GetOrigin(),
        fixed.GetSpacing(),
        fixed.GetDirection(),
    )
    disp_mm_3zyx = _disp_mm_3zyx_from_sitk(df)

    warped = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, float(np.min(moving_zyx)), sitk.sitkFloat32)
    warped_mask = sitk.Resample(moving_mask, fixed, final_tx, sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt16)
    moving_warped_zyx = np.asarray(sitk.GetArrayFromImage(warped), dtype=np.float32)
    moving_mask_warped_zyx = np.asarray(sitk.GetArrayFromImage(warped_mask), dtype=np.uint16)

    jac = jacobian_determinant_from_disp_mm(disp_mm_3zyx, spacing_zyx).astype(np.float32, copy=False)
    return moving_warped_zyx, moving_mask_warped_zyx, disp_mm_3zyx, jac
