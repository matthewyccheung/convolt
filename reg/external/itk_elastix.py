from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..demons import jacobian_determinant_from_disp_mm


def _require_itk_elastix() -> "object":
    try:
        import itk  # type: ignore

        # Make sure elastix bindings are present.
        _ = itk.ElastixRegistrationMethod  # noqa: F841
        _ = itk.TransformixFilter  # noqa: F841
        return itk
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "itk-elastix is required for this backend. Install it with `pip install itk-elastix` "
            "and rerun with --method itk_elastix_bspline."
        ) from e


def _itk_image_from_zyx(arr_zyx: np.ndarray, spacing_zyx: Tuple[float, float, float]) -> "object":
    itk = _require_itk_elastix()
    img = itk.image_from_array(np.asarray(arr_zyx))
    dz, dy, dx = map(float, spacing_zyx)
    img.SetSpacing((dx, dy, dz))
    return img


def _zyx_from_itk_image(img: "object", dtype=np.float32) -> np.ndarray:
    itk = _require_itk_elastix()
    return np.asarray(itk.array_from_image(img), dtype=dtype)


def _as_uint8_mask_zyx(mask_zyx: np.ndarray) -> np.ndarray:
    return (np.asarray(mask_zyx) > 0).astype(np.uint8, copy=False)


def _parameter_object_affine_bspline() -> "object":
    itk = _require_itk_elastix()
    po = itk.ParameterObject.New()
    # Default maps are reasonable starting points.
    po.AddParameterMap(po.GetDefaultParameterMap("affine"))
    po.AddParameterMap(po.GetDefaultParameterMap("bspline"))
    return po


def _parameter_object_affine_bspline_with_metric(metric: str | None) -> "object":
    """
    Build a ParameterObject and optionally override the Metric for each stage.
    """
    itk = _require_itk_elastix()
    po = _parameter_object_affine_bspline()
    if metric is None:
        return po
    metric = str(metric)
    n_maps = int(po.GetNumberOfParameterMaps())
    out = itk.ParameterObject.New()
    for i in range(n_maps):
        pm = dict(po.GetParameterMap(i))
        pm["Metric"] = [metric]
        out.AddParameterMap(pm)
    return out


def _parameter_object_nearest_neighbor(transform_parameter_object: "object") -> "object":
    """
    Copy a TransformParameterObject and force nearest-neighbor interpolation for label warps.
    """
    itk = _require_itk_elastix()
    tpo = itk.ParameterObject.New()
    n_maps = int(transform_parameter_object.GetNumberOfParameterMaps())
    for i in range(n_maps):
        pm = dict(transform_parameter_object.GetParameterMap(i))
        # ParameterMap values are lists of strings in elastix.
        pm["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        pm["FinalBSplineInterpolationOrder"] = ["0"]
        tpo.AddParameterMap(pm)
    return tpo


def _enable_logging(obj: "object", work_dir: Path, prefix: str) -> None:
    """
    Best-effort enabling of elastix/transformix logs to a file under work_dir.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    # ElastixRegistrationMethod exposes LogToConsoleOn/Off, LogToFileOn/Off on many builds.
    if hasattr(obj, "LogToConsoleOff"):
        try:
            obj.LogToConsoleOff()
        except Exception:
            pass
    if hasattr(obj, "LogToFileOn"):
        try:
            obj.LogToFileOn()
        except Exception:
            pass
    # Output directory / logfile name vary across versions.
    for name in ("SetOutputDirectory", "SetOutputFolder"):
        if hasattr(obj, name):
            try:
                getattr(obj, name)(str(work_dir))
            except Exception:
                pass
    for name in ("SetLogFileName", "SetLogFile", "SetLogFilePath"):
        if hasattr(obj, name):
            try:
                getattr(obj, name)(str(work_dir / f"{prefix}.log"))
            except Exception:
                pass


def _transformix_warp(
    *,
    moving_img_itk: "object",
    transform_parameter_object: "object",
) -> "object":
    itk = _require_itk_elastix()
    tx = itk.TransformixFilter.New(moving_img_itk)
    tx.SetTransformParameterObject(transform_parameter_object)
    tx.LogToConsoleOff()
    tx.UpdateLargestPossibleRegion()
    return tx.GetOutput()


def _transformix_displacement_field(
    *,
    fixed_like_itk: "object",
    transform_parameter_object: "object",
) -> np.ndarray:
    """
    Ask Transformix to compute a displacement field by transforming an identity field.

    Different itk-elastix versions expose different APIs; we try the common ones.
    Returns disp_mm_3zyx (float32) in physical units (mm).
    """
    itk = _require_itk_elastix()

    # Create a dummy scalar image to drive transformix, but request deformation field output.
    dummy = fixed_like_itk
    tx = itk.TransformixFilter.New(dummy)
    tx.SetTransformParameterObject(transform_parameter_object)
    tx.LogToConsoleOff()

    # Try to enable deformation field computation.
    enabled = False
    for name in ("ComputeDeformationFieldOn", "SetComputeDeformationField"):
        if hasattr(tx, name):
            try:
                getattr(tx, name)(True) if name.startswith("Set") else getattr(tx, name)()
                enabled = True
            except Exception:
                pass
    if not enabled:
        raise RuntimeError("TransformixFilter does not support deformation field output in this itk-elastix build.")

    tx.UpdateLargestPossibleRegion()

    # Try to fetch the deformation field output.
    df_img = None
    for name in ("GetDeformationField", "GetOutputDeformationField"):
        if hasattr(tx, name):
            try:
                df_img = getattr(tx, name)()
                break
            except Exception:
                pass
    if df_img is None:
        # Some builds provide the deformation field as an additional output.
        try:
            df_img = tx.GetOutputDeformationField()
        except Exception as e:
            raise RuntimeError("Could not retrieve deformation field from TransformixFilter.") from e

    a = np.asarray(itk.array_from_image(df_img), dtype=np.float32)
    # Common shape: (Z,Y,X,3) with components (x,y,z).
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"Unexpected deformation field array shape from transformix: {a.shape}")
    disp_mm_3zyx = np.moveaxis(a[..., [2, 1, 0]], -1, 0).astype(np.float32, copy=False)
    return disp_mm_3zyx


def register_affine_bspline(
    *,
    fixed_zyx: np.ndarray,
    moving_zyx: np.ndarray,
    fixed_mask_zyx: np.ndarray,
    moving_mask_zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    work_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    itk-elastix affine + B-spline FFD.

    Returns:
      moving_warped_zyx, moving_mask_warped_zyx, disp_mm_3zyx, jac_det_zyx
    """
    itk = _require_itk_elastix()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fixed = _itk_image_from_zyx(np.asarray(fixed_zyx, dtype=np.float32), spacing_zyx)
    moving = _itk_image_from_zyx(np.asarray(moving_zyx, dtype=np.float32), spacing_zyx)
    fixed_mask = _itk_image_from_zyx(_as_uint8_mask_zyx(fixed_mask_zyx), spacing_zyx)
    # moving_mask_zyx can be binary masks or integer label maps; use a binarized mask for registration,
    # but warp the original label map with NN interpolation for downstream segmentation.
    moving_mask_bin = _itk_image_from_zyx(_as_uint8_mask_zyx(moving_mask_zyx), spacing_zyx)
    moving_label_arr = np.asarray(moving_mask_zyx)
    if moving_label_arr.dtype.kind in {"f"}:
        moving_label_arr = np.rint(moving_label_arr).astype(np.int32)
    moving_label = _itk_image_from_zyx(moving_label_arr.astype(np.uint16, copy=False), spacing_zyx)

    # If masks are empty, don't pass them to elastix (it can error internally).
    n_fixed = int(np.count_nonzero(_as_uint8_mask_zyx(fixed_mask_zyx)))
    n_moving = int(np.count_nonzero(_as_uint8_mask_zyx(moving_mask_zyx)))

    def _try_run(*, use_fixed_mask: bool, use_moving_mask: bool, metric: str | None) -> tuple["object", "object"]:
        po = _parameter_object_affine_bspline_with_metric(metric)
        elx = itk.ElastixRegistrationMethod.New(fixed, moving)
        elx.SetParameterObject(po)
        _enable_logging(elx, work_dir, "elastix")
        if use_fixed_mask and n_fixed > 0 and hasattr(elx, "SetFixedMask"):
            elx.SetFixedMask(fixed_mask)
        if use_moving_mask and n_moving > 0 and hasattr(elx, "SetMovingMask"):
            elx.SetMovingMask(moving_mask_bin)
        elx.UpdateLargestPossibleRegion()
        return elx.GetOutput(), elx.GetTransformParameterObject()

    try:
        warped, tpo = _try_run(use_fixed_mask=True, use_moving_mask=True, metric=None)
    except Exception:
        # Retry in increasingly permissive modes. This makes the backend far more robust,
        # especially for small/empty masks (e.g., thin structures in ACDC).
        try:
            warped, tpo = _try_run(use_fixed_mask=True, use_moving_mask=False, metric=None)
        except Exception:
            try:
                warped, tpo = _try_run(use_fixed_mask=False, use_moving_mask=False, metric=None)
            except Exception:
                # Final attempt: override metric to a simpler one.
                warped, tpo = _try_run(use_fixed_mask=True, use_moving_mask=False, metric="AdvancedMeanSquares")

    warped_mask = _transformix_warp(moving_img_itk=moving_label, transform_parameter_object=_parameter_object_nearest_neighbor(tpo))
    disp_mm_3zyx = _transformix_displacement_field(fixed_like_itk=fixed, transform_parameter_object=tpo)
    jac = jacobian_determinant_from_disp_mm(disp_mm_3zyx, spacing_zyx).astype(np.float32, copy=False)

    return (
        _zyx_from_itk_image(warped, dtype=np.float32),
        _zyx_from_itk_image(warped_mask, dtype=np.uint16),
        disp_mm_3zyx,
        jac,
    )
