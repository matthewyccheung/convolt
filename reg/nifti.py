from __future__ import annotations

import gzip
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class NiftiImage:
    data_zyx: np.ndarray
    spacing_zyx: Tuple[float, float, float]
    affine_xyz: np.ndarray | None
    header: Dict[str, Any]


_DTYPE_BY_CODE: dict[int, np.dtype] = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
    1024: np.int64,
}


def _open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _unpack(fmt: str, buf: bytes, offset: int, endian: str):
    return struct.unpack_from(endian + fmt, buf, offset)


def _affine_from_qform(
    pixdim_xyz: Tuple[float, float, float],
    quatern_bcd: Tuple[float, float, float],
    qoffset_xyz: Tuple[float, float, float],
    qfac: float,
) -> np.ndarray:
    b, c, d = quatern_bcd
    a_sq = 1.0 - (b * b + c * c + d * d)
    a = math.sqrt(max(a_sq, 0.0))

    # Rotation (from NIfTI-1 spec).
    r11 = a * a + b * b - c * c - d * d
    r12 = 2 * (b * c - a * d)
    r13 = 2 * (b * d + a * c)
    r21 = 2 * (b * c + a * d)
    r22 = a * a + c * c - b * b - d * d
    r23 = 2 * (c * d - a * b)
    r31 = 2 * (b * d - a * c)
    r32 = 2 * (c * d + a * b)
    r33 = a * a + d * d - c * c - b * b

    dx, dy, dz = pixdim_xyz
    dz = dz * (1.0 if qfac >= 0 else -1.0)

    aff = np.eye(4, dtype=np.float64)
    aff[:3, :3] = np.array(
        [
            [r11 * dx, r12 * dy, r13 * dz],
            [r21 * dx, r22 * dy, r23 * dz],
            [r31 * dx, r32 * dy, r33 * dz],
        ],
        dtype=np.float64,
    )
    aff[:3, 3] = np.array(qoffset_xyz, dtype=np.float64)
    return aff


def read_nifti(path: str | Path) -> NiftiImage:
    """
    Minimal NIfTI-1 reader for .nii/.nii.gz that returns data in Z,Y,X order.

    This is intentionally small (no external deps). It supports common scalar dtypes
    and reads qform/sform affine when present.
    """
    path = Path(path)
    with _open_maybe_gzip(path) as f:
        hdr = f.read(352)  # 348-byte header + 4-byte extension flag
        if len(hdr) < 348:
            raise ValueError(f"{path} is too small to be a NIfTI file")

        sizeof_hdr_le = struct.unpack_from("<i", hdr, 0)[0]
        if sizeof_hdr_le == 348:
            endian = "<"
        else:
            sizeof_hdr_be = struct.unpack_from(">i", hdr, 0)[0]
            if sizeof_hdr_be != 348:
                raise ValueError(f"{path} does not look like a NIfTI-1 file (bad sizeof_hdr)")
            endian = ">"

        dim = _unpack("8h", hdr, 40, endian)
        ndim = int(dim[0])
        if ndim < 3:
            raise ValueError(f"{path} has ndim={ndim}; expected at least 3")
        dims = tuple(int(d) for d in dim[1 : 1 + ndim])
        if len(dims) < 3:
            raise ValueError(f"{path} has dims={dims}; expected at least 3 spatial dims")
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

        datatype = int(_unpack("h", hdr, 70, endian)[0])
        bitpix = int(_unpack("h", hdr, 72, endian)[0])
        dtype = _DTYPE_BY_CODE.get(datatype)
        if dtype is None:
            raise ValueError(f"{path} has unsupported datatype code {datatype}")
        if np.dtype(dtype).itemsize * 8 != bitpix:
            # Not necessarily fatal, but likely indicates something we can't parse safely.
            raise ValueError(f"{path} bitpix mismatch: datatype={datatype} implies {np.dtype(dtype).itemsize*8} but bitpix={bitpix}")

        pixdim = _unpack("8f", hdr, 76, endian)
        dx, dy, dz = float(pixdim[1]), float(pixdim[2]), float(pixdim[3])
        qfac = float(pixdim[0]) if pixdim[0] != 0 else 1.0

        # Some files have invalid/zero pixdims; clamp to positive values to avoid downstream ITK/SimpleITK errors.
        def _sanitize_spacing(v: float) -> float:
            v = float(v)
            if not np.isfinite(v) or v <= 0:
                return 1.0
            return v

        dx, dy, dz = _sanitize_spacing(dx), _sanitize_spacing(dy), _sanitize_spacing(dz)

        vox_offset = float(_unpack("f", hdr, 108, endian)[0])
        scl_slope = float(_unpack("f", hdr, 112, endian)[0])
        scl_inter = float(_unpack("f", hdr, 116, endian)[0])

        qform_code = int(_unpack("h", hdr, 252, endian)[0])
        sform_code = int(_unpack("h", hdr, 254, endian)[0])
        quatern_b = float(_unpack("f", hdr, 256, endian)[0])
        quatern_c = float(_unpack("f", hdr, 260, endian)[0])
        quatern_d = float(_unpack("f", hdr, 264, endian)[0])
        qoffset_x = float(_unpack("f", hdr, 268, endian)[0])
        qoffset_y = float(_unpack("f", hdr, 272, endian)[0])
        qoffset_z = float(_unpack("f", hdr, 276, endian)[0])

        srow_x = _unpack("4f", hdr, 280, endian)
        srow_y = _unpack("4f", hdr, 296, endian)
        srow_z = _unpack("4f", hdr, 312, endian)

        # Jump to voxel data and read.
        f.seek(int(vox_offset))
        nvox = int(np.prod(dims))
        raw = f.read(nvox * np.dtype(dtype).itemsize)
        if len(raw) != nvox * np.dtype(dtype).itemsize:
            raise ValueError(f"{path} truncated voxel data")
        data = np.frombuffer(raw, dtype=dtype)
        # NIfTI stores data with X fastest. Reshape to reversed dims (.., Z, Y, X).
        shape_rev = tuple(reversed(dims))
        data = data.reshape(shape_rev, order="C")

    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        # Apply scaling in float for safety.
        slope = 1.0 if scl_slope == 0.0 else scl_slope
        data = data.astype(np.float32) * slope + np.float32(scl_inter)

    affine = None
    if sform_code > 0:
        affine = np.array([srow_x, srow_y, srow_z, (0.0, 0.0, 0.0, 1.0)], dtype=np.float64)
    elif qform_code > 0:
        affine = _affine_from_qform(
            pixdim_xyz=(dx, dy, dz),
            quatern_bcd=(quatern_b, quatern_c, quatern_d),
            qoffset_xyz=(qoffset_x, qoffset_y, qoffset_z),
            qfac=qfac,
        )

    header_out: Dict[str, Any] = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "ndim": ndim,
        "dims": dims,
        "datatype": datatype,
        "bitpix": bitpix,
        "pixdim_xyz": (dx, dy, dz),
        "vox_offset": vox_offset,
        "scl_slope": scl_slope,
        "scl_inter": scl_inter,
        "qform_code": qform_code,
        "sform_code": sform_code,
    }
    spacing_zyx = (dz, dy, dx)
    return NiftiImage(data_zyx=data, spacing_zyx=spacing_zyx, affine_xyz=affine, header=header_out)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write to avoid corrupt .npz files if interrupted mid-write.
    # NOTE: numpy.savez_compressed appends ".npz" if the filename does not end with ".npz".
    # Ensure our temp path ends with ".npz" so os.replace() can find it.
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        np.savez_compressed(str(tmp), **arrays)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def resample_zyx_to_spacing(
    data_zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    target_spacing_zyx: Tuple[float, float, float],
    *,
    order: int,
) -> tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Resample a Z,Y,X array to a new spacing using scipy.ndimage.zoom.

    Returns (resampled_data_zyx, target_spacing_zyx).
    """
    zoom = tuple(float(s_in) / float(s_out) for s_in, s_out in zip(spacing_zyx, target_spacing_zyx))
    out = ndimage.zoom(data_zyx, zoom=zoom, order=order, mode="nearest")
    return out, target_spacing_zyx


def resample_zyx_to_shape(
    data_zyx: np.ndarray,
    target_shape_zyx: Tuple[int, int, int],
    *,
    order: int,
) -> np.ndarray:
    zoom = tuple(t / s for t, s in zip(target_shape_zyx, data_zyx.shape))
    return ndimage.zoom(data_zyx, zoom=zoom, order=order, mode="nearest")


def write_nifti_scalar_zyx(
    path: str | Path,
    data_zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    *,
    dtype: np.dtype | None = None,
) -> None:
    """
    Minimal NIfTI-1 writer for 3D scalar volumes in Z,Y,X array order.

    - Writes an identity-ish sform using spacing only.
    - Intended for saving derived outputs (warps/Jacobians/etc).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data_zyx.ndim != 3:
        raise ValueError("write_nifti_scalar_zyx expects a 3D array (Z,Y,X)")

    if dtype is None:
        if np.issubdtype(data_zyx.dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(data_zyx.dtype, np.integer):
            dtype = np.int16
        else:
            dtype = np.float32
    data = np.asarray(data_zyx, dtype=dtype)

    nz, ny, nx = data.shape
    dz, dy, dx = map(float, spacing_zyx)

    # Header: NIfTI-1 single-file format.
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)  # sizeof_hdr

    # dim[0]=3, dim[1]=nx, dim[2]=ny, dim[3]=nz
    struct.pack_into("<8h", hdr, 40, 3, nx, ny, nz, 1, 1, 1, 1)

    # datatype + bitpix
    if data.dtype == np.uint8:
        datatype, bitpix = 2, 8
    elif data.dtype == np.int16:
        datatype, bitpix = 4, 16
    elif data.dtype == np.int32:
        datatype, bitpix = 8, 32
    elif data.dtype == np.float32:
        datatype, bitpix = 16, 32
    elif data.dtype == np.float64:
        datatype, bitpix = 64, 64
    else:
        raise ValueError(f"Unsupported dtype for NIfTI writer: {data.dtype}")
    struct.pack_into("<h", hdr, 70, datatype)
    struct.pack_into("<h", hdr, 72, bitpix)

    # pixdim (x,y,z). pixdim[0]=qfac (unused here).
    struct.pack_into("<8f", hdr, 76, 1.0, dx, dy, dz, 0.0, 0.0, 0.0, 0.0)

    # vox_offset: write header(348) + extension(4) = 352
    struct.pack_into("<f", hdr, 108, 352.0)

    # scaling
    struct.pack_into("<f", hdr, 112, 1.0)  # scl_slope
    struct.pack_into("<f", hdr, 116, 0.0)  # scl_inter

    # sform: simple diagonal affine. (x,y,z) -> mm
    struct.pack_into("<h", hdr, 254, 1)  # sform_code
    struct.pack_into("<4f", hdr, 280, dx, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", hdr, 296, 0.0, dy, 0.0, 0.0)
    struct.pack_into("<4f", hdr, 312, 0.0, 0.0, dz, 0.0)

    # magic for single-file NIfTI.
    hdr[344:348] = b"n+1\0"

    ext = b"\0\0\0\0"
    raw = data.tobytes(order="C")  # with (Z,Y,X) this matches x-fastest layout.

    if str(path).endswith(".gz"):
        with gzip.open(path, "wb") as f:
            f.write(hdr)
            f.write(ext)
            f.write(raw)
    else:
        with open(path, "wb") as f:
            f.write(hdr)
            f.write(ext)
            f.write(raw)
