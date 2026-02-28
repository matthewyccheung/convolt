from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


_NP_BY_MET = {
    "MET_UCHAR": np.uint8,
    "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16,
    "MET_SHORT": np.int16,
    "MET_UINT": np.uint32,
    "MET_INT": np.int32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


@dataclass(frozen=True)
class MetaImage:
    data: np.ndarray
    header: Dict[str, str]


def read_mhd(path: str | Path) -> MetaImage:
    """
    Minimal MetaImage (.mhd + .raw) reader sufficient for Elastix/Transformix outputs.
    """
    path = Path(path)
    hdr: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        hdr[k.strip()] = v.strip()

    ndims = int(hdr.get("NDims", "3"))
    dim_sizes = tuple(int(x) for x in hdr["DimSize"].split())
    if len(dim_sizes) != ndims:
        raise ValueError(f"DimSize/NDims mismatch in {path}")

    elem_type = hdr["ElementType"]
    dtype = _NP_BY_MET.get(elem_type)
    if dtype is None:
        raise ValueError(f"Unsupported ElementType {elem_type} in {path}")

    n_channels = int(hdr.get("ElementNumberOfChannels", "1"))
    raw_name = hdr.get("ElementDataFile")
    if not raw_name:
        raise ValueError(f"Missing ElementDataFile in {path}")
    raw_path = (path.parent / raw_name).resolve()

    raw = raw_path.read_bytes()
    arr = np.frombuffer(raw, dtype=dtype)

    # MetaImage uses C-order with first dimension varying fastest (x).
    # We'll return array in (C, Z, Y, X) if channels>1 else (Z, Y, X),
    # mirroring our NIfTI reader's "reversed" convention.
    spatial = tuple(reversed(dim_sizes))  # (Z,Y,X)
    if n_channels > 1:
        expected = n_channels * int(np.prod(dim_sizes))
        if arr.size != expected:
            raise ValueError(f"Raw size mismatch in {raw_path}: got {arr.size}, expected {expected}")
        arr = arr.reshape((n_channels,) + spatial)
    else:
        expected = int(np.prod(dim_sizes))
        if arr.size != expected:
            raise ValueError(f"Raw size mismatch in {raw_path}: got {arr.size}, expected {expected}")
        arr = arr.reshape(spatial)

    return MetaImage(data=arr, header=hdr)

