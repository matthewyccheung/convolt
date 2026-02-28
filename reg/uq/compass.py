from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class CompassParams:
    beta_bounds: Tuple[float, float] = (-1.5, 1.5)
    beta_tol: float = 1e-5
    max_iter: int = 80


def delta_from_beta(
    *,
    beta: np.ndarray,
    exhale_pred0_ml: np.ndarray,
    inhale_vol_ml: np.ndarray,
) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64)
    v = np.asarray(exhale_pred0_ml, dtype=np.float64)
    inh = np.asarray(inhale_vol_ml, dtype=np.float64)
    return np.exp(beta) * v - inh


def beta_from_exhale_volume(
    *,
    exhale_ml: np.ndarray,
    exhale_pred0_ml: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    exhale_ml = np.asarray(exhale_ml, dtype=np.float64)
    exhale_pred0_ml = np.asarray(exhale_pred0_ml, dtype=np.float64)
    return np.log((exhale_ml + eps) / (exhale_pred0_ml + eps))


def beta_from_delta_analytic(
    *,
    delta_ml: np.ndarray,
    exhale_pred0_ml: np.ndarray,
    inhale_vol_ml: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Invert delta(beta) analytically:
      delta = exp(beta)*V0 - Vinh  => beta = log((delta + Vinh)/V0)
    """
    delta_ml = np.asarray(delta_ml, dtype=np.float64)
    v0 = np.asarray(exhale_pred0_ml, dtype=np.float64)
    inh = np.asarray(inhale_vol_ml, dtype=np.float64)
    return np.log((delta_ml + inh + eps) / (v0 + eps))


def beta_from_delta_binary_search(
    *,
    target_delta_ml: float,
    exhale_pred0_ml: float,
    inhale_vol_ml: float,
    params: CompassParams = CompassParams(),
) -> float:
    """
    Monotone binary search for beta such that delta(beta) ~= target_delta_ml.
    """
    lo, hi = map(float, params.beta_bounds)
    tgt = float(target_delta_ml)
    v0 = float(exhale_pred0_ml)
    inh = float(inhale_vol_ml)

    for _ in range(int(params.max_iter)):
        mid = 0.5 * (lo + hi)
        dmid = float(delta_from_beta(beta=np.array(mid), exhale_pred0_ml=np.array(v0), inhale_vol_ml=np.array(inh)))
        if abs(dmid - tgt) <= float(params.beta_tol):
            return mid
        if dmid < tgt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def monotonicity_check_beta_grid(
    *,
    beta_grid: np.ndarray,
    exhale_pred0_ml: float,
    inhale_vol_ml: float,
) -> bool:
    beta_grid = np.asarray(beta_grid, dtype=np.float64)
    d = delta_from_beta(beta=beta_grid, exhale_pred0_ml=np.array(exhale_pred0_ml), inhale_vol_ml=np.array(inhale_vol_ml))
    return bool(np.all(np.diff(d) > 0))

