from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile: k = ceil((n+1)*(1-alpha)), q = k-th smallest.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("No finite calibration scores")
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    k = int(np.clip(k, 1, n))
    return float(np.partition(s, k - 1)[k - 1])


@dataclass(frozen=True)
class Interval:
    lo: np.ndarray
    hi: np.ndarray

    def size(self) -> np.ndarray:
        return self.hi - self.lo

    def coverage(self, y: np.ndarray) -> np.ndarray:
        return (y >= self.lo) & (y <= self.hi)


def split_cp_symmetric(
    *,
    yhat_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat_test: np.ndarray,
    alpha: float,
) -> tuple[Interval, Dict[str, float]]:
    scores = np.abs(y_cal - yhat_cal)
    q = conformal_quantile(scores, alpha)
    interval = Interval(lo=yhat_test - q, hi=yhat_test + q)
    return interval, {"q": float(q)}


def weighted_split_cp_symmetric(
    *,
    yhat_cal: np.ndarray,
    y_cal: np.ndarray,
    yhat_test: np.ndarray,
    w_cal: np.ndarray,
    w_test: np.ndarray,
    alpha: float,
) -> tuple[Interval, Dict[str, float]]:
    w_cal = np.asarray(w_cal, dtype=np.float64).reshape(-1)
    w_test = np.asarray(w_test, dtype=np.float64).reshape(-1)
    w_cal = np.clip(w_cal, 1e-6, np.inf)
    w_test = np.clip(w_test, 1e-6, np.inf)
    scores = np.abs(y_cal - yhat_cal) / w_cal
    q = conformal_quantile(scores, alpha)
    interval = Interval(lo=yhat_test - q * w_test, hi=yhat_test + q * w_test)
    return interval, {"q": float(q)}


def local_quantile_1d(
    *,
    s_cal: np.ndarray,
    r_cal: np.ndarray,
    s_test: np.ndarray,
    alpha: float,
    k: int,
) -> np.ndarray:
    """
    1D kNN conformal quantile for adaptive interval sizes:
      q(x) = quantile of residuals among k nearest calibration points in s-space.
    """
    s_cal = np.asarray(s_cal, dtype=np.float64).reshape(-1)
    r_cal = np.asarray(r_cal, dtype=np.float64).reshape(-1)
    s_test = np.asarray(s_test, dtype=np.float64).reshape(-1)
    if s_cal.size != r_cal.size:
        raise ValueError("s_cal and r_cal must have same length")
    if s_cal.size == 0:
        raise ValueError("Empty calibration set")

    k = int(np.clip(k, 1, s_cal.size))
    out = np.zeros_like(s_test, dtype=np.float64)
    for i, s in enumerate(s_test):
        idx = np.argsort(np.abs(s_cal - s))[:k]
        out[i] = conformal_quantile(r_cal[idx], alpha)
    return out.astype(np.float32)

