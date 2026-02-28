from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RidgeRegressor:
    feature_keys: Tuple[str, ...]
    coef_: np.ndarray  # shape (d,)
    intercept_: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + float(self.intercept_)


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """
    Closed-form ridge regression with intercept (not penalized).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of rows")

    # Center X, y.
    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    Xc = X - x_mean
    yc = y - y_mean

    d = X.shape[1]
    A = Xc.T @ Xc + float(l2) * np.eye(d)
    b = Xc.T @ yc
    coef = np.linalg.solve(A, b)
    intercept = y_mean - float((x_mean.reshape(-1) @ coef))
    return coef.astype(np.float32), float(intercept)


def fit_ridge_weighted(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """
    Weighted ridge regression with intercept (intercept not penalized):
      min_{b,a} sum_i w_i (y_i - a - x_i^T b)^2 + l2 ||b||^2
    Implemented by weighted centering + solving normal equations.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] != y.shape[0] or X.shape[0] != w.shape[0]:
        raise ValueError("X, y, w must have same number of rows")

    w = np.clip(w, 0.0, np.inf)
    ws = float(np.sum(w))
    if ws <= 0:
        # Fall back to unweighted.
        return fit_ridge(X, y, l2=float(l2))

    # Weighted means.
    x_mean = (w[:, None] * X).sum(axis=0, keepdims=True) / ws
    y_mean = float((w * y).sum() / ws)
    Xc = X - x_mean
    yc = y - y_mean

    # Apply sqrt weights.
    sw = np.sqrt(w)[:, None]
    Xw = Xc * sw
    yw = yc * sw.reshape(-1)

    d = X.shape[1]
    A = Xw.T @ Xw + float(l2) * np.eye(d)
    b = Xw.T @ yw
    coef = np.linalg.solve(A, b)
    intercept = y_mean - float((x_mean.reshape(-1) @ coef))
    return coef.astype(np.float32), float(intercept)


@dataclass(frozen=True)
class QuantileRegressor:
    tau: float
    coef_: np.ndarray  # shape (d,), in standardized coordinates
    intercept_: float  # in standardized coordinates
    x_mean_: np.ndarray  # shape (d,)
    x_scale_: np.ndarray  # shape (d,)
    y_mean_: float
    y_scale_: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        Xz = (X - self.x_mean_[None, :]) / self.x_scale_[None, :]
        yz = Xz @ self.coef_.astype(np.float64) + float(self.intercept_)
        return (float(self.y_scale_) * yz + float(self.y_mean_)).astype(np.float32)


def fit_quantile_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tau: float,
    l2: float = 1e-3,
    n_iter: int = 3000,
    lr: float = 5e-2,
    seed: int = 0,
) -> QuantileRegressor:
    """
    Linear quantile regression with L2 penalty using Adam on the pinball loss.

      min_{b,a} mean_i ρ_tau(y_i - a - x_i^T b) + l2 ||b||^2

    This is used for CQR (Romano et al., 2019).
    """
    tau = float(tau)
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0,1)")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of rows")
    if X.shape[0] == 0:
        raise ValueError("Empty dataset")

    # Standardize X.
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    Xz = (X - x_mean[None, :]) / x_scale[None, :]

    # Standardize y for a stable optimizer step size.
    y_mean = float(y.mean())
    y_scale = float(y.std())
    if not np.isfinite(y_scale) or y_scale < 1e-12:
        y_scale = 1.0
    yz = (y - y_mean) / y_scale

    n, d = Xz.shape
    rng = np.random.default_rng(int(seed))
    w = np.zeros(d, dtype=np.float64)
    b = float(0.0)

    # Adam state.
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = 0.0
    v_b = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    l2 = float(l2)
    lr = float(lr)
    for t in range(1, int(n_iter) + 1):
        # Full-batch gradients on standardized data.
        yhat = Xz @ w + b
        u = yz - yhat
        # d/dyhat pinball = I(u<0) - tau
        g = (u < 0.0).astype(np.float64) - tau
        grad_w = (Xz.T @ g) / float(n) + 2.0 * l2 * w
        grad_b = float(np.mean(g))

        # Adam updates.
        m_w = beta1 * m_w + (1.0 - beta1) * grad_w
        v_w = beta2 * v_w + (1.0 - beta2) * (grad_w * grad_w)
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b
        v_b = beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b)

        m_w_hat = m_w / (1.0 - beta1**t)
        v_w_hat = v_w / (1.0 - beta2**t)
        m_b_hat = m_b / (1.0 - beta1**t)
        v_b_hat = v_b / (1.0 - beta2**t)

        step_w = lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        step_b = lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
        w -= step_w
        b -= float(step_b)

        # Occasional tiny jitter helps avoid pathological plateaus on very small datasets.
        if t % 500 == 0 and d > 0:
            w += rng.normal(0.0, 1e-6, size=w.shape)

    return QuantileRegressor(
        tau=tau,
        coef_=w.astype(np.float32),
        intercept_=float(b),
        x_mean_=x_mean.astype(np.float32),
        x_scale_=x_scale.astype(np.float32),
        y_mean_=float(y_mean),
        y_scale_=float(y_scale),
    )
