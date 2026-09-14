"""
Non-learning baselines for comparison:

1. Fixed Nominal: use the optimal gain for the nominal condition (closest to mp=1.0, fv=1.0)
2. Mean Gain: simple average of all training gains

DR (Domain Randomization) is handled separately via TuRBO on CasADi rollout cost.
"""
from __future__ import annotations

import numpy as np


def fixed_nominal_gains(phys_train: np.ndarray, gains_train: np.ndarray,
                        nominal: tuple = (1.0, 1.0)) -> np.ndarray:
    """Return the gain vector from the training set closest to the nominal condition.

    phys_train: (N, 2) — [m_p, fv_mult]
    gains_train: (N, G)
    Returns: (G,) gain vector
    """
    dists = np.sqrt(((phys_train - np.array(nominal)) ** 2).sum(axis=1))
    idx = np.argmin(dists)
    return gains_train[idx].copy()


def mean_gains(gains_train: np.ndarray) -> np.ndarray:
    """Return the simple mean of all training gains.

    gains_train: (N, G)
    Returns: (G,)
    """
    return gains_train.mean(axis=0).copy()


def evaluate_fixed_baseline(
    fixed_gains: np.ndarray,
    true_gains_test: np.ndarray,
) -> dict:
    """Evaluate a fixed (non-adaptive) gain vector against test set.

    fixed_gains: (G,) — single gain vector applied to all test points
    true_gains_test: (M, G) — per-point optimal gains

    Returns dict with MSE and per-sample errors.
    """
    pred = np.broadcast_to(fixed_gains, true_gains_test.shape)
    per_sample_mse = np.mean((pred - true_gains_test) ** 2, axis=1)
    return {
        "gain_mse": float(np.mean(per_sample_mse)),
        "gain_mse_std": float(np.std(per_sample_mse)),
        "per_sample_mse": per_sample_mse,
        "fixed_gains": fixed_gains,
    }
