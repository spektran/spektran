"""Benchmark evaluation metrics (plan §6).

All metrics take numpy arrays; predictions and truths are aligned by caller
(evaluate.py joins on record_id, so ordering here is guaranteed)."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error [same unit as y]."""
    return float(np.mean(np.abs(y_pred - y_true)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Mean absolute percentage error [%]. Truths of 0 are guarded by eps."""
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def spectral_rmse(spec_true: np.ndarray, spec_pred: np.ndarray) -> float:
    """RMSE over all points of all spectra (denoising task, plan T2)."""
    return float(np.sqrt(np.mean((spec_pred - spec_true) ** 2)))


def peak_weighted_rmse(
    spec_true: np.ndarray, spec_pred: np.ndarray, quantile: float = 0.9
) -> float:
    """RMSE restricted to the strongest-absorbance region of each spectrum.

    Per spectrum, points with true absorbance above its ``quantile`` are the
    'peak region'; errors there matter most for concentration retrieval.
    """
    err2, n = 0.0, 0
    for t, p in zip(np.atleast_2d(spec_true), np.atleast_2d(spec_pred)):
        thresh = np.quantile(t, quantile)
        mask = t >= thresh
        err2 += float(np.sum((p[mask] - t[mask]) ** 2))
        n += int(mask.sum())
    return float(np.sqrt(err2 / max(n, 1)))


def degradation_ratio(metric_held_out: float, metric_in_dist: float) -> float:
    """Cross-instrument degradation: held-out metric / in-distribution metric.

    1.0 = no degradation; the flagship T3 headline number (plan §6.1).
    """
    return float(metric_held_out / max(metric_in_dist, 1e-12))
