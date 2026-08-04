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


def peak_height_ratio_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE for WMS 2f peak-height concentration regression (T4)."""
    return mae(y_true, y_pred)


def allan_variance(
    series: np.ndarray, tau_points: int = 20, dt: float = 1.0
) -> dict[str, list[float]]:
    """Overlapping Allan deviation for a 1-D time series."""
    n = len(series)
    max_m = n // 2
    ms = np.unique(np.geomspace(1, max_m, tau_points).astype(int))
    s = np.insert(np.cumsum(series), 0, 0.0)
    taus, adevs = [], []
    for m in ms:
        tau = m * dt
        segments = n - 2 * m
        if segments < 1:
            continue
        # s has n+1 entries (leading 0), so all three windows below must be
        # sliced to the same `segments` length explicitly — s[m:-m] is off
        # by one whenever m != 0, which silently breaks the m=0 edge case.
        diff2 = (s[2 * m : 2 * m + segments] - 2 * s[m : m + segments] + s[:segments]) ** 2
        adev = float(np.sqrt(np.mean(diff2) / (2.0 * tau * tau)))
        taus.append(float(tau))
        adevs.append(adev)
    return {"taus": taus, "adevs": adevs}


def ood_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area under ROC curve for OOD instrument detection (T6).

    y_true: 0 = in-distribution, 1 = out-of-distribution.
    y_scores: model confidence that the sample is OOD.
    """
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    count = 0.0
    for p in pos:
        count += np.sum(neg < p) + 0.5 * np.sum(neg == p)
    return float(count / (n_pos * n_neg))
