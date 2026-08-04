"""Optical-path effects: baseline drift, intensity fluctuation, transmittance decay."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def baseline_polynomial(ramp: np.ndarray, coeffs_rel: list[float]) -> np.ndarray:
    """Multiplicative slow baseline over the scan: 1 + sum_k c_k * (u-1/2)^(k+1)."""
    du = ramp - 0.5
    base = np.ones_like(ramp, dtype=np.float64)
    for k, c in enumerate(coeffs_rel):
        base = base + c * du ** (k + 1)
    return base


def intensity_fluctuation(
    rng: np.random.Generator,
    n: int,
    sigma_rel: float,
    cutoff_norm: float = 0.01,
) -> np.ndarray:
    """Low-frequency multiplicative intensity fluctuation (1 + delta(t)).

    Gaussian noise low-pass filtered at ``cutoff_norm`` (fraction of the
    Nyquist rate), renormalized to sigma_rel.
    """
    raw = rng.normal(0.0, 1.0, n)
    sos = butter(2, cutoff_norm, btype="low", output="sos")
    slow = sosfiltfilt(sos, raw)
    std = slow.std()
    if std > 0.0:
        slow = slow * (sigma_rel / std)
    return 1.0 + slow


def transmittance_decay(t_s: np.ndarray, decay_rel_per_s: float) -> np.ndarray:
    """Slow multiplicative transmittance loss (window fouling), linearized."""
    return np.clip(1.0 - decay_rel_per_s * t_s, 0.0, None)
