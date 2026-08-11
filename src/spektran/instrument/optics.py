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


def window_contamination(
    nu_cm1: np.ndarray,
    contamination_rel: float,
    spectral_slope: float = 0.0,
) -> np.ndarray:
    """Multiplicative transmission loss from window fouling (dust, condensation,
    chemical deposits) accumulated on cell optics.

    Broadband loss ``contamination_rel`` plus an optional Rayleigh-like
    wavelength-dependent scattering term (~(nu/nu_center)^spectral_slope;
    0 = flat, 4 = full Rayleigh), renormalized so the mean loss across the
    scan still equals ``contamination_rel``.
    """
    if contamination_rel <= 0.0:
        return np.ones_like(nu_cm1, dtype=np.float64)
    if spectral_slope == 0.0:
        return np.full_like(nu_cm1, 1.0 - contamination_rel, dtype=np.float64)
    nu_center = 0.5 * (nu_cm1[0] + nu_cm1[-1])
    spectral_factor = (nu_cm1 / nu_center) ** spectral_slope
    mean_factor = np.mean(spectral_factor)
    loss = contamination_rel * spectral_factor / mean_factor
    return np.clip(1.0 - loss, 0.0, 1.0)


def beam_wander(
    rng: np.random.Generator,
    n: int,
    sigma_rel: float,
    cutoff_norm: float = 0.05,
) -> np.ndarray:
    """Low-frequency multiplicative intensity modulation from beam wander.

    Mechanical vibration and thermal drift shift the laser spot on the
    detector. Modeled like ``intensity_fluctuation`` (band-limited Gaussian
    noise) but with a higher default cutoff, since beam wander carries more
    energy near mechanical resonance modes (~10-100 Hz) than pure thermal
    drift.
    """
    if sigma_rel <= 0.0 or n < 2:
        return np.ones(max(n, 1), dtype=np.float64)
    raw = rng.normal(0.0, 1.0, n)
    sos = butter(2, cutoff_norm, btype="low", output="sos")
    slow = sosfiltfilt(sos, raw)
    std = slow.std()
    if std > 0.0:
        slow = slow * (sigma_rel / std)
    return 1.0 + slow


def gas_flow_turbulence(
    rng: np.random.Generator,
    n: int,
    sigma_rel: float,
    cutoff_norm: float = 0.1,
) -> np.ndarray:
    """Multiplicative refractive-index fluctuation from gas flow turbulence.

    Turbulent eddies in flow-through gas cells cause optical path length
    variations. Produces band-limited multiplicative noise (mid-frequency).
    Returns 1 + delta.
    """
    if sigma_rel <= 0.0 or n < 4:
        return np.ones(max(n, 1), dtype=np.float64)
    raw = rng.normal(0.0, 1.0, n)
    sos = butter(2, min(cutoff_norm, 0.49), btype="low", output="sos")
    filtered = sosfiltfilt(sos, raw)
    std = filtered.std()
    if std > 0.0:
        filtered = filtered * (sigma_rel / std)
    return 1.0 + filtered


def correlated_baseline_drift(
    rng: np.random.Generator,
    n: int,
    coeffs_sigma: list[float],
    previous_coeffs: list[float] | None = None,
    tau_scans: float = 50.0,
) -> tuple[np.ndarray, list[float]]:
    """Baseline polynomial with Ornstein-Uhlenbeck drift across scans.

    Coefficients evolve as correlated random walks (OU processes) instead
    of being re-sampled independently. Returns (baseline_array, new_coeffs)
    so the caller can chain scans in time-series mode.
    """
    du = np.linspace(-0.5, 0.5, n)
    if previous_coeffs is None:
        coeffs = [rng.normal(0.0, s) for s in coeffs_sigma]
    else:
        alpha = np.exp(-1.0 / max(tau_scans, 1e-6))
        coeffs = [
            alpha * prev + np.sqrt(1.0 - alpha * alpha) * rng.normal(0.0, s)
            for prev, s in zip(previous_coeffs, coeffs_sigma)
        ]
    base = np.ones(n, dtype=np.float64)
    for k, c in enumerate(coeffs):
        base = base + c * du ** (k + 1)
    return base, coeffs
