"""CRDS instrument noise chain.

Noise sources for cavity ring-down spectroscopy, modeled from:

- Shot noise on photon counting (Poisson statistics)
- Mirror reflectivity drift (Ornstein-Uhlenbeck process)
- Mode matching jitter (random transverse mode excitation)
- Detector electronics noise (additive Gaussian)
- Baseline loss drift (linear + temperature correlation)
- Exponential fitting residual (from multi-mode excitation)

References:
- B.A. Paldus et al., "Cavity-locked ring-down spectroscopy",
  J. Appl. Phys. 83 (1998) 3991, doi:10.1063/1.367155
- D.A. Long et al., "Frequency-stabilized cavity ring-down
  spectroscopy", Appl. Opt. 54 (2015) 1001,
  doi:10.1364/AO.54.001001
"""

from __future__ import annotations

import numpy as np


def shot_noise_tau(
    rng: np.random.Generator,
    tau_s: float | np.ndarray,
    n_photons: float = 1e6,
) -> float | np.ndarray:
    """Add shot-noise-limited uncertainty to ring-down time.

    For N detected photons per ring-down event, the fractional
    precision is delta_tau/tau ~ 1/sqrt(2*N) (Lehmann & Romanini 1996).
    """
    fractional_sigma = 1.0 / np.sqrt(2.0 * n_photons)
    tau = np.asarray(tau_s, dtype=np.float64)
    noise = rng.normal(0.0, fractional_sigma, tau.shape) * tau
    return tau + noise


def mirror_drift(
    rng: np.random.Generator,
    n_points: int,
    base_reflectivity: float,
    drift_sigma: float = 1e-7,
    mean_reversion_rate: float = 0.01,
) -> np.ndarray:
    """Ornstein-Uhlenbeck mirror reflectivity drift.

    Models slow contamination/thermal effects on cavity mirrors.
    Long et al. (2015) report sub-ppm drift for temperature-stabilized
    cavities.
    """
    R = np.empty(n_points)
    R[0] = base_reflectivity
    for i in range(1, n_points):
        dR = -mean_reversion_rate * (R[i - 1] - base_reflectivity)
        dR += rng.normal(0.0, drift_sigma)
        R[i] = np.clip(R[i - 1] + dR, 0.9999, 0.999999)
    return R


def mode_matching_jitter(
    rng: np.random.Generator,
    n_points: int,
    coupling_efficiency_mean: float = 0.95,
    coupling_efficiency_sigma: float = 0.02,
) -> np.ndarray:
    """Random mode-matching efficiency variation.

    Imperfect alignment excites higher-order transverse modes (TEM_mn),
    each with different round-trip loss.  Models the effective coupling
    as a noisy scalar. Paldus et al. (1998).
    """
    eta = rng.normal(coupling_efficiency_mean, coupling_efficiency_sigma, n_points)
    return np.clip(eta, 0.5, 1.0)


def detector_noise(
    rng: np.random.Generator,
    n_points: int,
    noise_sigma_rel: float = 1e-4,
) -> np.ndarray:
    """Additive detector electronics noise (Gaussian white noise).

    Relative to the ring-down signal amplitude.
    """
    return rng.normal(0.0, noise_sigma_rel, n_points)


def baseline_loss_drift(
    rng: np.random.Generator,
    n_points: int,
    drift_rate_per_point: float = 1e-8,
    temperature_sensitivity: float = 1e-7,
) -> np.ndarray:
    """Slow baseline cavity loss drift.

    Linear drift plus temperature-correlated fluctuations.
    Models contamination buildup on mirrors over a spectral scan.
    """
    linear = np.arange(n_points, dtype=np.float64) * drift_rate_per_point
    temp_fluct = rng.normal(0.0, temperature_sensitivity, n_points)
    return np.cumsum(linear + temp_fluct)


def fitting_residual(
    rng: np.random.Generator,
    tau_s: float | np.ndarray,
    n_modes: int = 1,
    mode_spread_fraction: float = 0.05,
) -> float | np.ndarray:
    """Systematic fitting residual from multi-transverse-mode excitation.

    When multiple TEM modes are excited, the decay is multi-exponential.
    Single-exponential fit introduces a systematic bias proportional to
    the number of modes and their loss spread.
    """
    if n_modes <= 1:
        return np.zeros_like(np.asarray(tau_s))
    bias_fraction = (n_modes - 1) * mode_spread_fraction * 0.1
    tau = np.asarray(tau_s, dtype=np.float64)
    systematic = bias_fraction * tau
    jitter = rng.normal(0.0, bias_fraction * 0.3, tau.shape) * tau
    return systematic + jitter
