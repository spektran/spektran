"""Laser imperfections: scan nonlinearity, intensity ramp/RAM, drift, linewidth.

All functions are deterministic given their inputs; stochastic parameter
values are sampled upstream by the virtual-instrument layer and recorded in
each record's provenance (plan §5.3).
"""

from __future__ import annotations

import numpy as np

from ..physics.lineshape import lorentz_profile


def scan_frequency_axis(
    ramp: np.ndarray,
    center_wavenumber_cm1: float,
    scan_range_cm1: float,
    nonlinearity_poly_cm1: list[float] | None = None,
) -> np.ndarray:
    """Optical frequency vs normalized ramp position u in [0, 1).

    nu(u) = nu_c + R*(u - 1/2) + sum_k p_k * (u - 1/2)^(k+2)

    The polynomial models current->frequency tuning nonlinearity (order >= 2
    terms only, so endpoints of the ideal linear scan are preserved to first
    order).
    """
    du = ramp - 0.5
    nu = center_wavenumber_cm1 + scan_range_cm1 * du
    if nonlinearity_poly_cm1:
        for k, p in enumerate(nonlinearity_poly_cm1):
            nu = nu + p * du ** (k + 2)
    return nu


def intensity_ramp(
    ramp: np.ndarray,
    mean_intensity: float = 1.0,
    slope_rel: float = 0.0,
    curvature_rel: float = 0.0,
) -> np.ndarray:
    """Laser intensity vs ramp position (current sweep changes output power).

    I(u) = Ibar * (1 + s*(u - 1/2) + c*(u - 1/2)^2)
    """
    du = ramp - 0.5
    return mean_intensity * (1.0 + slope_rel * du + curvature_rel * du * du)


def center_drift_cm1(
    t_s: np.ndarray, drift_rate_cm1_per_s: float, phase_s: float = 0.0
) -> np.ndarray:
    """Slow linear drift of the laser center frequency."""
    return drift_rate_cm1_per_s * (t_s - phase_s)


def linewidth_convolve(
    absorbance: np.ndarray,
    step_cm1: float,
    linewidth_MHz: float,
) -> np.ndarray:
    """Convolve an absorbance spectrum with the laser Lorentzian lineshape.

    Valid in the optically thin limit where the measured absorbance is
    approximately the true absorbance convolved with the (area-normalized)
    laser profile. linewidth_MHz is the FWHM; 0 returns the input unchanged.
    """
    if linewidth_MHz <= 0.0:
        return absorbance
    hwhm_cm1 = 0.5 * linewidth_MHz * 1e6 / 2.99792458e10  # MHz -> cm-1
    half_span = max(50.0 * hwhm_cm1, 5.0 * step_cm1)
    n_half = int(np.ceil(half_span / step_cm1))
    grid = np.arange(-n_half, n_half + 1) * step_cm1
    kernel = lorentz_profile(grid, 0.0, hwhm_cm1) * step_cm1
    kernel = kernel / kernel.sum()  # renormalize truncated tails
    return np.convolve(absorbance, kernel, mode="same")
