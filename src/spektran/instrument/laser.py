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
    tuning_model: str | None = None,
    tuning_params: dict | None = None,
) -> np.ndarray:
    """Optical frequency vs normalized ramp position u in [0, 1).

    nu(u) = nu_c + R*(u - 1/2) + sum_k p_k * (u - 1/2)^(k+2)

    The polynomial models current->frequency tuning nonlinearity (order >= 2
    terms only, so endpoints of the ideal linear scan are preserved to first
    order). Pass ``tuning_model="thermal_chirp"`` to instead use the
    physically motivated thermal-lag model of ``current_tuning_model``
    (``tuning_params`` forwards its keyword arguments); this is opt-in and
    the polynomial path remains the default for backward compatibility.
    """
    if tuning_model == "thermal_chirp":
        params = tuning_params or {}
        return current_tuning_model(ramp, center_wavenumber_cm1, scan_range_cm1, **params)
    du = ramp - 0.5
    nu = center_wavenumber_cm1 + scan_range_cm1 * du
    if nonlinearity_poly_cm1:
        for k, p in enumerate(nonlinearity_poly_cm1):
            nu = nu + p * du ** (k + 2)
    return nu


def current_tuning_model(
    ramp: np.ndarray,
    center_wavenumber_cm1: float,
    scan_range_cm1: float,
    thermal_fraction: float = 0.8,
    thermal_tau_norm: float = 0.3,
) -> np.ndarray:
    """DFB/VCSEL current-tuning model with thermal chirp lag.

    Real diode lasers tune frequency via two superposed mechanisms: an
    instantaneous carrier-density (plasma) chirp that tracks the drive
    current linearly, and a thermal chirp from active-region self-heating
    that lags the current through a first-order thermal time constant
    tau_th. The thermal mechanism dominates in DFB lasers (~70-90% of total
    tuning) and its lag produces the characteristic slow-start/fast-finish
    S-shaped nonlinearity of a current-ramp scan.

    For a linear ramp I(t) = t driven into a first-order lag
    tau*dy/dt + y = I(t) with y(0) = 0, the closed-form response is
    y(t) = t - tau*(1 - exp(-t/tau)); using it avoids an explicit
    convolution with the exponential thermal impulse response.

    Parameters
    ----------
    ramp : array
        Normalized ramp position in [0, 1), linear sawtooth.
    center_wavenumber_cm1 : float
        Center optical frequency of the scan.
    scan_range_cm1 : float
        Total scan range (thermal + current contributions).
    thermal_fraction : float
        Fraction of total tuning due to the thermal effect (0 to 1).
        Typical DFB: 0.7-0.9. VCSEL: 0.5-0.7.
    thermal_tau_norm : float
        Thermal time constant normalized to scan duration (tau_th / T_scan).
        0 means an instantaneous thermal response (recovers the linear scan).
        Typical: 0.1-0.5 for standard scan rates.

    Returns
    -------
    nu_cm1 : array
        Optical frequency at each ramp point [cm-1].
    """
    t = ramp
    current_contrib = (1.0 - thermal_fraction) * (t - 0.5)
    if thermal_tau_norm > 0 and thermal_fraction > 0:
        tau = thermal_tau_norm
        thermal_response = t - tau * (1.0 - np.exp(-t / tau))
        mid = 0.5 - tau * (1.0 - np.exp(-0.5 / tau))
        thermal_contrib = thermal_fraction * (thermal_response - mid)
    else:
        thermal_contrib = thermal_fraction * (t - 0.5)
    return center_wavenumber_cm1 + scan_range_cm1 * (current_contrib + thermal_contrib)


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
