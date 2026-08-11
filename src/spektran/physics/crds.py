"""CRDS forward model: cavity ring-down spectroscopy.

Cavity Ring-Down Spectroscopy measures the exponential decay time of light
trapped in a high-finesse optical cavity.  The ring-down time tau depends
on total cavity losses (mirror transmission + gas absorption):

    tau  = L / [c * (1 - R + alpha(nu) * L)]
    tau0 = L / [c * (1 - R)]                    (empty cavity)

The absorption coefficient is recovered from:

    alpha(nu) = (1/tau - 1/tau0) / c

where alpha(nu) comes from the same HITRAN line-by-line model used in
TDLAS.  HITRAN line intensities are literally calibrated against NIST
FS-CRDS measurements (Hodges et al., doi:10.1364/AO.54.001001), making
CRDS simulation self-consistent with the spectroscopic database.

Forward model chain::

    HITRAN line data
      -> Beer-Lambert alpha(nu) [cm-1]
        -> Ring-down time tau(nu) = L / [c * (1-R + alpha*L)]
          -> Ring-down trace I(t) = I0 * exp(-t/tau) + noise
            -> Exponential fit -> recovered tau -> alpha -> concentration

References:
- G. Berden and R. Engeln, "Cavity Ring-Down Spectroscopy: Techniques
  and Applications", Wiley (2009), doi:10.1002/9781444308259
- J.T. Hodges et al., "Frequency-stabilized cavity ring-down
  spectroscopy", Appl. Opt. 54 (2015) 1001,
  doi:10.1364/AO.54.001001
- K.K. Lehmann and D. Romanini, "The superposition principle and cavity
  ring-down spectroscopy", J. Chem. Phys. 105 (1996) 10263,
  doi:10.1063/1.472955
"""

from __future__ import annotations

import numpy as np

from .absorption import absorption_coefficient
from .constants import C_CM_PER_S
from .hitran import LineList


def ring_down_time(
    cavity_length_cm: float,
    mirror_reflectivity: float,
    absorption_cm1: float = 0.0,
) -> float:
    """Cavity ring-down time tau [seconds].

    tau = L / [c * (1 - R + alpha * L)]

    Berden & Engeln (2009) eq. 2.1, doi:10.1002/9781444308259.ch2
    """
    total_loss = 1.0 - mirror_reflectivity + absorption_cm1 * cavity_length_cm
    return cavity_length_cm / (C_CM_PER_S * total_loss)


def empty_cavity_tau(
    cavity_length_cm: float,
    mirror_reflectivity: float,
) -> float:
    """Empty-cavity ring-down time tau0 [seconds]."""
    return ring_down_time(cavity_length_cm, mirror_reflectivity, 0.0)


def absorption_from_tau(
    tau: float | np.ndarray,
    tau0: float,
    cavity_length_cm: float,
) -> float | np.ndarray:
    """Recover absorption coefficient [cm-1] from ring-down times.

    alpha = (1/tau - 1/tau0) / c

    Lehmann & Romanini (1996) eq. 5, doi:10.1063/1.472955
    """
    return (1.0 / tau - 1.0 / tau0) / C_CM_PER_S


def ring_down_trace(
    t_s: np.ndarray,
    tau_s: float,
    I0: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Generate ideal ring-down intensity trace I(t) = I0 * exp(-t/tau) + offset."""
    return I0 * np.exp(-t_s / tau_s) + offset


def cavity_finesse(mirror_reflectivity: float) -> float:
    """Cavity finesse F = pi * sqrt(R) / (1 - R).

    Berden & Engeln (2009) eq. 2.5.
    """
    return np.pi * np.sqrt(mirror_reflectivity) / (1.0 - mirror_reflectivity)


def nea_cm1(
    cavity_length_cm: float,
    mirror_reflectivity: float,
    delta_tau_over_tau: float,
) -> float:
    """Noise-equivalent absorption [cm-1] from fractional tau precision.

    NEA = (1 - R) / L * (delta_tau / tau)

    Paldus et al., J. Appl. Phys. 83 (1998) 3991,
    doi:10.1063/1.367155
    """
    return (1.0 - mirror_reflectivity) / cavity_length_cm * delta_tau_over_tau


def simulate_crds_spectrum(
    lines: LineList,
    molecule: str,
    concentration_ppm: float,
    temperature_K: float,
    pressure_atm: float,
    cavity_length_m: float,
    mirror_reflectivity: float,
    wavenumber_start_cm1: float = 6046.0,
    wavenumber_end_cm1: float = 6048.0,
    n_spectral_points: int = 200,
    interferents: list[dict] | None = None,
) -> dict:
    """Simulate a clean CRDS absorption spectrum (no noise).

    Scans the laser across wavenumber range, computing ring-down time
    at each spectral point.  Returns the tau spectrum and derived
    absorption coefficient spectrum.

    Returns dict with keys: ``nu_cm1``, ``tau_spectrum_s``, ``tau0_s``,
    ``alpha_spectrum_cm1``, ``absorbance_spectrum``, ``concentration_ppm``.
    """
    cavity_length_cm = cavity_length_m * 100.0
    nu = np.linspace(wavenumber_start_cm1, wavenumber_end_cm1, n_spectral_points)

    mole_fraction = concentration_ppm * 1e-6
    alpha = absorption_coefficient(
        nu, lines, mole_fraction, temperature_K, pressure_atm,
    )

    if interferents:
        for interf in interferents:
            alpha_i = absorption_coefficient(
                nu,
                interf["lines"],
                interf["concentration_ppm"] * 1e-6,
                temperature_K,
                pressure_atm,
            )
            alpha = alpha + alpha_i

    tau0 = empty_cavity_tau(cavity_length_cm, mirror_reflectivity)
    tau_spectrum = np.array([
        ring_down_time(cavity_length_cm, mirror_reflectivity, a)
        for a in alpha
    ])

    path_cm = cavity_length_cm
    absorbance = alpha * path_cm

    return {
        "nu_cm1": nu,
        "tau_spectrum_s": tau_spectrum,
        "tau0_s": tau0,
        "alpha_spectrum_cm1": alpha,
        "absorbance_spectrum": absorbance,
        "concentration_ppm": concentration_ppm,
        "cavity_length_cm": cavity_length_cm,
        "mirror_reflectivity": mirror_reflectivity,
        "finesse": cavity_finesse(mirror_reflectivity),
    }


def fit_ring_down(
    t_s: np.ndarray,
    intensity: np.ndarray,
) -> tuple[float, float, float]:
    """Fit single-exponential ring-down to recover tau.

    Uses linear regression on log(I) for speed (no iterative optimizer).
    Returns (tau_s, I0, offset_estimate).

    For noisy data with significant offset, a proper Levenberg-Marquardt
    fit would be better, but log-linear is standard for high-SNR CRDS
    (Chen et al., Atmos. Meas. Tech. 3 (2010) 375,
    doi:10.5194/amt-3-375-2010).
    """
    mask = intensity > 0
    t_fit = t_s[mask]
    log_I = np.log(intensity[mask])

    coeffs = np.polyfit(t_fit, log_I, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    tau_s = -1.0 / slope if slope < 0 else 1e-3
    I0 = np.exp(intercept)
    return tau_s, I0, 0.0
