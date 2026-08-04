"""NDIR forward model: broadband source, bandpass filter, integrated absorption.

Non-Dispersive Infrared (NDIR) spectroscopy uses a broadband thermal source
and optical bandpass filters to measure gas concentration via integrated band
absorption.  The active channel filter selects an absorption band of the
target gas; the reference channel selects a non-absorbing region.  The ratio
of the two integrated detector signals cancels source intensity fluctuations.

Forward model chain::

    Planck source B(nu, T)
      -> Beer-Lambert transmission exp(-A(nu))
        -> Optical bandpass filter F(nu)
          -> Detector integration S = integral B F exp(-A) dnu
            -> Ratio R = S_active / S_reference

References:
- J. Hodgkinson and R.P. Tatam, "Optical gas sensing: a review",
  Meas. Sci. Technol. 24 (2013) 012004,
  doi:10.1088/0957-0233/24/1/012004
"""

from __future__ import annotations

import numpy as np

from .absorption import absorption_coefficient
from .constants import C_CM_PER_S, H_ERG_S, K_ERG_PER_K
from .hitran import LineList

_LN2 = np.log(2.0)

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz  # type: ignore[attr-defined]


def planck_spectral_radiance(
    nu_cm1: np.ndarray,
    temperature_K: float,
) -> np.ndarray:
    """Planck function B(nu, T) in wavenumber space.

    B(nu, T) = 2 h c^2 nu^3 / (exp(h c nu / (k T)) - 1)

    All constants are CGS (erg, cm, s); result units are
    [erg s^-1 sr^-1 cm^-2 / cm^-1].
    """
    nu = np.asarray(nu_cm1, dtype=np.float64)
    exponent = H_ERG_S * C_CM_PER_S * nu / (K_ERG_PER_K * temperature_K)
    return 2.0 * H_ERG_S * C_CM_PER_S**2 * nu**3 / (
        np.exp(exponent) - 1.0
    )


def bandpass_filter(
    nu_cm1: np.ndarray,
    center_cm1: float,
    fwhm_cm1: float,
    shape: str = "gaussian",
) -> np.ndarray:
    """Optical bandpass filter transmission profile in [0, 1].

    Supported shapes:

    - ``"gaussian"``: exp(-4 ln2 (nu - nu0)^2 / FWHM^2)
    - ``"tophat"``: 1 where |nu - nu0| <= FWHM/2, else 0
    """
    nu = np.asarray(nu_cm1, dtype=np.float64)
    if shape == "gaussian":
        return np.exp(
            -4.0 * _LN2 * (nu - center_cm1) ** 2 / fwhm_cm1**2
        )
    if shape == "tophat":
        return np.where(
            np.abs(nu - center_cm1) <= fwhm_cm1 / 2.0, 1.0, 0.0
        )
    raise ValueError(
        f"Unknown filter shape {shape!r}; use 'gaussian' or 'tophat'"
    )


def ndir_detector_signal(
    nu_cm1: np.ndarray,
    absorbance: np.ndarray,
    filter_transmission: np.ndarray,
    source_radiance: np.ndarray,
) -> float:
    """Integrated detector signal S = integral B(nu) F(nu) exp(-A(nu)) dnu.

    Trapezoidal integration over the wavenumber grid.  Returns a scalar.
    """
    integrand = (
        source_radiance * filter_transmission * np.exp(-absorbance)
    )
    return float(_trapezoid(integrand, nu_cm1))


def ndir_ratio(
    active_signal: float,
    reference_signal: float,
) -> float:
    """NDIR measurement ratio R = S_active / S_reference.

    The ratio cancels source intensity fluctuations and common-mode
    drift.
    """
    return active_signal / reference_signal


def simulate_ndir(
    lines: LineList,
    molecule: str,
    concentration_ppm: float,
    temperature_K: float,
    pressure_atm: float,
    path_length_m: float,
    source_temperature_K: float = 800.0,
    active_filter_center_cm1: float = 3018.0,
    active_filter_fwhm_cm1: float = 100.0,
    reference_filter_center_cm1: float = 2500.0,
    reference_filter_fwhm_cm1: float = 100.0,
    filter_shape: str = "gaussian",
    n_integration_points: int = 500,
    interferents: list[dict] | None = None,
) -> dict:
    """Simulate a clean NDIR measurement.

    Both active and reference channels share the same gas cell; the
    reference filter is placed at a non-absorbing wavelength so that
    absorption there is naturally negligible.

    Returns a dict with keys: ``active_signal``, ``reference_signal``,
    ``ratio``, ``concentration_ppm``, ``nu_cm1``, ``absorbance``,
    ``source_radiance``, ``active_filter``, ``reference_filter``.
    """
    margin = 3.0
    lo = min(
        active_filter_center_cm1 - margin * active_filter_fwhm_cm1,
        reference_filter_center_cm1 - margin * reference_filter_fwhm_cm1,
    )
    hi = max(
        active_filter_center_cm1 + margin * active_filter_fwhm_cm1,
        reference_filter_center_cm1 + margin * reference_filter_fwhm_cm1,
    )
    nu = np.linspace(lo, hi, n_integration_points)

    path_cm = path_length_m * 100.0
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

    absorbance = alpha * path_cm

    radiance = planck_spectral_radiance(nu, source_temperature_K)
    active_filt = bandpass_filter(
        nu,
        active_filter_center_cm1,
        active_filter_fwhm_cm1,
        filter_shape,
    )
    ref_filt = bandpass_filter(
        nu,
        reference_filter_center_cm1,
        reference_filter_fwhm_cm1,
        filter_shape,
    )

    active_sig = ndir_detector_signal(
        nu, absorbance, active_filt, radiance,
    )
    ref_sig = ndir_detector_signal(
        nu, absorbance, ref_filt, radiance,
    )

    return {
        "active_signal": active_sig,
        "reference_signal": ref_sig,
        "ratio": ndir_ratio(active_sig, ref_sig),
        "concentration_ppm": concentration_ppm,
        "nu_cm1": nu,
        "absorbance": absorbance,
        "source_radiance": radiance,
        "active_filter": active_filt,
        "reference_filter": ref_filt,
    }
