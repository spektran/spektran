"""Spectral line-shape functions: Doppler (Gaussian), Lorentz, Voigt.

All profiles are area-normalized: integral over wavenumber [cm-1] equals 1,
so profile values carry units of cm.

Main Voigt implementation uses the Faddeeva function ``w(z)`` via
``scipy.special.wofz`` (Poppe & Wijers algorithm; see also Humlicek's
rational approximations). References:

- F. Schreier, "Optimized implementations of rational approximations for the
  Voigt and complex error function", JQSRT 112 (2011) 1010,
  doi:10.1016/j.jqsrt.2010.12.010
- J. Humlicek, "Optimized computation of the Voigt and complex probability
  functions", JQSRT 27 (1982) 437, doi:10.1016/0022-4073(82)90078-4

An independent numerical-convolution reference implementation lives in
``tests/reference_impl/ref_lineshape.py`` (Gate G3); the two must never share
code.
"""

from __future__ import annotations

import numpy as np
from scipy.special import wofz

from .constants import AMU_G, C_CM_PER_S, K_ERG_PER_K

_SQRT_LN2 = np.sqrt(np.log(2.0))
_SQRT_2LN2 = np.sqrt(2.0 * np.log(2.0))
_SQRT_PI = np.sqrt(np.pi)
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def doppler_hwhm_cm1(nu0_cm1: float, temperature_K: float, molar_mass_amu: float) -> float:
    """Doppler (Gaussian) half-width at half-maximum [cm-1].

    alpha_D = (nu0/c) * sqrt(2 ln2 kT / m)

    Standard thermal-Doppler broadening; see e.g. W. Demtroeder, "Laser
    Spectroscopy" 5th ed., Springer (2014), doi:10.1007/978-3-642-53859-9
    (its Eq. (3.43) gives the FWHM; this function returns the HWHM, i.e.
    half that value).
    """
    m_g = molar_mass_amu * AMU_G
    return (nu0_cm1 / C_CM_PER_S) * np.sqrt(
        2.0 * np.log(2.0) * K_ERG_PER_K * temperature_K / m_g
    )


def lorentz_hwhm_cm1(
    pressure_atm: float,
    temperature_K: float,
    gamma_air_cm1_per_atm: float,
    gamma_self_cm1_per_atm: float,
    mole_fraction: float,
    n_air: float,
    T_ref_K: float = 296.0,
) -> float:
    """Pressure-broadened Lorentzian HWHM [cm-1] per the HITRAN convention.

    gamma(P, T) = (T_ref/T)^n_air * (gamma_air * P_partial_foreign
                                     + gamma_self * P_partial_self)

    Reference: L.S. Rothman et al., "The HITRAN 2008 molecular spectroscopic
    database", JQSRT 110 (2009) 533, Eq. (6), doi:10.1016/j.jqsrt.2009.02.013
    """
    p_self = mole_fraction * pressure_atm
    p_foreign = pressure_atm - p_self
    return (T_ref_K / temperature_K) ** n_air * (
        gamma_air_cm1_per_atm * p_foreign + gamma_self_cm1_per_atm * p_self
    )


def gaussian_profile(nu_cm1: np.ndarray, nu0_cm1: float, hwhm_cm1: float) -> np.ndarray:
    """Area-normalized Gaussian (Doppler) profile [cm]."""
    sigma = hwhm_cm1 / _SQRT_2LN2
    x = (nu_cm1 - nu0_cm1) / sigma
    return np.exp(-0.5 * x * x) / (sigma * _SQRT_2PI)


def lorentz_profile(nu_cm1: np.ndarray, nu0_cm1: float, hwhm_cm1: float) -> np.ndarray:
    """Area-normalized Lorentzian profile [cm]."""
    d = nu_cm1 - nu0_cm1
    return (hwhm_cm1 / np.pi) / (d * d + hwhm_cm1 * hwhm_cm1)


def voigt_profile(
    nu_cm1: np.ndarray,
    nu0_cm1: float,
    doppler_hwhm: float,
    lorentz_hwhm: float,
) -> np.ndarray:
    """Area-normalized Voigt profile [cm] via the Faddeeva function.

    phi_V(nu) = Re[w(z)] / (sigma_G * sqrt(2*pi)),
    z = ((nu - nu0) + i*gamma_L) / (sigma_G * sqrt(2))

    where sigma_G = doppler_hwhm / sqrt(2 ln2). See module docstring for
    references. Degenerates gracefully: for doppler_hwhm -> 0 use
    ``lorentz_profile``; for lorentz_hwhm = 0 this reduces to the Gaussian.
    """
    if doppler_hwhm <= 0.0:
        if lorentz_hwhm <= 0.0:
            raise ValueError("At least one of doppler_hwhm, lorentz_hwhm must be > 0")
        return lorentz_profile(nu_cm1, nu0_cm1, lorentz_hwhm)
    sigma = doppler_hwhm / _SQRT_2LN2
    z = ((nu_cm1 - nu0_cm1) + 1j * lorentz_hwhm) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * _SQRT_2PI)
