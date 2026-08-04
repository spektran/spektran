"""REFERENCE implementation of the Beer-Lambert forward model — Gate G3.

Recomputes absorbance at a single wavenumber from HITRAN line parameters using
its own code path: scalar math, quadrature-based Voigt (ref_lineshape), and an
independently written transcription of the HITRAN equations (Rothman et al.,
JQSRT 110 (2009) 533, Appendix A).

MUST NOT import from or share code with ``opensensorsim``.
"""

from __future__ import annotations

import math

from .ref_lineshape import voigt_profile_ref

# Independent constant definitions (CODATA 2018 exact values)
_C_CM_S = 2.99792458e10
_K_ERG_K = 1.380649e-16
_H_ERG_S = 6.62607015e-27
_C2 = _H_ERG_S * _C_CM_S / _K_ERG_K
_AMU_G = 1.66053906660e-24
_ATM_CGS = 1.01325e6
_TREF = 296.0


def absorbance_ref(
    nu_cm1: float,
    *,
    nu0_cm1: float,
    sw_ref: float,
    gamma_air: float,
    gamma_self: float,
    n_air: float,
    delta_air: float,
    elower_cm1: float,
    molar_mass_amu: float,
    mole_fraction: float,
    temperature_K: float,
    pressure_atm: float,
    path_length_cm: float,
    q_ratio_value: float,
) -> float:
    """Napierian absorbance of a single line at a single wavenumber."""
    T, P = temperature_K, pressure_atm

    # Line intensity temperature scaling (Rothman 2009, Eq. A11)
    s = (
        sw_ref
        * q_ratio_value
        * math.exp(-_C2 * elower_cm1 / T)
        / math.exp(-_C2 * elower_cm1 / _TREF)
        * (1.0 - math.exp(-_C2 * nu0_cm1 / T))
        / (1.0 - math.exp(-_C2 * nu0_cm1 / _TREF))
    )

    # Pressure shift and widths
    nu0 = nu0_cm1 + delta_air * P
    p_self = mole_fraction * P
    gamma_l = (_TREF / T) ** n_air * (gamma_air * (P - p_self) + gamma_self * p_self)
    alpha_d = (nu0 / _C_CM_S) * math.sqrt(
        2.0 * math.log(2.0) * _K_ERG_K * T / (molar_mass_amu * _AMU_G)
    )

    # Number density of the absorber
    n_abs = mole_fraction * P * _ATM_CGS / (_K_ERG_K * T)

    phi = voigt_profile_ref(nu_cm1, nu0, alpha_d, gamma_l)
    return s * phi * n_abs * path_length_cm
