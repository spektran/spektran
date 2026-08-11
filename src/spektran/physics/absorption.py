"""Beer-Lambert forward model: line strengths, absorption coefficient, absorbance.

Implements the standard HITRAN line-by-line forward calculation:

    A(nu) = sum_j S_j(T) * phi_j(nu) * n * x * L        (napierian absorbance)

with S_j(T) the temperature-scaled line intensity, phi_j the area-normalized
Voigt profile, n the total number density, x the absorber mole fraction and L
the path length in cm.

References:
- L.S. Rothman et al., "The HITRAN 2008 molecular spectroscopic database",
  JQSRT 110 (2009) 533, Appendix A, doi:10.1016/j.jqsrt.2009.02.013
- I.E. Gordon et al., "The HITRAN2020 molecular spectroscopic database",
  JQSRT 277 (2022) 107949, doi:10.1016/j.jqsrt.2021.107949

An independent reference implementation (different code path) lives in
``tests/reference_impl/ref_absorption.py`` for Gate G3 cross-validation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import wofz

from .constants import C2_CM_K, T_REF_K, number_density_cm3
from .hitran import LineList, demo_ch4_2nu3
from .lineshape import doppler_hwhm_cm1, lorentz_hwhm_cm1
from .lineshape_htp import htp_profile
from .tips import tips_q_ratio

_SQRT_2LN2 = np.sqrt(2.0 * np.log(2.0))
_SQRT_2PI = np.sqrt(2.0 * np.pi)

# Default partition-function ratio approximation Q(T_ref)/Q(T) ~ (T_ref/T)^m.
# m = 3/2 for nonlinear polyatomics (rotational partition function; classical
# limit), m = 1 for linear molecules. Exact only at T = T_ref; kept for
# backward compatibility and as the accuracy baseline that `tips_q_ratio`
# (see tips.py) is benchmarked against in tests/test_tips.py. Superseded as
# the default by `tips_q_ratio` below.
_LINEAR_MOLECULES = {"CO2", "N2O", "CO", "O2", "NO", "HCl", "HF"}


def default_q_ratio(molecule: str, temperature_K: float, T_ref_K: float = T_REF_K) -> float:
    """Approximate partition-sum ratio Q(T_ref)/Q(T) by a power law.

    Exact at T = T_ref (ratio = 1), which is why HITRAN-comparison tests pin
    T = 296 K. Kept for backward compatibility; :func:`absorption_coefficient`
    now defaults to the more accurate :func:`tips.tips_q_ratio` (R.R. Gamache
    et al., JQSRT 203 (2017) 70, doi:10.1016/j.jqsrt.2017.03.045) unless a
    ``q_ratio`` callable is passed explicitly.
    """
    m = 1.0 if molecule in _LINEAR_MOLECULES else 1.5
    return (T_ref_K / temperature_K) ** m


def line_strength_at_T(
    sw_ref: np.ndarray,
    nu0_cm1: np.ndarray,
    elower_cm1: np.ndarray,
    temperature_K: float,
    q_ratio_value: float,
    T_ref_K: float = T_REF_K,
) -> np.ndarray:
    """Scale HITRAN line intensities from 296 K to temperature T.

    S(T) = S(T_ref) * [Q(T_ref)/Q(T)]
           * exp(-c2 E''/T) / exp(-c2 E''/T_ref)
           * [1 - exp(-c2 nu0/T)] / [1 - exp(-c2 nu0/T_ref)]

    Standard HITRAN intensity-scaling relation; originally formulated in the
    HITRAN/HAWKS appendix (L.S. Rothman et al., JQSRT 60 (1998) 665,
    doi:10.1016/S0022-4073(98)00078-8) and restated in later editions
    (Rothman et al., JQSRT 110 (2009) 533, doi:10.1016/j.jqsrt.2009.02.013).
    """
    boltzmann = np.exp(-C2_CM_K * elower_cm1 / temperature_K) / np.exp(
        -C2_CM_K * elower_cm1 / T_ref_K
    )
    stim_emission = (1.0 - np.exp(-C2_CM_K * nu0_cm1 / temperature_K)) / (
        1.0 - np.exp(-C2_CM_K * nu0_cm1 / T_ref_K)
    )
    return sw_ref * q_ratio_value * boltzmann * stim_emission


def absorption_coefficient(
    nu_cm1: np.ndarray,
    lines: LineList,
    mole_fraction: float,
    temperature_K: float,
    pressure_atm: float,
    q_ratio: Callable[[str, float], float] | None = None,
    wing_cutoff_cm1: float = 0.0,
    line_profile: str = "voigt",
) -> np.ndarray:
    """Absorption coefficient alpha(nu) [cm-1] for one absorber.

    alpha(nu) = n_total * x * sum_j S_j(T) * phi_j(nu; ...)

    with pressure-shifted line centers nu0' = nu0 + delta_air * P.

    Parameters
    ----------
    wing_cutoff_cm1 : float
        When > 0, zero out contributions where |nu - nu0'| exceeds this
        value.  Accelerates computation for large line lists by masking
        negligible far-wing contributions.  Default 0.0 (no cutoff).
    line_profile : str
        Line-shape model: ``"voigt"`` (default) or ``"htp"`` (Hartmann-Tran
        Profile, HITRAN2016+ recommendation).  ``"htp"`` requires that
        ``lines.has_htp_params`` is True.  Ngo et al., JQSRT 129 (2013) 89,
        doi:10.1016/j.jqsrt.2013.05.034
    """
    if not 0.0 <= mole_fraction <= 1.0:
        raise ValueError(f"mole_fraction must be in [0, 1], got {mole_fraction}")
    if line_profile not in ("voigt", "htp"):
        raise ValueError(f"line_profile must be 'voigt' or 'htp', got {line_profile!r}")
    if len(lines) == 0:
        return np.zeros_like(nu_cm1, dtype=np.float64)

    q = (q_ratio or tips_q_ratio)(lines.molecule, temperature_K)
    strengths = line_strength_at_T(
        lines.sw_cm_per_molec, lines.nu0_cm1, lines.elower_cm1, temperature_K, q
    )
    n_absorber = number_density_cm3(pressure_atm, temperature_K) * mole_fraction

    nu0_shifted = lines.nu0_cm1 + lines.delta_air * pressure_atm
    a_d = doppler_hwhm_cm1(nu0_shifted, temperature_K, lines.molar_mass_amu)
    g_l = lorentz_hwhm_cm1(
        pressure_atm,
        temperature_K,
        lines.gamma_air,
        lines.gamma_self,
        mole_fraction,
        lines.n_air,
    )

    if line_profile == "htp" and lines.has_htp_params:
        alpha = np.zeros_like(nu_cm1, dtype=np.float64)
        for j in range(len(lines)):
            gamma_2_j = lines.gamma_2[j] * pressure_atm
            delta_2_j = lines.delta_2[j] * pressure_atm
            nu_vc_j = lines.nu_vc[j] * pressure_atm
            eta_j = lines.eta[j]
            delta_0_j = lines.delta_air[j] * pressure_atm

            phi_j = htp_profile(
                nu_cm1, nu0_shifted[j], a_d[j], g_l[j], delta_0_j,
                gamma_2=gamma_2_j, delta_2=delta_2_j,
                nu_vc=nu_vc_j, eta=eta_j,
            )
            if wing_cutoff_cm1 > 0.0:
                phi_j[np.abs(nu_cm1 - nu0_shifted[j]) > wing_cutoff_cm1] = 0.0
            alpha += strengths[j] * phi_j
    else:
        sigma = a_d / _SQRT_2LN2
        delta_nu = nu_cm1[np.newaxis, :] - nu0_shifted[:, np.newaxis]
        z = (delta_nu + 1j * g_l[:, np.newaxis]) / (sigma[:, np.newaxis] * np.sqrt(2.0))
        phi = np.real(wofz(z)) / (sigma[:, np.newaxis] * _SQRT_2PI)

        if wing_cutoff_cm1 > 0.0:
            phi[np.abs(delta_nu) > wing_cutoff_cm1] = 0.0

        alpha = np.sum(strengths[:, np.newaxis] * phi, axis=0)

    return n_absorber * alpha


def rosenkranz_line_mixing(
    nu_cm1: np.ndarray,
    lines: LineList,
    temperature_K: float,
    pressure_atm: float,
    mole_fraction: float,
    y_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """First-order Rosenkranz line-mixing correction to absorption coefficient.

    Line mixing arises from collisional transfer of population between
    rotational states. The first-order (pressure-linear) approximation
    adds an antisymmetric dispersive component to each line:

        alpha_mix(nu) += n * x * sum_j S_j * Y_j * P * psi_j(nu)

    where psi_j is the dispersive (imaginary part of Faddeeva) counterpart
    of the Voigt profile phi_j, and Y_j is the first-order mixing coefficient
    [cm-1/atm]. Rosenkranz (1975) doi:10.1109/TAP.1975.1141105

    When ``y_coeffs`` is None, uses a simple empirical scaling of the
    air-broadening coefficient as a rough estimate.
    """
    if len(lines) == 0:
        return np.zeros_like(nu_cm1, dtype=np.float64)

    q = tips_q_ratio(lines.molecule, temperature_K)
    strengths = line_strength_at_T(
        lines.sw_cm_per_molec, lines.nu0_cm1, lines.elower_cm1, temperature_K, q
    )
    n_absorber = number_density_cm3(pressure_atm, temperature_K) * mole_fraction

    nu0_shifted = lines.nu0_cm1 + lines.delta_air * pressure_atm
    a_d = doppler_hwhm_cm1(nu0_shifted, temperature_K, lines.molar_mass_amu)
    g_l = lorentz_hwhm_cm1(
        pressure_atm, temperature_K,
        lines.gamma_air, lines.gamma_self, mole_fraction, lines.n_air,
    )

    if y_coeffs is None:
        y_coeffs = -0.01 * lines.gamma_air

    sigma = a_d / _SQRT_2LN2
    delta_nu = nu_cm1[np.newaxis, :] - nu0_shifted[:, np.newaxis]
    z = (delta_nu + 1j * g_l[:, np.newaxis]) / (sigma[:, np.newaxis] * np.sqrt(2.0))
    w = wofz(z)
    psi = np.imag(w) / (sigma[:, np.newaxis] * _SQRT_2PI)

    mixing_contrib = np.sum(
        strengths[:, np.newaxis] * y_coeffs[:, np.newaxis] * pressure_atm * psi, axis=0
    )
    return n_absorber * mixing_contrib


def simulate_absorbance(
    molecule: str = "CH4",
    concentration_ppm: float = 100.0,
    temperature_K: float = 296.0,
    pressure_atm: float = 1.0,
    path_length_m: float = 10.0,
    wavenumber_start_cm1: float = 6046.0,
    wavenumber_end_cm1: float = 6048.0,
    n_points: int = 2000,
    lines: LineList | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a clean direct-absorption napierian absorbance spectrum.

    Returns ``(nu_cm1, absorbance)``. If ``lines`` is None, uses the built-in
    approximate CH4 demo line list (offline); pass a hapi-fetched
    :class:`LineList` for authoritative results.
    """
    if lines is None:
        if molecule != "CH4":
            raise ValueError("Built-in demo lines exist only for CH4; pass `lines`")
        lines = demo_ch4_2nu3()
    nu = np.linspace(wavenumber_start_cm1, wavenumber_end_cm1, n_points)
    alpha = absorption_coefficient(
        nu,
        lines,
        mole_fraction=concentration_ppm * 1e-6,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        q_ratio=None,
    )
    path_cm = path_length_m * 100.0
    return nu, alpha * path_cm
