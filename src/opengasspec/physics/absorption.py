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

from .constants import C2_CM_K, T_REF_K, number_density_cm3
from .hitran import LineList, demo_ch4_2nu3
from .lineshape import doppler_hwhm_cm1, lorentz_hwhm_cm1, voigt_profile

# Default partition-function ratio approximation Q(T_ref)/Q(T) ~ (T_ref/T)^m.
# m = 3/2 for nonlinear polyatomics (rotational partition function; classical
# limit), m = 1 for linear molecules. Adequate near room temperature; official
# generation injects hapi's TIPS partition sums via `q_ratio` for accuracy.
_LINEAR_MOLECULES = {"CO2", "N2O", "CO", "O2"}


def default_q_ratio(molecule: str, temperature_K: float, T_ref_K: float = T_REF_K) -> float:
    """Approximate partition-sum ratio Q(T_ref)/Q(T) by a power law.

    Exact at T = T_ref (ratio = 1), which is why HITRAN-comparison tests pin
    T = 296 K. For production-quality temperature scaling pass a TIPS-based
    ``q_ratio`` callable to :func:`absorption_coefficient` (see hapi's
    partitionSum; R.R. Gamache et al., JQSRT 203 (2017) 70,
    doi:10.1016/j.jqsrt.2017.03.045).
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
) -> np.ndarray:
    """Absorption coefficient alpha(nu) [cm-1] for one absorber.

    alpha(nu) = n_total * x * sum_j S_j(T) * phi_V(nu; nu0_j', alpha_D_j, gamma_L_j)

    with pressure-shifted line centers nu0' = nu0 + delta_air * P.
    """
    if not 0.0 <= mole_fraction <= 1.0:
        raise ValueError(f"mole_fraction must be in [0, 1], got {mole_fraction}")
    q = (q_ratio or default_q_ratio)(lines.molecule, temperature_K)
    strengths = line_strength_at_T(
        lines.sw_cm_per_molec, lines.nu0_cm1, lines.elower_cm1, temperature_K, q
    )
    n_absorber = number_density_cm3(pressure_atm, temperature_K) * mole_fraction

    alpha = np.zeros_like(nu_cm1, dtype=np.float64)
    for j in range(len(lines)):
        nu0_shifted = lines.nu0_cm1[j] + lines.delta_air[j] * pressure_atm
        a_d = doppler_hwhm_cm1(nu0_shifted, temperature_K, lines.molar_mass_amu)
        g_l = lorentz_hwhm_cm1(
            pressure_atm,
            temperature_K,
            lines.gamma_air[j],
            lines.gamma_self[j],
            mole_fraction,
            lines.n_air[j],
        )
        alpha += strengths[j] * voigt_profile(nu_cm1, nu0_shifted, a_d, g_l)
    return n_absorber * alpha


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
