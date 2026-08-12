"""DOAS forward model: Differential Optical Absorption Spectroscopy.

UV/Vis absorption spectroscopy that separates narrow-band molecular
features from broadband extinction (Rayleigh/Mie scattering, instrument
response). The core operation is polynomial high-pass filtering of
optical density to extract differential structures.

Forward model chain::

    UV/Vis cross sections sigma(lambda)
      -> Beer-Lambert: OD = sigma * n * L
        -> Add broadband: Rayleigh scattering, Mie aerosol, instrument response
          -> Total optical density
            -> High-pass filter (polynomial subtraction)
              -> Differential optical density (DOAS signal)

Key physics:

1. **Beer-Lambert in UV/Vis**: I = I0 * exp(-sum_j sigma_j(lambda) * N_j * L)
   where sigma_j is the absorption cross-section [cm^2/molecule],
   N_j is the number density [molecules/cm^3], and L is path length.

2. **Differential formulation**: sigma(lambda) = sigma_slow(lambda) + sigma_diff(lambda)
   The slow-varying part is absorbed into a polynomial; only sigma_diff remains.

3. **Rayleigh scattering**: sigma_R(lambda) ~ lambda^(-4), Bodhaine et al. (1999).

4. **Mie scattering**: sigma_M(lambda) ~ lambda^(-alpha), alpha = 0.5-2.0
   (Angstrom exponent).

References:
- U. Platt and J. Stutz, "Differential Optical Absorption Spectroscopy:
  Principles and Applications", Springer (2008), doi:10.1007/978-3-540-75776-4
- T. Wagner et al., "MAX-DOAS observations of tropospheric trace gases",
  Atmos. Meas. Tech. 4 (2011) 655, doi:10.5194/amt-4-655-2011
- B.A. Bodhaine et al., "On Rayleigh Optical Depth Calculations",
  J. Atmos. Ocean. Tech. 16 (1999) 1854,
  doi:10.1175/1520-0426(1999)016<1854:ORODC>2.0.CO;2
"""

from __future__ import annotations

import numpy as np

from .constants import K_ERG_PER_K


def number_density(
    concentration_ppm: float,
    temperature_K: float,
    pressure_atm: float,
) -> float:
    """Convert mixing ratio [ppm] to number density [molecules/cm^3].

    n = x * P / (k_B * T), using ideal gas law.
    """
    x = concentration_ppm * 1e-6
    P_dyn = pressure_atm * 1.01325e6  # atm -> dyne/cm^2
    return x * P_dyn / (K_ERG_PER_K * temperature_K)


def rayleigh_cross_section(wavelength_nm: np.ndarray) -> np.ndarray:
    """Rayleigh scattering cross section [cm^2/molecule].

    Simplified Bodhaine et al. (1999) for dry air at STP.
    sigma_R(lambda) = 4.02e-28 / lambda_cm^4  (approximate)
    """
    lambda_cm = wavelength_nm * 1e-7
    return 4.02e-28 / lambda_cm**4


def mie_extinction(
    wavelength_nm: np.ndarray,
    tau_ref: float = 0.1,
    lambda_ref_nm: float = 550.0,
    angstrom_exp: float = 1.3,
) -> np.ndarray:
    """Mie aerosol extinction optical depth per km.

    tau_mie(lambda) = tau_ref * (lambda / lambda_ref)^(-alpha)

    Returns optical depth per km of path.
    """
    return tau_ref * (wavelength_nm / lambda_ref_nm) ** (-angstrom_exp)


def simulate_doas_cross_section(
    wavelength_nm: np.ndarray,
    center_nm: float = 330.0,
    peak_cross_section_cm2: float = 6e-19,
    n_features: int = 5,
    feature_width_nm: float = 0.8,
    feature_spacing_nm: float = 1.5,
    broad_width_nm: float = 30.0,
) -> np.ndarray:
    """Generate a synthetic UV/Vis absorption cross section.

    Combines a broad Gaussian envelope (representing the slow-varying part)
    with narrow differential features. This mimics the structure of real
    UV/Vis cross sections (e.g., SO2 Huggins band, NO2 visible).
    """
    broad = np.exp(-0.5 * ((wavelength_nm - center_nm) / broad_width_nm) ** 2)

    diff = np.zeros_like(wavelength_nm)
    for i in range(n_features):
        offset = (i - n_features / 2) * feature_spacing_nm
        diff += (0.5 + 0.5 * np.cos(np.pi * i / n_features)) * np.exp(
            -0.5 * ((wavelength_nm - center_nm - offset) / feature_width_nm) ** 2
        )

    sigma = peak_cross_section_cm2 * (0.3 * broad + 0.7 * diff / max(np.max(diff), 1e-30))
    return sigma


def doas_optical_density(
    wavelength_nm: np.ndarray,
    cross_sections: list[dict],
    temperature_K: float = 296.0,
    pressure_atm: float = 1.0,
    path_length_m: float = 1000.0,
    rayleigh: bool = True,
    mie_tau_ref: float = 0.0,
    mie_angstrom: float = 1.3,
) -> dict:
    """Compute total and differential optical density for DOAS.

    Parameters
    ----------
    cross_sections : list of dict
        Each dict has keys: 'sigma_cm2' (array), 'concentration_ppm', 'molecule'.

    Returns
    -------
    dict with keys:
        'wavelength_nm', 'od_total', 'od_differential', 'od_molecular',
        'od_rayleigh', 'od_mie', 'concentration_ppm' (of first species).
    """
    path_cm = path_length_m * 100.0
    n_total = pressure_atm * 1.01325e6 / (K_ERG_PER_K * temperature_K)

    od_molecular = np.zeros_like(wavelength_nm)
    for cs in cross_sections:
        N = number_density(cs["concentration_ppm"], temperature_K, pressure_atm)
        od_molecular = od_molecular + cs["sigma_cm2"] * N * path_cm

    od_rayleigh_arr = np.zeros_like(wavelength_nm)
    if rayleigh:
        sigma_r = rayleigh_cross_section(wavelength_nm)
        od_rayleigh_arr = sigma_r * n_total * path_cm

    od_mie_arr = np.zeros_like(wavelength_nm)
    if mie_tau_ref > 0:
        mie_per_km = mie_extinction(
            wavelength_nm, mie_tau_ref, angstrom_exp=mie_angstrom,
        )
        od_mie_arr = mie_per_km * (path_length_m / 1000.0)

    od_total = od_molecular + od_rayleigh_arr + od_mie_arr

    return {
        "wavelength_nm": wavelength_nm,
        "od_total": od_total,
        "od_molecular": od_molecular,
        "od_rayleigh": od_rayleigh_arr,
        "od_mie": od_mie_arr,
    }


def polynomial_high_pass(
    od: np.ndarray,
    poly_order: int = 5,
) -> np.ndarray:
    """Remove broadband structure via polynomial subtraction.

    This is the core DOAS operation: fit and subtract a polynomial
    to isolate the differential (narrow-band) absorption features.
    """
    x = np.linspace(-1, 1, len(od))
    coeffs = np.polyfit(x, od, poly_order)
    broadband = np.polyval(coeffs, x)
    return od - broadband


def simulate_doas_spectrum(
    wavelength_nm: np.ndarray,
    target_sigma_cm2: np.ndarray,
    target_concentration_ppm: float,
    temperature_K: float = 296.0,
    pressure_atm: float = 1.0,
    path_length_m: float = 1000.0,
    poly_order: int = 5,
    rayleigh: bool = True,
    mie_tau_ref: float = 0.1,
    mie_angstrom: float = 1.3,
    interferent_sigmas: list[dict] | None = None,
) -> dict:
    """Simulate a complete DOAS measurement.

    Full forward chain: cross sections → Beer-Lambert → broadband extinction
    → polynomial high-pass → differential optical density.

    Returns dict with keys: 'wavelength_nm', 'doas_spectrum' (differential OD),
    'od_total', 'od_molecular', 'transmittance', 'concentration_ppm'.
    """
    cross_sections = [{
        "sigma_cm2": target_sigma_cm2,
        "concentration_ppm": target_concentration_ppm,
        "molecule": "target",
    }]

    if interferent_sigmas:
        cross_sections.extend(interferent_sigmas)

    result = doas_optical_density(
        wavelength_nm=wavelength_nm,
        cross_sections=cross_sections,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        path_length_m=path_length_m,
        rayleigh=rayleigh,
        mie_tau_ref=mie_tau_ref,
        mie_angstrom=mie_angstrom,
    )

    doas_signal = polynomial_high_pass(result["od_total"], poly_order)

    transmittance = np.exp(-result["od_total"])

    return {
        "wavelength_nm": wavelength_nm,
        "doas_spectrum": doas_signal,
        "od_total": result["od_total"],
        "od_molecular": result["od_molecular"],
        "od_rayleigh": result["od_rayleigh"],
        "od_mie": result["od_mie"],
        "transmittance": transmittance,
        "concentration_ppm": target_concentration_ppm,
        "poly_order": poly_order,
    }
