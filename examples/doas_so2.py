"""Example: simulate a DOAS measurement — UV/Vis Beer-Lambert → differential OD.

Demonstrates the DOAS forward chain: synthetic cross section → Beer-Lambert
with Rayleigh/Mie scattering → polynomial high-pass → differential OD.
Uses SO2 Huggins-band-like parameters.
"""

import numpy as np
from spektran.physics.doas import (
    simulate_doas_cross_section,
    simulate_doas_spectrum,
)

wavelength = np.linspace(300.0, 360.0, 500)

sigma = simulate_doas_cross_section(
    wavelength,
    center_nm=330.0,
    peak_cross_section_cm2=6e-19,
    n_features=5,
    feature_width_nm=0.8,
)

result = simulate_doas_spectrum(
    wavelength_nm=wavelength,
    target_sigma_cm2=sigma,
    target_concentration_ppm=1.0,
    temperature_K=296.0,
    pressure_atm=1.0,
    path_length_m=100.0,
    poly_order=5,
    rayleigh=False,
    mie_tau_ref=0.0,
)

print(f"Wavelength range: {wavelength[0]:.0f}–{wavelength[-1]:.0f} nm")
print(f"Total OD range: {np.min(result['od_total']):.3f} – {np.max(result['od_total']):.3f}")
print(f"Differential OD range: {np.min(result['doas_spectrum']):.6f} – "
      f"{np.max(result['doas_spectrum']):.6f}")
print(f"Transmittance range: {np.min(result['transmittance']):.4f} – "
      f"{np.max(result['transmittance']):.4f}")
