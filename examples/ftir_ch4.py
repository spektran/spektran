"""Example: simulate an FTIR measurement — interferogram -> apodized spectrum.

Demonstrates the FTIR forward chain: HITRAN absorption -> high-res spectrum
-> interferogram -> truncate at max OPD -> apodize -> FFT -> ILS-broadened spectrum.
"""

import numpy as np
from spektran.physics.hitran import demo_ch4_2nu3
from spektran.physics.ftir import (
    simulate_ftir_spectrum,
    spectral_resolution_cm1,
    apodization_function,
)

lines = demo_ch4_2nu3()

result = simulate_ftir_spectrum(
    lines=lines,
    molecule="CH4",
    concentration_ppm=500.0,
    temperature_K=296.0,
    pressure_atm=1.0,
    path_length_m=10.0,
    max_opd_cm=10.0,
    wavenumber_start_cm1=6045.0,
    wavenumber_end_cm1=6049.0,
    n_hires_points=10000,
    n_output_points=500,
    apod_function="happ_genzel",
)

res = spectral_resolution_cm1(10.0)
print(f"Spectral resolution: {res:.3f} cm^-1 (max OPD = 10 cm)")
print(f"Output grid: {len(result['nu_cm1'])} points, "
      f"{result['nu_cm1'][0]:.1f}--{result['nu_cm1'][-1]:.1f} cm^-1")

hires = result["spectrum_hires"]
print(f"High-res transmittance range: {np.min(hires):.4f} -- {np.max(hires):.4f}")
print(f"Peak absorption depth: {1 - np.min(hires):.4f}")

apod_types = ["boxcar", "triangular", "happ_genzel", "norton_beer_medium", "norton_beer_strong"]
print(f"\nAvailable apodization functions: {apod_types}")
