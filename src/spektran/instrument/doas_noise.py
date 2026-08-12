"""DOAS instrument noise chain.

Noise sources for Differential Optical Absorption Spectroscopy:

- Photon noise: shot noise on UV/Vis detector (CCD/photodiode)
- Stray light: broadband offset from scattered light in spectrograph
- Ring effect: rotational Raman filling-in of Fraunhofer lines
  (Chance & Spurr, 1997, doi:10.1029/97GL00237)
- Wavelength shift: spectrograph wavelength calibration drift
- Dark current: CCD/photodiode thermal dark current
- Readout noise: CCD readout electronics noise

References:
- U. Platt and J. Stutz, "Differential Optical Absorption Spectroscopy",
  Springer (2008), doi:10.1007/978-3-540-75776-4
- K.P. Chance and R.J.D. Spurr, "Ring effect studies",
  Geophys. Res. Lett. 24 (1997) 3012, doi:10.1029/97GL00237
"""

from __future__ import annotations

import numpy as np


def photon_noise(
    rng: np.random.Generator,
    signal: np.ndarray,
    photon_count_ref: float = 1e6,
) -> np.ndarray:
    """Shot noise on UV/Vis detector.

    sigma = signal / sqrt(N_photons)
    Returns noise array (additive to optical density).
    """
    sigma = np.abs(signal) / np.sqrt(photon_count_ref)
    sigma = np.maximum(sigma, 1e-20)
    return rng.normal(0.0, sigma)


def stray_light(
    rng: np.random.Generator,
    n_points: int,
    stray_fraction: float = 1e-4,
) -> np.ndarray:
    """Broadband stray light offset in the spectrograph.

    Adds a small spectrally-smooth offset to the measured spectrum.
    """
    base = rng.normal(0, 1, n_points)
    kernel_size = max(n_points // 10, 3)
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(base, kernel, mode="same")
    return stray_fraction * smooth / (np.std(smooth) + 1e-20)


def ring_effect(
    wavelength_nm: np.ndarray,
    amplitude: float = 0.02,
    filling_width_nm: float = 0.5,
    n_lines: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simplified Ring effect: rotational Raman filling-in.

    Models filling-in of solar Fraunhofer lines by inelastic
    scattering. Creates pseudo-absorption features that interfere
    with DOAS retrieval.
    """
    ring = np.zeros_like(wavelength_nm)
    wl_range = wavelength_nm[-1] - wavelength_nm[0]

    if rng is not None:
        positions = rng.uniform(wavelength_nm[0], wavelength_nm[-1], n_lines)
    else:
        positions = np.linspace(
            wavelength_nm[0] + 0.1 * wl_range,
            wavelength_nm[-1] - 0.1 * wl_range,
            n_lines,
        )

    for pos in positions:
        ring += np.exp(-0.5 * ((wavelength_nm - pos) / filling_width_nm) ** 2)

    ring = amplitude * ring / (np.max(ring) + 1e-20)
    return ring


def wavelength_shift(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    shift_nm: float = 0.01,
    squeeze: float = 0.0,
) -> np.ndarray:
    """Apply wavelength calibration shift and squeeze.

    Models spectrograph drift: lambda_true = lambda_meas + shift + squeeze * lambda_meas
    Returns the spectrum on the shifted grid (interpolated back).
    """
    wl_shifted = wavelength_nm + shift_nm + squeeze * (wavelength_nm - np.mean(wavelength_nm))
    return np.interp(wavelength_nm, wl_shifted, spectrum)


def dark_current_noise(
    rng: np.random.Generator,
    n_points: int,
    dark_sigma: float = 1e-5,
) -> np.ndarray:
    """CCD/photodiode dark current noise (white Gaussian)."""
    return rng.normal(0.0, dark_sigma, n_points)


def readout_noise(
    rng: np.random.Generator,
    n_points: int,
    readout_sigma: float = 5e-5,
) -> np.ndarray:
    """CCD readout electronics noise."""
    return rng.normal(0.0, readout_sigma, n_points)
