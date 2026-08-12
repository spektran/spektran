"""FTIR instrument noise chain.

Noise sources for Fourier Transform Infrared spectroscopy:

- Detector noise: NEP-limited, frequency-dependent (InSb/MCT/InGaAs)
- Source intensity fluctuation: 1/f + white
- Phase error: random phase jitter on interferogram from alignment quality
- Channel spectra: etalon from beamsplitter/windows
- Sampling position error: OPD axis jitter from HeNe laser stability
- Self-apodization: off-axis angle effect on ILS

References:
- P.R. Griffiths and J.A. de Haseth, "Fourier Transform Infrared
  Spectrometry", 2nd ed., Wiley (2007), doi:10.1002/047010631X
- F. Hase et al., "An empirical line-by-line model for the infrared
  solar transmittance spectrum", JQSRT 72 (2002) 45,
  doi:10.1016/S0022-4073(01)00055-7
"""

from __future__ import annotations

import numpy as np


def detector_noise(
    rng: np.random.Generator,
    n_points: int,
    noise_sigma_rel: float = 1e-4,
) -> np.ndarray:
    """Additive detector noise (white Gaussian, relative to signal peak)."""
    return rng.normal(0.0, noise_sigma_rel, n_points)


def source_fluctuation(
    rng: np.random.Generator,
    n_points: int,
    fluctuation_sigma: float = 1e-4,
    one_over_f_fraction: float = 0.3,
) -> np.ndarray:
    """Source intensity fluctuation: white + 1/f components.

    Returns multiplicative factor (1 + noise).
    """
    white = rng.normal(0.0, fluctuation_sigma * (1 - one_over_f_fraction), n_points)

    if one_over_f_fraction > 0 and n_points > 1:
        fft_white = np.fft.rfft(rng.normal(0, 1, n_points))
        freqs = np.fft.rfftfreq(n_points)
        freqs[0] = 1.0
        fft_pink = fft_white / np.sqrt(freqs)
        pink = np.fft.irfft(fft_pink, n=n_points)
        pink = pink / (np.std(pink) + 1e-20) * fluctuation_sigma * one_over_f_fraction
    else:
        pink = np.zeros(n_points)

    return white + pink


def phase_error(
    rng: np.random.Generator,
    n_points: int,
    phase_sigma_rad: float = 0.01,
) -> np.ndarray:
    """Random phase error on interferogram from alignment imperfections.

    Hase et al. (2002). Adds asymmetric distortion to interferogram
    that converts to spectral artifacts after FFT.
    """
    return rng.normal(0.0, phase_sigma_rad, n_points)


def channel_spectrum(
    nu_cm1: np.ndarray,
    amplitude_rel: float = 1e-3,
    fsr_cm1: float = 5.0,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Parasitic channel spectrum from beamsplitter/window etalon.

    Sinusoidal modulation of the spectrum from multiple reflections
    in plane-parallel optical elements.
    """
    return amplitude_rel * np.cos(2.0 * np.pi * nu_cm1 / fsr_cm1 + phase_rad)


def sampling_error(
    rng: np.random.Generator,
    opd_cm: np.ndarray,
    jitter_sigma_cm: float = 1e-7,
) -> np.ndarray:
    """OPD sampling position jitter from HeNe reference laser instability.

    Returns perturbed OPD array.
    """
    jitter = rng.normal(0.0, jitter_sigma_cm, len(opd_cm))
    return opd_cm + jitter


def self_apodization(
    nu_cm1: np.ndarray,
    max_opd_cm: float,
    fov_half_angle_rad: float = 0.01,
) -> np.ndarray:
    """Self-apodization from finite field of view (off-axis rays).

    Off-axis rays see a slightly different OPD, broadening the ILS.
    The effect is a sinc-like envelope modulating the spectrum.
    Yang et al., Sensors 20 (2020) 2181, doi:10.3390/s20082181
    """
    delta_nu = nu_cm1 * fov_half_angle_rad**2 / 2.0
    x = np.pi * delta_nu * max_opd_cm
    sinc_env = np.where(np.abs(x) > 1e-10, np.sin(x) / x, 1.0)
    return sinc_env
