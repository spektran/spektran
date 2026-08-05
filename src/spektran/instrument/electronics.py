"""Signal-chain electronics: laser RIN, TIA bandwidth, detector responsivity.

Completes the instrument noise model with three effects commonly missing from
simplified TDLAS simulations:

- Relative Intensity Noise (RIN): multiplicative laser noise from spontaneous
  emission coupling. Flat spectral density below the relaxation oscillation
  frequency.
- Transimpedance amplifier (TIA) bandwidth: low-pass filtering that limits
  signal bandwidth and determines noise-equivalent bandwidth.
- Detector responsivity: wavelength-dependent photocurrent conversion
  efficiency of InGaAs photodiodes (roll-off near cutoff wavelength).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def rin_noise(
    rng: np.random.Generator,
    n: int,
    rin_dBc_Hz: float,
    bandwidth_Hz: float,
    sampling_rate_Hz: float,
) -> np.ndarray:
    """Laser relative intensity noise (multiplicative, zero-mean).

    RIN is specified as a single-sided spectral density in dBc/Hz. The
    total noise variance over a detection bandwidth B is:

        sigma^2 = 10^(RIN_dBc_Hz/10) * B

    Returns a zero-mean array to be ADDED to the normalized intensity
    (i.e., I_noisy = I * (1 + rin_noise(...))).

    Typical values: -120 to -150 dBc/Hz for DFB lasers near threshold,
    -155 to -170 dBc/Hz for well-above-threshold DFBs and VCSELs.

    Reference: K. Petermann, "Laser Diode Modulation and Noise", Springer
    (1991), doi:10.1007/978-94-015-7979-0
    """
    rin_linear = 10.0 ** (rin_dBc_Hz / 10.0)
    sigma = float(np.sqrt(rin_linear * bandwidth_Hz))
    noise = rng.normal(0.0, sigma, n)
    nyquist = sampling_rate_Hz / 2.0
    if bandwidth_Hz < nyquist:
        sos = butter(2, bandwidth_Hz, btype="low", fs=sampling_rate_Hz, output="sos")
        noise = sosfilt(sos, noise)
        std = noise.std()
        if std > 0:
            noise = noise * (sigma / std)
    return noise


def tia_bandwidth_filter(
    signal: np.ndarray,
    bandwidth_Hz: float,
    sampling_rate_Hz: float,
    order: int = 2,
) -> np.ndarray:
    """Apply TIA bandwidth limit (Butterworth low-pass).

    Models the transimpedance amplifier's finite bandwidth, which acts as
    an anti-aliasing filter before the ADC. Typical TIA bandwidths for
    TDLAS: 1-10 MHz for fast scans, 10-100 kHz for slow scans.
    """
    nyquist = sampling_rate_Hz / 2.0
    if bandwidth_Hz >= nyquist:
        return signal
    sos = butter(order, bandwidth_Hz, btype="low", fs=sampling_rate_Hz, output="sos")
    return sosfilt(sos, signal)


def detector_responsivity(
    nu_cm1: np.ndarray,
    cutoff_cm1: float = 5882.0,
    peak_responsivity: float = 1.0,
    rolloff_width_cm1: float = 200.0,
) -> np.ndarray:
    """InGaAs detector responsivity vs wavenumber.

    Models the long-wavelength (low-wavenumber) cutoff of an InGaAs
    photodiode. Responsivity is flat above the cutoff region and drops
    with a sigmoid roll-off below cutoff_cm1. Default cutoff at 5882 cm-1
    corresponds to ~1.7 um (standard InGaAs).

    Returns a multiplicative factor in [0, peak_responsivity].
    """
    x = (nu_cm1 - cutoff_cm1) / rolloff_width_cm1
    return peak_responsivity / (1.0 + np.exp(-x))
