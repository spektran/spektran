"""Wavelength-modulation spectroscopy (WMS): modulation + lock-in demodulation.

Simulates the full WMS signal chain in the time domain, the way a real
instrument works:

1. Optical frequency:  nu(t) = nu_scan(t) + a * cos(2*pi*f_m*t)
2. Laser intensity with residual amplitude modulation (RAM):
       I0(t) = Ibar(t) * (1 + i0*cos(2*pi*f_m*t + psi1)
                            + i2*cos(4*pi*f_m*t + psi2))
3. Transmission through the gas: I(t) = I0(t) * exp(-alpha(nu(t)) * L)
4. Digital lock-in at harmonic n: multiply by cos/sin(2*pi*n*f_m*t + phi),
   low-pass filter -> X_nf(t), Y_nf(t).

Conventions follow the calibration-free WMS literature:

- G.B. Rieker, J.B. Jeffries, R.K. Hanson, "Calibration-free
  wavelength-modulation spectroscopy for measurements of gas temperature and
  concentration in harsh environments", Appl. Opt. 48 (2009) 5546,
  doi:10.1364/AO.48.005546
- K. Sun et al., "Analysis of calibration-free wavelength-scanned wavelength
  modulation spectroscopy for practical gas sensing using tunable diode
  lasers", Meas. Sci. Technol. 24 (2013) 125203,
  doi:10.1088/0957-0233/24/12/125203

The independent Gate G3 reference implementation (Fourier-coefficient
quadrature; Arndt's analytic Lorentzian harmonics for the optically-thin
limit) lives in ``tests/reference_impl/ref_wms.py`` and must not share code
with this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class WMSConfig:
    """Parameters of the WMS chain (record-schema field names mirrored)."""

    modulation_frequency_Hz: float
    modulation_depth_cm1: float
    sampling_rate_Hz: float
    duration_s: float
    center_wavenumber_cm1: float
    scan_range_cm1: float = 0.0  # 0 = fixed center (no slow scan)
    scan_rate_Hz: float = 0.0  # sawtooth ramp rate when scan_range > 0
    im_i0_rel: float = 0.0  # linear RAM amplitude
    im_i2_rel: float = 0.0  # nonlinear RAM amplitude
    fm_im_phase1_rad: float = 0.0
    fm_im_phase2_rad: float = 0.0
    lockin_phase_rad: float = 0.0
    lowpass_cutoff_Hz: float = 0.0  # 0 -> auto: f_m / 10


def optical_frequency(cfg: WMSConfig, t_s: np.ndarray) -> np.ndarray:
    """Instantaneous optical frequency nu(t) [cm-1]."""
    nu = cfg.center_wavenumber_cm1 + cfg.modulation_depth_cm1 * np.cos(
        2.0 * np.pi * cfg.modulation_frequency_Hz * t_s
    )
    if cfg.scan_range_cm1 > 0.0 and cfg.scan_rate_Hz > 0.0:
        ramp = (t_s * cfg.scan_rate_Hz) % 1.0  # normalized sawtooth [0, 1)
        nu = nu + cfg.scan_range_cm1 * (ramp - 0.5)
    return nu


def laser_intensity(cfg: WMSConfig, t_s: np.ndarray, mean_intensity: float = 1.0) -> np.ndarray:
    """Laser intensity I0(t) with linear + nonlinear RAM (Rieker convention)."""
    wt = 2.0 * np.pi * cfg.modulation_frequency_Hz * t_s
    return mean_intensity * (
        1.0
        + cfg.im_i0_rel * np.cos(wt + cfg.fm_im_phase1_rad)
        + cfg.im_i2_rel * np.cos(2.0 * wt + cfg.fm_im_phase2_rad)
    )


def lockin_demodulate(
    signal: np.ndarray,
    t_s: np.ndarray,
    reference_frequency_Hz: float,
    harmonic: int,
    lockin_phase_rad: float = 0.0,
    lowpass_cutoff_Hz: float = 0.0,
    sampling_rate_Hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Digital lock-in: return (X, Y) at the given harmonic.

    X = LP[ 2 * s(t) * cos(2*pi*n*f*t + phi) ],
    Y = LP[-2 * s(t) * sin(2*pi*n*f*t + phi) ]

    The factor 2 makes X equal the cosine-quadrature Fourier coefficient of
    s(t) at n*f (so a signal A*cos(n*w*t) demodulates to X = A at phi = 0).
    Zero-phase filtering (filtfilt) avoids group-delay distortion of the
    scan-resolved harmonic envelope.
    """
    if sampling_rate_Hz is None:
        dt = float(t_s[1] - t_s[0])
        sampling_rate_Hz = 1.0 / dt
    wt = 2.0 * np.pi * harmonic * reference_frequency_Hz * t_s + lockin_phase_rad
    x_raw = 2.0 * signal * np.cos(wt)
    y_raw = -2.0 * signal * np.sin(wt)
    cutoff = lowpass_cutoff_Hz or reference_frequency_Hz / 10.0
    sos = butter(4, cutoff, btype="low", fs=sampling_rate_Hz, output="sos")
    return sosfiltfilt(sos, x_raw), sosfiltfilt(sos, y_raw)


def simulate_wms(
    cfg: WMSConfig,
    absorbance_of_nu,
    mean_intensity: float = 1.0,
    harmonics: tuple[int, ...] = (1, 2),
) -> dict:
    """Run the full WMS chain against an absorbance function alpha*L = A(nu).

    ``absorbance_of_nu``: callable mapping wavenumber array [cm-1] to napierian
    absorbance (dimensionless). Returns a dict with time axis, nu(t),
    transmitted intensity, and per-harmonic (X, Y, R) lock-in outputs.

    Deterministic: no randomness lives here; noise is injected by the
    instrument layer upstream of/around this chain.
    """
    n = int(round(cfg.duration_s * cfg.sampling_rate_Hz))
    t = np.arange(n) / cfg.sampling_rate_Hz
    nu = optical_frequency(cfg, t)
    i0 = laser_intensity(cfg, t, mean_intensity)
    transmitted = i0 * np.exp(-absorbance_of_nu(nu))
    out = {"t_s": t, "nu_cm1": nu, "intensity": transmitted}
    for h in harmonics:
        x, y = lockin_demodulate(
            transmitted,
            t,
            cfg.modulation_frequency_Hz,
            h,
            cfg.lockin_phase_rad,
            cfg.lowpass_cutoff_Hz,
            cfg.sampling_rate_Hz,
        )
        out[f"x_{h}f"] = x
        out[f"y_{h}f"] = y
        out[f"r_{h}f"] = np.hypot(x, y)
    return out
