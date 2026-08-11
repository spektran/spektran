"""Etalon fringes: periodic parasitic interference in the optical path.

Low-finesse parasitic etalons (windows, fiber ends, lens surfaces) impose a
sinusoidal transmission ripple with free spectral range FSR = 1/(2nL). Real
systems typically show 1-3 fringe systems whose phases drift slowly with
temperature (plan §5.2)."""

from __future__ import annotations

import numpy as np


def etalon_transmission(
    nu_cm1: np.ndarray,
    free_spectral_range_cm1: float,
    amplitude_rel: float,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Multiplicative transmission ripple of one low-finesse etalon.

    T(nu) = 1 + A * cos(2*pi*nu/FSR + phi)

    (first-order Airy expansion, valid for low reflectivity surfaces).
    """
    if free_spectral_range_cm1 <= 0.0:
        raise ValueError("free_spectral_range_cm1 must be > 0")
    return 1.0 + amplitude_rel * np.cos(
        2.0 * np.pi * nu_cm1 / free_spectral_range_cm1 + phase_rad
    )


def multipass_etalon_transmission(
    nu_cm1: np.ndarray,
    n_passes: int,
    base_fsr_cm1: float,
    base_amplitude_rel: float,
    phase_rad: float = 0.0,
    amplitude_decay: float = 0.85,
) -> np.ndarray:
    """Etalon pattern from a multi-pass absorption cell (Herriott/White).

    In multi-pass cells, the beam reflects N times between mirrors. Each
    reflection pair acts as a weak Fabry-Perot cavity with progressively
    shifted FSR (due to slightly different path lengths at each pass) and
    decaying amplitude (alignment loss per pass). The result is a
    superposition of N/2 fringe systems that is richer and harder to
    subtract than single-pass etalons.

    Parameters
    ----------
    n_passes : int
        Number of beam passes through the cell (typically 20-100).
    base_fsr_cm1 : float
        FSR of the fundamental (single-pass) cavity.
    base_amplitude_rel : float
        Fringe amplitude of the first pass.
    amplitude_decay : float
        Multiplicative decay of fringe amplitude per pass (0 to 1).
    """
    trans = np.ones_like(nu_cm1, dtype=np.float64)
    n_cavities = max(1, n_passes // 2)
    for k in range(n_cavities):
        fsr_k = base_fsr_cm1 / (1.0 + 0.02 * k)
        amp_k = base_amplitude_rel * (amplitude_decay ** k)
        phase_k = phase_rad + 0.5 * k
        trans = trans * etalon_transmission(nu_cm1, fsr_k, amp_k, phase_k)
    return trans


def multi_etalon_transmission(
    nu_cm1: np.ndarray,
    etalons: list[dict],
    t_s: float = 0.0,
) -> np.ndarray:
    """Combined ripple of several etalons, with optional phase drift in time.

    Each dict: {free_spectral_range_cm1, amplitude_rel, phase_rad,
    phase_drift_rad_per_s (optional)}.
    """
    trans = np.ones_like(nu_cm1, dtype=np.float64)
    for e in etalons:
        phase = e.get("phase_rad", 0.0) + e.get("phase_drift_rad_per_s", 0.0) * t_s
        trans = trans * etalon_transmission(
            nu_cm1, e["free_spectral_range_cm1"], e["amplitude_rel"], phase
        )
    return trans
