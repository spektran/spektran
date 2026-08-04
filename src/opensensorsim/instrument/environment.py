"""Environmental perturbations: temperature/pressure jitter around nominals.

The jittered values feed the forward model (line strengths and widths change
accordingly), producing physically consistent lineshape distortion rather
than ad-hoc signal warping (plan §5.2 item 6)."""

from __future__ import annotations

import numpy as np


def jittered_conditions(
    rng: np.random.Generator,
    temperature_K: float,
    pressure_atm: float,
    temperature_jitter_K: float = 0.0,
    pressure_jitter_atm: float = 0.0,
) -> tuple[float, float]:
    """Sample per-record (T, P) around nominal values (Gaussian, 1 sigma)."""
    t = temperature_K + (rng.normal(0.0, temperature_jitter_K) if temperature_jitter_K else 0.0)
    p = pressure_atm + (rng.normal(0.0, pressure_jitter_atm) if pressure_jitter_atm else 0.0)
    if t <= 0.0 or p <= 0.0:
        raise ValueError(f"jitter produced unphysical conditions T={t}, P={p}")
    return float(t), float(p)
