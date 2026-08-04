"""Instrument-effects layer: the noise chain that makes simulated spectra real."""

from .detector import adc_quantize, gain_nonlinearity, one_over_f_noise, white_noise
from .environment import jittered_conditions
from .etalon import etalon_transmission, multi_etalon_transmission
from .laser import (
    center_drift_cm1,
    current_tuning_model,
    intensity_ramp,
    linewidth_convolve,
    scan_frequency_axis,
)
from .optics import (
    baseline_polynomial,
    beam_wander,
    intensity_fluctuation,
    transmittance_decay,
    window_contamination,
)

__all__ = [
    "adc_quantize",
    "baseline_polynomial",
    "beam_wander",
    "center_drift_cm1",
    "current_tuning_model",
    "etalon_transmission",
    "gain_nonlinearity",
    "intensity_fluctuation",
    "intensity_ramp",
    "jittered_conditions",
    "linewidth_convolve",
    "multi_etalon_transmission",
    "one_over_f_noise",
    "scan_frequency_axis",
    "transmittance_decay",
    "white_noise",
    "window_contamination",
]
