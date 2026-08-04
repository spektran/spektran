"""Forward physics: HITRAN line data, line shapes, Beer-Lambert absorption."""

from .absorption import absorption_coefficient, line_strength_at_T, simulate_absorbance
from .hitran import LineList, demo_ch4_2nu3, demo_co, demo_co2, demo_h2o, fetch_lines
from .lineshape import (
    doppler_hwhm_cm1,
    gaussian_profile,
    lorentz_hwhm_cm1,
    lorentz_profile,
    voigt_profile,
)
from .tips import tips_q_ratio
from .wms import WMSConfig, lockin_demodulate, simulate_wms

__all__ = [
    "LineList",
    "WMSConfig",
    "absorption_coefficient",
    "demo_ch4_2nu3",
    "demo_co",
    "demo_co2",
    "demo_h2o",
    "doppler_hwhm_cm1",
    "fetch_lines",
    "gaussian_profile",
    "line_strength_at_T",
    "lockin_demodulate",
    "lorentz_hwhm_cm1",
    "lorentz_profile",
    "simulate_absorbance",
    "simulate_wms",
    "tips_q_ratio",
    "voigt_profile",
]
