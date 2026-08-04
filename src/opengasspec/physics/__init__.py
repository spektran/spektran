"""Forward physics: HITRAN line data, line shapes, Beer-Lambert absorption."""

from .absorption import absorption_coefficient, line_strength_at_T, simulate_absorbance
from .hitran import LineList, demo_ch4_2nu3, fetch_lines
from .lineshape import (
    doppler_hwhm_cm1,
    gaussian_profile,
    lorentz_hwhm_cm1,
    lorentz_profile,
    voigt_profile,
)

__all__ = [
    "LineList",
    "absorption_coefficient",
    "demo_ch4_2nu3",
    "doppler_hwhm_cm1",
    "fetch_lines",
    "gaussian_profile",
    "line_strength_at_T",
    "lorentz_hwhm_cm1",
    "lorentz_profile",
    "simulate_absorbance",
    "voigt_profile",
]
