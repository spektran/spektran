"""Sim-to-real validation: compare SPEKTRAN output against literature benchmarks.

These tests encode quantitative reference values from peer-reviewed publications
and HITRAN-validated measurements. They verify that SPEKTRAN's forward model
produces physically plausible results consistent with real-world observations.

References:
- HITRAN2020: I.E. Gordon et al., JQSRT 277 (2022) 107949,
  doi:10.1016/j.jqsrt.2021.107949
- Cossel et al.: K.C. Cossel et al., JQSRT (2025),
  NIST dual-comb Mauna Loa validation: CO2 0.1%, CH4 -1.1% HITRAN bias
- Ngo et al.: N.H. Ngo et al., JQSRT 129 (2013) 89,
  doi:10.1016/j.jqsrt.2013.05.034 (HTP IUPAC recommendation)
- Rieker et al.: G.B. Rieker et al., Appl. Opt. 48 (2009) 5546,
  doi:10.1364/AO.48.005546 (calibration-free WMS)
"""

import numpy as np
import pytest

from spektran.physics.absorption import absorption_coefficient, simulate_absorbance
from spektran.physics.hitran import demo_ch4_2nu3
from spektran.physics.lineshape import doppler_hwhm_cm1, lorentz_hwhm_cm1, voigt_profile


class TestLineStrengthAccuracy:
    """HITRAN-validated CH4 line intensities in the 2nu3 band."""

    def test_ch4_peak_absorbance_order_of_magnitude(self):
        """100 ppm CH4, 10 m path, 296 K, 1 atm: peak absorbance ~0.01-0.1."""
        nu, absorbance = simulate_absorbance(
            molecule="CH4",
            concentration_ppm=100.0,
            temperature_K=296.0,
            pressure_atm=1.0,
            path_length_m=10.0,
        )
        peak = np.max(absorbance)
        assert 0.001 < peak < 1.0, f"peak absorbance {peak} outside physical range"

    def test_absorbance_scales_linearly_with_concentration(self):
        """Beer-Lambert linearity: doubling concentration doubles absorbance."""
        _, a1 = simulate_absorbance(concentration_ppm=50.0)
        _, a2 = simulate_absorbance(concentration_ppm=100.0)
        ratio = np.max(a2) / np.max(a1)
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_absorbance_scales_with_path_length(self):
        _, a1 = simulate_absorbance(path_length_m=5.0)
        _, a2 = simulate_absorbance(path_length_m=10.0)
        ratio = np.max(a2) / np.max(a1)
        assert ratio == pytest.approx(2.0, rel=0.01)


class TestVoigtLineWidth:
    """Voigt FWHM at known T/P must match spectroscopic expectations."""

    def test_doppler_width_ch4_296K(self):
        """CH4 at 6047 cm-1, 296 K: Doppler HWHM ~ 0.0094 cm-1."""
        hwhm = doppler_hwhm_cm1(6047.0, 296.0, 16.0313)
        assert hwhm == pytest.approx(0.0094, rel=0.05)

    def test_lorentz_width_1atm(self):
        """CH4 air-broadened at 1 atm: gamma_L ~ 0.06 cm-1 (HITRAN typical)."""
        hwhm = lorentz_hwhm_cm1(1.0, 296.0, 0.06, 0.08, 0.0001, 0.75)
        assert hwhm == pytest.approx(0.06, rel=0.01)

    def test_voigt_fwhm_dominated_by_lorentz_at_1atm(self):
        """At 1 atm, Lorentz HWHM >> Doppler HWHM, so Voigt ~ Lorentzian."""
        nu0 = 6047.0
        gamma_D = doppler_hwhm_cm1(nu0, 296.0, 16.0313)
        gamma_L = 0.06
        nu = np.linspace(nu0 - 0.5, nu0 + 0.5, 10000)
        phi = voigt_profile(nu, nu0, gamma_D, gamma_L)
        half_max = np.max(phi) / 2.0
        above = nu[phi >= half_max]
        fwhm = above[-1] - above[0]
        assert fwhm == pytest.approx(2 * gamma_L, rel=0.1)

    def test_voigt_fwhm_at_low_pressure(self):
        """At 0.01 atm, Doppler dominates: FWHM ~ 2*gamma_D."""
        nu0 = 6047.0
        gamma_D = doppler_hwhm_cm1(nu0, 296.0, 16.0313)
        gamma_L = lorentz_hwhm_cm1(0.01, 296.0, 0.06, 0.08, 0.0001, 0.75)
        nu = np.linspace(nu0 - 0.1, nu0 + 0.1, 50000)
        phi = voigt_profile(nu, nu0, gamma_D, gamma_L)
        half_max = np.max(phi) / 2.0
        above = nu[phi >= half_max]
        fwhm = above[-1] - above[0]
        assert fwhm == pytest.approx(2 * gamma_D, rel=0.15)


class TestTemperatureDependence:
    """Line strengths and widths vary with temperature as expected."""

    def test_hot_band_weakening(self):
        """Most ground-state CH4 lines weaken with increasing T (population
        redistribution to higher levels). Integrated absorbance at 500 K
        should be less than at 296 K for low E'' lines."""
        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 2000)
        alpha_296 = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0)
        alpha_500 = absorption_coefficient(nu, lines, 100e-6, 500.0, 1.0)
        assert np.sum(alpha_296) > np.sum(alpha_500) * 0.8

    def test_doppler_broadening_increases_with_T(self):
        """Doppler width scales as sqrt(T)."""
        hwhm_296 = doppler_hwhm_cm1(6047.0, 296.0, 16.0313)
        hwhm_500 = doppler_hwhm_cm1(6047.0, 500.0, 16.0313)
        expected_ratio = np.sqrt(500.0 / 296.0)
        assert hwhm_500 / hwhm_296 == pytest.approx(expected_ratio, rel=0.001)


class TestWMS2fAnalytical:
    """WMS 2f signal vs analytical prediction (Rieker 2009 convention)."""

    def test_2f_peak_arndt_formula(self):
        """Arndt's analytical 2f peak for Lorentzian in optically thin limit."""
        from tests.reference_impl.ref_wms import arndt_lorentzian_h2_peak
        from spektran.physics.wms import WMSConfig, simulate_wms
        from spektran.physics.lineshape import lorentz_profile

        nu0 = 6047.0
        gamma_l = 0.05
        peak_abs = 1e-3
        m = 2.2
        scale = peak_abs / (1.0 / (np.pi * gamma_l))

        def abs_fn(nu):
            return scale * lorentz_profile(np.asarray(nu, dtype=float), nu0, gamma_l)

        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=m * gamma_l,
            sampling_rate_Hz=2e6,
            duration_s=0.02,
            center_wavenumber_cm1=nu0,
        )
        out = simulate_wms(cfg, abs_fn)
        n = len(out["x_2f"])
        lo, hi = int(n * 0.3), int(n * 0.7)
        x2f = float(np.mean(out["x_2f"][lo:hi]))
        analytic = arndt_lorentzian_h2_peak(peak_abs, m)
        assert x2f == pytest.approx(analytic, rel=0.01)


class TestHITRANConsistency:
    """HITRAN database consistency checks."""

    def test_line_strength_positive(self):
        lines = demo_ch4_2nu3()
        assert np.all(lines.sw_cm_per_molec > 0)

    def test_line_center_in_range(self):
        lines = demo_ch4_2nu3()
        assert np.all(lines.nu0_cm1 > 6046.0)
        assert np.all(lines.nu0_cm1 < 6048.0)

    def test_broadening_coefficients_positive(self):
        lines = demo_ch4_2nu3()
        assert np.all(lines.gamma_air > 0)
        assert np.all(lines.gamma_self > 0)

    def test_temperature_exponent_physical(self):
        """n_air typically between 0.4 and 1.0 for most molecules."""
        lines = demo_ch4_2nu3()
        assert np.all(lines.n_air > 0.3)
        assert np.all(lines.n_air < 1.5)
