"""Physics-correctness tests for the NDIR forward model."""

from __future__ import annotations

import numpy as np
import pytest

from spektran.physics import (
    bandpass_filter,
    demo_ch4_2nu3,
    ndir_detector_signal,
    ndir_ratio,
    planck_spectral_radiance,
    simulate_ndir,
)


class TestPlanckSpectralRadiance:
    """Planck function B(nu, T) basic sanity checks."""

    def test_peak_position_wien(self):
        """Wien displacement law: nu_max(cm-1) ~ 1.961 * T."""
        nu = np.linspace(100.0, 5000.0, 10000)
        b = planck_spectral_radiance(nu, 800.0)
        peak_nu = nu[np.argmax(b)]
        assert abs(peak_nu - 1569.0) < 100.0

    def test_positivity(self):
        nu = np.linspace(100.0, 10000.0, 500)
        b = planck_spectral_radiance(nu, 800.0)
        assert np.all(b > 0.0)

    def test_monotonic_in_temperature_rayleigh_jeans(self):
        """At low nu (Rayleigh-Jeans regime), higher T gives higher B."""
        nu = np.array([100.0, 200.0, 300.0])
        b_low = planck_spectral_radiance(nu, 600.0)
        b_high = planck_spectral_radiance(nu, 1000.0)
        assert np.all(b_high > b_low)

    def test_stefan_boltzmann_t4_ratio(self):
        """Integral of B(nu,T) over all nu is proportional to T^4."""
        nu = np.linspace(1.0, 30000.0, 50000)
        integral_600 = np.trapezoid(
            planck_spectral_radiance(nu, 600.0), nu,
        )
        integral_900 = np.trapezoid(
            planck_spectral_radiance(nu, 900.0), nu,
        )
        ratio = integral_900 / integral_600
        expected = (900.0 / 600.0) ** 4
        assert ratio == pytest.approx(expected, rel=0.02)


class TestBandpassFilter:
    """Optical bandpass filter profile correctness."""

    def test_gaussian_peak_is_one(self):
        val = bandpass_filter(np.array([3000.0]), 3000.0, 50.0, "gaussian")
        assert val[0] == pytest.approx(1.0, abs=1e-14)

    def test_gaussian_fwhm(self):
        center, fwhm = 3000.0, 50.0
        half_points = np.array([center - fwhm / 2, center + fwhm / 2])
        vals = bandpass_filter(half_points, center, fwhm, "gaussian")
        assert vals[0] == pytest.approx(0.5, abs=1e-10)
        assert vals[1] == pytest.approx(0.5, abs=1e-10)

    def test_tophat_inside_outside(self):
        center, fwhm = 3000.0, 100.0
        nu = np.array([2940.0, 2960.0, 3000.0, 3040.0, 3060.0])
        f = bandpass_filter(nu, center, fwhm, "tophat")
        assert f[0] == 0.0
        assert f[1] == 1.0
        assert f[2] == 1.0
        assert f[3] == 1.0
        assert f[4] == 0.0

    def test_values_in_zero_one(self):
        nu = np.linspace(2000.0, 4000.0, 1000)
        for shape in ("gaussian", "tophat"):
            f = bandpass_filter(nu, 3000.0, 200.0, shape)
            assert np.all(f >= 0.0)
            assert np.all(f <= 1.0)

    def test_unknown_shape_raises(self):
        nu = np.array([3000.0])
        with pytest.raises(ValueError, match="Unknown filter shape"):
            bandpass_filter(nu, 3000.0, 50.0, "triangular")


class TestNDIRDetectorSignal:
    """Integrated detector signal properties."""

    def _make_inputs(self, absorbance_level: float = 0.0):
        nu = np.linspace(2900.0, 3100.0, 300)
        radiance = planck_spectral_radiance(nu, 800.0)
        filt = bandpass_filter(nu, 3000.0, 50.0, "gaussian")
        absorbance = np.full_like(nu, absorbance_level)
        return nu, absorbance, filt, radiance

    def test_signal_positive(self):
        sig = ndir_detector_signal(*self._make_inputs(0.0))
        assert sig > 0.0

    def test_signal_decreases_with_absorbance(self):
        sig_zero = ndir_detector_signal(*self._make_inputs(0.0))
        sig_low = ndir_detector_signal(*self._make_inputs(0.5))
        sig_high = ndir_detector_signal(*self._make_inputs(2.0))
        assert sig_zero > sig_low > sig_high > 0.0

    def test_zero_absorbance_is_maximum(self):
        sig_zero = ndir_detector_signal(*self._make_inputs(0.0))
        for level in (0.1, 0.5, 1.0, 5.0):
            sig = ndir_detector_signal(*self._make_inputs(level))
            assert sig < sig_zero


class TestNDIRRatio:
    """NDIR ratio R = active / reference."""

    def test_equal_signals_give_unity(self):
        assert ndir_ratio(1.0, 1.0) == pytest.approx(1.0)

    def test_ratio_decreases_with_active_absorption(self):
        ref = 100.0
        ratio_low = ndir_ratio(90.0, ref)
        ratio_high = ndir_ratio(50.0, ref)
        assert ratio_low > ratio_high

    def test_ratio_scales_linearly(self):
        assert ndir_ratio(50.0, 100.0) == pytest.approx(0.5)


class TestSimulateNDIR:
    """End-to-end NDIR simulation with demo CH4 lines."""

    ACTIVE_CENTER = 6047.0
    ACTIVE_FWHM = 5.0
    REF_CENTER = 6100.0
    REF_FWHM = 5.0

    def _run(self, concentration_ppm: float, **kwargs):
        lines = demo_ch4_2nu3()
        defaults = dict(
            lines=lines,
            molecule="CH4",
            concentration_ppm=concentration_ppm,
            temperature_K=296.0,
            pressure_atm=1.0,
            path_length_m=10.0,
            source_temperature_K=800.0,
            active_filter_center_cm1=self.ACTIVE_CENTER,
            active_filter_fwhm_cm1=self.ACTIVE_FWHM,
            reference_filter_center_cm1=self.REF_CENTER,
            reference_filter_fwhm_cm1=self.REF_FWHM,
            filter_shape="gaussian",
            n_integration_points=500,
        )
        defaults.update(kwargs)
        return simulate_ndir(**defaults)

    def test_return_keys(self):
        result = self._run(100.0)
        expected_keys = {
            "active_signal",
            "reference_signal",
            "ratio",
            "concentration_ppm",
            "nu_cm1",
            "absorbance",
            "source_radiance",
            "active_filter",
            "reference_filter",
        }
        assert set(result.keys()) == expected_keys

    def test_zero_concentration_baseline(self):
        result = self._run(0.0)
        assert result["active_signal"] > 0.0
        assert result["reference_signal"] > 0.0
        assert result["ratio"] > 0.0

    def test_higher_concentration_lower_ratio(self):
        r_low = self._run(100.0)["ratio"]
        r_high = self._run(10000.0)["ratio"]
        assert r_low > r_high

    def test_reference_channel_constant(self):
        """Reference channel signal nearly unaffected by target gas.

        At very high concentrations far Voigt wings can reach the
        reference filter, so we allow 0.1 % tolerance.
        """
        ref_0 = self._run(0.0)["reference_signal"]
        ref_100 = self._run(100.0)["reference_signal"]
        ref_10k = self._run(10000.0)["reference_signal"]
        assert ref_100 == pytest.approx(ref_0, rel=1e-6)
        assert ref_10k == pytest.approx(ref_0, rel=1e-3)

    def test_active_signal_decreases(self):
        act_0 = self._run(0.0)["active_signal"]
        act_10k = self._run(10000.0)["active_signal"]
        assert act_10k < act_0

    def test_concentration_recorded(self):
        result = self._run(42.0)
        assert result["concentration_ppm"] == 42.0

    def test_interferent_increases_absorption(self):
        """Adding an interferent in the active band lowers the ratio."""
        r_clean = self._run(100.0)["ratio"]
        interf_lines = demo_ch4_2nu3()
        r_with = self._run(
            100.0,
            interferents=[
                {"lines": interf_lines, "concentration_ppm": 5000.0},
            ],
        )["ratio"]
        assert r_with < r_clean

    def test_tophat_filter(self):
        result = self._run(100.0, filter_shape="tophat")
        assert result["active_signal"] > 0.0
        assert result["ratio"] > 0.0
