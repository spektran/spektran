"""Regression and performance tests for the vectorized absorption coefficient."""

import numpy as np
import pytest

from spektran.physics.absorption import absorption_coefficient, line_strength_at_T
from spektran.physics.constants import number_density_cm3
from spektran.physics.hitran import LineList, demo_ch4_2nu3
from spektran.physics.lineshape import doppler_hwhm_cm1, lorentz_hwhm_cm1, voigt_profile
from spektran.physics.tips import tips_q_ratio


def _reference_loop(nu_cm1, lines, mole_fraction, temperature_K, pressure_atm, q_ratio=None):
    """Per-line loop reference (the pre-vectorization algorithm)."""
    q = (q_ratio or tips_q_ratio)(lines.molecule, temperature_K)
    strengths = line_strength_at_T(
        lines.sw_cm_per_molec, lines.nu0_cm1, lines.elower_cm1, temperature_K, q
    )
    n_absorber = number_density_cm3(pressure_atm, temperature_K) * mole_fraction
    alpha = np.zeros_like(nu_cm1, dtype=np.float64)
    for j in range(len(lines)):
        nu0_shifted = lines.nu0_cm1[j] + lines.delta_air[j] * pressure_atm
        a_d = doppler_hwhm_cm1(nu0_shifted, temperature_K, lines.molar_mass_amu)
        g_l = lorentz_hwhm_cm1(
            pressure_atm,
            temperature_K,
            lines.gamma_air[j],
            lines.gamma_self[j],
            mole_fraction,
            lines.n_air[j],
        )
        alpha += strengths[j] * voigt_profile(nu_cm1, nu0_shifted, a_d, g_l)
    return n_absorber * alpha


_CONDITIONS = [
    (296.0, 1.0, 100e-6),
    (350.0, 0.5, 50e-6),
    (250.0, 2.0, 500e-6),
    (300.0, 0.1, 1e-6),
    (400.0, 1.5, 1000e-6),
]


class TestVectorizedRegression:
    """Vectorized absorption_coefficient must match the per-line reference loop."""

    @pytest.mark.parametrize("T,P,x", _CONDITIONS)
    def test_multi_condition(self, T, P, x):
        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 2000)
        expected = _reference_loop(nu, lines, x, T, P)
        actual = absorption_coefficient(nu, lines, x, T, P)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    @pytest.mark.parametrize("T,P,x", _CONDITIONS)
    def test_custom_q_ratio(self, T, P, x):
        from spektran.physics.absorption import default_q_ratio

        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 500)
        expected = _reference_loop(nu, lines, x, T, P, q_ratio=default_q_ratio)
        actual = absorption_coefficient(nu, lines, x, T, P, q_ratio=default_q_ratio)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


class TestEdgeCases:
    def test_empty_lines(self):
        empty = LineList(
            molecule="CH4",
            nu0_cm1=np.array([]),
            sw_cm_per_molec=np.array([]),
            gamma_air=np.array([]),
            gamma_self=np.array([]),
            n_air=np.array([]),
            delta_air=np.array([]),
            elower_cm1=np.array([]),
        )
        nu = np.linspace(6046.0, 6048.0, 100)
        alpha = absorption_coefficient(nu, empty, 100e-6, 296.0, 1.0)
        assert alpha.shape == nu.shape
        assert np.all(alpha == 0.0)

    def test_single_line(self):
        lines = demo_ch4_2nu3()
        single = LineList(
            molecule="CH4",
            nu0_cm1=lines.nu0_cm1[:1],
            sw_cm_per_molec=lines.sw_cm_per_molec[:1],
            gamma_air=lines.gamma_air[:1],
            gamma_self=lines.gamma_self[:1],
            n_air=lines.n_air[:1],
            delta_air=lines.delta_air[:1],
            elower_cm1=lines.elower_cm1[:1],
        )
        nu = np.linspace(6046.0, 6048.0, 500)
        expected = _reference_loop(nu, single, 100e-6, 296.0, 1.0)
        actual = absorption_coefficient(nu, single, 100e-6, 296.0, 1.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_multi_line(self):
        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 500)
        expected = _reference_loop(nu, lines, 100e-6, 296.0, 1.0)
        actual = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


class TestWingCutoff:
    def test_no_cutoff_matches_default(self):
        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 500)
        alpha_default = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0)
        alpha_no_cutoff = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0, wing_cutoff_cm1=0.0)
        np.testing.assert_array_equal(alpha_default, alpha_no_cutoff)

    def test_cutoff_reduces_absorption(self):
        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 500)
        alpha_full = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0)
        alpha_cut = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0, wing_cutoff_cm1=0.1)
        assert np.all(alpha_cut <= alpha_full + 1e-30)

    def test_very_large_cutoff_matches_full(self):
        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 500)
        alpha_full = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0)
        alpha_big = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0, wing_cutoff_cm1=1000.0)
        np.testing.assert_array_equal(alpha_full, alpha_big)

    def test_narrow_cutoff_zeros_far_wing(self):
        lines = demo_ch4_2nu3()
        nu_far = np.array([lines.nu0_cm1[0] + 5.0])
        alpha_full = absorption_coefficient(nu_far, lines, 100e-6, 296.0, 1.0)
        alpha_cut = absorption_coefficient(nu_far, lines, 100e-6, 296.0, 1.0, wing_cutoff_cm1=0.5)
        assert alpha_cut[0] <= alpha_full[0]


class TestSyntheticPerformance:
    """Verify vectorized results on a large synthetic line list."""

    @staticmethod
    def _make_synthetic_lines(n_lines, rng):
        center = 6047.0
        return LineList(
            molecule="CH4",
            nu0_cm1=center + rng.uniform(-1.0, 1.0, n_lines),
            sw_cm_per_molec=10.0 ** rng.uniform(-24, -20, n_lines),
            gamma_air=rng.uniform(0.04, 0.08, n_lines),
            gamma_self=rng.uniform(0.06, 0.10, n_lines),
            n_air=rng.uniform(0.5, 0.8, n_lines),
            delta_air=rng.uniform(-0.01, 0.0, n_lines),
            elower_cm1=rng.uniform(0.0, 500.0, n_lines),
        )

    def test_200_lines_matches_loop(self):
        rng = np.random.default_rng(42)
        lines = self._make_synthetic_lines(200, rng)
        nu = np.linspace(6046.0, 6048.0, 500)
        expected = _reference_loop(nu, lines, 100e-6, 296.0, 1.0)
        actual = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_200_lines_spot_check(self):
        rng = np.random.default_rng(42)
        lines = self._make_synthetic_lines(200, rng)
        nu = np.linspace(6046.0, 6048.0, 500)
        alpha = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0)
        assert alpha.shape == (500,)
        assert np.all(np.isfinite(alpha))
        assert np.all(alpha >= 0.0)
        assert alpha.max() > 0.0
