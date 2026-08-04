"""Physics-correctness tests for line shapes (plan §8 hard requirements)."""

import numpy as np
import pytest

from opengasspec.physics import (
    doppler_hwhm_cm1,
    gaussian_profile,
    lorentz_profile,
    voigt_profile,
)
from tests.reference_impl.ref_lineshape import voigt_profile_ref

RNG_SEED = 20260804


def _wide_grid(nu0, width, half_span_widths=2000.0, n=4_000_001):
    return np.linspace(nu0 - half_span_widths * width, nu0 + half_span_widths * width, n)


class TestNormalization:
    def test_gaussian_area_is_one(self):
        nu = np.linspace(-1.0, 1.0, 200001)
        area = np.trapezoid(gaussian_profile(nu, 0.0, 0.01), nu)
        assert abs(area - 1.0) < 1e-8

    def test_lorentz_area_is_one(self):
        # Lorentzian tails decay as 1/x^2 — needs a very wide grid
        nu = _wide_grid(0.0, 0.01)
        area = np.trapezoid(lorentz_profile(nu, 0.0, 0.01), nu)
        assert abs(area - 1.0) < 1e-3

    def test_voigt_area_is_one(self):
        nu = _wide_grid(0.0, 0.01)
        area = np.trapezoid(voigt_profile(nu, 0.0, 0.005, 0.01), nu)
        assert abs(area - 1.0) < 1e-3


class TestLimits:
    """Low-pressure limit -> Doppler; high-pressure limit -> Lorentz.

    Plan §8: RMSE vs analytic profile < 0.5% of peak value.
    """

    def test_doppler_limit(self):
        a_d = 0.01
        nu = np.linspace(-0.1, 0.1, 20001)
        v = voigt_profile(nu, 0.0, a_d, a_d * 1e-6)
        g = gaussian_profile(nu, 0.0, a_d)
        rmse = np.sqrt(np.mean((v - g) ** 2))
        assert rmse < 0.005 * g.max()

    def test_lorentz_limit(self):
        g_l = 0.1
        nu = np.linspace(-2.0, 2.0, 20001)
        v = voigt_profile(nu, 0.0, g_l * 1e-6, g_l)
        lor = lorentz_profile(nu, 0.0, g_l)
        rmse = np.sqrt(np.mean((v - lor) ** 2))
        assert rmse < 0.005 * lor.max()


class TestDopplerWidth:
    def test_ch4_room_temperature_value(self):
        # Analytic sanity check: alpha_D = (nu0/c) sqrt(2 ln2 kT/m)
        # CH4 (16.0313 amu) at 296 K, nu0 = 6046.96 cm-1
        import math

        got = doppler_hwhm_cm1(6046.9647, 296.0, 16.0313)
        expected = (6046.9647 / 2.99792458e10) * math.sqrt(
            2.0 * math.log(2.0) * 1.380649e-16 * 296.0 / (16.0313 * 1.66053906660e-24)
        )
        assert got == pytest.approx(expected, rel=1e-12)
        # Physical plausibility: a few 1e-3 cm-1 in the NIR
        assert 0.005 < got < 0.02

    def test_scaling_laws(self):
        base = doppler_hwhm_cm1(6000.0, 296.0, 16.0)
        assert doppler_hwhm_cm1(6000.0, 4 * 296.0, 16.0) == pytest.approx(2 * base, rel=1e-12)
        assert doppler_hwhm_cm1(6000.0, 296.0, 4 * 16.0) == pytest.approx(base / 2, rel=1e-12)
        assert doppler_hwhm_cm1(12000.0, 296.0, 16.0) == pytest.approx(2 * base, rel=1e-12)


class TestVoigtCrossValidation:
    """Gate G3: main (Faddeeva) vs reference (quadrature) — 1000 random points,
    relative deviation < 0.1%."""

    def test_1000_random_points(self):
        rng = np.random.default_rng(RNG_SEED)
        n = 1000
        max_rel = 0.0
        for _ in range(n):
            a_d = 10.0 ** rng.uniform(-3.5, -1.0)  # 3e-4 .. 0.1 cm-1
            g_l = 10.0 ** rng.uniform(-4.0, 0.0)  # 1e-4 .. 1 cm-1
            width = max(a_d, g_l)
            offset = rng.uniform(-10.0, 10.0) * width
            nu0 = rng.uniform(1000.0, 8000.0)
            main = voigt_profile(np.array([nu0 + offset]), nu0, a_d, g_l)[0]
            ref = voigt_profile_ref(nu0 + offset, nu0, a_d, g_l)
            rel = abs(main - ref) / abs(ref)
            max_rel = max(max_rel, rel)
        assert max_rel < 1e-3, f"max relative deviation {max_rel:.2e} exceeds 0.1%"
