"""Tests for laser RIN, TIA bandwidth, and detector responsivity."""

import numpy as np
import pytest

from spektran.instrument.electronics import (
    detector_responsivity,
    rin_noise,
    tia_bandwidth_filter,
)


class TestRIN:
    def test_zero_mean(self):
        rng = np.random.default_rng(42)
        noise = rin_noise(rng, 10000, rin_dBc_Hz=-140.0,
                          bandwidth_Hz=1e6, sampling_rate_Hz=10e6)
        assert abs(noise.mean()) < 0.01

    def test_variance_scales_with_rin(self):
        rng1 = np.random.default_rng(42)
        n1 = rin_noise(rng1, 50000, rin_dBc_Hz=-130.0,
                       bandwidth_Hz=1e6, sampling_rate_Hz=10e6)
        rng2 = np.random.default_rng(42)
        n2 = rin_noise(rng2, 50000, rin_dBc_Hz=-140.0,
                       bandwidth_Hz=1e6, sampling_rate_Hz=10e6)
        assert n1.std() > n2.std() * 2.0

    def test_variance_scales_with_bandwidth(self):
        rng1 = np.random.default_rng(42)
        n1 = rin_noise(rng1, 50000, rin_dBc_Hz=-140.0,
                       bandwidth_Hz=2e6, sampling_rate_Hz=10e6)
        rng2 = np.random.default_rng(42)
        n2 = rin_noise(rng2, 50000, rin_dBc_Hz=-140.0,
                       bandwidth_Hz=1e6, sampling_rate_Hz=10e6)
        assert n1.std() > n2.std()

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        noise = rin_noise(rng, 2000, -140.0, 1e6, 10e6)
        assert noise.shape == (2000,)


class TestTIABandwidth:
    def test_identity_when_bandwidth_exceeds_nyquist(self):
        sig = np.array([1.0, 0.5, 0.2, 0.8, 1.0])
        out = tia_bandwidth_filter(sig, bandwidth_Hz=1e6, sampling_rate_Hz=1e6)
        np.testing.assert_array_equal(sig, out)

    def test_attenuates_high_frequency(self):
        fs = 1e6
        n = 10000
        t = np.arange(n) / fs
        low_freq = np.sin(2 * np.pi * 1000 * t)
        high_freq = np.sin(2 * np.pi * 100000 * t)
        sig = low_freq + high_freq
        out = tia_bandwidth_filter(sig, bandwidth_Hz=10000, sampling_rate_Hz=fs)
        power_in_high = np.var(high_freq)
        residual_high = out - low_freq
        power_out_high = np.var(residual_high[1000:-1000])
        assert power_out_high < power_in_high * 0.1

    def test_preserves_shape(self):
        sig = np.random.default_rng(42).normal(0, 1, 500)
        out = tia_bandwidth_filter(sig, 50000, 1e6)
        assert out.shape == sig.shape


class TestDetectorResponsivity:
    def test_flat_above_cutoff(self):
        nu = np.linspace(6500.0, 7000.0, 100)
        r = detector_responsivity(nu, cutoff_cm1=5882.0, peak_responsivity=1.0)
        assert np.all(r > 0.95)

    def test_drops_below_cutoff(self):
        nu = np.array([5000.0, 5500.0, 5882.0, 6500.0])
        r = detector_responsivity(nu, cutoff_cm1=5882.0, peak_responsivity=1.0)
        assert r[0] < 0.1
        assert r[-1] > 0.9

    def test_peak_responsivity_scaling(self):
        nu = np.array([7000.0])
        r1 = detector_responsivity(nu, peak_responsivity=1.0)
        r2 = detector_responsivity(nu, peak_responsivity=0.8)
        assert r2[0] / r1[0] == pytest.approx(0.8, rel=0.01)

    def test_sigmoid_at_cutoff(self):
        nu = np.array([5882.0])
        r = detector_responsivity(nu, cutoff_cm1=5882.0, peak_responsivity=1.0)
        assert r[0] == pytest.approx(0.5, rel=0.01)
