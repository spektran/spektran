"""Noise-statistics self-checks (plan §8: PSD fits within 5% of config) and
determinism tests for the instrument-effects layer."""

import numpy as np
import pytest
from scipy.signal import welch

from opengasspec.instrument import (
    adc_quantize,
    baseline_polynomial,
    etalon_transmission,
    gain_nonlinearity,
    intensity_ramp,
    jittered_conditions,
    linewidth_convolve,
    multi_etalon_transmission,
    one_over_f_noise,
    scan_frequency_axis,
    white_noise,
)

SEED = 20260809


class TestWhiteNoise:
    def test_sigma_within_5pct(self):
        rng = np.random.default_rng(SEED)
        x = white_noise(rng, 200_000, sigma=3.2e-4)
        assert x.std() == pytest.approx(3.2e-4, rel=0.05)

    def test_psd_flat(self):
        rng = np.random.default_rng(SEED)
        sigma = 1e-3
        x = white_noise(rng, 400_000, sigma)
        f, pxx = welch(x, fs=1.0, nperseg=4096)
        # log-PSD slope of white noise ~ 0; level = 2*sigma^2 (one-sided, fs=1)
        mask = f > 0
        slope = np.polyfit(np.log10(f[mask]), np.log10(pxx[mask]), 1)[0]
        assert abs(slope) < 0.05
        assert np.mean(pxx[mask]) == pytest.approx(2.0 * sigma**2, rel=0.05)

    def test_deterministic(self):
        a = white_noise(np.random.default_rng(7), 1000, 1e-3)
        b = white_noise(np.random.default_rng(7), 1000, 1e-3)
        assert a.tobytes() == b.tobytes()


class TestOneOverFNoise:
    @pytest.mark.parametrize("slope", [0.8, 1.0, 1.3])
    def test_psd_slope_within_5pct(self, slope):
        rng = np.random.default_rng(SEED)
        x = one_over_f_noise(rng, 2**20, sigma=1e-3, slope=slope)
        f, pxx = welch(x, fs=1.0, nperseg=2**14)
        # Fit log-log slope over two decades away from DC and Nyquist
        mask = (f > 1e-4) & (f < 1e-1)
        fit = np.polyfit(np.log10(f[mask]), np.log10(pxx[mask]), 1)[0]
        assert -fit == pytest.approx(slope, rel=0.05), f"PSD slope {-fit:.3f} vs {slope}"

    def test_total_sigma(self):
        rng = np.random.default_rng(SEED)
        x = one_over_f_noise(rng, 2**18, sigma=2e-4, slope=1.0)
        assert x.std() == pytest.approx(2e-4, rel=1e-9)


class TestEtalon:
    def test_ripple_amplitude_and_period(self):
        nu = np.linspace(6046.0, 6048.0, 20001)
        fsr, amp = 0.15, 3e-3
        t = etalon_transmission(nu, fsr, amp)
        assert t.max() == pytest.approx(1.0 + amp, abs=1e-6)
        assert t.min() == pytest.approx(1.0 - amp, abs=1e-6)
        # Dominant FFT period equals FSR (tolerance limited by the FFT bin
        # width: df = 1/span = 0.5 cycles/cm-1 -> ~4% at 1/FSR = 6.7)
        step = nu[1] - nu[0]
        spec = np.abs(np.fft.rfft(t - t.mean()))
        f_axis = np.fft.rfftfreq(len(nu), d=step)
        assert 1.0 / f_axis[np.argmax(spec)] == pytest.approx(fsr, rel=0.05)

    def test_multi_etalon_drift(self):
        nu = np.linspace(6046.0, 6048.0, 2001)
        etalons = [
            {"free_spectral_range_cm1": 0.2, "amplitude_rel": 2e-3, "phase_rad": 0.0,
             "phase_drift_rad_per_s": 0.1},
        ]
        t0 = multi_etalon_transmission(nu, etalons, t_s=0.0)
        t1 = multi_etalon_transmission(nu, etalons, t_s=10.0)  # 1 rad shift
        assert not np.allclose(t0, t1)


class TestDetectorChain:
    def test_gain_nonlinearity_zero_is_identity(self):
        s = np.linspace(0.0, 1.0, 101)
        assert np.array_equal(gain_nonlinearity(s, 0.0), s)

    def test_adc_step_size(self):
        s = np.linspace(0.0, 1.0, 10001)
        q = adc_quantize(s, bits=12, full_scale=1.0)
        steps = np.unique(np.round(np.diff(np.unique(q)), 12))
        assert steps[0] == pytest.approx(1.0 / (2**12 - 1), rel=1e-9)
        assert np.abs(q - s).max() <= 0.5 / (2**12 - 1) + 1e-12

    def test_adc_clips(self):
        q = adc_quantize(np.array([-0.5, 1.5]), bits=8, full_scale=1.0)
        assert q[0] == 0.0 and q[1] == 1.0


class TestLaserEffects:
    def test_scan_axis_linear_when_no_poly(self):
        u = np.linspace(0.0, 1.0, 101)
        nu = scan_frequency_axis(u, 6047.0, 2.0)
        assert nu[0] == pytest.approx(6046.0) and nu[-1] == pytest.approx(6048.0)

    def test_nonlinearity_bends_axis(self):
        u = np.linspace(0.0, 1.0, 101)
        nu = scan_frequency_axis(u, 6047.0, 2.0, [0.1])
        lin = scan_frequency_axis(u, 6047.0, 2.0)
        dev = nu - lin
        assert abs(dev[50]) < 1e-12  # zero at center
        assert dev[0] == pytest.approx(0.1 * 0.25, rel=1e-9)

    def test_intensity_ramp(self):
        u = np.linspace(0.0, 1.0, 5)
        i = intensity_ramp(u, 1.0, slope_rel=0.2)
        assert i[0] == pytest.approx(0.9) and i[-1] == pytest.approx(1.1)

    def test_linewidth_convolution_broadens_and_conserves_area(self):
        from opengasspec.physics.lineshape import lorentz_profile

        step = 1e-4
        nu = np.arange(6046.0, 6048.0, step)
        a = lorentz_profile(nu, 6047.0, 0.02)
        conv = linewidth_convolve(a, step, linewidth_MHz=300.0)
        assert conv.max() < a.max()  # broadened -> lower peak
        assert np.trapezoid(conv, nu) == pytest.approx(np.trapezoid(a, nu), rel=1e-3)

    def test_zero_linewidth_identity(self):
        a = np.ones(100)
        assert linewidth_convolve(a, 1e-4, 0.0) is a


class TestOpticsAndEnvironment:
    def test_baseline_polynomial(self):
        u = np.linspace(0.0, 1.0, 3)
        b = baseline_polynomial(u, [0.1])
        assert b[1] == pytest.approx(1.0)
        assert b[-1] == pytest.approx(1.05)

    def test_jittered_conditions_deterministic_and_centered(self):
        vals = [
            jittered_conditions(np.random.default_rng(s), 296.0, 1.0, 0.5, 0.01)
            for s in range(2000)
        ]
        ts = np.array([v[0] for v in vals])
        assert ts.mean() == pytest.approx(296.0, abs=0.05)
        assert ts.std() == pytest.approx(0.5, rel=0.1)
        a = jittered_conditions(np.random.default_rng(42), 296.0, 1.0, 0.5, 0.01)
        b = jittered_conditions(np.random.default_rng(42), 296.0, 1.0, 0.5, 0.01)
        assert a == b

    def test_jitter_rejects_unphysical(self):
        # Deterministic trigger: negative nominal temperature with zero jitter
        with pytest.raises(ValueError):
            jittered_conditions(np.random.default_rng(0), -5.0, 1.0, 0.0, 0.0)
