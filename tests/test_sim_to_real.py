"""Tests for sim-to-real gap improvements: shot noise, periodic interference,
speckle noise, clock jitter, enhanced gain nonlinearity, mode hop, multi-pass
etalon, gas flow turbulence, correlated baseline drift, H2O continuum, line
mixing, non-uniform gas path, reference channel integration."""

import numpy as np
import pytest

from spektran.instrument.detector import (
    clock_jitter,
    gain_nonlinearity,
    periodic_interference,
    shot_noise,
    speckle_noise,
)
from spektran.instrument.etalon import multipass_etalon_transmission
from spektran.instrument.laser import apply_mode_hop
from spektran.instrument.optics import correlated_baseline_drift, gas_flow_turbulence
from spektran.physics.absorption import rosenkranz_line_mixing
from spektran.physics.continuum import h2o_continuum_absorbance
from spektran.physics.hitran import demo_ch4_2nu3

SEED = 20260811


class TestShotNoise:
    def test_signal_dependent_variance(self):
        rng = np.random.default_rng(SEED)
        low_sig = np.full(50000, 0.1)
        high_sig = np.full(50000, 1.0)
        noise_low = shot_noise(rng, low_sig, gain=1e-3)
        noise_high = shot_noise(np.random.default_rng(SEED + 1), high_sig, gain=1e-3)
        assert noise_high.std() > noise_low.std() * 2.5

    def test_zero_signal_zero_noise(self):
        rng = np.random.default_rng(SEED)
        sig = np.zeros(1000)
        n = shot_noise(rng, sig, gain=1.0)
        assert np.allclose(n, 0.0)

    def test_deterministic(self):
        sig = np.ones(100)
        a = shot_noise(np.random.default_rng(42), sig, 1e-3)
        b = shot_noise(np.random.default_rng(42), sig, 1e-3)
        assert a.tobytes() == b.tobytes()


class TestPeriodicInterference:
    def test_correct_frequency(self):
        rng = np.random.default_rng(SEED)
        n = 100000
        scan_rate = 10.0
        sig = periodic_interference(rng, n, [50.0], [1e-3], scan_rate, n)
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(n, d=1.0 / (scan_rate * n))
        peak_freq = freqs[np.argmax(spec[1:]) + 1]
        assert peak_freq == pytest.approx(50.0, rel=0.02)

    def test_amplitude_matches(self):
        rng = np.random.default_rng(SEED)
        sig = periodic_interference(rng, 100000, [50.0], [0.005], 10.0, 100000)
        assert sig.max() == pytest.approx(0.005, rel=0.02)


class TestSpeckleNoise:
    def test_spectral_correlation(self):
        rng = np.random.default_rng(SEED)
        uncorr = speckle_noise(rng, 10000, 1e-3, correlation_length=1)
        corr = speckle_noise(np.random.default_rng(SEED + 1), 10000, 1e-3, correlation_length=50)
        autocorr_uncorr = np.correlate(uncorr[:200], uncorr[:200], mode="full")
        autocorr_corr = np.correlate(corr[:200], corr[:200], mode="full")
        mid = len(autocorr_uncorr) // 2
        ratio_uncorr = autocorr_uncorr[mid + 5] / autocorr_uncorr[mid]
        ratio_corr = autocorr_corr[mid + 5] / autocorr_corr[mid]
        assert ratio_corr > ratio_uncorr

    def test_sigma_matches(self):
        rng = np.random.default_rng(SEED)
        sig = speckle_noise(rng, 100000, 2e-4, correlation_length=10)
        assert sig.std() == pytest.approx(2e-4, rel=0.05)


class TestClockJitter:
    def test_statistics(self):
        rng = np.random.default_rng(SEED)
        j = clock_jitter(rng, 50000, 1e-5)
        assert j.std() == pytest.approx(1e-5, rel=0.05)
        assert abs(j.mean()) < 1e-7


class TestEnhancedGainNonlinearity:
    def test_cubic_term(self):
        s = np.array([0.0, 0.5, 1.0])
        out = gain_nonlinearity(s, 0.0, cubic_rel=0.1)
        expected = s + 0.1 * s**3
        np.testing.assert_allclose(out, expected)

    def test_saturation_clamps(self):
        s = np.array([0.0, 5.0, 100.0])
        out = gain_nonlinearity(s, 0.0, saturation_level=1.0)
        assert out.max() < 1.0 + 1e-10
        assert out[-1] == pytest.approx(np.tanh(100.0), rel=1e-6)


class TestModeHop:
    def test_no_hop_when_zero_probability(self):
        nu = np.linspace(6046.0, 6048.0, 1000)
        result = apply_mode_hop(nu, np.random.default_rng(SEED), probability=0.0)
        assert result is nu

    def test_hop_creates_discontinuity(self):
        nu = np.linspace(6046.0, 6048.0, 1000)
        hopped = apply_mode_hop(nu, np.random.default_rng(42), probability=1.0, hop_size_cm1=0.5)
        diffs = np.diff(hopped)
        normal_step = (6048.0 - 6046.0) / 999
        assert np.max(np.abs(diffs)) > 10 * normal_step

    def test_deterministic(self):
        nu = np.linspace(6046.0, 6048.0, 500)
        a = apply_mode_hop(nu, np.random.default_rng(7), probability=1.0)
        b = apply_mode_hop(nu, np.random.default_rng(7), probability=1.0)
        np.testing.assert_array_equal(a, b)


class TestMultipassEtalon:
    def test_differs_from_single_pass(self):
        nu = np.linspace(6046.0, 6048.0, 10000)
        from spektran.instrument.etalon import etalon_transmission
        single = etalon_transmission(nu, 0.15, 3e-3)
        multi = multipass_etalon_transmission(nu, 20, 0.15, 3e-3)
        assert not np.allclose(single, multi)
        corr = np.corrcoef(single - 1.0, multi - 1.0)[0, 1]
        assert corr < 0.99

    def test_amplitude_decays(self):
        nu = np.linspace(6046.0, 6048.0, 5000)
        t_nodecay = multipass_etalon_transmission(nu, 10, 0.15, 3e-3, amplitude_decay=1.0)
        t_decay = multipass_etalon_transmission(nu, 10, 0.15, 3e-3, amplitude_decay=0.5)
        assert (t_nodecay - 1.0).std() > (t_decay - 1.0).std()


class TestGasFlowTurbulence:
    def test_returns_ones_when_zero_sigma(self):
        rng = np.random.default_rng(SEED)
        result = gas_flow_turbulence(rng, 1000, 0.0)
        np.testing.assert_array_equal(result, np.ones(1000))

    def test_sigma_matches(self):
        rng = np.random.default_rng(SEED)
        result = gas_flow_turbulence(rng, 50000, 1e-4)
        assert (result - 1.0).std() == pytest.approx(1e-4, rel=0.05)

    def test_band_limited(self):
        rng = np.random.default_rng(SEED)
        result = gas_flow_turbulence(rng, 10000, 1e-3, cutoff_norm=0.05)
        spec = np.abs(np.fft.rfft(result - result.mean()))
        freqs = np.fft.rfftfreq(len(result))
        high_energy = np.sum(spec[freqs > 0.2] ** 2)
        low_energy = np.sum(spec[(freqs > 0.001) & (freqs < 0.1)] ** 2)
        assert high_energy < low_energy * 0.01


class TestCorrelatedBaselineDrift:
    def test_uncorrelated_initial_draw(self):
        rng = np.random.default_rng(SEED)
        base, coeffs = correlated_baseline_drift(rng, 1000, [0.01, 0.005])
        assert len(coeffs) == 2
        assert base.shape == (1000,)
        assert base[500] != pytest.approx(1.0, abs=1e-10)

    def test_ou_correlation(self):
        coeffs_sigma = [0.02]
        all_c0 = []
        prev = None
        rng = np.random.default_rng(SEED)
        for _ in range(500):
            _, coeffs = correlated_baseline_drift(
                rng, 100, coeffs_sigma, previous_coeffs=prev, tau_scans=10.0,
            )
            all_c0.append(coeffs[0])
            prev = coeffs
        c0 = np.array(all_c0)
        autocorr_1 = np.corrcoef(c0[:-1], c0[1:])[0, 1]
        assert autocorr_1 > 0.85

    def test_independent_without_previous(self):
        results = []
        for s in range(100):
            _, coeffs = correlated_baseline_drift(
                np.random.default_rng(s), 100, [0.02],
            )
            results.append(coeffs[0])
        assert abs(np.corrcoef(results[:-1], results[1:])[0, 1]) < 0.3


class TestH2OContinuum:
    def test_zero_for_dry_gas(self):
        nu = np.linspace(4000.0, 5000.0, 500)
        a = h2o_continuum_absorbance(nu, 0.0, 296.0, 1.0, 10.0)
        np.testing.assert_array_equal(a, np.zeros(500))

    def test_positive_for_wet_gas(self):
        nu = np.linspace(4000.0, 5000.0, 500)
        a = h2o_continuum_absorbance(nu, 0.02, 296.0, 1.0, 10.0)
        assert np.all(a > 0)

    def test_increases_with_path_length(self):
        nu = np.linspace(4000.0, 5000.0, 100)
        a1 = h2o_continuum_absorbance(nu, 0.01, 296.0, 1.0, 1.0)
        a10 = h2o_continuum_absorbance(nu, 0.01, 296.0, 1.0, 10.0)
        np.testing.assert_allclose(a10, a1 * 10.0, rtol=1e-10)

    def test_scales_with_pressure(self):
        nu = np.linspace(4000.0, 5000.0, 100)
        a1 = h2o_continuum_absorbance(nu, 0.01, 296.0, 1.0, 5.0)
        a2 = h2o_continuum_absorbance(nu, 0.01, 296.0, 2.0, 5.0)
        assert np.all(a2 > a1)


class TestRosenkranzLineMixing:
    def test_nonzero_contribution(self):
        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 2000)
        mixing = rosenkranz_line_mixing(
            nu, lines, 296.0, 5.0, mole_fraction=0.01,
        )
        assert np.max(np.abs(mixing)) > 1e-8
        assert mixing.min() < 0 and mixing.max() > 0

    def test_linear_in_pressure(self):
        lines = demo_ch4_2nu3()
        nu = np.linspace(6046.0, 6048.0, 500)
        m1 = rosenkranz_line_mixing(nu, lines, 296.0, 0.5, mole_fraction=100e-6)
        m2 = rosenkranz_line_mixing(nu, lines, 296.0, 1.0, mole_fraction=100e-6)
        ratio = np.max(np.abs(m2)) / np.max(np.abs(m1))
        assert ratio == pytest.approx(2.0, rel=0.3)


class TestGeneratorIntegration:
    """Integration test: all new effects fire without error in a full record."""

    def _make_spec(self):
        from spektran.generator import GenerationSpec
        from spektran.physics.hitran import demo_ch4_2nu3
        return GenerationSpec(
            lines=demo_ch4_2nu3(),
            molecule="CH4",
            concentration_ppm_low=50.0,
            concentration_ppm_high=500.0,
        )

    def _base_instrument(self):
        return {
            "schema_version": "0.2",
            "instrument_config_id": "test-sim2real",
            "technique": "TDLAS-DA",
            "laser": {
                "center_wavenumber_cm1": 6047.0,
                "scan_range_cm1": 2.0,
                "scan_rate_Hz": 100.0,
                "mode_hop_probability": 0.5,
                "mode_hop_size_cm1": 0.2,
            },
            "detector": {
                "white_noise_rel": 1e-4,
                "one_over_f_sigma_rel": 5e-5,
                "one_over_f_slope": 1.0,
                "adc_bits": 16,
                "shot_noise_gain": 1e-4,
                "emi_frequencies_Hz": [50.0],
                "emi_amplitudes_rel": [2e-5],
                "speckle_noise_sigma": 3e-5,
                "speckle_correlation_length": 8,
                "gain_nonlinearity_rel": 1e-3,
                "gain_cubic_rel": 5e-4,
                "saturation_level": 1.5,
                "clock_jitter_rel": 1e-6,
            },
            "optics": {
                "baseline_poly_rel": [0.001, -0.0005],
                "beam_wander_sigma_rel": 5e-5,
                "gas_flow_turbulence_sigma_rel": 3e-5,
            },
            "etalons": [
                {"free_spectral_range_cm1": 0.15, "amplitude_rel": 1e-4, "phase_rad": 0.0},
            ],
            "physics": {
                "line_mixing": True,
                "h2o_continuum_fraction": 0.01,
            },
            "reference_channel": {
                "noise_sigma_rel": 5e-5,
            },
        }

    def test_full_record_with_all_effects(self):
        from spektran.generator import generate_record
        spec = self._make_spec()
        cfg = self._base_instrument()
        seed_seq = np.random.SeedSequence(SEED)
        record = generate_record(spec, cfg, seed_seq)
        assert "raw_scan" in record["arrays"]
        assert "absorbance_clean" in record["arrays"]
        assert record["arrays"]["raw_scan"].shape == (2000,)
        assert np.isfinite(record["arrays"]["raw_scan"]).all()

    def test_multipass_etalon_integration(self):
        from spektran.generator import generate_record
        spec = self._make_spec()
        cfg = self._base_instrument()
        del cfg["etalons"]
        cfg["multipass_etalon"] = {
            "n_passes": 20,
            "base_fsr_cm1": 0.15,
            "base_amplitude_rel": 1e-4,
        }
        record = generate_record(spec, cfg, np.random.SeedSequence(SEED))
        assert np.isfinite(record["arrays"]["raw_scan"]).all()

    def test_layered_gas_path(self):
        from spektran.generator import generate_record
        spec = self._make_spec()
        cfg = self._base_instrument()
        cfg["gas_path_layers"] = [
            {"length_fraction": 0.3, "temperature_K": 400.0, "concentration_scale": 1.2},
            {"length_fraction": 0.4, "temperature_K": 296.0, "concentration_scale": 1.0},
            {"length_fraction": 0.3, "temperature_K": 350.0, "concentration_scale": 0.8},
        ]
        record = generate_record(spec, cfg, np.random.SeedSequence(SEED))
        assert np.isfinite(record["arrays"]["raw_scan"]).all()

    def test_correlated_baseline(self):
        from spektran.generator import generate_record
        spec = self._make_spec()
        cfg = self._base_instrument()
        cfg["optics"]["baseline_drift_coeffs_sigma"] = [0.005, 0.002]
        cfg["optics"]["baseline_drift_tau"] = 20.0
        del cfg["optics"]["baseline_poly_rel"]
        record = generate_record(spec, cfg, np.random.SeedSequence(SEED))
        assert np.isfinite(record["arrays"]["raw_scan"]).all()
