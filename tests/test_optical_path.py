"""Tests for window contamination and beam wander optical path effects."""

from pathlib import Path

import numpy as np

from spektran.instrument.optics import beam_wander, window_contamination

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "instruments"


class TestWindowContamination:
    def test_clean_window(self):
        """Zero contamination returns all ones."""
        nu = np.linspace(6046, 6048, 500)
        t = window_contamination(nu, 0.0)
        np.testing.assert_array_equal(t, 1.0)

    def test_flat_contamination(self):
        """spectral_slope=0 gives uniform loss."""
        nu = np.linspace(6046, 6048, 500)
        t = window_contamination(nu, 0.05, spectral_slope=0.0)
        np.testing.assert_allclose(t, 0.95)

    def test_spectral_dependence(self):
        """Non-zero slope makes higher frequencies attenuate more."""
        nu = np.linspace(6046, 6048, 500)
        t = window_contamination(nu, 0.05, spectral_slope=2.0)
        # Higher wavenumber = more scattering = lower transmission
        assert t[0] > t[-1]

    def test_mean_loss_matches(self):
        """Mean transmission loss should approximately match contamination_rel."""
        nu = np.linspace(6046, 6048, 1000)
        t = window_contamination(nu, 0.05, spectral_slope=2.0)
        mean_loss = 1.0 - np.mean(t)
        np.testing.assert_allclose(mean_loss, 0.05, atol=0.005)

    def test_output_in_range(self):
        """Transmission should be in (0, 1]."""
        nu = np.linspace(6046, 6048, 500)
        t = window_contamination(nu, 0.20, spectral_slope=4.0)
        assert np.all(t > 0)
        assert np.all(t <= 1.0)


class TestBeamWander:
    def test_zero_sigma_returns_ones(self):
        """Zero sigma means no beam wander."""
        rng = np.random.default_rng(42)
        bw = beam_wander(rng, 500, 0.0)
        np.testing.assert_array_equal(bw, 1.0)

    def test_centered_on_one(self):
        """Beam wander should be centered around 1.0."""
        rng = np.random.default_rng(42)
        bw = beam_wander(rng, 5000, 0.01)
        np.testing.assert_allclose(np.mean(bw), 1.0, atol=0.01)

    def test_correct_sigma(self):
        """Standard deviation should match requested sigma."""
        rng = np.random.default_rng(42)
        bw = beam_wander(rng, 10000, 0.005)
        np.testing.assert_allclose(np.std(bw), 0.005, atol=0.001)

    def test_low_frequency_content(self):
        """Beam wander should be low-frequency (smooth)."""
        rng = np.random.default_rng(42)
        bw = beam_wander(rng, 2000, 0.01)
        # First derivative should be small relative to the signal
        deriv_std = np.std(np.diff(bw))
        assert deriv_std < 0.005

    def test_reproducibility(self):
        """Same seed produces same beam wander."""
        bw1 = beam_wander(np.random.default_rng(99), 500, 0.01)
        bw2 = beam_wander(np.random.default_rng(99), 500, 0.01)
        np.testing.assert_array_equal(bw1, bw2)


class TestOpticalPathIntegration:
    def test_instrument_config_with_optics(self):
        """Full generation with window contamination + beam wander works."""
        from spektran.generator import GenerationSpec, generate_dataset
        from spektran.instrument.sampling import load_instrument_config
        from spektran.physics.hitran import demo_ch4_2nu3

        cfg = load_instrument_config(CONFIG_DIR / "vi-da-contaminated-11.yaml")
        spec = GenerationSpec(lines=demo_ch4_2nu3())
        records = generate_dataset(spec, cfg, 3, master_seed=12345)
        assert len(records) == 3
        for rec in records:
            assert "raw_scan" in rec["arrays"]
            assert np.all(np.isfinite(rec["arrays"]["raw_scan"]))
