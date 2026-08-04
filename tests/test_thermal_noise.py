"""Tests for temperature-dependent detector noise."""
import numpy as np
import pytest

from spektran.instrument.detector import dark_current_noise, thermal_noise_scale


class TestThermalNoiseScale:
    def test_unity_at_reference(self):
        assert thermal_noise_scale(296.0, 296.0) == 1.0

    def test_increases_with_temperature(self):
        assert thermal_noise_scale(400.0, 296.0) > 1.0

    def test_decreases_below_reference(self):
        assert thermal_noise_scale(200.0, 296.0) < 1.0

    def test_sqrt_relationship(self):
        scale = thermal_noise_scale(4 * 296.0, 296.0)
        np.testing.assert_allclose(scale, 2.0, atol=1e-10)

    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError):
            thermal_noise_scale(-100.0, 296.0)

    def test_rejects_zero_temperature(self):
        with pytest.raises(ValueError):
            thermal_noise_scale(0.0, 296.0)


class TestDarkCurrentNoise:
    def test_higher_temp_more_noise(self):
        """Dark current noise should increase with temperature."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        n1 = dark_current_noise(rng1, 50000, 1e-4, 296.0)
        n2 = dark_current_noise(rng2, 50000, 1e-4, 340.0)
        assert np.std(n2) > np.std(n1)

    def test_zero_at_reference(self):
        """At reference temperature, sigma should equal sigma_ref."""
        rng = np.random.default_rng(42)
        n = dark_current_noise(rng, 100000, 1e-3, 296.0, 296.0)
        np.testing.assert_allclose(np.std(n), 1e-3, rtol=0.05)

    def test_deterministic(self):
        a = dark_current_noise(np.random.default_rng(7), 1000, 1e-4, 310.0)
        b = dark_current_noise(np.random.default_rng(7), 1000, 1e-4, 310.0)
        np.testing.assert_array_equal(a, b)

    def test_activation_energy_effect(self):
        """Higher activation energy = more temperature sensitivity."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        # Same T but different activation energies
        n_low_ea = dark_current_noise(rng1, 50000, 1e-4, 340.0,
                                       activation_energy_eV=0.18)
        n_high_ea = dark_current_noise(rng2, 50000, 1e-4, 340.0,
                                        activation_energy_eV=0.56)
        # Ea=0 gives scale=1 (no T dependence) regardless of T; the Arrhenius
        # exponent's magnitude grows with Ea, so larger Ea -> larger deviation
        # from sigma_ref at T != T_ref -> more noise here (T=340 > T_ref=296).
        assert np.std(n_high_ea) > np.std(n_low_ea)


class TestThermalNoiseIntegration:
    def test_generation_with_thermal_noise(self):
        """Full generation with temperature-dependent detector noise works."""
        from pathlib import Path

        from spektran.generator import GenerationSpec, generate_dataset
        from spektran.instrument.sampling import load_instrument_config
        from spektran.physics.hitran import demo_ch4_2nu3
        cfg = load_instrument_config(
            Path(__file__).resolve().parents[1]
            / "configs/instruments/vi-da-thermal-noise-12.yaml"
        )
        spec = GenerationSpec(lines=demo_ch4_2nu3())
        records = generate_dataset(spec, cfg, 3, master_seed=54321)
        assert len(records) == 3
        for rec in records:
            assert np.all(np.isfinite(rec["arrays"]["raw_scan"]))

    def test_backward_compatible(self):
        """Existing configs without thermal noise params still work identically."""
        from pathlib import Path

        from spektran.generator import GenerationSpec, generate_dataset
        from spektran.instrument.sampling import load_instrument_config
        from spektran.physics.hitran import demo_ch4_2nu3
        cfg = load_instrument_config(
            Path(__file__).resolve().parents[1]
            / "configs/instruments/vi-da-medium-02.yaml"
        )
        spec = GenerationSpec(lines=demo_ch4_2nu3())
        rec1 = generate_dataset(spec, cfg, 1, master_seed=99999)
        rec2 = generate_dataset(spec, cfg, 1, master_seed=99999)
        np.testing.assert_array_equal(
            rec1[0]["arrays"]["raw_scan"],
            rec2[0]["arrays"]["raw_scan"]
        )
