"""Tests for the thermal chirp laser tuning model."""

from pathlib import Path

import numpy as np
import pytest

from spektran.generator import GenerationSpec, generate_dataset
from spektran.instrument.laser import current_tuning_model, scan_frequency_axis
from spektran.instrument.sampling import load_instrument_config
from spektran.physics import demo_ch4_2nu3
from spektran.validate import validate_record

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "instruments"


class TestCurrentTuningModel:
    def test_zero_thermal_fraction_is_linear(self):
        """With no thermal contribution, tuning should be perfectly linear."""
        ramp = np.linspace(0, 1, 1000, endpoint=False)
        nu = current_tuning_model(ramp, 6047.0, 2.0, thermal_fraction=0.0)
        expected = 6047.0 + 2.0 * (ramp - 0.5)
        np.testing.assert_allclose(nu, expected, atol=1e-12)

    def test_zero_tau_is_linear(self):
        """With zero time constant, thermal response is instantaneous = linear."""
        ramp = np.linspace(0, 1, 1000, endpoint=False)
        nu = current_tuning_model(
            ramp, 6047.0, 2.0, thermal_fraction=0.8, thermal_tau_norm=0.0
        )
        expected = 6047.0 + 2.0 * (ramp - 0.5)
        np.testing.assert_allclose(nu, expected, atol=1e-12)

    def test_scan_range_preserved(self):
        """Total frequency span should approximately equal scan_range."""
        ramp = np.linspace(0, 1, 2000, endpoint=False)
        nu = current_tuning_model(
            ramp, 6047.0, 2.0, thermal_fraction=0.8, thermal_tau_norm=0.3
        )
        span = nu[-1] - nu[0]
        assert 1.5 < span < 2.5

    def test_monotonically_increasing(self):
        """Frequency should be monotonically increasing for positive scan range."""
        ramp = np.linspace(0, 1, 2000, endpoint=False)
        nu = current_tuning_model(
            ramp, 6047.0, 2.0, thermal_fraction=0.8, thermal_tau_norm=0.3
        )
        assert np.all(np.diff(nu) > 0)

    def test_nonlinearity_present(self):
        """With thermal chirp, the scan should deviate from linear."""
        ramp = np.linspace(0, 1, 1000, endpoint=False)
        nu_nonlinear = current_tuning_model(
            ramp, 6047.0, 2.0, thermal_fraction=0.8, thermal_tau_norm=0.3
        )
        nu_linear = current_tuning_model(ramp, 6047.0, 2.0, thermal_fraction=0.0)
        deviation = np.max(np.abs(nu_nonlinear - nu_linear))
        assert deviation > 0.01

    def test_thermal_lag_slow_start(self):
        """Thermal lag means frequency changes slowly at start, fast at end."""
        ramp = np.linspace(0, 1, 2000, endpoint=False)
        nu = current_tuning_model(
            ramp, 6047.0, 2.0, thermal_fraction=0.8, thermal_tau_norm=0.3
        )
        dnu = np.diff(nu)
        n = len(dnu)
        assert np.mean(dnu[: n // 4]) < np.mean(dnu[3 * n // 4 :])

    def test_centered_at_midpoint(self):
        """nu(u=0.5) must equal center_wavenumber_cm1, matching the linear model."""
        ramp = np.array([0.5])
        nu = current_tuning_model(
            ramp, 6047.0, 2.0, thermal_fraction=0.8, thermal_tau_norm=0.3
        )
        assert nu[0] == pytest.approx(6047.0, abs=1e-10)


class TestScanFrequencyAxisTuningModel:
    def test_thermal_chirp_model_selection(self):
        """tuning_model='thermal_chirp' should use the thermal chirp model."""
        ramp = np.linspace(0, 1, 500, endpoint=False)
        nu = scan_frequency_axis(
            ramp,
            6047.0,
            2.0,
            tuning_model="thermal_chirp",
            tuning_params={"thermal_fraction": 0.8, "thermal_tau_norm": 0.3},
        )
        assert np.all(np.diff(nu) > 0)
        linear = 6047.0 + 2.0 * (ramp - 0.5)
        assert np.max(np.abs(nu - linear)) > 0.01

    def test_thermal_chirp_matches_direct_call(self):
        """scan_frequency_axis dispatch must match calling current_tuning_model directly."""
        ramp = np.linspace(0, 1, 500, endpoint=False)
        via_dispatch = scan_frequency_axis(
            ramp,
            6047.0,
            2.0,
            tuning_model="thermal_chirp",
            tuning_params={"thermal_fraction": 0.7, "thermal_tau_norm": 0.2},
        )
        direct = current_tuning_model(
            ramp, 6047.0, 2.0, thermal_fraction=0.7, thermal_tau_norm=0.2
        )
        np.testing.assert_array_equal(via_dispatch, direct)

    def test_thermal_chirp_default_params(self):
        """tuning_params may be omitted; current_tuning_model's own defaults apply."""
        ramp = np.linspace(0, 1, 500, endpoint=False)
        nu = scan_frequency_axis(ramp, 6047.0, 2.0, tuning_model="thermal_chirp")
        expected = current_tuning_model(ramp, 6047.0, 2.0)
        np.testing.assert_array_equal(nu, expected)

    def test_backward_compatible_default(self):
        """Default (no tuning_model) should use polynomial path."""
        ramp = np.linspace(0, 1, 500, endpoint=False)
        nu = scan_frequency_axis(ramp, 6047.0, 2.0)
        expected = 6047.0 + 2.0 * (ramp - 0.5)
        np.testing.assert_allclose(nu, expected, atol=1e-12)

    def test_backward_compatible_with_poly(self):
        """tuning_model=None must still honor nonlinearity_poly_cm1 (unaffected by the new path)."""
        ramp = np.linspace(0, 1, 500, endpoint=False)
        nu = scan_frequency_axis(ramp, 6047.0, 2.0, nonlinearity_poly_cm1=[0.1])
        du = ramp - 0.5
        expected = 6047.0 + 2.0 * du + 0.1 * du**2
        np.testing.assert_allclose(nu, expected, atol=1e-12)


class TestThermalChirpInstrumentConfig:
    """Integration: the vi-da-thermal-10 config exercises the full generation pipeline."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return load_instrument_config(CONFIG_DIR / "vi-da-thermal-10.yaml")

    def test_config_loads_and_validates(self, cfg):
        assert cfg["laser"]["tuning_model"] == "thermal_chirp"

    def test_generated_records_validate(self, cfg):
        spec = GenerationSpec(lines=demo_ch4_2nu3(), n_points=400)
        for rec in generate_dataset(spec, cfg, n_records=3, master_seed=2026):
            assert validate_record(rec["meta"]) == [], rec["meta"]["record_id"]

    def test_sampled_tuning_params_recorded_in_provenance(self, cfg):
        spec = GenerationSpec(lines=demo_ch4_2nu3(), n_points=400)
        rec = generate_dataset(spec, cfg, n_records=1, master_seed=2026)[0]
        sampled_laser = rec["meta"]["provenance"]["noise_config"]["sampled"]["laser"]
        assert sampled_laser["tuning_model"] == "thermal_chirp"
        params = sampled_laser["tuning_params"]
        assert 0.70 <= params["thermal_fraction"] <= 0.90
        assert 0.10 <= params["thermal_tau_norm"] <= 0.50

    def test_reproducible(self, cfg):
        spec = GenerationSpec(lines=demo_ch4_2nu3(), n_points=400)
        a = generate_dataset(spec, cfg, n_records=2, master_seed=77)
        b = generate_dataset(spec, cfg, n_records=2, master_seed=77)
        for ra, rb in zip(a, b):
            assert ra["meta"] == rb["meta"]
            assert ra["arrays"]["raw_scan"].tobytes() == rb["arrays"]["raw_scan"].tobytes()
