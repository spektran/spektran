"""NDIR generator tests: structure, noise, drift, reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from spektran.ndir_generator import (
    NDIRGenerationSpec,
    generate_ndir_dataset,
    generate_ndir_record,
    generate_ndir_time_series,
)
from spektran.physics import demo_ch4_2nu3, simulate_ndir

TEST_NDIR_CONFIG = {
    "instrument_config_id": "vi-ndir-test",
    "schema_version": "0.1",
    "technique": "NDIR",
    "source": {
        "temperature_K": 800.0,
        "temperature_drift_K_per_s": 0.0,
        "intensity_fluctuation_rel": 0.0,
    },
    "filters": {
        "active_center_cm1": 6047.0,
        "active_fwhm_cm1": 2.0,
        "reference_center_cm1": 6040.0,
        "reference_fwhm_cm1": 2.0,
        "shape": "gaussian",
    },
    "detector": {
        "white_noise_rel": 0.0,
    },
    "environment": {
        "temperature_K": 296.0,
        "pressure_atm": 1.0,
    },
}


def _make_spec(**overrides) -> NDIRGenerationSpec:
    defaults = dict(
        lines=demo_ch4_2nu3(),
        molecule="CH4",
        concentration_ppm_low=100.0,
        concentration_ppm_high=100.0,
        log_uniform_concentration=False,
        path_length_m=0.10,
        n_integration_points=500,
    )
    defaults.update(overrides)
    return NDIRGenerationSpec(**defaults)


def _noisy_config(**overrides) -> dict:
    cfg = {
        "instrument_config_id": "vi-ndir-noisy",
        "schema_version": "0.1",
        "technique": "NDIR",
        "source": {
            "temperature_K": 800.0,
            "temperature_drift_K_per_s": 0.0,
            "intensity_fluctuation_rel": 0.01,
        },
        "filters": {
            "active_center_cm1": 6047.0,
            "active_fwhm_cm1": 2.0,
            "reference_center_cm1": 6040.0,
            "reference_fwhm_cm1": 2.0,
            "shape": "gaussian",
        },
        "detector": {
            "white_noise_rel": 0.001,
        },
        "environment": {
            "temperature_K": 296.0,
            "pressure_atm": 1.0,
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


class TestRecordStructure:
    def test_returns_meta_and_arrays(self):
        seed = np.random.SeedSequence(42)
        rec = generate_ndir_record(
            _make_spec(), TEST_NDIR_CONFIG, seed,
        )
        assert "meta" in rec
        assert "arrays" in rec

    def test_meta_technique_is_ndir(self):
        seed = np.random.SeedSequence(42)
        rec = generate_ndir_record(
            _make_spec(), TEST_NDIR_CONFIG, seed,
        )
        assert rec["meta"]["technique"] == "NDIR"

    def test_arrays_keys(self):
        seed = np.random.SeedSequence(42)
        rec = generate_ndir_record(
            _make_spec(), TEST_NDIR_CONFIG, seed,
        )
        expected = {
            "active_channel",
            "reference_channel",
            "ratio",
            "ratio_clean",
        }
        assert set(rec["arrays"].keys()) == expected

    def test_arrays_are_scalar_floats(self):
        seed = np.random.SeedSequence(42)
        rec = generate_ndir_record(
            _make_spec(), TEST_NDIR_CONFIG, seed,
        )
        for key, val in rec["arrays"].items():
            assert isinstance(val, float), (
                f"{key} is {type(val)}, expected float"
            )

    def test_ratio_clean_matches_forward_model(self):
        spec = _make_spec()
        seed = np.random.SeedSequence(42)
        rec = generate_ndir_record(spec, TEST_NDIR_CONFIG, seed)
        conc = rec["meta"]["labels"]["species"][0][
            "concentration_ppm"
        ]
        clean = simulate_ndir(
            lines=spec.lines,
            molecule="CH4",
            concentration_ppm=conc,
            temperature_K=rec["meta"]["conditions"]["temperature_K"],
            pressure_atm=rec["meta"]["conditions"]["pressure_atm"],
            path_length_m=spec.path_length_m,
            source_temperature_K=800.0,
            active_filter_center_cm1=6047.0,
            active_filter_fwhm_cm1=2.0,
            reference_filter_center_cm1=6040.0,
            reference_filter_fwhm_cm1=2.0,
            filter_shape="gaussian",
            n_integration_points=500,
        )
        assert rec["arrays"]["ratio_clean"] == pytest.approx(
            clean["ratio"], rel=1e-10,
        )

    def test_meta_has_required_fields(self):
        seed = np.random.SeedSequence(42)
        rec = generate_ndir_record(
            _make_spec(), TEST_NDIR_CONFIG, seed,
        )
        meta = rec["meta"]
        assert "record_id" in meta
        assert meta["schema_version"] == "0.2"
        assert meta["data_origin"] == "simulated"
        assert "provenance" in meta
        assert "signals" in meta
        assert "labels" in meta
        assert "conditions" in meta
        assert "instrument" in meta

    def test_signals_entries(self):
        seed = np.random.SeedSequence(42)
        rec = generate_ndir_record(
            _make_spec(), TEST_NDIR_CONFIG, seed,
        )
        signals = rec["meta"]["signals"]
        for key in ("active_channel", "reference_channel", "ratio"):
            assert key in signals
            assert signals[key]["n_samples"] == 1


class TestNoiseEffects:
    def test_noise_makes_ratio_differ_from_clean(self):
        spec = _make_spec()
        cfg = _noisy_config()
        seed = np.random.SeedSequence(99)
        rec = generate_ndir_record(spec, cfg, seed)
        assert rec["arrays"]["ratio"] != pytest.approx(
            rec["arrays"]["ratio_clean"], abs=0,
        )

    def test_zero_noise_ratio_equals_clean(self):
        spec = _make_spec()
        seed = np.random.SeedSequence(42)
        rec = generate_ndir_record(spec, TEST_NDIR_CONFIG, seed)
        assert rec["arrays"]["ratio"] == pytest.approx(
            rec["arrays"]["ratio_clean"], rel=1e-12,
        )

    def test_higher_noise_larger_deviation(self):
        spec = _make_spec()
        n_trials = 50
        deviations_low = []
        deviations_high = []
        for i in range(n_trials):
            seed = np.random.SeedSequence(1000 + i)
            cfg_low = _noisy_config(
                detector={"white_noise_rel": 1e-5},
                source={
                    "temperature_K": 800.0,
                    "temperature_drift_K_per_s": 0.0,
                    "intensity_fluctuation_rel": 0.0,
                },
            )
            rec_low = generate_ndir_record(spec, cfg_low, seed)
            deviations_low.append(
                abs(
                    rec_low["arrays"]["ratio"]
                    - rec_low["arrays"]["ratio_clean"]
                )
            )

            seed2 = np.random.SeedSequence(1000 + i)
            cfg_high = _noisy_config(
                detector={"white_noise_rel": 1e-2},
                source={
                    "temperature_K": 800.0,
                    "temperature_drift_K_per_s": 0.0,
                    "intensity_fluctuation_rel": 0.0,
                },
            )
            rec_high = generate_ndir_record(spec, cfg_high, seed2)
            deviations_high.append(
                abs(
                    rec_high["arrays"]["ratio"]
                    - rec_high["arrays"]["ratio_clean"]
                )
            )
        assert np.mean(deviations_high) > np.mean(deviations_low)


class TestSourceDrift:
    def test_drift_changes_ratio_over_time(self):
        spec = _make_spec()
        cfg = {
            **TEST_NDIR_CONFIG,
            "source": {
                "temperature_K": 800.0,
                "temperature_drift_K_per_s": 10.0,
                "intensity_fluctuation_rel": 0.0,
            },
        }
        seed0 = np.random.SeedSequence(50)
        seed1 = np.random.SeedSequence(50)
        rec_t0 = generate_ndir_record(
            spec, cfg, seed0, scan_time_s=0.0,
        )
        rec_t100 = generate_ndir_record(
            spec, cfg, seed1, scan_time_s=100.0,
        )
        assert rec_t0["arrays"]["ratio_clean"] != pytest.approx(
            rec_t100["arrays"]["ratio_clean"], abs=0,
        )


class TestConcentrationVariation:
    def test_higher_concentration_lower_ratio(self):
        spec_low = _make_spec(
            concentration_ppm_low=10.0,
            concentration_ppm_high=10.0,
        )
        spec_high = _make_spec(
            concentration_ppm_low=500.0,
            concentration_ppm_high=500.0,
        )
        seed_low = np.random.SeedSequence(7)
        seed_high = np.random.SeedSequence(7)
        rec_low = generate_ndir_record(
            spec_low, TEST_NDIR_CONFIG, seed_low,
        )
        rec_high = generate_ndir_record(
            spec_high, TEST_NDIR_CONFIG, seed_high,
        )
        assert (
            rec_low["arrays"]["ratio_clean"]
            > rec_high["arrays"]["ratio_clean"]
        )

    def test_zero_concentration_positive_ratio(self):
        spec = _make_spec(
            concentration_ppm_low=1e-10,
            concentration_ppm_high=1e-10,
        )
        seed = np.random.SeedSequence(8)
        rec = generate_ndir_record(spec, TEST_NDIR_CONFIG, seed)
        assert rec["arrays"]["ratio_clean"] > 0.0


class TestReproducibility:
    def test_same_seed_identical_output(self):
        spec = _make_spec()
        seed_a = np.random.SeedSequence(42)
        seed_b = np.random.SeedSequence(42)
        rec_a = generate_ndir_record(spec, TEST_NDIR_CONFIG, seed_a)
        rec_b = generate_ndir_record(spec, TEST_NDIR_CONFIG, seed_b)
        assert rec_a["meta"] == rec_b["meta"]
        for k in rec_a["arrays"]:
            assert rec_a["arrays"][k] == rec_b["arrays"][k]

    def test_different_seed_different_output(self):
        spec = _make_spec(
            concentration_ppm_low=1.0,
            concentration_ppm_high=1000.0,
            log_uniform_concentration=True,
        )
        cfg = _noisy_config()
        seed_a = np.random.SeedSequence(1)
        seed_b = np.random.SeedSequence(2)
        rec_a = generate_ndir_record(spec, cfg, seed_a)
        rec_b = generate_ndir_record(spec, cfg, seed_b)
        assert rec_a["arrays"]["ratio"] != rec_b["arrays"]["ratio"]


class TestDataset:
    def test_returns_correct_count(self):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, TEST_NDIR_CONFIG, 5, master_seed=10,
        )
        assert len(recs) == 5

    def test_all_records_have_valid_structure(self):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, TEST_NDIR_CONFIG, 3, master_seed=11,
        )
        for rec in recs:
            assert "meta" in rec and "arrays" in rec
            assert rec["meta"]["technique"] == "NDIR"
            for k in (
                "active_channel",
                "reference_channel",
                "ratio",
                "ratio_clean",
            ):
                assert k in rec["arrays"]
                assert isinstance(rec["arrays"][k], float)

    def test_unique_record_ids(self):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, TEST_NDIR_CONFIG, 10, master_seed=12,
        )
        ids = [r["meta"]["record_id"] for r in recs]
        assert len(set(ids)) == 10


class TestTimeSeries:
    def test_returns_correct_count(self):
        spec = _make_spec()
        recs = generate_ndir_time_series(
            spec,
            TEST_NDIR_CONFIG,
            n_scans=5,
            master_seed=20,
            scan_interval_s=1.0,
        )
        assert len(recs) == 5

    def test_frozen_instrument_shared(self):
        spec = _make_spec()
        recs = generate_ndir_time_series(
            spec,
            TEST_NDIR_CONFIG,
            n_scans=4,
            master_seed=21,
            scan_interval_s=1.0,
        )
        inst_ids = {
            r["meta"]["provenance"]["instrument_config_id"]
            for r in recs
        }
        assert len(inst_ids) == 1

    def test_all_records_valid(self):
        spec = _make_spec()
        recs = generate_ndir_time_series(
            spec,
            TEST_NDIR_CONFIG,
            n_scans=3,
            master_seed=22,
            scan_interval_s=2.0,
        )
        for rec in recs:
            assert rec["meta"]["technique"] == "NDIR"
            assert isinstance(rec["arrays"]["ratio"], float)
