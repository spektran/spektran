"""NDIR integration tests: schema validation, dataset generation, HDF5 round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spektran.io import read_records, write_records
from spektran.ndir_generator import (
    NDIRGenerationSpec,
    generate_ndir_dataset,
    generate_ndir_record,
)
from spektran.physics import demo_ch4_2nu3
from spektran.validate import validate_record

NDIR_INSTRUMENT_CFG = {
    "instrument_config_id": "vi-ndir-integ",
    "schema_version": "0.1",
    "technique": "NDIR",
    "source": {
        "temperature_K": 800.0,
        "temperature_drift_K_per_s": 0.0,
        "intensity_fluctuation_rel": 1e-4,
    },
    "filters": {
        "active_center_cm1": 6047.0,
        "active_fwhm_cm1": 2.0,
        "reference_center_cm1": 6040.0,
        "reference_fwhm_cm1": 2.0,
        "shape": "gaussian",
    },
    "detector": {
        "white_noise_rel": 1e-4,
    },
    "environment": {
        "temperature_K": 296.0,
        "pressure_atm": 1.0,
        "temperature_jitter_K": 0.1,
        "pressure_jitter_atm": 0.001,
    },
}


def _make_spec(**overrides) -> NDIRGenerationSpec:
    defaults = dict(
        lines=demo_ch4_2nu3(),
        molecule="CH4",
        concentration_ppm_low=10.0,
        concentration_ppm_high=500.0,
        log_uniform_concentration=True,
        path_length_m=10.0,
        n_integration_points=500,
    )
    defaults.update(overrides)
    return NDIRGenerationSpec(**defaults)


class TestNDIRSchemaValidation:
    def test_ndir_record_passes_schema(self):
        spec = _make_spec()
        seed = np.random.SeedSequence(100)
        rec = generate_ndir_record(spec, NDIR_INSTRUMENT_CFG, seed)
        errors = validate_record(rec["meta"])
        assert errors == [], f"NDIR record should validate: {errors}"

    def test_ndir_dataset_all_records_validate(self):
        spec = _make_spec()
        recs = generate_ndir_dataset(spec, NDIR_INSTRUMENT_CFG, 5, 200)
        for rec in recs:
            errors = validate_record(rec["meta"])
            assert errors == [], (
                f"Record {rec['meta']['record_id']} failed: {errors}"
            )

    def test_ndir_record_missing_ratio_fails(self):
        spec = _make_spec()
        seed = np.random.SeedSequence(101)
        rec = generate_ndir_record(spec, NDIR_INSTRUMENT_CFG, seed)
        del rec["meta"]["signals"]["ratio"]
        errors = validate_record(rec["meta"])
        assert errors, "NDIR record without ratio signal should fail"

    def test_tdlas_da_still_validates(self):
        """Existing TDLAS-DA records must still pass after schema changes."""
        import uuid

        rec = {
            "record_id": str(uuid.uuid4()),
            "schema_version": "0.1",
            "data_origin": "simulated",
            "technique": "TDLAS-DA",
            "provenance": {
                "generator_version": "0.1.0",
                "hitran_data_version": "n/a (demo)",
                "random_seed": 42,
                "instrument_config_id": "vi-clean-00",
                "noise_config": {},
            },
            "signals": {
                "absorbance": {
                    "array_ref": "/records/r0/absorbance",
                    "n_samples": 2000,
                    "wavenumber_axis": {
                        "start_cm1": 6046.0,
                        "step_cm1": 0.001,
                    },
                }
            },
            "labels": {
                "species": [
                    {
                        "molecule": "CH4",
                        "hitran_molecule_id": 6,
                        "concentration_ppm": 100.0,
                        "concentration_uncertainty_ppm": 0.0,
                    }
                ]
            },
            "conditions": {
                "temperature_K": 296.0,
                "pressure_atm": 1.0,
                "path_length_m": 10.0,
                "matrix_gas": "N2",
            },
            "instrument": {
                "laser": {
                    "center_wavenumber_cm1": 6047.0,
                    "scan_range_cm1": 2.0,
                },
                "detector": {"type": "InGaAs photodiode"},
                "target_lines": [
                    {"hitran_molecule_id": 6, "wavenumber_cm1": 6046.96}
                ],
            },
        }
        errors = validate_record(rec)
        assert errors == [], f"TDLAS-DA record should still validate: {errors}"


class TestNDIRDatasetGeneration:
    def test_generates_correct_count(self):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, NDIR_INSTRUMENT_CFG, 10, master_seed=300,
        )
        assert len(recs) == 10

    def test_all_records_technique_ndir(self):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, NDIR_INSTRUMENT_CFG, 10, master_seed=301,
        )
        for rec in recs:
            assert rec["meta"]["technique"] == "NDIR"

    def test_all_ratios_positive(self):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, NDIR_INSTRUMENT_CFG, 10, master_seed=302,
        )
        for rec in recs:
            assert rec["arrays"]["ratio"] > 0

    def test_concentrations_in_expected_range(self):
        spec = _make_spec(
            concentration_ppm_low=10.0,
            concentration_ppm_high=500.0,
        )
        recs = generate_ndir_dataset(
            spec, NDIR_INSTRUMENT_CFG, 20, master_seed=303,
        )
        for rec in recs:
            conc = rec["meta"]["labels"]["species"][0]["concentration_ppm"]
            assert 10.0 <= conc <= 500.0

    def test_unique_record_ids(self):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, NDIR_INSTRUMENT_CFG, 15, master_seed=304,
        )
        ids = [r["meta"]["record_id"] for r in recs]
        assert len(set(ids)) == 15


class TestNDIRHDF5RoundTrip:
    def test_write_and_read_back(self, tmp_path: Path):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, NDIR_INSTRUMENT_CFG, 5, master_seed=400,
        )
        out = tmp_path / "ndir_test.h5"
        write_records(out, recs, validate=True)
        loaded = read_records(out)
        assert len(loaded) == 5

    def test_scalar_values_survive_roundtrip(self, tmp_path: Path):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, NDIR_INSTRUMENT_CFG, 3, master_seed=401,
        )
        out = tmp_path / "ndir_scalars.h5"
        write_records(out, recs, validate=True)
        loaded = read_records(out)
        loaded_by_id = {r["meta"]["record_id"]: r for r in loaded}
        for orig in recs:
            rid = orig["meta"]["record_id"]
            back = loaded_by_id[rid]
            for key in ("active_channel", "reference_channel", "ratio"):
                assert float(back["arrays"][key]) == pytest.approx(
                    orig["arrays"][key], rel=1e-12,
                )

    def test_metadata_preserved(self, tmp_path: Path):
        spec = _make_spec()
        recs = generate_ndir_dataset(
            spec, NDIR_INSTRUMENT_CFG, 3, master_seed=402,
        )
        out = tmp_path / "ndir_meta.h5"
        write_records(out, recs, validate=True)
        loaded = read_records(out)
        loaded_by_id = {r["meta"]["record_id"]: r for r in loaded}
        for orig in recs:
            rid = orig["meta"]["record_id"]
            back = loaded_by_id[rid]
            assert back["meta"]["technique"] == "NDIR"
            assert (
                back["meta"]["labels"]["species"][0]["concentration_ppm"]
                == orig["meta"]["labels"]["species"][0]["concentration_ppm"]
            )
            assert (
                back["meta"]["conditions"]["temperature_K"]
                == orig["meta"]["conditions"]["temperature_K"]
            )
