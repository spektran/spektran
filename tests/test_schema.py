"""Schema meta-validation and record validation tests."""

import uuid

from opengasspec.validate import instrument_validator, record_validator, validate_record


def make_valid_da_record() -> dict:
    return {
        "record_id": str(uuid.uuid4()),
        "schema_version": "0.1",
        "data_origin": "simulated",
        "technique": "TDLAS-DA",
        "provenance": {
            "generator_version": "0.1.0.dev0+g0000000",
            "hitran_data_version": "n/a (demo)",
            "random_seed": 42,
            "instrument_config_id": "vi-clean-00",
            "noise_config": {},
        },
        "signals": {
            "absorbance": {
                "array_ref": "/records/r0/absorbance",
                "n_samples": 2000,
                "wavenumber_axis": {"start_cm1": 6046.0, "step_cm1": 0.001},
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
            "laser": {"center_wavenumber_cm1": 6047.0, "scan_range_cm1": 2.0},
            "detector": {"type": "InGaAs photodiode", "bandwidth_Hz": 1.0e6},
            "target_lines": [{"hitran_molecule_id": 6, "wavenumber_cm1": 6046.9647}],
        },
    }


class TestMetaValidation:
    def test_record_schema_is_valid(self):
        record_validator()  # check_schema inside

    def test_instrument_schema_is_valid(self):
        instrument_validator()


class TestRecordValidation:
    def test_valid_da_record_passes(self):
        assert validate_record(make_valid_da_record()) == []

    def test_missing_provenance_fails_for_simulated(self):
        rec = make_valid_da_record()
        del rec["provenance"]
        assert any("provenance" in e for e in validate_record(rec))

    def test_wms_requires_modulation_and_demod(self):
        rec = make_valid_da_record()
        rec["technique"] = "TDLAS-WMS"
        errors = validate_record(rec)
        assert errors, "WMS record without modulation/demod signals must fail"

    def test_wms_valid_with_modulation_and_2f(self):
        rec = make_valid_da_record()
        rec["technique"] = "TDLAS-WMS"
        rec["instrument"]["modulation"] = {"frequency_Hz": 10000.0, "depth_cm1": 0.05}
        rec["signals"]["demod_2f"] = {
            "array_ref": "/records/r0/demod_2f",
            "n_samples": 2000,
            "lowpass_cutoff_Hz": 100.0,
        }
        assert validate_record(rec) == []

    def test_unknown_field_rejected(self):
        rec = make_valid_da_record()
        rec["surprise"] = 1
        assert any("surprise" in e for e in validate_record(rec))

    def test_negative_concentration_rejected(self):
        rec = make_valid_da_record()
        rec["labels"]["species"][0]["concentration_ppm"] = -1.0
        assert validate_record(rec)


class TestInstrumentConfigValidation:
    def test_minimal_instrument_config(self):
        cfg = {
            "instrument_config_id": "vi-easy-01",
            "schema_version": "0.1",
            "technique": "TDLAS-DA",
            "laser": {
                "center_wavenumber_cm1": 6047.0,
                "scan_range_cm1": {"dist": "uniform", "low": 1.8, "high": 2.2},
            },
            "detector": {
                "white_noise_rel": {"dist": "loguniform", "low": 1e-5, "high": 1e-4}
            },
        }
        v = instrument_validator()
        assert list(v.iter_errors(cfg)) == []

    def test_bad_distribution_rejected(self):
        cfg = {
            "instrument_config_id": "vi-bad",
            "schema_version": "0.1",
            "technique": "TDLAS-DA",
            "laser": {
                "center_wavenumber_cm1": 6047.0,
                "scan_range_cm1": {"dist": "gaussian", "mu": 2.0},  # wrong name/fields
            },
            "detector": {},
        }
        v = instrument_validator()
        assert list(v.iter_errors(cfg))
