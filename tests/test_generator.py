"""Generator + IO tests: reproducibility, schema validity, HDF5 round-trip."""

from pathlib import Path

import numpy as np
import pytest

from opensensorsim.generator import GenerationSpec, generate_dataset, generate_record
from opensensorsim.instrument.sampling import load_instrument_config
from opensensorsim.io import read_records, write_records
from opensensorsim.physics import demo_ch4_2nu3
from opensensorsim.validate import validate_record

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "instruments"


def make_spec(n_points: int = 400) -> GenerationSpec:
    return GenerationSpec(lines=demo_ch4_2nu3(), n_points=n_points)


@pytest.fixture(scope="module")
def da_cfg():
    return load_instrument_config(CONFIG_DIR / "vi-da-medium-02.yaml")


@pytest.fixture(scope="module")
def wms_cfg():
    return load_instrument_config(CONFIG_DIR / "vi-wms-easy-04.yaml")


class TestReproducibility:
    def test_same_master_seed_bit_identical(self, da_cfg):
        a = generate_dataset(make_spec(), da_cfg, n_records=3, master_seed=1234)
        b = generate_dataset(make_spec(), da_cfg, n_records=3, master_seed=1234)
        for ra, rb in zip(a, b):
            assert ra["meta"] == rb["meta"]
            for k in ra["arrays"]:
                assert ra["arrays"][k].tobytes() == rb["arrays"][k].tobytes()

    def test_different_seed_differs(self, da_cfg):
        a = generate_dataset(make_spec(), da_cfg, 1, master_seed=1)
        b = generate_dataset(make_spec(), da_cfg, 1, master_seed=2)
        assert a[0]["arrays"]["raw_scan"].tobytes() != b[0]["arrays"]["raw_scan"].tobytes()

    def test_single_record_regenerable_independently(self, da_cfg):
        full = generate_dataset(make_spec(), da_cfg, 5, master_seed=99)
        child_2 = np.random.SeedSequence(99).spawn(5)[2]
        solo = generate_record(make_spec(), da_cfg, child_2)
        assert solo["meta"] == full[2]["meta"]
        assert solo["arrays"]["raw_scan"].tobytes() == full[2]["arrays"]["raw_scan"].tobytes()


class TestSchemaValidity:
    def test_da_records_validate(self, da_cfg):
        for rec in generate_dataset(make_spec(), da_cfg, 5, master_seed=7):
            assert validate_record(rec["meta"]) == [], rec["meta"]["record_id"]

    def test_wms_records_validate(self, wms_cfg):
        for rec in generate_dataset(make_spec(200), wms_cfg, 2, master_seed=7):
            assert validate_record(rec["meta"]) == []
            assert "demod_2f" in rec["arrays"]
            assert rec["meta"]["technique"] == "TDLAS-WMS"

    def test_provenance_carries_sampled_parameters(self, da_cfg):
        rec = generate_dataset(make_spec(), da_cfg, 1, master_seed=5)[0]
        sampled = rec["meta"]["provenance"]["noise_config"]["sampled"]
        # Distributions must have been collapsed to concrete numbers
        assert isinstance(sampled["detector"]["white_noise_rel"], float)
        assert isinstance(sampled["laser"]["scan_range_cm1"], float)


class TestPhysicalSanity:
    def test_absorption_dip_visible_in_raw_scan(self, da_cfg):
        spec = make_spec(2000)
        spec.concentration_ppm_low = spec.concentration_ppm_high = 500.0
        rec = generate_dataset(spec, da_cfg, 1, master_seed=11)[0]
        clean = rec["arrays"]["absorbance_clean"]
        assert clean.max() > 1e-3  # line present
        assert clean.min() >= 0.0

    def test_noise_level_scales_with_tier(self):
        easy = load_instrument_config(CONFIG_DIR / "vi-da-easy-01.yaml")
        hard = load_instrument_config(CONFIG_DIR / "vi-da-hard-03.yaml")
        spec = make_spec(2000)
        spec.concentration_ppm_low = spec.concentration_ppm_high = 1e-3  # ~no gas

        def hf_noise(cfg, seed):
            rec = generate_dataset(spec, cfg, 1, master_seed=seed)[0]
            raw = rec["arrays"]["raw_scan"]
            return np.std(np.diff(raw))  # high-frequency component

        assert hf_noise(hard, 3) > hf_noise(easy, 3)


class TestHDF5RoundTrip:
    def test_write_read_lossless(self, da_cfg, tmp_path):
        recs = generate_dataset(make_spec(), da_cfg, 4, master_seed=21)
        path = tmp_path / "test.h5"
        write_records(path, recs)
        back = read_records(path)
        by_id = {r["meta"]["record_id"]: r for r in back}
        assert len(back) == 4
        for rec in recs:
            b = by_id[rec["meta"]["record_id"]]
            assert b["meta"] == rec["meta"]
            for k in rec["arrays"]:
                assert np.array_equal(b["arrays"][k], rec["arrays"][k])

    def test_write_rejects_invalid_record(self, da_cfg, tmp_path):
        recs = generate_dataset(make_spec(), da_cfg, 1, master_seed=22)
        del recs[0]["meta"]["conditions"]
        with pytest.raises(ValueError):
            write_records(tmp_path / "bad.h5", recs)
