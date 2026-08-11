"""Generator + IO tests: reproducibility, schema validity, HDF5 round-trip."""

from pathlib import Path

import numpy as np
import pytest

from spektran.generator import GenerationSpec, generate_dataset, generate_record
from spektran.instrument.sampling import load_instrument_config
from spektran.io import read_records, write_records
from spektran.physics import demo_ch4_2nu3
from spektran.validate import validate_record

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
        spec.concentration_ppm_low = spec.concentration_ppm_high = 500.0

        def noise_metric(cfg, seed):
            rec = generate_dataset(spec, cfg, 1, master_seed=seed)[0]
            raw = rec["arrays"]["raw_scan"]
            smooth = np.convolve(raw, np.ones(20) / 20, mode="same")
            residual = raw - smooth
            ptp = max(float(np.ptp(raw)), 1e-12)
            return np.std(residual) / ptp

        easy_nrel = np.median([noise_metric(easy, s) for s in range(5)])
        hard_nrel = np.median([noise_metric(hard, s) for s in range(5)])
        assert hard_nrel > easy_nrel


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


class TestTimeSeries:
    def test_write_read_round_trip(self, da_cfg, tmp_path):
        from spektran.generator import generate_time_series
        from spektran.io import read_time_series, write_time_series

        recs = generate_time_series(
            make_spec(n_points=100), da_cfg, n_scans=5, master_seed=31, scan_interval_s=2.0
        )
        path = tmp_path / "ts.h5"
        write_time_series(path, recs, scan_interval_s=2.0)
        back, interval = read_time_series(path)

        assert interval == 2.0
        assert [r["meta"]["record_id"] for r in back] == [r["meta"]["record_id"] for r in recs]
        for a, b in zip(recs, back):
            assert a["meta"] == b["meta"]
            for k in a["arrays"]:
                assert np.array_equal(a["arrays"][k], b["arrays"][k])

    def test_frozen_instrument_shared_across_scans(self, da_cfg):
        from spektran.generator import generate_time_series

        recs = generate_time_series(
            make_spec(n_points=100), da_cfg, n_scans=4, master_seed=32, scan_interval_s=1.0
        )
        inst_ids = {r["meta"]["provenance"]["instrument_config_id"] for r in recs}
        assert len(inst_ids) == 1

    def test_fixed_concentration_series_is_bit_identical(self):
        """A degenerate (low == high) spec must yield the exact same truth every draw.

        This underpins evaluate_drift's series-boundary detection (a change in
        truth concentration marks a new series), so it must hold exactly, not
        just approximately.
        """
        from spektran.generator import GenerationSpec, sample_concentration

        spec = GenerationSpec(
            lines=demo_ch4_2nu3(),
            concentration_ppm_low=77.0,
            concentration_ppm_high=77.0,
            log_uniform_concentration=False,
        )
        rng = np.random.default_rng(5)
        vals = [sample_concentration(spec, rng) for _ in range(10)]
        assert all(v == 77.0 for v in vals)


def test_generate_record_with_interferent():
    """Records with interferents must list them in conditions and add their absorption."""
    import numpy as np

    from spektran.generator import GenerationSpec, generate_record
    from spektran.physics.hitran import demo_ch4_2nu3, demo_h2o

    spec = GenerationSpec(
        lines=demo_ch4_2nu3(),
        molecule="CH4",
        concentration_ppm_low=100.0,
        concentration_ppm_high=100.0,
        interferents=[
            {
                "molecule": "H2O",
                "lines": demo_h2o(),
                "concentration_ppm": 10000.0,
            }
        ],
    )
    inst_cfg = {
        "instrument_config_id": "test-multi",
        "schema_version": "0.1",
        "technique": "TDLAS-DA",
        "laser": {"center_wavenumber_cm1": 6047.0, "scan_range_cm1": 2.0},
        "detector": {},
    }
    seed = np.random.SeedSequence(42)
    rec = generate_record(spec, inst_cfg, seed)
    meta = rec["meta"]
    assert len(meta["labels"]["species"]) == 1  # only target species in labels
    assert "interferents" in meta["conditions"]
    assert meta["conditions"]["interferents"][0]["molecule"] == "H2O"
    assert meta["conditions"]["interferents"][0]["concentration_ppm"] == 10000.0
