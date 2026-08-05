"""Tests for the T7 cross-modality benchmark track."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

NDIR_INST = {
    "instrument_config_id": "vi-ndir-test-t7",
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
    "detector": {"white_noise_rel": 1e-4},
    "environment": {"temperature_K": 296.0, "pressure_atm": 1.0},
}


def test_cross_modality_dataset_config_valid():
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "datasets"
        / "ch4-cross-modality-test-v0.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["dataset_id"] == "ch4-cross-modality-test-v0"
    assert cfg["technique"] == "NDIR"
    assert cfg["n_records"] == 1000
    assert cfg["master_seed"] == 501001
    assert cfg["gas"]["molecule"] == "CH4"
    assert cfg["gas"]["concentration_ppm"]["low"] == 1.0
    assert cfg["gas"]["concentration_ppm"]["high"] == 1000.0
    assert len(cfg["instrument_config"]) == 3


def test_cross_modality_evaluation_with_ndir_records(tmp_path):
    """evaluate_concentration works on NDIR records (same labels structure)."""
    from spektran.benchmark.evaluate import evaluate_concentration
    from spektran.io import write_records
    from spektran.ndir_generator import NDIRGenerationSpec, generate_ndir_dataset
    from spektran.physics import demo_ch4_2nu3

    spec = NDIRGenerationSpec(lines=demo_ch4_2nu3(), path_length_m=10.0)
    records = generate_ndir_dataset(spec, NDIR_INST, n_records=5, master_seed=900)

    truth_path = tmp_path / "truth.h5"
    write_records(truth_path, records)

    preds_path = tmp_path / "preds.csv"
    with open(preds_path, "w") as f:
        f.write("record_id,concentration_ppm\n")
        for r in records:
            truth_c = r["meta"]["labels"]["species"][0][
                "concentration_ppm"
            ]
            f.write(f"{r['meta']['record_id']},{truth_c + 1.0}\n")

    scores = evaluate_concentration(truth_path, preds_path)
    assert scores["n_records"] == 5
    assert scores["mae_ppm"] == pytest.approx(1.0, abs=1e-6)
    assert scores["mape_pct"] > 0
    assert scores["rmse_ppm"] == pytest.approx(1.0, abs=1e-6)


def test_t7_task_registered():
    from spektran.benchmark.tasks import TASK_SPECS, TASKS

    assert "T7-cross-modality" in TASKS
    assert "T7-cross-modality" in TASK_SPECS
    spec = TASK_SPECS["T7-cross-modality"]
    assert spec.primary_metric == "mae"
    assert spec.input_signal == "ndir_ratio"


def test_t7_evaluate_dispatch_recognized():
    """The evaluate CLI accepts T7-cross-modality as a valid task choice."""
    from spektran.benchmark.evaluate import main

    # FileNotFoundError from missing truth file, NOT argparse "invalid choice"
    with pytest.raises((SystemExit, FileNotFoundError)):
        main(["--task", "T7-cross-modality", "--truth", "x.h5",
              "--predictions", "x.csv"])


def test_t7_cross_modality_degradation(tmp_path):
    """T7 computes cross_modality_degradation when --t1-mae is given."""
    from spektran.benchmark.evaluate import main
    from spektran.io import write_records
    from spektran.ndir_generator import NDIRGenerationSpec, generate_ndir_dataset
    from spektran.physics import demo_ch4_2nu3

    spec = NDIRGenerationSpec(lines=demo_ch4_2nu3(), path_length_m=10.0)
    records = generate_ndir_dataset(
        spec, NDIR_INST, n_records=5, master_seed=901,
    )

    truth_path = tmp_path / "truth.h5"
    write_records(truth_path, records)

    preds_path = tmp_path / "preds.csv"
    with open(preds_path, "w") as f:
        f.write("record_id,concentration_ppm\n")
        for r in records:
            truth_c = r["meta"]["labels"]["species"][0][
                "concentration_ppm"
            ]
            f.write(f"{r['meta']['record_id']},{truth_c + 5.0}\n")

    import json

    json_out = tmp_path / "scores.json"
    rc = main([
        "--task", "T7-cross-modality",
        "--truth", str(truth_path),
        "--predictions", str(preds_path),
        "--t1-mae", "2.5",
        "--json-out", str(json_out),
    ])
    assert rc == 0

    result = json.loads(json_out.read_text())
    assert result["task"] == "T7-cross-modality"
    assert result["scores"]["mae_ppm"] == pytest.approx(5.0, abs=1e-6)
    assert result["scores"]["cross_modality_degradation"] == pytest.approx(
        5.0 / 2.5, abs=1e-6,
    )
