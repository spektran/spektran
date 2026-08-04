"""Tests for v0.2 benchmark metrics and task specs."""
from pathlib import Path

import numpy as np
import pytest


def test_peak_height_ratio_mae():
    from spektran.benchmark.metrics import peak_height_ratio_mae

    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    result = peak_height_ratio_mae(y_true, y_pred)
    assert result == pytest.approx(10.0)


def test_allan_variance_single_series():
    from spektran.benchmark.metrics import allan_variance

    rng = np.random.default_rng(42)
    white = rng.normal(0.0, 1.0, 1000)
    av = allan_variance(white, tau_points=10)
    # np.geomspace(1, n//2, 10).astype(int) does not always yield 10 unique
    # taus (dedup + segments<1 exclusion near the top of the range), so we
    # assert the shape invariant instead of an exact count.
    assert 0 < len(av["taus"]) <= 10
    assert len(av["adevs"]) == len(av["taus"])
    assert all(a > 0 for a in av["adevs"])


def test_ood_auroc_perfect():
    from spektran.benchmark.metrics import ood_auroc

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ood_auroc(y_true, y_scores) == pytest.approx(1.0)


def test_ood_auroc_random():
    from spektran.benchmark.metrics import ood_auroc

    rng = np.random.default_rng(42)
    y_true = np.concatenate([np.zeros(500), np.ones(500)])
    y_scores = rng.uniform(0, 1, 1000)
    result = ood_auroc(y_true, y_scores)
    assert 0.4 < result < 0.6


def test_task_specs_include_t4_t5_t6():
    from spektran.benchmark.tasks import TASK_SPECS

    assert "T4-wms-concentration" in TASK_SPECS
    assert "T5-drift-compensation" in TASK_SPECS
    assert "T6-ood-instrument" in TASK_SPECS


def test_evaluate_drift_separates_series_boundaries(tmp_path):
    """A multi-series truth file must never let Allan variance cross a series
    boundary -- this is the whole reason evaluate_drift recovers series from
    truth-concentration jumps instead of just flattening the file."""
    from spektran.benchmark.evaluate import evaluate_drift
    from spektran.generator import GenerationSpec, generate_time_series
    from spektran.instrument.sampling import load_instrument_config
    from spektran.io import write_time_series
    from spektran.physics import demo_ch4_2nu3

    cfg_dir = Path(__file__).resolve().parents[1] / "configs" / "instruments"
    inst = load_instrument_config(cfg_dir / "vi-da-easy-01.yaml")

    all_records = []
    for i, c in enumerate([80.0, 120.0]):
        spec = GenerationSpec(
            lines=demo_ch4_2nu3(), n_points=100,
            concentration_ppm_low=c, concentration_ppm_high=c,
            log_uniform_concentration=False,
        )
        all_records.extend(
            generate_time_series(spec, inst, n_scans=6, master_seed=500 + i, scan_interval_s=1.0)
        )

    truth_path = tmp_path / "truth.h5"
    write_time_series(truth_path, all_records, scan_interval_s=1.0)

    preds_path = tmp_path / "preds.csv"
    with open(preds_path, "w") as f:
        f.write("record_id,concentration_ppm\n")
        for r in all_records:
            truth = r["meta"]["labels"]["species"][0]["concentration_ppm"]
            f.write(f"{r['meta']['record_id']},{truth + 0.5}\n")

    scores = evaluate_drift(truth_path, preds_path)
    assert scores["n_scans"] == 12
    assert scores["n_series"] == 2
    assert scores["mae_ppm"] == pytest.approx(0.5, abs=1e-6)
    assert len(scores["adev_curve"]) == len(scores["adev_taus_s"])
