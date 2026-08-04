"""Tests for v0.2 benchmark metrics and task specs."""
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
