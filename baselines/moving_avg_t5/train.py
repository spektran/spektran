#!/usr/bin/env python
"""Moving-average baseline for T5 (drift compensation).

Per-scan concentration via ridge on raw_scan, smoothed with a sliding window
per time series. Smoothing never crosses a series boundary (a boundary is a
change in true concentration -- each series is one frozen instrument's fixed-
concentration run; see `spektran.benchmark.evaluate.evaluate_drift`), so the
window is applied independently within each series and the results
concatenated back in truth order. The window is selected by MAE on the
TRAINING series themselves (never on test, matching the val-based tuning the
other baselines use) -- T5 v0 has no dedicated val split, so this is
in-sample rather than held-out, a known simplification for this baseline.
Reproduce:

    python baselines/moving_avg_t5/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_time_series_split, write_predictions_csv  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent

WINDOWS = [1, 3, 5, 10, 20, 50, 100]


def series_boundaries(y_true: np.ndarray) -> np.ndarray:
    """Index of each series' first scan (after the first), from truth jumps.

    Every scan in one `generate_time_series` run shares an exactly equal true
    concentration by construction, so a change marks a new series.
    """
    return np.flatnonzero(np.diff(y_true) != 0) + 1


def moving_average(series: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, renormalized at the edges.

    A plain ``np.convolve(series, np.ones(window) / window, mode="same")``
    implicitly zero-pads past the array ends, biasing edge values toward zero
    (verified: a constant 100-valued series gets pulled to 50 at its first
    sample for window=10) -- large enough on our 200-scan series to dominate
    MAE at window >= 5 and make every window choice look worse than none.
    Dividing by the actual overlap count at each position (a truncated/
    adaptive moving average, matching `pandas.rolling(center=True,
    min_periods=1)`) fixes this.
    """
    if window <= 1:
        return series.copy()
    kernel = np.ones(window)
    sums = np.convolve(series, kernel, mode="same")
    counts = np.convolve(np.ones_like(series), kernel, mode="same")
    return sums / counts


def smooth_per_series(y_pred_raw: np.ndarray, boundaries: np.ndarray, window: int) -> np.ndarray:
    blocks = np.split(y_pred_raw, boundaries)
    return np.concatenate([moving_average(b, window) for b in blocks])


def main() -> int:
    X_tr, y_tr, _ = load_time_series_split("ch4-t5-train-v0")
    X_te, y_te, ids_te = load_time_series_split("ch4-t5-test-v0")

    scaler = StandardScaler().fit(X_tr)
    model = Ridge(alpha=10.0).fit(scaler.transform(X_tr), y_tr)

    train_raw_preds = model.predict(scaler.transform(X_tr))
    train_boundaries = series_boundaries(y_tr)
    train_mae_unsmoothed = float(np.mean(np.abs(train_raw_preds - y_tr)))
    print(f"window=1 (raw): train MAE {train_mae_unsmoothed:.3f} ppm")

    best_w, best_mae = 1, train_mae_unsmoothed
    for w in WINDOWS[1:]:
        smoothed = smooth_per_series(train_raw_preds, train_boundaries, w)
        this_mae = float(np.mean(np.abs(smoothed - y_tr)))
        print(f"window={w}: train MAE {this_mae:.3f} ppm")
        if this_mae < best_mae:
            best_w, best_mae = w, this_mae

    test_raw_preds = model.predict(scaler.transform(X_te))
    test_boundaries = series_boundaries(y_te)
    final_preds = smooth_per_series(test_raw_preds, test_boundaries, best_w)
    test_mae = float(np.mean(np.abs(final_preds - y_te)))

    write_predictions_csv(OUT / "predictions_t5-test.csv", ids_te, final_preds)
    (OUT / "hyperparams.json").write_text(json.dumps({
        "ridge_alpha": 10.0,
        "window": best_w,
        "window_selected_on": "train (in-sample MAE, per-series smoothing)",
        "train_mae_ppm": best_mae,
        "test_mae_ppm": test_mae,
    }, indent=2))
    print(f"window={best_w} (selected on train), test MAE {test_mae:.3f} ppm, "
          f"predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
