#!/usr/bin/env python
"""Evaluate DL baseline results (U-Net T2, TCN T5) after training completes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spektran.benchmark.evaluate import evaluate_denoising  # noqa: E402
from spektran.benchmark.metrics import spectral_rmse, peak_weighted_rmse  # noqa: E402
from spektran.io import read_records  # noqa: E402

DATA = REPO / "data"
BASELINES = REPO / "baselines"


def evaluate_unet_t2() -> dict | None:
    pred_path = BASELINES / "unet_t2" / "predictions_t2-test.h5"
    truth_path = DATA / "ch4-t1-test-v0.h5"
    if not pred_path.exists():
        print("U-Net T2: predictions not found yet")
        return None

    scores = evaluate_denoising(truth_path, pred_path)
    out = BASELINES / "unet_t2" / "scores_t2.json"
    out.write_text(json.dumps(scores, indent=2))
    print(f"U-Net T2: spectral RMSE = {scores['spectral_rmse']:.6f}, "
          f"peak-weighted RMSE = {scores['peak_weighted_rmse']:.6f}")
    return scores


def evaluate_tcn_t5() -> dict | None:
    pred_path = BASELINES / "tcn_t5" / "predictions_t5-test.csv"
    if not pred_path.exists():
        print("TCN T5: predictions not found yet")
        return None

    from spektran.benchmark.evaluate import evaluate_drift  # noqa: E402
    truth_path = DATA / "ch4-t5-test-v0.h5"
    scores = evaluate_drift(truth_path, pred_path)
    out = BASELINES / "tcn_t5" / "scores_t5.json"
    out.write_text(json.dumps(scores, indent=2))
    print(f"TCN T5: MAE = {scores['mae_ppm']:.4f}, "
          f"ADEV@1s = {scores['adev_1s']:.6f}, "
          f"ADEV@last = {scores['adev_last']:.6f}")
    return scores


def main() -> int:
    print("=" * 60)
    print("DL Baseline Evaluation")
    print("=" * 60)

    unet = evaluate_unet_t2()
    tcn = evaluate_tcn_t5()

    if unet is None and tcn is None:
        print("\nNo DL results ready yet.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
