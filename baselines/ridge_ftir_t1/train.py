#!/usr/bin/env python
"""Ridge-regression baseline for FTIR T1 concentration regression.

Input: ftir_spectrum (recovered FTIR spectrum after apodized FFT).
Target: CH4 concentration [ppm].

Reproduce:
    python baselines/ridge_ftir_t1/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_ftir_split, write_predictions_csv  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> int:
    X_tr, y_tr, _ = load_ftir_split("ch4-ftir-t1-train-v0")
    X_va, y_va, _ = load_ftir_split("ch4-ftir-t1-val-v0")
    X_te, _, ids_te = load_ftir_split("ch4-ftir-t1-test-v0")
    X_ho, _, ids_ho = load_ftir_split("ch4-ftir-t3-heldout-v0")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)

    best_alpha, best_mae = None, np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
        mae = float(np.mean(np.abs(model.predict(X_va_s) - y_va)))
        print(f"alpha={alpha:g}: val MAE {mae:.4f} ppm")
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    model = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)

    y_te = model.predict(scaler.transform(X_te))
    y_ho = model.predict(scaler.transform(X_ho))

    write_predictions_csv(OUT / "predictions_ftir-t1-test.csv", ids_te, y_te)
    write_predictions_csv(OUT / "predictions_ftir-t3-heldout.csv", ids_ho, y_ho)

    X_te_full, y_te_full, _ = load_ftir_split("ch4-ftir-t1-test-v0")
    mae_test = float(np.mean(np.abs(y_te - y_te_full)))
    mape_test = float(np.mean(np.abs((y_te - y_te_full) / np.maximum(y_te_full, 0.01)))) * 100

    X_ho_full, y_ho_full, _ = load_ftir_split("ch4-ftir-t3-heldout-v0")
    mae_ho = float(np.mean(np.abs(y_ho - y_ho_full)))

    scores_t1 = {"mae_ppm": round(mae_test, 4), "mape_pct": round(mape_test, 2)}
    scores_t3 = {
        "mae_ppm": round(mae_ho, 4),
        "degradation_ratio": round(mae_ho / max(mae_test, 0.001), 2),
    }

    (OUT / "scores_ftir_t1.json").write_text(json.dumps(scores_t1, indent=2))
    (OUT / "scores_ftir_t3.json").write_text(json.dumps(scores_t3, indent=2))
    (OUT / "hyperparams.json").write_text(
        json.dumps({"alpha": best_alpha, "val_mae_ppm": best_mae, "scaler": "standard"})
    )

    print(f"\nFTIR T1 test: MAE={mae_test:.4f} ppm, MAPE={mape_test:.2f}%")
    print(f"FTIR T3 heldout: MAE={mae_ho:.4f} ppm, degradation={mae_ho/max(mae_test,0.001):.2f}x")
    print(f"Selected alpha={best_alpha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
