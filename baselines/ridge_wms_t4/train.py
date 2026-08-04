#!/usr/bin/env python
"""Ridge-regression baseline for T4 (WMS 2f concentration regression).

Deliberately simple linear reference: per-feature standardization + ridge on
the 2f demodulated signal. Alpha selected on the official validation split.
Deterministic (no stochastic steps). Reproduce:

    python baselines/ridge_wms_t4/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_wms_split, write_predictions_csv  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> int:
    X_tr, y_tr, _ = load_wms_split("ch4-t4-train-v0")
    X_va, y_va, _ = load_wms_split("ch4-t4-val-v0")
    X_te, _, ids_te = load_wms_split("ch4-t4-test-v0")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)

    best_alpha, best_mae = None, np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
        mae = float(np.mean(np.abs(model.predict(X_va_s) - y_va)))
        print(f"alpha={alpha:g}: val MAE {mae:.3f} ppm")
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    model = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)
    preds = model.predict(scaler.transform(X_te))
    write_predictions_csv(OUT / "predictions_t4-test.csv", ids_te, preds)

    (OUT / "hyperparams.json").write_text(
        json.dumps({"alpha": best_alpha, "val_mae_ppm": best_mae, "scaler": "standard"})
    )
    print(f"selected alpha={best_alpha}, predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
