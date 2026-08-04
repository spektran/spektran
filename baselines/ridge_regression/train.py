#!/usr/bin/env python
"""Ridge-regression baseline for T1 (and T3 held-out evaluation).

Deliberately simple linear reference: per-feature standardization + ridge on
the raw scan. Alpha selected on the official validation split. Deterministic
(no stochastic steps). Reproduce:

    python baselines/ridge_regression/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_split, write_predictions_csv  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> int:
    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    X_va, y_va, _ = load_split("ch4-t1-val-v0")
    X_te, _, ids_te = load_split("ch4-t1-test-v0")
    X_ho, _, ids_ho = load_split("ch4-t3-test-heldout-v0")

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
    for X, ids, tag in [(X_te, ids_te, "t1-test"), (X_ho, ids_ho, "t3-test-heldout")]:
        preds = model.predict(scaler.transform(X))
        write_predictions_csv(OUT / f"predictions_{tag}.csv", ids, preds)

    (OUT / "hyperparams.json").write_text(
        json.dumps({"alpha": best_alpha, "val_mae_ppm": best_mae, "scaler": "standard"})
    )
    print(f"selected alpha={best_alpha}, predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
