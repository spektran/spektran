#!/usr/bin/env python
"""Ridge-regression baseline for T9 (temperature regression).

Fixed CH4 concentration, regress gas temperature from line-shape changes.
Reproduce:

    python baselines/ridge_temp_t9/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_temperature_split, write_temperature_csv  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> int:
    X_tr, y_tr, _ = load_temperature_split("ch4-t9-train-v0")
    X_te, _, ids_te = load_temperature_split("ch4-t9-test-v0")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    best_alpha, best_mae = None, np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        m = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
        mae = float(np.mean(np.abs(m.predict(X_tr_s) - y_tr)))
        print(f"alpha={alpha:g}: train MAE = {mae:.2f} K")
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    model = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    write_temperature_csv(OUT / "predictions_t9-test.csv", ids_te, y_pred)

    (OUT / "hyperparams.json").write_text(json.dumps({
        "alpha": best_alpha,
        "train_mae_K": best_mae,
        "scaler": "standard",
    }))
    print(f"selected alpha={best_alpha}, predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
