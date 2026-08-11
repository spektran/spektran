#!/usr/bin/env python
"""Random Forest + Gradient Boosting baselines for T1 (and T3 held-out).

Non-linear tree ensemble baselines widely used in TDLAS literature for
gas concentration retrieval. Two models trained side-by-side; the better
one on validation is used for final predictions. Reproduce:

    python baselines/random_forest_t1/train.py

References:
    Semi-supervised methane detection with TDLAS + Random Forest,
    Optoelectronics Letters (2025), doi:10.1007/s11801-025-4140-7
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_split, write_predictions_csv  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20260811


def main() -> int:
    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    X_va, y_va, _ = load_split("ch4-t1-val-v0")
    X_te, _, ids_te = load_split("ch4-t1-test-v0")
    X_ho, _, ids_ho = load_split("ch4-t3-test-heldout-v0")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)

    y_log = np.log1p(y_tr)

    results = {}

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_tr_s, y_log)
    rf_preds_va = np.expm1(rf.predict(X_va_s))
    rf_mae = float(np.mean(np.abs(rf_preds_va - y_va)))
    results["rf"] = {"model": rf, "val_mae_ppm": rf_mae}
    print(f"Random Forest: val MAE {rf_mae:.3f} ppm")

    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=SEED,
    )
    gbr.fit(X_tr_s, y_log)
    gbr_preds_va = np.expm1(gbr.predict(X_va_s))
    gbr_mae = float(np.mean(np.abs(gbr_preds_va - y_va)))
    results["gbr"] = {"model": gbr, "val_mae_ppm": gbr_mae}
    print(f"Gradient Boosting: val MAE {gbr_mae:.3f} ppm")

    best_name = min(results, key=lambda k: results[k]["val_mae_ppm"])
    best = results[best_name]
    print(f"Selected: {best_name} (MAE {best['val_mae_ppm']:.3f} ppm)")

    model = best["model"]
    for X, ids, tag in [(X_te, ids_te, "t1-test"), (X_ho, ids_ho, "t3-test-heldout")]:
        preds = np.expm1(model.predict(scaler.transform(X)))
        write_predictions_csv(OUT / f"predictions_{tag}.csv", ids, preds)

    (OUT / "hyperparams.json").write_text(json.dumps({
        "selected_model": best_name,
        "seed": SEED,
        "rf_val_mae_ppm": results["rf"]["val_mae_ppm"],
        "gbr_val_mae_ppm": results["gbr"]["val_mae_ppm"],
        "rf_params": {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 5},
        "gbr_params": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8},
        "target_transform": "log1p",
        "scaler": "standard",
    }, indent=2))
    print(f"Predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
