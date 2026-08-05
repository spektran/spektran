#!/usr/bin/env python
"""Ridge-regression baseline for T8 (multi-species: CH4 + H2O).

Two independent ridge regressors, one per target species. Reproduce:

    python baselines/ridge_multispecies_t8/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_multispecies_split, write_multispecies_csv  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> int:
    X_tr, ch4_tr, h2o_tr, _ = load_multispecies_split("ch4-h2o-t8-train-v0")
    X_te, _, _, ids_te = load_multispecies_split("ch4-h2o-t8-test-v0")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    best_alpha, best_mae = None, np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        m_ch4 = Ridge(alpha=alpha).fit(X_tr_s, ch4_tr)
        m_h2o = Ridge(alpha=alpha).fit(X_tr_s, h2o_tr)
        mae_ch4 = float(np.mean(np.abs(m_ch4.predict(X_tr_s) - ch4_tr)))
        mae_h2o = float(np.mean(np.abs(m_h2o.predict(X_tr_s) - h2o_tr)))
        mae = (mae_ch4 + mae_h2o) / 2
        print(f"alpha={alpha:g}: train MAE ch4={mae_ch4:.1f} h2o={mae_h2o:.1f} agg={mae:.1f}")
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    m_ch4 = Ridge(alpha=best_alpha).fit(X_tr_s, ch4_tr)
    m_h2o = Ridge(alpha=best_alpha).fit(X_tr_s, h2o_tr)
    ch4_pred = m_ch4.predict(X_te_s)
    h2o_pred = m_h2o.predict(X_te_s)
    write_multispecies_csv(OUT / "predictions_t8-test.csv", ids_te, ch4_pred, h2o_pred)

    (OUT / "hyperparams.json").write_text(json.dumps({
        "alpha": best_alpha,
        "train_mae_aggregate_ppm": best_mae,
        "scaler": "standard",
    }))
    print(f"selected alpha={best_alpha}, predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
