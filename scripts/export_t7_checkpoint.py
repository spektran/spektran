#!/usr/bin/env python
"""Export Ridge T7 cross-modality checkpoint to checkpoints/ directory.

Uses joblib for scikit-learn model serialization (standard practice for
sklearn models in production, consistent with existing checkpoint exports).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))
sys.path.insert(0, str(REPO / "baselines" / "ridge_cross_modality_t7"))

from common import load_split  # noqa: E402
from train import tdlas_integrated_absorbance  # noqa: E402


def main() -> int:
    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    A_tr = tdlas_integrated_absorbance(X_tr)

    model = Ridge(alpha=1.0)
    model.fit(A_tr, np.log1p(y_tr))

    out_dir = REPO / "checkpoints" / "ridge-t7-cross-modality"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")

    meta = {
        "model": "Ridge",
        "task": "T7",
        "feature": "integrated absorbance (TDLAS wing-baseline, NDIR Planck-normalized ratio)",
        "target_transform": "log1p",
        "training_data": "ch4-t1-train-v0 (TDLAS, 5000 records)",
        "test_data": "ch4-cross-modality-test-v0 (NDIR, 1000 records)",
        "mae_ppm": 130.68,
        "degradation_vs_t1": 46.02,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported to {out_dir.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
