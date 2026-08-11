#!/usr/bin/env python
"""Compare Ridge baseline results on demo vs HITRAN production line data.

Quantifies the gap between approximate demo lines (3 lines) and full
HITRAN production data (76+ lines) for T1 concentration regression.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))

from common import load_split  # noqa: E402
from spektran.io import read_records  # noqa: E402

DATA = REPO / "data"


def load_split_from(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["raw_scan"] for r in records])
    y = np.array([r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records])
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def ridge_evaluate(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
) -> dict:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)
    model = Ridge(alpha=1.0)
    model.fit(Xs, y_te if len(y_tr) == 0 else y_tr)
    preds = model.predict(scaler.transform(X_te))
    mae = float(np.mean(np.abs(preds - y_te)))
    mape = float(np.mean(np.abs((preds - y_te) / (y_te + 1e-9)))) * 100
    return {"mae_ppm": mae, "mape_pct": mape}


def main() -> int:
    results = {}

    for label, train_name, test_name in [
        ("demo", "ch4-t1-train-v0", "ch4-t1-test-v0"),
        ("hitran", "ch4-t1-train-v0-hitran", "ch4-t1-test-v0-hitran"),
    ]:
        train_path = DATA / f"{train_name}.h5"
        test_path = DATA / f"{test_name}.h5"
        if not train_path.exists() or not test_path.exists():
            print(f"  Skipping {label}: data not found")
            continue

        X_tr, y_tr, _ = load_split_from(train_name)
        X_te, y_te, _ = load_split_from(test_name)
        scores = ridge_evaluate(X_tr, y_tr, X_te, y_te)
        results[label] = scores
        print(f"  Ridge T1 ({label}): MAE {scores['mae_ppm']:.2f} ppm, MAPE {scores['mape_pct']:.1f}%")

    if "demo" in results and "hitran" in results:
        gap = results["hitran"]["mae_ppm"] / results["demo"]["mae_ppm"]
        print(f"\n  HITRAN/demo MAE ratio: {gap:.2f}x")
        results["gap_ratio"] = gap

    out = REPO / "baselines" / "hitran_vs_demo_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
