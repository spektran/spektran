#!/usr/bin/env python
"""Run Ridge baselines on HITRAN production data vs demo line data.

Compares results across T1, T3, T4 to quantify the impact of using
76 HITRAN lines vs 3 approximate demo lines.
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

from spektran.io import read_records  # noqa: E402

DATA = REPO / "data"


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["raw_scan"] for r in records])
    y = np.array([r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records])
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def load_wms_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["demod_2f"] for r in records])
    y = np.array([r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records])
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def ridge_run(X_tr, y_tr, X_te, y_te) -> dict:
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    m = Ridge(alpha=1.0)
    m.fit(Xs, y_tr)
    preds = m.predict(sc.transform(X_te))
    mae = float(np.mean(np.abs(preds - y_te)))
    mape = float(np.mean(np.abs((preds - y_te) / (y_te + 1e-9)))) * 100
    return {"mae_ppm": mae, "mape_pct": mape}


def main() -> int:
    results = {}

    print("=" * 60)
    print("HITRAN vs Demo Baseline Comparison")
    print("=" * 60)

    for label, suffix in [("demo", ""), ("hitran", "-hitran")]:
        tr = f"ch4-t1-train-v0{suffix}"
        te = f"ch4-t1-test-v0{suffix}"
        if not (DATA / f"{tr}.h5").exists():
            continue
        X_tr, y_tr, _ = load_split(tr)
        X_te, y_te, _ = load_split(te)
        s = ridge_run(X_tr, y_tr, X_te, y_te)
        results[f"ridge_t1_{label}"] = s
        print(f"  Ridge T1 ({label}): MAE {s['mae_ppm']:.2f} ppm, MAPE {s['mape_pct']:.1f}%")

    for label, suffix in [("demo", ""), ("hitran", "-hitran")]:
        tr = f"ch4-t1-train-v0{suffix}"
        te = f"ch4-t3-test-heldout-v0{suffix}"
        if not (DATA / f"{tr}.h5").exists() or not (DATA / f"{te}.h5").exists():
            continue
        X_tr, y_tr, _ = load_split(tr)
        X_te, y_te, _ = load_split(te)
        s = ridge_run(X_tr, y_tr, X_te, y_te)
        results[f"ridge_t3_{label}"] = s
        print(f"  Ridge T3 ({label}): MAE {s['mae_ppm']:.2f} ppm, MAPE {s['mape_pct']:.1f}%")

    for label, suffix in [("demo", ""), ("hitran", "-hitran")]:
        tr = f"ch4-t4-train-v0{suffix}"
        te = f"ch4-t4-test-v0{suffix}"
        if not (DATA / f"{tr}.h5").exists() or not (DATA / f"{te}.h5").exists():
            continue
        X_tr, y_tr, _ = load_wms_split(tr)
        X_te, y_te, _ = load_wms_split(te)
        s = ridge_run(X_tr, y_tr, X_te, y_te)
        results[f"ridge_t4_{label}"] = s
        print(f"  Ridge T4 ({label}): MAE {s['mae_ppm']:.2f} ppm, MAPE {s['mape_pct']:.1f}%")

    print("\n" + "=" * 60)
    print("SUMMARY: HITRAN / Demo MAE ratios")
    print("=" * 60)
    for task in ["t1", "t3", "t4"]:
        demo_key = f"ridge_{task}_demo"
        hitran_key = f"ridge_{task}_hitran"
        if demo_key in results and hitran_key in results:
            ratio = results[hitran_key]["mae_ppm"] / results[demo_key]["mae_ppm"]
            print(f"  {task.upper()}: {ratio:.3f}x (HITRAN {results[hitran_key]['mae_ppm']:.2f} vs demo {results[demo_key]['mae_ppm']:.2f})")

    out = REPO / "baselines" / "hitran_vs_demo_full_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
