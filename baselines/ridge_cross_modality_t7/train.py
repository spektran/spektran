#!/usr/bin/env python
"""Ridge baseline for T7 (cross-modality transfer: TDLAS → NDIR).

The NDIR ratio = S_active / S_ref is dominated by filter parameter
variation (each virtual instrument has different filter FWHMs and centers).
To extract the gas absorption signal, we compute the zero-gas baseline
ratio from the Planck function and filter parameters, then normalize:

    transmittance = ratio_observed / ratio_zero_gas

This gives the actual gas transmittance, from which we extract
A_integrated = -ln(transmittance).

For TDLAS, we extract integrated absorbance from the wing-corrected
raw scan. Both features live in the same physical space (Beer-Lambert
absorbance), enabling genuine cross-modality transfer.

Reproduce:

    python baselines/ridge_cross_modality_t7/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, REPO, load_split, write_predictions_csv  # noqa: E402

from spektran.io import read_records  # noqa: E402
from spektran.physics.ndir import bandpass_filter, planck_spectral_radiance  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20260819


def compute_zero_gas_ratio(inst: dict) -> float:
    """Compute the NDIR ratio with no gas present for this instrument config."""
    filt = inst["filters"]
    src_T = inst["source"]["temperature_K"]
    active_center = filt["active_center_cm1"]
    active_fwhm = filt["active_fwhm_cm1"]
    ref_center = filt["reference_center_cm1"]
    ref_fwhm = filt["reference_fwhm_cm1"]
    shape = filt.get("shape", "gaussian")

    lo = min(active_center - 3 * active_fwhm, ref_center - 3 * ref_fwhm)
    hi = max(active_center + 3 * active_fwhm, ref_center + 3 * ref_fwhm)
    nu = np.linspace(lo, hi, 500)

    B = planck_spectral_radiance(nu, src_T)
    F_a = bandpass_filter(nu, active_center, active_fwhm, shape)
    F_r = bandpass_filter(nu, ref_center, ref_fwhm, shape)

    S_a = float(np.trapezoid(B * F_a, nu))
    S_r = float(np.trapezoid(B * F_r, nu))
    return S_a / (S_r + 1e-30)


def load_ndir_normalized(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load NDIR data with physics-normalized transmittance."""
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])

    features = []
    for r in records:
        ratio = float(r["arrays"]["ratio"])
        R0 = compute_zero_gas_ratio(r["meta"]["instrument"])
        transmittance = np.clip(ratio / R0, 1e-6, 2.0)
        A_int = -np.log(transmittance)
        features.append(A_int)

    X = np.array(features).reshape(-1, 1)
    y = np.array(
        [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    )
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def tdlas_integrated_absorbance(X: np.ndarray) -> np.ndarray:
    """Extract integrated absorbance from TDLAS raw scans."""
    n = X.shape[1]
    x = np.linspace(0, 1, n)
    wing_mask = (x < 0.15) | (x > 0.85)
    center = slice(int(n * 0.30), int(n * 0.70))

    features = np.empty(len(X))
    for i in range(len(X)):
        poly = np.polyfit(x[wing_mask], X[i, wing_mask], 3)
        baseline = np.polyval(poly, x)
        T = np.clip(X[i] / (baseline + 1e-12), 1e-6, None)
        features[i] = -np.log(T[center]).mean()
    return features.reshape(-1, 1)


def main() -> int:
    np.random.seed(SEED)

    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    A_tr = tdlas_integrated_absorbance(X_tr)

    model = Ridge(alpha=1.0)
    model.fit(A_tr, np.log1p(y_tr))

    train_preds = np.expm1(model.predict(A_tr))
    train_mae = float(np.mean(np.abs(train_preds - y_tr)))
    print(f"TDLAS train MAE: {train_mae:.2f} ppm (sanity check)")

    X_ndir_te, y_ndir_te, ids_te = load_ndir_normalized("ch4-cross-modality-test-v0")
    preds = np.expm1(model.predict(X_ndir_te))
    preds = np.clip(preds, 0, None)

    mae = float(np.mean(np.abs(preds - y_ndir_te)))
    mape = float(np.mean(np.abs((preds - y_ndir_te) / (y_ndir_te + 1e-9)))) * 100

    t1_mae = 2.84
    degradation = mae / t1_mae

    OUT.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(OUT / "predictions_t7-test.csv", ids_te, preds)
    scores = {
        "task": "T7",
        "model": "Ridge (TDLAS→NDIR, Planck-normalized absorbance)",
        "mae_ppm": mae,
        "mape_pct": mape,
        "t1_ridge_mae": t1_mae,
        "degradation_vs_t1": degradation,
    }
    (OUT / "scores.json").write_text(json.dumps(scores, indent=2))
    (OUT / "hyperparams.json").write_text(json.dumps({
        "seed": SEED,
        "alpha": 1.0,
        "feature": "integrated absorbance (TDLAS: wing-baseline; NDIR: Planck-normalized ratio)",
        "target_transform": "log1p",
        "training_data": "ch4-t1-train-v0 (TDLAS, 5000 records)",
        "test_data": "ch4-cross-modality-test-v0 (NDIR, 1000 records)",
    }, indent=2))

    print(f"T7 Ridge: MAE {mae:.2f} ppm, MAPE {mape:.1f}%, degradation {degradation:.2f}x vs T1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
