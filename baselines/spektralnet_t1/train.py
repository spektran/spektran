#!/usr/bin/env python
"""SpektralNet — Dual-Domain Physics-Augmented Ridge for T1/T3.

The empirical finding across all SPEKTRAN baselines: Ridge dominates
because Beer-Lambert absorbance is linear in concentration. Deep models
overfit instrument artifacts and generalize poorly (T3 degradation).

SpektralNet enhances Ridge with physics-informed feature augmentation:
the same raw scan viewed in complementary domains (raw intensity,
estimated absorbance, physics scalars) provides a richer feature space
that maintains linearity while adding instrument-invariant information.

    python baselines/spektralnet_t1/train.py
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
SEED = 20260811


def extract_physics_features(raw_scan: np.ndarray) -> np.ndarray:
    """Extract Beer-Lambert physics features from raw spectral scans.

    These scalars capture the physically meaningful properties of the
    absorption signal and are partially decoupled from instrument-specific
    baseline and fringe patterns.
    """
    n = raw_scan.shape[-1]
    wing = int(n * 0.15)

    wing_left = raw_scan[:, :wing].mean(axis=1)
    wing_right = raw_scan[:, -wing:].mean(axis=1)
    baseline = (wing_left + wing_right) / 2.0

    min_val = raw_scan.min(axis=1)
    transmittance = np.clip(min_val / np.clip(baseline, 1e-9, None), 1e-9, 1.0)
    peak_abs = -np.log(transmittance)

    center = raw_scan[:, n // 4: 3 * n // 4]
    center_depth = baseline - center.min(axis=1)
    center_depth_norm = center_depth / np.clip(baseline, 1e-9, None)

    center_mean = center.mean(axis=1)
    center_contrast = (baseline - center_mean) / np.clip(baseline, 1e-9, None)

    wing_asymmetry = (wing_left - wing_right) / np.clip(baseline, 1e-9, None)

    absorption_proxy = baseline.reshape(-1, 1) - raw_scan
    abs_positive = np.clip(absorption_proxy, 0, None)
    integrated = abs_positive.sum(axis=1) / n / np.clip(baseline, 1e-9, None)

    weights = abs_positive / (abs_positive.sum(axis=1, keepdims=True) + 1e-9)
    positions = np.arange(n).reshape(1, -1).astype(np.float64)
    mean_pos = (weights * positions).sum(axis=1)
    variance = (weights * (positions - mean_pos.reshape(-1, 1)) ** 2).sum(axis=1)
    width = np.sqrt(np.clip(variance, 1e-9, None)) / n

    return np.column_stack([
        peak_abs, center_depth_norm, center_contrast,
        wing_asymmetry, integrated, width,
    ])


def main() -> int:
    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    X_va, y_va, _ = load_split("ch4-t1-val-v0")
    X_te, _, ids_te = load_split("ch4-t1-test-v0")
    X_ho, _, ids_ho = load_split("ch4-t3-test-heldout-v0")

    phys_tr = extract_physics_features(X_tr)
    phys_va = extract_physics_features(X_va)
    phys_te = extract_physics_features(X_te)
    phys_ho = extract_physics_features(X_ho)

    X_aug_tr = np.hstack([X_tr, phys_tr])
    X_aug_va = np.hstack([X_va, phys_va])
    X_aug_te = np.hstack([X_te, phys_te])
    X_aug_ho = np.hstack([X_ho, phys_ho])

    configs = {
        "raw_ridge": (X_tr, X_va, X_te, X_ho),
        "augmented_ridge": (X_aug_tr, X_aug_va, X_aug_te, X_aug_ho),
        "physics_only": (phys_tr, phys_va, phys_te, phys_ho),
    }

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    results = {}

    for name, (d_tr, d_va, d_te, d_ho) in configs.items():
        scaler = StandardScaler()
        d_tr_s = scaler.fit_transform(d_tr)
        d_va_s = scaler.transform(d_va)
        d_te_s = scaler.transform(d_te)
        d_ho_s = scaler.transform(d_ho)

        best_alpha, best_mae, best_model = None, np.inf, None
        for alpha in alphas:
            ridge = Ridge(alpha=alpha, random_state=SEED)
            ridge.fit(d_tr_s, y_tr)
            pred_va = ridge.predict(d_va_s)
            mae = float(np.mean(np.abs(pred_va - y_va)))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
                best_model = ridge

        pred_te = best_model.predict(d_te_s)
        pred_ho = best_model.predict(d_ho_s)

        results[name] = {
            "alpha": best_alpha,
            "val_mae": best_mae,
            "model": best_model,
            "scaler": scaler,
            "pred_te": pred_te,
            "pred_ho": pred_ho,
        }
        print(f"  {name}: alpha={best_alpha}, val MAE={best_mae:.3f} ppm")

    best_name = min(results, key=lambda k: results[k]["val_mae"])
    best = results[best_name]
    print(f"\nBest single model: {best_name} (val MAE={best['val_mae']:.3f} ppm)")

    print("\nTrying optimal blend of raw + augmented...")
    raw_va_pred = results["raw_ridge"]["model"].predict(
        results["raw_ridge"]["scaler"].transform(X_va))
    aug_va_pred = results["augmented_ridge"]["model"].predict(
        results["augmented_ridge"]["scaler"].transform(X_aug_va))

    best_w, best_blend_mae = 0.0, np.inf
    for w in np.arange(0, 1.01, 0.05):
        blend = w * aug_va_pred + (1 - w) * raw_va_pred
        mae = float(np.mean(np.abs(blend - y_va)))
        if mae < best_blend_mae:
            best_blend_mae = mae
            best_w = round(w, 2)

    print(f"  Best blend: w_augmented={best_w}, val MAE={best_blend_mae:.3f} ppm")

    if best_blend_mae < best["val_mae"]:
        print(f"  Blend wins! ({best_blend_mae:.3f} < {best['val_mae']:.3f})")
        final_val_mae = best_blend_mae
        for X_raw, X_aug, ids, tag in [
            (X_te, X_aug_te, ids_te, "t1-test"),
            (X_ho, X_aug_ho, ids_ho, "t3-test-heldout"),
        ]:
            raw_pred = results["raw_ridge"]["model"].predict(
                results["raw_ridge"]["scaler"].transform(X_raw))
            aug_pred = results["augmented_ridge"]["model"].predict(
                results["augmented_ridge"]["scaler"].transform(X_aug))
            final_pred = best_w * aug_pred + (1 - best_w) * raw_pred
            write_predictions_csv(OUT / f"predictions_{tag}.csv", ids, final_pred)
        strategy = f"blend(w_aug={best_w})"
    else:
        print(f"  Single model wins ({best['val_mae']:.3f})")
        final_val_mae = best["val_mae"]
        write_predictions_csv(OUT / "predictions_t1-test.csv",
                              ids_te, best["pred_te"])
        write_predictions_csv(OUT / "predictions_t3-test-heldout.csv",
                              ids_ho, best["pred_ho"])
        strategy = f"single({best_name})"

    (OUT / "hyperparams.json").write_text(json.dumps({
        "seed": SEED,
        "architecture": "SpektralNet: Dual-Domain Physics-Augmented Ridge",
        "strategy": strategy,
        "blend_weight_augmented": best_w if best_blend_mae < best["val_mae"] else None,
        "individual_results": {
            name: {"alpha": r["alpha"], "val_mae_ppm": r["val_mae"]}
            for name, r in results.items()
        },
        "best_val_mae_ppm": final_val_mae,
        "physics_features": [
            "peak_absorbance", "center_depth_normalized",
            "center_contrast", "wing_asymmetry",
            "integrated_absorption", "spectral_width",
        ],
        "target_transform": "none (raw ppm, matching standard Ridge baseline)",
        "design_rationale": "Ridge dominates because Beer-Lambert is linear. SpektralNet "
                            "augments Ridge's feature space with 6 physics-informed scalars "
                            "extracted from the raw scan (peak absorbance, center depth, "
                            "contrast, wing asymmetry, integrated absorption, spectral "
                            "width). An optional blend with pure Ridge captures complementary "
                            "information from both the full spectral and physics-scalar views.",
    }, indent=2))
    print(f"\nFinal: val MAE {final_val_mae:.3f} ppm; "
          f"predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
