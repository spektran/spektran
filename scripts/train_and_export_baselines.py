#!/usr/bin/env python
"""Train core baselines and export weights for HF Model Hub.

Trains ridge + CNN on T1 (DA concentration), saves checkpoints with
normalization statistics so users can do inference without retraining.

    python scripts/train_and_export_baselines.py --out checkpoints/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))

from common import load_split, load_wms_split, load_temperature_split  # noqa: E402


def train_ridge_t1(out: Path) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    print("\n=== Ridge T1 (DA concentration) ===")
    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    X_va, y_va, _ = load_split("ch4-t1-val-v0")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)

    best_alpha, best_mae = None, np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
        mae = float(np.mean(np.abs(model.predict(X_va_s) - y_va)))
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    model = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)

    d = out / "ridge-t1-da"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / "weights.npz",
        coef=model.coef_,
        intercept=np.array([model.intercept_]),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    meta = {
        "model": "Ridge",
        "task": "T1-concentration",
        "technique": "TDLAS-DA",
        "alpha": best_alpha,
        "val_mae_ppm": round(best_mae, 3),
        "input": "raw_scan (2000 points)",
        "output": "concentration_ppm",
        "normalization": "StandardScaler (per-feature)",
    }
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"  alpha={best_alpha}, val MAE={best_mae:.3f} ppm -> {d}")
    return meta


def train_cnn_t1(out: Path) -> dict:
    import torch
    import torch.nn as nn

    print("\n=== CNN T1 (DA concentration) ===")
    SEED, EPOCHS, BATCH, LR = 20260812, 60, 64, 3e-4

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    X_va, y_va, _ = load_split("ch4-t1-val-v0")

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9

    def prep(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32).unsqueeze(1)

    Xt, Xv = prep(X_tr), prep(X_va)
    ylog = np.log1p(y_tr)
    y_mu, y_sd = float(ylog.mean()), float(ylog.std())
    yt = torch.tensor((ylog - y_mu) / y_sd, dtype=torch.float32)

    model = nn.Sequential(
        nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.ReLU(),
        nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU(),
        nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool1d(8), nn.Flatten(),
        nn.Linear(64 * 8, 64), nn.ReLU(), nn.Linear(64, 1),
    )

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_state, best_val = None, np.inf
    g = torch.Generator().manual_seed(SEED)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
        model.train(False)
        with torch.no_grad():
            val_preds = np.expm1(model(Xv).squeeze(-1).numpy() * y_sd + y_mu)
            val_mae_ppm = float(np.mean(np.abs(val_preds - y_va)))
        if val_mae_ppm < best_val:
            best_val = val_mae_ppm
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:02d}: val MAE {val_mae_ppm:.3f} ppm")

    d = out / "cnn1d-t1-da"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, d / "model.pt")
    np.savez(
        d / "normalization.npz",
        input_mean=mu,
        input_std=sd,
        target_mean=np.array([y_mu]),
        target_std=np.array([y_sd]),
    )
    meta = {
        "model": "CNN1D",
        "task": "T1-concentration",
        "technique": "TDLAS-DA",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "lr": LR,
        "val_mae_ppm": round(best_val, 3),
        "input": "raw_scan (2000 points)",
        "output": "concentration_ppm",
        "target_transform": "log1p -> z-score",
        "architecture": "Conv1d(1->16->32->64) + AdaptiveAvgPool + Linear(512->64->1)",
    }
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"  best val MAE={best_val:.3f} ppm -> {d}")
    return meta


def train_ridge_t4_wms(out: Path) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    print("\n=== Ridge T4 (WMS 2f concentration) ===")
    X_tr, y_tr, _ = load_wms_split("ch4-t4-train-v0")
    X_va, y_va, _ = load_wms_split("ch4-t4-val-v0")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)

    best_alpha, best_mae = None, np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
        mae = float(np.mean(np.abs(model.predict(X_va_s) - y_va)))
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    model = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)

    d = out / "ridge-t4-wms"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / "weights.npz",
        coef=model.coef_,
        intercept=np.array([model.intercept_]),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    meta = {
        "model": "Ridge",
        "task": "T4-wms-concentration",
        "technique": "TDLAS-WMS",
        "alpha": best_alpha,
        "val_mae_ppm": round(best_mae, 3),
        "input": "demod_2f signal",
        "output": "concentration_ppm",
    }
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"  alpha={best_alpha}, val MAE={best_mae:.3f} ppm -> {d}")
    return meta


def train_ridge_t9_temp(out: Path) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    print("\n=== Ridge T9 (temperature regression) ===")
    X_tr, y_tr, _ = load_temperature_split("ch4-t9-train-v0")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)

    best_alpha, best_mae = None, np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(X_tr_s, y_tr)
        preds = model.predict(X_tr_s)
        mae = float(np.mean(np.abs(preds - y_tr)))
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    model = Ridge(alpha=best_alpha).fit(X_tr_s, y_tr)

    d = out / "ridge-t9-temperature"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / "weights.npz",
        coef=model.coef_,
        intercept=np.array([model.intercept_]),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    meta = {
        "model": "Ridge",
        "task": "T9-temperature",
        "technique": "TDLAS-DA",
        "alpha": best_alpha,
        "train_mae_K": round(best_mae, 2),
        "input": "raw_scan (2000 points)",
        "output": "temperature_K",
    }
    (d / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"  alpha={best_alpha}, train MAE={best_mae:.2f} K -> {d}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints", help="Output directory")
    args = ap.parse_args()
    out = Path(args.out)

    results = {}
    results["ridge_t1"] = train_ridge_t1(out)
    results["ridge_t4"] = train_ridge_t4_wms(out)
    results["ridge_t9"] = train_ridge_t9_temp(out)
    results["cnn_t1"] = train_cnn_t1(out)

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll baselines trained. Checkpoints in {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
