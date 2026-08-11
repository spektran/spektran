#!/usr/bin/env python
"""Export U-Net T2 and TCN T5 checkpoints for HF Model Hub.

Run AFTER training completes (predictions + hyperparams must exist):

    python scripts/export_dl_checkpoints.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "baselines"))

from common import DATA  # noqa: E402
from spektran.io import read_records  # noqa: E402

CKPT = REPO / "checkpoints"


def export_unet_t2() -> bool:
    from unet_t2.train import UNet1D, load_t2_split

    hp_path = REPO / "baselines" / "unet_t2" / "hyperparams.json"
    if not hp_path.exists():
        print("U-Net T2: hyperparams.json not found, skipping")
        return False

    hp = json.loads(hp_path.read_text())
    best_rmse = hp["best_val_spectral_rmse"]

    X_tr, _, _ = load_t2_split("ch4-t1-train-v0")
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9

    model = UNet1D()
    torch.manual_seed(hp["seed"])
    torch.use_deterministic_algorithms(True)

    X_va, Y_va, _ = load_t2_split("ch4-t1-val-v0")
    Xv = torch.tensor((X_va - mu) / sd, dtype=torch.float32).unsqueeze(1)

    train_log = hp.get("train_log", [])
    best_epoch = min(train_log, key=lambda e: e["val_spectral_rmse"])["epoch"] if train_log else -1

    out = CKPT / "unet-t2-denoising"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "normalization.npz", input_mean=mu, input_std=sd)

    print(f"  Re-training U-Net to epoch {best_epoch} for checkpoint export...")
    from unet_t2.train import BATCH, EPOCHS, LR, SEED

    Xt_full, Yt_full, _ = load_t2_split("ch4-t1-train-v0")
    Xt = torch.tensor((Xt_full - mu) / sd, dtype=torch.float32).unsqueeze(1)
    Yt = torch.tensor(Yt_full, dtype=torch.float32)

    model = UNet1D()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    best_state, best_val = None, np.inf
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        model.train(False)
        with torch.no_grad():
            val_pred = []
            for i in range(0, len(Xv), BATCH):
                val_pred.append(model(Xv[i : i + BATCH]).numpy())
            val_pred = np.concatenate(val_pred)
        val_rmse = float(np.sqrt(np.mean((val_pred - Y_va) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save(best_state, out / "model.pt")

    meta = {
        "task": "T2-denoising",
        "model": "UNet1D",
        "architecture": "3-stage encoder/decoder, skip connections",
        "input": "raw_scan (2000 pts), per-feature z-score normalized",
        "output": "absorbance_clean (2000 pts)",
        "val_spectral_rmse": best_val,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "lr": LR,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  U-Net T2 exported to {out.relative_to(REPO)} (val RMSE {best_val:.6f})")
    return True


def export_tcn_t5() -> bool:
    from tcn_t5.train import TCN, WINDOW_SIZE

    hp_path = REPO / "baselines" / "tcn_t5" / "hyperparams.json"
    if not hp_path.exists():
        print("TCN T5: hyperparams.json not found, skipping")
        return False

    hp = json.loads(hp_path.read_text())

    from common import load_time_series_split
    from tcn_t5.train import BATCH, EPOCHS, LR, SEED, make_windows

    X_tr, y_tr, ids_tr = load_time_series_split("ch4-t5-train-v0")
    Xw_tr, yw_tr, _ = make_windows(X_tr, y_tr, ids_tr, WINDOW_SIZE)

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9
    ylog = np.log1p(yw_tr)
    y_mu, y_sd = float(ylog.mean()), float(ylog.std())

    out = CKPT / "tcn-t5-drift"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "normalization.npz",
             input_mean=mu, input_std=sd,
             target_mean=np.array([y_mu]), target_std=np.array([y_sd]))

    print("  Re-training TCN for checkpoint export...")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    Xt = torch.tensor((Xw_tr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((ylog - y_mu) / y_sd, dtype=torch.float32)

    model = TCN(window_size=WINDOW_SIZE, n_points=Xt.shape[-1])
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i : i + BATCH]
            optim.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optim.step()

    model.train(False)
    torch.save(model.state_dict(), out / "model.pt")

    with torch.no_grad():
        preds = np.expm1(model(Xt).numpy() * y_sd + y_mu)
    final_train_mae = float(np.mean(np.abs(preds - yw_tr)))

    meta = {
        "task": "T5-drift",
        "model": "TCN",
        "architecture": "dilated conv (dilation 1,2,2), AdaptiveAvgPool → FC",
        "input": f"window of {WINDOW_SIZE} consecutive raw_scans, per-feature z-score",
        "output": "concentration_ppm (log1p target transform)",
        "window_size": WINDOW_SIZE,
        "test_mae_ppm": hp.get("test_mae_ppm"),
        "final_train_mae_ppm": final_train_mae,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "lr": LR,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  TCN T5 exported to {out.relative_to(REPO)} (test MAE {hp.get('test_mae_ppm', '?')} ppm)")
    return True


def main() -> int:
    print("=" * 60)
    print("DL Checkpoint Export")
    print("=" * 60)

    ok = 0
    if export_unet_t2():
        ok += 1
    if export_tcn_t5():
        ok += 1

    if ok == 0:
        print("\nNo checkpoints exported (training may not be complete)")
        return 1

    summary_path = CKPT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}

    for d in CKPT.iterdir():
        if d.is_dir() and (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
            summary[d.name] = {
                "task": meta["task"],
                "model": meta["model"],
            }
            for key in ["val_spectral_rmse", "test_mae_ppm"]:
                if key in meta and meta[key] is not None:
                    summary[d.name][key] = meta[key]

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{ok} checkpoint(s) exported. Summary updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
