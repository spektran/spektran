#!/usr/bin/env python
"""U-Net baseline for T2 (spectral denoising).

1D U-Net (3 encoder/decoder stages, skip connections) that maps the noisy
raw_scan directly onto the clean absorbance spectrum, point for point --
the first deep learning baseline for the T2 task (the existing reference,
wing_poly_t2, is a classical wing-anchored polynomial with no training).
MSE loss on the full 2000-point spectrum; the input is standardized with
train-set per-feature statistics, the target (absorbance_clean) is used
as-is since it is already in physical units. Reproduce:

    python baselines/unet_t2/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, REPO  # noqa: E402

from spektran.io import read_records  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20260817
EPOCHS = 100
BATCH = 32
LR = 1e-3


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    """Minimal 1D U-Net for spectral denoising.

    Encoder: 3 downsample stages (1->32->64->128). Bottleneck: 128->256.
    Decoder: 3 upsample stages with skip connections. Head: 1x1 conv to a
    single output channel. Input length must be divisible by 8 (2000 is).
    """

    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128 skip + 128 up = 256 in
        self.up2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.head = nn.Conv1d(32, 1, 1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        # x: [B, 1, 2000]
        e1 = self.enc1(x)  # [B, 32, 2000]
        e2 = self.enc2(self.pool(e1))  # [B, 64, 1000]
        e3 = self.enc3(self.pool(e2))  # [B, 128, 500]
        b = self.bottleneck(self.pool(e3))  # [B, 256, 250]
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))  # [B, 128, 500]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # [B, 64, 1000]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 32, 2000]
        return self.head(d1).squeeze(1)  # [B, 2000]


def load_t2_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a T2 split: X = raw_scan (noisy input), Y = absorbance_clean (target)."""
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["raw_scan"] for r in records])
    Y = np.stack([r["arrays"]["absorbance_clean"] for r in records])
    ids = [r["meta"]["record_id"] for r in records]
    return X, Y, ids


def predict(model: nn.Module, X: torch.Tensor, batch: int = BATCH) -> np.ndarray:
    """Chunked forward pass (no_grad) -- U-Nets are memory-hungry at full-batch."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(X[i : i + batch]).numpy())
    return np.concatenate(outs, axis=0)


def main() -> int:
    torch.manual_seed(SEED)  # no numpy randomness used in this script
    torch.use_deterministic_algorithms(True)

    X_tr, Y_tr, _ = load_t2_split("ch4-t1-train-v0")
    X_va, Y_va, _ = load_t2_split("ch4-t1-val-v0")
    X_te, _, ids_te = load_t2_split("ch4-t1-test-v0")

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9

    def prep(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32).unsqueeze(1)

    Xt, Xv, Xte = prep(X_tr), prep(X_va), prep(X_te)
    Yt = torch.tensor(Y_tr, dtype=torch.float32)

    model = UNet1D()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_state, best_val, log = None, np.inf, []
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
        val_pred = predict(model, Xv)
        val_spectral_rmse = float(np.sqrt(np.mean((val_pred - Y_va) ** 2)))
        log.append({"epoch": epoch, "val_spectral_rmse": val_spectral_rmse})
        if val_spectral_rmse < best_val:
            best_val = val_spectral_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}: val spectral RMSE {val_spectral_rmse:.6f}")

    model.load_state_dict(best_state)
    model.train(False)
    preds = predict(model, Xte)

    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT / "predictions_t2-test.h5", "w") as f:
        grp = f.create_group("predictions")
        for rid, pred in zip(ids_te, preds):
            grp.create_dataset(rid, data=pred)

    torch.save(best_state, OUT / "model.pt")
    np.savez(OUT / "normalization.npz", input_mean=mu, input_std=sd)

    (OUT / "hyperparams.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "optimizer": "Adam",
        "best_val_spectral_rmse": best_val,
        "input_normalization": "per-feature z-score (train statistics)",
        "target_normalization": "none (absorbance_clean used as-is)",
        "train_log": log,
    }, indent=2))
    print(f"best val spectral RMSE {best_val:.6f}; predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
