#!/usr/bin/env python
"""LSTM Denoising AutoEncoder (LSTM-DAE) baseline for T2.

A different denoising paradigm from U-Net: the encoder compresses the
noisy scan into a low-dimensional bottleneck via LSTM, forcing the model
to learn a compact representation that discards noise; the decoder LSTM
reconstructs the clean absorbance from this bottleneck. No skip connections
— all information must pass through the bottleneck, which acts as an
information filter. Reproduce:

    python baselines/lstm_dae_t2/train.py

Reference:
    Z. Li et al., "A TDLAS signal denoising method based on LSTM-DAE",
    Optics Communications 508 (2024) 130272,
    doi:10.1016/j.optcom.2024.130272
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
SEED = 20260811
EPOCHS = 100
BATCH = 32
LR = 1e-3


class LSTMDAE(nn.Module):
    """LSTM-based Denoising AutoEncoder.

    Encoder: Conv1d downsamples 4x, then LSTM compresses to a fixed-length
    latent vector (last hidden state). Decoder: repeat latent across time,
    LSTM decodes, Conv1dTranspose upsamples back to original resolution.
    """

    def __init__(self, n_points: int = 2000, latent: int = 64, lstm_h: int = 128):
        super().__init__()
        self.n_points = n_points
        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.seq_len = (n_points + 3) // 4
        self.enc_lstm = nn.LSTM(64, lstm_h, num_layers=2, batch_first=True)
        self.compress = nn.Linear(lstm_h, latent)
        self.expand = nn.Linear(latent, lstm_h)
        self.dec_lstm = nn.LSTM(lstm_h, 64, num_layers=2, batch_first=True)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        # x: [B, 1, n_points]
        enc = self.enc_conv(x)  # [B, 64, seq_len]
        enc_seq = enc.permute(0, 2, 1)  # [B, seq_len, 64]
        _, (h_n, _) = self.enc_lstm(enc_seq)  # h_n: [2, B, lstm_h]
        latent = self.compress(h_n[-1])  # [B, latent]

        dec_input = self.expand(latent).unsqueeze(1).expand(-1, self.seq_len, -1)
        dec_seq, _ = self.dec_lstm(dec_input)  # [B, seq_len, 64]
        dec_feat = dec_seq.permute(0, 2, 1)  # [B, 64, seq_len]
        out = self.dec_conv(dec_feat)  # [B, 1, ~n_points]
        return out[:, 0, :self.n_points]  # [B, n_points]


def load_t2_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["raw_scan"] for r in records])
    Y = np.stack([r["arrays"]["absorbance_clean"] for r in records])
    ids = [r["meta"]["record_id"] for r in records]
    return X, Y, ids


def predict(model: nn.Module, X: torch.Tensor, batch: int = BATCH) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(X[i : i + batch]).numpy())
    return np.concatenate(outs, axis=0)


def main() -> int:
    torch.manual_seed(SEED)

    X_tr, Y_tr, _ = load_t2_split("ch4-t1-train-v0")
    X_va, Y_va, _ = load_t2_split("ch4-t1-val-v0")
    X_te, _, ids_te = load_t2_split("ch4-t1-test-v0")

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9

    def prep(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32).unsqueeze(1)

    Xt, Xv, Xte = prep(X_tr), prep(X_va), prep(X_te)
    Yt = torch.tensor(Y_tr, dtype=torch.float32)

    model = LSTMDAE(n_points=Xt.shape[-1])
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
        val_rmse = float(np.sqrt(np.mean((val_pred - Y_va) ** 2)))
        log.append({"epoch": epoch, "val_spectral_rmse": val_rmse})
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"epoch {epoch:02d}: val spectral RMSE {val_rmse:.6f}")

    model.load_state_dict(best_state)
    model.train(False)
    preds = predict(model, Xte)

    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT / "predictions_t2-test.h5", "w") as f:
        grp = f.create_group("predictions")
        for rid, pred in zip(ids_te, preds):
            grp.create_dataset(rid, data=pred)

    (OUT / "hyperparams.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "architecture": "Conv1d(4x down)->LSTM enc->latent(64)->LSTM dec->ConvT1d(4x up)",
        "best_val_spectral_rmse": best_val,
        "paradigm": "bottleneck compression (no skip connections)",
        "train_log": log,
    }, indent=2))
    print(f"best val spectral RMSE {best_val:.6f}; predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
