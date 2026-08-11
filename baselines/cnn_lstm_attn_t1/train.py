#!/usr/bin/env python
"""CNN-LSTM-Attention hybrid baseline for T1 (and T3 held-out).

The dominant architecture in 2024-2025 TDLAS+ML papers: 1D CNN extracts
local spectral features, BiLSTM captures sequential dependencies along
the wavenumber axis, and a scaled dot-product attention layer learns to
focus on informative spectral regions (absorption peaks). Reproduce:

    python baselines/cnn_lstm_attn_t1/train.py

References:
    W. Wang et al., "CO concentration detection based on deep learning
    and TDLAS technology", Optics and Lasers in Engineering (2024)
    doi:10.1016/j.optlaseng.2024.108420
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_split, write_predictions_csv  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20260811
EPOCHS = 60
BATCH = 64
LR = 3e-4


class SelfAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        # x: [B, T, D]
        q = self.query(x)  # [B, T, D]
        k = self.key(x)    # [B, T, D]
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [B, T, T]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, x)  # [B, T, D]
        return out.mean(dim=1)  # [B, D]  — attention-weighted pooling


class CNNLSTMAttention(nn.Module):
    def __init__(self, n_points: int, cnn_ch: int = 32, lstm_h: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_ch, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(cnn_ch, cnn_ch * 2, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_ch * 2, hidden_size=lstm_h,
            num_layers=2, batch_first=True, bidirectional=True,
        )
        self.attention = SelfAttentionPool(lstm_h * 2)
        self.head = nn.Sequential(
            nn.Linear(lstm_h * 2, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, 1, n_points]
        cnn_out = self.cnn(x)  # [B, 64, n_points//4]
        seq = cnn_out.permute(0, 2, 1)  # [B, T, 64]
        lstm_out, _ = self.lstm(seq)  # [B, T, 128]
        pooled = self.attention(lstm_out)  # [B, 128]
        return self.head(pooled).squeeze(-1)


def main() -> int:
    torch.manual_seed(SEED)

    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    X_va, y_va, _ = load_split("ch4-t1-val-v0")
    X_te, _, ids_te = load_split("ch4-t1-test-v0")
    X_ho, _, ids_ho = load_split("ch4-t3-test-heldout-v0")

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9

    def prep(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32).unsqueeze(1)

    Xt, Xv = prep(X_tr), prep(X_va)
    ylog = np.log1p(y_tr)
    y_mu, y_sd = float(ylog.mean()), float(ylog.std())
    yt = torch.tensor((ylog - y_mu) / y_sd, dtype=torch.float32)

    model = CNNLSTMAttention(Xt.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    best_state, best_val, log = None, np.inf, []
    g = torch.Generator().manual_seed(SEED)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        scheduler.step()
        model.train(False)
        with torch.no_grad():
            val_pred = np.expm1(model(Xv).numpy() * y_sd + y_mu)
            val_mae_ppm = float(np.mean(np.abs(val_pred - y_va)))
        log.append({"epoch": epoch, "val_mae_ppm": val_mae_ppm})
        if val_mae_ppm < best_val:
            best_val = val_mae_ppm
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"epoch {epoch:02d}: val MAE {val_mae_ppm:.3f} ppm")

    model.load_state_dict(best_state)
    model.train(False)
    with torch.no_grad():
        for X, ids, tag in [(X_te, ids_te, "t1-test"), (X_ho, ids_ho, "t3-test-heldout")]:
            preds = np.expm1(model(prep(X)).numpy() * y_sd + y_mu)
            write_predictions_csv(OUT / f"predictions_{tag}.csv", ids, preds)

    (OUT / "hyperparams.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "architecture": "CNN(2-layer, stride-2) -> BiLSTM(2-layer, h=64) -> SelfAttention -> MLP",
        "scheduler": "CosineAnnealingLR",
        "best_val_mae_ppm": best_val,
        "target_transform": "log1p, z-standardized",
        "train_log": log,
    }, indent=2))
    print(f"best val MAE {best_val:.3f} ppm; predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
