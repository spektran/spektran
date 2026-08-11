#!/usr/bin/env python
"""Bidirectional LSTM baseline for T1 (and T3 held-out).

The most widely cited RNN architecture in TDLAS+ML literature for gas
concentration retrieval. Treats the raw scan as a sequence along the
wavenumber axis; BiLSTM captures dependencies in both scan directions.
Final hidden states are concatenated and passed through a regression head.

    python baselines/bilstm_t1/train.py

Reference:
    Y. Ma et al., "Methane concentration inversion using LSTM neural
    network", Spectrochimica Acta Part A (2022)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_split, write_predictions_csv  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20260811
EPOCHS = 60
BATCH = 64
LR = 1e-3


class BiLSTM(nn.Module):
    def __init__(self, n_points: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=4, padding=3), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=16, hidden_size=hidden,
            num_layers=n_layers, batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, 1, n_points]
        feat = self.downsample(x)  # [B, 16, seq_len]
        feat = feat.permute(0, 2, 1)  # [B, seq_len, 16]
        _, (h_n, _) = self.lstm(feat)  # h_n: [n_layers*2, B, hidden]
        h_fwd = h_n[-2]  # last layer forward
        h_bwd = h_n[-1]  # last layer backward
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # [B, hidden*2]
        return self.head(h_cat).squeeze(-1)


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

    model = BiLSTM(Xt.shape[-1])
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
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
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
        "architecture": "Conv1d(4x downsample) -> BiLSTM(2 layers, hidden=64) -> MLP head",
        "best_val_mae_ppm": best_val,
        "target_transform": "log1p, z-standardized",
        "train_log": log,
    }, indent=2))
    print(f"best val MAE {best_val:.3f} ppm; predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
