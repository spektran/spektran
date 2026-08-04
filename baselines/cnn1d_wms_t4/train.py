#!/usr/bin/env python
"""1D-CNN baseline for T4 (WMS 2f concentration regression).

Small deterministic torch CNN on the 2f demodulated signal; trains on
log1p(concentration) (the labels are log-uniform over three decades) and
exponentiates at predict time. Hyperparameters are fixed here — no tuning
beyond the val-split early stop. Reproduce:

    python baselines/cnn1d_wms_t4/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_wms_split, write_predictions_csv  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20260813
EPOCHS = 60
BATCH = 64
LR = 3e-4


class CNN1D(nn.Module):
    def __init__(self, n_points: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(64 * 8, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> int:
    torch.manual_seed(SEED)  # no numpy randomness used in this script
    torch.use_deterministic_algorithms(True)

    X_tr, y_tr, _ = load_wms_split("ch4-t4-train-v0")
    X_va, y_va, _ = load_wms_split("ch4-t4-val-v0")
    X_te, _, ids_te = load_wms_split("ch4-t4-test-v0")

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9

    def prep(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32).unsqueeze(1)

    Xt, Xv = prep(X_tr), prep(X_va)
    ylog = np.log1p(y_tr)
    y_mu, y_sd = float(ylog.mean()), float(ylog.std())
    yt = torch.tensor((ylog - y_mu) / y_sd, dtype=torch.float32)

    model = CNN1D(Xt.shape[-1])
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
            val_mae_ppm = float(
                np.mean(np.abs(np.expm1(model(Xv).numpy() * y_sd + y_mu) - y_va))
            )
        log.append({"epoch": epoch, "val_mae_ppm": val_mae_ppm})
        if val_mae_ppm < best_val:
            best_val = val_mae_ppm
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}: val MAE {val_mae_ppm:.3f} ppm")

    model.load_state_dict(best_state)
    model.train(False)
    with torch.no_grad():
        preds = np.expm1(model(prep(X_te)).numpy() * y_sd + y_mu)
        write_predictions_csv(OUT / "predictions_t4-test.csv", ids_te, preds)

    (OUT / "hyperparams.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "best_val_mae_ppm": best_val, "target_transform": "log1p, z-standardized",
        "normalization": "per-feature (train statistics)",
        "train_log": log,
    }, indent=2))
    print(f"best val MAE {best_val:.3f} ppm; predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
