#!/usr/bin/env python
"""Multi-Layer Perceptron (BPNN) baseline for T1 (and T3 held-out).

The classic fully-connected neural network used by Ma et al. and Li et al.
for CH4 concentration inversion in TDLAS. Simple 3-layer MLP via PyTorch
with log1p target transform and early stopping on validation MAE. Reproduce:

    python baselines/mlp_t1/train.py
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
EPOCHS = 80
BATCH = 64
LR = 1e-3


class MLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> int:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    X_tr, y_tr, _ = load_split("ch4-t1-train-v0")
    X_va, y_va, _ = load_split("ch4-t1-val-v0")
    X_te, _, ids_te = load_split("ch4-t1-test-v0")
    X_ho, _, ids_ho = load_split("ch4-t3-test-heldout-v0")

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9

    def prep(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32)

    Xt, Xv = prep(X_tr), prep(X_va)
    ylog = np.log1p(y_tr)
    y_mu, y_sd = float(ylog.mean()), float(ylog.std())
    yt = torch.tensor((ylog - y_mu) / y_sd, dtype=torch.float32)

    model = MLP(Xt.shape[1])
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
        "architecture": "256-128-64-1 with ReLU + Dropout(0.1)",
        "best_val_mae_ppm": best_val,
        "target_transform": "log1p, z-standardized",
        "train_log": log,
    }, indent=2))
    print(f"best val MAE {best_val:.3f} ppm; predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
