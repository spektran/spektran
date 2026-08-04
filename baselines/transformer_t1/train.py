#!/usr/bin/env python
"""1D-Transformer baseline for T1 (and T3 held-out evaluation).

Patchified transformer encoder on the raw scan: splits the 2000-point scan
into 50-point patches, projects each patch to d_model, adds learnable
position embeddings, and runs a standard transformer encoder before
global-average pooling to a scalar. Trains on log1p(concentration) (the
labels are log-uniform over three decades) and exponentiates at predict
time. Hyperparameters are fixed here — no tuning beyond the val-split
early stop. Reproduce:

    python baselines/transformer_t1/train.py
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
SEED = 20260815
EPOCHS = 80
BATCH = 64
LR = 1e-4
WEIGHT_DECAY = 0.01
PATCH_SIZE = 50
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 128
DROPOUT = 0.1


class TransformerRegressor(nn.Module):
    """Patchified 1D Transformer encoder for spectral regression.

    Splits the 2000-point scan into patches, projects to d_model, adds
    learnable position embeddings, runs through transformer encoder
    layers, and pools to a scalar output via global average pooling.
    """

    def __init__(
        self,
        n_points: int = 2000,
        patch_size: int = 50,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_patches = n_points // patch_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        # x: [B, 1, 2000] -> patch -> [B, 40, 50]
        B = x.shape[0]
        x = x.squeeze(1).reshape(B, -1, self.patch_size)  # [B, 40, 50]
        x = self.proj(x) + self.pos_embed  # [B, 40, d_model]
        x = self.encoder(x)  # [B, 40, d_model]
        x = x.mean(dim=1)  # [B, d_model] — global average pooling
        return self.head(x).squeeze(-1)  # [B]


def main() -> int:
    torch.manual_seed(SEED)  # no numpy randomness used in this script
    torch.use_deterministic_algorithms(True)

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

    model = TransformerRegressor(
        n_points=Xt.shape[-1], patch_size=PATCH_SIZE, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
        for X, ids, tag in [(X_te, ids_te, "t1-test"), (X_ho, ids_ho, "t3-test-heldout")]:
            preds = np.expm1(model(prep(X)).numpy() * y_sd + y_mu)
            write_predictions_csv(OUT / f"predictions_{tag}.csv", ids, preds)

    (OUT / "hyperparams.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW",
        "patch_size": PATCH_SIZE, "d_model": D_MODEL, "nhead": NHEAD,
        "num_layers": NUM_LAYERS, "dim_ff": DIM_FF, "dropout": DROPOUT,
        "best_val_mae_ppm": best_val, "target_transform": "log1p, z-standardized",
        "normalization": "per-feature (train statistics)",
        "train_log": log,
    }, indent=2))
    print(f"best val MAE {best_val:.3f} ppm; predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
