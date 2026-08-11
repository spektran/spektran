#!/usr/bin/env python
"""Physics-Informed Neural Network (PINN) baseline for T1 (and T3 held-out).

Augments a standard MLP regression loss with a physics consistency term:
given the predicted concentration, we forward-simulate the expected peak
absorbance via Beer-Lambert law and compare against the observed peak
absorbance extracted from the raw scan. This physics regularizer forces
the network to produce concentrations that are physically consistent with
the observed absorption signal. Reproduce:

    python baselines/pinn_t1/train.py

Reference:
    Nature Scientific Reports (2025), "Unsupervised spectra information
    extraction using physics-informed neural networks"
    doi:10.1038/s41598-025-25573-5
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
PHYSICS_WEIGHT = 0.1

# Beer-Lambert constants for CH4 2nu3 R(3) at 6047 cm-1
SW_EFF = 1.37e-21      # effective line strength [cm/molecule]
N_DENSITY = 2.479e19   # number density at STP [molecules/cm3]
PATH_CM = 10.0 * 100   # 10 m path in cm


class PhysicsMLP(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.concentration_head = nn.Linear(64, 1)

    def forward(self, x):
        feat = self.features(x)
        conc = self.concentration_head(feat).squeeze(-1)
        return conc


def extract_peak_absorbance(raw_scan: torch.Tensor) -> torch.Tensor:
    """Estimate peak absorbance from raw scan using wing baseline."""
    n = raw_scan.shape[-1]
    wing = int(n * 0.15)
    wing_left = raw_scan[:, :wing].mean(dim=1)
    wing_right = raw_scan[:, -wing:].mean(dim=1)
    baseline = (wing_left + wing_right) / 2.0
    min_val = raw_scan.min(dim=1).values
    transmittance = min_val / baseline.clamp(min=1e-9)
    return -torch.log(transmittance.clamp(min=1e-9))


def beer_lambert_absorbance(conc_ppm: torch.Tensor) -> torch.Tensor:
    """Theoretical peak absorbance from Beer-Lambert law."""
    conc_frac = conc_ppm * 1e-6
    return torch.tensor(SW_EFF * N_DENSITY * PATH_CM) * conc_frac


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

    def raw_tensor(X):
        return torch.tensor(X, dtype=torch.float32)

    Xt, Xv = prep(X_tr), prep(X_va)
    Xt_raw = raw_tensor(X_tr)
    ylog = np.log1p(y_tr)
    y_mu, y_sd = float(ylog.mean()), float(ylog.std())
    yt = torch.tensor((ylog - y_mu) / y_sd, dtype=torch.float32)

    model = PhysicsMLP(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse_fn = nn.MSELoss()
    best_state, best_val, log = None, np.inf, []
    g = torch.Generator().manual_seed(SEED)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()

            pred_norm = model(Xt[idx])
            data_loss = mse_fn(pred_norm, yt[idx])

            pred_ppm = torch.expm1(pred_norm * y_sd + y_mu)
            pred_abs = beer_lambert_absorbance(pred_ppm)
            obs_abs = extract_peak_absorbance(Xt_raw[idx])
            physics_loss = mse_fn(pred_abs, obs_abs)

            loss = data_loss + PHYSICS_WEIGHT * physics_loss
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
        "physics_weight": PHYSICS_WEIGHT,
        "architecture": "256-128-64-1 MLP with Beer-Lambert physics loss",
        "physics_constraint": "Beer-Lambert: predicted conc -> theoretical absorbance vs observed",
        "best_val_mae_ppm": best_val,
        "target_transform": "log1p, z-standardized",
        "train_log": log,
    }, indent=2))
    print(f"best val MAE {best_val:.3f} ppm; predictions under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
