#!/usr/bin/env python
"""TCN baseline for T5 (drift compensation).

Dilated-convolution TCN over a sliding window of W=5 consecutive scans,
stacked as input channels: every channel is a raw_scan over the same
2000-point (wavelength) axis, so the shared per-feature train statistics
used to standardize a single scan in the other torch baselines are applied
identically to each channel. Convolutions run along the scan-position axis
(not across the window), with increasing dilation to widen the receptive
field over that axis. Each window predicts the concentration of its most
recent (current) scan; series boundaries (truth-concentration jumps, same
detection `evaluate_drift` uses) are never crossed -- the start of a series
is padded with its own first scan repeated. Trains on log1p(concentration)
like the other torch baselines. T5 v0 has no dedicated val split (see
`moving_avg_t5`), so there is no early stopping or checkpoint selection --
the model trains for a fixed EPOCHS and is evaluated once at the end.
Reproduce:

    python baselines/tcn_t5/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_time_series_split, write_predictions_csv  # noqa: E402

OUT = Path(__file__).resolve().parent
SEED = 20260818
EPOCHS = 60
BATCH = 64
LR = 3e-4
WINDOW_SIZE = 5


class TCN(nn.Module):
    """Temporal Convolutional Network for drift-compensated concentration.

    Takes a window of W consecutive scans [B, W, 2000] and predicts
    the concentration of the center (most recent) scan. Dilated
    convolutions capture multi-scale temporal patterns.
    """

    def __init__(self, window_size: int = 5, n_points: int = 2000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(window_size, 32, 15, padding=7, dilation=1), nn.ReLU(),
            nn.Conv1d(32, 32, 9, padding=8, dilation=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=4, dilation=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(64 * 8, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_windows(
    X: np.ndarray, y: np.ndarray, ids: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create sliding windows from time-series data.

    The T5 data is a concatenation of independent series, each with a
    fixed true concentration. Series boundaries are detected by
    concentration jumps (same as the evaluator).
    """
    boundaries = np.flatnonzero(np.diff(y) != 0) + 1
    series_list = np.split(np.arange(len(X)), boundaries)

    X_win, y_win, ids_win = [], [], []
    for series_idx in series_list:
        for i, idx in enumerate(series_idx):
            win_indices = []
            for w in range(window_size - 1, -1, -1):
                j = max(0, i - w)
                win_indices.append(series_idx[j])
            X_win.append(X[win_indices])
            y_win.append(y[idx])
            ids_win.append(ids[idx])

    return np.stack(X_win), np.array(y_win), ids_win


def main() -> int:
    torch.manual_seed(SEED)  # no numpy randomness used in this script
    torch.use_deterministic_algorithms(True)

    X_tr, y_tr, ids_tr = load_time_series_split("ch4-t5-train-v0")
    X_te, y_te, ids_te = load_time_series_split("ch4-t5-test-v0")

    Xw_tr, yw_tr, _ = make_windows(X_tr, y_tr, ids_tr, WINDOW_SIZE)
    Xw_te, yw_te, ids_win_te = make_windows(X_te, y_te, ids_te, WINDOW_SIZE)

    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-9

    def prep(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32)

    Xt, Xte = prep(Xw_tr), prep(Xw_te)
    ylog = np.log1p(yw_tr)
    y_mu, y_sd = float(ylog.mean()), float(ylog.std())
    yt = torch.tensor((ylog - y_mu) / y_sd, dtype=torch.float32)

    model = TCN(window_size=WINDOW_SIZE, n_points=Xt.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    log = []
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
            train_mae_ppm = float(
                np.mean(np.abs(np.expm1(model(Xt).numpy() * y_sd + y_mu) - yw_tr))
            )
        log.append({"epoch": epoch, "train_mae_ppm": train_mae_ppm})
        print(f"epoch {epoch:02d}: train MAE {train_mae_ppm:.3f} ppm")

    model.train(False)
    with torch.no_grad():
        preds = np.expm1(model(Xte).numpy() * y_sd + y_mu)
    test_mae = float(np.mean(np.abs(preds - yw_te)))

    write_predictions_csv(OUT / "predictions_t5-test.csv", ids_win_te, preds)
    (OUT / "hyperparams.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "window_size": WINDOW_SIZE, "optimizer": "Adam",
        "target_transform": "log1p, z-standardized",
        "normalization": "per-feature (train statistics), shared across window channels",
        "model_selection": "final epoch (no val split for T5 v0, see moving_avg_t5)",
        "final_train_mae_ppm": log[-1]["train_mae_ppm"],
        "test_mae_ppm": test_mae,
        "train_log": log,
    }, indent=2))
    print(f"test MAE {test_mae:.3f} ppm; predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
