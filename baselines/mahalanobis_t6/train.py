#!/usr/bin/env python
"""Mahalanobis-distance baseline for T6 (OOD instrument detection).

Fits a PCA-whitened Gaussian to in-distribution training scans; the OOD
score is the Mahalanobis distance of a test scan from that fitted
distribution -- simple, interpretable, no neural network. Fully
unsupervised: it never reads an ood_label, at training or scoring time
(predictions here are compared to truth only downstream, by
`spektran.benchmark.evaluate.evaluate_ood`).

Reproduce:

    python baselines/mahalanobis_t6/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_split  # noqa: E402

OUT = Path(__file__).resolve().parent

N_COMPONENTS = 50
COV_REG_EPS = 1e-6


def fit_pca_gaussian(X: np.ndarray, n_components: int) -> dict:
    """PCA-whitened Gaussian fit to in-distribution scans X."""
    mu = X.mean(axis=0)
    x_centered = X - mu
    _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    k = min(n_components, len(s))
    v = vt[:k].T
    proj = x_centered @ v
    cov = np.cov(proj, rowvar=False)
    cov_inv = np.linalg.inv(cov + COV_REG_EPS * np.eye(k))
    return {"mu": mu, "v": v, "proj_mean": proj.mean(axis=0), "cov_inv": cov_inv, "k": k}


def mahalanobis_scores(X: np.ndarray, model: dict) -> np.ndarray:
    proj = (X - model["mu"]) @ model["v"]
    diffs = proj - model["proj_mean"]
    return np.sqrt(np.sum(diffs @ model["cov_inv"] * diffs, axis=1))


def main() -> int:
    X_tr, _, _ = load_split("ch4-t6-train-v0")
    X_te, _, ids_te = load_split("ch4-t6-test-v0")

    model = fit_pca_gaussian(X_tr, N_COMPONENTS)
    scores = mahalanobis_scores(X_te, model)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "predictions_t6-test.csv", "w") as f:
        f.write("record_id,ood_score\n")
        for rid, score in zip(ids_te, scores):
            f.write(f"{rid},{score:.6f}\n")

    (OUT / "hyperparams.json").write_text(json.dumps({
        "method": "PCA + Mahalanobis distance",
        "n_components": model["k"],
        "cov_regularization_eps": COV_REG_EPS,
        "n_train": int(X_tr.shape[0]),
        "n_features": int(X_tr.shape[1]),
    }, indent=2))
    print(f"{len(ids_te)} predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
