"""Shared data loading for baselines: HDF5 splits -> (X, y, record_ids)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spektran.io import read_records  # noqa: E402

DATA = REPO / "data"


def load_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a split by dataset_id (e.g. 'ch4-t1-train-v0').

    Returns (X, y, record_ids): X = raw_scan matrix [n, n_points],
    y = CH4 concentration [ppm].
    """
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["raw_scan"] for r in records])
    y = np.array(
        [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    )
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def load_wms_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a WMS split: X = demod_2f [n, n_points], y = concentration [ppm]."""
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["demod_2f"] for r in records])
    y = np.array(
        [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    )
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def write_predictions_csv(path: Path, ids: list[str], y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("record_id,concentration_ppm\n")
        for i, v in zip(ids, y_pred):
            f.write(f"{i},{float(v):.6f}\n")
