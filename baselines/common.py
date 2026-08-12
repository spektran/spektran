"""Shared data loading for baselines: HDF5 splits -> (X, y, record_ids)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spektran.io import read_records, read_time_series  # noqa: E402

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


def load_time_series_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a T5 time-series split: X = raw_scan [n, n_points], y = concentration [ppm].

    Unlike load_split/load_wms_split, records are kept in temporal
    (generation) order, NOT sorted by record_id -- downstream series-boundary
    detection (spektran.benchmark.evaluate.evaluate_drift) depends on it.
    """
    records, _ = read_time_series(DATA / f"{name}.h5")
    X = np.stack([r["arrays"]["raw_scan"] for r in records])
    y = np.array(
        [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    )
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def load_multispecies_split(
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load T8 multi-species split: X, y_ch4, y_h2o, ids."""
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["raw_scan"] for r in records])
    y_ch4 = np.array(
        [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    )
    y_h2o = np.array([
        r["meta"]["conditions"].get("interferents", [{}])[0].get("concentration_ppm", 0.0)
        for r in records
    ])
    ids = [r["meta"]["record_id"] for r in records]
    return X, y_ch4, y_h2o, ids


def load_temperature_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load T9 temperature regression split: X, y_temperature_K, ids."""
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["raw_scan"] for r in records])
    y = np.array(
        [r["meta"]["conditions"]["temperature_K"] for r in records]
    )
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def load_crds_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a CRDS split: X = tau_spectrum [n, n_points], y = concentration [ppm]."""
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["tau_spectrum"] for r in records])
    y = np.array(
        [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    )
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def load_doas_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a DOAS split: X = doas_spectrum [n, n_points], y = concentration [ppm]."""
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["doas_spectrum"] for r in records])
    y = np.array(
        [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    )
    ids = [r["meta"]["record_id"] for r in records]
    return X, y, ids


def load_ftir_split(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load an FTIR split: X = ftir_spectrum [n, n_points], y = concentration [ppm]."""
    records = read_records(DATA / f"{name}.h5")
    records.sort(key=lambda r: r["meta"]["record_id"])
    X = np.stack([r["arrays"]["ftir_spectrum"] for r in records])
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


def write_multispecies_csv(
    path: Path, ids: list[str], ch4_pred: np.ndarray, h2o_pred: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("record_id,ch4_ppm,h2o_ppm\n")
        for i, c, h in zip(ids, ch4_pred, h2o_pred):
            f.write(f"{i},{float(c):.6f},{float(h):.6f}\n")


def write_temperature_csv(path: Path, ids: list[str], y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("record_id,temperature_K\n")
        for i, v in zip(ids, y_pred):
            f.write(f"{i},{float(v):.4f}\n")
