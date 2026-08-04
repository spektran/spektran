"""Benchmark task definitions and official split logic (plan §6).

A *task release* is fully specified by dataset-config YAMLs under
``configs/datasets/`` — the split is defined by generation configs and master
seeds, not by shipping static files, so anyone can regenerate the official
data bit-for-bit with the pinned generator version.

Tasks (v0):

- **T1 concentration-regression**: input = noisy raw scan (DA) or demod_2f
  (WMS); output = CH4 concentration [ppm]. Metrics: MAE (primary), MAPE.
- **T2 denoising**: input = noisy raw scan; output = clean absorbance
  spectrum. Metrics: spectral RMSE (primary), peak-weighted RMSE.
- **T3 cross-instrument generalization**: same I/O as T1, but the test split
  comes from held-out instruments (vi-da-heldout-07 / vi-wms-heldout-08)
  whose parameter ranges are excluded from the training distribution.
  Metrics: MAE (primary), degradation ratio vs the T1 test MAE.

Split seeds are disjoint by construction (different master seeds per split;
per-record streams spawn from them independently).
"""

from __future__ import annotations

from dataclasses import dataclass, field

TASKS = ("T1-concentration", "T2-denoising", "T3-generalization")

# Difficulty tiers -> instrument configs used in the train/val/test mixtures
TIER_INSTRUMENTS = {
    "easy": ["vi-da-easy-01"],
    "medium": ["vi-da-medium-02"],
    "hard": ["vi-da-hard-03"],
}
HELD_OUT_INSTRUMENTS = ["vi-da-heldout-07"]

# v0 scale (plan §6.2 notes scale is negotiable; v0 keeps generation fast for
# contributors — the config format scales to 50k/5k/10k unchanged)
V0_SPLIT_SIZES = {"train": 5000, "val": 500, "test": 1000}

# Master seeds per (task, split) — disjoint, never reuse
SPLIT_SEEDS = {
    ("T1", "train"): 101_001,
    ("T1", "val"): 101_002,
    ("T1", "test"): 101_003,
    ("T3", "test"): 103_001,  # held-out instruments; train/val shared with T1
}


@dataclass
class TaskSpec:
    task_id: str
    input_signal: str
    target: str
    primary_metric: str
    secondary_metrics: list[str] = field(default_factory=list)


TASK_SPECS = {
    "T1-concentration": TaskSpec(
        task_id="T1-concentration",
        input_signal="raw_scan",
        target="labels.species[0].concentration_ppm",
        primary_metric="mae",
        secondary_metrics=["mape"],
    ),
    "T2-denoising": TaskSpec(
        task_id="T2-denoising",
        input_signal="raw_scan",
        target="absorbance_clean",
        primary_metric="spectral_rmse",
        secondary_metrics=["peak_weighted_rmse"],
    ),
    "T3-generalization": TaskSpec(
        task_id="T3-generalization",
        input_signal="raw_scan",
        target="labels.species[0].concentration_ppm",
        primary_metric="mae",
        secondary_metrics=["mape", "degradation_ratio_vs_T1"],
    ),
}
