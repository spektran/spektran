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

Tasks (v0.2 additions):

- **T4 WMS concentration-regression**: input = demod_2f (WMS lock-in output);
  output = CH4 concentration [ppm]. Metrics: MAE (primary), MAPE. Dataset
  configs shipped (``ch4-t4-*-v0.yaml``); evaluation reuses the T1 pipeline.
- **T5 drift compensation**: input = raw_scan_timeseries; output =
  drift-corrected concentration [ppm]. Metrics: Allan variance improvement
  (primary), MAE. Dataset configs shipped (``ch4-t5-*-v0.yaml``, time-series
  HDF5 layout via ``io.write_time_series``/``read_time_series``); evaluation
  computes Allan deviation per series (see ``evaluate.evaluate_drift``).
- **T6 OOD instrument detection**: input = raw_scan; output = ood_label
  (in-/out-of-distribution). Metrics: AUROC. Dataset configs shipped
  (``ch4-t6-*-v0.yaml``; the test split mixes in-distribution and held-out
  instruments via the ``ood_task`` CLI branch, which stamps
  ``labels.ood_label`` onto each record after generation). Full evaluation
  pipeline (``evaluate.evaluate_ood``) and a PCA+Mahalanobis baseline shipped.
- **T7 cross-modality transfer**: train on TDLAS (T1 training split), test on
  NDIR (scalar ratio). Same gas and concentration range, different measurement
  physics. Metrics: MAE (primary), MAPE, cross-modality degradation vs T1.

Split seeds are disjoint by construction (different master seeds per split;
per-record streams spawn from them independently).
"""

from __future__ import annotations

from dataclasses import dataclass, field

TASKS = (
    "T1-concentration", "T2-denoising", "T3-generalization",
    "T4-wms-concentration", "T5-drift-compensation", "T6-ood-instrument",
    "T7-cross-modality",
)

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
SPLIT_SEEDS.update({
    ("T4", "train"): 104_001,
    ("T4", "val"): 104_002,
    ("T4", "test"): 104_003,
    ("T5", "train"): 105_001,
    ("T5", "test"): 105_002,
    ("T6", "train"): 106_001,
    ("T6", "test"): 106_002,
    ("T7", "test"): 501_001,
})


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
TASK_SPECS.update({
    "T4-wms-concentration": TaskSpec(
        task_id="T4-wms-concentration",
        input_signal="demod_2f",
        target="labels.species[0].concentration_ppm",
        primary_metric="mae",
        secondary_metrics=["mape"],
    ),
    "T5-drift-compensation": TaskSpec(
        task_id="T5-drift-compensation",
        input_signal="raw_scan_timeseries",
        target="labels.species[0].concentration_ppm",
        primary_metric="allan_variance_improvement",
        secondary_metrics=["mae"],
    ),
    "T6-ood-instrument": TaskSpec(
        task_id="T6-ood-instrument",
        input_signal="raw_scan",
        target="labels.ood_label",
        primary_metric="auroc",
        secondary_metrics=[],
    ),
    "T7-cross-modality": TaskSpec(
        task_id="T7-cross-modality",
        input_signal="ndir_ratio",
        target="labels.species[0].concentration_ppm",
        primary_metric="mae",
        secondary_metrics=["mape", "cross_modality_degradation"],
    ),
})
