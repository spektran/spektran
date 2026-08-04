# Benchmark rules (v0)

## Tasks

| Task | Input | Output | Primary metric |
|---|---|---|---|
| T1 concentration regression | noisy raw scan | CH4 concentration (ppm) | MAE |
| T2 denoising | noisy raw scan | clean absorbance spectrum | spectral RMSE |
| T3 cross-instrument generalization | same as T1 | concentration (ppm) | MAE + degradation vs T1 |

## Official splits (v0)

Defined entirely by configs under `configs/datasets/` (instrument mixtures +
disjoint master seeds). Regenerate locally with `scripts/generate_dataset.py`;
identical bytes for everyone at the same generator version.

- T1: train 5000 / val 500 / test 1000, mixture of easy+medium+hard DA
  instruments.
- T3 test: 1000 records from the held-out instrument `vi-da-heldout-07`,
  whose parameter ranges are excluded from the training distribution.

## Rules

1. Train on the official train split; tune only on val. The test truths ship
   with the data (simulation is open) — the leaderboard is honor-system plus
   mandatory reproducible code links; CI re-runs submissions on regenerated
   splits with a different verification seed where feasible.
2. No use of `provenance.noise_config` or `absorbance_clean` at inference
   time for T1/T3 (they are labels/oracle signals, not inputs).
3. Report all metrics from `python -m opensensorsim.benchmark.evaluate`
   unmodified, and the exact command used.
4. T3 submissions must use the SAME model/weights as T1 (no held-out-specific
   tuning): the track measures transfer, not adaptation.
