# Benchmark rules (v0)

## Tasks

| Task | Input | Output | Primary metric |
|---|---|---|---|
| T1 concentration regression | noisy raw scan | CH4 concentration (ppm) | MAE |
| T2 denoising | noisy raw scan | clean absorbance spectrum | spectral RMSE |
| T3 cross-instrument generalization | same as T1 | concentration (ppm) | MAE + degradation vs T1 |
| T4 WMS concentration | noisy 2f signal (WMS) | CH4 concentration (ppm) | MAE |
| T5 drift compensation | time-series raw scans | drift-corrected concentrations | Allan variance improvement |
| T6 OOD instrument detection | raw scan | in-dist vs OOD binary | AUROC |

## Official splits (v0)

Defined entirely by configs under `configs/datasets/` (instrument mixtures +
disjoint master seeds). Regenerate locally with `scripts/generate_dataset.py`;
identical bytes for everyone at the same generator version.

- T1: train 5000 / val 500 / test 1000, mixture of easy+medium+hard DA
  instruments.
- T3 test: 1000 records from the held-out instrument `vi-da-heldout-07`,
  whose parameter ranges are excluded from the training distribution.
- T4: train 5000 / val 500 / test 1000, mixture of easy+medium+hard WMS
  instruments. Uses the same generation pipeline as T1 but with WMS configs.
- T5: train 20 / test 10 time series of 200 consecutive 1 s scans each, one
  frozen `vi-da-medium-02` realization per series (`mode: time_series`).
- T6: train 3000 (in-distribution only: easy+medium+hard DA) / test 1000
  (500 in-distribution + 500 held-out `vi-da-heldout-07`, `ood_task: true`).

## Dataset scale options

The official splits above use moderate sizes for fast iteration. For training
larger models or studying scaling behavior, large-scale configs are available:

| Config | Records | Seed | Purpose |
|---|---|---|---|
| `ch4-t1-train-v0` | 5,000 | 101001 | Standard training |
| `ch4-t1-train-v0-50k` | 50,000 | 201001 | Large-scale training |
| `ch4-t1-val-v0` | 500 | 101002 | Standard validation |
| `ch4-t1-val-v0-5k` | 5,000 | 201002 | Large-scale validation |
| `ch4-t1-test-v0` | 1,000 | 101003 | Standard test |
| `ch4-t1-test-v0-10k` | 10,000 | 201003 | Large-scale test |

Generate large-scale data:

```bash
for s in t1-train-v0-50k t1-val-v0-5k t1-test-v0-10k; do
  spektran generate configs/datasets/ch4-$s.yaml --out data
done
```

Large-scale seeds (201xxx) are disjoint from standard seeds (101xxx), so no
records overlap between scale tiers.

## Rules

1. Train on the official train split; tune only on val. The test truths ship
   with the data (simulation is open) — the leaderboard is honor-system plus
   mandatory reproducible code links; CI re-runs submissions on regenerated
   splits with a different verification seed where feasible.
2. No use of `provenance.noise_config` or `absorbance_clean` at inference
   time for T1/T3 (they are labels/oracle signals, not inputs).
3. Report all metrics from `python -m spektran.benchmark.evaluate`
   unmodified, and the exact command used.
4. T3 submissions must use the SAME model/weights as T1 (no held-out-specific
   tuning): the track measures transfer, not adaptation.

## New tasks (v0.2)

### T4: WMS 2f concentration

Same evaluation pipeline as T1, but input is the 2f demodulated signal from
WMS instruments. Tests whether models can extract concentration from the
calibration-free 2f peak-height.

### T5: Drift compensation

Input: a time series of repeated scans from a single instrument session
(`spektran generate` with `mode: time_series` in the config, one frozen
instrument realization per series; the true concentration is fixed per
series -- only the *measured* value drifts). Output: drift-corrected
concentration trajectory. Primary metric: Allan deviation of the prediction
error (`n_scans`, `mae_ppm`, `adev_shortest_tau`/`adev_longest_tau`, and the
full `adev_curve` over `adev_taus_s`). `evaluate_drift` recovers series
boundaries from truth-concentration jumps (every scan in a series shares an
exactly equal true concentration by construction) so Allan deviation is
computed within each series and averaged, never across a series boundary.
Prediction format is the same CSV as T1 (`record_id,concentration_ppm`).
Reference baseline: `baselines/moving_avg_t5` (ridge + per-series moving
average).

### T6: OOD instrument detection

Input: raw scan. Output: binary classification (in-distribution vs
out-of-distribution instrument). Primary metric: AUROC. Tests whether models
can identify spectra from instruments outside their training distribution.

Training data (`ch4-t6-train-v0`) is drawn only from the three in-distribution
DA instruments -- it carries no `ood_label` at all. The test split
(`ch4-t6-test-v0`) is generated from an `ood_task: true` dataset config:
`spektran generate` loads two disjoint instrument pools
(`instrument_config_in_dist`, `instrument_config_ood`), generates each
independently via the normal `generate_dataset` path, then stamps
`labels.ood_label` (0 or 1) onto every record's metadata afterward -- the
label is a property of which instrument pool produced a scan, not something
the physics model itself predicts. Prediction format is a CSV of
`record_id,ood_score` (higher = more confidently OOD; need not be a
probability -- `evaluate_ood`/`ood_auroc` are rank-based). Reference
baseline: `baselines/mahalanobis_t6` (PCA-whitened Gaussian fit to
in-distribution training scans; OOD score = Mahalanobis distance from that
fit).
