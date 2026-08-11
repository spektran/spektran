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
| T7 cross-modality transfer | TDLAS train, NDIR test | concentration (ppm) | MAE + degradation |
| T8 multi-species regression | raw scan (CH4+H2O) | CH4 + H2O concentrations (ppm) | aggregate MAE |
| T9 temperature regression | raw scan (fixed conc) | gas temperature (K) | MAE |

## Official splits (v0)

Defined entirely by configs under `configs/datasets/` (instrument mixtures +
disjoint master seeds). Regenerate locally with `spektran generate` (or
`scripts/generate_dataset.py`); identical bytes for everyone at the same
generator version. All CLI commands support `--json` for AI agent integration.

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
- T7: train on T1 TDLAS splits / test 1000 NDIR records
  (`ch4-cross-modality-test-v0`). NDIR splits also available: train 3000 /
  test 1000 / heldout 500 (4 NDIR virtual instruments).
- T8: train 5000 / test 1000 multi-species (CH4 + H2O).
- T9: train 5000 / test 1000 temperature regression (fixed 100 ppm CH4).

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

## HITRAN production data

The standard splits use approximate demo line lists (3 lines) for fast offline
generation. HITRAN production variants use the full 76-line CH4 list from
HITRAN2020 (fetched via hapi). Available for T1/T3 and T4:

```bash
for s in t1-train-v0-hitran t1-val-v0-hitran t1-test-v0-hitran \
         t3-test-heldout-v0-hitran \
         t4-train-v0-hitran t4-val-v0-hitran t4-test-v0-hitran; do
  spektran generate configs/datasets/ch4-$s.yaml --out data
done
```

Ridge baseline comparison (demo vs HITRAN): T1 MAE nearly identical (2.84 →
2.77 ppm, -2.5%), T3 improves 13% (3.72 → 3.24 ppm), T4 WMS becomes harder
(15.32 → 24.87 ppm, +62%) due to richer 2f spectral complexity from 76 lines.

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

### T7: Cross-modality transfer (TDLAS → NDIR)

Train on TDLAS direct-absorption scans (T1 training split), test on NDIR
scalar ratios. Same gas (CH4) and concentration range, entirely different
measurement physics: TDLAS provides a 2000-point spectrum while NDIR
collapses to a single active/reference detector ratio.

Dataset configs: `ch4-ndir-{train,test,test-heldout}-v0.yaml` (NDIR splits),
`ch4-cross-modality-test-v0.yaml` (the actual T7 test set: NDIR ratios with
concentrations drawn from the same distribution as T1). Training uses the
standard T1 TDLAS splits — the challenge is zero-shot transfer to the NDIR
modality. Primary metric: MAE + degradation ratio vs T1 Ridge.

Reference baseline: `baselines/ridge_cross_modality_t7` (Planck-normalized
integrated absorbance bridge; MAE 130.68 ppm, 46.02x degradation). The 46x
degradation decomposes as ~44x from information reduction (2000 spectral
points → 1 scalar) and ~1.05x from actual domain gap. The physics bridge
computes zero-gas Planck baseline ratios per instrument, normalizes observed
ratios to transmittance, and extracts -ln(transmittance) as the absorption
feature — placing both TDLAS and NDIR features in the same Beer-Lambert
absorbance space.

### T8: Multi-species regression (CH4 + H2O)

Input: raw DA scan containing overlapping absorption from CH4 (target) and
H2O (interferent at random concentrations). Output: both CH4 and H2O
concentrations. Primary metric: aggregate MAE (average of per-species MAE).
Tests whether models can disentangle overlapping absorption features from
two species in the same spectral window.

Dataset configs: `ch4-h2o-t8-{train,test}-v0.yaml`. CH4 range 1-1000 ppm
(log-uniform), H2O range 100-20000 ppm (log-uniform). Instrument:
`vi-da-multispecies-13`. Prediction format: CSV with
`record_id,ch4_ppm,h2o_ppm`. Reference baseline:
`baselines/ridge_multispecies_t8` (two independent ridge regressors; CH4
MAE 0.89 ppm, H2O MAE 3937 ppm — H2O is poorly resolved because the
instrument is tuned to CH4's 2nu3 band).

### T9: Temperature regression

Input: raw DA scan at fixed CH4 concentration (100 ppm). Output: gas
temperature (K). Primary metric: MAE (K). Tests whether models can infer
temperature from temperature-dependent line-shape changes (Doppler width
scales as sqrt(T), Boltzmann population redistributes across rotational
states).

Dataset configs: `ch4-t9-{train,test}-v0.yaml`. Temperature range 250-800 K
(uniform). Instrument: `vi-da-temp-regression-14` (low noise, wide T range).
Prediction format: CSV with `record_id,temperature_K`. Reference baseline:
`baselines/ridge_temp_t9` (ridge regression; MAE 9.4 K, MAPE 2.0%).

---

## Additional molecule datasets

Beyond the CH4 v0 benchmark, SPEKTRAN provides pre-built datasets for other
molecules and multi-gas scenarios on Hugging Face:

| Dataset | Molecules | Configs | Use case |
|---------|-----------|---------|----------|
| [spektran-co2-v0](https://huggingface.co/datasets/spektran/spektran-co2-v0) | CO2 | `da`, `wms` | CO2 concentration + cross-instrument + WMS |
| [spektran-industrial-v0](https://huggingface.co/datasets/spektran/spektran-industrial-v0) | SO2, NO, CO | `so2`, `no`, `co` | Industrial emission monitoring |
| [spektran-multigas-v0](https://huggingface.co/datasets/spektran/spektran-multigas-v0) | CH4+CO2+H2O, CO+CO2 | `ch4_co2_h2o`, `co_co2` | Multi-species mixture regression |

These datasets use the same task structure (T1-style concentration regression)
and can be loaded with one line:

```python
from datasets import load_dataset
ds = load_dataset("spektran/spektran-co2-v0", "da")
