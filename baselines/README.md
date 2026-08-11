# Baselines

22 reference models for the SPEKTRAN benchmark. They chase reproducibility,
not state of the art: fixed seeds, published hyperparameters, one command each.

All baselines are registered in [`registry.yaml`](registry.yaml) and can be
trained via the AI Agent-ready CLI:

```bash
spektran train --baseline ridge --json    # auto-generates data, trains, reports scores
spektran list baselines --json            # list all 22 baselines with metadata
```

### Architecture coverage (added 2026-08-11)

| Category | Models | Task |
|----------|--------|------|
| Linear | Ridge | T1/T3/T4/T7/T8/T9 |
| Tree ensemble | Random Forest, Gradient Boosting | T1/T3 |
| MLP | MLP (BPNN) | T1/T3 |
| CNN | 1D CNN | T1/T3/T4 |
| RNN | BiLSTM | T1/T3 |
| Hybrid | CNN-LSTM-Attention | T1/T3 |
| Transformer | Patchified Transformer | T1/T3/T4 |
| Denoising | U-Net, LSTM-DAE, Wing-Poly | T2 |
| Drift | Moving Average, TCN | T5 |
| OOD | PCA+Mahalanobis | T6 |
| Physics | Voigt Fit (LM), PINN | T1/T3 |
| Ensemble (novel) | SpektralNet | T1/T3 |

## Setup (manual)

```bash
pip install -e ".[dev]" scikit-learn torch
# generate the official v0 splits first:
for s in t1-train t1-val t1-test t3-test-heldout; do
  spektran generate configs/datasets/ch4-$s-v0.yaml --out data
done
```

## Results (v0 splits, CH4 DA, 2026-08-12)

| Model | T1 MAE (ppm) | T1 MAPE (%) | T3 MAE (ppm) | T3 degradation |
|---|---|---|---|---|
| **SpektralNet** | **2.27** | **22.5** | **3.51** | 1.54x |
| Ridge regression | 2.84 | 29.9 | 3.72 | **1.31x** |
| Random Forest | 5.27 | 24.1 | 10.89 | 2.07x |
| PINN | 7.29 | 49.1 | 15.62 | 2.14x |
| Transformer | 7.39 | 22.7 | 10.81 | 1.46x |
| MLP (BPNN) | 8.08 | 44.5 | 9.85 | 1.22x |
| 1D CNN | 15.58 | 42.2 | 28.30 | 1.82x |
| BiLSTM | 29.47 | 61.7 | 51.04 | 1.73x |
| CNN-LSTM-Attention | 38.39 | 69.4 | 71.03 | 1.85x |

### T2 Denoising

| Model | Spectral RMSE | Peak-weighted RMSE |
|---|---|---|
| **U-Net** | **0.00362** | **0.00858** |
| Wing-Poly | 0.00631 | 0.00860 |
| LSTM-DAE | 0.00994 | 0.02785 |

### Observations (honest, not tuned away)

- **SpektralNet** achieves the best T1 MAE (2.27 ppm, 20% over Ridge) by
  augmenting Ridge's raw-scan features with 6 physics-informed scalars
  (peak absorbance, center depth, spectral width, etc.) extracted from the
  same scan. The optimal blend is 80% augmented + 20% raw Ridge.
- **Linear models dominate**: in the optically thin regime Beer-Lambert
  absorbance is linear in concentration; Ridge exploits this directly and
  all deep models (CNN, LSTM, Transformer, PINN) perform worse.
- **Complexity hurts**: model complexity inversely correlates with T1 MAE
  on this 3000-sample benchmark. Ridge (201 params) > MLP (50K) > CNN >
  BiLSTM > CNN-LSTM-Attention.
- **T3 generalization**: MLP has the best cross-instrument generalization
  among deep models (1.22x) — simplicity is robustness. PINN's physics
  loss actually hurts T3 (2.14x) because fixed Beer-Lambert constants
  overfit to training instrument path lengths.
- **T2 denoising**: U-Net dominates; skip connections preserve spectral
  detail that the LSTM-DAE bottleneck destroys.

## T4 Results (v0 WMS splits, CH4 2f, 2026-08-05)

| Model | T4 test MAE (ppm) | T4 MAPE (%) |
|---|---|---|
| Ridge (2f) | 15.15 | 61.4 |
| 1D CNN (2f) | 20.35 | 23.2 |

Ridge wins on absolute MAE; the CNN wins on MAPE — its log1p target transform
(see `cnn1d_wms_t4/train.py`) trades absolute error at high concentrations for
better relative accuracy across the log-uniform range. Reproduce:

```bash
python baselines/ridge_wms_t4/train.py    # ~1 min
python baselines/cnn1d_wms_t4/train.py    # ~7 min on CPU (60 epochs)
```

```bash
python -m spektran.benchmark.evaluate --task T4-wms-concentration \
  --truth data/ch4-t4-test-v0.h5 \
  --predictions baselines/ridge_wms_t4/predictions_t4-test.csv
python -m spektran.benchmark.evaluate --task T4-wms-concentration \
  --truth data/ch4-t4-test-v0.h5 \
  --predictions baselines/cnn1d_wms_t4/predictions_t4-test.csv
```

## T5 Results (v0 time-series split, CH4 DA, 2026-08-05)

| Model | Window | Test MAE (ppm) | ADEV @ tau=1s | ADEV @ tau=78s |
|---|---|---|---|---|
| Moving average (ridge + per-series smoothing) | 100 | **0.270** | 0.0040 | 0.129 |

The official v0 split is 20 train / 10 test time series of 200 consecutive
1 s scans each, one frozen `vi-da-medium-02` realization per series (the true
concentration is fixed per series -- only the *measured* value drifts).
`evaluate_drift` recovers series boundaries from truth-concentration jumps
with no extra metadata (every scan in a series shares an exactly equal true
concentration by construction), computes Allan deviation within each series
so a series boundary is never mistaken for a drift jump, and averages the
per-series ADEV curves. Reproduce:

```bash
for s in t5-train t5-test; do
  spektran generate configs/datasets/ch4-$s-v0.yaml --out data
done
python baselines/moving_avg_t5/train.py    # ~10 s
python -m spektran.benchmark.evaluate --task T5-drift-compensation \
  --truth data/ch4-t5-test-v0.h5 \
  --predictions baselines/moving_avg_t5/predictions_t5-test.csv
```

Observations (honest, not tuned away):

- Per-scan ridge alone (window=1) gets 0.41 ppm test MAE; a per-series
  moving average keeps reducing it as the window grows, still improving at
  window=100 (half the series length) with no turnaround. For this v0 split
  (medium-tier instrument, 200 s series) averageable noise dominates and no
  Allan-variance "bathtub" turnaround from drift shows up within this
  timescale -- a longer series or a harder-tier instrument would be needed
  to see one.
- The Allan-deviation curve of the *residual* (smoothed prediction minus
  truth) still rises from 0.0040 ppm at tau=1s to 0.129 ppm at tau=78s: the
  moving average removes fast per-scan noise but leaves a slower structured
  error uncorrected -- exactly the gap a purpose-built drift-compensation
  model should close relative to this reference smoother.
- The window is selected by MAE on the training series themselves, not on
  test (T5 v0 has no dedicated val split -- seeds are reserved per split in
  `benchmark/tasks.py` and none is reserved for a T5 val set yet).
- A naive `np.convolve(..., mode="same")` moving average implicitly zero-pads
  past each series' ends, biasing edge predictions toward zero badly enough
  to make every window choice look worse than no smoothing at all; the
  shipped implementation renormalizes by the true per-position overlap count
  instead (see `moving_avg_t5/train.py:moving_average`).

## T6 Results (v0 OOD split, CH4 DA, 2026-08-05)

| Model | Test AUROC | n in-dist | n OOD |
|---|---|---|---|
| PCA + Mahalanobis distance | **0.672** | 500 | 500 |

The official v0 split trains on 3000 in-distribution records (easy+medium+
hard DA instruments; no `ood_label` at all) and tests on 1000 records: 500
more from those same three instruments plus 500 from the held-out
`vi-da-heldout-07` instrument (`ood_task: true` dataset config -- `spektran
generate` stamps `labels.ood_label` after generation; see
`docs/benchmark.md`). Reproduce:

```bash
for s in t6-train t6-test; do
  spektran generate configs/datasets/ch4-$s-v0.yaml --out data
done
python baselines/mahalanobis_t6/train.py    # ~5 s
python -m spektran.benchmark.evaluate --task T6-ood-instrument \
  --truth data/ch4-t6-test-v0.h5 \
  --predictions baselines/mahalanobis_t6/predictions_t6-test.csv
```

Observations (honest, not tuned away):

- 0.672 AUROC is well above chance (0.5) but far from perfect separation:
  `vi-da-heldout-07`'s parameter ranges were deliberately placed between the
  medium and hard training tiers (distinct etalon FSR bands and
  scan-nonlinearity signatures, but not an obviously different noise
  regime -- see the instrument config comment), so a fair share of its scans
  land inside the in-distribution Mahalanobis-distance range.
- The baseline is fully unsupervised: it never reads a single `ood_label`,
  at training or scoring time. It fits a 50-component PCA-whitened Gaussian
  to the in-distribution training scans alone and scores test scans by
  distance from that fit. A method that gets to see even a few
  held-out-instrument examples during model selection should beat this
  comfortably.

## T8 Results (v0 multi-species split, CH4+H2O DA, 2026-08-05)

| Model | CH4 MAE (ppm) | H2O MAE (ppm) | Aggregate MAE (ppm) |
|---|---|---|---|
| Ridge (dual) | **0.89** | **3937** | **1969** |

Two independent ridge regressors, one per target species. CH4 is well-recovered
(instrument tuned to 2nu3 band) but H2O is poorly resolved from the same spectral
window. Reproduce:

```bash
for s in h2o-t8-train h2o-t8-test; do
  spektran generate configs/datasets/ch4-$s-v0.yaml --out data
done
python baselines/ridge_multispecies_t8/train.py    # ~5 s
python -m spektran.benchmark.evaluate --task T8-multispecies \
  --truth data/ch4-h2o-t8-test-v0.h5 \
  --predictions baselines/ridge_multispecies_t8/predictions_t8-test.csv
```

## T9 Results (v0 temperature regression split, CH4 DA, 2026-08-05)

| Model | MAE (K) | MAPE (%) | RMSE (K) |
|---|---|---|---|
| Ridge | **9.4** | **2.0** | **11.8** |

Fixed CH4 concentration (100 ppm), temperature range 250-800 K. Reproduce:

```bash
for s in t9-train t9-test; do
  spektran generate configs/datasets/ch4-$s-v0.yaml --out data
done
python baselines/ridge_temp_t9/train.py    # ~5 s
python -m spektran.benchmark.evaluate --task T9-temperature \
  --truth data/ch4-t9-test-v0.h5 \
  --predictions baselines/ridge_temp_t9/predictions_t9-test.csv
```

## SpektralNet — Novel TDLAS-Native Model

SpektralNet is a dual-domain physics-augmented Ridge ensemble designed
specifically for TDLAS concentration recovery. It achieves the best T1 MAE
(2.27 ppm, 20% improvement over standard Ridge) by augmenting the raw scan
features with 6 physics-informed scalars extracted via Beer-Lambert wing-baseline
correction:

1. **Peak absorbance** — `-log(min_transmittance)`, direct Beer-Lambert measure
2. **Center depth (normalized)** — depth of absorption dip relative to baseline
3. **Center contrast** — mean absorption in center vs wings
4. **Wing asymmetry** — left/right baseline imbalance (instrument signature)
5. **Integrated absorption** — total area under absorption proxy curve
6. **Spectral width** — variance-based width of the absorption feature

The ensemble blends two Ridge regressors (80% augmented + 20% raw features),
selected by validation MAE grid search. The key insight: in the optically thin
regime where Beer-Lambert absorbance is linear in concentration, enhancing
Ridge's input space with physics features outperforms replacing Ridge with
deeper models.

```bash
python baselines/spektralnet_t1/train.py   # ~30 s
```

## Reproduce

```bash
python baselines/ridge_regression/train.py   # ~20 s
python baselines/spektralnet_t1/train.py     # ~30 s (best T1 MAE)
python baselines/cnn1d/train.py              # ~7 min on CPU (60 epochs)
```

Score each model on T1, then on T3. The `--t1-mae` flag is the SAME MODEL's
T1 test MAE (from its T1 evaluate output) — it feeds the T3 degradation
ratio, so always substitute your own model's number:

```bash
# ridge
python -m spektran.benchmark.evaluate --task T1-concentration \
  --truth data/ch4-t1-test-v0.h5 \
  --predictions baselines/ridge_regression/predictions_t1-test.csv
python -m spektran.benchmark.evaluate --task T3-generalization \
  --truth data/ch4-t3-test-heldout-v0.h5 \
  --predictions baselines/ridge_regression/predictions_t3-test-heldout.csv \
  --t1-mae 2.8426   # <- ridge's own T1 MAE from the previous command

# cnn1d
python -m spektran.benchmark.evaluate --task T1-concentration \
  --truth data/ch4-t1-test-v0.h5 \
  --predictions baselines/cnn1d/predictions_t1-test.csv
python -m spektran.benchmark.evaluate --task T3-generalization \
  --truth data/ch4-t3-test-heldout-v0.h5 \
  --predictions baselines/cnn1d/predictions_t3-test-heldout.csv \
  --t1-mae 15.5807  # <- cnn1d's own T1 MAE
```

Hyperparameters live in each model directory's `hyperparams.json` (written at
train time, including the full validation curve for the CNN). Prediction
files are NOT shipped in the repository — running `train.py` regenerates
them; determinism means your regenerated files should match the official
scores exactly (the leaderboard table rounds to 2 decimals; full precision is
in `scores_*.json`).

## T4/T5/T6 tasks (v0.2)

Added in schema v0.2:

| Task | Input | Output | Primary metric | Status |
|---|---|---|---|---|
| T4 WMS concentration | 2f demod signal | ppm | MAE | Dataset configs shipped; baselines shipped |
| T5 Drift compensation | Time-series scans | Drift-corrected ppm | Allan variance improvement | Full pipeline shipped |
| T6 OOD instrument | Raw scan | In/out-of-distribution | AUROC | Full pipeline shipped |

T4 dataset generation:

```bash
for s in t4-train t4-val t4-test; do
  python scripts/generate_dataset.py configs/datasets/ch4-$s-v0.yaml --out data
done
```

T4 evaluation uses the same pipeline as T1:

```bash
python -m spektran.benchmark.evaluate --task T4-wms-concentration \
  --truth data/ch4-t4-test-v0.h5 --predictions preds_t4.csv
```

### T5: Drift compensation

T5 dataset generation (time-series mode: `spektran generate` detects
`mode: time_series` in the config and calls `generate_time_series` per
series instead of `generate_dataset`):

```bash
for s in t5-train t5-test; do
  spektran generate configs/datasets/ch4-$s-v0.yaml --out data
done
```

T5 evaluation reads the time-series HDF5 layout (`spektran.io.read_time_series`)
and computes Allan deviation of the prediction error within each series:

```bash
python -m spektran.benchmark.evaluate --task T5-drift-compensation \
  --truth data/ch4-t5-test-v0.h5 --predictions preds_t5.csv
```

### T6: OOD instrument detection

T6 dataset generation: the train split is a normal `generate_dataset` run
(in-distribution instruments only, no OOD label). The test split needs
`ood_task: true` in the config -- `spektran generate` detects it, calls
`generate_dataset` once per instrument pool (`instrument_config_in_dist`,
`instrument_config_ood`), and stamps `labels.ood_label` onto each record
afterward (0 for the in-distribution pool, 1 for the OOD pool):

```bash
for s in t6-train t6-test; do
  spektran generate configs/datasets/ch4-$s-v0.yaml --out data
done
```

T6 evaluation reads `labels.ood_label` from the truth file and computes
AUROC (`spektran.benchmark.metrics.ood_auroc`) against a predictions CSV of
`record_id,ood_score` (higher = more confidently OOD):

```bash
python -m spektran.benchmark.evaluate --task T6-ood-instrument \
  --truth data/ch4-t6-test-v0.h5 --predictions preds_t6.csv
```

## Cross-Modality Track (T7)

Train on TDLAS (direct absorption scans), test on NDIR (broadband ratio).
Same gas (CH4), same concentration range, different measurement physics.

The challenge: a model that learns spectral features from TDLAS 2000-point
scans must generalize to predicting concentration from a single NDIR ratio
value. This tests whether the model learns the underlying gas physics or
merely overfits to modality-specific signal characteristics.

Dataset generation:

```bash
# Training data: use existing TDLAS split
spektran generate configs/datasets/ch4-t1-train-v0.yaml --out data
# Cross-modality test data: NDIR
spektran generate configs/datasets/ch4-cross-modality-test-v0.yaml --out data
```

Evaluation:

```bash
python -m spektran.benchmark.evaluate --task T7-cross-modality \
  --truth data/ch4-cross-modality-test-v0.h5 \
  --predictions preds_t7.csv \
  --t1-mae <your_t1_mae>
```

Reference baseline: `baselines/ridge_cross_modality_t7/`

```bash
python baselines/ridge_cross_modality_t7/train.py
```

The Ridge baseline uses a physics-bridged approach: it extracts integrated
Beer-Lambert absorbance from both modalities. For TDLAS, a wing-anchored cubic
polynomial baseline is subtracted and -ln(transmittance) integrated over the
absorption center. For NDIR, the observed active/reference ratio is normalized
by the zero-gas Planck baseline ratio (computed from source temperature and
filter parameters), yielding transmittance in the same physical space.

Result: MAE 130.68 ppm, degradation 46.02x vs T1 Ridge. The 46x decomposes as
~44x from information reduction (2000 spectral points → 1 scalar) and ~1.05x
from actual domain gap — the physics bridge successfully places both modalities
in the same feature space.
