# Leaderboard

All results on the official **v0 splits** (CH4, direct absorption unless noted).
Baselines chase **reproducibility**, not state of the art — fixed seeds, published
hyperparameters, one command each. See [`baselines/README.md`](https://github.com/spektran/spektran/blob/main/baselines/README.md)
for reproduction instructions.

!!! info "Submit your results"
    Run `python -m spektran.benchmark.evaluate` on your predictions and open a PR
    adding your row with a link to reproducible code.

---

## T1 — Concentration Regression (DA)

| Model | Type | MAE (ppm) | MAPE (%) | Code |
|---|---|---|---|---|
| Ridge regression | Linear | **2.84** | **29.9** | `baselines/ridge_regression/` |
| 1D CNN | Deep | 15.58 | 42.2 | `baselines/cnn1d/` |
| Patchified Transformer | Deep | — | — | `baselines/transformer_t1/` |

The linear model wins on this v0 configuration: napierian absorbance is linear
in concentration in the optically thin regime, and ridge averages fringe/noise
structure effectively. The CNN (60 epochs, CPU) is under-trained by design — it
is a reference point, not a ceiling.

---

## T2 — Spectral Denoising

| Model | Type | Spectral RMSE | Peak-weighted RMSE | Code |
|---|---|---|---|---|
| Wing-anchored cubic polynomial | Classical | 6.31e-3 | 8.60e-3 | `baselines/wing_poly_t2/` |
| 1D U-Net | Deep | — | — | `baselines/unet_t2/` |

Input: noisy raw scan. Output: clean absorbance spectrum. The classical baseline
fits a cubic polynomial to the absorption-free wings and subtracts it.

---

## T3 — Cross-Instrument Generalization

| Model | Type | MAE (ppm) | Degradation vs T1 | Code |
|---|---|---|---|---|
| Ridge regression | Linear | **3.72** | **1.31x** | `baselines/ridge_regression/` |
| 1D CNN | Deep | 28.30 | 1.82x | `baselines/cnn1d/` |

The flagship finding: the CNN degrades more across held-out instruments (1.82x)
than ridge (1.31x) — deep models can overfit *instrument signatures*, which is
exactly what the cross-instrument track measures.

---

## T4 — WMS Concentration (2f)

| Model | Type | MAE (ppm) | MAPE (%) | Code |
|---|---|---|---|---|
| Ridge (2f) | Linear | **15.15** | 61.4 | `baselines/ridge_wms_t4/` |
| 1D CNN (2f) | Deep | 20.35 | **23.2** | `baselines/cnn1d_wms_t4/` |
| Patchified Transformer | Deep | — | — | `baselines/transformer_t4/` |

Ridge wins on absolute MAE; the CNN wins on MAPE — its log1p target transform
trades absolute error at high concentrations for better relative accuracy across
the log-uniform range.

---

## T5 — Drift Compensation

| Model | Type | Window | MAE (ppm) | ADEV @ 1s | ADEV @ 78s | Code |
|---|---|---|---|---|---|---|
| Moving average | Classical | 100 | **0.270** | 0.0040 | 0.129 | `baselines/moving_avg_t5/` |
| TCN | Deep | 5 | — | — | — | `baselines/tcn_t5/` |

The moving average removes fast per-scan noise but leaves slower structured
error uncorrected — exactly the gap a purpose-built drift-compensation model
should close.

---

## T6 — OOD Instrument Detection

| Model | Type | AUROC | n in-dist | n OOD | Code |
|---|---|---|---|---|---|
| PCA + Mahalanobis | Unsupervised | **0.672** | 500 | 500 | `baselines/mahalanobis_t6/` |

The baseline is fully unsupervised: it fits a PCA-whitened Gaussian to
in-distribution training scans alone and scores test scans by Mahalanobis
distance. 0.672 AUROC is above chance but far from perfect — the held-out
instrument's parameters were deliberately placed between existing tiers.

---

## T7 — Cross-Modality Transfer

| Model | Type | MAE (ppm) | Degradation vs T1 | Code |
|---|---|---|---|---|
| Ridge (TDLAS→NDIR) | Linear | — | — | `baselines/ridge_regression/` |

Train on TDLAS (T1 training split), test on NDIR (scalar active/reference ratio).
Same gas and concentration range, entirely different measurement physics.

---

## T8 — Multi-Species Regression (CH4 + H2O)

| Model | Type | CH4 MAE (ppm) | H2O MAE (ppm) | Aggregate MAE | Code |
|---|---|---|---|---|---|
| Ridge (dual) | Linear | **0.89** | **3937** | **1969** | `baselines/ridge_multispecies_t8/` |

Two independent ridge regressors, one per target species. CH4 is well-recovered
(instrument tuned to 2nu3 band) but H2O is poorly resolved from the same spectral
window — the main challenge for DL models on this track.

---

## T9 — Temperature Regression

| Model | Type | MAE (K) | MAPE (%) | RMSE (K) | Code |
|---|---|---|---|---|---|
| Ridge | Linear | **9.4** | **2.0** | **11.8** | `baselines/ridge_temp_t9/` |

Fixed CH4 concentration (100 ppm), temperature range 250–800 K. The regression
target is gas temperature, inferred from temperature-dependent line-shape changes
(Doppler broadening, Boltzmann population redistribution).

---

## Submitting Results

1. Generate predictions on the official test split using `spektran generate`.
2. Run `python -m spektran.benchmark.evaluate --task <TASK> --truth <truth.h5> --predictions <preds.csv>`.
3. Open a PR adding your row to this page with:
    - Model name and type
    - All reported metrics (same columns as above)
    - Link to reproducible code (repository or inline script)
    - The exact `spektran` version and dataset config used

All submissions must be reproducible from a single command with a fixed seed.
