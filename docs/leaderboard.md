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
| **SpektralNet** | Ensemble | **2.27** | 22.5 | `baselines/spektralnet_t1/` |
| Ridge regression | Linear | 2.84 | 29.9 | `baselines/ridge_regression/` |
| Random Forest | Tree | 5.27 | 24.1 | `baselines/random_forest_t1/` |
| PINN | Physics | 7.29 | 49.1 | `baselines/pinn_t1/` |
| Patchified Transformer | Deep | 7.39 | **22.7** | `baselines/transformer_t1/` |
| MLP (BPNN) | Deep | 8.08 | 44.5 | `baselines/mlp_t1/` |
| 1D CNN | Deep | 15.58 | 42.2 | `baselines/cnn1d/` |
| BiLSTM | Deep | 29.47 | 61.7 | `baselines/bilstm_t1/` |
| CNN-LSTM-Attention | Deep | 38.39 | 69.4 | `baselines/cnn_lstm_attn_t1/` |

SpektralNet achieves SOTA by augmenting Ridge's raw-scan features with 6
physics-informed Beer-Lambert scalars and blending two Ridge regressors (80%
augmented + 20% raw). Linear models dominate because absorbance is linear in
concentration; model complexity inversely correlates with performance on this
3000-sample benchmark.

---

## T2 — Spectral Denoising

| Model | Type | Spectral RMSE | Peak-weighted RMSE | Code |
|---|---|---|---|---|
| 1D U-Net | Deep | **3.62e-3** | **8.58e-3** | `baselines/unet_t2/` |
| Wing-anchored cubic polynomial | Classical | 6.31e-3 | 8.60e-3 | `baselines/wing_poly_t2/` |
| LSTM-DAE | Deep | 9.94e-3 | 27.85e-3 | `baselines/lstm_dae_t2/` |

The U-Net dominates both metrics; its skip connections preserve spectral detail
that the LSTM-DAE's bottleneck destroys. The classical wing polynomial is
competitive on peak-weighted RMSE — it performs well precisely where it matters
most (the absorption peak region).

---

## T3 — Cross-Instrument Generalization

| Model | Type | MAE (ppm) | Degradation vs T1 | Code |
|---|---|---|---|---|
| **SpektralNet** | Ensemble | **3.51** | 1.54x | `baselines/spektralnet_t1/` |
| Ridge regression | Linear | 3.72 | **1.31x** | `baselines/ridge_regression/` |
| MLP (BPNN) | Deep | 9.85 | 1.22x | `baselines/mlp_t1/` |
| Patchified Transformer | Deep | 10.81 | 1.46x | `baselines/transformer_t1/` |
| Random Forest | Tree | 10.89 | 2.07x | `baselines/random_forest_t1/` |
| PINN | Physics | 15.62 | 2.14x | `baselines/pinn_t1/` |
| 1D CNN | Deep | 28.30 | 1.82x | `baselines/cnn1d/` |
| BiLSTM | Deep | 51.04 | 1.73x | `baselines/bilstm_t1/` |
| CNN-LSTM-Attention | Deep | 71.03 | 1.85x | `baselines/cnn_lstm_attn_t1/` |

SpektralNet has the best absolute T3 MAE (3.51 ppm) but not the best
degradation ratio. MLP (1.22x) and Ridge (1.31x) generalize best — simplicity
is robustness. PINN's physics loss actually hurts T3 (2.14x) because fixed
Beer-Lambert constants overfit to training instrument path lengths.

---

## T4 — WMS Concentration (2f)

| Model | Type | MAE (ppm) | MAPE (%) | Code |
|---|---|---|---|---|
| Ridge (2f) | Linear | **15.15** | 61.4 | `baselines/ridge_wms_t4/` |
| Patchified Transformer | Deep | 17.83 | **16.4** | `baselines/transformer_t4/` |
| 1D CNN (2f) | Deep | 20.35 | 23.2 | `baselines/cnn1d_wms_t4/` |

Ridge wins on absolute MAE; the Transformer achieves the best MAPE (16.4%) —
its log1p target transform trades absolute error at high concentrations for
better relative accuracy across the log-uniform range.

---

## T5 — Drift Compensation

| Model | Type | Window | MAE (ppm) | ADEV @ 1s | ADEV @ 78s | Code |
|---|---|---|---|---|---|---|
| Moving average | Classical | 100 | **0.270** | **0.0040** | **0.129** | `baselines/moving_avg_t5/` |
| TCN | Deep | 5 | 4.233 | 0.132 | 0.415 | `baselines/tcn_t5/` |

The moving average dominates: its 100-scan window matches the slow drift
timescale, while the TCN's 5-scan context misses the low-frequency structure
entirely. The TCN also overfits badly (train MAE 0.43 vs test 4.23 ppm) —
without a dedicated T5 validation split, no early stopping is possible.
Temporal context length matters more than model complexity for drift.

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
| Ridge (TDLAS→NDIR) | Linear | **130.68** | **46.02x** | `baselines/ridge_cross_modality_t7/` |

Train on TDLAS (T1 training split), test on NDIR (scalar active/reference ratio).
The 46x degradation decomposes as ~44x from information reduction (2000-point
spectrum → 1 scalar ratio) and ~1.05x from actual domain gap. The physics bridge
uses Planck-normalized integrated absorbance: the TDLAS model extracts integrated
Beer-Lambert absorbance from wing-baseline-corrected scans, while NDIR ratios are
normalized by the zero-gas Planck baseline to yield transmittance in the same
physical space.

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
