# Baselines

Reference models for the SPEKTRAN benchmark. They chase reproducibility,
not state of the art (plan §6.3): fixed seeds, published hyperparameters,
one command each.

## Setup

```bash
pip install -e ".[dev]" scikit-learn torch
# generate the official v0 splits first:
for s in t1-train t1-val t1-test t3-test-heldout; do
  python scripts/generate_dataset.py configs/datasets/ch4-$s-v0.yaml --out data
done
```

## Results (v0 splits, CH4 DA, 2026-08-04)

| Model | T1 test MAE (ppm) | T1 MAPE (%) | T3 held-out MAE (ppm) | T3 degradation vs T1 |
|---|---|---|---|---|
| Ridge regression | **2.84** | **29.9** | **3.72** | **1.31x** |
| 1D CNN | 15.58 | 42.2 | 28.30 | 1.82x |

T2 denoising (classical reference, `wing_poly_t2`): spectral RMSE 6.31e-3,
peak-weighted RMSE 8.60e-3 on the T1 test split.

Observations (honest, not tuned away):

- The linear model wins on this v0 configuration: napierian absorbance is
  linear in concentration in the optically thin regime, and ridge averages
  fringe/noise structure effectively. The CNN (small, 60 epochs, CPU,
  log-target) is under-trained by design — it is a reference point, not a
  ceiling.
- The flagship T3 finding: the CNN degrades more across held-out instruments
  (1.82x) than ridge (1.31x) — models can overfit *instrument signatures*,
  which is exactly what the cross-instrument track measures.

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

## Reproduce

```bash
python baselines/ridge_regression/train.py   # ~20 s
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
| T5 Drift compensation | Time-series scans | Drift-corrected ppm | Allan variance improvement | Evaluation stub; baselines pending |
| T6 OOD instrument | Raw scan | In/out-of-distribution | AUROC | Evaluation stub; baselines pending |

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
