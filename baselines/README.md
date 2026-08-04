# Baselines

Reference models for the OpenGasSpec benchmark. They chase reproducibility,
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

Observations (honest, not tuned away):

- The linear model wins on this v0 configuration: napierian absorbance is
  linear in concentration in the optically thin regime, and ridge averages
  fringe/noise structure effectively. The CNN (small, 60 epochs, CPU,
  log-target) is under-trained by design — it is a reference point, not a
  ceiling.
- The flagship T3 finding: the CNN degrades more across held-out instruments
  (1.82x) than ridge (1.31x) — models can overfit *instrument signatures*,
  which is exactly what the cross-instrument track measures.

## Reproduce

```bash
python baselines/ridge_regression/train.py
python baselines/cnn1d/train.py
# score any predictions file:
python -m opengasspec.benchmark.evaluate --task T1-concentration \
  --truth data/ch4-t1-test-v0.h5 \
  --predictions baselines/ridge_regression/predictions_t1-test.csv
python -m opengasspec.benchmark.evaluate --task T3-generalization \
  --truth data/ch4-t3-test-heldout-v0.h5 \
  --predictions baselines/ridge_regression/predictions_t3-test-heldout.csv \
  --t1-mae 2.8426
```

Hyperparameters live in each model directory's `hyperparams.json` (written at
train time, including the full validation curve for the CNN).
