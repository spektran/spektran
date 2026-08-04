# Quickstart

## Install

```bash
git clone https://github.com/opengasspec/opengasspec
cd opengasspec
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## One clean spectrum (offline, ~10 lines)

```python
from opengasspec.physics import simulate_absorbance

nu, absorbance = simulate_absorbance(
    molecule="CH4", concentration_ppm=100.0,
    temperature_K=296.0, pressure_atm=1.0, path_length_m=10.0,
    wavenumber_start_cm1=6046.0, wavenumber_end_cm1=6048.0,
)
```

Uses the built-in approximate CH4 demo line list. For authoritative spectra
install the HITRAN extra (`pip install -e ".[hitran]"`) and pass
`lines=fetch_lines("CH4", 6045.0, 6049.0)` (network required on first call).

## Generate an official dataset split

```bash
python scripts/generate_dataset.py configs/datasets/ch4-t1-train-v0.yaml --out data
```

Reproducible bit-for-bit: the config pins the master seed, instrument
configs, and gas-truth distributions. The four official v0 splits total
~285 MB on disk and generate in well under a minute.
(`configs/datasets/ch4-da-medium-v0.yaml` is a standalone 10k-record demo
config, not part of the benchmark splits.)

## Train and score a baseline

```bash
pip install scikit-learn torch   # torch only needed for the CNN baseline
for s in t1-train t1-val t1-test t3-test-heldout; do
  python scripts/generate_dataset.py configs/datasets/ch4-$s-v0.yaml --out data
done
python baselines/ridge_regression/train.py   # ~20 s
python -m opengasspec.benchmark.evaluate --task T1-concentration \
  --truth data/ch4-t1-test-v0.h5 \
  --predictions baselines/ridge_regression/predictions_t1-test.csv
```

The CNN baseline (`baselines/cnn1d/train.py`) takes ~7 minutes on CPU; full
scoring commands for both models, including the T3 `--t1-mae` convention,
are in [baselines/README.md](https://github.com/opengasspec/opengasspec/blob/main/baselines/README.md).
