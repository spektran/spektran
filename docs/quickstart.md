# Quickstart

## Option 0: AI Agent (zero code)

If you use an AI coding agent (Claude Code, Cursor, GitHub Copilot, Windsurf),
just tell it what you want:

> "Train the ridge baseline on the concentration task and show me the MAE"

The agent reads [`AGENTS.md`](https://github.com/spektran/spektran/blob/main/AGENTS.md)
and operates the full pipeline through `spektran` CLI commands with `--json` output.
No manual steps needed. See the [AI Agent interface](https://github.com/spektran/spektran/blob/main/AGENTS.md) for the full command reference.

## Option 1: just the data (no install)

The official v0 splits are hosted on Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("spektran/spektran-ch4-v0")                          # CH4 benchmark
ds = load_dataset("spektran/spektran-co2-v0", "da")                    # CO2 DA benchmark
ds = load_dataset("spektran/spektran-industrial-v0", "so2")            # SO2 industrial
ds = load_dataset("spektran/spektran-multigas-v0", "ch4_co2_h2o")      # Multi-gas mixture
```

Everything below is for running the engine, regenerating data bit-for-bit,
and scoring benchmark submissions.

## Install

```bash
git clone https://github.com/spektran/spektran
cd spektran
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## One clean spectrum (offline, ~10 lines)

```python
from spektran.physics import simulate_absorbance

nu, absorbance = simulate_absorbance(
    molecule="CH4", concentration_ppm=100.0,
    temperature_K=296.0, pressure_atm=1.0, path_length_m=10.0,
    wavenumber_start_cm1=6046.0, wavenumber_end_cm1=6048.0,
)
```

Uses the built-in approximate CH4 demo line list. For authoritative spectra
install the HITRAN extra (`pip install -e ".[hitran]"`) and pass
`lines=fetch_lines("CH4", 6045.0, 6049.0)` (network required on first call).

## Multi-species example

```python
from spektran.physics import demo_ch4_2nu3, demo_h2o, absorption_coefficient
import numpy as np

nu = np.linspace(6046.0, 6048.0, 2000)
alpha_ch4 = absorption_coefficient(nu, demo_ch4_2nu3(), 100e-6, 296.0, 1.0)
alpha_h2o = absorption_coefficient(nu, demo_h2o(), 0.01, 296.0, 1.0)  # 1% H2O
total = alpha_ch4 + alpha_h2o  # Beer-Lambert linear superposition
```

Demo line lists are also available for CO2 (`demo_co2()`) and CO (`demo_co()`).
Each demo line list is centered on its own representative sensing band (CH4
~6047 cm⁻¹, H2O ~7187 cm⁻¹, CO2 ~4978 cm⁻¹, CO ~2171 cm⁻¹), so combining two
demo species over one narrow scan window mostly exercises the superposition
code path rather than showing visible spectral overlap. For a window with
genuine cross-species interference, fetch HITRAN lines for both species over
the same range with `fetch_lines()`.

## Generate an official dataset split

Using the CLI (v0.2+):

```bash
spektran generate configs/datasets/ch4-t1-train-v0.yaml --out data
```

Or using the script directly:

```bash
python scripts/generate_dataset.py configs/datasets/ch4-t1-train-v0.yaml --out data
```

Reproducible bit-for-bit: the config pins the master seed, instrument
configs, and gas-truth distributions. The four official v0 splits total
~285 MB on disk and generate in well under a minute.
(`configs/datasets/ch4-da-medium-v0.yaml` is a standalone 10k-record demo
config, not part of the benchmark splits.)

## Train and score a baseline

**One command** (auto-generates data if missing):

```bash
spektran train --baseline ridge --json
```

**Step by step** (manual control):

```bash
pip install scikit-learn torch   # torch only needed for the CNN baseline
for s in t1-train t1-val t1-test t3-test-heldout; do
  spektran generate configs/datasets/ch4-$s-v0.yaml --out data
done
python baselines/ridge_regression/train.py   # ~20 s
spektran benchmark --task T1-concentration \
  --truth data/ch4-t1-test-v0.h5 \
  --predictions baselines/ridge_regression/predictions_t1-test.csv
```

The CNN baseline (`baselines/cnn1d/train.py`) takes ~7 minutes on CPU; full
scoring commands for both models, including the T3 `--t1-mae` convention,
are in [baselines/README.md](https://github.com/spektran/spektran/blob/main/baselines/README.md).
