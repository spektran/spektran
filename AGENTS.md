# AGENTS.md — AI Agent Interface for SPEKTRAN

> **You are an AI agent.** This file tells you everything you need to operate
> SPEKTRAN's full ML pipeline — simulate, generate, train, evaluate — via CLI.
> No prior knowledge of spectroscopy required.

## What is SPEKTRAN?

An open-source simulation engine + ML benchmark for optical gas sensing (TDLAS/NDIR).
It generates physically rigorous synthetic training data and provides 9 benchmark tasks
with 14 baselines. Think "MNIST for gas sensing" — but grounded in real physics.

- **Version**: 0.5.0
- **License**: Apache-2.0 (code), CC BY 4.0 (data)
- **Python**: 3.10+
- **Install**: `pip install spektran`

## Bootstrap: Your First Command

Run this to understand the project state:

```bash
spektran info --json
```

This returns: version, available tasks, baselines, data directory, and how much data
has been generated. Every command supports `--json` for structured output.

## Available Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `spektran info --json` | Project overview (start here) | See above |
| `spektran list tasks --json` | All 9 benchmark tasks with metrics | Discovery |
| `spektran list baselines --json` | All 14 baselines with scores | Discovery |
| `spektran list instruments --json` | All 18 virtual instruments | Discovery |
| `spektran list datasets --json` | All datasets + generation status | Discovery |
| `spektran status --json` | Data & training state | Inspection |
| `spektran generate <config> --out data --json` | Generate dataset from YAML config | Data pipeline |
| `spektran train --baseline <name> --json` | Train baseline (auto-generates data) | Training |
| `spektran benchmark <args>` | Evaluate predictions | Evaluation |
| `spektran download` | Show HF dataset download instructions | Data access |

## Common Workflows

### 1. Train a baseline (one command)

```bash
spektran train --baseline ridge --json
```

This automatically: checks if data exists -> generates missing splits -> runs training
script -> reports scores. No manual steps needed.

### 2. Generate data, train, evaluate (step by step)

```bash
# Generate training data
spektran generate configs/datasets/ch4-t1-train-v0.yaml --out data --json

# Generate test data
spektran generate configs/datasets/ch4-t1-test-v0.yaml --out data --json

# Train a baseline
spektran train --baseline ridge --task T1 --json

# Evaluate predictions
spektran benchmark --task T1-concentration \
  --truth data/ch4-t1-test-v0.h5 \
  --predictions baselines/ridge_regression/predictions_t1-test.csv
```

### 3. Explore and compare baselines

```bash
# What tasks are available?
spektran list tasks --json

# What baselines exist for a task?
spektran list baselines --json | python -c "
import sys, json
for b in json.load(sys.stdin):
    if 'T1' in b['tasks']:
        print(f\"{b['name']:20s} {b.get('scores',{}).get('T1',{}).get('scores',{}).get('mae_ppm','N/A')}\")
"

# Train all T1 baselines and compare
for b in ridge cnn1d transformer; do
  spektran train --baseline $b --task T1 --json
done
```

### 4. Custom model development

```python
# Load data programmatically
from spektran.io import read_records
import numpy as np

records = read_records("data/ch4-t1-train-v0.h5")
X = np.stack([r["arrays"]["raw_scan"] for r in records])
y = np.array([r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records])

# Train your model
# model.fit(X, y)

# Write predictions for evaluation
with open("my_predictions.csv", "w") as f:
    f.write("record_id,concentration_ppm\n")
    for r, pred in zip(records, predictions):
        f.write(f"{r['meta']['record_id']},{pred:.6f}\n")
```

Then evaluate:
```bash
spektran benchmark --task T1-concentration \
  --truth data/ch4-t1-test-v0.h5 \
  --predictions my_predictions.csv
```

## Error Handling

All errors in `--json` mode return structured output:

```json
{"error": "Unknown baseline 'foo'. Available: ridge, cnn1d, ...", "code": 1}
```

Non-zero exit codes always indicate failure. Parse `error` field for diagnostics.

## Project Structure

```
spektran/
  src/spektran/          # Python package (physics engine, CLI, benchmark)
  configs/
    instruments/         # 18 virtual instrument configs (vi-*.yaml)
    datasets/            # 25+ dataset split configs (ch4-*.yaml)
  baselines/
    registry.yaml        # Baseline metadata registry (14 baselines)
    ridge_regression/    # Each baseline has: train.py, predictions, scores
    cnn1d/
    transformer_t1/
    ...
  data/                  # Generated HDF5 datasets (not in git)
  schema/                # JSON Schema for records and instruments
```

## Key Concepts

- **Task**: A benchmark problem (T1-T9). Each has input signal type, target, and metrics.
- **Baseline**: A reference ML model. Registry at `baselines/registry.yaml`.
- **Virtual instrument**: A YAML config defining noise/artifact characteristics.
- **Dataset split**: A YAML config + generated HDF5 file (train/val/test).
- **Record**: One simulated spectrum with metadata, stored in HDF5.

## Benchmark Tasks

| ID | Name | Input | Target | Metric |
|----|------|-------|--------|--------|
| T1 | Concentration regression | DA scan | ppm | MAE |
| T2 | Spectral denoising | Noisy spectrum | Clean absorbance | RMSE |
| T3 | Cross-instrument generalization | DA scan (held-out) | ppm | Degradation ratio |
| T4 | WMS concentration | 2f signal | ppm | MAE |
| T5 | Drift compensation | Time-series | Corrected ppm | Allan variance |
| T6 | OOD instrument detection | DA scan | In/OOD label | AUROC |
| T7 | Cross-modality transfer | TDLAS -> NDIR | ppm | Degradation ratio |
| T8 | Multi-species regression | DA scan | CH4 + H2O ppm | Aggregate MAE |
| T9 | Temperature regression | DA scan | Temperature (K) | MAE |

## Pre-built Datasets (No Generation Needed)

```python
from datasets import load_dataset
ds = load_dataset("spektran/spektran-ch4-v0", "da")  # T1/T3 concentration
```

Available configs: `da`, `wms`, `drift`, `ood`, `ndir`, `multispecies`, `temperature`,
`da_hitran`, `wms_hitran`, `da_large`.

## Extending the Project

To add a new baseline:

1. Create `baselines/<name>/train.py` following existing patterns (see `baselines/common.py`)
2. Add entry to `baselines/registry.yaml`
3. Run: `spektran train --baseline <name> --json`

To modify simulation physics:

- Forward models: `src/spektran/physics/` (lineshape, absorption, wms, ndir)
- Noise chain: `src/spektran/instrument/` (laser, detector, etalon, optics)
- Instrument configs: `configs/instruments/vi-*.yaml`

## Links

- [GitHub](https://github.com/spektran/spektran)
- [Documentation](https://spektran.github.io/spektran/)
- [Leaderboard](https://spektran.github.io/spektran/leaderboard/)
- [HF Dataset](https://huggingface.co/datasets/spektran/spektran-ch4-v0)
- [PyPI](https://pypi.org/project/spektran/)
