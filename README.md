<div align="center">

[**English**](README.md) | [中文](README_zh.md)

<img src="assets/logo.jpg" alt="SPEKTRAN" width="560">

### The MNIST of Gas Sensing

**AI Agent-Ready simulation engine + ML benchmark for optical gas sensing**<br>
*HITRAN-grade physics. 9 tasks. 14 baselines. Full pipeline via natural language.*

<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spektran/spektran/blob/main/notebooks/quickstart_colab.ipynb)
[![GitHub Stars](https://img.shields.io/github/stars/spektran/spektran?style=flat-square)](https://github.com/spektran/spektran/stargazers)
[![CI](https://img.shields.io/github/actions/workflow/status/spektran/spektran/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/spektran/spektran/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spektran?style=flat-square&color=blue)](https://pypi.org/project/spektran/)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/spektran/)
[![License](https://img.shields.io/badge/code-Apache%202.0-green?style=flat-square)](LICENSE)
[![License](https://img.shields.io/badge/data-CC%20BY%204.0-green?style=flat-square)](LICENSE-DATA)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21790394-blue?style=flat-square)](https://doi.org/10.5281/zenodo.21790394)
[![AI Agent Ready](https://img.shields.io/badge/%F0%9F%A4%96_AI_Agent-Ready-blueviolet?style=flat-square)](AGENTS.md)

[**Documentation**](https://spektran.github.io/spektran/) &nbsp;&middot;&nbsp;
[**Interactive Demo**](https://huggingface.co/spaces/spektran/spektran-demo) &nbsp;&middot;&nbsp;
[**Leaderboard**](https://spektran.github.io/spektran/leaderboard/) &nbsp;&middot;&nbsp;
[**Datasets**](https://huggingface.co/spektran) &nbsp;&middot;&nbsp;
[**Baselines**](https://huggingface.co/spektran/spektran-baselines-v0)

</div>

<br>

> **You don't need to be a spectroscopist.** &nbsp;If you work on regression, denoising, domain generalization, or anomaly detection, SPEKTRAN gives you 9 ready-to-use benchmark tasks backed by real physics — with the same convenience as MNIST, but grounded in a domain where ML has direct industrial impact.
>
> **You don't even need to type code.** &nbsp;SPEKTRAN is fully **AI Agent-ready** — tell Claude Code, Cursor, or any coding agent what you want, and it operates the entire pipeline through natural language. See [AGENTS.md](AGENTS.md).

<br>

## Highlights

<table>
<tr>
<td width="50%" valign="top">

### Simulation Engine
- **10 molecules** — CH4, H2O, CO2, CO, NH3, NO, NO2, SO2, HCl, HF
- **2 modalities** — TDLAS (DA + WMS) and NDIR
- **Advanced line shapes** — Voigt & Hartmann-Tran Profile
- **14+ virtual instruments** with realistic noise chains
- **WMS 1f–4f** demodulation + 2f/1f calibration-free ratio

</td>
<td width="50%" valign="top">

### ML Benchmark
- **9 tasks** (T1–T9) — regression, denoising, OOD, transfer, multi-species
- **14 baselines** — Ridge, CNN, Transformer, U-Net, TCN, ...
- **Official splits** — train / val / test / held-out instrument
- **AI Agent-ready CLI** — `--json` on every command, discoverable API
- **Public leaderboard** on GitHub Pages

</td>
</tr>
</table>

<br>

## Quick Start

**Zero-install** — load data from Hugging Face in one line:

```python
from datasets import load_dataset
ds = load_dataset("spektran/spektran-ch4-v0")       # CH4 benchmark
ds = load_dataset("spektran/spektran-co2-v0", "da") # CO2 benchmark
ds = load_dataset("spektran/spektran-industrial-v0", "so2")  # Industrial SO2
ds = load_dataset("spektran/spektran-multigas-v0", "ch4_co2_h2o")  # Multi-gas
```

**Full engine** — simulate spectra locally:

```bash
pip install spektran
```

```python
from spektran.physics import simulate_absorbance

nu, absorbance = simulate_absorbance(
    molecule="CH4", concentration_ppm=100.0,
    temperature_K=296.0, pressure_atm=1.0,
    path_length_m=10.0,
    wavenumber_start_cm1=6046.0, wavenumber_end_cm1=6048.0,
)
```

<details>
<summary><b>More examples</b> — WMS, multi-species, CLI</summary>

<br>

**WMS 2f signal:**

```python
from spektran.physics.wms import WMSConfig, simulate_wms
```

See [`examples/wms_ch4.py`](examples/wms_ch4.py) for a complete WMS example.

**Multi-species (CH4 + H2O interferent):**

```python
from spektran.physics import absorption_coefficient
from spektran.physics.hitran import demo_ch4_2nu3, demo_h2o

alpha_ch4 = absorption_coefficient(nu, demo_ch4_2nu3(), 100e-6, 296.0, 1.0)
alpha_h2o = absorption_coefficient(nu, demo_h2o(), 0.01, 296.0, 1.0)
```

See [`examples/multispecies_ch4_h2o.py`](examples/multispecies_ch4_h2o.py).

**CLI** (all commands support `--json` for AI agents):

```bash
spektran info --json                      # Project discovery (agent bootstrap)
spektran list tasks --json                # Available benchmark tasks
spektran train --baseline ridge --json    # Train with auto data generation
spektran generate configs/datasets/ch4-t1-train-v0.yaml --out data --json
spektran benchmark --task T1-concentration --truth data/test.h5 --predictions preds.csv
```

</details>

<br>

## AI Agent Ready

SPEKTRAN is designed from the ground up for the AI agent era. Every CLI command outputs
structured JSON, every resource is discoverable, and the full ML pipeline — **simulate →
generate → train → evaluate** — runs with zero manual steps.

**Works with**: Claude Code, Cursor, GitHub Copilot, Windsurf, Cline, and any agent that
can run shell commands.

**Agent interface**: [`AGENTS.md`](AGENTS.md) — the agent reads this file and immediately
understands how to operate the entire project.

```
You: "Train the ridge baseline on T1 and show me the scores"

Agent: spektran train --baseline ridge --task T1 --json
       → {"baseline": "ridge", "scores": {"T1": {"mae_ppm": 2.84, "mape_pct": 29.87}}}
```

<details>
<summary><b>Agent workflow examples</b></summary>

<br>

**Discovery** — agent bootstraps itself:
```bash
spektran info --json           # What is this project? What's available?
spektran list tasks --json     # 9 tasks with metrics and available baselines
spektran list baselines --json # 14 baselines with pre-computed scores
spektran status --json         # What data exists? What's been trained?
```

**One-command training** — agent trains any baseline:
```bash
spektran train --baseline ridge --json        # Auto-generates data if missing
spektran train --baseline transformer --json  # Works for any registered baseline
spektran train --baseline cnn1d --task T1 --json  # Target a specific task
```

**Compare all baselines** — agent scripts a leaderboard run:
```bash
for baseline in ridge cnn1d transformer; do
  spektran train --baseline $baseline --task T1 --json
done
```

**Custom model** — agent writes code, evaluates with CLI:
```bash
# Agent generates training code using spektran.io.read_records
# Agent writes predictions CSV
spektran benchmark --task T1-concentration \
  --truth data/ch4-t1-test-v0.h5 \
  --predictions my_model_predictions.csv
```

</details>

<br>

## Benchmark Tasks

| Task | Domain | Primary Metric |
|:-----|:-------|:---------------|
| **T1** Concentration regression | DA spectrum → ppm | MAE |
| **T2** Spectral denoising | Noisy → clean spectrum | RMSE |
| **T3** Cross-instrument generalization | Held-out instruments | Degradation vs T1 |
| **T4** WMS concentration | 2f signal → ppm | MAE |
| **T5** Drift compensation | Time-series scans | Allan variance |
| **T6** OOD instrument detection | In-dist vs OOD | AUROC |
| **T7** Cross-modality transfer | TDLAS → NDIR | Degradation vs T1 |
| **T8** Multi-species regression | CH4 + H2O → both ppm | Aggregate MAE |
| **T9** Temperature regression | Spectrum → gas temp (K) | MAE |

<br>

## Leaderboard

**T1 Concentration + T3 Cross-Instrument** (v0 splits, CH4 DA):

| Model | T1 MAE ↓ | T1 MAPE ↓ | T3 MAE ↓ | T3 Degradation |
|:------|:--------:|:---------:|:--------:|:--------------:|
| Ridge regression | **2.84** | 29.9% | **3.72** | **1.31x** |
| Patchified Transformer | 7.39 | **22.7%** | 10.81 | 1.46x |
| 1D CNN | 15.58 | 42.2% | 28.30 | 1.82x |

> Model complexity correlates with instrument overfitting: Ridge 1.31x → Transformer 1.46x → CNN 1.82x. Can you build a model that breaks this pattern?

<details>
<summary><b>Results for other tasks</b></summary>

| Task | Best Model | Score |
|:-----|:-----------|:------|
| **T2** Denoising | 1D U-Net | RMSE 3.62e-3 |
| **T4** WMS | Ridge | MAE 15.15 ppm |
| **T5** Drift | Moving average | MAE 0.270 ppm |
| **T6** OOD | PCA + Mahalanobis | AUROC 0.672 |
| **T7** Cross-modality | Ridge (TDLAS→NDIR) | MAE 130.68 ppm (46x) |
| **T8** Multi-species | Ridge (dual) | CH4 0.89 / H2O 3937 ppm |
| **T9** Temperature | Ridge | MAE 9.4 K |

</details>

[**Full leaderboard →**](https://spektran.github.io/spektran/leaderboard/) &nbsp;&middot;&nbsp;
[**Submit results →**](https://spektran.github.io/spektran/leaderboard/#submitting-results)

<br>

## Real-World Impact

SPEKTRAN's benchmarks mirror problems that matter in industry and environmental science:

- **Methane leak detection** — oil & gas facilities, landfills, livestock operations (T1, T3)
- **CO2 monitoring** — greenhouse gas quantification, indoor air quality, process control ([CO2 dataset](https://huggingface.co/datasets/spektran/spektran-co2-v0))
- **Industrial emissions monitoring** — SO2, NO, CO stack gas analysis ([industrial dataset](https://huggingface.co/datasets/spektran/spektran-industrial-v0))
- **Multi-gas mixtures** — disentangling overlapping species in combustion exhaust ([multi-gas dataset](https://huggingface.co/datasets/spektran/spektran-multigas-v0))
- **Medical breath analysis** — trace-gas biomarkers at ppb levels (T1, T9)
- **Instrument-agnostic deployment** — models that transfer across hardware without recalibration (T3, T7)
- **Drift-resilient field sensors** — long-term autonomous monitoring in harsh environments (T5)

<br>

## How It Works

```
 HITRAN line data        Instrument configs         Benchmark
 ───────────────        ──────────────────         ─────────
 Line positions    ──►  Virtual instruments   ──►  Official splits
 Line strengths         (noise, fringes,           (train/val/test/
 Broadening params       drift, chirp)              held-out)
        │                      │                       │
        ▼                      ▼                       ▼
   Forward physics  ──►  Noisy spectra   ──►   Evaluate & rank
   (Voigt / HTP)         with provenance       on leaderboard
```

<br>

## Physics You Can Trust

- **Dual-implementation validation** — independent reference implementations cross-checked against HITRAN/hapi
- **Literature-anchored noise** — instrument noise parameters surveyed from 18 published systems
- **Sim-to-real gap report** — known gap sources documented in [G5 report](gates/reports/)
- **Automated quality gates** — G1–G5 checkpoints enforced in CI

<br>

## Citing

```bibtex
@software{spektran,
  title     = {SPEKTRAN: Simulation Engine and ML Benchmark for Optical Gas Sensing},
  url       = {https://github.com/spektran/spektran},
  doi       = {10.5281/zenodo.21790394},
  version   = {0.5.1},
  license   = {Apache-2.0}
}
```

See [CITATION.cff](CITATION.cff) or use the DOI: [10.5281/zenodo.21790394](https://doi.org/10.5281/zenodo.21790394).

## Star History

<a href="https://star-history.com/#spektran/spektran&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=spektran/spektran&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=spektran/spektran&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=spektran/spektran&type=Date" width="600" />
 </picture>
</a>

<br>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome new baselines, modalities, and line lists.

<div align="center">
<sub>Code: Apache-2.0 &nbsp;·&nbsp; Data & Schema: CC BY 4.0</sub>
</div>
