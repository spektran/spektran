# SPEKTRAN

**Open-source simulation engines, data standards, and ML benchmarks for physical sensing — every modality where physically rigorous synthetic training data can be generated.**

SPEKTRAN builds one platform pattern — parameterized forward physics + literature-anchored instrument-noise chains + reproducible benchmark splits — and applies it modality by modality. The first shipped domain is **laser gas absorption spectroscopy (TDLAS)**: fully reproducible synthetic spectra with realistic instrument noise (laser scan nonlinearity, etalon fringes, 1/f noise, baseline drift) and standardized tasks for concentration regression, spectral denoising, and cross-instrument generalization. NDIR, photoacoustic, and cavity-ringdown spectroscopy are the next planned techniques; the record schema carries a `technique` field from day one so new modalities extend, not fork, the standard.

- **Code**: Apache-2.0 ([LICENSE](LICENSE))
- **Data & schema**: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA))

> ⚠️ **Status: alpha (v0.2.1).** Gates G1–G5 all pass with archived adversarial reviews. Multi-species support (CH4, H2O, CO2, CO), higher-harmonic WMS (3f/4f), and 6 benchmark tasks — all with complete pipelines and reference baselines — now available. APIs and schema may still change until v1.0.

## Why SPEKTRAN?

Machine learning for physical sensing lacks what computer vision has had for a decade: standard datasets, standard tasks, and comparable baselines. Every paper simulates (or measures) its own signals, with its own noise assumptions, and reports metrics nobody else can reproduce.

SPEKTRAN attacks this with three assets:

1. **A parameterized simulation engine** — HITRAN-based forward physics (direct absorption and wavelength-modulation spectroscopy) plus a modular instrument-noise chain modeled after real hardware.
2. **A data standard** — a JSON Schema for spectra records with explicit units, full provenance (generator version, random seed, every sampled noise parameter), and a `technique` field ready for NDIR / PAS / CRDS extensions.
3. **A tiered benchmark** — official train/val/test splits, three difficulty levels, six tasks (concentration regression, denoising, cross-instrument generalization, WMS concentration, drift compensation, OOD detection), and a flagship *cross-instrument generalization* track built on held-out virtual instruments.

All shipped data is simulation-born and labeled `data_origin: simulated`. The sim-to-real gap is not hidden — it is the research topic of the generalization track.

## Quick start

Zero-install data access (Hugging Face):

```python
from datasets import load_dataset
ds = load_dataset("spektran/spektran-ch4-v0")  # train/validation/test/test_heldout_instrument
```

Full engine + benchmark tooling:

```bash
git clone https://github.com/spektran/spektran
cd spektran
pip install -e ".[dev]"
```

```python
from spektran.physics import simulate_absorbance

# Clean CH4 direct-absorption spectrum near 6046.9 cm-1 (1653 nm, 2v3 band)
nu, absorbance = simulate_absorbance(
    molecule="CH4",
    concentration_ppm=100.0,
    temperature_K=296.0,
    pressure_atm=1.0,
    path_length_m=10.0,
    wavenumber_start_cm1=6046.0,
    wavenumber_end_cm1=6048.0,
)
```

```python
# Multi-species: CH4 with H2O interferent
from spektran.physics import demo_ch4_2nu3, demo_h2o, absorption_coefficient
import numpy as np

nu = np.linspace(6046.0, 6048.0, 2000)
alpha_ch4 = absorption_coefficient(nu, demo_ch4_2nu3(), 100e-6, 296.0, 1.0)
alpha_h2o = absorption_coefficient(nu, demo_h2o(), 0.01, 296.0, 1.0)  # 1% H2O
```

## Benchmark tasks

| Task | Input | Output | Primary metric |
|---|---|---|---|
| **T1 Concentration regression** | Noisy raw scan (DA) | CH₄ concentration (ppm) | MAE, MAPE |
| **T2 Denoising / baseline correction** | Raw spectrum with fringes & drift | Clean absorbance spectrum | RMSE, peak-weighted RMSE |
| **T3 Cross-instrument generalization** | Same as T1, held-out instruments | Concentration (ppm) | Generalization MAE, degradation vs T1 |
| **T4 WMS concentration** | Noisy 2f signal (WMS) | CH₄ concentration (ppm) | MAE |
| **T5 Drift compensation** | Time-series raw scans | Drift-corrected concentrations | Allan variance improvement |
| **T6 OOD instrument detection** | Raw scan | In-dist vs OOD binary | AUROC |

### Leaderboard (v0 splits, CH4 DA)

| Model | T1 MAE (ppm) | T1 MAPE (%) | T3 MAE (ppm) | T3 degradation |
|---|---|---|---|---|
| Ridge regression (baseline) | 2.84 | 29.9 | 3.72 | 1.31x |
| 1D CNN (baseline) | 15.58 | 42.2 | 28.30 | 1.82x |

T2 denoising (same test split): wing-anchored cubic-polynomial baseline
(classical reference) — spectral RMSE 6.31e-3, peak-weighted RMSE 8.60e-3.

T4 WMS concentration (2f demodulated signal, held-out test split): ridge
regression 15.15 ppm MAE, 1D CNN 20.35 ppm MAE — the linear model wins again,
same pattern as T1/T3.

T5 drift compensation (10 held-out time series, 200 scans each): moving-average
baseline 0.270 ppm MAE.

T6 OOD instrument detection (in-distribution vs. held-out instrument): PCA +
Mahalanobis-distance baseline 0.672 AUROC.

Reproduce with [`baselines/README.md`](baselines/README.md). Note the T3
lesson already visible in the baselines: the deep model overfits instrument
signatures harder than the linear one. Submissions: run
`python -m spektran.benchmark.evaluate` on your predictions and open a PR
adding your row with a link to reproducible code.

## CLI

```bash
spektran generate configs/datasets/ch4-t1-train-v0.yaml --out data
spektran benchmark --task T1-concentration --truth data/test.h5 --predictions preds.csv
spektran validate data/ch4-t1-train-v0.h5
spektran download
```

## Project quality gates

Development is gated by automated, quantitative checkpoints (G1 naming … G5 cold-start usability). Every gate report is version-controlled under [`gates/reports/`](gates/reports/) as public evidence. Physics correctness is enforced by dual-implementation cross-validation (independent reference implementations in `tests/reference_impl/`) and CI tests against HITRAN/hapi references.

## Citing

See [CITATION.cff](CITATION.cff). A Zenodo DOI will be minted at v1.0.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Data contributions require the Data Submission Agreement (CC BY 4.0 grant).
