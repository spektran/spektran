# SPEKTRAN

**Open-source simulation engines, data standards, and ML benchmarks for physical sensing — every modality where physically rigorous synthetic training data can be generated.**

SPEKTRAN builds one platform pattern — parameterized forward physics + literature-anchored instrument-noise chains + reproducible benchmark splits — and applies it modality by modality. The first shipped domain is **laser gas absorption spectroscopy (TDLAS)**: fully reproducible synthetic spectra with realistic instrument noise (laser scan nonlinearity, etalon fringes, 1/f noise, baseline drift) and standardized tasks for concentration regression, spectral denoising, and cross-instrument generalization. NDIR, photoacoustic, and cavity-ringdown spectroscopy are the next planned techniques; the record schema carries a `technique` field from day one so new modalities extend, not fork, the standard.

- **Code**: Apache-2.0 ([LICENSE](LICENSE))
- **Data & schema**: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA))

> ⚠️ **Status: alpha.** Gates G1–G5 (naming, schema completeness, physics correctness, noise realism, cold-start usability) all pass with archived adversarial reviews; APIs and schema v0.1 may still change until v1.0.

## Why SPEKTRAN?

Machine learning for physical sensing lacks what computer vision has had for a decade: standard datasets, standard tasks, and comparable baselines. Every paper simulates (or measures) its own signals, with its own noise assumptions, and reports metrics nobody else can reproduce.

SPEKTRAN attacks this with three assets:

1. **A parameterized simulation engine** — HITRAN-based forward physics (direct absorption and wavelength-modulation spectroscopy) plus a modular instrument-noise chain modeled after real hardware.
2. **A data standard** — a JSON Schema for spectra records with explicit units, full provenance (generator version, random seed, every sampled noise parameter), and a `technique` field ready for NDIR / PAS / CRDS extensions.
3. **A tiered benchmark** — official train/val/test splits, three difficulty levels, and a flagship *cross-instrument generalization* track built on held-out virtual instruments.

All shipped data is simulation-born and labeled `data_origin: simulated`. The sim-to-real gap is not hidden — it is the research topic of the generalization track.

## Quick start

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

## Benchmark tasks

| Task | Input | Output | Primary metric |
|---|---|---|---|
| **T1 Concentration regression** | Noisy raw scan (DA) | CH₄ concentration (ppm) | MAE, MAPE |
| **T2 Denoising / baseline correction** | Raw spectrum with fringes & drift | Clean absorbance spectrum | RMSE, peak-weighted RMSE |
| **T3 Cross-instrument generalization** | Same as T1, held-out instruments | Concentration (ppm) | Generalization MAE, degradation vs T1 |

### Leaderboard (v0 splits, CH4 DA)

| Model | T1 MAE (ppm) | T1 MAPE (%) | T3 MAE (ppm) | T3 degradation |
|---|---|---|---|---|
| Ridge regression (baseline) | 2.84 | 29.9 | 3.72 | 1.31x |
| 1D CNN (baseline) | 15.58 | 42.2 | 28.30 | 1.82x |

T2 denoising (same test split): wing-anchored cubic-polynomial baseline
(classical reference) — spectral RMSE 6.31e-3, peak-weighted RMSE 8.60e-3.

Reproduce with [`baselines/README.md`](baselines/README.md). Note the T3
lesson already visible in the baselines: the deep model overfits instrument
signatures harder than the linear one. Submissions: run
`python -m spektran.benchmark.evaluate` on your predictions and open a PR
adding your row with a link to reproducible code.

## Project quality gates

Development is gated by automated, quantitative checkpoints (G1 naming … G5 cold-start usability). Every gate report is version-controlled under [`gates/reports/`](gates/reports/) as public evidence. Physics correctness is enforced by dual-implementation cross-validation (independent reference implementations in `tests/reference_impl/`) and CI tests against HITRAN/hapi references.

## Citing

See [CITATION.cff](CITATION.cff). A Zenodo DOI will be minted at v1.0.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Data contributions require the Data Submission Agreement (CC BY 4.0 grant).
