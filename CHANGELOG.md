# Changelog

All notable changes to SPEKTRAN are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [0.5.1] — 2026-08-11 — AI Agent Ready

### Added
- **AI Agent-ready CLI** — 8 commands (`info`, `list`, `status`, `train`, `generate`, `validate`, `benchmark`, `download`), all with `--json` structured output for AI agent integration
- **AGENTS.md** — public AI agent interface file; any coding agent reads this to operate the full pipeline
- **`spektran train`** — one-command baseline training with auto data generation and score reporting
- **Baseline registry** (`baselines/registry.yaml`) — declarative manifest of all 14 baselines
- **AI Agent Guide** — dedicated documentation page (`docs/agent.md`)

### Changed
- All project branding updated to "AI Agent-Ready" positioning across README, README_zh, docs site, pyproject.toml, CITATION.cff, .zenodo.json, Colab notebook, HF dataset card, and HF Space
- Quickstart docs now lead with "Option 0: AI Agent (zero code)"
- GitHub repo description and topics updated (`agent-ready`, `ai-agent`)

---

## [0.5.0] — 2026-08-05 — TDLAS Deep Dive

### Added
- **Hartmann-Tran Profile (HTP)** — beyond-Voigt line shape via hapi's `pcqsdhc`, with speed-dependent broadening, Dicke narrowing, and correlation
- **WMS 2f/1f calibration-free ratio** (Rieker et al. 2009) — auto-computed when both harmonics are demodulated
- **Etalon fringes in WMS chain** — physically correct parasitic fringe demodulation
- **Laser RIN** (relative intensity noise), **TIA bandwidth**, and **detector responsivity** modeling
- **Isotopologue handling** — per-line isotopologue ID, natural abundance lookup, configurable line-wing cutoff
- **T8: Multi-species regression** — CH4 + H2O overlapping absorption, both concentrations as targets (ridge baseline: CH4 0.89 ppm, H2O 3937 ppm)
- **T9: Temperature regression** — gas temperature from line-shape changes, 250–800 K (ridge baseline: 9.4 K MAE)
- **Sim-to-real validation** against published literature (G5 report)
- **Vectorized absorption coefficient** — NumPy broadcasting, ~2.5x speedup

### Changed
- **hapi is now a core dependency** (was optional) — TIPS uses PYTIPS2021, no more hand-fit polynomials
- TIPS values differ slightly from v0.4.0; regenerate datasets for exact reproducibility

---

## [0.4.0] — 2026-08-05 — Multi-Modality

### Added
- **NDIR modality** — Planck source + bandpass filter forward model, 4 virtual instruments
- **T7: Cross-modality transfer** — train on TDLAS, test on NDIR (same gas, different physics)
- NDIR noise chain (source drift, intensity fluctuation, detector noise)
- 3 NDIR dataset configs, CLI support for NDIR generation

---

## [0.3.1] — 2026-08-05 — ML Baselines & Leaderboard

### Added
- **Patchified Transformer** baseline (T1: 7.39 ppm MAE, T4: 17.83 ppm MAE)
- **1D U-Net** baseline for T2 spectral denoising
- **TCN** baseline for T5 drift compensation
- **Static leaderboard** on GitHub Pages

---

## [0.3.0] — 2026-08-05 — Deeper TDLAS Physics

### Added
- **6 new molecules**: NH3, NO, NO2, SO2, HCl, HF (total: 10)
- **Nonlinear laser current-tuning** model (DFB/VCSEL thermal chirp)
- **Window contamination** and **beam wander** optical path effects
- **Temperature-dependent detector noise** (Johnson-Nyquist + Arrhenius dark current)
- 3 new specialized virtual instrument configs

---

## [0.2.1] — 2026-08-05 — Complete Benchmark Suite

### Added
- **T4 WMS baselines**: ridge 15.15 ppm MAE, 1D CNN 20.35 ppm MAE
- **T5 drift compensation** full pipeline + moving-average baseline (0.270 ppm MAE)
- **T6 OOD instrument detection** full pipeline + PCA+Mahalanobis baseline (0.672 AUROC)
- HITRAN production data fetching (7 opt-in `-hitran` dataset configs)
- Large-scale dataset configs (50K/5K/10K splits)

---

## [0.2.0] — 2026-08-04 — Depth & Polish

### Added
- **H2O, CO2, CO** target molecules (total: 4)
- **TIPS partition-function polynomial** for accurate Q(T) ratio
- **Multi-species records** (interferent + target in same cell)
- **WMS 3f/4f** higher harmonics in demod chain
- **T4 WMS concentration** benchmark task
- **T5 drift compensation** and **T6 OOD detection** tasks
- `spektran generate`, `spektran benchmark`, `spektran validate` CLI commands

---

## [0.1.0] — 2026-08-04 — First Public Release

### Added
- CH4 direct-absorption forward model (Voigt profile, HITRAN line-by-line)
- Instrument noise chain (scan nonlinearity, RAM, etalon, 1/f, baseline drift, ADC quantization)
- WMS 1f/2f lock-in demodulation
- T1 concentration regression, T2 denoising, T3 cross-instrument generalization
- Ridge regression and 1D CNN baselines
- JSON Schema for spectra records (v0.1)
- HDF5 persistence + Hugging Face Hub integration
- MkDocs documentation, CI/CD pipeline, PyPI publishing

[0.5.1]: https://github.com/spektran/spektran/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/spektran/spektran/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/spektran/spektran/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/spektran/spektran/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/spektran/spektran/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/spektran/spektran/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/spektran/spektran/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/spektran/spektran/releases/tag/v0.1.0
