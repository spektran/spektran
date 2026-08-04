# SPEKTRAN Roadmap

This roadmap is public and technical. It describes what the project intends to build, not what has been promised. Timelines are estimates. Contributions that accelerate any milestone are welcome.

## Current state: v0.3.1 (2026-08)

The TDLAS domain has deep physics coverage and ML baselines:
- Direct absorption (DA) and wavelength modulation (WMS) forward physics
- 10 target molecules: CH4, H2O, CO2, CO, NH3, NO, NO2, SO2, HCl, HF
- TIPS partition-function polynomial for all 10 molecules
- Nonlinear laser current-tuning model (DFB/VCSEL thermal chirp)
- Window contamination (wavelength-dependent scattering) and beam wander
- Temperature-dependent detector noise (Johnson-Nyquist + dark current)
- 12 virtual instruments spanning easy → hard + held-out + specialized tiers
- 6 benchmark tasks (T1-T6) with 22 dataset configs
- 10 baseline models (classical + deep learning: Transformer, U-Net, TCN)
- Static leaderboard on GitHub Pages
- Dual-implementation physics cross-validation (G3)
- Noise realism envelope checks against 18-paper survey (G4)
- HDF5 persistence + Hugging Face Hub integration
- MkDocs documentation on GitHub Pages
- Schema v0.2 (higher harmonics, measurement block, backward compatible with v0.1)
- Expanded CLI: `spektran generate`, `spektran benchmark`, `spektran download`

---

## Phase 3: Depth & polish (v0.2.0) — shipped 2026-08

### 3.1 HITRAN production run
- [x] Add H2O, CO2, CO target molecules (demo line lists)
- [x] Implement TIPS polynomial for accurate Q(T) ratio (replace power-law approximation)
- [x] HITRAN production fetch for T1/T3/T4 official splits (7 opt-in `-hitran`
  dataset configs via `scripts/fetch_hitran.py`); T5/T6 and default splits remain
  on offline demo lines *(full replacement deferred to Phase 4)*

### 3.2 Schema v0.2
- [x] Multi-species records (interferent + target in same cell)
- [x] Higher harmonics (demod_3f, demod_4f) signal slots
- [x] Measurement block for experimental data fields
- [x] Backward compatibility with v0.1 records
- [ ] Versioned schema migration tool *(deferred to Phase 4)*

### 3.3 Enhanced WMS
- [x] Higher harmonics (3f, 4f) in demod chain and generator
- [x] WMS benchmark task (T4: 2f peak-height ratio → concentration)
- [ ] Nonlinear intensity modulation (laser diode current-tuning model) *(deferred to Phase 4)*

### 3.4 Benchmark expansion
- [x] T4: WMS 2f concentration regression (MAE) — full pipeline shipped (dataset
  configs, evaluation, ridge + CNN baselines)
- [x] T5: Time-series drift compensation (Allan variance improvement) — full pipeline shipped (dataset configs, evaluation, moving-average baseline)
- [x] T6: Anomaly detection / OOD instrument identification (AUROC) — full pipeline shipped (dataset configs, ood_task generation, evaluation, PCA+Mahalanobis baseline)
- [ ] Community leaderboard *(deferred to Phase 6)*

### 3.5 Quality-of-life
- [x] `spektran generate` CLI with timing output
- [x] `spektran benchmark` one-command evaluation
- [x] `spektran download` instructions for pre-built datasets
- [ ] Multi-format output (Parquet, CSV) *(deferred to Phase 4)*
- [ ] Docker image for reproducible environments *(deferred to Phase 4)*

---

## Phase 3.1: Complete benchmark suite (v0.2.1) — shipped 2026-08

### 3.1.1 HITRAN production data
- [x] HITRAN production fetch for T1/T3/T4 official splits (7 opt-in `-hitran`
  dataset configs, `scripts/fetch_hitran.py`)
- [x] Large-scale dataset configs for scaling experiments (50K train / 5K val /
  10K test)
- [ ] HITRAN fetch extended to T5/T6 official splits *(deferred to Phase 4)*
- [ ] Isotopologue filtering for multi-species *(deferred to Phase 4)*
- [ ] CI-pinned HITRAN data snapshots for reproducibility *(deferred to Phase 4)*

### 3.1.2 Benchmark baselines
- [x] T4 WMS baselines: ridge regression + 1D CNN on demod_2f (ridge 15.15 ppm
  MAE, CNN 20.35 ppm MAE)
- [x] T5 drift compensation: time-series IO, evaluation, moving-average baseline
  (0.270 ppm MAE)
- [x] T6 OOD instrument detection: evaluation, Mahalanobis baseline (0.672 AUROC)

### 3.1.3 Developer experience *(deferred to Phase 4)*
- [ ] Parquet and CSV output formats in `spektran generate`
- [ ] tqdm progress bar for generation
- [ ] Docker image for reproducible environments
- [ ] Versioned schema migration tool

All six benchmark tasks (T1-T6) now have complete pipelines and at least one
reference baseline. This closes out Phase 3.1 and the v0.2.x line.

---

## Phase 3.2: Deeper TDLAS Physics (v0.3.0) — shipped 2026-08

### 3.2.1 New target molecules
- [x] 6 new industrial gases: NH3, NO, NO2, SO2, HCl, HF (demo line lists +
  TIPS polynomials + reference implementation cross-validation)
- [x] All 10 molecules available as target or interferent in CLI
- [ ] N2O, C2H2 line lists *(deferred to Phase 5)*
- [ ] Virtual instruments and dataset configs for new species *(deferred to Phase 5)*

### 3.2.2 Enhanced forward physics
- [x] Nonlinear laser current-tuning model (DFB/VCSEL thermal chirp with
  analytic first-order lag solution)
- [x] Window contamination (broadband + wavelength-dependent Rayleigh-like
  scattering from fouled cell windows)
- [x] Beam wander (low-frequency intensity modulation from mechanical vibration)
- [x] Temperature-dependent detector noise (Johnson-Nyquist thermal scaling +
  Arrhenius dark-current shot noise)
- [x] 3 new specialized instrument configs (vi-da-thermal-10, vi-da-contaminated-11,
  vi-da-thermal-noise-12)
- [ ] HITRAN production fetch extended to T5/T6 official splits *(deferred)*
- [ ] Isotopologue filtering for multi-species HITRAN fetches *(deferred)*

### 3.2.3 Developer experience *(deferred to Phase 5)*
- [ ] Parquet and CSV output formats in `spektran generate`
- [ ] tqdm progress bar for generation
- [ ] Docker image for reproducible environments
- [ ] Versioned schema migration tool

---

## Phase 3.3: ML Baselines & Leaderboard (v0.3.1) — shipped 2026-08

### 3.3.1 Modern ML baselines
- [x] Patchified Transformer baseline for T1/T4 concentration regression
- [x] 1D U-Net baseline for T2 spectral denoising
- [x] TCN (Temporal Convolutional Network) baseline for T5 drift compensation

### 3.3.2 Community
- [x] Static leaderboard page (GitHub Pages)

---

## Phase 4: Multi-modality (v0.4.0)

### 4.1 NDIR (Non-Dispersive Infrared)
- Broadband source + bandpass filter forward model
- Detector noise model (thermopile, pyroelectric)
- 4–6 virtual instruments, benchmark splits

### 4.2 Cross-modality benchmark track
- Train on one modality, test on another (shared gas/concentration, different physics)
- Task IDs to be assigned when multi-modality data is available
- Requires the `technique` field already present in the schema

---

## Phase 5: Additional modalities (v0.5.0+)

### 5.1 PAS (Photoacoustic Spectroscopy)
- Acoustic resonator model
- Microphone noise chain
- Concentration regression benchmark

### 5.2 CRDS (Cavity Ring-Down Spectroscopy)
- Ring-down time fitting forward model
- Mirror reflectivity degradation noise
- Benchmark: ring-down time → concentration

---

## Phase 6: Community & ecosystem (v1.0)

### 6.1 Experimental data ingestion
- Curated pipeline: schema validation + physics-plausibility check + human review
- `data_origin: measured` support with uncertainty fields
- Mixed sim+real training datasets

### 6.2 PyPI stable release
- Semantic versioning from v1.0
- Schema frozen (non-breaking changes only within major version)
- Zenodo DOI for citation

### 6.3 Agent & API
- REST API for on-demand spectrum generation
- Python client library for programmatic access
- Integration with popular ML frameworks (PyTorch DataLoader, HF datasets streaming)

### 6.4 Contributed modalities
- Standardized template for adding new sensing domains
- Modality-specific gate scripts (G3/G4 equivalents)
- Community maintainers per modality

---

## Non-goals (out of scope)

- Real-time instrument control or data acquisition
- Replacing HITRAN or other spectroscopic databases
- Providing measurement-grade calibration (SPEKTRAN is for ML training data)
- Supporting non-optical sensing (acoustic, electrochemical, etc. — unless physics simulation is feasible)

---

## How to contribute to the roadmap

Open an issue or discussion on GitHub with the `roadmap` label. Proposals that include a physics reference (DOI) and a sketch of the forward model are fast-tracked for review.
