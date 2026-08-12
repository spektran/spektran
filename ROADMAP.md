# SPEKTRAN Roadmap

This roadmap is public and technical. It describes what the project intends to build, not what has been promised. Timelines are estimates. Contributions that accelerate any milestone are welcome.

## Current state: v0.6.0 (2026-08)

Five modalities shipped — TDLAS, NDIR, CRDS, FTIR, DOAS:
- **TDLAS**: DA + WMS forward physics, 10 molecules, 14+ virtual instruments,
  TIPS polynomials, thermal chirp tuning, window/beam path effects,
  temperature-dependent detector noise
- **NDIR**: Planck source + bandpass filter forward model, 4 virtual
  instruments (easy/medium/hard/heldout), thermopile/pyroelectric noise
- **CRDS**: Ring-down time → absorption coefficient forward model, mirror
  degradation + mode-matching noise chain, 4 virtual instruments
- **FTIR**: Interferogram ↔ spectrum forward model with 5 apodization
  functions, channel spectra + self-apodization noise chain, 4 virtual instruments
- **DOAS**: UV/Vis Beer-Lambert + Rayleigh/Mie scattering + polynomial
  high-pass forward model, Ring effect + stray light noise chain, 4 virtual instruments
- 9 benchmark tasks (T1-T9) with 50+ dataset configs
- T7: cross-modality transfer; T8: multi-species regression; T9: temperature regression
- 25 baseline models (classical + Transformer, U-Net, TCN, SpektralNet, PINN)
- 46 virtual instruments across all modalities
- Static leaderboard on GitHub Pages (T1-TDLAS, T1-CRDS, T1-FTIR, T1-DOAS)
- Dual-implementation physics cross-validation (G3)
- Noise realism envelope checks against 18-paper survey (G4)
- Literature-anchored noise parameters for all modalities
- HDF5 persistence + Hugging Face Hub integration
- MkDocs documentation on GitHub Pages
- Schema extended for all 5 techniques (record + instrument schemas)
- CLI supports generation for all modalities

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

## Phase 4: Multi-modality (v0.4.0) — shipped 2026-08

### 4.1 NDIR (Non-Dispersive Infrared)
- [x] Planck source + bandpass filter forward model (Gaussian + tophat filters)
- [x] NDIR noise chain: source drift, intensity fluctuation, detector noise
- [x] NDIR generator with SeedSequence reproducibility
- [x] 4 virtual instruments (easy/medium/hard/heldout tiers)
- [x] 3 NDIR dataset configs (train/test/test-heldout)
- [x] Record + instrument schema extended for NDIR technique
- [x] CLI NDIR generation support

### 4.2 Cross-modality benchmark track
- [x] T7: train on TDLAS, test on NDIR (shared gas/concentration, different physics)
- [x] Cross-modality degradation metric (vs T1 baseline)
- [x] Dataset config and evaluation pipeline

---

## Phase 4.1: TDLAS Deep Dive (v0.5.0) — shipped 2026-08

### 4.1.1 Advanced line-shape physics
- [x] Hartmann-Tran Profile (HTP) implementation with speed-dependent
  broadening/shifting, Dicke narrowing, correlation parameter
- [x] Independent HTP reference implementation (quadrature-based) + G3 cross-validation
- [x] Isotopologue handling: per-line isotopologue ID, natural abundance lookup,
  filtering by isotopologue
- [x] Configurable line-wing cutoff (per-molecule defaults, e.g. 500 cm-1 for CO2)
- [x] Vectorized absorption coefficient (NumPy broadcasting, ~2.5x speedup)

### 4.1.2 Enhanced WMS chain
- [x] WMS 2f/1f calibration-free ratio (Rieker et al. 2009)
- [x] Etalon transmission wired into WMS time-domain chain (parasitic fringes
  demodulated alongside gas absorption)
- [x] `ratio_2f1f` signal type in record schema

### 4.1.3 Instrument electronics
- [x] Laser RIN (relative intensity noise): multiplicative noise from spontaneous
  emission coupling, specified in dBc/Hz
- [x] TIA bandwidth filter: transimpedance amplifier low-pass before ADC
- [x] Detector responsivity: wavelength-dependent InGaAs response with sigmoid cutoff
- [x] 6 new instrument schema fields (rin_dBc_Hz, rin_bandwidth_Hz,
  tia_bandwidth_Hz, responsivity_cutoff_cm1, peak_responsivity,
  responsivity_rolloff_cm1)

### 4.1.4 New benchmark tasks
- [x] T8: multi-species regression (CH4 + H2O overlapping absorption, both
  concentrations as targets). Ridge baseline: CH4 MAE 0.89 ppm, H2O MAE 3937 ppm
- [x] T9: temperature regression (fixed concentration, regress gas temperature
  from line-shape changes, 250-800 K). Ridge baseline: MAE 9.4 K

### 4.1.5 Validation
- [x] Sim-to-real validation against literature: HITRAN line strengths, Voigt
  line width vs pressure, temperature dependence, WMS 2f analytical prediction
- [x] G5 report documenting known sim-to-real gap sources

---

## Phase 5: Additional modalities (v0.6.0) — shipped 2026-08

### 5.1 CRDS (Cavity Ring-Down Spectroscopy)
- [x] Ring-down time fitting forward model (`physics/crds.py`)
- [x] 6-effect noise chain: shot noise, mirror drift, mode matching, detector noise, baseline loss, temperature sensitivity
- [x] 4 virtual instruments (lab/field/Picarro-class/held-out)
- [x] 4 dataset configs (train/val/test/heldout), Ridge baseline (MAE 36.5 ppm)

### 5.2 FTIR (Fourier Transform Infrared)
- [x] Interferogram ↔ spectrum forward model with 5 apodization functions (`physics/ftir.py`)
- [x] 6-effect noise chain: detector, source 1/f, phase error, channel spectra, sampling error, self-apodization
- [x] 4 virtual instruments (lab/field/TCCON-class/held-out)
- [x] 4 dataset configs (train/val/test/heldout), Ridge baseline (MAE 83.7 ppm)

### 5.3 DOAS (Differential Optical Absorption Spectroscopy)
- [x] UV/Vis Beer-Lambert + Rayleigh/Mie scattering + polynomial high-pass forward model (`physics/doas.py`)
- [x] 6-effect noise chain: photon noise, stray light, Ring effect, wavelength shift, dark current, readout noise
- [x] 4 virtual instruments (zenith-sky/MAX-DOAS/long-path/held-out)
- [x] 4 dataset configs (train/val/test/heldout), Ridge baseline (MAE 1.40 ppm)

### 5.4 PAS (Photoacoustic Spectroscopy) — future
- Acoustic resonator model
- Microphone noise chain
- Concentration regression benchmark

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
