# SPEKTRAN Roadmap

This roadmap is public and technical. It describes what the project intends to build, not what has been promised. Timelines are estimates. Contributions that accelerate any milestone are welcome.

## Current state: v0.2.0-alpha (2026-08)

The TDLAS domain is fully functional:
- Direct absorption (DA) and wavelength modulation (WMS) forward physics
- 8 literature-anchored virtual instruments spanning easy → hard + held-out tiers
- 5 official benchmark splits (train/val/test/T3-heldout + demo)
- 3 baseline models (ridge regression, 1D CNN, wing-anchored polynomial)
- Dual-implementation physics cross-validation (G3)
- Noise realism envelope checks against 18-paper survey (G4)
- HDF5 persistence + Hugging Face Hub integration
- MkDocs documentation on GitHub Pages
- TIPS partition-function polynomial for temperature scaling (CH4, H2O, CO2, CO)
- Multi-species forward model with Beer-Lambert superposition
- Schema v0.2 (higher harmonics, measurement block, backward compatible with v0.1)
- 3f/4f WMS demodulation in generator
- 9 virtual instruments (added 4-harmonic WMS config)
- 6 benchmark tasks (T1-T6) with 8 dataset configs
- Expanded CLI: `spektran generate`, `spektran benchmark`, `spektran download`

---

## Phase 3: Depth & polish (v0.2.0) — shipped 2026-08

### 3.1 HITRAN production run
- [x] Add H2O, CO2, CO target molecules (demo line lists)
- [x] Implement TIPS polynomial for accurate Q(T) ratio (replace power-law approximation)
- [ ] Replace offline demo lines with full HITRAN fetch for all official splits *(deferred to Phase 3.1)*

### 3.2 Schema v0.2
- [x] Multi-species records (interferent + target in same cell)
- [x] Higher harmonics (demod_3f, demod_4f) signal slots
- [x] Measurement block for experimental data fields
- [x] Backward compatibility with v0.1 records
- [ ] Versioned schema migration tool *(deferred to Phase 3.1)*

### 3.3 Enhanced WMS
- [x] Higher harmonics (3f, 4f) in demod chain and generator
- [x] WMS benchmark task (T4: 2f peak-height ratio → concentration)
- [ ] Nonlinear intensity modulation (laser diode current-tuning model) *(deferred to Phase 4)*

### 3.4 Benchmark expansion
- [x] T4: WMS 2f concentration regression (MAE)
- [x] T5: Time-series drift compensation (Allan variance improvement) — metrics and task spec defined; evaluation stub
- [x] T6: Anomaly detection / OOD instrument identification (AUROC) — metrics and task spec defined; evaluation stub
- [ ] Community leaderboard *(deferred to Phase 5)*

### 3.5 Quality-of-life
- [x] `spektran generate` CLI with timing output
- [x] `spektran benchmark` one-command evaluation
- [x] `spektran download` instructions for pre-built datasets
- [ ] Multi-format output (Parquet, CSV) *(deferred to Phase 3.1)*
- [ ] Docker image for reproducible environments *(deferred to Phase 3.1)*

---

## Phase 3.1: Polish & production (v0.2.x)

### 3.1.1 HITRAN production splits
- Full HITRAN fetch for all 8 official dataset configs
- Isotopologue filtering for multi-species
- CI-pinned HITRAN data snapshots for reproducibility

### 3.1.2 Evaluation pipeline completion
- T5 time-series evaluation (requires time-series HDF5 layout)
- T6 OOD evaluation pipeline (requires OOD label format)
- Full baselines for T4, T5, T6

### 3.1.3 Developer experience
- Parquet and CSV output formats in `spektran generate`
- tqdm progress bar for generation
- Docker image for reproducible environments
- Versioned schema migration tool

---

## Phase 4: Multi-modality (v0.3.0+)

### 4.1 NDIR (Non-Dispersive Infrared)
- Broadband source + bandpass filter forward model
- Detector noise model (thermopile, pyroelectric)
- 4–6 virtual instruments, benchmark splits

### 4.2 PAS (Photoacoustic Spectroscopy)
- Acoustic resonator model
- Microphone noise chain
- Concentration regression benchmark

### 4.3 CRDS (Cavity Ring-Down Spectroscopy)
- Ring-down time fitting forward model
- Mirror reflectivity degradation noise
- Benchmark: ring-down time → concentration

### 4.4 Cross-modality benchmark track
- Train on one modality, test on another (shared gas/concentration, different physics)
- Task IDs to be assigned when multi-modality data is available
- Requires the `technique` field already present in the schema

---

## Phase 5: Community & ecosystem (v1.0)

### 5.1 Experimental data ingestion
- Curated pipeline: schema validation + physics-plausibility check + human review
- `data_origin: measured` support with uncertainty fields
- Mixed sim+real training datasets

### 5.2 PyPI stable release
- Semantic versioning from v1.0
- Schema frozen (non-breaking changes only within major version)
- Zenodo DOI for citation

### 5.3 Agent & API
- REST API for on-demand spectrum generation
- Python client library for programmatic access
- Integration with popular ML frameworks (PyTorch DataLoader, HF datasets streaming)

### 5.4 Contributed modalities
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
