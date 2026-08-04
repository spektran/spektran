# SPEKTRAN Roadmap

This roadmap is public and technical. It describes what the project intends to build, not what has been promised. Timelines are estimates. Contributions that accelerate any milestone are welcome.

## Current state: v0.1.0-alpha (2026-08)

The TDLAS domain is fully functional:
- Direct absorption (DA) and wavelength modulation (WMS) forward physics
- 8 literature-anchored virtual instruments spanning easy → hard + held-out tiers
- 5 official benchmark splits (train/val/test/T3-heldout + demo)
- 3 baseline models (ridge regression, 1D CNN, wing-anchored polynomial)
- Dual-implementation physics cross-validation (G3)
- Noise realism envelope checks against 18-paper survey (G4)
- HDF5 persistence + Hugging Face Hub integration
- MkDocs documentation on GitHub Pages

---

## Phase 3: Depth & polish (v0.2.0)

### 3.1 HITRAN production run
- Replace offline 3-line CH4 demo with full HITRAN fetch for all official splits
- Add H2O, CO2, CO target molecules
- Implement TIPS polynomial for accurate Q(T) ratio (replace power-law approximation)

### 3.2 Schema v0.2
- Multi-species records (interferent + target in same cell)
- Experimental-data fields (`data_origin: measured`, instrument metadata, uncertainty)
- Versioned schema migration tool

### 3.3 Enhanced WMS
- Higher harmonics (3f, 4f) in demod chain
- Nonlinear intensity modulation (laser diode current-tuning model)
- WMS benchmark tasks (T4: 2f peak-height ratio → concentration)

### 3.4 Benchmark expansion
- Time-series tasks (Allan variance prediction, drift compensation)
- Anomaly detection task (out-of-distribution instrument identification)
- Community leaderboard (static page or lightweight API)

### 3.5 Quality-of-life
- `spektran generate` CLI with progress bar and multi-format output (HDF5, Parquet, CSV)
- `spektran benchmark run` one-command train-and-score for baselines
- Pre-built dataset downloads via `spektran download`
- Docker image for reproducible environments

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
- T5: Train on one modality, test on another (shared gas/concentration, different physics)
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
