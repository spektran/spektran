# CLAUDE.md — AI Agent Handoff Guide for SPEKTRAN

This file gives an AI coding agent (Claude Code, Cursor, Copilot, etc.) everything it needs to work on this project without prior context.

## What is SPEKTRAN?

An open-source platform that generates physically rigorous synthetic ML training data for sensing modalities. The first (and currently only) shipped domain is **TDLAS** (Tunable Diode Laser Absorption Spectroscopy). The platform pattern — parameterized forward physics + literature-anchored instrument-noise chains + reproducible benchmark splits — is designed to extend modality by modality (NDIR, PAS, CRDS next).

- **Name origin**: SPEK(tral) + (HI)TRAN. The `-TRAN` suffix nods to HITRAN, the spectroscopic database the engine sources line data from.
- **License**: Apache-2.0 (code), CC BY 4.0 (data/schema).

## Quick reference

| What | Where |
|---|---|
| Package name | `spektran` |
| Python versions | 3.10, 3.11, 3.12 |
| Source code | `src/spektran/` |
| Tests | `tests/` (66 tests) |
| JSON Schemas | `schema/record.schema.json`, `schema/instrument.schema.json` |
| Virtual instruments | `configs/instruments/vi-*.yaml` (8 configs) |
| Dataset split configs | `configs/datasets/ch4-*.yaml` (5 configs) |
| ML baselines | `baselines/` (ridge, CNN, wing-poly) |
| Documentation site | `docs/` → MkDocs Material → GitHub Pages |
| Gate reports | `gates/reports/` |
| Gate scripts | `gates/g2_*.py`, `gates/g3_*.py`, `gates/g4_*.py` |

## Build & test commands

```bash
# Setup (from repo root)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (exclude network-dependent HITRAN tests)
pytest -m "not hitran_online"

# Run all tests including HITRAN online reference
pytest

# Lint
ruff check src tests

# Validate schemas against configs
python -m spektran.validate configs/

# Generate a dataset
python scripts/generate_dataset.py configs/datasets/ch4-t1-train-v0.yaml --out data

# Build documentation locally
pip install mkdocs-material
mkdocs serve   # → http://127.0.0.1:8000

# Push dataset to Hugging Face
python scripts/push_to_hf.py data/ch4-t1-train-v0.h5 --repo Deepnight/spektran-ch4-v0
```

## Architecture

### Signal generation chain (the core of the engine)

```
HITRAN line data
  → Voigt line profile (Faddeeva wofz)
    → Beer-Lambert absorbance
      → [WMS branch: scan modulation → RAM → lock-in demodulation]
        → Instrument noise chain:
           scan nonlinearity → etalon fringes → baseline drift
           → linewidth convolution → white + 1/f noise
           → gain nonlinearity → ADC quantization
            → Final noisy signal + ground-truth labels
```

### Module map

```
src/spektran/
├── physics/           # Forward physics (HITRAN → absorbance → WMS)
│   ├── lineshape.py   # Voigt/Gaussian/Lorentzian profiles (area-normalized, cm units)
│   ├── absorption.py  # Beer-Lambert forward model, line-strength temperature scaling
│   ├── wms.py         # WMS: scan modulation, RAM, lock-in demodulation (1f/2f)
│   ├── hitran.py      # HITRAN line list fetching (hapi wrapper) + offline demo lines
│   └── constants.py   # Physical constants (CODATA 2018)
├── instrument/        # Noise/artifact modules
│   ├── laser.py       # Scan axis, center drift, intensity ramp, linewidth convolution
│   ├── etalon.py      # Single/multi etalon fringes with phase drift
│   ├── detector.py    # White noise, 1/f noise (Timmer-König), ADC, gain nonlinearity
│   ├── optics.py      # Baseline polynomial, transmittance decay, intensity fluctuation
│   ├── environment.py # Temperature/pressure jitter
│   └── sampling.py    # dist_or_number sampling, YAML config loading, schema validation
├── benchmark/         # ML benchmark tasks and evaluation
│   ├── tasks.py       # T1/T2/T3 task specs, split seeds, tier instruments
│   ├── metrics.py     # MAE, MAPE, RMSE, spectral RMSE, peak-weighted RMSE, degradation
│   └── evaluate.py    # CLI: --task + --truth + --predictions → JSON scores
├── generator.py       # Main generation: generate_record, generate_dataset, generate_time_series
├── io.py              # HDF5 persistence (write_records, read_records, read_meta_index)
├── validate.py        # JSON Schema validation (Draft 2020-12), CLI
└── cli.py             # CLI entry point (spektran validate)
```

### Key design decisions

1. **Reproducibility via SeedSequence.spawn**: Every record gets its own child seed spawned from the master seed. This means any single record can be regenerated without generating the full dataset. The seed index IS the record index.

2. **Schema-first**: Both `record.schema.json` and `instrument.schema.json` are the source of truth. Every generated record is validated against the schema on write. The schema uses JSON Schema draft-2020-12 with conditional requirements (e.g., WMS records must have modulation+demod fields).

3. **Dual-implementation testing**: Physics correctness is verified by having two completely independent implementations (in `tests/reference_impl/`) that use different algorithms and their own transcribed constants. No shared code between main and reference.

4. **Literature-anchored noise**: The 8 virtual instruments span noise levels anchored to an 18-paper survey (`configs/instruments/literature_anchors.yaml`). Gate G4 checks that generated signals fall within the documented envelope (NEA, SNR, fringe amplitude, Allan variance turnover).

5. **dist_or_number**: Instrument config parameters are either a fixed number OR a distribution spec (`{dist: uniform, low: ..., high: ...}`). The `sampling.py` module handles this transparently.

6. **WMS convention**: Follows Rieker/Hanson calibration-free WMS convention. The lock-in uses zero-phase Butterworth filter with documented X/Y sign convention (Y sign differs between Rieker and some other references — see `wms.py` docstring).

## External services

### GitHub
- **Organization**: `spektran` (github.com/spektran)
- **Repository**: `spektran/spektran`
- **Pages**: `spektran.github.io/spektran/` (MkDocs Material, deployed via `.github/workflows/docs.yml`)
- **CI**: `.github/workflows/ci.yml` — test matrix (3.10/3.11/3.12), lint, schema validation, HITRAN reference tests on push
- **CLI auth**: `gh` CLI authenticated as user `spectramaster` with repo+workflow scopes
- **To verify**: `gh auth status`

### Hugging Face
- **User**: `Deepnight`
- **Dataset**: `Deepnight/spektran-ch4-v0` (4 splits: train, validation, test, test_heldout_instrument)
- **License on HF**: CC BY 4.0
- **CLI auth**: `huggingface-cli` authenticated via `hf auth login`
- **To verify**: `huggingface-cli whoami`
- **Push script**: `scripts/push_to_hf.py`

### PyPI
- **Package name reserved**: `spektran` (not yet published; will publish at v0.1.0 stable)

## Conventions

- **Units in field names**: Every numeric field name includes its unit suffix (e.g., `wavenumber_cm1`, `temperature_K`, `pressure_atm`, `path_length_m`). This is enforced by schema lint in Gate G2.
- **DOI citations**: Every physics formula must cite its source DOI in the docstring.
- **Deterministic by default**: All stochastic functions take an explicit `seed` or `rng` argument. Non-reproducible defaults are rejected in review.
- **No comments except for "why"**: Code should be self-documenting. Only add comments when the reason is non-obvious (workarounds, sign conventions, subtle invariants).
- **Gate integrity**: Gate threshold scripts (`gates/`) must never be modified in the same PR that tries to pass them.
- **Line length**: 100 chars (ruff).
- **Import order**: isort via ruff (I rules).

## Quality gate system

| Gate | What it checks | Script |
|---|---|---|
| G1 | Name availability (GitHub, PyPI, HF, web confusion) | Manual (report in `gates/reports/g1_*.json`) |
| G2 | Schema completeness: 23-paper literature coverage, unit lint, 100-record round-trip | `gates/g2_schema_completeness.py` |
| G3-DA | Physics correctness (DA): 1000-point dual-impl cross-validation, R² > 0.9999 | `gates/g3_physics_da.py` |
| G3-WMS | Physics correctness (WMS): 60-point cross-validation + Arndt anchor | `gates/g3_physics_wms.py` |
| G4 | Noise realism: NEA/SNR/fringe/Allan envelopes vs 18-paper survey | `gates/g4_noise_realism.py` |
| G5 | Cold-start usability: byte-identical reproduction from docs alone | Manual (report in `gates/reports/g5_report.md`) |

All gates passed with independent adversarial reviews archived in `gates/reports/`.

## Benchmark tasks

| Task | ID | Input → Output | Key metric |
|---|---|---|---|
| Concentration regression | T1 | Noisy DA scan → ppm | MAE |
| Spectral denoising | T2 | Raw spectrum → clean absorbance | RMSE |
| Cross-instrument generalization | T3 | T1 on held-out instruments | Degradation ratio vs T1 |

Official splits use instruments vi-01 through vi-06 for training, vi-07/vi-08 (held-out) for T3. The `held_out: true` flag in instrument configs marks them.

## Data flow

```
YAML instrument config + YAML dataset config
  → scripts/generate_dataset.py
    → HDF5 file (schema-validated on write)
      → scripts/push_to_hf.py → Hugging Face Hub
```

HDF5 layout: `/records/<record_id>/<signal_name>` arrays + `.attrs["meta"]` JSON blob.

## Adding a new sensing modality (future)

1. Add physics module under `src/spektran/physics/` (e.g., `ndir.py`)
2. Add instrument noise modules if the noise model differs
3. Add `technique: NDIR` instrument configs under `configs/instruments/`
4. Extend `record.schema.json` with conditional fields for the new technique
5. Add dataset configs and benchmark tasks
6. Run gates G2–G4 for the new modality

The `technique` field in the schema already supports extension — new modalities extend, not fork.

## Known limitations & honest gaps

- **Q(T) ratio**: Uses power-law approximation (exact at 296K). TIPS polynomial planned for v1.0.
- **Sim-to-real gap**: Not hidden — it's the research topic of the T3 track. No experimental data yet.
- **Single species**: Only CH4 2ν3 band near 6047 cm⁻¹ currently. Multi-species is schema-ready but not exercised.
- **No experimental validation**: All data is simulation-born (`data_origin: simulated`).

## File naming conventions

- Instrument configs: `vi-{technique}-{tier}-{number}.yaml` (e.g., `vi-da-easy-01.yaml`)
- Dataset configs: `{molecule}-{task}-{split}-v{version}.yaml` (e.g., `ch4-t1-train-v0.yaml`)
- Gate reports: `g{N}_report.json` or `g{N}_review.md`

## PR and contribution flow

- Fork → feature branch → PR against `main`
- CI must pass: tests + lint + schema validation
- Physics PRs require DOI citation
- Schema changes re-trigger G2
- Data contributions require Data Submission Agreement (CC BY 4.0 grant)
- See `CONTRIBUTING.md` for full details
