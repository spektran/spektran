# Contributing to SPEKTRAN

Thank you for your interest! Read access is fully open; write access (data ingestion, schema changes) goes through a curated flow: CI automated validation + maintainer review.

## Contributor License Agreement (CLA)

By submitting a pull request to this repository, you agree to the following terms:

**Code contributions (Apache-2.0):** You grant the SPEKTRAN project and its maintainers a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to use, reproduce, modify, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute your contribution and any derivative works under the Apache License 2.0. You represent that you have the legal right to grant this license.

**Data contributions (CC BY 4.0):** If your contribution includes data (spectra, instrument configs, benchmark results), you grant a license under the Creative Commons Attribution 4.0 International License. You represent that you have the legal right to license the data under CC BY 4.0.

**No obligation:** Submitting a contribution does not obligate the project to accept or merge it.

The PR template includes a CLA checkbox that must be checked before merging.

## Code contributions

1. Fork, create a feature branch, open a PR.
2. CI must pass: unit tests, physics-correctness tests, schema validation, lint (`ruff`).
3. Physics formulas must cite their literature source (DOI) in the docstring.
4. All stochastic code must accept an explicit `seed`/`rng` argument. Non-reproducible default behavior is rejected.
5. Code license: Apache-2.0. By submitting a PR you agree to the CLA above.

## Adding a new baseline

SPEKTRAN is AI Agent-ready — new baselines are automatically available to AI agents
via the registry.

1. Create `baselines/<name>/train.py` following the pattern in existing baselines (see `baselines/common.py` for data loading helpers).
2. Add an entry to `baselines/registry.yaml` with display name, description, tasks, datasets, and score file paths.
3. Verify: `spektran train --baseline <name> --json` should auto-generate data, run training, and report scores.
4. Open a PR with your baseline code and the registry entry.

## Schema changes

Schema modifications are write-privileged:

- Any change goes through a PR and re-triggers the Gate G2 completeness check.
- Every change is recorded in `schema/CHANGELOG.md` with rationale.
- Breaking changes bump the minor version pre-1.0 (`0.1` → `0.2`).

## Adding a new modality

SPEKTRAN ships 5 modalities (TDLAS, NDIR, CRDS, FTIR, DOAS). To add a sixth:

1. Add physics module(s) under `src/spektran/physics/` (forward model).
2. Add noise module(s) under `src/spektran/instrument/` (instrument noise chain).
3. Create virtual instrument configs under `configs/instruments/vi-{technique}-*.yaml`.
4. Extend `record.schema.json` with conditional fields for the new technique.
5. Add a generator module (`src/spektran/{technique}_generator.py`) and CLI routing in `cli.py`.
6. Add dataset configs, benchmark tasks, and at least one baseline.
7. Run gates G2–G4 for the new modality.
8. Update `docs/`, `README.md`, `README_zh.md`, and `AGENTS.md`.

The `technique` field in the schema already supports extension — new modalities extend, not fork.

## Data contributions (future)

External data (experimental or simulated) will be accepted once the curated ingestion pipeline lands (Phase 6). Requirements preview:

- Records must validate against the current `record.schema.json` (`spektran validate`).
- Unit-consistency lint must pass with zero warnings.
- Physics-plausibility check: CI re-runs a HITRAN forward simulation from your metadata and compares it against the submitted spectrum; large deviations are flagged for human review.
- **Data Submission Agreement**: the PR template requires you to affirm that you have the right to license the submitted data under CC BY 4.0.

## Gate integrity rules

- Gate thresholds and check scripts under `gates/` must not be modified in the same PR that is trying to pass them. Threshold changes require an independent PR with a CHANGELOG entry explaining the rationale.
