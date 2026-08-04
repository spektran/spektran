# Contributing to SPEKTRAN

Thank you for your interest! Read access is fully open; write access (data ingestion, schema changes) goes through a curated flow: CI automated validation + maintainer review.

## Code contributions

1. Fork, create a feature branch, open a PR.
2. CI must pass: unit tests, physics-correctness tests, schema validation, lint (`ruff`).
3. Physics formulas must cite their literature source (DOI) in the docstring.
4. All stochastic code must accept an explicit `seed`/`rng` argument. Non-reproducible default behavior is rejected.
5. Code license: Apache-2.0. By submitting a PR you agree to license your contribution under Apache-2.0.

## Schema changes

Schema modifications are write-privileged:

- Any change goes through a PR and re-triggers the Gate G2 completeness check.
- Every change is recorded in `schema/CHANGELOG.md` with rationale.
- Breaking changes bump the minor version pre-1.0 (`0.1` → `0.2`).

## Data contributions (future)

External data (experimental or simulated) will be accepted once the curated ingestion pipeline lands (Phase 3). Requirements preview:

- Records must validate against the current `record.schema.json` (`spektran validate`).
- Unit-consistency lint must pass with zero warnings.
- Physics-plausibility check: CI re-runs a HITRAN forward simulation from your metadata and compares it against the submitted spectrum; large deviations are flagged for human review.
- **Data Submission Agreement**: the PR template requires you to affirm that you have the right to license the submitted data under CC BY 4.0.

## Gate integrity rules

- Gate thresholds and check scripts under `gates/` must not be modified in the same PR that is trying to pass them. Threshold changes require an independent PR with a CHANGELOG entry explaining the rationale.
