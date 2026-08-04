# Phase 0 exit report — OpenGasSpec

Date: 2026-08-04. Plan reference: project plan §10 (Phase 0), §9 (gates).

## Gate status

| Gate | Scope | Self-check | Independent review | Status |
|---|---|---|---|---|
| G1 | Naming availability | PASS | Internally consistent (advisory: register accounts promptly) | **CLOSED** |
| G2 | Schema completeness (literature-anchored) | PASS (61/61 coverage, round-trip, unit lint) | PASS after 1 blocking DOI fix (applied + re-verified) | **CLOSED** |
| G3 | Physics correctness — DA scope | PASS (3 cross-validations < 1e-13 vs 1e-3 threshold) | PASS (seed-robust, independence verified, 10/10 DOI spot-check) | **CLOSED** (WMS part due in Phase 1) |

## Acceptance criterion

Plan: "`pip install -e .` then ~10 lines of code produce a clean CH4
absorption spectrum passing schema validation." — **Met**:
`examples/quickstart_ch4.py` produces a 2000-point spectrum (peak napierian
absorbance 1.56e-2 at 100 ppm, 10 m, 296 K, 1 atm; order-of-magnitude
hand-verified) and the assembled record validates against
`record.schema.json`.

## Deliverables

- Repo skeleton: dual license (Apache-2.0 code / CC BY 4.0 data), pyproject,
  CI workflow, CONTRIBUTING with gate-integrity rules, CITATION.cff
- `schema/`: record + instrument JSON Schemas (draft 2020-12), v0.1 with
  literature-driven extension; CHANGELOG; G2 field mapping (61 parameters)
- `docs/literature/`: 23-paper survey with verified citations; 61-parameter
  superset
- `src/opengasspec/`: physics layer (HITRAN access, Voigt via Faddeeva,
  HITRAN T-scaling, Beer-Lambert), validate CLI, package entry point
- `tests/`: 27 offline tests incl. plan §8 physics red lines; independent
  reference implementations (quadrature Voigt, scalar Beer-Lambert chain);
  online HITRAN comparison test (runs in CI `hitran-reference` job)
- `gates/`: G2 + G3 check scripts; all reports + both adversarial review
  verdicts archived under `gates/reports/`

## Known limitations (honest disclosure)

1. Built-in CH4 demo line list is approximate — examples/tests only; official
   generation must use hapi `fetch_lines` (enforced by `hitran_online` test).
2. Partition-function ratio defaults to a power law; exact at 296 K. TIPS
   injection planned for official dataset generation (Phase 1).
3. G3 chain validation covers 250–350 K, 0.1–2 atm, CH4; mtorr and
   combustion regimes to be added with the WMS gate revision (Phase 1).
4. The `hitran_online` HITRAN/hapi <0.1% comparison test has not yet been
   executed in this environment (no hapi/network); it gates CI on push.

## Human actions required (plan §9, unchanged)

1. Register `opengasspec` on GitHub / PyPI / Hugging Face; configure tokens.
   Note: G1 availability was checked on 2026-08-04 and is perishable.
2. No external communications pending in Phase 0.

## Phase 1 plan (next)

Per plan §10: WMS chain (modulation + lock-in demodulation → G3 WMS part),
all instrument-noise modules (laser nonlinearity/RAM/drift, etalons, optics
drift, detector noise incl. 1/f, environment jitter), virtual-instrument
sampling mechanism, 6–8 predefined instruments → G4 literature anchoring
(anchors table from the 23-paper survey), dataset generator + HDF5/Parquet IO
+ validate CLI extension. Exit: G3 (full) + G4; one command generates 10k
reproducible noisy records from a YAML config.
