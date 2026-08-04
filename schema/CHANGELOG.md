# Schema changelog

## v0.1 (2026-08-04 — G2 candidate)

- 2026-08-04: Literature-driven extension after the Gate G2 survey
  (23 papers, 61-parameter superset; see docs/literature/ and
  schema/g2_field_mapping.yaml):
  - `record`: laser type/waveform/setpoints/power/tuning polynomial; WMS
    calibration-free parameters (harmonic_scheme, modulation_index, i0/i2,
    psi1/psi2); optional `instrument.cell` block (type, passes, volume,
    temperature control); optional line-parameter snapshot on `target_lines`
    (S, E'', gamma_air/self, n_air, delta_air); `conditions` sample flow and
    conditioning; new optional top-level `processing` block (calibration,
    baseline, fitting, demodulation, denoising, averaging, uncertainty).
  - `instrument`: optional `performance` block (SNR, detection limit, Allan
    minimum, precision/accuracy, linearity, response time, drift, rate) —
    fixed metrics anchored against literature ranges in Gate G4.
  - Coverage: 61/61 parameters covered; field-path existence machine-verified
    by gates/g2_schema_completeness.py.

## v0.1 (initial draft)

- 2026-08-04: Initial draft of `record.schema.json` and `instrument.schema.json`.
  - `record`: required blocks `record_id / schema_version / data_origin / technique / signals / labels / conditions / instrument`; `provenance` conditionally required for `data_origin: simulated`; WMS records require `modulation` and at least one demod signal.
  - `instrument`: distribution-valued parameters (`uniform / normal / loguniform / choice`), `held_out` flag for generalization-track instruments.
  - Unit convention: units embedded in field names (`_ppm`, `_K`, `_atm`, `_cm1`, `_Hz`, `_m`, `_MHz`, `_rad`).

**Finalization of v0.1 is gated on G2** (literature-anchored field-coverage ≥ 95% over ≥ 20 quantitative TDLAS papers, meta-validation, 100-record round-trip, zero unit-lint warnings). Until then this draft may change without a version bump.
