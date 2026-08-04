# Schema changelog

## v0.1 (draft — NOT finalized)

- 2026-08-04: Initial draft of `record.schema.json` and `instrument.schema.json`.
  - `record`: required blocks `record_id / schema_version / data_origin / technique / signals / labels / conditions / instrument`; `provenance` conditionally required for `data_origin: simulated`; WMS records require `modulation` and at least one demod signal.
  - `instrument`: distribution-valued parameters (`uniform / normal / loguniform / choice`), `held_out` flag for generalization-track instruments.
  - Unit convention: units embedded in field names (`_ppm`, `_K`, `_atm`, `_cm1`, `_Hz`, `_m`, `_MHz`, `_rad`).

**Finalization of v0.1 is gated on G2** (literature-anchored field-coverage ≥ 95% over ≥ 20 quantitative TDLAS papers, meta-validation, 100-record round-trip, zero unit-lint warnings). Until then this draft may change without a version bump.
