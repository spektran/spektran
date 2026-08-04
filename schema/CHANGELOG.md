# Schema changelog

## v0.2 (2026-08 — Phase 3)

- `schema_version` accepts `"0.1"` and `"0.2"` (backward compatible).
- New signal slots: `demod_3f`, `demod_4f` for higher-harmonic WMS.
- `modulation.harmonic_scheme` enum extended with `"3f"`, `"4f"`.
- New optional top-level `measurement` block for experimental records
  (`operator`, `date_utc`, `facility`, `instrument_serial`, `notes`).
- Generator now emits `schema_version: "0.2"`.
- New optional `labels.ood_label` (0/1): out-of-distribution flag for the T6
  instrument-detection task, stamped by `spektran generate` on records built
  from an `ood_task: true` dataset config. Additive; absent on all other
  records.

## v0.1 (2026-08-04 — generator integration)

- 2026-08-04 (later): `instrument` schema additions driven by the generator
  implementation: `detector.one_over_f_sigma_rel` (total sigma of the 1/f
  component), `optics.intensity_ramp_slope_rel` /
  `optics.intensity_ramp_curvature_rel` (laser power change over the current
  scan), `etalons[].phase_rad` (per-record fringe phase). All carry unit
  suffixes; unit lint re-run clean; G2 coverage unaffected (additive).

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
