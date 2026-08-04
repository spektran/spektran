# Data schema (v0.1)

Every record = signal arrays (HDF5) + JSON metadata validated by
[`schema/record.schema.json`](https://github.com/opengasspec/opengasspec/blob/main/schema/record.schema.json).

Key design rules:

- **Units live in field names** (`concentration_ppm`, `temperature_K`,
  `depth_cm1`) — enforced by a unit-consistency lint with a documented
  dimensionless whitelist.
- **Conditional requirements**: `data_origin: simulated` requires a full
  `provenance` block (generator version, seed, instrument config id, every
  sampled noise parameter); `technique: TDLAS-WMS` requires `modulation`
  parameters and at least one demodulated signal.
- **`technique` is an enum** ready to extend to NDIR / PAS / CRDS (schema
  v0.2+).
- Anchored in the literature: the field set covers 61/61 parameters distilled
  from a 23-paper survey of quantitative TDLAS work (Gate G2; see
  `docs/literature/` and `schema/g2_field_mapping.yaml`).

Virtual instruments are configs of **distributions** (uniform / normal /
loguniform / choice) validated by `instrument.schema.json`; the generator
samples concrete values per record and writes them to provenance.

Schema changes go through PR + G2 re-check and are logged in
`schema/CHANGELOG.md`.
