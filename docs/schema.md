# Data schema

## Schema versions

### v0.2 (current)

Backward compatible with v0.1. New features:

- **`schema_version`** is now an enum (`"0.1"` or `"0.2"`) instead of a const
- **Higher harmonics**: `demod_3f` and `demod_4f` signal array slots for 3rd and 4th harmonic WMS demodulation
- **`harmonic_scheme`** enum extended with `"3f"` and `"4f"` (full set: `"1f"`, `"2f"`, `"2f/1f"`, `"3f"`, `"4f"`, `"other"`)
- **`measurement` block**: optional block for experimental-record metadata (`operator`, `date_utc`, `facility`, `instrument_serial`, `notes`), primarily for `data_origin: experimental/augmented`
- **`harmonics` array** in the instrument config's `modulation` block: specifies which harmonics the instrument demodulates (default `[1, 2]`, extendable to `[1, 2, 3, 4]`)

All v0.1 records validate against the v0.2 schema without modification.

See [`schema/CHANGELOG.md`](https://github.com/spektran/spektran/blob/main/schema/CHANGELOG.md) for the full change history.

### v0.1

Every record = signal arrays (HDF5) + JSON metadata validated by
[`schema/record.schema.json`](https://github.com/spektran/spektran/blob/main/schema/record.schema.json).

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
