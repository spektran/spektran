# Quality gates

Development is gated by quantitative, adversarially reviewed checkpoints.
All reports and reviewer verdicts are version-controlled under
`gates/reports/` as public evidence.

| Gate | What it proves | Status |
|---|---|---|
| G1 | Name availability across GitHub/PyPI/HF | PASS (2026-08-04) |
| G2 | Schema covers 61/61 parameters from a 23-paper survey; round-trip; unit lint | PASS + independent review |
| G3 | Dual-implementation physics cross-validation (DA < 0.1%, WMS < 1%, 1000 pts each) | PASS + independent review |
| G3-CRDS | CRDS ring-down time + round-trip identity (< 0.1%, 1000 pts) | PASS (2026-08-12) |
| G3-FTIR | FTIR resolution, apodization, forward-chain consistency | PASS (2026-08-12) |
| G3-DOAS | DOAS Beer-Lambert, Rayleigh scaling, polynomial high-pass | PASS (2026-08-12) |
| G4 | Instrument noise statistics inside literature envelopes (18-paper anchors) | PASS + independent review |
| G5 | Independent cold-start: install -> data -> train -> score from docs alone | see gates/reports/ |

### Phase 3 extensions

Phase 3 engine additions (TIPS polynomial, multi-species superposition,
3f/4f demodulation) are covered by the existing G3 dual-implementation
framework:

- TIPS has its own cross-validation: independent reference implementation
  (`tests/reference_impl/ref_tips.py`) with separately derived coefficients,
  verified to < 0.5% relative error against the main implementation.
- Multi-species superposition: `test_multi_species_absorbance_superposition`
  (`tests/test_absorption.py`) confirms Beer-Lambert additivity numerically
  (< 0.01% relative); `test_generate_record_with_interferent`
  (`tests/test_generator.py`) exercises the full generator pipeline with a
  CH4 target species plus an H2O interferent end to end.
- 3f/4f demodulation reuses the existing `simulate_wms()` harmonic machinery
  and the same independent reference implementation used for 1f/2f
  (`ref_wms.py`'s Fourier-quadrature harmonic coefficients are generic in
  harmonic order, not special-cased to 1f/2f). Dedicated tests check 3f/4f
  output shape and physical plausibility; the G3-WMS random-point numerical
  cross-validation sweep itself currently still samples only 1f/2f, so full
  dual-implementation coverage of 3f/4f is open follow-up work rather than a
  completed gate pass.

No new gates were added: TIPS and multi-species superposition extend the
existing G3 test suite directly, and 3f/4f demodulation shares G3's
machinery and reference implementation without requiring new threshold
scripts.

Design principles: reviewers run in fresh sessions with read-only access;
gate thresholds cannot be changed in the PR that passes them; every claim
in the anchor tables is traceable to a cited paper.

### v0.6.0 extensions — CRDS, FTIR, DOAS

Each new modality has its own G3 physics gate script:

- **G3-CRDS** (`gates/g3_physics_crds.py`): ring-down time cross-validation
  against analytic `tau = L / (c * (1-R + alpha*L))`, round-trip identity
  (absorption_from_tau recovers alpha to < 0.01%), empty-cavity tau check.
- **G3-FTIR** (`gates/g3_physics_ftir.py`): spectral resolution `1/(2*OPD)`
  identity, 5 apodization functions boundary/range checks, forward-chain
  consistency (transmittance in [0,1], absorption depth scales with conc).
- **G3-DOAS** (`gates/g3_physics_doas.py`): Beer-Lambert OD cross-validation
  vs ideal-gas number density formula, Rayleigh lambda^-4 scaling, polynomial
  high-pass null-check, molecular OD consistency with `simulate_doas_spectrum`.

Literature anchors for CRDS/FTIR/DOAS added to `configs/instruments/literature_anchors.yaml`
referencing Romanini (1997), Crosson (2008), Paldus & Kachanov (2005),
Griffiths & de Haseth (2007), Wunch et al. (2011), Platt & Stutz (2008),
Pinardi et al. (2013), and Bogumil et al. (2003).

Known honest limitations: G4 validates statistical similarity to published
systems, not point-wise truth; official CH4 line data must come from HITRAN
via hapi (the built-in demo line list is approximate and so labeled).
CRDS/FTIR use the same demo line lists; DOAS uses synthetic cross sections
by design.
