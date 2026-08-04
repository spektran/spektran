# Quality gates

Development is gated by quantitative, adversarially reviewed checkpoints.
All reports and reviewer verdicts are version-controlled under
`gates/reports/` as public evidence.

| Gate | What it proves | Status |
|---|---|---|
| G1 | Name availability across GitHub/PyPI/HF | PASS (2026-08-04) |
| G2 | Schema covers 61/61 parameters from a 23-paper survey; round-trip; unit lint | PASS + independent review |
| G3 | Dual-implementation physics cross-validation (DA < 0.1%, WMS < 1%, 1000 pts each) | PASS + independent review |
| G4 | Instrument noise statistics inside literature envelopes (18-paper anchors) | PASS + independent review |
| G5 | Independent cold-start: install -> data -> train -> score from docs alone | see gates/reports/ |

Design principles: reviewers run in fresh sessions with read-only access;
gate thresholds cannot be changed in the PR that passes them; every claim
in the anchor tables is traceable to a cited paper.

Known honest limitations: G4 validates statistical similarity to published
systems, not point-wise truth; official CH4 line data must come from HITRAN
via hapi (the built-in demo line list is approximate and so labeled).
