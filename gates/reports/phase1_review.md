# Independent adversarial review — Phase 1 exit (G3-WMS + G4)

- Date: 2026-08-04
- Reviewer: independent agent session (read-only)
- Verdict: **PASS on both gates. Zero blocking findings.**

Highlights of the reviewer's evidence:
- G3(WMS): re-run matches archive exactly; reviewer independently derived
  the Arndt Lorentzian H2 closed form and confirmed the implementation to
  machine precision (rel. err <= 2e-12 across m = 0.1..3.0); seed-robustness
  re-checked with 3 seeds (max 1.1e-6 vs 1% threshold); reference impl has
  zero opengasspec imports.
- G4: report re-run byte-identical; envelope numbers recomputed from the
  anchors YAML (no hardcoding); 5/5 anchor spot-checks traced to primary
  papers with exact figures (chang2023, lou2019, zhao2016, klein2014,
  li2016); NEA/SNR/Allan computed from generated signals, not configs;
  median-based checks judged the defensible statistical interpretation
  (per-record P95 against single-number-per-paper anchors would be
  incoherent); generator single-record regeneration test confirmed real.

Non-blocking flags (dispositions in phase1_exit_report.md): NEA envelope
union, fringe metric provenance, easy-tier SNR tail, single-commit gate
provenance, lock-in Y-sign convention (now documented in wms.py).
