# Independent adversarial review — Gate G2

- Date: 2026-08-04
- Reviewer: independent agent session (read-only; did not author code or survey)
- Verdict: **G2 genuinely passes on substance, conditional on one blocking
  fix (applied, see disposition).**

## Evidence summary

1. **Gate re-run**: exits 0; report matches (61/61 covered, 0 round-trip
   failures, 0 lint warnings).
2. **Field-path resolver**: probed adversarially (nonsense leaves, wrong
   prefix, `[]` on non-arrays all rejected); re-ran independently on all 61
   mappings. Two theoretical cheat vectors (container-only paths,
   semantically unrelated fields) — 15+ mappings spot-checked semantically,
   none exploits them. i0/i2/psi1/psi2 mapping matches the Rieker/Hanson
   calibration-free WMS convention exactly.
3. **Literature**: all 23 papers real; 17/18 explicit DOIs resolve; 10
   cross-examined against Crossref with exact metadata match; the 5
   `doi-unverified` entries located as real papers.
4. **Superset**: internally consistent (union of per-paper checklists equals
   superset exactly; no padding); no must-have gaps for a simulated
   benchmark.
5. **Schema conditionals**: negative tests all reject correctly (WMS without
   modulation/demod, simulated without provenance, junk fields).
6. **Test suite**: 27 passed, 1 skipped.

## Findings and disposition

| # | Severity | Finding | Disposition |
|---|---|---|---|
| 1 | **Blocking** | shao2019 DOI `10.1016/j.saa.2019.117118` constructed from article number, does not resolve, yet not flagged `doi-unverified` | **Fixed**: corrected to `10.1016/j.saa.2019.05.023` (re-verified via doi.org, HTTP 302) |
| 2 | Non-blocking | Unit lint skipped `$ref`'d (`dist_or_number`) properties — most of the instrument schema | **Fixed**: lint now resolves local $refs; immediately caught `scan_nonlinearity_poly` (renamed `scan_nonlinearity_poly_cm1`) |
| 3 | Non-blocking | Round-trip generator exercised ~half the schema surface | **Fixed**: generator now also emits raw_scan, demod_1f, im-parameters, interferents, multi-species labels, cell, laser details, processing, line snapshots |
| 4 | Non-blocking | goldenstein2014_hp author list inaccurate | **Fixed** from Crossref: Goldenstein, Spearrin, Schultz, Jeffries, Hanson |
| 5 | Non-blocking | 5 entries `doi-unverified` | **Upgraded**: reviewer-located DOIs re-verified via doi.org and written in |
| 6 | Non-blocking | `mole_fraction_range` covers value, not range (range lives in dataset configs) | Accepted as documented design; noted in mapping file |
| 7 | Non-blocking | `uncertainty_budget` free-text rather than structured | Deferred to schema v0.2 |

Post-fix status: G2 gate re-run PASS (all four checks), full test suite green.
Per plan §9 the reviewer's conclusion prevails; with the blocking fix applied,
**G2 is certified PASS.**
