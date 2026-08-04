# Independent adversarial review — Gate G3 (DA scope) + G1

- Date: 2026-08-04
- Reviewer: independent agent session (read-only; did not author the code;
  execution agent's reasoning not visible to reviewer)
- Verdict: **G3 (DA scope) GENUINELY PASSES. G1 internally consistent.**

## Evidence summary

1. **Gate re-run**: `gates/g3_physics_da.py` exits 0; numbers reproduce the
   archived report exactly. Lineshape max rel. deviation 1.075e-13, forward
   chain 1.211e-14 — ~10 orders of magnitude below the 1e-3 threshold.
2. **Test suite**: 27 passed, 1 skipped (online HITRAN test, correctly
   excluded offline).
3. **Reference-implementation independence**: zero code imports of the main
   package in `tests/reference_impl/`; genuinely different algorithm
   (QUADPACK quadrature of Armstrong's K(x,y) integral vs Faddeeva/wofz);
   reference redefines its own CODATA constants.
4. **No cherry-picking**: reviewer re-ran with seeds 1, 42, 987654321
   (lineshape max 5.4e-12) and 3, 777 (chain max 1.3e-14); stress-tested
   offsets to ±100 linewidths (max 1.1e-14). Seed-robust.
5. **DOI spot-check (10/10)**: all cited references real and formulas match
   sources. Two minor attribution nits (fixed post-review, see below).
6. **Independent physics sanity**: CH4 Doppler HWHM at 296 K / 6047 cm-1 =
   9.3052e-3 cm-1 (analytic, matches implementation to 7 digits); number
   density reproduces Loschmidt constant at 273.15 K; Voigt limits converge
   at 9e-7 / 3e-10 relative error.

## Non-blocking weaknesses flagged (disposition)

| # | Finding | Disposition |
|---|---|---|
| 1 | Partition-function power-law ratio is a shared input, not cross-validated | Acknowledged in docstring; TIPS injection planned for official generation (Phase 1). Tracked. |
| 2 | Demo CH4 line list is approximate | Already labeled; official generation must use `fetch_lines`. Enforced by `hitran_online` CI test. |
| 3 | Forward-chain check was single-line only | **Fixed**: multi-line summation check added to the gate (stricter). |
| 4 | Chain parameter ranges exclude mtorr / combustion regimes | Tracked for Phase 1 (WMS gate revision). |
| 5 | Citation-precision nits (Demtröder Eq. 3.43 FWHM vs HWHM; "(A11)" attribution) | **Fixed** in docstrings. |
| 6 | Gate report overwrite, repo not under git at review time | Repo committed to git immediately after review snapshot (360b13f); reports now version-controlled. |

Per plan §9, reviewer conclusions prevail over the execution agent's; here
both agree: PASS.
