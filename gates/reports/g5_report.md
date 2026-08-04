# Gate G5 report — cold-start end-to-end usability

- Date: 2026-08-04
- Tester: independent agent, fresh git clone + fresh Python 3.11 venv,
  instructions taken ONLY from README.md, baselines/README.md, and docs/
  pages. Did not read source code to complete any step.
- Environment deviations from the plan's letter (documented honestly):
  fresh venv instead of a container (no container runtime in this
  environment); data obtained by regenerating from pinned configs instead of
  Hugging Face download (HF upload pending human token — regeneration is the
  documented path and is the stronger reproducibility test).
- Process note: the tester session was interrupted once by an API usage
  quota (not a project defect) and resumed; one interim evaluation was run
  by the execution agent on the tester's artifacts and later re-run by the
  tester itself — final numbers below are the tester's own.

## Verdict: **PASS** (all four conditions)

| Condition | Result |
|---|---|
| End-to-end without human intervention | PASS — every documented command ran unmodified first try; only gap was guessing CNN evaluate paths (defect #1, fixed) |
| Total < 60 min excluding model training | PASS — ~3 min stage wall time (~10 min including doc reading); CNN training 423 s excluded per gate definition |
| Baseline score deviation < 5% | PASS — max deviation 0.20% across all 8 leaderboard numbers; retraining regenerated the prediction CSVs **byte-for-byte** (git detected no modification) |
| Doc ambiguities recorded and fixed | 8 defects recorded; 4 actionable ones fixed (below), fixes re-verified |

## Stage timings (tester's machine)

install 8 s · quickstart example 42 s · generate 4 splits (285 MB) 24 s ·
baseline deps 76 s · ridge training 21 s · CNN training 423 s (excluded) ·
4 evaluations ~4 s.

## Documentation defects and dispositions

| # | Defect | Disposition |
|---|---|---|
| 1 | No CNN evaluate commands; paths and `--t1-mae` had to be guessed | **Fixed**: full per-model commands added to baselines/README.md; fixed command re-verified verbatim |
| 2 | `--t1-mae 2.8426` semantics undocumented | **Fixed**: documented as "the same model's own T1 test MAE" |
| 3 | Shipped prediction CSVs allow accidental no-training "reproduction" | **Fixed**: prediction files untracked from git (regenerable; already gitignored); note added to baselines/README.md |
| 4 | quickstart install line omitted `torch` | **Fixed**: `pip install scikit-learn torch` + pointer to full commands |
| 5 | README quickstart lacked clone/venv steps; unused numpy import | **Fixed** |
| 6 | No runtime/size expectations (CNN ~7 min, data ~285 MB) | **Fixed** in quickstart + baselines README |
| 7 | No pointer that HF-hosted data is planned | Noted; will land with the HF push (HUMAN_ACTIONS.md) |
| 8 | Cosmetic: rounding inconsistency; undocumented demo config | **Fixed**: full-precision note; demo config labeled in quickstart |

## Scores vs leaderboard (tester's runs)

Ridge T1 MAE 2.8426 (Δ0.09%) · Ridge T3 3.7184 (Δ0.04%) · CNN T1 15.5807
(Δ0.004%) · CNN T3 28.3007 (Δ0.003%) · degradations 1.3081 / 1.8164
(Δ≤0.2%). Prediction files byte-identical to the official ones.
