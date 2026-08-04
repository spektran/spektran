# Phase 2 exit report — OpenGasSpec

Date: 2026-08-04. Plan reference: §10 (Phase 2), §9 (G5).

## Gate status

| Gate | Self-check | Independent verification | Status |
|---|---|---|---|
| G5 cold-start usability | n/a (G5 is inherently independent) | PASS — fresh clone + fresh venv, docs-only navigation, all 4 conditions met, byte-identical reproduction | **CLOSED** (8 doc defects recorded, 7 fixed, 1 pending HF upload) |

With G5 closed, **all five gates (G1–G5) defined in the plan are now closed**
within the scope achievable without platform credentials.

## Deliverables

- Official v0 splits as pinned configs (T1 train/val/test 5000/500/1000,
  T3 held-out test 1000) — regenerable bit-for-bit by anyone
- Three baselines with public hyperparameters and exact scores:
  ridge (T1 MAE 2.84 ppm, T3 degradation 1.31x), 1D CNN (15.58 ppm, 1.82x),
  classical wing-poly T2 reference (spectral RMSE 6.31e-3)
- First scientific observation from the flagship track: the deep baseline
  overfits instrument signatures harder than the linear one
- One-command evaluation (`python -m opengasspec.benchmark.evaluate`)
- Docs site source (MkDocs): quickstart, schema, benchmark rules,
  reproducibility contract, gates evidence page
- Publishing machinery ready: `scripts/push_to_hf.py`, `.zenodo.json`

## Deviations from the plan (documented)

1. Split scale v0 = 5000/500/1000 vs plan's 50k/5k/10k (plan marks scale
   negotiable; configs scale by editing one number).
2. v0 tasks input DA raw scans; WMS-input task variants ship with the
   engine (instruments exist) but official WMS splits are deferred to v0.2.
3. G5 ran in a fresh venv rather than a container, and regenerated data
   instead of HF download (upload pending human token) — see g5_report.md.
4. Official generation currently uses the approximate built-in CH4 demo
   lines; switching `line_source: hitran` (hapi fetch) is wired but needs a
   network-enabled release run + TIPS partition sums before v1.0 claims
   HITRAN-accurate spectra. This is the top pre-release engineering item.

## Blocked on human (see HUMAN_ACTIONS.md)

Platform registration (GitHub/PyPI/HF/Zenodo), tokens, HF push, v0.1.0 tag
for the DOI, GitHub Pages activation.

## Suggested next (Phase 3 preview, per plan)

Data-descriptor paper draft; literature-scan agent for external data
sources; schema v0.2 (NDIR/PAS/CRDS technique extension, structured
uncertainty budget); WMS official splits; TIPS injection + HITRAN release
run; signal-based fringe estimator for G4.
