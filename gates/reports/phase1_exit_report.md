# Phase 1 exit report — OpenGasSpec

Date: 2026-08-04. Plan reference: §10 (Phase 1), §9 (gates).

## Gate status

| Gate | Scope | Self-check | Independent review | Status |
|---|---|---|---|---|
| G3 (WMS) | WMS chain dual-implementation | PASS (1000 pts, max 5.2e-6 vs 1% threshold; Arndt anchor 9.8e-4) | PASS — reviewer independently derived the Arndt closed form; seed-robust; reference independence verified | **CLOSED** |
| G4 | Noise realism, literature-anchored | PASS (8/8 instruments in envelopes; family spread OK; attempt 2 of 3) | PASS — anchors traced to primary papers (5/5 spot-checks exact); envelopes read from YAML, not hardcoded; metrics computed from generated data | **CLOSED** |

G3 is now closed in full (DA part closed in Phase 0).

## Acceptance criterion

Plan: "one command generates a 10k-record reproducible noisy dataset from a
YAML config" — **Met**: `python scripts/generate_dataset.py
configs/datasets/ch4-da-medium-v0.yaml` produced 10,000 schema-valid records
in ~12 s (generation 5.8 s + write/validate 5.8 s).

## Deliverables

- WMS chain (time-domain modulation + RAM + digital lock-in) with
  Fourier-quadrature reference implementation and Arndt analytic anchor
- Instrument-effects layer: laser (nonlinearity/ramp/drift/linewidth),
  etalons (multi-system, phase drift), detector (white, 1/f Timmer-Koenig,
  gain nonlinearity, ADC), optics (baseline/fluctuation/decay), environment
  (physically consistent T/P jitter)
- Virtual-instrument sampling (distribution configs -> concrete provenance),
  generator with SeedSequence.spawn per-record streams, time-series mode
  (frozen instrument + evolving drift), HDF5 IO
- 8 schema-valid virtual instruments (easy/medium/hard x DA/WMS + 2 held-out)
  tuned into literature envelopes from 18-paper numeric anchor survey
- 66 offline tests; all gate reports + review verdicts archived

## Review flags carried forward (non-blocking, tracked)

1. G4 NEA envelope is the union of pure-NEA anchors and zhao2016 fringe/RIN
   absorbance floors; the hard tier passes NEA only via that union
   (disclosed in gate code). Revisit when more NEA anchors are collected.
2. Fringe metric is taken from (verified-faithful) sampled provenance;
   a signal-based estimator would be strictly stronger. Planned improvement.
3. Easy-instrument per-record SNR tail exceeds the literature P95 for
   ~20-40% of records (medians are in-envelope). Mild realism gap, disclosed.
4. Gate/anchors/configs landed in one commit, so attempt-1->2 integrity rests
   on commit-message disclosure + anchor paper-traceability. Future gates:
   commit the gate script before the first attempt.
5. Lock-in Y-quadrature sign convention differs from the reference's sine
   coefficient (magnitude comparison unaffected); documented in wms.py.
6. From Phase 0: TIPS partition-function injection for official generation
   still pending (power-law default is exact at 296 K, the official
   generation temperature).

## Phase 2 (started)

Official v0 splits generated (T1 train/val/test + T3 held-out test;
5000/500/1000, reduced from the plan's 50k/5k/10k — scale is config-driven
and the plan marks it negotiable). Baselines trained and evaluated; docs,
G5 cold-start, and HF/Zenodo packaging next. Human actions still pending:
platform account registration + tokens.
