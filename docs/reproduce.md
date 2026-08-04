# Reproducibility contract

- All randomness flows from explicit seeds. Datasets: a master seed spawns
  per-record independent streams (`numpy.random.SeedSequence.spawn`), so the
  full dataset is bit-reproducible AND any single record can be regenerated
  alone (`tests/test_generator.py` verifies both).
- Every simulated record's `provenance` holds: generator version, HITRAN data
  version, per-record seed/spawn key, instrument config id, and every sampled
  noise parameter value.
- Baseline trainings pin seeds and hyperparameters; `hyperparams.json` is
  written at train time including validation curves.
- Physics correctness is enforced by dual independent implementations
  (Faddeeva vs quadrature; time-domain lock-in vs Fourier coefficients) that
  must agree to 0.1% / 1% on 1000 random points — see [Quality gates](gates.md).
