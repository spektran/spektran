# Validation Hashes

This directory contains SHA-256 hashes of private validation datasets used to verify benchmark submissions. The actual seeds and ground-truth data are held privately by maintainers.

## How it works

1. Maintainers generate validation seeds using `scripts/generate_validation_seeds.py`
2. Only the hash fingerprints are committed here (never the seeds)
3. Benchmark submitters submit predictions on the public test sets
4. For leaderboard verification, maintainers regenerate the private test set and score against it using `scripts/verify_submission.py`

This prevents data leakage: submitters cannot reverse-engineer the validation set from published hashes.
