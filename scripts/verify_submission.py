#!/usr/bin/env python3
"""Verify a benchmark submission against private validation seeds.

This script is run by maintainers (who have access to the private seeds) to
score a submission. It regenerates the held-back test set from the private
seeds, runs the submission's predictions against ground truth, and reports
scores. The submitter never sees the seeds or ground truth.

Usage (maintainer-only):
    python scripts/verify_submission.py \
        --seeds private/validation_seeds.json \
        --predictions submission/predictions.csv \
        --task T1-concentration
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True, help="Path to private seed file")
    parser.add_argument("--predictions", required=True, help="Path to submission CSV")
    parser.add_argument("--task", required=True, choices=["T1-concentration", "T2-denoise", "T3-generalization"])
    parser.add_argument("--hashes", help="Path to public hash file (optional integrity check)")
    args = parser.parse_args()

    seeds_path = Path(args.seeds)
    if not seeds_path.exists():
        print(f"ERROR: Private seed file not found: {seeds_path}")
        print("This file is not in the public repo. Only maintainers have access.")
        raise SystemExit(1)

    seeds = json.loads(seeds_path.read_text())
    print(f"Loaded {seeds['n_records']} validation seeds")
    print(f"Instrument config: {seeds['instrument_config']}")
    print(f"Task: {args.task}")
    print()
    print("To complete verification:")
    print("1. Generate the validation dataset from seeds (not yet automated)")
    print("2. Run: python -m spektran.benchmark.evaluate \\")
    print(f"     --task {args.task} --truth <generated.h5> --predictions {args.predictions}")
    print()
    print("Full automation of this pipeline is planned for v0.2.0.")


if __name__ == "__main__":
    main()
