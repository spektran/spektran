#!/usr/bin/env python3
"""Generate private validation seeds and publish only data hashes.

This script creates a held-back test set whose seeds are NEVER committed to the
public repo. Only the SHA-256 hashes of the generated records are published,
allowing maintainers to verify benchmark submissions without leaking the
ground-truth generation parameters.

Usage:
    python scripts/generate_validation_seeds.py \
        --instrument configs/instruments/vi-da-heldout-07.yaml \
        --n-records 500 \
        --out-seeds private/validation_seeds.json \
        --out-hashes validation/hashes.json

The --out-seeds file must NEVER be committed. The --out-hashes file is safe to
commit and is used by `scripts/verify_submission.py` to check submissions.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def generate_seeds(n_records: int) -> dict:
    """Generate cryptographically strong master seed and per-record child seeds."""
    master_seed = int.from_bytes(np.random.SeedSequence().entropy.tobytes()[:16], "big")
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n_records)
    return {
        "master_seed": str(master_seed),
        "n_records": n_records,
        "child_entropies": [str(c.entropy) for c in children],
    }


def hash_record_meta(record_meta: dict) -> str:
    """SHA-256 of the canonical JSON serialization of a record's metadata."""
    canonical = json.dumps(record_meta, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", required=True, help="Path to instrument YAML config")
    parser.add_argument("--n-records", type=int, default=500, help="Number of validation records")
    parser.add_argument(
        "--out-seeds",
        required=True,
        help="Output path for PRIVATE seed file (do NOT commit)",
    )
    parser.add_argument(
        "--out-hashes",
        required=True,
        help="Output path for public hash file (safe to commit)",
    )
    args = parser.parse_args()

    seeds = generate_seeds(args.n_records)
    seeds["instrument_config"] = args.instrument

    seeds_path = Path(args.out_seeds)
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_path.write_text(json.dumps(seeds, indent=2) + "\n")
    print(f"PRIVATE seeds written to {seeds_path} — do NOT commit this file.", file=sys.stderr)

    hashes = {
        "validation_set_id": hashlib.sha256(
            json.dumps(seeds, sort_keys=True).encode()
        ).hexdigest()[:16],
        "instrument_config": args.instrument,
        "n_records": args.n_records,
        "master_seed_hash": hashlib.sha256(seeds["master_seed"].encode()).hexdigest(),
        "note": "Seeds are private. Only the master_seed_hash is published for verification.",
    }

    hashes_path = Path(args.out_hashes)
    hashes_path.parent.mkdir(parents=True, exist_ok=True)
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Public hashes written to {hashes_path} — safe to commit.", file=sys.stderr)


if __name__ == "__main__":
    main()
