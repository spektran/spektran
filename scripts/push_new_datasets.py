#!/usr/bin/env python
"""Push new molecule datasets to Hugging Face Hub.

Creates three new dataset repos:
  - spektran/spektran-co2-v0     (CO2 benchmark: DA + WMS)
  - spektran/spektran-industrial-v0  (SO2 + NO + CO)
  - spektran/spektran-multigas-v0    (CH4+CO2+H2O, CO+CO2 mixtures)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA_DIR = REPO / "data_export"

REPOS = {
    "spektran/spektran-co2-v0": {
        "da": {
            "train": "co2-t1-train-v0.h5",
            "validation": "co2-t1-val-v0.h5",
            "test": "co2-t1-test-v0.h5",
            "test_heldout_instrument": "co2-t3-test-heldout-v0.h5",
        },
        "wms": {
            "train": "co2-t4-train-v0.h5",
            "validation": "co2-t4-val-v0.h5",
            "test": "co2-t4-test-v0.h5",
        },
    },
    "spektran/spektran-industrial-v0": {
        "so2": {
            "train": "so2-t1-train-v0.h5",
            "test": "so2-t1-test-v0.h5",
        },
        "no": {
            "train": "no-t1-train-v0.h5",
            "test": "no-t1-test-v0.h5",
        },
        "co": {
            "train": "co-t1-train-v0.h5",
            "test": "co-t1-test-v0.h5",
        },
    },
    "spektran/spektran-multigas-v0": {
        "ch4_co2_h2o": {
            "train": "ch4-co2-h2o-train-v0.h5",
            "test": "ch4-co2-h2o-test-v0.h5",
        },
        "co_co2": {
            "train": "co-co2-train-v0.h5",
            "test": "co-co2-test-v0.h5",
        },
    },
}


def _record_to_row(rec: dict) -> dict:
    meta = rec["meta"]
    arrays = rec["arrays"]
    species = meta["labels"]["species"]
    technique = meta.get("technique", "TDLAS")

    row = {
        "record_id": meta["record_id"],
        "concentration_ppm": species[0]["concentration_ppm"],
        "molecule": species[0]["molecule"],
        "temperature_K": meta["conditions"]["temperature_K"],
        "pressure_atm": meta["conditions"]["pressure_atm"],
        "path_length_m": meta["conditions"]["path_length_m"],
        "instrument_config_id": meta["provenance"]["instrument_config_id"],
        "technique": technique,
    }

    if technique == "NDIR":
        row["ratio"] = float(arrays["ratio"])
    else:
        row["raw_scan"] = arrays["raw_scan"].tolist()
        row["absorbance_clean"] = arrays["absorbance_clean"].tolist()

    if "demod_1f" in arrays:
        row["demod_1f"] = arrays["demod_1f"].tolist()
    if "demod_2f" in arrays:
        row["demod_2f"] = arrays["demod_2f"].tolist()

    for i, sp in enumerate(species[1:], 1):
        row[f"interferent_{i}_molecule"] = sp["molecule"]
        row[f"interferent_{i}_concentration_ppm"] = sp["concentration_ppm"]

    return row


def build_config_dataset(split_files: dict[str, str]):
    from datasets import Dataset, DatasetDict

    from spektran.io import read_records

    splits = {}
    for split, fname in split_files.items():
        path = DATA_DIR / fname
        if not path.is_file():
            print(f"  skipping {split}: {path.name} not found")
            continue
        rows = [_record_to_row(r) for r in read_records(path)]
        splits[split] = Dataset.from_list(rows)
        print(f"  {split}: {len(rows)} records")
    return DatasetDict(splits) if splits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", nargs="*", default=list(REPOS.keys()),
                     help="Which repos to push (default: all)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    for repo_id in args.repos:
        if repo_id not in REPOS:
            print(f"Unknown repo: {repo_id}")
            continue

        print(f"\n{'='*60}")
        print(f"REPO: {repo_id}")
        print(f"{'='*60}")

        configs = REPOS[repo_id]
        for config_name, split_files in configs.items():
            print(f"\n--- Config: {config_name} ---")
            dd = build_config_dataset(split_files)
            if dd is None:
                print(f"  no data files found, skipping")
                continue
            dd.push_to_hub(
                repo_id,
                config_name=config_name,
                private=args.private,
            )
            print(f"  pushed to https://huggingface.co/datasets/{repo_id} ({config_name})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
