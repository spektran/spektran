#!/usr/bin/env python
"""Push official OpenSensorSim splits to the Hugging Face Hub.

HUMAN ACTION REQUIRED before this works (plan §9, human responsibilities):
  1. Create the 'opensensorsim' organization on huggingface.co
  2. `pip install huggingface_hub datasets` and `hf auth login` with a
     write token (or set HF_TOKEN)

Usage:
    python scripts/push_to_hf.py [--repo opensensorsim/ch4-v0] [--private]

Converts the generated HDF5 splits under data/ into a datasets.DatasetDict
(raw_scan + labels + conditions + record_id) and pushes with the CC BY 4.0
license tag. ML users then need one line:

    load_dataset("opensensorsim/ch4-v0")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SPLIT_FILES = {
    "train": "ch4-t1-train-v0.h5",
    "validation": "ch4-t1-val-v0.h5",
    "test": "ch4-t1-test-v0.h5",
    "test_heldout_instrument": "ch4-t3-test-heldout-v0.h5",
}


def build_dataset_dict():
    from datasets import Dataset, DatasetDict

    from opensensorsim.io import read_records

    splits = {}
    for split, fname in SPLIT_FILES.items():
        path = REPO / "data" / fname
        if not path.is_file():
            raise SystemExit(
                f"{path} missing — generate splits first (see docs/quickstart.md)"
            )
        rows = []
        for r in read_records(path):
            meta = r["meta"]
            rows.append(
                {
                    "record_id": meta["record_id"],
                    "raw_scan": r["arrays"]["raw_scan"].tolist(),
                    "absorbance_clean": r["arrays"]["absorbance_clean"].tolist(),
                    "concentration_ppm": meta["labels"]["species"][0][
                        "concentration_ppm"
                    ],
                    "temperature_K": meta["conditions"]["temperature_K"],
                    "pressure_atm": meta["conditions"]["pressure_atm"],
                    "path_length_m": meta["conditions"]["path_length_m"],
                    "instrument_config_id": meta["provenance"]["instrument_config_id"],
                    "technique": meta["technique"],
                }
            )
        splits[split] = Dataset.from_list(rows)
    return DatasetDict(splits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="opensensorsim/ch4-v0")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    dd = build_dataset_dict()
    print({k: len(v) for k, v in dd.items()})
    dd.push_to_hub(args.repo, private=args.private)
    print(f"pushed to https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
