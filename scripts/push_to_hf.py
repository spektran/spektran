#!/usr/bin/env python
"""Push official SPEKTRAN splits to the Hugging Face Hub.

Usage:
    python scripts/push_to_hf.py [--repo spektran/spektran-ch4-v0] [--private]

Converts generated HDF5 splits under data/ into multiple HF dataset configs
and pushes with the CC BY 4.0 license tag. ML users then need one line:

    load_dataset("spektran/spektran-ch4-v0", "da")         # T1/T3
    load_dataset("spektran/spektran-ch4-v0", "wms")        # T4
    load_dataset("spektran/spektran-ch4-v0", "drift")      # T5
    load_dataset("spektran/spektran-ch4-v0", "ood")        # T6
    load_dataset("spektran/spektran-ch4-v0", "multispecies")  # T8
    load_dataset("spektran/spektran-ch4-v0", "temperature")   # T9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CONFIGS = {
    "da": {
        "train": "ch4-t1-train-v0.h5",
        "validation": "ch4-t1-val-v0.h5",
        "test": "ch4-t1-test-v0.h5",
        "test_heldout_instrument": "ch4-t3-test-heldout-v0.h5",
    },
    "wms": {
        "t4_train": "ch4-t4-train-v0.h5",
        "t4_validation": "ch4-t4-val-v0.h5",
        "t4_test": "ch4-t4-test-v0.h5",
    },
    "drift": {
        "t5_train": "ch4-t5-train-v0.h5",
        "t5_test": "ch4-t5-test-v0.h5",
    },
    "ood": {
        "t6_train": "ch4-t6-train-v0.h5",
        "t6_test": "ch4-t6-test-v0.h5",
    },
    "multispecies": {
        "t8_train": "ch4-h2o-t8-train-v0.h5",
        "t8_test": "ch4-h2o-t8-test-v0.h5",
    },
    "temperature": {
        "t9_train": "ch4-t9-train-v0.h5",
        "t9_test": "ch4-t9-test-v0.h5",
    },
    "ndir": {
        "ndir_train": "ch4-ndir-train-v0.h5",
        "ndir_test": "ch4-ndir-test-v0.h5",
        "ndir_heldout": "ch4-ndir-test-heldout-v0.h5",
        "cross_modality_test": "ch4-cross-modality-test-v0.h5",
    },
    "da_hitran": {
        "train": "ch4-t1-train-v0-hitran.h5",
        "validation": "ch4-t1-val-v0-hitran.h5",
        "test": "ch4-t1-test-v0-hitran.h5",
        "test_heldout_instrument": "ch4-t3-test-heldout-v0-hitran.h5",
    },
    "wms_hitran": {
        "t4_train": "ch4-t4-train-v0-hitran.h5",
        "t4_validation": "ch4-t4-val-v0-hitran.h5",
        "t4_test": "ch4-t4-test-v0-hitran.h5",
    },
    "da_large": {
        "train_50k": "ch4-t1-train-v0-50k.h5",
        "validation_5k": "ch4-t1-val-v0-5k.h5",
        "test_10k": "ch4-t1-test-v0-10k.h5",
    },
}


def _record_to_row(rec: dict, include_ood: bool = False) -> dict:
    meta = rec["meta"]
    arrays = rec["arrays"]
    species = meta["labels"]["species"]
    technique = meta.get("technique", "TDLAS")

    row = {
        "record_id": meta["record_id"],
        "concentration_ppm": species[0]["concentration_ppm"],
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

    if include_ood:
        row["ood_label"] = int(meta["labels"].get("ood_label", 0))

    if len(species) > 1:
        row["interferent_molecule"] = species[1]["molecule"]
        row["interferent_concentration_ppm"] = species[1]["concentration_ppm"]

    return row


def build_config_dataset(split_files: dict[str, str], include_ood: bool = False):
    from datasets import Dataset, DatasetDict

    from spektran.io import read_records

    splits = {}
    for split, fname in split_files.items():
        path = REPO / "data" / fname
        if not path.is_file():
            print(f"  skipping {split}: {path.name} not found")
            continue
        rows = [_record_to_row(r, include_ood=include_ood) for r in read_records(path)]
        splits[split] = Dataset.from_list(rows)
        print(f"  {split}: {len(rows)} records")
    return DatasetDict(splits) if splits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="spektran/spektran-ch4-v0")
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--configs",
        nargs="*",
        default=list(CONFIGS.keys()),
        help="Which configs to push (default: all)",
    )
    ap.add_argument("--create-pr", action="store_true", help="Push via PR instead of direct commit")
    args = ap.parse_args()

    for config_name in args.configs:
        if config_name not in CONFIGS:
            print(f"Unknown config: {config_name}")
            continue
        print(f"\n=== Config: {config_name} ===")
        dd = build_config_dataset(CONFIGS[config_name], include_ood=(config_name == "ood"))
        if dd is None:
            print(f"  no data files found, skipping")
            continue
        dd.push_to_hub(
            args.repo,
            config_name=config_name,
            private=args.private,
            create_pr=args.create_pr,
        )
        print(f"  pushed to https://huggingface.co/datasets/{args.repo} ({config_name})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
