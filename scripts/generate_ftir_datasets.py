#!/usr/bin/env python
"""Generate all FTIR benchmark datasets (train/val/test/heldout)."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spektran.ftir_generator import FTIRGenerationSpec, generate_ftir_dataset
from spektran.instrument.sampling import load_instrument_config
from spektran.io import write_records
from spektran.physics.hitran import demo_ch4_2nu3

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
DATA.mkdir(exist_ok=True)


def generate_from_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_id = cfg["dataset_id"]
    out_path = DATA / f"{dataset_id}.h5"

    if out_path.exists():
        print(f"  {dataset_id}: already exists, skipping")
        return

    inst_configs = cfg["instrument_config"]
    if isinstance(inst_configs, str):
        inst_configs = [inst_configs]

    loaded_insts = []
    for ic_path in inst_configs:
        full_path = REPO / ic_path
        loaded_insts.append(load_instrument_config(full_path))

    lines = demo_ch4_2nu3()
    gas = cfg.get("gas", {})

    spec = FTIRGenerationSpec(
        lines=lines,
        molecule=gas.get("molecule", "CH4"),
        concentration_ppm_low=gas.get("concentration_ppm", {}).get("low", 0.5),
        concentration_ppm_high=gas.get("concentration_ppm", {}).get("high", 500.0),
        log_uniform_concentration=gas.get("concentration_ppm", {}).get("log_uniform", True),
        path_length_m=gas.get("path_length_m", 10.0),
        matrix_gas=gas.get("matrix_gas", "N2"),
        wavenumber_start_cm1=gas.get("wavenumber_start_cm1", 6000.0),
        wavenumber_end_cm1=gas.get("wavenumber_end_cm1", 6100.0),
        n_output_points=gas.get("n_output_points", 500),
    )

    n_records = cfg["n_records"]
    master_seed = cfg["master_seed"]
    n_per_inst = n_records // len(loaded_insts)
    remainder = n_records - n_per_inst * len(loaded_insts)

    all_records = []
    seed_offset = 0
    for i, inst_cfg in enumerate(loaded_insts):
        n = n_per_inst + (1 if i < remainder else 0)
        records = generate_ftir_dataset(
            spec, inst_cfg, n, master_seed + seed_offset,
        )
        all_records.extend(records)
        seed_offset += n

    print(f"  {dataset_id}: writing {len(all_records)} records to {out_path.name}")
    write_records(out_path, all_records, validate=True)
    print(f"  {dataset_id}: done")


def main():
    configs = [
        "ch4-ftir-t1-train-v0",
        "ch4-ftir-t1-val-v0",
        "ch4-ftir-t1-test-v0",
        "ch4-ftir-t3-heldout-v0",
    ]
    for name in configs:
        config_path = REPO / "configs" / "datasets" / f"{name}.yaml"
        print(f"Generating {name}...")
        generate_from_config(config_path)


if __name__ == "__main__":
    main()
