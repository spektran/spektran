#!/usr/bin/env python
"""Generate an OpenGasSpec dataset from a dataset-config YAML.

Usage:
    python scripts/generate_dataset.py configs/datasets/<name>.yaml [--out DIR]

A dataset config fully determines the output (bit-for-bit): instrument
config, record count, master seed, gas-truth distribution. Example:

    dataset_id: ch4-da-medium-v0
    instrument_config: configs/instruments/vi-da-medium-02.yaml
    n_records: 10000
    master_seed: 20260810
    gas:
      molecule: CH4
      concentration_ppm: {low: 1.0, high: 1000.0, log_uniform: true}
      path_length_m: 10.0
      matrix_gas: N2
    n_points: 2000
    line_source: demo   # 'demo' (built-in approximate) or 'hitran' (hapi fetch)
    wavenumber_range_cm1: [6045.0, 6049.0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from opengasspec.generator import GenerationSpec, generate_dataset  # noqa: E402
from opengasspec.instrument.sampling import load_instrument_config  # noqa: E402
from opengasspec.io import write_records  # noqa: E402


def load_spec(cfg: dict) -> GenerationSpec:
    from opengasspec.physics import demo_ch4_2nu3, fetch_lines

    gas = cfg.get("gas", {})
    conc = gas.get("concentration_ppm", {})
    source = cfg.get("line_source", "demo")
    molecule = gas.get("molecule", "CH4")
    if source == "hitran":
        lo, hi = cfg.get("wavenumber_range_cm1", [6045.0, 6049.0])
        lines = fetch_lines(molecule, lo, hi)
    elif source == "demo":
        if molecule != "CH4":
            raise SystemExit("line_source: demo only provides CH4")
        lines = demo_ch4_2nu3()
    else:
        raise SystemExit(f"unknown line_source: {source}")
    return GenerationSpec(
        lines=lines,
        molecule=molecule,
        concentration_ppm_low=float(conc.get("low", 1.0)),
        concentration_ppm_high=float(conc.get("high", 1000.0)),
        log_uniform_concentration=bool(conc.get("log_uniform", True)),
        path_length_m=float(gas.get("path_length_m", 10.0)),
        matrix_gas=gas.get("matrix_gas", "N2"),
        n_points=int(cfg.get("n_points", 2000)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="dataset config YAML")
    ap.add_argument("--out", default="data", help="output directory (default: data/)")
    ap.add_argument("--n", type=int, default=None, help="override n_records")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    # 'instrument_config' may be one path or a list (records split evenly, each
    # sub-config with its own derived master seed for disjoint streams)
    inst_paths = cfg["instrument_config"]
    if isinstance(inst_paths, str):
        inst_paths = [inst_paths]
    instruments = [load_instrument_config(REPO / p) for p in inst_paths]
    spec = load_spec(cfg)
    n = args.n if args.n is not None else int(cfg["n_records"])
    seed = int(cfg["master_seed"])

    t0 = time.time()
    records = []
    per = n // len(instruments)
    counts = [per + (1 if i < n - per * len(instruments) else 0) for i in range(len(instruments))]
    for i, (inst, n_i) in enumerate(zip(instruments, counts)):
        records.extend(generate_dataset(spec, inst, n_i, seed + i))
    t1 = time.time()
    out_path = Path(args.out) / f"{cfg['dataset_id']}.h5"
    write_records(out_path, records, validate=True)
    t2 = time.time()
    print(
        f"{cfg['dataset_id']}: {n} records "
        f"(gen {t1 - t0:.1f}s, write+validate {t2 - t1:.1f}s) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
