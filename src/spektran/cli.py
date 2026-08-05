"""Command-line entry point: ``spektran <subcommand>``."""
from __future__ import annotations

import argparse
import sys

from . import __version__


def cmd_validate(args: argparse.Namespace) -> int:
    from .validate import main as validate_main
    return validate_main(args.files)


def cmd_generate(args: argparse.Namespace) -> int:
    import time
    from dataclasses import replace
    from pathlib import Path

    import numpy as np
    import yaml

    from .generator import (
        GenerationSpec,
        generate_dataset,
        generate_time_series,
        sample_concentration,
    )
    from .instrument.sampling import load_instrument_config
    from .io import write_records, write_time_series

    repo = Path(args.config).resolve().parents[0]
    while not (repo / "pyproject.toml").exists() and repo != repo.parent:
        repo = repo.parent
    if not (repo / "pyproject.toml").exists():
        # Config lives outside any repo checkout (e.g. a tmp config file) --
        # fall back to CWD, which by convention is the repo root when
        # invoking `spektran generate`.
        repo = Path.cwd()

    cfg = yaml.safe_load(Path(args.config).read_text())

    def _load_instruments(key: str) -> list[dict]:
        paths = cfg[key]
        if isinstance(paths, str):
            paths = [paths]
        return [load_instrument_config(repo / p) for p in paths]

    def _split_counts(n_total: int, n_instruments: int) -> list[int]:
        per = n_total // n_instruments
        return [per + (1 if i < n_total - per * n_instruments else 0)
                for i in range(n_instruments)]

    # T6 (ood_task) configs give two disjoint instrument pools instead of one
    # `instrument_config` list, so the standard load below is skipped for them.
    ood_task = bool(cfg.get("ood_task"))
    if ood_task:
        in_dist_instruments = _load_instruments("instrument_config_in_dist")
        ood_instruments = _load_instruments("instrument_config_ood")
    else:
        instruments = _load_instruments("instrument_config")

    from .physics import (
        demo_ch4_2nu3,
        demo_co,
        demo_co2,
        demo_h2o,
        demo_hcl,
        demo_hf,
        demo_nh3,
        demo_no,
        demo_no2,
        demo_so2,
        fetch_lines,
    )

    gas = cfg.get("gas", {})
    conc = gas.get("concentration_ppm", {})
    source = cfg.get("line_source", "demo")
    molecule = gas.get("molecule", "CH4")
    if source == "hitran":
        lo, hi = cfg.get("wavenumber_range_cm1", [6045.0, 6049.0])
        lines = fetch_lines(molecule, lo, hi)
    elif source == "demo":
        demo_fns = {
            "CH4": demo_ch4_2nu3, "H2O": demo_h2o, "CO2": demo_co2, "CO": demo_co,
            "NH3": demo_nh3, "NO": demo_no, "NO2": demo_no2,
            "SO2": demo_so2, "HCl": demo_hcl, "HF": demo_hf,
        }
        if molecule not in demo_fns:
            print(f"No demo lines for {molecule}", file=sys.stderr)
            return 1
        lines = demo_fns[molecule]()
    else:
        print(f"Unknown line_source: {source}", file=sys.stderr)
        return 1

    interferent_specs = []
    for interf_cfg in gas.get("interferents", cfg.get("interferents", [])):
        mol = interf_cfg["molecule"]
        if source == "demo":
            demo_fns_i = {
                "H2O": demo_h2o, "CO2": demo_co2, "CO": demo_co,
                "CH4": demo_ch4_2nu3, "NH3": demo_nh3, "NO": demo_no,
                "NO2": demo_no2, "SO2": demo_so2, "HCl": demo_hcl, "HF": demo_hf,
            }
            if mol not in demo_fns_i:
                print(f"No demo lines for interferent {mol}", file=sys.stderr)
                return 1
            i_lines = demo_fns_i[mol]()
        else:
            i_lo, i_hi = interf_cfg.get("wavenumber_range_cm1",
                                        cfg.get("wavenumber_range_cm1", [6045.0, 6049.0]))
            i_lines = fetch_lines(mol, i_lo, i_hi)
        conc_spec = interf_cfg["concentration_ppm"]
        if isinstance(conc_spec, dict):
            interferent_specs.append({
                "molecule": mol, "lines": i_lines,
                "concentration_ppm_low": float(conc_spec["low"]),
                "concentration_ppm_high": float(conc_spec["high"]),
                "log_uniform": bool(conc_spec.get("log_uniform", False)),
            })
        else:
            interferent_specs.append({
                "molecule": mol, "lines": i_lines,
                "concentration_ppm": float(conc_spec),
            })

    technique = cfg.get("technique")
    if technique is None:
        if ood_task:
            technique = in_dist_instruments[0].get("technique", "TDLAS-DA")
        else:
            technique = instruments[0].get("technique", "TDLAS-DA")

    seed = int(cfg["master_seed"])
    out_dir = Path(args.out)
    out_path = out_dir / f"{cfg['dataset_id']}.h5"

    if technique == "NDIR":
        from .ndir_generator import NDIRGenerationSpec, generate_ndir_dataset

        ndir_spec = NDIRGenerationSpec(
            lines=lines, molecule=molecule,
            concentration_ppm_low=float(conc.get("low", 1.0)),
            concentration_ppm_high=float(conc.get("high", 1000.0)),
            log_uniform_concentration=bool(conc.get("log_uniform", True)),
            path_length_m=float(gas.get("path_length_m", 10.0)),
            matrix_gas=gas.get("matrix_gas", "N2"),
            interferents=interferent_specs,
        )
        n = args.n if args.n else int(cfg["n_records"])
        t0 = time.time()
        records = []
        counts = _split_counts(n, len(instruments))
        for i, (inst, n_i) in enumerate(zip(instruments, counts)):
            records.extend(
                generate_ndir_dataset(ndir_spec, inst, n_i, seed + i)
            )
        t1 = time.time()
        write_records(out_path, records, validate=True)
        t2 = time.time()
        print(f"{cfg['dataset_id']}: {n} records "
              f"(gen {t1 - t0:.1f}s, write {t2 - t1:.1f}s) -> {out_path}")
        return 0

    spec = GenerationSpec(
        lines=lines, molecule=molecule,
        concentration_ppm_low=float(conc.get("low", 1.0)),
        concentration_ppm_high=float(conc.get("high", 1000.0)),
        log_uniform_concentration=bool(conc.get("log_uniform", True)),
        path_length_m=float(gas.get("path_length_m", 10.0)),
        matrix_gas=gas.get("matrix_gas", "N2"),
        n_points=int(cfg.get("n_points", 2000)),
        interferents=interferent_specs,
    )

    if cfg.get("mode") == "time_series":
        # Time-series mode (T5 drift compensation): one frozen instrument per
        # series, so the true concentration is fixed for every scan in that
        # series (drift shows up in the MEASURED value, not the truth). Only
        # a single instrument config is meaningful here -- extra entries
        # would silently apply to no series.
        n_series = int(cfg.get("n_series", 1))
        n_scans = int(cfg["n_scans_per_series"])
        interval = float(cfg.get("scan_interval_s", 1.0))
        inst = instruments[0]

        t0 = time.time()
        series_rng = np.random.default_rng(seed)
        all_records = []
        for s in range(n_series):
            c = sample_concentration(spec, series_rng)
            series_spec = replace(
                spec,
                concentration_ppm_low=c,
                concentration_ppm_high=c,
                log_uniform_concentration=False,
            )
            all_records.extend(
                generate_time_series(series_spec, inst, n_scans, seed + s + 1, interval)
            )
        t1 = time.time()

        write_time_series(out_path, all_records, interval)
        t2 = time.time()
        print(f"{cfg['dataset_id']}: {n_series} series x {n_scans} scans "
              f"(gen {t1 - t0:.1f}s, write {t2 - t1:.1f}s) -> {out_path}")
        return 0

    if ood_task:
        # generate_record/generate_dataset are OOD-agnostic -- T6's label is a
        # property of which instrument pool produced a scan, not of the
        # physics, so it is stamped onto each record's metadata afterward.
        n_in_dist = int(cfg["n_records_in_dist"])
        n_ood = int(cfg["n_records_ood"])

        t0 = time.time()
        records = []
        seed_i = 0
        counts_in = _split_counts(n_in_dist, len(in_dist_instruments))
        for inst, n_i in zip(in_dist_instruments, counts_in):
            recs = generate_dataset(spec, inst, n_i, seed + seed_i)
            for r in recs:
                r["meta"]["labels"]["ood_label"] = 0
            records.extend(recs)
            seed_i += 1
        counts_ood = _split_counts(n_ood, len(ood_instruments))
        for inst, n_i in zip(ood_instruments, counts_ood):
            recs = generate_dataset(spec, inst, n_i, seed + seed_i)
            for r in recs:
                r["meta"]["labels"]["ood_label"] = 1
            records.extend(recs)
            seed_i += 1
        t1 = time.time()

        write_records(out_path, records, validate=True)
        t2 = time.time()
        print(f"{cfg['dataset_id']}: {len(records)} records "
              f"({n_in_dist} in-dist + {n_ood} ood) "
              f"(gen {t1 - t0:.1f}s, write {t2 - t1:.1f}s) -> {out_path}")
        return 0

    n = args.n if args.n else int(cfg["n_records"])

    t0 = time.time()
    records = []
    counts = _split_counts(n, len(instruments))
    for i, (inst, n_i) in enumerate(zip(instruments, counts)):
        records.extend(generate_dataset(spec, inst, n_i, seed + i))
    t1 = time.time()

    write_records(out_path, records, validate=True)
    t2 = time.time()
    print(f"{cfg['dataset_id']}: {n} records "
          f"(gen {t1 - t0:.1f}s, write {t2 - t1:.1f}s) -> {out_path}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark.evaluate import main as eval_main
    return eval_main(args.eval_args)


def cmd_download(args: argparse.Namespace) -> int:
    print("spektran download: fetches pre-built datasets from Hugging Face.")
    print("Usage: pip install datasets && python -c "
          "\"from datasets import load_dataset; "
          "ds = load_dataset('spektran/spektran-ch4-v0')\"")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spektran", description="SPEKTRAN CLI")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_val = sub.add_parser("validate", help="validate records against the schema")
    p_val.add_argument("files", nargs="+")

    p_gen = sub.add_parser("generate", help="generate a dataset from a YAML config")
    p_gen.add_argument("config", help="dataset config YAML")
    p_gen.add_argument("--out", default="data", help="output directory")
    p_gen.add_argument("--n", type=int, default=None, help="override n_records")

    p_bench = sub.add_parser("benchmark", help="run benchmark evaluation")
    p_bench.add_argument("eval_args", nargs=argparse.REMAINDER,
                         help="args passed to evaluate.py (--task, --truth, --predictions)")

    sub.add_parser("download", help="show download instructions for pre-built datasets")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    handlers = {
        "validate": cmd_validate,
        "generate": cmd_generate,
        "benchmark": cmd_benchmark,
        "download": cmd_download,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
