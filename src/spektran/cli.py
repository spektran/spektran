"""Command-line entry point: ``spektran <subcommand>``.

Agent-ready CLI — every command supports ``--json`` for structured output.
Discovery: ``spektran info``, ``spektran list {tasks,baselines,instruments,datasets}``.
Pipeline:  ``spektran generate``, ``spektran train``, ``spektran benchmark``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_repo(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    while not (p / "pyproject.toml").exists() and p != p.parent:
        p = p.parent
    return p if (p / "pyproject.toml").exists() else Path.cwd()


def _load_registry(repo: Path) -> dict:
    import yaml
    reg_path = repo / "baselines" / "registry.yaml"
    if not reg_path.exists():
        return {"baselines": {}}
    return yaml.safe_load(reg_path.read_text())


def _data_status(repo: Path, data_dir: Path) -> dict[str, dict]:
    import yaml
    configs_dir = repo / "configs" / "datasets"
    status = {}
    if not configs_dir.exists():
        return status
    for p in sorted(configs_dir.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        ds_id = cfg.get("dataset_id", p.stem)
        h5 = data_dir / f"{ds_id}.h5"
        entry: dict = {"config": str(p.relative_to(repo)), "exists": h5.exists()}
        if h5.exists():
            entry["path"] = str(h5)
            entry["size_mb"] = round(h5.stat().st_size / 1e6, 1)
        status[ds_id] = entry
    return status


def _load_scores(repo: Path, baseline_dir: str, score_file: str) -> dict | None:
    p = repo / "baselines" / baseline_dir / score_file
    if p.exists():
        return json.loads(p.read_text())
    return None


def _json_or_print(data, use_json: bool) -> None:
    if use_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        _pretty_print(data)


def _pretty_print(data, indent: int = 0) -> None:
    prefix = "  " * indent
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _pretty_print(item, indent)
                print()
            else:
                print(f"{prefix}- {item}")
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}:")
                _pretty_print(v, indent + 1)
            else:
                print(f"{prefix}{k}: {v}")
    else:
        print(f"{prefix}{data}")


# ---------------------------------------------------------------------------
# cmd_info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    from .benchmark.tasks import TASK_SPECS

    repo = _find_repo()
    registry = _load_registry(repo)
    data_dir = Path(args.data_dir) if hasattr(args, "data_dir") else repo / "data"

    tasks_summary = []
    for tid, spec in TASK_SPECS.items():
        tasks_summary.append({
            "id": tid,
            "primary_metric": spec.primary_metric,
            "input": spec.input_signal,
            "target": spec.target,
        })

    baselines_summary = []
    for name, bl in registry.get("baselines", {}).items():
        baselines_summary.append({
            "name": name,
            "display_name": bl["display_name"],
            "tasks": list(bl.get("tasks", {}).keys()),
        })

    result = {
        "version": __version__,
        "repo": str(repo),
        "tasks": tasks_summary,
        "baselines": baselines_summary,
        "data_dir": str(data_dir),
        "data_generated": sum(
            1 for v in _data_status(repo, data_dir).values() if v["exists"]
        ),
        "data_total": len(_data_status(repo, data_dir)),
        "agent_hint": (
            "Use 'spektran list tasks --json' for task details, "
            "'spektran list baselines --json' for baseline details, "
            "'spektran train --baseline <name> --json' to train."
        ),
    }
    _json_or_print(result, getattr(args, "json", False))
    return 0


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    resource = args.resource
    handlers = {
        "tasks": _list_tasks,
        "baselines": _list_baselines,
        "instruments": _list_instruments,
        "datasets": _list_datasets,
    }
    if resource not in handlers:
        print(f"Unknown resource: {resource}. Choose from: {', '.join(handlers)}", file=sys.stderr)
        return 1
    return handlers[resource](args)


def _list_tasks(args: argparse.Namespace) -> int:
    from .benchmark.tasks import TASK_SPECS

    repo = _find_repo()
    registry = _load_registry(repo)

    baseline_map: dict[str, list[str]] = {}
    for name, bl in registry.get("baselines", {}).items():
        for task_key in bl.get("tasks", {}):
            baseline_map.setdefault(f"T{task_key}" if not task_key.startswith("T") else task_key, [])
            short_task = task_key.replace("T", "") if task_key.startswith("T") else task_key
            for full_id in TASK_SPECS:
                if full_id.startswith(f"T{short_task}-") or full_id.startswith(f"{task_key}-"):
                    baseline_map.setdefault(full_id, []).append(name)

    result = []
    for tid, spec in TASK_SPECS.items():
        entry = {
            "id": tid,
            "input_signal": spec.input_signal,
            "target": spec.target,
            "primary_metric": spec.primary_metric,
            "secondary_metrics": spec.secondary_metrics,
            "available_baselines": sorted(set(baseline_map.get(tid, []))),
        }
        result.append(entry)

    _json_or_print(result, getattr(args, "json", False))
    return 0


def _list_baselines(args: argparse.Namespace) -> int:
    repo = _find_repo()
    registry = _load_registry(repo)

    result = []
    for name, bl in registry.get("baselines", {}).items():
        entry = {
            "name": name,
            "display_name": bl["display_name"],
            "description": bl.get("description", ""),
            "directory": f"baselines/{bl['directory']}",
            "tasks": list(bl.get("tasks", {}).keys()),
        }
        scores = {}
        for task_key, task_cfg in bl.get("tasks", {}).items():
            for score_task, score_file in task_cfg.get("scores", {}).items():
                s = _load_scores(repo, bl["directory"], score_file)
                if s:
                    scores[score_task] = s
        if scores:
            entry["scores"] = scores
        result.append(entry)

    _json_or_print(result, getattr(args, "json", False))
    return 0


def _list_instruments(args: argparse.Namespace) -> int:
    import yaml

    repo = _find_repo()
    inst_dir = repo / "configs" / "instruments"
    result = []
    for p in sorted(inst_dir.glob("vi-*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        parts = p.stem.split("-")
        tier = next((t for t in parts if t in ("easy", "medium", "hard", "heldout")), "unknown")
        entry = {
            "name": p.stem,
            "config": str(p.relative_to(repo)),
            "technique": cfg.get("technique", "TDLAS-DA"),
            "tier": tier,
            "held_out": cfg.get("held_out", False),
        }
        result.append(entry)

    _json_or_print(result, getattr(args, "json", False))
    return 0


def _list_datasets(args: argparse.Namespace) -> int:
    repo = _find_repo()
    data_dir = Path(args.data_dir) if hasattr(args, "data_dir") else repo / "data"
    status = _data_status(repo, data_dir)

    result = []
    for ds_id, info in status.items():
        result.append({"dataset_id": ds_id, **info})

    _json_or_print(result, getattr(args, "json", False))
    return 0


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    repo = _find_repo()
    data_dir = Path(args.data_dir) if hasattr(args, "data_dir") else repo / "data"
    registry = _load_registry(repo)

    data = _data_status(repo, data_dir)
    baselines_status = {}
    for name, bl in registry.get("baselines", {}).items():
        bl_dir = repo / "baselines" / bl["directory"]
        predictions = sorted(str(p.name) for p in bl_dir.glob("predictions_*"))
        scores = {}
        for task_key, task_cfg in bl.get("tasks", {}).items():
            for score_task, score_file in task_cfg.get("scores", {}).items():
                s = _load_scores(repo, bl["directory"], score_file)
                if s:
                    scores[score_task] = s
        baselines_status[name] = {
            "trained": len(predictions) > 0,
            "predictions": predictions,
            "scores": scores,
        }

    result = {"data": data, "baselines": baselines_status}
    _json_or_print(result, getattr(args, "json", False))
    return 0


# ---------------------------------------------------------------------------
# cmd_train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    repo = _find_repo()
    registry = _load_registry(repo)
    data_dir = Path(args.data_dir) if hasattr(args, "data_dir") else repo / "data"
    use_json = getattr(args, "json", False)

    bl_name = args.baseline
    baselines = registry.get("baselines", {})
    if bl_name not in baselines:
        available = ", ".join(sorted(baselines.keys()))
        msg = f"Unknown baseline '{bl_name}'. Available: {available}"
        if use_json:
            print(json.dumps({"error": msg, "code": 1}))
        else:
            print(msg, file=sys.stderr)
        return 1

    bl = baselines[bl_name]
    task_key = args.task
    if task_key and task_key not in bl.get("tasks", {}):
        available = ", ".join(bl.get("tasks", {}).keys())
        msg = f"Baseline '{bl_name}' does not support task '{task_key}'. Supported: {available}"
        if use_json:
            print(json.dumps({"error": msg, "code": 1}))
        else:
            print(msg, file=sys.stderr)
        return 1

    tasks_to_run = {task_key: bl["tasks"][task_key]} if task_key else bl.get("tasks", {})

    all_datasets: set[str] = set()
    for tcfg in tasks_to_run.values():
        all_datasets.update(tcfg.get("datasets", []))

    generated = []
    for ds_id in sorted(all_datasets):
        h5 = data_dir / f"{ds_id}.h5"
        if h5.exists():
            continue
        cfg_path = repo / "configs" / "datasets" / f"{ds_id}.yaml"
        if not cfg_path.exists():
            msg = f"Dataset config not found: {cfg_path}"
            if use_json:
                print(json.dumps({"error": msg, "code": 1}))
            else:
                print(msg, file=sys.stderr)
            return 1
        if not use_json:
            print(f"Generating {ds_id}...")
        gen_argv = ["generate", str(cfg_path), "--out", str(data_dir)]
        rc = cmd_generate(argparse.Namespace(
            config=str(cfg_path), out=str(data_dir), n=None, json=False,
        ))
        if rc != 0:
            return rc
        generated.append(ds_id)

    for tk, tcfg in tasks_to_run.items():
        script = tcfg.get("script", "train.py")
        script_path = repo / "baselines" / bl["directory"] / script
        if not script_path.exists():
            msg = f"Training script not found: {script_path}"
            if use_json:
                print(json.dumps({"error": msg, "code": 1}))
            else:
                print(msg, file=sys.stderr)
            return 1

        if not use_json:
            print(f"Training {bl_name} on {tk}...")

        env = os.environ.copy()
        env["SPEKTRAN_DATA_DIR"] = str(data_dir)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(repo),
            env=env,
            capture_output=use_json,
            text=True,
        )
        if result.returncode != 0:
            msg = f"Training failed (exit {result.returncode})"
            if use_json:
                print(json.dumps({
                    "error": msg, "code": result.returncode,
                    "stderr": result.stderr[:500] if result.stderr else "",
                }))
            else:
                print(msg, file=sys.stderr)
            return result.returncode

    scores = {}
    for tk, tcfg in tasks_to_run.items():
        for score_task, score_file in tcfg.get("scores", {}).items():
            s = _load_scores(repo, bl["directory"], score_file)
            if s:
                scores[score_task] = s

    output = {
        "baseline": bl_name,
        "display_name": bl["display_name"],
        "tasks_trained": list(tasks_to_run.keys()),
        "data_generated": generated,
        "scores": scores,
    }
    if use_json:
        print(json.dumps(output, indent=2))
    else:
        print(f"\nDone. Baseline '{bl_name}' trained.")
        if scores:
            print("Scores:")
            for task, s in scores.items():
                metrics = ", ".join(f"{k}: {v}" for k, v in s.items() if isinstance(v, (int, float)))
                print(f"  {task}: {metrics}")
    return 0


# ---------------------------------------------------------------------------
# cmd_validate (unchanged)
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    from .validate import main as validate_main
    return validate_main(args.files)


# ---------------------------------------------------------------------------
# cmd_generate (added --json support)
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> int:
    import time
    from dataclasses import replace

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
        repo = Path.cwd()

    cfg = yaml.safe_load(Path(args.config).read_text())
    use_json = getattr(args, "json", False)

    def _load_instruments(key: str) -> list[dict]:
        paths = cfg[key]
        if isinstance(paths, str):
            paths = [paths]
        return [load_instrument_config(repo / p) for p in paths]

    def _split_counts(n_total: int, n_instruments: int) -> list[int]:
        per = n_total // n_instruments
        return [per + (1 if i < n_total - per * n_instruments else 0)
                for i in range(n_instruments)]

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
            msg = f"No demo lines for {molecule}"
            if use_json:
                print(json.dumps({"error": msg, "code": 1}))
            else:
                print(msg, file=sys.stderr)
            return 1
        lines = demo_fns[molecule]()
    else:
        msg = f"Unknown line_source: {source}"
        if use_json:
            print(json.dumps({"error": msg, "code": 1}))
        else:
            print(msg, file=sys.stderr)
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
                msg = f"No demo lines for interferent {mol}"
                if use_json:
                    print(json.dumps({"error": msg, "code": 1}))
                else:
                    print(msg, file=sys.stderr)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['dataset_id']}.h5"

    def _report(dataset_id: str, n: int, t_gen: float, t_write: float) -> None:
        if use_json:
            print(json.dumps({
                "dataset_id": dataset_id, "n_records": n,
                "path": str(out_path),
                "generation_s": round(t_gen, 2), "write_s": round(t_write, 2),
            }))
        else:
            print(f"{dataset_id}: {n} records "
                  f"(gen {t_gen:.1f}s, write {t_write:.1f}s) -> {out_path}")

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
            records.extend(generate_ndir_dataset(ndir_spec, inst, n_i, seed + i))
        t1 = time.time()
        write_records(out_path, records, validate=True)
        t2 = time.time()
        _report(cfg["dataset_id"], n, t1 - t0, t2 - t1)
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
        n_total = n_series * n_scans
        _report(cfg["dataset_id"], n_total, t1 - t0, t2 - t1)
        return 0

    if ood_task:
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
        _report(cfg["dataset_id"], len(records), t1 - t0, t2 - t1)
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
    _report(cfg["dataset_id"], n, t1 - t0, t2 - t1)
    return 0


# ---------------------------------------------------------------------------
# cmd_benchmark (unchanged)
# ---------------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark.evaluate import main as eval_main
    return eval_main(args.eval_args)


# ---------------------------------------------------------------------------
# cmd_download (unchanged)
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> int:
    print("spektran download: fetches pre-built datasets from Hugging Face.\n")
    print("pip install datasets\n")
    configs = [
        ("da", "T1/T3 concentration + generalization (TDLAS DA)"),
        ("wms", "T4 WMS concentration"),
        ("drift", "T5 drift compensation (time series)"),
        ("ood", "T6 OOD instrument detection"),
        ("ndir", "T7 NDIR + cross-modality transfer"),
        ("multispecies", "T8 multi-species regression (CH4+H2O)"),
        ("temperature", "T9 temperature regression"),
        ("da_hitran", "T1/T3 with HITRAN production lines (76 lines)"),
        ("wms_hitran", "T4 with HITRAN production lines"),
        ("da_large", "T1 large-scale (50K train / 5K val / 10K test)"),
    ]
    print("Available configs:")
    for name, desc in configs:
        print(f"  {name:15s} {desc}")
    print("\nExample:")
    print('  from datasets import load_dataset')
    print('  ds = load_dataset("spektran/spektran-ch4-v0", "da")')
    return 0


# ---------------------------------------------------------------------------
# main — argument parser
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spektran",
        description="SPEKTRAN — TDLAS/NDIR simulation engine and ML benchmark. "
                    "Agent-ready: use --json on any command for structured output.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    # --- info ---
    p_info = sub.add_parser("info", help="project overview (agent discovery)")
    p_info.add_argument("--json", action="store_true", help="JSON output")
    p_info.add_argument("--data-dir", default="data", help="data directory")

    # --- list ---
    p_list = sub.add_parser(
        "list", help="list {tasks, baselines, instruments, datasets}",
    )
    p_list.add_argument(
        "resource", choices=["tasks", "baselines", "instruments", "datasets"],
        help="resource type to list",
    )
    p_list.add_argument("--json", action="store_true", help="JSON output")
    p_list.add_argument("--data-dir", default="data", help="data directory")

    # --- status ---
    p_status = sub.add_parser("status", help="data and baseline training status")
    p_status.add_argument("--json", action="store_true", help="JSON output")
    p_status.add_argument("--data-dir", default="data", help="data directory")

    # --- train ---
    p_train = sub.add_parser("train", help="train a baseline (auto-generates data)")
    p_train.add_argument("--baseline", required=True, help="baseline name (from registry)")
    p_train.add_argument("--task", default=None, help="specific task (default: all tasks for baseline)")
    p_train.add_argument("--data-dir", default="data", help="data directory")
    p_train.add_argument("--json", action="store_true", help="JSON output")

    # --- generate ---
    p_gen = sub.add_parser("generate", help="generate a dataset from a YAML config")
    p_gen.add_argument("config", help="dataset config YAML")
    p_gen.add_argument("--out", default="data", help="output directory")
    p_gen.add_argument("--n", type=int, default=None, help="override n_records")
    p_gen.add_argument("--json", action="store_true", help="JSON output")

    # --- validate ---
    p_val = sub.add_parser("validate", help="validate records against the schema")
    p_val.add_argument("files", nargs="+")

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="run benchmark evaluation")
    p_bench.add_argument("eval_args", nargs=argparse.REMAINDER,
                         help="args passed to evaluate.py")

    # --- download ---
    sub.add_parser("download", help="show download instructions for pre-built datasets")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "info": cmd_info,
        "list": cmd_list,
        "status": cmd_status,
        "train": cmd_train,
        "validate": cmd_validate,
        "generate": cmd_generate,
        "benchmark": cmd_benchmark,
        "download": cmd_download,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
