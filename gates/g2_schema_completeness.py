#!/usr/bin/env python
"""Gate G2: schema completeness (literature-anchored) + round-trip + unit lint.

Pass conditions (plan §9, G2):
  1. Field coverage >= 95% of the literature-derived parameter superset
     (docs/literature/g2_parameter_superset.yaml), with every uncovered
     parameter listed and justified in schema/g2_field_mapping.yaml.
  2. JSON Schema meta-validation passes for both schemas.
  3. 100 randomly generated records survive a lossless
     serialize -> validate -> parse round-trip.
  4. Unit-consistency lint: every numeric leaf field embeds a unit suffix or
     is an explicitly whitelisted dimensionless quantity. Zero warnings.

Writes gates/reports/g2_report.json. Exit 0 = PASS.
Coverage check reports status "pending" until the literature survey files
exist; the gate FAILS while pending (G2 cannot pass without the survey).
"""

from __future__ import annotations

import datetime
import json
import sys
import uuid
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

COVERAGE_THRESHOLD = 0.95
N_ROUNDTRIP = 100
SEED = 20260806

UNIT_SUFFIXES = (
    "_ppm", "_K", "_atm", "_cm1", "_Hz", "_MHz", "_m", "_rad", "_amu",
    "_rel", "_per_s", "_per_atm", "_bits", "_mA", "_mW", "_cm3", "_s",
    "_per_min", "_per_h", "_cm_per_molec",
)
DIMENSIONLESS_WHITELIST = {
    "n_samples",          # sample count
    "random_seed",        # RNG seed
    "hitran_molecule_id", # HITRAN catalogue number
    "isotopologue_id",    # HITRAN catalogue number
    "adc_bits",           # bit count
    "n_air",              # dimensionless temperature exponent (HITRAN)
    "one_over_f_slope",   # dimensionless PSD slope
    "modulation_index",   # dimensionless ratio a/HWHM
    "averaging_n_scans",  # scan count
    "number_of_passes",   # pass count
    "snr_typical",        # dimensionless ratio
    "linearity_r2",       # dimensionless R^2
    "low", "high", "mean", "sigma", "values",  # distribution descriptors: unit
                                               # comes from the parent field
}


def _meta_validation() -> dict:
    from jsonschema import Draft202012Validator

    ok = True
    for name in ("record.schema.json", "instrument.schema.json"):
        Draft202012Validator.check_schema(json.loads((REPO / "schema" / name).read_text()))
    return {"pass": ok}


def _random_record(rng) -> dict:
    """Generate a random schema-valid record (DA or WMS)."""
    technique = "TDLAS-WMS" if rng.random() < 0.5 else "TDLAS-DA"
    n = int(rng.integers(500, 5000))
    rec = {
        "record_id": str(uuid.UUID(bytes=rng.bytes(16), version=4)),
        "schema_version": "0.1",
        "data_origin": "simulated",
        "technique": technique,
        "provenance": {
            "generator_version": "0.1.0.dev0",
            "hitran_data_version": "HITRAN2020",
            "random_seed": int(rng.integers(0, 2**31)),
            "instrument_config_id": f"vi-{int(rng.integers(0, 8)):02d}",
            "noise_config": {"white_noise_rel": float(rng.uniform(1e-5, 1e-3))},
        },
        "signals": {},
        "labels": {
            "species": [
                {
                    "molecule": "CH4",
                    "hitran_molecule_id": 6,
                    "concentration_ppm": float(rng.uniform(0.1, 1000.0)),
                    "concentration_uncertainty_ppm": 0.0,
                }
            ]
        },
        "conditions": {
            "temperature_K": float(rng.uniform(250.0, 350.0)),
            "pressure_atm": float(rng.uniform(0.1, 2.0)),
            "path_length_m": float(rng.uniform(0.1, 100.0)),
            "matrix_gas": "N2" if rng.random() < 0.5 else "air",
        },
        "instrument": {
            "laser": {
                "center_wavenumber_cm1": 6047.0,
                "scan_range_cm1": float(rng.uniform(1.0, 3.0)),
            },
            "detector": {"type": "InGaAs photodiode"},
            "target_lines": [{"hitran_molecule_id": 6, "wavenumber_cm1": 6046.9647}],
        },
    }
    if technique == "TDLAS-DA":
        rec["signals"]["absorbance"] = {
            "array_ref": f"/records/{rec['record_id']}/absorbance",
            "n_samples": n,
            "wavenumber_axis": {"start_cm1": 6046.0, "step_cm1": 0.001},
        }
    else:
        rec["instrument"]["modulation"] = {
            "frequency_Hz": float(rng.uniform(1e3, 5e4)),
            "depth_cm1": float(rng.uniform(0.01, 0.2)),
            "harmonic_scheme": "2f/1f" if rng.random() < 0.5 else "2f",
            "im_i0_rel": float(rng.uniform(0.05, 0.5)),
            "im_i2_rel": float(rng.uniform(0.0, 0.02)),
            "fm_im_phase1_rad": float(rng.uniform(-3.14, 3.14)),
            "fm_im_phase2_rad": float(rng.uniform(-3.14, 3.14)),
        }
        rec["signals"]["demod_2f"] = {
            "array_ref": f"/records/{rec['record_id']}/demod_2f",
            "n_samples": n,
        }
        if rng.random() < 0.5:
            rec["signals"]["demod_1f"] = {
                "array_ref": f"/records/{rec['record_id']}/demod_1f",
                "n_samples": n,
            }
    # Exercise wider schema surface (independent review: previous generator
    # touched roughly half the fields)
    if rng.random() < 0.5:
        rec["signals"]["raw_scan"] = {
            "array_ref": f"/records/{rec['record_id']}/raw_scan",
            "n_samples": n,
            "sampling_rate_Hz": float(rng.uniform(1e5, 1e8)),
        }
    if rng.random() < 0.4:
        rec["conditions"]["interferents"] = [
            {
                "molecule": "H2O",
                "hitran_molecule_id": 1,
                "concentration_ppm": float(rng.uniform(100.0, 20000.0)),
            }
        ]
    if rng.random() < 0.4:
        rec["labels"]["species"].append(
            {
                "molecule": "CO2",
                "hitran_molecule_id": 2,
                "concentration_ppm": float(rng.uniform(10.0, 5000.0)),
            }
        )
    if rng.random() < 0.5:
        rec["instrument"]["cell"] = {
            "type": "Herriott multipass",
            "number_of_passes": int(rng.integers(2, 200)),
            "volume_cm3": float(rng.uniform(10.0, 5000.0)),
        }
    if rng.random() < 0.5:
        rec["instrument"]["laser"].update(
            {
                "type": "DFB",
                "scan_waveform": "sawtooth",
                "operating_temperature_K": float(rng.uniform(283.0, 313.0)),
                "injection_current_mA": float(rng.uniform(20.0, 150.0)),
                "output_power_mW": float(rng.uniform(1.0, 40.0)),
                "tuning_poly_cm1": [float(v) for v in rng.normal(0.0, 0.1, 3)],
            }
        )
    if rng.random() < 0.5:
        rec["processing"] = {
            "calibration_method": "calibration-free (first-principles)",
            "baseline_treatment": "3rd-order polynomial on non-absorbing wings",
            "line_shape_model": "Voigt",
            "averaging_n_scans": int(rng.integers(1, 1000)),
            "averaging_time_s": float(rng.uniform(0.01, 100.0)),
        }
    if rng.random() < 0.3:
        rec["instrument"]["target_lines"][0].update(
            {
                "isotopologue_id": 1,
                "line_strength_cm_per_molec": 1.2e-21,
                "elower_cm1": 62.88,
                "gamma_air_cm1_per_atm": 0.06,
                "gamma_self_cm1_per_atm": 0.075,
                "n_air": 0.72,
                "delta_air_cm1_per_atm": -0.008,
            }
        )
    return rec


def _round_trip() -> dict:
    import numpy as np

    from opengasspec.validate import validate_record

    rng = np.random.default_rng(SEED)
    failures = []
    for i in range(N_ROUNDTRIP):
        rec = _random_record(rng)
        errors = validate_record(rec)
        if errors:
            failures.append({"i": i, "stage": "validate", "errors": errors[:3]})
            continue
        back = json.loads(json.dumps(rec))
        if back != rec:
            failures.append({"i": i, "stage": "roundtrip", "errors": ["lossy round-trip"]})
        elif validate_record(back):
            failures.append({"i": i, "stage": "revalidate", "errors": ["parsed copy invalid"]})
    return {
        "n_records": N_ROUNDTRIP,
        "seed": SEED,
        "failures": failures,
        "pass": not failures,
    }


def _iter_numeric_leaves(schema_node: dict, root: dict | None = None, path: str = ""):
    """Yield (field_path, field_name) for numeric-typed leaf properties.

    Resolves local ``$ref``s so that properties pointing at shared $defs —
    notably the instrument schema's ``dist_or_number`` — are linted under the
    referencing property's own name (independent review found the previous
    version skipped every $ref'd property).
    """
    root = root if root is not None else schema_node

    def resolve(node: dict) -> dict:
        ref = node.get("$ref", "")
        if ref.startswith("#/"):
            cur = root
            for seg in ref[2:].split("/"):
                cur = cur[seg]
            return cur
        return node

    props = schema_node.get("properties", {})
    for name, sub in props.items():
        p = f"{path}/{name}"
        rsub = resolve(sub)
        is_dist = sub.get("$ref", "").endswith("dist_or_number")
        if rsub.get("type") in ("number", "integer") or is_dist:
            yield p, name
        if not is_dist:  # dist internals (low/high/mean/...) inherit parent unit
            yield from _iter_numeric_leaves(rsub, root, p)
            items = rsub.get("items")
            if isinstance(items, dict):
                r_items = resolve(items)
                if r_items.get("type") in ("number", "integer") or items.get(
                    "$ref", ""
                ).endswith("dist_or_number"):
                    yield p + "[]", name
                else:
                    yield from _iter_numeric_leaves(r_items, root, p + "[]")
    for kw in ("allOf", "anyOf", "oneOf"):
        for sub in schema_node.get(kw, []) or []:
            yield from _iter_numeric_leaves(sub, root, path)


def _unit_lint() -> dict:
    warnings = []
    for schema_name in ("record.schema.json", "instrument.schema.json"):
        schema = json.loads((REPO / "schema" / schema_name).read_text())
        for fpath, fname in _iter_numeric_leaves(schema):
            if fname in DIMENSIONLESS_WHITELIST:
                continue
            if not any(fname.endswith(s) for s in UNIT_SUFFIXES):
                warnings.append(f"{schema_name}:{fpath} — no unit suffix")
    return {"warnings": warnings, "pass": not warnings}


def _resolve_ref(schema: dict, node: dict) -> dict:
    ref = node.get("$ref")
    if ref and ref.startswith("#/"):
        cur = schema
        for seg in ref[2:].split("/"):
            cur = cur[seg]
        return cur
    return node


def _field_path_exists(schemas: dict, spec: str) -> bool:
    """Check a mapping field path like 'R:/labels/species[]/molecule' exists."""
    prefix, _, path = spec.partition(":/")
    schema = schemas.get(prefix)
    if schema is None:
        return False
    node = schema
    for raw_seg in path.strip("/").split("/"):
        is_array = raw_seg.endswith("[]")
        seg = raw_seg[:-2] if is_array else raw_seg
        node = _resolve_ref(schema, node)
        props = node.get("properties", {})
        if seg not in props:
            return False
        node = _resolve_ref(schema, props[seg])
        if is_array:
            node = _resolve_ref(schema, node.get("items", {}))
    return True


def _coverage() -> dict:
    superset_path = REPO / "docs" / "literature" / "g2_parameter_superset.yaml"
    mapping_path = REPO / "schema" / "g2_field_mapping.yaml"
    if not superset_path.is_file() or not mapping_path.is_file():
        return {
            "status": "pending",
            "detail": "literature survey and/or field mapping not yet available",
            "pass": False,
        }
    superset = yaml.safe_load(superset_path.read_text())
    mapping = yaml.safe_load(mapping_path.read_text())
    schemas = {
        "R": json.loads((REPO / "schema" / "record.schema.json").read_text()),
        "I": json.loads((REPO / "schema" / "instrument.schema.json").read_text()),
    }
    params = superset["parameters"] if isinstance(superset, dict) else superset
    covered, excluded, unmapped, bad_paths = [], [], [], []
    for p in params:
        name = p["canonical_name"] if isinstance(p, dict) else p
        m = mapping.get("parameters", {}).get(name)
        if m is None:
            unmapped.append(name)
        elif m.get("status") == "covered":
            # A "covered" claim is only valid if every referenced field path
            # actually resolves inside the schemas.
            missing = [f for f in m.get("fields", []) if not _field_path_exists(schemas, f)]
            if missing or not m.get("fields"):
                bad_paths.append({"parameter": name, "missing_fields": missing})
            else:
                covered.append(name)
        elif m.get("status") == "excluded" and m.get("justification"):
            excluded.append(name)
        else:
            unmapped.append(name)
    total = len(covered) + len(excluded) + len(unmapped) + len(bad_paths)
    coverage = len(covered) / total if total else 0.0
    return {
        "n_parameters": total,
        "covered": len(covered),
        "excluded_with_justification": len(excluded),
        "unmapped": unmapped,
        "covered_claims_with_nonexistent_fields": bad_paths,
        "coverage_of_all": coverage,
        "threshold": COVERAGE_THRESHOLD,
        "pass": coverage >= COVERAGE_THRESHOLD and not unmapped and not bad_paths,
    }


def main() -> int:
    report = {
        "gate": "G2",
        "date": datetime.date.today().isoformat(),
        "checks": {
            "meta_validation": _meta_validation(),
            "round_trip_100": _round_trip(),
            "unit_lint": _unit_lint(),
            "literature_coverage": _coverage(),
        },
    }
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    out = REPO / "gates" / "reports" / "g2_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v.get("pass") for k, v in report["checks"].items()}, indent=2))
    print(f"Gate G2: {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
