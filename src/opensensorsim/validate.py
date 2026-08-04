"""Record/config validation against the OpenSensorSim JSON Schemas.

Used by the CLI (``opensensorsim validate``), by CI, and by external data
contributors. Unit-consistency linting beyond structural validation lands
with Gate G2.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator

# Repo layout: <root>/src/opensensorsim/validate.py -> <root>/schema/
_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"


def _load_schema(name: str) -> dict:
    # Installed-package location first, repo layout as fallback for editable installs
    try:
        ref = resources.files("opensensorsim").joinpath(f"schema/{name}")
        if ref.is_file():
            return json.loads(ref.read_text())
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    path = _SCHEMA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Schema {name} not found (looked in package and {path})")
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def record_validator() -> Draft202012Validator:
    schema = _load_schema("record.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@lru_cache(maxsize=None)
def instrument_validator() -> Draft202012Validator:
    schema = _load_schema("instrument.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_record(record: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    v = record_validator()
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in v.iter_errors(record)
    ]


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m opensensorsim.validate RECORD.json [...]", file=sys.stderr)
        return 2
    n_bad = 0
    for arg in args:
        p = Path(arg)
        files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
        for f in files:
            errors = validate_record(json.loads(f.read_text()))
            if errors:
                n_bad += 1
                print(f"FAIL {f}")
                for e in errors:
                    print(f"  - {e}")
            else:
                print(f"OK   {f}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
