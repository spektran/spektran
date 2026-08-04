"""HDF5 persistence for generated datasets, with schema validation on write.

Layout of an SPEKTRAN HDF5 file:

    /records/<record_id>/<signal_name>   float64 arrays
    /records/<record_id>.attrs["meta"]   JSON record metadata (schema v0.1)
    /.attrs["spektran_version"], ["created_utc"], ["n_records"]

Parquet export for ML-friendly flat access ships with the benchmark layer.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from . import __version__
from .validate import validate_record


def write_records(path: str | Path, records: list[dict], validate: bool = True) -> None:
    """Write generated records ({'meta','arrays'}) to an HDF5 file."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["spektran_version"] = __version__
        f.attrs["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        f.attrs["n_records"] = len(records)
        grp = f.create_group("records")
        for rec in records:
            meta = rec["meta"]
            if validate:
                errors = validate_record(meta)
                if errors:
                    raise ValueError(
                        f"Record {meta.get('record_id')} fails schema: {errors[:3]}"
                    )
            g = grp.create_group(meta["record_id"])
            g.attrs["meta"] = json.dumps(meta)
            for name, arr in rec["arrays"].items():
                g.create_dataset(name, data=np.asarray(arr, dtype=np.float64))


def read_records(path: str | Path) -> list[dict]:
    """Read all records back as {'meta', 'arrays'} dicts."""
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        for rid in f["records"]:
            g = f["records"][rid]
            meta = json.loads(g.attrs["meta"])
            arrays = {name: g[name][()] for name in g}
            out.append({"meta": meta, "arrays": arrays})
    return out


def read_meta_index(path: str | Path) -> list[dict]:
    """Read only the metadata of every record (cheap index scan)."""
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        for rid in f["records"]:
            out.append(json.loads(f["records"][rid].attrs["meta"]))
    return out


def write_time_series(
    path: str | Path,
    records: list[dict],
    scan_interval_s: float,
    validate: bool = True,
) -> None:
    """Write a time series (temporally-ordered records) to HDF5.

    Adds time-series metadata beyond ``write_records``: ``scan_interval_s``
    and ``record_order`` (JSON list of record IDs in the exact order given --
    callers that concatenate several consecutive-scan runs, e.g. one per
    frozen instrument realization, must append them in generation order so
    downstream Allan-variance analysis can recover run boundaries).
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["spektran_version"] = __version__
        f.attrs["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        f.attrs["n_records"] = len(records)
        f.attrs["scan_interval_s"] = scan_interval_s
        f.attrs["record_order"] = json.dumps([r["meta"]["record_id"] for r in records])
        grp = f.create_group("records")
        for rec in records:
            meta = rec["meta"]
            if validate:
                errors = validate_record(meta)
                if errors:
                    raise ValueError(
                        f"Record {meta.get('record_id')} fails schema: {errors[:3]}"
                    )
            g = grp.create_group(meta["record_id"])
            g.attrs["meta"] = json.dumps(meta)
            for name, arr in rec["arrays"].items():
                g.create_dataset(name, data=np.asarray(arr, dtype=np.float64))


def read_time_series(path: str | Path) -> tuple[list[dict], float]:
    """Read a time-series HDF5, returning (records in temporal order, scan_interval_s)."""
    import h5py

    with h5py.File(path, "r") as f:
        order = json.loads(f.attrs["record_order"])
        scan_interval_s = float(f.attrs["scan_interval_s"])
        records = []
        for rid in order:
            g = f["records"][rid]
            meta = json.loads(g.attrs["meta"])
            arrays = {name: g[name][()] for name in g}
            records.append({"meta": meta, "arrays": arrays})
    return records, scan_interval_s
