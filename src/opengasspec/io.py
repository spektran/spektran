"""HDF5 persistence for generated datasets, with schema validation on write.

Layout of an OpenGasSpec HDF5 file:

    /records/<record_id>/<signal_name>   float64 arrays
    /records/<record_id>.attrs["meta"]   JSON record metadata (schema v0.1)
    /.attrs["opengasspec_version"], ["created_utc"], ["n_records"]

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
        f.attrs["opengasspec_version"] = __version__
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
