#!/usr/bin/env python
"""Fetch and cache HITRAN line data for all official SPEKTRAN measurement regions.

Usage:
    python scripts/fetch_hitran.py [--cache-dir .hitran_cache]

Requires network access to hitran.org on first run; subsequent runs use the
local cache. Pin the cache directory in version control or CI artifacts to
freeze the line-data snapshot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from spektran.physics.hitran import fetch_lines  # noqa: E402

REGIONS = {
    "CH4": (6045.0, 6049.0),
    "H2O": (7183.0, 7192.0),
    "CO2": (4976.0, 4980.0),
    "CO": (2168.0, 2174.0),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cache-dir", default=".hitran_cache",
        help="directory for cached HITRAN data (default: .hitran_cache)",
    )
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for mol, (lo, hi) in REGIONS.items():
        print(f"Fetching {mol} [{lo:.1f} - {hi:.1f}] cm-1 ...")
        try:
            ll = fetch_lines(mol, lo, hi, cache_dir=str(cache))
            n = len(ll)
            print(f"  -> {n} lines")
            manifest[mol] = {"region_cm1": [lo, hi], "n_lines": n}
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            manifest[mol] = {"region_cm1": [lo, hi], "error": str(exc)}

    mpath = cache / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest written to {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
