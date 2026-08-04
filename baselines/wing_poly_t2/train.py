#!/usr/bin/env python
"""Classical (non-ML) T2 denoising baseline: wing-anchored polynomial baseline.

The standard practitioner's method: fit a cubic polynomial to the
non-absorbing wings of each raw scan, take -log(raw/baseline) as the
absorbance estimate. No training, fully deterministic — this is the
'what a spectroscopist would do first' reference for the T2 task.

    python baselines/wing_poly_t2/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, load_split  # noqa: E402

OUT = Path(__file__).resolve().parent


def wing_poly_absorbance(raw: np.ndarray, wing_frac: float = 0.2, order: int = 3) -> np.ndarray:
    n = len(raw)
    idx = np.arange(n)
    wings = np.r_[0 : int(n * wing_frac), int(n * (1 - wing_frac)) : n]
    coeff = np.polyfit(wings, raw[wings], order)
    baseline = np.polyval(coeff, idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(np.clip(raw / baseline, 1e-9, None))


def main() -> int:
    splits = [("t1-test", "ch4-t1-test-v0"), ("t3-test-heldout", "ch4-t3-test-heldout-v0")]
    for tag, split in splits:
        X, _, ids = load_split(split)
        out_path = OUT / f"predictions_{tag}.h5"
        with h5py.File(out_path, "w") as f:
            grp = f.create_group("predictions")
            for rid, raw in zip(ids, X):
                grp.create_dataset(rid, data=wing_poly_absorbance(raw))
        print(f"{tag}: {len(ids)} predictions -> {out_path.name}")
    print("truth files:", DATA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
