"""One-command evaluation: predictions file + truth HDF5 -> metric scores.

Usage (also exposed as ``python -m spektran.benchmark.evaluate``):

    evaluate.py --task T1-concentration --truth data/test.h5 \
                --predictions preds.csv [--json-out scores.json]

Prediction formats:
- T1/T3/T4: CSV with header ``record_id,concentration_ppm``
- T2: HDF5 with /predictions/<record_id> arrays (denoised absorbance)
- T5/T6: not yet implemented (raises NotImplementedError; see docs/benchmark.md)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ..io import read_records
from . import metrics as M


def _truth_concentrations(records: list[dict]) -> dict[str, float]:
    return {
        r["meta"]["record_id"]: r["meta"]["labels"]["species"][0]["concentration_ppm"]
        for r in records
    }


def evaluate_concentration(truth_h5: Path, predictions_csv: Path) -> dict:
    records = read_records(truth_h5)
    truth = _truth_concentrations(records)
    preds: dict[str, float] = {}
    with open(predictions_csv) as f:
        for row in csv.DictReader(f):
            preds[row["record_id"]] = float(row["concentration_ppm"])
    missing = sorted(set(truth) - set(preds))
    if missing:
        raise SystemExit(
            f"predictions missing {len(missing)} record_ids (first: {missing[:3]})"
        )
    ids = sorted(truth)
    y_true = np.array([truth[i] for i in ids])
    y_pred = np.array([preds[i] for i in ids])
    return {
        "n_records": len(ids),
        "mae_ppm": M.mae(y_true, y_pred),
        "mape_pct": M.mape(y_true, y_pred),
        "rmse_ppm": M.rmse(y_true, y_pred),
    }


def evaluate_denoising(truth_h5: Path, predictions_h5: Path) -> dict:
    import h5py

    records = read_records(truth_h5)
    with h5py.File(predictions_h5, "r") as f:
        grp = f["predictions"]
        spec_true, spec_pred = [], []
        for r in records:
            rid = r["meta"]["record_id"]
            if rid not in grp:
                raise SystemExit(f"predictions missing record {rid}")
            spec_true.append(r["arrays"]["absorbance_clean"])
            spec_pred.append(grp[rid][()])
    spec_true = np.asarray(spec_true)
    spec_pred = np.asarray(spec_pred)
    if spec_true.shape != spec_pred.shape:
        raise SystemExit(f"shape mismatch: truth {spec_true.shape} vs pred {spec_pred.shape}")
    return {
        "n_records": len(spec_true),
        "spectral_rmse": M.spectral_rmse(spec_true, spec_pred),
        "peak_weighted_rmse": M.peak_weighted_rmse(spec_true, spec_pred),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    choices=["T1-concentration", "T2-denoising", "T3-generalization",
                             "T4-wms-concentration", "T5-drift-compensation",
                             "T6-ood-instrument"])
    ap.add_argument("--truth", required=True, help="truth HDF5 file")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--t1-mae", type=float, default=None,
                    help="T3 only: in-distribution T1 test MAE for the degradation ratio")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if args.task in ("T1-concentration", "T3-generalization", "T4-wms-concentration"):
        scores = evaluate_concentration(Path(args.truth), Path(args.predictions))
        if args.task == "T3-generalization" and args.t1_mae:
            scores["degradation_ratio_vs_T1"] = M.degradation_ratio(
                scores["mae_ppm"], args.t1_mae
            )
    elif args.task == "T2-denoising":
        scores = evaluate_denoising(Path(args.truth), Path(args.predictions))
    elif args.task == "T5-drift-compensation":
        raise NotImplementedError(
            "T5 evaluation requires time-series data format; see docs/benchmark.md"
        )
    elif args.task == "T6-ood-instrument":
        raise NotImplementedError(
            "T6 evaluation requires OOD label format; see docs/benchmark.md"
        )

    out = {"task": args.task, "scores": scores}
    print(json.dumps(out, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
