"""One-command evaluation: predictions file + truth HDF5 -> metric scores.

Usage (also exposed as ``python -m spektran.benchmark.evaluate``):

    evaluate.py --task T1-concentration --truth data/test.h5 \
                --predictions preds.csv [--json-out scores.json]

Prediction formats:
- T1/T3/T4/T5: CSV with header ``record_id,concentration_ppm``
- T2: HDF5 with /predictions/<record_id> arrays (denoised absorbance)
- T6: CSV with header ``record_id,ood_score`` (higher = more confidently OOD)
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


def evaluate_drift(truth_h5: Path, predictions_csv: Path) -> dict:
    """T5: Allan-deviation curve of the drift-corrected concentration error.

    A T5 truth file is a concatenation of independent time series -- each a
    fixed-truth-concentration run of one frozen instrument realization (see
    ``cli.py``'s ``mode: time_series`` generation). Records from different
    series must never be treated as consecutive scans of the same drift
    process, or Allan variance would pick up a spurious jump at every series
    boundary. Series are recovered without extra metadata: every scan in one
    series shares an exactly equal true concentration by construction (each
    series uses a degenerate ``low == high`` truth range, and
    ``generator.sample_concentration`` returns that bound exactly every draw
    -- see its docstring), so a change in truth concentration marks a new
    series. Allan deviation is
    computed within each series and averaged across series (they share the
    same ``taus`` grid since every official T5 series has the same length).
    """
    from ..io import read_time_series

    records, scan_interval_s = read_time_series(truth_h5)
    ids = [r["meta"]["record_id"] for r in records]
    truth_conc = np.array(
        [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
    )

    preds: dict[str, float] = {}
    with open(predictions_csv) as f:
        for row in csv.DictReader(f):
            preds[row["record_id"]] = float(row["concentration_ppm"])
    missing = [rid for rid in ids if rid not in preds]
    if missing:
        raise SystemExit(
            f"predictions missing {len(missing)} record_ids (first: {missing[:3]})"
        )
    pred_conc = np.array([preds[rid] for rid in ids])

    errors = pred_conc - truth_conc
    boundaries = np.flatnonzero(np.diff(truth_conc) != 0) + 1
    series = np.split(errors, boundaries)

    if len({len(s) for s in series}) == 1:
        curves = [M.allan_variance(s, tau_points=20, dt=scan_interval_s) for s in series]
        taus = curves[0]["taus"]
        adevs = np.mean([c["adevs"] for c in curves], axis=0).tolist()
    else:
        # Irregular series lengths (e.g. a hand-built truth file): fall back
        # to one Allan-variance pass over the full concatenation rather than
        # averaging incompatible tau grids.
        av = M.allan_variance(errors, tau_points=20, dt=scan_interval_s)
        taus, adevs = av["taus"], av["adevs"]

    return {
        "n_scans": len(records),
        "n_series": len(series),
        "mae_ppm": M.mae(truth_conc, pred_conc),
        "adev_shortest_tau": adevs[0] if adevs else 0.0,
        "adev_longest_tau": adevs[-1] if adevs else 0.0,
        "adev_taus_s": taus,
        "adev_curve": adevs,
    }


def evaluate_ood(truth_h5: Path, predictions_csv: Path) -> dict:
    """T6: OOD instrument detection (AUROC).

    Ground truth is ``labels.ood_label`` in each record's metadata (0 = in-
    distribution instrument, 1 = held-out/OOD instrument). Missing defaults to
    0 -- records generated outside the ``ood_task`` CLI branch (e.g. the T6
    training split) never carry the field at all, and are in-distribution by
    construction. Predictions are a CSV of ``record_id,ood_score``; higher
    means more confidently OOD. The score need not be a probability --
    ``ood_auroc`` is rank-based.
    """
    records = read_records(truth_h5)
    truth_labels = {
        r["meta"]["record_id"]: r["meta"]["labels"].get("ood_label", 0) for r in records
    }

    preds: dict[str, float] = {}
    with open(predictions_csv) as f:
        for row in csv.DictReader(f):
            preds[row["record_id"]] = float(row["ood_score"])
    missing = sorted(set(truth_labels) - set(preds))
    if missing:
        raise SystemExit(
            f"predictions missing {len(missing)} record_ids (first: {missing[:3]})"
        )

    ids = sorted(truth_labels)
    y_true = np.array([truth_labels[i] for i in ids])
    y_scores = np.array([preds[i] for i in ids])

    return {
        "n_records": len(ids),
        "n_in_dist": int(np.sum(y_true == 0)),
        "n_ood": int(np.sum(y_true == 1)),
        "auroc": M.ood_auroc(y_true, y_scores),
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
        scores = evaluate_drift(Path(args.truth), Path(args.predictions))
    elif args.task == "T6-ood-instrument":
        scores = evaluate_ood(Path(args.truth), Path(args.predictions))

    out = {"task": args.task, "scores": scores}
    print(json.dumps(out, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
