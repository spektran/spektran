#!/usr/bin/env python
"""Gate G3 (CRDS): physics cross-validation for cavity ring-down spectroscopy.

Checks:
  1. Ring-down time: tau = L / (c * (1 - R + alpha*L)) from Beer-Lambert cavity
     physics. Cross-validate main implementation vs analytic formula over 1000
     random (L, R, alpha) points; max relative deviation < 0.1%.
  2. Round-trip identity: absorption_from_tau(ring_down_time(L, R, alpha), tau0, L)
     must recover alpha to < 0.01% over 1000 random points.
  3. Empty-cavity tau: tau0 = L / (c * (1 - R)) matches textbook formula.

Writes gates/reports/g3_crds_report.json. Exit code 0 = PASS.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

THRESHOLD_TAU = 1e-3
THRESHOLD_ROUNDTRIP = 1e-4
N_POINTS = 1000
SEED = 20260812

SPEED_OF_LIGHT_CM_S = 2.99792458e10


def check_ring_down_time() -> dict:
    from spektran.physics.crds import ring_down_time

    rng = np.random.default_rng(SEED)
    max_rel, deviations = 0.0, []
    for _ in range(N_POINTS):
        L_cm = rng.uniform(10.0, 100.0)
        R = 1.0 - 10.0 ** rng.uniform(-6.0, -3.0)
        alpha = 10.0 ** rng.uniform(-10.0, -4.0)
        main = ring_down_time(L_cm, R, alpha)
        ref = L_cm / (SPEED_OF_LIGHT_CM_S * ((1.0 - R) + alpha * L_cm))
        rel = abs(main - ref) / abs(ref)
        deviations.append(rel)
        max_rel = max(max_rel, rel)
    return {
        "n_points": N_POINTS,
        "seed": SEED,
        "max_relative_deviation": max_rel,
        "median_relative_deviation": float(np.median(deviations)),
        "threshold": THRESHOLD_TAU,
        "pass": bool(max_rel < THRESHOLD_TAU),
    }


def check_roundtrip_identity() -> dict:
    from spektran.physics.crds import absorption_from_tau, empty_cavity_tau, ring_down_time

    rng = np.random.default_rng(SEED + 1)
    max_rel, deviations = 0.0, []
    for _ in range(N_POINTS):
        L_cm = rng.uniform(10.0, 100.0)
        R = 1.0 - 10.0 ** rng.uniform(-6.0, -3.0)
        alpha = 10.0 ** rng.uniform(-9.0, -4.0)
        tau0 = empty_cavity_tau(L_cm, R)
        tau = ring_down_time(L_cm, R, alpha)
        alpha_recovered = absorption_from_tau(tau, tau0, L_cm)
        rel = abs(alpha_recovered - alpha) / abs(alpha)
        deviations.append(rel)
        max_rel = max(max_rel, rel)
    return {
        "n_points": N_POINTS,
        "seed": SEED + 1,
        "max_relative_deviation": max_rel,
        "median_relative_deviation": float(np.median(deviations)),
        "threshold": THRESHOLD_ROUNDTRIP,
        "pass": bool(max_rel < THRESHOLD_ROUNDTRIP),
    }


def check_empty_cavity_tau() -> dict:
    from spektran.physics.crds import empty_cavity_tau

    rng = np.random.default_rng(SEED + 2)
    max_rel = 0.0
    for _ in range(N_POINTS):
        L_cm = rng.uniform(10.0, 100.0)
        R = 1.0 - 10.0 ** rng.uniform(-6.0, -3.0)
        main = empty_cavity_tau(L_cm, R)
        ref = L_cm / (SPEED_OF_LIGHT_CM_S * (1.0 - R))
        rel = abs(main - ref) / abs(ref)
        max_rel = max(max_rel, rel)
    return {
        "n_points": N_POINTS,
        "seed": SEED + 2,
        "max_relative_deviation": max_rel,
        "threshold": 1e-12,
        "pass": bool(max_rel < 1e-12),
    }


def main() -> int:
    report = {
        "gate": "G3",
        "scope": "CRDS forward physics (ring-down time, round-trip identity, empty cavity)",
        "date": datetime.date.today().isoformat(),
        "checks": {
            "ring_down_time": check_ring_down_time(),
            "roundtrip_identity": check_roundtrip_identity(),
            "empty_cavity_tau": check_empty_cavity_tau(),
        },
    }
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    out = REPO / "gates" / "reports" / "g3_crds_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nGate G3 (CRDS): {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
