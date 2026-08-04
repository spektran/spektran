#!/usr/bin/env python
"""Gate G3 (DA part): dual-implementation cross-validation of forward physics.

Checks (plan §9, G3):
  1. Voigt lineshape: main (Faddeeva/wofz) vs reference (adaptive quadrature),
     >= 1000 random parameter points, max relative deviation < 0.1%.
  2. Full DA forward chain (line strength + widths + Voigt + Beer-Lambert):
     main vs independent reference transcription, >= 1000 random points, < 0.1%.
  3. Full physics-correctness pytest suite (plan §8) passes.
  4. Formula docstrings carry literature DOIs (counted; independent reviewer
     spot-checks 10 against the sources).

Writes gates/reports/g3_da_report.json. Exit code 0 = PASS.

NOTE (gate integrity): thresholds below must not be edited in the PR that is
trying to pass the gate. See CONTRIBUTING.md.
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

THRESHOLD_LINESHAPE = 1e-3  # < 0.1 %
THRESHOLD_CHAIN = 1e-3  # < 0.1 %
N_POINTS = 1000
SEED_LINESHAPE = 20260804
SEED_CHAIN = 20260805


def check_lineshape() -> dict:
    from opengasspec.physics import voigt_profile
    from tests.reference_impl.ref_lineshape import voigt_profile_ref

    rng = np.random.default_rng(SEED_LINESHAPE)
    max_rel, deviations = 0.0, []
    for _ in range(N_POINTS):
        a_d = 10.0 ** rng.uniform(-3.5, -1.0)
        g_l = 10.0 ** rng.uniform(-4.0, 0.0)
        width = max(a_d, g_l)
        offset = rng.uniform(-10.0, 10.0) * width
        nu0 = rng.uniform(1000.0, 8000.0)
        main = voigt_profile(np.array([nu0 + offset]), nu0, a_d, g_l)[0]
        ref = voigt_profile_ref(nu0 + offset, nu0, a_d, g_l)
        rel = abs(main - ref) / abs(ref)
        deviations.append(rel)
        max_rel = max(max_rel, rel)
    return {
        "n_points": N_POINTS,
        "seed": SEED_LINESHAPE,
        "max_relative_deviation": max_rel,
        "median_relative_deviation": float(np.median(deviations)),
        "threshold": THRESHOLD_LINESHAPE,
        "pass": bool(max_rel < THRESHOLD_LINESHAPE),
    }


def check_forward_chain() -> dict:
    from opengasspec.physics import demo_ch4_2nu3
    from opengasspec.physics.absorption import absorption_coefficient, default_q_ratio
    from opengasspec.physics.hitran import LineList
    from tests.reference_impl.ref_absorption import absorbance_ref

    rng = np.random.default_rng(SEED_CHAIN)
    lines = demo_ch4_2nu3()
    max_rel, deviations = 0.0, []
    for _ in range(N_POINTS):
        j = int(rng.integers(0, len(lines)))
        T = rng.uniform(250.0, 350.0)
        P = 10.0 ** rng.uniform(-1.0, 0.3)
        x = 10.0 ** rng.uniform(-6.0, -3.0)
        L_cm = rng.uniform(10.0, 5000.0)
        nu = float(lines.nu0_cm1[j]) + rng.uniform(-0.3, 0.3)
        q = default_q_ratio("CH4", T)
        single = LineList(
            molecule="CH4",
            nu0_cm1=lines.nu0_cm1[j : j + 1],
            sw_cm_per_molec=lines.sw_cm_per_molec[j : j + 1],
            gamma_air=lines.gamma_air[j : j + 1],
            gamma_self=lines.gamma_self[j : j + 1],
            n_air=lines.n_air[j : j + 1],
            delta_air=lines.delta_air[j : j + 1],
            elower_cm1=lines.elower_cm1[j : j + 1],
        )
        main = absorption_coefficient(np.array([nu]), single, x, T, P)[0] * L_cm
        ref = absorbance_ref(
            nu,
            nu0_cm1=float(lines.nu0_cm1[j]),
            sw_ref=float(lines.sw_cm_per_molec[j]),
            gamma_air=float(lines.gamma_air[j]),
            gamma_self=float(lines.gamma_self[j]),
            n_air=float(lines.n_air[j]),
            delta_air=float(lines.delta_air[j]),
            elower_cm1=float(lines.elower_cm1[j]),
            molar_mass_amu=lines.molar_mass_amu,
            mole_fraction=x,
            temperature_K=T,
            pressure_atm=P,
            path_length_cm=L_cm,
            q_ratio_value=q,
        )
        rel = abs(main - ref) / abs(ref)
        deviations.append(rel)
        max_rel = max(max_rel, rel)
    return {
        "n_points": N_POINTS,
        "seed": SEED_CHAIN,
        "max_relative_deviation": max_rel,
        "median_relative_deviation": float(np.median(deviations)),
        "threshold": THRESHOLD_CHAIN,
        "pass": bool(max_rel < THRESHOLD_CHAIN),
    }


def check_pytest() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not hitran_online"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    return {"exit_code": proc.returncode, "summary": tail, "pass": proc.returncode == 0}


def check_doi_citations() -> dict:
    doi_re = re.compile(r"doi:\s*10\.\d{4,9}/\S+", re.IGNORECASE)
    files = sorted((REPO / "src" / "opengasspec" / "physics").glob("*.py")) + sorted(
        (REPO / "tests" / "reference_impl").glob("ref_*.py")
    )
    per_file = {}
    for f in files:
        per_file[str(f.relative_to(REPO))] = len(doi_re.findall(f.read_text()))
    total = sum(per_file.values())
    return {
        "doi_citations_per_file": per_file,
        "total": total,
        "pass": total >= 10,
        "note": "Independent reviewer must spot-check 10 citations against sources.",
    }


def main() -> int:
    report = {
        "gate": "G3",
        "scope": "DA forward physics (lineshape + Beer-Lambert). WMS pending Phase 1.",
        "date": datetime.date.today().isoformat(),
        "checks": {
            "lineshape_cross_validation": check_lineshape(),
            "forward_chain_cross_validation": check_forward_chain(),
            "physics_test_suite": check_pytest(),
            "doi_citations": check_doi_citations(),
        },
    }
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    out = REPO / "gates" / "reports" / "g3_da_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nGate G3 (DA): {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
