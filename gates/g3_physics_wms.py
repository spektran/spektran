#!/usr/bin/env python
"""Gate G3 (WMS part): dual-implementation cross-validation of the WMS chain.

Checks (plan §9 G3, §8 red lines):
  1. Full WMS chain (modulation + RAM + transmission + lock-in): time-domain
     main implementation vs Fourier-quadrature reference, >= 1000 random
     parameter points, relative deviation < 1%.
  2. Analytic anchor: optically-thin Lorentzian 2f peak vs Arndt closed form
     across modulation indices, < 1%.
  3. WMS pytest suite passes.

Writes gates/reports/g3_wms_report.json. Exit 0 = PASS.
Gate integrity: thresholds must not be edited in the PR that passes the gate.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

THRESHOLD = 1e-2  # < 1% (WMS full chain, plan §9)
N_POINTS = 1000
SEED = 20260808

NU0 = 6047.0
GAMMA_L = 0.05


def _absorbance_fn(peak: float):
    from opengasspec.physics.lineshape import lorentz_profile

    scale = peak / (1.0 / (np.pi * GAMMA_L))

    def absorbance(nu):
        return scale * lorentz_profile(np.asarray(nu, dtype=float), NU0, GAMMA_L)

    return absorbance


def _settled_mean(x: np.ndarray, frac: float = 0.3) -> float:
    n = len(x)
    return float(np.mean(x[int(n * frac) : int(n * (1.0 - frac))]))


def check_chain() -> dict:
    from opengasspec.physics.wms import WMSConfig, simulate_wms

    from tests.reference_impl.ref_wms import wms_harmonic_ref

    rng = np.random.default_rng(SEED)
    worst, deviations = 0.0, []
    for _ in range(N_POINTS):
        m = rng.uniform(0.2, 3.0)
        peak = 10.0 ** rng.uniform(-3.0, -0.3)
        offset = rng.uniform(-3.0, 3.0) * GAMMA_L
        i0 = rng.uniform(0.0, 0.5)
        i2 = rng.uniform(0.0, 0.05)
        psi1 = rng.uniform(-np.pi, np.pi)
        psi2 = rng.uniform(-np.pi, np.pi)
        harmonic = int(rng.integers(1, 3))
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=m * GAMMA_L,
            sampling_rate_Hz=1e6,
            duration_s=0.02,
            center_wavenumber_cm1=NU0 + offset,
            im_i0_rel=i0,
            im_i2_rel=i2,
            fm_im_phase1_rad=psi1,
            fm_im_phase2_rad=psi2,
        )
        absorb = _absorbance_fn(peak)
        out = simulate_wms(cfg, absorb, harmonics=(harmonic,))
        r_main = float(
            np.hypot(_settled_mean(out[f"x_{harmonic}f"]), _settled_mean(out[f"y_{harmonic}f"]))
        )
        x_ref, y_ref = wms_harmonic_ref(
            lambda nu: float(absorb(np.array([nu]))[0]),
            NU0 + offset,
            m * GAMMA_L,
            harmonic,
            im_i0_rel=i0,
            im_i2_rel=i2,
            fm_im_phase1_rad=psi1,
            fm_im_phase2_rad=psi2,
        )
        r_ref = float(np.hypot(x_ref, y_ref))
        rel = abs(r_main - r_ref) / max(abs(r_ref), 1e-12)
        deviations.append(rel)
        worst = max(worst, rel)
    return {
        "n_points": N_POINTS,
        "seed": SEED,
        "max_relative_deviation": worst,
        "median_relative_deviation": float(np.median(deviations)),
        "threshold": THRESHOLD,
        "pass": bool(worst < THRESHOLD),
    }


def check_arndt() -> dict:
    from opengasspec.physics.wms import WMSConfig, simulate_wms

    from tests.reference_impl.ref_wms import arndt_lorentzian_h2_peak

    worst = 0.0
    details = {}
    for m in (0.3, 0.5, 1.0, 1.5, 2.0, 2.2, 2.5):
        peak = 1e-3
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=m * GAMMA_L,
            sampling_rate_Hz=2e6,
            duration_s=0.02,
            center_wavenumber_cm1=NU0,
        )
        out = simulate_wms(cfg, _absorbance_fn(peak), harmonics=(2,))
        x2f = _settled_mean(out["x_2f"])
        analytic = arndt_lorentzian_h2_peak(peak, m)
        rel = abs(x2f - analytic) / abs(analytic)
        details[f"m={m}"] = rel
        worst = max(worst, rel)
    return {
        "relative_deviation_by_m": details,
        "max_relative_deviation": worst,
        "threshold": THRESHOLD,
        "pass": bool(worst < THRESHOLD),
    }


def check_pytest() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_wms.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    return {"exit_code": proc.returncode, "summary": tail, "pass": proc.returncode == 0}


def main() -> int:
    report = {
        "gate": "G3",
        "scope": "WMS chain (modulation + RAM + lock-in demodulation)",
        "date": datetime.date.today().isoformat(),
        "checks": {
            "wms_chain_cross_validation": check_chain(),
            "arndt_analytic_anchor": check_arndt(),
            "wms_test_suite": check_pytest(),
        },
    }
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    out = REPO / "gates" / "reports" / "g3_wms_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nGate G3 (WMS): {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
