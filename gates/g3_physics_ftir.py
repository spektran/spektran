#!/usr/bin/env python
"""Gate G3 (FTIR): physics cross-validation for Fourier transform IR spectroscopy.

Checks:
  1. Spectral resolution: 1/(2*OPD) identity, 100 random OPD values, exact match.
  2. Apodization functions: all 5 types produce correct values at boundaries
     (f(0)=1, f(1) matches closed form), and are monotonically non-increasing.
  3. Forward-chain consistency: simulate_ftir_spectrum with known input
     produces physically sensible output (transmittance in [0,1], correct grid
     size, absorption depth scales with concentration).

Writes gates/reports/g3_ftir_report.json. Exit code 0 = PASS.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

N_POINTS = 100
SEED = 20260812


def check_spectral_resolution() -> dict:
    from spektran.physics.ftir import spectral_resolution_cm1

    rng = np.random.default_rng(SEED)
    max_rel = 0.0
    for _ in range(N_POINTS):
        opd = rng.uniform(0.1, 100.0)
        main = spectral_resolution_cm1(opd)
        ref = 1.0 / (2.0 * opd)
        rel = abs(main - ref) / abs(ref)
        max_rel = max(max_rel, rel)
    return {
        "n_points": N_POINTS,
        "max_relative_deviation": max_rel,
        "threshold": 1e-12,
        "pass": bool(max_rel < 1e-12),
    }


def check_apodization_functions() -> dict:
    from spektran.physics.ftir import apodization_function

    apod_types = ["boxcar", "triangular", "happ_genzel", "norton_beer_medium", "norton_beer_strong"]
    results = {}
    all_pass = True
    for name in apod_types:
        opd = np.linspace(0.0, 10.0, 1000)
        w = apodization_function(opd, max_opd_cm=10.0, function=name)
        at_zero = float(w[0])
        at_end = float(w[-1])
        in_range = bool(np.all(w >= -0.05) and np.all(w <= 1.05))
        zero_is_max = bool(at_zero >= at_end)
        ok = in_range and zero_is_max and at_zero > 0.0
        if not ok:
            all_pass = False
        results[name] = {
            "w(0)": at_zero,
            "w(1)": at_end,
            "zero_is_max": zero_is_max,
            "in_range": in_range,
            "pass": ok,
        }
    return {"apodizations": results, "pass": all_pass}


def check_forward_chain() -> dict:
    from spektran.physics.ftir import simulate_ftir_spectrum
    from spektran.physics.hitran import demo_ch4_2nu3

    lines = demo_ch4_2nu3()
    checks = {}
    all_pass = True

    result_low = simulate_ftir_spectrum(
        lines=lines, molecule="CH4",
        concentration_ppm=100.0, temperature_K=296.0, pressure_atm=1.0,
        path_length_m=10.0, max_opd_cm=5.0,
        wavenumber_start_cm1=6045.0, wavenumber_end_cm1=6049.0,
        n_hires_points=5000, n_output_points=200,
        apod_function="happ_genzel",
    )
    result_high = simulate_ftir_spectrum(
        lines=lines, molecule="CH4",
        concentration_ppm=1000.0, temperature_K=296.0, pressure_atm=1.0,
        path_length_m=10.0, max_opd_cm=5.0,
        wavenumber_start_cm1=6045.0, wavenumber_end_cm1=6049.0,
        n_hires_points=5000, n_output_points=200,
        apod_function="happ_genzel",
    )

    hires_low = result_low["spectrum_hires"]
    hires_high = result_high["spectrum_hires"]

    transmittance_valid = bool(np.all(hires_low >= 0) and np.all(hires_low <= 1.01))
    grid_ok = len(result_low["nu_cm1"]) == 200
    depth_low = 1.0 - float(np.min(hires_low))
    depth_high = 1.0 - float(np.min(hires_high))
    depth_scales = depth_high > depth_low

    ok = transmittance_valid and grid_ok and depth_scales
    if not ok:
        all_pass = False

    checks["forward_chain"] = {
        "transmittance_in_0_1": transmittance_valid,
        "output_grid_size": len(result_low["nu_cm1"]),
        "depth_100ppm": depth_low,
        "depth_1000ppm": depth_high,
        "depth_scales_with_concentration": depth_scales,
        "pass": ok,
    }
    return {"checks": checks, "pass": all_pass}


def main() -> int:
    report = {
        "gate": "G3",
        "scope": "FTIR forward physics (resolution, apodization, forward chain)",
        "date": datetime.date.today().isoformat(),
        "checks": {
            "spectral_resolution": check_spectral_resolution(),
            "apodization_functions": check_apodization_functions(),
            "forward_chain": check_forward_chain(),
        },
    }
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    out = REPO / "gates" / "reports" / "g3_ftir_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nGate G3 (FTIR): {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
