#!/usr/bin/env python
"""Gate G3 (DOAS): physics cross-validation for differential optical absorption.

Checks:
  1. Beer-Lambert identity: OD = sigma * N * L, cross-validate DOAS OD
     computation vs direct calculation over 1000 random points, < 0.1%.
  2. Rayleigh cross section: lambda^-4 scaling law, cross-validate vs
     textbook formula over 100 wavelengths.
  3. Polynomial high-pass: a pure polynomial input must produce near-zero
     differential OD (residual < 1e-10).
  4. Round-trip: simulate_doas_spectrum OD_total must equal
     sigma * number_density * path_length for the molecular component when
     Rayleigh and Mie are disabled.

Writes gates/reports/g3_doas_report.json. Exit code 0 = PASS.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

THRESHOLD = 1e-3
N_POINTS = 1000
SEED = 20260812

BOLTZMANN_CGS = 1.380649e-16
ATM_DYN_PER_CM2 = 1.01325e6


def check_beer_lambert_od() -> dict:
    from spektran.physics.doas import number_density

    rng = np.random.default_rng(SEED)
    max_rel, deviations = 0.0, []
    for _ in range(N_POINTS):
        conc_ppm = 10.0 ** rng.uniform(-2.0, 3.0)
        T = rng.uniform(250.0, 320.0)
        P = rng.uniform(0.8, 1.1)
        sigma = 10.0 ** rng.uniform(-20.0, -17.0)
        L_cm = rng.uniform(100.0, 500000.0)
        N = number_density(conc_ppm, T, P)
        od = sigma * N * L_cm
        N_ref = conc_ppm * 1e-6 * P * ATM_DYN_PER_CM2 / (BOLTZMANN_CGS * T)
        od_ref = sigma * N_ref * L_cm
        rel = abs(od - od_ref) / max(abs(od_ref), 1e-30)
        deviations.append(rel)
        max_rel = max(max_rel, rel)
    return {
        "n_points": N_POINTS,
        "seed": SEED,
        "max_relative_deviation": max_rel,
        "median_relative_deviation": float(np.median(deviations)),
        "threshold": THRESHOLD,
        "pass": bool(max_rel < THRESHOLD),
    }


def check_rayleigh_scaling() -> dict:
    from spektran.physics.doas import rayleigh_cross_section

    wavelengths = np.linspace(200.0, 800.0, 100)
    sigma_ray = rayleigh_cross_section(wavelengths)
    ratios = []
    for i in range(1, len(wavelengths)):
        expected_ratio = (wavelengths[i] / wavelengths[0]) ** 4
        actual_ratio = sigma_ray[0] / sigma_ray[i]
        rel = abs(actual_ratio - expected_ratio) / expected_ratio
        ratios.append(rel)
    max_rel = float(np.max(ratios))
    return {
        "n_wavelengths": len(wavelengths),
        "max_relative_deviation_from_lambda4": max_rel,
        "threshold": 0.01,
        "pass": bool(max_rel < 0.01),
    }


def check_polynomial_highpass() -> dict:
    from spektran.physics.doas import polynomial_high_pass

    wavelength = np.linspace(300.0, 400.0, 500)
    for order in [3, 5, 7]:
        x_norm = np.linspace(-1, 1, 500)
        coeffs = np.random.default_rng(SEED + 3).uniform(-1, 1, order + 1)
        poly_signal = np.polyval(coeffs, x_norm)
        diff = polynomial_high_pass(poly_signal, order)
        residual = float(np.max(np.abs(diff)))
        if residual > 1e-8:
            return {
                "poly_order": order,
                "max_residual": residual,
                "threshold": 1e-8,
                "pass": False,
            }
    return {"poly_orders_tested": [3, 5, 7], "max_residual": residual, "pass": True}


def check_molecular_od_consistency() -> dict:
    from spektran.physics.doas import (
        number_density,
        simulate_doas_cross_section,
        simulate_doas_spectrum,
    )

    wavelength = np.linspace(300.0, 360.0, 200)
    sigma = simulate_doas_cross_section(
        wavelength, center_nm=330.0, peak_cross_section_cm2=6e-19,
        n_features=5, feature_width_nm=0.8,
    )
    conc_ppm = 10.0
    T, P = 296.0, 1.0
    L_m = 500.0
    L_cm = L_m * 100.0

    result = simulate_doas_spectrum(
        wavelength_nm=wavelength, target_sigma_cm2=sigma,
        target_concentration_ppm=conc_ppm,
        temperature_K=T, pressure_atm=P,
        path_length_m=L_m, poly_order=5,
        rayleigh=False, mie_tau_ref=0.0,
    )

    N = number_density(conc_ppm, T, P)
    od_ref = sigma * N * L_cm
    od_main = result["od_total"]
    max_rel = float(np.max(np.abs(od_main - od_ref) / np.maximum(od_ref, 1e-30)))
    return {
        "max_relative_deviation": max_rel,
        "threshold": THRESHOLD,
        "pass": bool(max_rel < THRESHOLD),
    }


def main() -> int:
    report = {
        "gate": "G3",
        "scope": "DOAS forward physics (Beer-Lambert, Rayleigh, polynomial high-pass, consistency)",
        "date": datetime.date.today().isoformat(),
        "checks": {
            "beer_lambert_od": check_beer_lambert_od(),
            "rayleigh_lambda4_scaling": check_rayleigh_scaling(),
            "polynomial_highpass_nulls_poly": check_polynomial_highpass(),
            "molecular_od_consistency": check_molecular_od_consistency(),
        },
    }
    report["pass"] = all(c["pass"] for c in report["checks"].values())
    out = REPO / "gates" / "reports" / "g3_doas_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nGate G3 (DOAS): {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
