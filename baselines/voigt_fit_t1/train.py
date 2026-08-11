#!/usr/bin/env python
"""Classical Voigt-fit concentration retrieval baseline for T1.

The standard spectroscopic method: for each raw scan, (1) estimate the
baseline via wing-region polynomial, (2) compute absorbance as -log(raw/baseline),
(3) fit a Voigt profile + polynomial residual baseline to the absorbance using
Levenberg-Marquardt, (4) recover concentration from the fitted peak area via
Beer-Lambert. No training data needed — fully physics-based, deterministic.

This is the gold-standard classical comparison: what a spectroscopist does
before any ML is applied. Scipy's curve_fit (trust-region reflective) handles
the nonlinear optimization.

    python baselines/voigt_fit_t1/train.py

Reference:
    P. Werle et al., "The limits of signal averaging in atmospheric
    trace-gas monitoring by tunable diode-laser absorption spectroscopy",
    Appl. Phys. B 57 (1993) 131, doi:10.1007/BF00425997
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, load_split, write_predictions_csv  # noqa: E402

OUT = Path(__file__).resolve().parent

NU_CENTER = 6047.0
NU_RANGE = 2.0
N_POINTS = 2000
PATH_CM = 10.0 * 100.0

_SQRT_2LN2 = np.sqrt(2.0 * np.log(2.0))
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def voigt_absorbance_model(nu, area, nu0, gamma_D, gamma_L, c0, c1, c2):
    sigma = gamma_D / _SQRT_2LN2
    z = ((nu - nu0) + 1j * gamma_L) / (sigma * np.sqrt(2.0))
    phi = np.real(wofz(z)) / (sigma * _SQRT_2PI)
    return area * phi + c0 + c1 * (nu - NU_CENTER) + c2 * (nu - NU_CENTER) ** 2


def raw_to_absorbance(raw, wing_frac=0.15):
    n = len(raw)
    idx = np.arange(n)
    wings = np.r_[0 : int(n * wing_frac), int(n * (1 - wing_frac)) : n]
    coeff = np.polyfit(wings, raw[wings], 3)
    baseline = np.polyval(coeff, idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(np.clip(raw / baseline, 1e-9, None))


def fit_single_scan(raw):
    absorbance = raw_to_absorbance(raw)
    nu = np.linspace(NU_CENTER - NU_RANGE / 2, NU_CENTER + NU_RANGE / 2, len(absorbance))

    peak_idx = np.argmax(absorbance)
    peak_area_guess = float(np.trapezoid(np.clip(absorbance, 0, None), nu))

    p0 = [
        max(peak_area_guess, 1e-6),
        float(nu[peak_idx]),
        0.015,
        0.03,
        0.0, 0.0, 0.0,
    ]
    bounds_lo = [0.0, nu[0], 1e-4, 1e-4, -0.1, -0.5, -1.0]
    bounds_hi = [10.0, nu[-1], 0.5, 0.5, 0.1, 0.5, 1.0]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                voigt_absorbance_model, nu, absorbance,
                p0=p0, bounds=(bounds_lo, bounds_hi),
                maxfev=2000,
            )
        fitted_area = popt[0]
    except (RuntimeError, ValueError):
        fitted_area = max(peak_area_guess, 0.0)

    sw_eff = 1.37e-21
    n_density = 2.479e19
    concentration = fitted_area / (sw_eff * n_density * PATH_CM)
    concentration_ppm = concentration * 1e6
    return max(concentration_ppm, 0.0)


def main() -> int:
    X_te, _, ids_te = load_split("ch4-t1-test-v0")
    X_ho, _, ids_ho = load_split("ch4-t3-test-heldout-v0")

    for X, ids, tag in [(X_te, ids_te, "t1-test"), (X_ho, ids_ho, "t3-test-heldout")]:
        preds = np.array([fit_single_scan(raw) for raw in X])
        write_predictions_csv(OUT / f"predictions_{tag}.csv", ids, preds)
        print(f"{tag}: {len(ids)} scans fitted -> predictions_{tag}.csv")

    (OUT / "hyperparams.json").write_text(json.dumps({
        "method": "Levenberg-Marquardt Voigt fit",
        "wing_fraction": 0.15,
        "baseline_order": 3,
        "note": "No training — pure physics-based retrieval",
    }))
    print(f"Voigt-fit predictions written under {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
