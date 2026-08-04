#!/usr/bin/env python
"""Gate G4: noise realism — literature-anchored derived metrics.

For each of the 8 virtual instruments, generate data and compute derived
metrics THE SAME WAY the anchor survey reports them, then require them to lie
inside the literature envelopes of configs/instruments/literature_anchors.yaml:

  1. NEA (noise-equivalent absorbance, per scan): median over records within
     [nea_min, fringe_floor_max] — the combined absorbance-floor envelope.
  2. SNR at the reference condition (100 ppm CH4, 10 m): median within
     [P5, P95] of the SNR anchor.
  3. Total etalon-fringe amplitude: median within the literature
     absorbance-floor band.
  4. Allan turnover time (time-series of scans, matched-filter concentration
     retrieval): within [P5, P95] of allan_optimum_time_s.
  5. Family spread: the instrument set must span from better-than-median to
     near-worst-tier (min median-NEA <= P50 anchor; max median-NEA >= half the
     anchor max; max median-SNR >= P50 anchor).

Median-based checks are used because anchor values are single reported
numbers per paper (heterogeneous conditions); per-record tails may exceed the
envelope without breaking the anchoring claim. Known limitation (plan §9 G4
honesty note): this validates statistical similarity to published systems,
not point-wise physical truth.

Writes gates/reports/g4_report.json. Exit 0 = PASS.
"""

from __future__ import annotations

import datetime
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SEED = 20260811
N_RECORDS = 24
N_SCANS = 480
SCAN_INTERVAL_S = 0.5
REF_CONC_PPM = 100.0


def load_anchors() -> dict:
    return yaml.safe_load(
        (REPO / "configs" / "instruments" / "literature_anchors.yaml").read_text()
    )


def envelopes(anchors: dict) -> dict:
    m = anchors["metrics"]

    def rng_of(name, lo_key="p5", hi_key="p95"):
        met = m[name]
        lo = met.get(lo_key, met.get("min"))
        hi = met.get(hi_key, met.get("max"))
        return float(lo), float(hi)

    nea_lo, nea_hi = rng_of("noise_equivalent_absorbance")
    fr_lo, fr_hi = rng_of("fringe_amplitude_rel_to_peak")
    return {
        "snr": rng_of("snr"),
        # Combined absorbance-floor envelope: NEA anchors + fringe floors
        "nea": (min(nea_lo, fr_lo), max(nea_hi, fr_hi)),
        "fringe": (fr_lo, fr_hi),
        "allan_turnover_s": rng_of("allan_optimum_time_s"),
        "nea_p50": float(m["noise_equivalent_absorbance"].get("p50",
                         m["noise_equivalent_absorbance"].get("median"))),
        "snr_p50": float(m["snr"]["p50"]),
    }


def _highpass_noise_std(x: np.ndarray) -> float:
    """Noise std after removing slow structure (baseline, fringes, line)."""
    sos = butter(4, 0.15, btype="high", output="sos")
    return float(np.std(sosfiltfilt(sos, x)))


def instrument_metrics(cfg_path: str, anchors_env: dict) -> dict:
    from spektran.generator import (
        GenerationSpec,
        generate_dataset,
        generate_time_series,
    )
    from spektran.instrument.sampling import load_instrument_config
    from spektran.physics import demo_ch4_2nu3

    cfg = load_instrument_config(cfg_path)
    spec = GenerationSpec(
        lines=demo_ch4_2nu3(),
        concentration_ppm_low=REF_CONC_PPM,
        concentration_ppm_high=REF_CONC_PPM,
        log_uniform_concentration=False,
        n_points=2000,
    )

    # --- per-record NEA, SNR, fringe amplitude ---
    recs = generate_dataset(spec, cfg, N_RECORDS, master_seed=SEED)
    neas, snrs, fringes = [], [], []
    for r in recs:
        raw = r["arrays"]["raw_scan"]
        clean_peak = float(r["arrays"]["absorbance_clean"].max())
        nea = _highpass_noise_std(raw)  # relative intensity ~ absorbance (thin)
        neas.append(nea)
        snrs.append(clean_peak / nea)
        sampled = r["meta"]["provenance"]["noise_config"]["sampled"]
        amp = sum(e["amplitude_rel"] for e in sampled.get("etalons", []) or [])
        fringes.append(amp)

    # --- Allan turnover from a scan time series (matched-filter retrieval) ---
    series = generate_time_series(spec, cfg, N_SCANS, SEED + 1, SCAN_INTERVAL_S)
    template = series[0]["arrays"]["absorbance_clean"]
    t2 = float(template @ template)
    n = len(template)
    wings = np.r_[0 : n // 5, 4 * n // 5 : n]
    conc = []
    for r in series:
        raw = r["arrays"]["raw_scan"]
        # crude DA retrieval: wing-anchored polynomial baseline -> absorbance
        coeff = np.polyfit(wings, raw[wings], 3)
        baseline = np.polyval(coeff, np.arange(n))
        with np.errstate(divide="ignore", invalid="ignore"):
            a_meas = -np.log(np.clip(raw / baseline, 1e-9, None))
        conc.append(float(a_meas @ template) / t2 * REF_CONC_PPM)
    conc = np.asarray(conc)

    taus, adevs = _allan(conc, SCAN_INTERVAL_S)
    turnover_s = float(taus[int(np.argmin(adevs))])

    return {
        "instrument": cfg["instrument_config_id"],
        "technique": cfg["technique"],
        "held_out": bool(cfg.get("held_out", False)),
        "nea_median": float(np.median(neas)),
        "snr_median": float(np.median(snrs)),
        "fringe_total_median": float(np.median(fringes)),
        "allan_turnover_s": turnover_s,
        "allan_min_ppm": float(np.min(adevs)),
    }


def _allan(y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping Allan deviation over averaging times tau = k*dt."""
    taus, adevs = [], []
    k = 1
    while k <= len(y) // 4:
        m = len(y) // k
        means = y[: m * k].reshape(m, k).mean(axis=1)
        adev = np.sqrt(0.5 * np.mean(np.diff(means) ** 2))
        taus.append(k * dt)
        adevs.append(adev)
        k *= 2
    return np.asarray(taus), np.asarray(adevs)


def main() -> int:
    anchors = load_anchors()
    env = envelopes(anchors)
    cfgs = sorted(glob.glob(str(REPO / "configs" / "instruments" / "vi-*.yaml")))
    per_inst, failures = [], []
    for c in cfgs:
        m = instrument_metrics(c, env)
        checks = {
            "nea_in_envelope": env["nea"][0] <= m["nea_median"] <= env["nea"][1],
            "snr_in_envelope": env["snr"][0] <= m["snr_median"] <= env["snr"][1],
            "fringe_in_envelope": env["fringe"][0]
            <= m["fringe_total_median"]
            <= env["fringe"][1],
            "allan_turnover_in_envelope": env["allan_turnover_s"][0]
            <= m["allan_turnover_s"]
            <= env["allan_turnover_s"][1],
        }
        m["checks"] = checks
        m["pass"] = all(checks.values())
        if not m["pass"]:
            failures.append(m["instrument"])
        per_inst.append(m)

    neas = [m["nea_median"] for m in per_inst]
    snrs = [m["snr_median"] for m in per_inst]
    spread = {
        "min_nea_below_anchor_p50": min(neas) <= env["nea_p50"],
        "max_nea_above_half_anchor_max": max(neas) >= 0.5 * env["nea"][1],
        "max_snr_above_anchor_p50": max(snrs) >= env["snr_p50"],
    }

    report = {
        "gate": "G4",
        "date": datetime.date.today().isoformat(),
        "anchors_source": "configs/instruments/literature_anchors.yaml (18 papers)",
        "method": {
            "n_records_per_instrument": N_RECORDS,
            "n_scans_time_series": N_SCANS,
            "scan_interval_s": SCAN_INTERVAL_S,
            "reference_condition": f"{REF_CONC_PPM} ppm CH4, 10 m, 296 K, 1 atm",
            "seed": SEED,
        },
        "envelopes": {k: v for k, v in env.items() if isinstance(v, tuple)},
        "instruments": per_inst,
        "family_spread": spread,
        "known_limitation": (
            "Validates statistical similarity of derived indicators to published "
            "systems, not point-wise physical truth (plan §9 G4)."
        ),
        "pass": not failures and all(spread.values()),
    }
    out = REPO / "gates" / "reports" / "g4_report.json"
    out.write_text(json.dumps(report, indent=2))
    for m in per_inst:
        flag = "PASS" if m["pass"] else "FAIL " + str(m["checks"])
        print(
            f"{m['instrument']:>18}  NEA {m['nea_median']:.2e}  SNR {m['snr_median']:7.1f}  "
            f"fringe {m['fringe_total_median']:.2e}  turnover {m['allan_turnover_s']:6.1f}s  {flag}"
        )
    print("spread:", spread)
    print(f"Gate G4: {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
