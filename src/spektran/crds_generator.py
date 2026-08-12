"""CRDS dataset generator: cavity ring-down spectroscopy -> noisy records.

Signal chain (per record):

1. Sample concrete instrument parameters from the virtual-instrument config
   distributions (per-record RNG stream).
2. Sample gas conditions (concentration, T/P jitter).
3. Forward physics: compute clean tau spectrum via simulate_crds_spectrum
   (from physics/crds.py).
4. Cavity noise: apply mirror reflectivity drift, mode matching jitter,
   shot noise on ring-down times, detector noise, baseline loss drift.
5. Output: noisy tau spectrum + derived alpha spectrum for ML regression.

Reproducibility: a master seed spawns per-record independent child streams
via numpy SeedSequence.spawn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .instrument.crds_noise import (
    baseline_loss_drift,
    detector_noise,
    fitting_residual,
    mirror_drift,
    mode_matching_jitter,
    shot_noise_tau,
)
from .instrument.environment import jittered_conditions
from .instrument.sampling import sample_instrument
from .physics.crds import (
    absorption_from_tau,
    ring_down_time,
    simulate_crds_spectrum,
)
from .physics.hitran import MOLECULE_IDS, LineList


@dataclass
class CRDSGenerationSpec:
    """What to generate for CRDS: gas truth + instrument config."""

    lines: LineList
    molecule: str = "CH4"
    concentration_ppm_low: float = 0.1
    concentration_ppm_high: float = 500.0
    log_uniform_concentration: bool = True
    temperature_K: float = 296.0
    pressure_atm: float = 1.0
    path_length_m: float = 0.50
    matrix_gas: str = "N2"
    wavenumber_start_cm1: float = 6046.0
    wavenumber_end_cm1: float = 6048.0
    n_spectral_points: int = 200
    extra_conditions: dict = field(default_factory=dict)
    interferents: list[dict] = field(default_factory=list)


def _sample_concentration(
    spec: CRDSGenerationSpec,
    rng: np.random.Generator,
) -> float:
    lo, hi = spec.concentration_ppm_low, spec.concentration_ppm_high
    if spec.log_uniform_concentration:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def generate_crds_record(
    spec: CRDSGenerationSpec,
    instrument_cfg: dict,
    seed_seq: np.random.SeedSequence,
    scan_time_s: float = 0.0,
    frozen_instrument: dict | None = None,
) -> dict:
    """Generate one CRDS measurement record.

    Returns {'meta': <schema record>, 'arrays': {...}} with:
    - arrays['tau_spectrum']: shape (n_spectral_points,) noisy ring-down times [s]
    - arrays['tau_spectrum_clean']: clean ring-down times [s]
    - arrays['alpha_spectrum']: derived absorption coefficient [cm-1]
    - arrays['nu_cm1']: wavenumber grid [cm-1]
    """
    rng = np.random.default_rng(seed_seq)
    if frozen_instrument is None:
        inst = sample_instrument(instrument_cfg, rng)
    else:
        inst = frozen_instrument
        sample_instrument(instrument_cfg, rng)

    concentration_ppm = _sample_concentration(spec, rng)
    env = inst.get("environment", {})
    temperature_K, pressure_atm = jittered_conditions(
        rng,
        env.get("temperature_K", spec.temperature_K),
        env.get("pressure_atm", spec.pressure_atm),
        env.get("temperature_jitter_K", 0.0),
        env.get("pressure_jitter_atm", 0.0),
    )

    cavity = inst.get("cavity", {})
    det = inst.get("detector", {})

    cavity_length_m = cavity.get("length_m", spec.path_length_m)
    mirror_R = cavity.get("mirror_reflectivity", 0.99995)

    clean = simulate_crds_spectrum(
        lines=spec.lines,
        molecule=spec.molecule,
        concentration_ppm=concentration_ppm,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        cavity_length_m=cavity_length_m,
        mirror_reflectivity=mirror_R,
        wavenumber_start_cm1=spec.wavenumber_start_cm1,
        wavenumber_end_cm1=spec.wavenumber_end_cm1,
        n_spectral_points=spec.n_spectral_points,
        interferents=spec.interferents or None,
    )

    tau_clean = clean["tau_spectrum_s"]
    tau0 = clean["tau0_s"]
    nu = clean["nu_cm1"]
    n_pts = len(nu)

    tau_noisy = tau_clean.copy()

    n_photons = cavity.get("n_photons_per_ringdown", 1e6)
    tau_noisy = shot_noise_tau(rng, tau_noisy, n_photons)

    R_drift_sigma = cavity.get("mirror_drift_sigma", 0.0)
    if R_drift_sigma > 0:
        R_array = mirror_drift(
            rng, n_pts, mirror_R,
            drift_sigma=R_drift_sigma,
            mean_reversion_rate=cavity.get("mirror_drift_reversion", 0.01),
        )
        cavity_length_cm = cavity_length_m * 100.0
        for i in range(n_pts):
            alpha_i = clean["alpha_spectrum_cm1"][i]
            tau_noisy[i] = ring_down_time(cavity_length_cm, R_array[i], alpha_i)
        tau_noisy = shot_noise_tau(rng, tau_noisy, n_photons)

    coupling_sigma = cavity.get("mode_matching_sigma", 0.0)
    if coupling_sigma > 0:
        eta = mode_matching_jitter(
            rng, n_pts,
            coupling_efficiency_mean=cavity.get("mode_matching_mean", 0.95),
            coupling_efficiency_sigma=coupling_sigma,
        )
        tau_noisy = tau_noisy * eta

    det_noise_sigma = det.get("noise_sigma_rel", 0.0)
    if det_noise_sigma > 0:
        noise = detector_noise(rng, n_pts, det_noise_sigma)
        tau_noisy = tau_noisy * (1.0 + noise)

    baseline_drift_rate = cavity.get("baseline_loss_drift_rate", 0.0)
    if baseline_drift_rate > 0:
        drift = baseline_loss_drift(
            rng, n_pts,
            drift_rate_per_point=baseline_drift_rate,
            temperature_sensitivity=cavity.get("baseline_temp_sensitivity", 1e-7),
        )
        cavity_length_cm = cavity_length_m * 100.0
        tau_noisy = tau_noisy / (1.0 + drift * cavity_length_cm)

    n_modes = cavity.get("n_transverse_modes", 1)
    if n_modes > 1:
        tau_noisy = tau_noisy + fitting_residual(
            rng, tau_noisy, n_modes,
            mode_spread_fraction=cavity.get("mode_spread_fraction", 0.05),
        )

    tau_noisy = np.maximum(tau_noisy, 1e-12)

    alpha_recovered = absorption_from_tau(tau_noisy, tau0, cavity_length_m * 100.0)

    record_id = str(uuid.UUID(bytes=rng.bytes(16), version=4))

    arrays = {
        "tau_spectrum": tau_noisy.astype(np.float32),
        "tau_spectrum_clean": tau_clean.astype(np.float32),
        "alpha_spectrum": alpha_recovered.astype(np.float32),
        "nu_cm1": nu.astype(np.float32),
    }

    meta = {
        "record_id": record_id,
        "schema_version": "0.2",
        "data_origin": "simulated",
        "technique": "CRDS",
        "provenance": {
            "generator_version": __version__,
            "hitran_data_version": spec.lines.hitran_data_version,
            "random_seed": int(seed_seq.entropy)
            if isinstance(seed_seq.entropy, int)
            else 0,
            "instrument_config_id": inst["instrument_config_id"],
            "noise_config": {
                "sampled": {
                    k: v
                    for k, v in inst.items()
                    if k
                    not in {
                        "instrument_config_id",
                        "schema_version",
                        "technique",
                        "held_out",
                        "description",
                        "performance",
                    }
                },
                "spawn_key": [int(k) for k in seed_seq.spawn_key],
            },
        },
        "signals": {
            "tau_spectrum": {
                "array_ref": f"/records/{record_id}/tau_spectrum",
                "n_samples": n_pts,
                "unit": "s",
            },
            "alpha_spectrum": {
                "array_ref": f"/records/{record_id}/alpha_spectrum",
                "n_samples": n_pts,
                "unit": "cm-1",
            },
        },
        "labels": {
            "species": [
                {
                    "molecule": spec.molecule,
                    "hitran_molecule_id": MOLECULE_IDS[spec.molecule],
                    "concentration_ppm": concentration_ppm,
                    "concentration_uncertainty_ppm": 0.0,
                }
            ]
        },
        "conditions": {
            "temperature_K": temperature_K,
            "pressure_atm": pressure_atm,
            "path_length_m": cavity_length_m,
            "matrix_gas": spec.matrix_gas,
            **(
                {
                    "interferents": [
                        {
                            "molecule": interf["molecule"],
                            "hitran_molecule_id": MOLECULE_IDS[interf["molecule"]],
                            "concentration_ppm": interf["concentration_ppm"],
                        }
                        for interf in spec.interferents
                    ]
                }
                if spec.interferents
                else {}
            ),
        },
        "instrument": {
            "cavity": {
                "length_m": float(cavity_length_m),
                "mirror_reflectivity": float(mirror_R),
                "finesse": float(clean["finesse"]),
                "tau0_s": float(tau0),
            },
            "detector": {
                "type": det.get("type", "APD (simulated)"),
            },
            "target_lines": [
                {
                    "hitran_molecule_id": MOLECULE_IDS[spec.molecule],
                    "wavenumber_cm1": float(w),
                }
                for w in spec.lines.nu0_cm1
            ],
        },
    }
    return {"meta": meta, "arrays": arrays}


def generate_crds_dataset(
    spec: CRDSGenerationSpec,
    instrument_cfg: dict,
    n_records: int,
    master_seed: int,
) -> list[dict]:
    """Generate ``n_records`` reproducible CRDS records."""
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_records)
    return [
        generate_crds_record(spec, instrument_cfg, child)
        for child in children
    ]


def generate_crds_time_series(
    spec: CRDSGenerationSpec,
    instrument_cfg: dict,
    n_scans: int,
    master_seed: int,
    scan_interval_s: float,
) -> list[dict]:
    """Consecutive CRDS measurements of ONE instrument realization."""
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_scans + 1)
    inst = sample_instrument(
        instrument_cfg, np.random.default_rng(children[0]),
    )
    return [
        generate_crds_record(
            spec,
            instrument_cfg,
            children[k + 1],
            scan_time_s=k * scan_interval_s,
            frozen_instrument=inst,
        )
        for k in range(n_scans)
    ]
