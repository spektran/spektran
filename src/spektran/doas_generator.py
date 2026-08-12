"""DOAS dataset generator: UV/Vis cross sections -> differential OD records.

Signal chain (per record):

1. Sample instrument parameters from virtual-instrument config distributions.
2. Sample gas conditions (concentration, T/P jitter).
3. Generate synthetic UV/Vis cross section for target molecule.
4. Forward physics: Beer-Lambert + Rayleigh/Mie + polynomial high-pass.
5. Apply noise: photon noise, stray light, Ring effect, wavelength shift.
6. Output: noisy DOAS differential OD for ML regression.

Reproducibility: SeedSequence.spawn per record.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .instrument.doas_noise import (
    dark_current_noise,
    photon_noise,
    readout_noise,
    ring_effect,
    stray_light,
    wavelength_shift,
)
from .instrument.environment import jittered_conditions
from .instrument.sampling import sample_instrument
from .physics.doas import (
    polynomial_high_pass,
    simulate_doas_cross_section,
    simulate_doas_spectrum,
)

DOAS_MOLECULE_IDS = {
    "SO2": 9,
    "NO2": 10,
    "O3": 3,
    "HCHO": 20,
    "BrO": 40,
    "OClO": 43,
}


@dataclass
class DOASGenerationSpec:
    """What to generate for DOAS: gas truth + instrument config."""

    molecule: str = "SO2"
    concentration_ppm_low: float = 0.001
    concentration_ppm_high: float = 10.0
    log_uniform_concentration: bool = True
    temperature_K: float = 296.0
    pressure_atm: float = 1.0
    path_length_m: float = 1000.0
    matrix_gas: str = "air"
    wavelength_start_nm: float = 300.0
    wavelength_end_nm: float = 360.0
    n_output_points: int = 500
    cross_section_center_nm: float = 330.0
    cross_section_peak_cm2: float = 6e-19
    n_features: int = 5
    feature_width_nm: float = 0.8
    feature_spacing_nm: float = 1.5
    poly_order: int = 5
    extra_conditions: dict = field(default_factory=dict)
    interferents: list[dict] = field(default_factory=list)


def _sample_concentration(
    spec: DOASGenerationSpec,
    rng: np.random.Generator,
) -> float:
    lo, hi = spec.concentration_ppm_low, spec.concentration_ppm_high
    if spec.log_uniform_concentration:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def generate_doas_record(
    spec: DOASGenerationSpec,
    instrument_cfg: dict,
    seed_seq: np.random.SeedSequence,
    frozen_instrument: dict | None = None,
) -> dict:
    """Generate one DOAS measurement record.

    Returns {'meta': <schema record>, 'arrays': {...}} with:
    - arrays['doas_spectrum']: shape (n_output_points,) noisy differential OD
    - arrays['doas_spectrum_clean']: clean differential OD
    - arrays['wavelength_nm']: wavelength grid [nm]
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

    spectro = inst.get("spectrograph", {})
    det = inst.get("detector", {})

    wavelength = np.linspace(
        spec.wavelength_start_nm, spec.wavelength_end_nm, spec.n_output_points,
    )

    target_sigma = simulate_doas_cross_section(
        wavelength,
        center_nm=spec.cross_section_center_nm,
        peak_cross_section_cm2=spec.cross_section_peak_cm2,
        n_features=spec.n_features,
        feature_width_nm=spec.feature_width_nm,
        feature_spacing_nm=spec.feature_spacing_nm,
    )

    interferent_list = []
    for interf in spec.interferents:
        i_sigma = simulate_doas_cross_section(
            wavelength,
            center_nm=interf.get("center_nm", spec.cross_section_center_nm + 10),
            peak_cross_section_cm2=interf.get("peak_cm2", 1e-19),
            n_features=interf.get("n_features", 3),
            feature_width_nm=interf.get("width_nm", 1.0),
            feature_spacing_nm=interf.get("spacing_nm", 2.0),
        )
        interferent_list.append({
            "sigma_cm2": i_sigma,
            "concentration_ppm": interf["concentration_ppm"],
            "molecule": interf["molecule"],
        })

    mie_tau = spectro.get("mie_tau_ref", 0.1)
    mie_ang = spectro.get("mie_angstrom_exp", 1.3)

    result = simulate_doas_spectrum(
        wavelength_nm=wavelength,
        target_sigma_cm2=target_sigma,
        target_concentration_ppm=concentration_ppm,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        path_length_m=spec.path_length_m,
        poly_order=spec.poly_order,
        rayleigh=True,
        mie_tau_ref=mie_tau,
        mie_angstrom=mie_ang,
        interferent_sigmas=interferent_list if interferent_list else None,
    )

    doas_clean = result["doas_spectrum"].copy()
    doas_noisy = doas_clean.copy()
    n_pts = len(wavelength)

    photon_ref = det.get("photon_count_ref", 1e6)
    if photon_ref > 0:
        pn = photon_noise(rng, np.exp(-result["od_total"]), photon_ref)
        doas_noisy = doas_noisy + polynomial_high_pass(pn, spec.poly_order)

    stray_frac = spectro.get("stray_light_fraction", 0.0)
    if stray_frac > 0:
        sl = stray_light(rng, n_pts, stray_frac)
        doas_noisy = doas_noisy + sl

    ring_amp = spectro.get("ring_effect_amplitude", 0.0)
    if ring_amp > 0:
        re = ring_effect(wavelength, ring_amp, rng=rng)
        re_hp = polynomial_high_pass(re, spec.poly_order)
        doas_noisy = doas_noisy + re_hp

    shift_nm = spectro.get("wavelength_shift_nm", 0.0)
    if abs(shift_nm) > 0:
        actual_shift = rng.normal(0, shift_nm)
        squeeze = rng.normal(0, shift_nm * 0.01)
        doas_noisy = wavelength_shift(wavelength, doas_noisy, actual_shift, squeeze)

    dark_sigma = det.get("dark_current_sigma", 0.0)
    if dark_sigma > 0:
        dc = dark_current_noise(rng, n_pts, dark_sigma)
        doas_noisy = doas_noisy + dc

    readout_sig = det.get("readout_noise_sigma", 0.0)
    if readout_sig > 0:
        rn = readout_noise(rng, n_pts, readout_sig)
        doas_noisy = doas_noisy + rn

    record_id = str(uuid.UUID(bytes=rng.bytes(16), version=4))

    arrays = {
        "doas_spectrum": doas_noisy.astype(np.float32),
        "doas_spectrum_clean": doas_clean.astype(np.float32),
        "wavelength_nm": wavelength.astype(np.float32),
    }

    meta = {
        "record_id": record_id,
        "schema_version": "0.2",
        "data_origin": "simulated",
        "technique": "DOAS",
        "provenance": {
            "generator_version": __version__,
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
            "doas_spectrum": {
                "array_ref": f"/records/{record_id}/doas_spectrum",
                "n_samples": n_pts,
                "unit": "OD_diff",
            },
        },
        "labels": {
            "species": [
                {
                    "molecule": spec.molecule,
                    "hitran_molecule_id": DOAS_MOLECULE_IDS.get(spec.molecule, 0),
                    "concentration_ppm": concentration_ppm,
                    "concentration_uncertainty_ppm": 0.0,
                }
            ]
        },
        "conditions": {
            "temperature_K": temperature_K,
            "pressure_atm": pressure_atm,
            "path_length_m": spec.path_length_m,
            "matrix_gas": spec.matrix_gas,
        },
        "instrument": {
            "spectrograph": {
                "poly_order": spec.poly_order,
                "mie_tau_ref": float(mie_tau),
            },
            "detector": {
                "type": det.get("type", "CCD (simulated)"),
            },
            "target_lines": [
                {
                    "hitran_molecule_id": DOAS_MOLECULE_IDS.get(spec.molecule, 1),
                    "wavenumber_cm1": 1e7 / spec.cross_section_center_nm,
                }
            ],
        },
    }
    return {"meta": meta, "arrays": arrays}


def generate_doas_dataset(
    spec: DOASGenerationSpec,
    instrument_cfg: dict,
    n_records: int,
    master_seed: int,
) -> list[dict]:
    """Generate ``n_records`` reproducible DOAS records."""
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_records)
    return [
        generate_doas_record(spec, instrument_cfg, child)
        for child in children
    ]
