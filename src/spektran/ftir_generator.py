"""FTIR dataset generator: interferogram -> FFT -> noisy spectrum records.

Signal chain (per record):

1. Sample concrete instrument parameters from the virtual-instrument config
   distributions (per-record RNG stream).
2. Sample gas conditions (concentration, T/P jitter).
3. Forward physics: compute clean FTIR spectrum via simulate_ftir_spectrum
   (from physics/ftir.py).
4. Apply noise: detector noise, source fluctuation, phase error, channel spectra.
5. Output: noisy FTIR spectrum for ML regression.

Reproducibility: a master seed spawns per-record independent child streams
via numpy SeedSequence.spawn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .instrument.environment import jittered_conditions
from .instrument.ftir_noise import (
    channel_spectrum,
    detector_noise,
    self_apodization,
    source_fluctuation,
)
from .instrument.sampling import sample_instrument
from .physics.ftir import simulate_ftir_spectrum
from .physics.hitran import MOLECULE_IDS, LineList


@dataclass
class FTIRGenerationSpec:
    """What to generate for FTIR: gas truth + instrument config."""

    lines: LineList
    molecule: str = "CH4"
    concentration_ppm_low: float = 0.5
    concentration_ppm_high: float = 500.0
    log_uniform_concentration: bool = True
    temperature_K: float = 296.0
    pressure_atm: float = 1.0
    path_length_m: float = 10.0
    matrix_gas: str = "N2"
    wavenumber_start_cm1: float = 6000.0
    wavenumber_end_cm1: float = 6100.0
    n_output_points: int = 500
    extra_conditions: dict = field(default_factory=dict)
    interferents: list[dict] = field(default_factory=list)


def _sample_concentration(
    spec: FTIRGenerationSpec,
    rng: np.random.Generator,
) -> float:
    lo, hi = spec.concentration_ppm_low, spec.concentration_ppm_high
    if spec.log_uniform_concentration:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def generate_ftir_record(
    spec: FTIRGenerationSpec,
    instrument_cfg: dict,
    seed_seq: np.random.SeedSequence,
    scan_time_s: float = 0.0,
    frozen_instrument: dict | None = None,
) -> dict:
    """Generate one FTIR measurement record.

    Returns {'meta': <schema record>, 'arrays': {...}} with:
    - arrays['ftir_spectrum']: shape (n_output_points,) noisy FTIR spectrum
    - arrays['ftir_spectrum_clean']: clean spectrum
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

    spectrometer = inst.get("spectrometer", {})
    det = inst.get("detector", {})

    max_opd = spectrometer.get("max_opd_cm", 45.0)
    apod = spectrometer.get("apodization", "norton_beer_medium")
    zero_fill = int(spectrometer.get("zero_fill_factor", 2))
    fov_angle = spectrometer.get("fov_half_angle_rad", 0.01)

    clean = simulate_ftir_spectrum(
        lines=spec.lines,
        molecule=spec.molecule,
        concentration_ppm=concentration_ppm,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        path_length_m=spec.path_length_m,
        max_opd_cm=max_opd,
        wavenumber_start_cm1=spec.wavenumber_start_cm1,
        wavenumber_end_cm1=spec.wavenumber_end_cm1,
        n_output_points=spec.n_output_points,
        apod_function=apod,
        zero_fill_factor=zero_fill,
        interferents=spec.interferents or None,
    )

    nu = clean["nu_cm1"]
    spectrum_clean = clean["spectrum"].copy()
    spectrum_noisy = spectrum_clean.copy()
    n_pts = len(nu)

    self_apod = self_apodization(nu, max_opd, fov_angle)
    spectrum_noisy = spectrum_noisy * self_apod

    det_sigma = det.get("noise_sigma_rel", 0.0)
    if det_sigma > 0:
        noise = detector_noise(rng, n_pts, det_sigma)
        spectrum_noisy = spectrum_noisy + noise * np.max(spectrum_clean)

    src_fluct = spectrometer.get("source_fluctuation_rel", 0.0)
    if src_fluct > 0:
        fluct = source_fluctuation(rng, n_pts, src_fluct)
        spectrum_noisy = spectrum_noisy * (1.0 + fluct)

    channel_amp = spectrometer.get("channel_spectrum_amplitude", 0.0)
    if channel_amp > 0:
        fsr = spectrometer.get("channel_spectrum_fsr_cm1", 5.0)
        phase = rng.uniform(0, 2 * np.pi)
        ch_spec = channel_spectrum(nu, channel_amp, fsr, phase)
        spectrum_noisy = spectrum_noisy * (1.0 + ch_spec)

    spectrum_noisy = np.maximum(spectrum_noisy, 1e-15)

    record_id = str(uuid.UUID(bytes=rng.bytes(16), version=4))

    arrays = {
        "ftir_spectrum": spectrum_noisy.astype(np.float32),
        "ftir_spectrum_clean": spectrum_clean.astype(np.float32),
        "nu_cm1": nu.astype(np.float32),
    }

    meta = {
        "record_id": record_id,
        "schema_version": "0.2",
        "data_origin": "simulated",
        "technique": "FTIR",
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
            "ftir_spectrum": {
                "array_ref": f"/records/{record_id}/ftir_spectrum",
                "n_samples": n_pts,
                "unit": "a.u.",
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
            "path_length_m": spec.path_length_m,
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
            "spectrometer": {
                "max_opd_cm": float(max_opd),
                "resolution_cm1": float(clean["resolution_cm1"]),
                "apodization": apod,
            },
            "detector": {
                "type": det.get("type", "InGaAs (simulated)"),
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


def generate_ftir_dataset(
    spec: FTIRGenerationSpec,
    instrument_cfg: dict,
    n_records: int,
    master_seed: int,
) -> list[dict]:
    """Generate ``n_records`` reproducible FTIR records."""
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_records)
    return [
        generate_ftir_record(spec, instrument_cfg, child)
        for child in children
    ]
