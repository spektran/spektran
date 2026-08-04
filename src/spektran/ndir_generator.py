"""NDIR dataset generator: broadband source + bandpass filters -> noisy records.

Signal chain (per record):

1. Sample concrete instrument parameters from the virtual-instrument config
   distributions (per-record RNG stream).
2. Sample gas conditions (concentration, T/P jitter).
3. Forward physics: clean active and reference signals via simulate_ndir
   (from physics/ndir.py).
4. Source noise: multiplicative source intensity fluctuation (common-mode
   to both channels).
5. Detector noise: independent white noise to each channel (thermopile /
   pyroelectric noise characteristics).
6. Source drift: slow temperature drift of the IR emitter.
7. Compute noisy ratio = noisy_active / noisy_reference.

Reproducibility: a master seed spawns per-record independent child streams
via numpy SeedSequence.spawn — same (generator version, master seed, configs)
reproduces every record bit-for-bit.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .instrument.detector import thermal_noise_scale, white_noise
from .instrument.environment import jittered_conditions
from .instrument.sampling import sample_instrument
from .physics.hitran import MOLECULE_IDS, LineList
from .physics.ndir import simulate_ndir


@dataclass
class NDIRGenerationSpec:
    """What to generate for NDIR: gas truth + instrument config."""

    lines: LineList
    molecule: str = "CH4"
    concentration_ppm_low: float = 1.0
    concentration_ppm_high: float = 1000.0
    log_uniform_concentration: bool = True
    temperature_K: float = 296.0
    pressure_atm: float = 1.0
    path_length_m: float = 0.10
    matrix_gas: str = "N2"
    n_integration_points: int = 500
    extra_conditions: dict = field(default_factory=dict)
    interferents: list[dict] = field(default_factory=list)


def _sample_concentration(
    spec: NDIRGenerationSpec,
    rng: np.random.Generator,
) -> float:
    lo, hi = spec.concentration_ppm_low, spec.concentration_ppm_high
    if spec.log_uniform_concentration:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def generate_ndir_record(
    spec: NDIRGenerationSpec,
    instrument_cfg: dict,
    seed_seq: np.random.SeedSequence,
    scan_time_s: float = 0.0,
    frozen_instrument: dict | None = None,
) -> dict:
    """Generate one NDIR measurement record.

    Signal chain:
    1. Sample instrument parameters (source T, filter specs, noise levels)
    2. Sample gas conditions (concentration, T/P jitter)
    3. Forward physics: compute clean active and reference signals via
       simulate_ndir (from physics/ndir.py)
    4. Source noise: add multiplicative source intensity fluctuation
       (common-mode to both channels)
    5. Detector noise: add independent white noise to each channel
       (thermopile/pyroelectric noise characteristics)
    6. Source drift: slow temperature drift of the IR emitter
    7. Compute noisy ratio = noisy_active / noisy_reference

    Returns {'meta': <schema record>, 'arrays': {...}} with:
    - arrays['active_channel']: scalar float (noisy active detector signal)
    - arrays['reference_channel']: scalar float (noisy reference signal)
    - arrays['ratio']: scalar float (noisy ratio)
    - arrays['ratio_clean']: scalar float (clean ratio, no noise)
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

    source = inst.get("source", {})
    filters = inst.get("filters", {})
    det = inst.get("detector", {})

    source_T_base = source.get("temperature_K", 800.0)
    drift_K_per_s = source.get(
        "temperature_drift_K_per_s", 0.0,
    )
    source_T_actual = source_T_base + drift_K_per_s * scan_time_s

    filter_shape = filters.get("shape", "gaussian")

    clean = simulate_ndir(
        lines=spec.lines,
        molecule=spec.molecule,
        concentration_ppm=concentration_ppm,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
        path_length_m=spec.path_length_m,
        source_temperature_K=source_T_actual,
        active_filter_center_cm1=filters.get(
            "active_center_cm1", 3018.0,
        ),
        active_filter_fwhm_cm1=filters.get(
            "active_fwhm_cm1", 100.0,
        ),
        reference_filter_center_cm1=filters.get(
            "reference_center_cm1", 2500.0,
        ),
        reference_filter_fwhm_cm1=filters.get(
            "reference_fwhm_cm1", 100.0,
        ),
        filter_shape=filter_shape,
        n_integration_points=spec.n_integration_points,
        interferents=spec.interferents or None,
    )

    ratio_clean = clean["ratio"]
    active = clean["active_signal"]
    reference = clean["reference_signal"]

    fluctuation_sigma = source.get("intensity_fluctuation_rel", 0.0)
    if fluctuation_sigma:
        factor = 1.0 + rng.normal(0.0, fluctuation_sigma)
        active *= factor
        reference *= factor

    sigma_w = det.get("white_noise_rel", 0.0)
    if sigma_w:
        det_temp = det.get("detector_temperature_K")
        if det_temp is not None:
            sigma_w = sigma_w * thermal_noise_scale(
                det_temp,
                det.get("reference_temperature_K", 296.0),
            )
        active += white_noise(rng, 1, sigma_w)[0]
        reference += white_noise(rng, 1, sigma_w)[0]

    ratio = active / reference

    record_id = str(uuid.UUID(bytes=rng.bytes(16), version=4))

    arrays = {
        "active_channel": float(active),
        "reference_channel": float(reference),
        "ratio": float(ratio),
        "ratio_clean": float(ratio_clean),
    }

    meta = {
        "record_id": record_id,
        "schema_version": "0.2",
        "data_origin": "simulated",
        "technique": "NDIR",
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
                "spawn_key": [
                    int(k) for k in seed_seq.spawn_key
                ],
            },
        },
        "signals": {
            "active_channel": {
                "array_ref": (
                    f"/records/{record_id}/active_channel"
                ),
                "n_samples": 1,
            },
            "reference_channel": {
                "array_ref": (
                    f"/records/{record_id}/reference_channel"
                ),
                "n_samples": 1,
            },
            "ratio": {
                "array_ref": f"/records/{record_id}/ratio",
                "n_samples": 1,
            },
        },
        "labels": {
            "species": [
                {
                    "molecule": spec.molecule,
                    "hitran_molecule_id": MOLECULE_IDS[
                        spec.molecule
                    ],
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
                            "hitran_molecule_id": MOLECULE_IDS[
                                interf["molecule"]
                            ],
                            "concentration_ppm": interf[
                                "concentration_ppm"
                            ],
                        }
                        for interf in spec.interferents
                    ]
                }
                if spec.interferents
                else {}
            ),
        },
        "instrument": {
            "source": {
                "temperature_K": float(source_T_actual),
            },
            "filters": {
                "active_center_cm1": float(
                    filters.get("active_center_cm1", 3018.0),
                ),
                "active_fwhm_cm1": float(
                    filters.get("active_fwhm_cm1", 100.0),
                ),
                "reference_center_cm1": float(
                    filters.get("reference_center_cm1", 2500.0),
                ),
                "reference_fwhm_cm1": float(
                    filters.get("reference_fwhm_cm1", 100.0),
                ),
                "shape": filter_shape,
            },
            "detector": {
                "type": det.get(
                    "type",
                    "thermopile (simulated)",
                ),
            },
            "target_lines": [
                {
                    "hitran_molecule_id": MOLECULE_IDS[
                        spec.molecule
                    ],
                    "wavenumber_cm1": float(w),
                }
                for w in spec.lines.nu0_cm1
            ],
        },
    }
    return {"meta": meta, "arrays": arrays}


def generate_ndir_dataset(
    spec: NDIRGenerationSpec,
    instrument_cfg: dict,
    n_records: int,
    master_seed: int,
) -> list[dict]:
    """Generate ``n_records`` reproducible NDIR records."""
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_records)
    return [
        generate_ndir_record(spec, instrument_cfg, child)
        for child in children
    ]


def generate_ndir_time_series(
    spec: NDIRGenerationSpec,
    instrument_cfg: dict,
    n_scans: int,
    master_seed: int,
    scan_interval_s: float,
) -> list[dict]:
    """Consecutive NDIR measurements of ONE instrument realization.

    The instrument parameters are sampled once (first child stream)
    and then frozen; per-scan noise stays independent while slow
    effects (source temperature drift) evolve with wall-clock time
    k * scan_interval_s.
    """
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_scans + 1)
    inst = sample_instrument(
        instrument_cfg, np.random.default_rng(children[0]),
    )
    return [
        generate_ndir_record(
            spec,
            instrument_cfg,
            children[k + 1],
            scan_time_s=k * scan_interval_s,
            frozen_instrument=inst,
        )
        for k in range(n_scans)
    ]
