"""Dataset generator: virtual instrument + forward physics -> noisy records.

Signal chain (plan §5), per record:

1. Sample concrete instrument parameters from the virtual-instrument config
   distributions (per-record RNG stream).
2. Sample gas conditions (concentration, T/P jitter).
3. Forward physics: absorbance on the (nonlinear) scan frequency axis.
4. Instrument effects: intensity ramp, etalon fringes, baseline, laser
   linewidth convolution, transmission.
5. Detector: white + 1/f noise, gain nonlinearity, ADC quantization.
6. WMS instruments additionally run the lock-in demodulation chain.

Reproducibility: a master seed spawns per-record independent child streams
via numpy SeedSequence.spawn — same (generator version, master seed, configs)
reproduces every record bit-for-bit, and any single record can be regenerated
without generating its predecessors. All sampled parameters are written to
``provenance.noise_config``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from . import __version__
from .instrument.detector import (
    adc_quantize,
    clock_jitter,
    dark_current_noise,
    gain_nonlinearity,
    one_over_f_noise,
    periodic_interference,
    shot_noise,
    speckle_noise,
    thermal_noise_scale,
    white_noise,
)
from .instrument.electronics import detector_responsivity, rin_noise, tia_bandwidth_filter
from .instrument.environment import jittered_conditions
from .instrument.etalon import multi_etalon_transmission, multipass_etalon_transmission
from .instrument.laser import (
    apply_mode_hop,
    intensity_ramp,
    linewidth_convolve,
    scan_frequency_axis,
)
from .instrument.optics import (
    baseline_polynomial,
    beam_wander,
    correlated_baseline_drift,
    gas_flow_turbulence,
    window_contamination,
)
from .instrument.sampling import sample_instrument
from .physics.absorption import absorption_coefficient, rosenkranz_line_mixing
from .physics.continuum import h2o_continuum_absorbance
from .physics.hitran import MOLECULE_IDS, LineList
from .physics.wms import WMSConfig, simulate_wms


@dataclass
class GenerationSpec:
    """What to generate: gas truth distribution + sampling grid.

    ``interferents`` are background absorbers superposed onto the target
    species' absorbance (Beer-Lambert linearity) but excluded from
    ``labels.species`` -- they are not a prediction target, only a source of
    spectral cross-interference the model must learn to be robust to. Each
    entry is a dict with keys ``molecule`` (str), ``lines`` (LineList) and
    ``concentration_ppm`` (float).
    """

    lines: LineList
    molecule: str = "CH4"
    concentration_ppm_low: float = 1.0
    concentration_ppm_high: float = 1000.0
    log_uniform_concentration: bool = True
    temperature_K: float = 296.0
    pressure_atm: float = 1.0
    path_length_m: float = 10.0
    matrix_gas: str = "N2"
    n_points: int = 2000
    extra_conditions: dict = field(default_factory=dict)
    interferents: list[dict] = field(default_factory=list)


def sample_concentration(spec: GenerationSpec, rng: np.random.Generator) -> float:
    """Sample one concentration [ppm] from ``spec``'s truth distribution.

    ``low == high`` is a degenerate point distribution (time-series mode
    fixes one true concentration per series): ``rng.uniform(low, high)``
    computes ``low + (high - low) * u``, and IEEE754 subtraction of two
    equal finite floats is always exactly 0.0, so this returns exactly
    ``low`` for every draw, not merely approximately -- callers that detect
    series boundaries by truth-concentration equality depend on this.
    """
    lo, hi = spec.concentration_ppm_low, spec.concentration_ppm_high
    if spec.log_uniform_concentration:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def generate_record(
    spec: GenerationSpec,
    instrument_cfg: dict,
    seed_seq: np.random.SeedSequence,
    scan_time_s: float = 0.0,
    frozen_instrument: dict | None = None,
) -> dict:
    """Generate one record: returns {'meta': <schema record>, 'arrays': {...}}.

    ``scan_time_s`` is the wall-clock time of this scan since instrument
    start; it drives slow time-dependent effects (etalon phase drift,
    transmittance decay). ``frozen_instrument`` reuses a previously sampled
    concrete instrument (time-series mode: one physical instrument, many
    consecutive scans) instead of re-sampling the config distributions.
    """
    rng = np.random.default_rng(seed_seq)
    if frozen_instrument is None:
        inst = sample_instrument(instrument_cfg, rng)
    else:
        inst = frozen_instrument
        sample_instrument(instrument_cfg, rng)  # keep stream layout identical
    technique = inst["technique"]

    concentration_ppm = sample_concentration(spec, rng)
    env = inst.get("environment", {})
    temperature_K, pressure_atm = jittered_conditions(
        rng,
        env.get("temperature_K", spec.temperature_K),
        env.get("pressure_atm", spec.pressure_atm),
        env.get("temperature_jitter_K", 0.0),
        env.get("pressure_jitter_atm", 0.0),
    )

    laser = inst["laser"]
    det = inst.get("detector", {})
    optics = inst.get("optics", {})
    n = spec.n_points
    ramp = np.arange(n) / n  # normalized sawtooth position in [0, 1)

    # --- frequency axis with scan nonlinearity ---
    nu = scan_frequency_axis(
        ramp,
        laser["center_wavenumber_cm1"],
        laser["scan_range_cm1"],
        laser.get("scan_nonlinearity_poly_cm1"),
        tuning_model=laser.get("tuning_model"),
        tuning_params=laser.get("tuning_params"),
    )
    mode_hop_prob = laser.get("mode_hop_probability", 0.0)
    if mode_hop_prob > 0:
        nu = apply_mode_hop(
            nu, rng, mode_hop_prob, laser.get("mode_hop_size_cm1", 0.3)
        )

    # --- forward physics: clean absorbance ---
    layers = inst.get("gas_path_layers")
    if layers and len(layers) > 1:
        absorbance = np.zeros_like(nu, dtype=np.float64)
        for layer in layers:
            l_frac = layer.get("length_fraction", 1.0 / len(layers))
            l_T = layer.get("temperature_K", temperature_K)
            l_P = layer.get("pressure_atm", pressure_atm)
            l_x = layer.get("concentration_scale", 1.0)
            a_layer = absorption_coefficient(
                nu, spec.lines,
                mole_fraction=concentration_ppm * 1e-6 * l_x,
                temperature_K=l_T, pressure_atm=l_P,
            )
            absorbance = absorbance + a_layer * spec.path_length_m * l_frac * 100.0
    else:
        alpha = absorption_coefficient(
            nu, spec.lines,
            mole_fraction=concentration_ppm * 1e-6,
            temperature_K=temperature_K, pressure_atm=pressure_atm,
        )
        absorbance = alpha * spec.path_length_m * 100.0
    physics_cfg = inst.get("physics", {})
    if physics_cfg.get("line_mixing", False):
        mixing_alpha = rosenkranz_line_mixing(
            nu, spec.lines, temperature_K, pressure_atm,
            mole_fraction=concentration_ppm * 1e-6,
        )
        absorbance = absorbance + mixing_alpha * spec.path_length_m * 100.0
    h2o_frac = physics_cfg.get("h2o_continuum_fraction", 0.0)
    if h2o_frac > 0:
        absorbance = absorbance + h2o_continuum_absorbance(
            nu, h2o_frac, temperature_K, pressure_atm, spec.path_length_m,
        )
    sampled_interferents = []
    for interf in spec.interferents:
        if "concentration_ppm" in interf:
            i_conc = interf["concentration_ppm"]
        else:
            lo = interf["concentration_ppm_low"]
            hi = interf["concentration_ppm_high"]
            if interf.get("log_uniform", False):
                i_conc = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                i_conc = float(rng.uniform(lo, hi))
        sampled_interferents.append({
            "molecule": interf["molecule"],
            "lines": interf["lines"],
            "concentration_ppm": i_conc,
        })
        alpha_interf = absorption_coefficient(
            nu,
            interf["lines"],
            mole_fraction=i_conc * 1e-6,
            temperature_K=temperature_K,
            pressure_atm=pressure_atm,
        )
        absorbance += alpha_interf * spec.path_length_m * 100.0
    step_est = abs(laser["scan_range_cm1"]) / n
    absorbance_measured = linewidth_convolve(
        absorbance, step_est, laser.get("linewidth_MHz", 0.0)
    )

    # --- optical chain ---
    intensity0 = intensity_ramp(
        ramp,
        1.0,
        optics.get("intensity_ramp_slope_rel", 0.0),
        optics.get("intensity_ramp_curvature_rel", 0.0),
    )
    baseline_coeffs_sigma = optics.get("baseline_drift_coeffs_sigma")
    if baseline_coeffs_sigma:
        baseline, _ou_coeffs = correlated_baseline_drift(
            rng, n, baseline_coeffs_sigma, tau_scans=optics.get("baseline_drift_tau", 50.0),
        )
    else:
        baseline = baseline_polynomial(ramp, optics.get("baseline_poly_rel", []) or [])
    etalons = inst.get("etalons", []) or []
    multipass = inst.get("multipass_etalon")
    if multipass:
        fringes = multipass_etalon_transmission(
            nu,
            int(multipass.get("n_passes", 20)),
            multipass["base_fsr_cm1"],
            multipass["base_amplitude_rel"],
            multipass.get("phase_rad", 0.0),
            multipass.get("amplitude_decay", 0.85),
        )
    elif etalons:
        fringes = multi_etalon_transmission(nu, etalons, t_s=scan_time_s)
    else:
        fringes = 1.0
    decay = 1.0 - optics.get("transmittance_drift_rel_per_s", 0.0) * scan_time_s
    contam_rel = optics.get("window_contamination_rel", 0.0)
    window_trans = (
        window_contamination(nu, contam_rel, optics.get("window_spectral_slope", 0.0))
        if contam_rel
        else 1.0
    )
    bw_sigma = optics.get("beam_wander_sigma_rel", 0.0)
    wander = (
        beam_wander(rng, n, bw_sigma, optics.get("beam_wander_cutoff_norm", 0.05))
        if bw_sigma
        else 1.0
    )
    turb_sigma = optics.get("gas_flow_turbulence_sigma_rel", 0.0)
    turbulence = (
        gas_flow_turbulence(rng, n, turb_sigma, optics.get("gas_flow_turbulence_cutoff", 0.1))
        if turb_sigma
        else 1.0
    )
    transmitted = (
        intensity0
        * baseline
        * fringes
        * window_trans
        * wander
        * turbulence
        * max(decay, 0.0)
        * np.exp(-absorbance_measured)
    )

    # --- laser RIN (multiplicative, before detection) ---
    rin_dBc = det.get("rin_dBc_Hz")
    if rin_dBc is not None:
        scan_rate_Hz = laser.get("scan_rate_Hz", 100.0)
        rin_bw = det.get("rin_bandwidth_Hz", scan_rate_Hz * n)
        rin_fs = scan_rate_Hz * n
        transmitted = transmitted * (1.0 + rin_noise(rng, n, rin_dBc, rin_bw, rin_fs))

    # --- detector responsivity (wavelength-dependent) ---
    resp_cutoff = det.get("responsivity_cutoff_cm1")
    if resp_cutoff is not None:
        resp = detector_responsivity(
            nu, resp_cutoff,
            det.get("peak_responsivity", 1.0),
            det.get("responsivity_rolloff_cm1", 200.0),
        )
        transmitted = transmitted * resp

    # --- detector chain ---
    shot_gain = det.get("shot_noise_gain", 0.0)
    if shot_gain:
        transmitted = transmitted + shot_noise(rng, transmitted, shot_gain)
    sigma_w = det.get("white_noise_rel", 0.0)
    if sigma_w:
        det_temp = det.get("detector_temperature_K")
        if det_temp is not None:
            sigma_w = sigma_w * thermal_noise_scale(
                det_temp, det.get("reference_temperature_K", 296.0)
            )
        transmitted = transmitted + white_noise(rng, n, sigma_w)
    sigma_dark = det.get("dark_current_sigma_rel", 0.0)
    if sigma_dark:
        transmitted = transmitted + dark_current_noise(
            rng,
            n,
            sigma_dark,
            det.get("detector_temperature_K", temperature_K),
            det.get("reference_temperature_K", 296.0),
            det.get("dark_current_activation_eV", 0.37),
        )
    sigma_f = det.get("one_over_f_sigma_rel", 0.0)
    if sigma_f:
        transmitted = transmitted + one_over_f_noise(
            rng, n, sigma_f, det.get("one_over_f_slope", 1.0)
        )
    emi_freqs = det.get("emi_frequencies_Hz")
    emi_amps = det.get("emi_amplitudes_rel")
    if emi_freqs and emi_amps:
        scan_rate_Hz = laser.get("scan_rate_Hz", 100.0)
        transmitted = transmitted + periodic_interference(
            rng, n, emi_freqs, emi_amps, scan_rate_Hz, n,
        )
    speckle_sigma = det.get("speckle_noise_sigma", 0.0)
    if speckle_sigma:
        transmitted = transmitted + speckle_noise(
            rng, n, speckle_sigma, int(det.get("speckle_correlation_length", 10)),
        )
    gnl = det.get("gain_nonlinearity_rel", 0.0)
    if gnl:
        transmitted = gain_nonlinearity(
            transmitted, gnl,
            cubic_rel=det.get("gain_cubic_rel", 0.0),
            saturation_level=det.get("saturation_level", 0.0),
        )
    tia_bw = det.get("tia_bandwidth_Hz")
    if tia_bw is not None:
        scan_rate_Hz = laser.get("scan_rate_Hz", 100.0)
        transmitted = tia_bandwidth_filter(transmitted, tia_bw, scan_rate_Hz * n)
    bits = det.get("adc_bits", 0)
    if bits:
        transmitted = adc_quantize(transmitted, int(round(bits)), full_scale=1.5)
    jitter_rel = det.get("clock_jitter_rel", 0.0)
    if jitter_rel:
        jitter = clock_jitter(rng, n, jitter_rel)
        indices = np.arange(n, dtype=np.float64) + jitter
        indices = np.clip(indices, 0, n - 1)
        transmitted = np.interp(indices, np.arange(n, dtype=np.float64), transmitted)

    ref_cfg = inst.get("reference_channel")
    if ref_cfg:
        ref_signal = intensity0 * baseline * max(decay, 0.0) * wander * turbulence
        ref_sigma = ref_cfg.get("noise_sigma_rel", sigma_w)
        if ref_sigma:
            ref_signal = ref_signal + white_noise(rng, n, ref_sigma)
        transmitted = transmitted / np.clip(ref_signal, 1e-10, None)

    record_id = str(uuid.UUID(bytes=rng.bytes(16), version=4))
    arrays = {"raw_scan": transmitted, "absorbance_clean": absorbance}
    signals: dict = {
        "raw_scan": {
            "array_ref": f"/records/{record_id}/raw_scan",
            "n_samples": n,
            "sampling_rate_Hz": float(laser.get("scan_rate_Hz", 100.0)) * n,
        },
        "absorbance": {
            "array_ref": f"/records/{record_id}/absorbance_clean",
            "n_samples": n,
            "wavenumber_axis": {
                "start_cm1": float(nu[0]),
                "step_cm1": float((nu[-1] - nu[0]) / (n - 1)),
            },
        },
    }

    # --- WMS branch: lock-in demodulation of a modulated time-domain chain ---
    mod = inst.get("modulation")
    if technique == "TDLAS-WMS":
        if not mod:
            raise ValueError("WMS instrument config lacks 'modulation' section")
        f_m = mod["frequency_Hz"]
        fs = f_m * 50.0
        scan_rate = laser.get("scan_rate_Hz", 10.0)
        # sample_instrument() casts every leaf (incl. list items) to float, so
        # cast back to int here -- f"x_{h}f" must key off "1", not "1.0".
        harmonics_cfg = mod.get("harmonics", [1, 2])
        harmonics = tuple(int(h) for h in harmonics_cfg)
        cfg = WMSConfig(
            modulation_frequency_Hz=f_m,
            modulation_depth_cm1=mod["depth_cm1"],
            sampling_rate_Hz=fs,
            duration_s=1.0 / scan_rate,
            center_wavenumber_cm1=laser["center_wavenumber_cm1"],
            scan_range_cm1=laser["scan_range_cm1"],
            scan_rate_Hz=scan_rate,
            im_i0_rel=inst.get("laser", {}).get("ram_amplitude_rel", 0.0),
            lockin_phase_rad=mod.get("lockin_phase_error_rad", 0.0),
            lowpass_cutoff_Hz=mod.get("lowpass_cutoff_Hz", 0.0),
        )

        def absorbance_fn(nu_arr):
            a = absorption_coefficient(
                np.asarray(nu_arr, dtype=float),
                spec.lines,
                mole_fraction=concentration_ppm * 1e-6,
                temperature_K=temperature_K,
                pressure_atm=pressure_atm,
            )
            total = a * spec.path_length_m * 100.0
            for interf in sampled_interferents:
                a_i = absorption_coefficient(
                    np.asarray(nu_arr, dtype=float),
                    interf["lines"],
                    mole_fraction=interf["concentration_ppm"] * 1e-6,
                    temperature_K=temperature_K,
                    pressure_atm=pressure_atm,
                )
                total += a_i * spec.path_length_m * 100.0
            return total

        etalon_fn = None
        if etalons:
            def etalon_fn(nu_arr, _et=etalons, _st=scan_time_s):
                return multi_etalon_transmission(nu_arr, _et, t_s=_st)
        out = simulate_wms(cfg, absorbance_fn, harmonics=harmonics,
                           etalon_transmission_of_nu=etalon_fn)
        n_t = len(out["t_s"])
        noisy = out["intensity"]
        if sigma_w:
            noisy = noisy + white_noise(rng, n_t, sigma_w)
        if sigma_f:
            noisy = noisy + one_over_f_noise(rng, n_t, sigma_f, det.get("one_over_f_slope", 1.0))
        from .physics.wms import lockin_demodulate

        stride = max(1, n_t // n)
        demod_r = {}
        for h in harmonics:
            x, y = lockin_demodulate(
                noisy, out["t_s"], f_m, h, cfg.lockin_phase_rad,
                cfg.lowpass_cutoff_Hz, fs,
            )
            key = f"demod_{h}f"
            arrays[key] = x[::stride][:n]
            demod_r[h] = np.hypot(x, y)[::stride][:n]
            signals[key] = {
                "array_ref": f"/records/{record_id}/{key}",
                "n_samples": int(len(arrays[key])),
                "lowpass_cutoff_Hz": float(cfg.lowpass_cutoff_Hz or f_m / 10.0),
            }
        if 1 in harmonics and 2 in harmonics:
            from .physics.wms import wms_2f_1f_ratio
            arrays["ratio_2f1f"] = wms_2f_1f_ratio(demod_r[2], demod_r[1])
            signals["ratio_2f1f"] = {
                "array_ref": f"/records/{record_id}/ratio_2f1f",
                "n_samples": int(len(arrays["ratio_2f1f"])),
            }

    meta = {
        "record_id": record_id,
        "schema_version": "0.2",
        "data_origin": "simulated",
        "technique": technique,
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
        "signals": signals,
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
                        for interf in sampled_interferents
                    ]
                }
                if sampled_interferents
                else {}
            ),
        },
        "instrument": {
            "laser": {
                "center_wavenumber_cm1": float(laser["center_wavenumber_cm1"]),
                "scan_range_cm1": float(laser["scan_range_cm1"]),
                **(
                    {"scan_rate_Hz": float(laser["scan_rate_Hz"])}
                    if "scan_rate_Hz" in laser
                    else {}
                ),
                **(
                    {"linewidth_MHz": float(laser["linewidth_MHz"])}
                    if "linewidth_MHz" in laser
                    else {}
                ),
            },
            **(
                {
                    "modulation": {
                        "frequency_Hz": float(mod["frequency_Hz"]),
                        "depth_cm1": float(mod["depth_cm1"]),
                    }
                }
                if technique == "TDLAS-WMS" and mod
                else {}
            ),
            "detector": {"type": det.get("type", "InGaAs photodiode (simulated)")},
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


def generate_dataset(
    spec: GenerationSpec,
    instrument_cfg: dict,
    n_records: int,
    master_seed: int,
) -> list[dict]:
    """Generate ``n_records`` reproducible records (list of {'meta','arrays'})."""
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_records)
    return [generate_record(spec, instrument_cfg, child) for child in children]


def generate_time_series(
    spec: GenerationSpec,
    instrument_cfg: dict,
    n_scans: int,
    master_seed: int,
    scan_interval_s: float,
) -> list[dict]:
    """Consecutive scans of ONE instrument realization (drift/Allan studies).

    The instrument parameters are sampled once (first child stream) and then
    frozen; per-scan noise stays independent while slow effects (etalon phase
    drift, transmittance decay) evolve with wall-clock time k*scan_interval_s.
    """
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(n_scans + 1)
    inst = sample_instrument(instrument_cfg, np.random.default_rng(children[0]))
    return [
        generate_record(
            spec,
            instrument_cfg,
            children[k + 1],
            scan_time_s=k * scan_interval_s,
            frozen_instrument=inst,
        )
        for k in range(n_scans)
    ]
