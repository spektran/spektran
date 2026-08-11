"""Tests for CRDS physics engine and generator."""

import numpy as np
import pytest

from spektran.physics.constants import C_CM_PER_S
from spektran.physics.crds import (
    absorption_from_tau,
    cavity_finesse,
    empty_cavity_tau,
    fit_ring_down,
    nea_cm1,
    ring_down_time,
    ring_down_trace,
    simulate_crds_spectrum,
)
from spektran.physics.hitran import demo_ch4_2nu3


class TestRingDownTime:
    def test_empty_cavity(self):
        L_cm = 50.0
        R = 0.99995
        tau0 = ring_down_time(L_cm, R, 0.0)
        expected = L_cm / (C_CM_PER_S * (1 - R))
        assert tau0 == pytest.approx(expected, rel=1e-12)

    def test_with_absorption(self):
        L_cm = 50.0
        R = 0.99995
        alpha = 1e-7
        tau = ring_down_time(L_cm, R, alpha)
        expected = L_cm / (C_CM_PER_S * (1 - R + alpha * L_cm))
        assert tau == pytest.approx(expected, rel=1e-12)

    def test_tau_decreases_with_absorption(self):
        L_cm = 50.0
        R = 0.99995
        tau0 = ring_down_time(L_cm, R, 0.0)
        tau = ring_down_time(L_cm, R, 1e-6)
        assert tau < tau0

    def test_tau_increases_with_reflectivity(self):
        L_cm = 50.0
        tau_low = ring_down_time(L_cm, 0.999, 0.0)
        tau_high = ring_down_time(L_cm, 0.99999, 0.0)
        assert tau_high > tau_low


class TestEmptyCavityTau:
    def test_typical_lab(self):
        tau0 = empty_cavity_tau(50.0, 0.99995)
        assert 20e-6 < tau0 < 50e-6

    def test_matches_ring_down_time(self):
        tau0 = empty_cavity_tau(50.0, 0.99995)
        tau0_alt = ring_down_time(50.0, 0.99995, 0.0)
        assert tau0 == pytest.approx(tau0_alt, rel=1e-15)


class TestAbsorptionFromTau:
    def test_roundtrip(self):
        L_cm = 50.0
        R = 0.99995
        alpha_in = 1e-7
        tau = ring_down_time(L_cm, R, alpha_in)
        tau0 = empty_cavity_tau(L_cm, R)
        alpha_out = absorption_from_tau(tau, tau0, L_cm)
        assert alpha_out == pytest.approx(alpha_in, rel=1e-10)

    def test_zero_absorption(self):
        L_cm = 50.0
        R = 0.99995
        tau0 = empty_cavity_tau(L_cm, R)
        alpha = absorption_from_tau(tau0, tau0, L_cm)
        assert alpha == pytest.approx(0.0, abs=1e-20)

    def test_vectorized(self):
        L_cm = 50.0
        R = 0.99995
        alphas = np.array([0.0, 1e-8, 1e-7, 1e-6])
        taus = np.array([ring_down_time(L_cm, R, a) for a in alphas])
        tau0 = empty_cavity_tau(L_cm, R)
        recovered = absorption_from_tau(taus, tau0, L_cm)
        np.testing.assert_allclose(recovered, alphas, rtol=1e-10)


class TestRingDownTrace:
    def test_pure_exponential(self):
        t = np.linspace(0, 100e-6, 1000)
        tau = 30e-6
        trace = ring_down_trace(t, tau, I0=1.0)
        expected = np.exp(-t / tau)
        np.testing.assert_allclose(trace, expected, rtol=1e-12)

    def test_with_offset(self):
        t = np.array([0.0, 1e-3])
        tau = 30e-6
        trace = ring_down_trace(t, tau, I0=1.0, offset=0.1)
        assert trace[0] == pytest.approx(1.1, rel=1e-10)
        assert trace[-1] == pytest.approx(0.1, rel=1e-5)

    def test_initial_amplitude(self):
        t = np.array([0.0])
        trace = ring_down_trace(t, 30e-6, I0=5.0)
        assert trace[0] == pytest.approx(5.0, rel=1e-12)


class TestCavityFinesse:
    def test_high_reflectivity(self):
        F = cavity_finesse(0.99999)
        assert 300_000 < F < 320_000

    def test_moderate_reflectivity(self):
        F = cavity_finesse(0.999)
        assert 3100 < F < 3200


class TestNEA:
    def test_typical_lab(self):
        val = nea_cm1(50.0, 0.99995, 1e-3)
        assert 1e-12 < val < 1e-8


class TestFitRingDown:
    def test_clean_exponential(self):
        tau_true = 30e-6
        t = np.linspace(0, 5 * tau_true, 500)
        I = ring_down_trace(t, tau_true, I0=1.0)
        tau_fit, I0_fit, _ = fit_ring_down(t, I)
        assert tau_fit == pytest.approx(tau_true, rel=1e-4)
        assert I0_fit == pytest.approx(1.0, rel=1e-3)

    def test_noisy_exponential(self):
        rng = np.random.default_rng(42)
        tau_true = 30e-6
        t = np.linspace(0, 2 * tau_true, 500)
        I = ring_down_trace(t, tau_true, I0=1.0) + rng.normal(0, 0.005, 500)
        I = np.maximum(I, 1e-10)
        tau_fit, _, _ = fit_ring_down(t, I)
        assert tau_fit == pytest.approx(tau_true, rel=0.15)


class TestSimulateCRDSSpectrum:
    def test_basic_output(self):
        lines = demo_ch4_2nu3()
        result = simulate_crds_spectrum(
            lines=lines,
            molecule="CH4",
            concentration_ppm=100.0,
            temperature_K=296.0,
            pressure_atm=1.0,
            cavity_length_m=0.50,
            mirror_reflectivity=0.99995,
            n_spectral_points=50,
        )
        assert "nu_cm1" in result
        assert "tau_spectrum_s" in result
        assert "tau0_s" in result
        assert "alpha_spectrum_cm1" in result
        assert len(result["nu_cm1"]) == 50
        assert len(result["tau_spectrum_s"]) == 50

    def test_tau_less_than_tau0(self):
        lines = demo_ch4_2nu3()
        result = simulate_crds_spectrum(
            lines=lines,
            molecule="CH4",
            concentration_ppm=100.0,
            temperature_K=296.0,
            pressure_atm=1.0,
            cavity_length_m=0.50,
            mirror_reflectivity=0.99995,
            n_spectral_points=50,
        )
        assert np.all(result["tau_spectrum_s"] <= result["tau0_s"] + 1e-15)

    def test_zero_concentration(self):
        lines = demo_ch4_2nu3()
        result = simulate_crds_spectrum(
            lines=lines,
            molecule="CH4",
            concentration_ppm=0.0,
            temperature_K=296.0,
            pressure_atm=1.0,
            cavity_length_m=0.50,
            mirror_reflectivity=0.99995,
            n_spectral_points=50,
        )
        np.testing.assert_allclose(
            result["tau_spectrum_s"], result["tau0_s"], rtol=1e-10,
        )

    def test_higher_concentration_lower_tau(self):
        lines = demo_ch4_2nu3()
        res_low = simulate_crds_spectrum(
            lines=lines, molecule="CH4", concentration_ppm=10.0,
            temperature_K=296.0, pressure_atm=1.0,
            cavity_length_m=0.50, mirror_reflectivity=0.99995,
            n_spectral_points=50,
        )
        res_high = simulate_crds_spectrum(
            lines=lines, molecule="CH4", concentration_ppm=500.0,
            temperature_K=296.0, pressure_atm=1.0,
            cavity_length_m=0.50, mirror_reflectivity=0.99995,
            n_spectral_points=50,
        )
        assert np.mean(res_high["tau_spectrum_s"]) < np.mean(res_low["tau_spectrum_s"])


class TestCRDSGenerator:
    def test_generate_single_record(self):
        from spektran.crds_generator import CRDSGenerationSpec, generate_crds_record

        lines = demo_ch4_2nu3()
        spec = CRDSGenerationSpec(
            lines=lines,
            molecule="CH4",
            n_spectral_points=50,
        )
        inst_cfg = {
            "instrument_config_id": "test-crds",
            "schema_version": "0.2",
            "technique": "CRDS",
            "held_out": False,
            "cavity": {
                "length_m": 0.50,
                "mirror_reflectivity": 0.99995,
                "n_photons_per_ringdown": 1e6,
            },
            "detector": {"noise_sigma_rel": 1e-4},
            "environment": {
                "temperature_K": 296.0,
                "pressure_atm": 1.0,
            },
        }
        seed = np.random.SeedSequence(42)
        child = seed.spawn(1)[0]
        record = generate_crds_record(spec, inst_cfg, child)

        assert "meta" in record
        assert "arrays" in record
        assert record["meta"]["technique"] == "CRDS"
        assert "tau_spectrum" in record["arrays"]
        assert "alpha_spectrum" in record["arrays"]
        assert len(record["arrays"]["tau_spectrum"]) == 50

    def test_generate_dataset(self):
        from spektran.crds_generator import CRDSGenerationSpec, generate_crds_dataset

        lines = demo_ch4_2nu3()
        spec = CRDSGenerationSpec(
            lines=lines, molecule="CH4", n_spectral_points=20,
        )
        inst_cfg = {
            "instrument_config_id": "test-crds",
            "schema_version": "0.2",
            "technique": "CRDS",
            "held_out": False,
            "cavity": {
                "length_m": 0.50,
                "mirror_reflectivity": 0.99995,
                "n_photons_per_ringdown": 1e6,
            },
            "detector": {"noise_sigma_rel": 1e-4},
            "environment": {
                "temperature_K": 296.0,
                "pressure_atm": 1.0,
            },
        }
        records = generate_crds_dataset(spec, inst_cfg, 5, master_seed=123)
        assert len(records) == 5
        concentrations = [
            r["meta"]["labels"]["species"][0]["concentration_ppm"]
            for r in records
        ]
        assert all(0.1 <= c <= 500.0 for c in concentrations)

    def test_reproducibility(self):
        from spektran.crds_generator import CRDSGenerationSpec, generate_crds_dataset

        lines = demo_ch4_2nu3()
        spec = CRDSGenerationSpec(
            lines=lines, molecule="CH4", n_spectral_points=20,
        )
        inst_cfg = {
            "instrument_config_id": "test-crds",
            "schema_version": "0.2",
            "technique": "CRDS",
            "held_out": False,
            "cavity": {
                "length_m": 0.50,
                "mirror_reflectivity": 0.99995,
                "n_photons_per_ringdown": 1e6,
            },
            "detector": {"noise_sigma_rel": 1e-4},
            "environment": {
                "temperature_K": 296.0,
                "pressure_atm": 1.0,
            },
        }
        ds1 = generate_crds_dataset(spec, inst_cfg, 3, master_seed=999)
        ds2 = generate_crds_dataset(spec, inst_cfg, 3, master_seed=999)
        for r1, r2 in zip(ds1, ds2):
            np.testing.assert_array_equal(
                r1["arrays"]["tau_spectrum"],
                r2["arrays"]["tau_spectrum"],
            )
