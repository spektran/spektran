"""Tests for FTIR forward model, noise chain, and generator."""

from __future__ import annotations

import numpy as np
import pytest

from spektran.physics.ftir import (
    apodization_function,
    generate_interferogram,
    interferogram_to_spectrum,
    simulate_ftir_spectrum,
    spectral_resolution_cm1,
)
from spektran.instrument.ftir_noise import (
    channel_spectrum,
    detector_noise,
    self_apodization,
    source_fluctuation,
    sampling_error,
    phase_error,
)
from spektran.physics.hitran import demo_ch4_2nu3


class TestApodization:
    def test_boxcar_is_ones(self):
        opd = np.linspace(-10, 10, 100)
        w = apodization_function(opd, 10.0, "boxcar")
        np.testing.assert_allclose(w, 1.0)

    def test_triangular_endpoints(self):
        opd = np.linspace(-10, 10, 201)
        w = apodization_function(opd, 10.0, "triangular")
        assert w[0] == pytest.approx(0.0, abs=0.01)
        assert w[100] == pytest.approx(1.0, abs=0.01)
        assert w[-1] == pytest.approx(0.0, abs=0.01)

    def test_happ_genzel_center(self):
        opd = np.array([0.0])
        w = apodization_function(opd, 10.0, "happ_genzel")
        assert w[0] == pytest.approx(1.0, abs=1e-10)

    def test_norton_beer_medium_range(self):
        opd = np.linspace(-10, 10, 100)
        w = apodization_function(opd, 10.0, "norton_beer_medium")
        assert np.all(w >= 0)
        assert np.all(w <= 1.1)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown apodization"):
            apodization_function(np.array([0.0]), 10.0, "fake")


class TestSpectralResolution:
    def test_standard(self):
        assert spectral_resolution_cm1(45.0) == pytest.approx(1.0 / 90.0, rel=1e-10)

    def test_low_res(self):
        assert spectral_resolution_cm1(0.5) == pytest.approx(1.0, rel=1e-10)


class TestInterferogram:
    def test_roundtrip_flat(self):
        nu = np.linspace(6000, 6100, 500)
        spectrum = np.ones_like(nu)
        opd, igram = generate_interferogram(nu, spectrum, max_opd_cm=5.0, n_opd_points=2048)
        assert len(opd) == 2048
        assert igram[len(igram) // 2] > 0

    def test_cosine_spectrum(self):
        nu = np.linspace(6000, 6100, 200)
        spectrum = 1.0 + 0.1 * np.cos(2 * np.pi * nu / 10.0)
        opd, igram = generate_interferogram(nu, spectrum, max_opd_cm=2.0, n_opd_points=512)
        assert len(igram) == 512
        assert np.max(np.abs(igram)) > 0


class TestInterferogramToSpectrum:
    def test_recovery_shape(self):
        nu = np.linspace(6000, 6100, 200)
        spectrum = np.ones_like(nu)
        opd, igram = generate_interferogram(nu, spectrum, max_opd_cm=5.0, n_opd_points=2048)
        nu_out = np.linspace(6000, 6100, 100)
        nu_rec, spec_rec = interferogram_to_spectrum(opd, igram, nu_out, "boxcar")
        assert len(spec_rec) == 100
        assert np.all(spec_rec >= 0)


class TestSimulateFTIRSpectrum:
    def test_output_keys(self):
        lines = demo_ch4_2nu3()
        result = simulate_ftir_spectrum(
            lines=lines,
            molecule="CH4",
            concentration_ppm=100.0,
            temperature_K=296.0,
            pressure_atm=1.0,
            path_length_m=10.0,
            max_opd_cm=5.0,
            n_output_points=200,
            n_hires_points=2000,
        )
        assert "nu_cm1" in result
        assert "spectrum" in result
        assert "resolution_cm1" in result
        assert "concentration_ppm" in result
        assert len(result["nu_cm1"]) == 200
        assert len(result["spectrum"]) == 200

    def test_zero_concentration(self):
        lines = demo_ch4_2nu3()
        result = simulate_ftir_spectrum(
            lines=lines,
            molecule="CH4",
            concentration_ppm=0.0,
            temperature_K=296.0,
            pressure_atm=1.0,
            path_length_m=10.0,
            max_opd_cm=5.0,
            n_output_points=100,
            n_hires_points=1000,
        )
        assert np.all(result["spectrum"] > 0)

    def test_higher_concentration_more_absorption(self):
        lines = demo_ch4_2nu3()
        r_low = simulate_ftir_spectrum(
            lines=lines, molecule="CH4", concentration_ppm=10.0,
            temperature_K=296.0, pressure_atm=1.0, path_length_m=10.0,
            max_opd_cm=5.0, n_output_points=200, n_hires_points=2000,
        )
        r_high = simulate_ftir_spectrum(
            lines=lines, molecule="CH4", concentration_ppm=200.0,
            temperature_K=296.0, pressure_atm=1.0, path_length_m=10.0,
            max_opd_cm=5.0, n_output_points=200, n_hires_points=2000,
        )
        assert np.min(r_high["spectrum"]) < np.min(r_low["spectrum"])

    def test_resolution_matches_opd(self):
        lines = demo_ch4_2nu3()
        result = simulate_ftir_spectrum(
            lines=lines, molecule="CH4", concentration_ppm=10.0,
            temperature_K=296.0, pressure_atm=1.0, path_length_m=10.0,
            max_opd_cm=45.0, n_output_points=200, n_hires_points=2000,
        )
        expected_res = 1.0 / (2.0 * 45.0)
        assert result["resolution_cm1"] == pytest.approx(expected_res, rel=1e-10)


class TestFTIRNoise:
    def test_detector_noise_shape(self):
        rng = np.random.default_rng(42)
        n = detector_noise(rng, 500, 1e-4)
        assert n.shape == (500,)
        assert np.std(n) > 0

    def test_source_fluctuation_shape(self):
        rng = np.random.default_rng(42)
        f = source_fluctuation(rng, 500, 1e-3)
        assert f.shape == (500,)

    def test_phase_error_shape(self):
        rng = np.random.default_rng(42)
        p = phase_error(rng, 500, 0.01)
        assert p.shape == (500,)

    def test_channel_spectrum_sinusoidal(self):
        nu = np.linspace(6000, 6100, 500)
        cs = channel_spectrum(nu, 0.001, 5.0, 0.0)
        assert cs.shape == (500,)
        assert np.max(np.abs(cs)) == pytest.approx(0.001, rel=0.01)

    def test_sampling_error_shape(self):
        rng = np.random.default_rng(42)
        opd = np.linspace(-5, 5, 1000)
        perturbed = sampling_error(rng, opd, 1e-7)
        assert perturbed.shape == opd.shape
        np.testing.assert_allclose(perturbed, opd, atol=1e-5)

    def test_self_apodization_small_angle(self):
        nu = np.array([6050.0])
        sa = self_apodization(nu, 45.0, 1e-4)
        assert sa[0] == pytest.approx(1.0, abs=1e-4)


class TestFTIRGenerator:
    def test_single_record(self):
        from spektran.ftir_generator import generate_ftir_record, FTIRGenerationSpec
        from spektran.instrument.sampling import load_instrument_config
        from pathlib import Path

        lines = demo_ch4_2nu3()
        spec = FTIRGenerationSpec(
            lines=lines,
            concentration_ppm_low=1.0,
            concentration_ppm_high=100.0,
            n_output_points=100,
            wavenumber_start_cm1=6000.0,
            wavenumber_end_cm1=6100.0,
        )

        repo = Path(__file__).resolve().parents[1]
        inst_cfg = load_instrument_config(repo / "configs/instruments/vi-ftir-lab-39.yaml")
        seed = np.random.SeedSequence(12345)

        record = generate_ftir_record(spec, inst_cfg, seed)
        assert "meta" in record
        assert "arrays" in record
        assert record["meta"]["technique"] == "FTIR"
        assert record["arrays"]["ftir_spectrum"].shape == (100,)
        assert record["arrays"]["ftir_spectrum_clean"].shape == (100,)
        assert record["arrays"]["nu_cm1"].shape == (100,)
        assert record["meta"]["labels"]["species"][0]["concentration_ppm"] > 0

    def test_reproducibility(self):
        from spektran.ftir_generator import generate_ftir_record, FTIRGenerationSpec
        from spektran.instrument.sampling import load_instrument_config
        from pathlib import Path

        lines = demo_ch4_2nu3()
        spec = FTIRGenerationSpec(
            lines=lines,
            concentration_ppm_low=10.0,
            concentration_ppm_high=50.0,
            n_output_points=50,
        )

        repo = Path(__file__).resolve().parents[1]
        inst_cfg = load_instrument_config(repo / "configs/instruments/vi-ftir-lab-39.yaml")

        seed1 = np.random.SeedSequence(99999)
        seed2 = np.random.SeedSequence(99999)

        r1 = generate_ftir_record(spec, inst_cfg, seed1)
        r2 = generate_ftir_record(spec, inst_cfg, seed2)

        np.testing.assert_array_equal(
            r1["arrays"]["ftir_spectrum"], r2["arrays"]["ftir_spectrum"]
        )

    def test_dataset_generation(self):
        from spektran.ftir_generator import generate_ftir_dataset, FTIRGenerationSpec
        from spektran.instrument.sampling import load_instrument_config
        from pathlib import Path

        lines = demo_ch4_2nu3()
        spec = FTIRGenerationSpec(
            lines=lines,
            concentration_ppm_low=1.0,
            concentration_ppm_high=100.0,
            n_output_points=50,
        )

        repo = Path(__file__).resolve().parents[1]
        inst_cfg = load_instrument_config(repo / "configs/instruments/vi-ftir-lab-39.yaml")

        records = generate_ftir_dataset(spec, inst_cfg, 5, master_seed=77777)
        assert len(records) == 5
        concs = [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
        assert len(set(concs)) == 5
