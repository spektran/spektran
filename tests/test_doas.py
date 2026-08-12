"""Tests for DOAS forward model, noise chain, and generator."""

from __future__ import annotations

import numpy as np
import pytest

from spektran.physics.doas import (
    doas_optical_density,
    mie_extinction,
    number_density,
    polynomial_high_pass,
    rayleigh_cross_section,
    simulate_doas_cross_section,
    simulate_doas_spectrum,
)
from spektran.instrument.doas_noise import (
    dark_current_noise,
    photon_noise,
    readout_noise,
    ring_effect,
    stray_light,
    wavelength_shift,
)


class TestNumberDensity:
    def test_stp(self):
        n = number_density(1e6, 296.0, 1.0)
        assert 2.4e19 < n < 2.6e19

    def test_zero_concentration(self):
        assert number_density(0.0, 296.0, 1.0) == 0.0

    def test_higher_pressure(self):
        n1 = number_density(100.0, 296.0, 1.0)
        n2 = number_density(100.0, 296.0, 2.0)
        assert n2 == pytest.approx(2.0 * n1, rel=1e-10)


class TestRayleigh:
    def test_lambda4_scaling(self):
        sigma_300 = rayleigh_cross_section(np.array([300.0]))[0]
        sigma_600 = rayleigh_cross_section(np.array([600.0]))[0]
        ratio = sigma_300 / sigma_600
        assert ratio == pytest.approx(16.0, rel=0.01)

    def test_positive(self):
        sigma = rayleigh_cross_section(np.linspace(200, 800, 100))
        assert np.all(sigma > 0)


class TestMie:
    def test_angstrom_scaling(self):
        wl = np.array([300.0, 550.0])
        tau = mie_extinction(wl, tau_ref=0.1, lambda_ref_nm=550.0, angstrom_exp=1.3)
        assert tau[0] > tau[1]

    def test_reference_point(self):
        tau = mie_extinction(np.array([550.0]), tau_ref=0.1, lambda_ref_nm=550.0)
        assert tau[0] == pytest.approx(0.1, rel=1e-10)


class TestCrossSection:
    def test_shape(self):
        wl = np.linspace(300, 360, 500)
        sigma = simulate_doas_cross_section(wl)
        assert sigma.shape == (500,)
        assert np.all(sigma >= 0)

    def test_peak_near_center(self):
        wl = np.linspace(300, 360, 500)
        sigma = simulate_doas_cross_section(wl, center_nm=330.0)
        peak_idx = np.argmax(sigma)
        assert 310 < wl[peak_idx] < 350


class TestOpticalDensity:
    def test_basic(self):
        wl = np.linspace(300, 360, 200)
        sigma = simulate_doas_cross_section(wl)
        result = doas_optical_density(
            wl,
            [{"sigma_cm2": sigma, "concentration_ppm": 1.0, "molecule": "SO2"}],
            path_length_m=1000.0,
        )
        assert "od_total" in result
        assert "od_molecular" in result
        assert np.all(result["od_total"] >= 0)

    def test_zero_concentration(self):
        wl = np.linspace(300, 360, 200)
        sigma = simulate_doas_cross_section(wl)
        result = doas_optical_density(
            wl,
            [{"sigma_cm2": sigma, "concentration_ppm": 0.0, "molecule": "SO2"}],
            rayleigh=False, mie_tau_ref=0.0,
        )
        np.testing.assert_allclose(result["od_molecular"], 0.0, atol=1e-30)


class TestPolynomialHighPass:
    def test_removes_polynomial(self):
        x = np.linspace(-1, 1, 500)
        poly = 3 * x**3 + 2 * x**2 + x + 5
        narrow = 0.1 * np.exp(-0.5 * (x / 0.02)**2)
        signal = poly + narrow
        filtered = polynomial_high_pass(signal, poly_order=3)
        assert np.max(np.abs(filtered - narrow)) < 0.05

    def test_preserves_narrow(self):
        x = np.linspace(-1, 1, 500)
        narrow = np.zeros_like(x)
        for i in range(5):
            narrow += np.exp(-0.5 * ((x - 0.1 * i) / 0.01) ** 2)
        filtered = polynomial_high_pass(narrow, poly_order=5)
        assert np.max(np.abs(filtered)) > 0.5 * np.max(np.abs(narrow))


class TestSimulateDOAS:
    def test_output_keys(self):
        wl = np.linspace(300, 360, 200)
        sigma = simulate_doas_cross_section(wl)
        result = simulate_doas_spectrum(wl, sigma, 1.0, path_length_m=1000.0)
        assert "doas_spectrum" in result
        assert "od_total" in result
        assert "transmittance" in result
        assert "concentration_ppm" in result
        assert len(result["doas_spectrum"]) == 200

    def test_higher_conc_larger_signal(self):
        wl = np.linspace(300, 360, 200)
        sigma = simulate_doas_cross_section(wl)
        r_low = simulate_doas_spectrum(
            wl, sigma, 0.1, path_length_m=1000.0,
            rayleigh=False, mie_tau_ref=0.0,
        )
        r_high = simulate_doas_spectrum(
            wl, sigma, 5.0, path_length_m=1000.0,
            rayleigh=False, mie_tau_ref=0.0,
        )
        assert np.max(np.abs(r_high["doas_spectrum"])) > np.max(np.abs(r_low["doas_spectrum"]))


class TestDOASNoise:
    def test_photon_noise_shape(self):
        rng = np.random.default_rng(42)
        signal = np.ones(500) * 0.8
        pn = photon_noise(rng, signal, 1e6)
        assert pn.shape == (500,)

    def test_stray_light_smooth(self):
        rng = np.random.default_rng(42)
        sl = stray_light(rng, 500, 1e-4)
        assert sl.shape == (500,)

    def test_ring_effect_shape(self):
        wl = np.linspace(300, 360, 500)
        re = ring_effect(wl, 0.02, n_lines=8)
        assert re.shape == (500,)
        assert np.max(re) == pytest.approx(0.02, rel=0.1)

    def test_wavelength_shift_identity(self):
        wl = np.linspace(300, 360, 200)
        spec = np.sin(2 * np.pi * wl / 5.0)
        shifted = wavelength_shift(wl, spec, 0.0, 0.0)
        np.testing.assert_allclose(shifted, spec, atol=1e-10)

    def test_dark_current_shape(self):
        rng = np.random.default_rng(42)
        dc = dark_current_noise(rng, 500, 1e-5)
        assert dc.shape == (500,)

    def test_readout_noise_shape(self):
        rng = np.random.default_rng(42)
        rn = readout_noise(rng, 500, 5e-5)
        assert rn.shape == (500,)


class TestDOASGenerator:
    def test_single_record(self):
        from spektran.doas_generator import generate_doas_record, DOASGenerationSpec
        from spektran.instrument.sampling import load_instrument_config
        from pathlib import Path

        spec = DOASGenerationSpec(
            concentration_ppm_low=0.01,
            concentration_ppm_high=5.0,
            n_output_points=200,
        )

        repo = Path(__file__).resolve().parents[1]
        inst_cfg = load_instrument_config(repo / "configs/instruments/vi-doas-zenith-43.yaml")
        seed = np.random.SeedSequence(54321)

        record = generate_doas_record(spec, inst_cfg, seed)
        assert record["meta"]["technique"] == "DOAS"
        assert record["arrays"]["doas_spectrum"].shape == (200,)
        assert record["arrays"]["doas_spectrum_clean"].shape == (200,)
        assert record["arrays"]["wavelength_nm"].shape == (200,)
        assert record["meta"]["labels"]["species"][0]["concentration_ppm"] > 0

    def test_reproducibility(self):
        from spektran.doas_generator import generate_doas_record, DOASGenerationSpec
        from spektran.instrument.sampling import load_instrument_config
        from pathlib import Path

        spec = DOASGenerationSpec(
            concentration_ppm_low=0.1,
            concentration_ppm_high=1.0,
            n_output_points=100,
        )

        repo = Path(__file__).resolve().parents[1]
        inst_cfg = load_instrument_config(repo / "configs/instruments/vi-doas-zenith-43.yaml")

        seed1 = np.random.SeedSequence(88888)
        seed2 = np.random.SeedSequence(88888)
        r1 = generate_doas_record(spec, inst_cfg, seed1)
        r2 = generate_doas_record(spec, inst_cfg, seed2)
        np.testing.assert_array_equal(
            r1["arrays"]["doas_spectrum"], r2["arrays"]["doas_spectrum"]
        )

    def test_dataset_generation(self):
        from spektran.doas_generator import generate_doas_dataset, DOASGenerationSpec
        from spektran.instrument.sampling import load_instrument_config
        from pathlib import Path

        spec = DOASGenerationSpec(
            concentration_ppm_low=0.01,
            concentration_ppm_high=5.0,
            n_output_points=100,
        )

        repo = Path(__file__).resolve().parents[1]
        inst_cfg = load_instrument_config(repo / "configs/instruments/vi-doas-zenith-43.yaml")

        records = generate_doas_dataset(spec, inst_cfg, 5, master_seed=66666)
        assert len(records) == 5
        concs = [r["meta"]["labels"]["species"][0]["concentration_ppm"] for r in records]
        assert len(set(concs)) == 5
