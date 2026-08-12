"""SPEKTRAN Interactive Demo — try optical gas sensing simulation in 30 seconds.

Hosted on Hugging Face Spaces: https://huggingface.co/spaces/spektran/spektran-demo
5 modalities: TDLAS (DA + WMS), NDIR, CRDS, FTIR, DOAS.
"""

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from spektran.physics import (
    WMSConfig,
    absorption_coefficient,
    demo_ch4_2nu3,
    demo_co,
    demo_co2,
    demo_h2o,
    demo_hcl,
    demo_hf,
    demo_nh3,
    demo_no,
    demo_no2,
    demo_so2,
    simulate_wms,
)
from spektran.physics.crds import empty_cavity_tau, ring_down_time, ring_down_trace
from spektran.physics.ftir import simulate_ftir_spectrum, spectral_resolution_cm1
from spektran.physics.doas import simulate_doas_cross_section, simulate_doas_spectrum

MOLECULES = {
    "CH4": {"lines_fn": demo_ch4_2nu3, "nu_start": 6045.5, "nu_end": 6048.5, "band": "2nu3 R-branch, ~1653 nm"},
    "H2O": {"lines_fn": demo_h2o, "nu_start": 7184.0, "nu_end": 7191.0, "band": "combination band, ~1392 nm"},
    "CO2": {"lines_fn": demo_co2, "nu_start": 4976.5, "nu_end": 4979.5, "band": "combination band, ~2009 nm"},
    "CO": {"lines_fn": demo_co, "nu_start": 2168.0, "nu_end": 2174.0, "band": "v=1-0 R-branch, ~4604 nm"},
    "NH3": {"lines_fn": demo_nh3, "nu_start": 6547.0, "nu_end": 6550.0, "band": "nu1+nu3, ~1527 nm"},
    "NO": {"lines_fn": demo_no, "nu_start": 1899.5, "nu_end": 1901.0, "band": "v=1-0 R-branch, ~5263 nm"},
    "NO2": {"lines_fn": demo_no2, "nu_start": 6323.5, "nu_end": 6326.0, "band": "2nu3 overtone, ~1581 nm"},
    "SO2": {"lines_fn": demo_so2, "nu_start": 2499.5, "nu_end": 2501.5, "band": "nu3 fundamental, ~4000 nm"},
    "HCl": {"lines_fn": demo_hcl, "nu_start": 2885.0, "nu_end": 2887.0, "band": "v=1-0 R-branch, ~3465 nm"},
    "HF": {"lines_fn": demo_hf, "nu_start": 4137.5, "nu_end": 4140.0, "band": "v=1-0 R-branch, ~2416 nm"},
}


def _get_lines_and_range(molecule):
    cfg = MOLECULES[molecule]
    return cfg["lines_fn"](), cfg["nu_start"], cfg["nu_end"]


def _make_absorbance_spectrum(molecule, concentration_ppm, temperature_K, pressure_atm, path_length_m, n_points=2000):
    lines, nu_start, nu_end = _get_lines_and_range(molecule)
    nu = np.linspace(nu_start, nu_end, n_points)
    alpha = absorption_coefficient(
        nu, lines,
        mole_fraction=concentration_ppm * 1e-6,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
    )
    absorbance = alpha * path_length_m * 100.0
    return nu, absorbance


def plot_da(molecule, concentration_ppm, temperature_K, pressure_atm, path_length_m):
    nu, absorbance = _make_absorbance_spectrum(
        molecule, concentration_ppm, temperature_K, pressure_atm, path_length_m
    )
    band = MOLECULES[molecule]["band"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nu, absorbance, color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance (napierian)")
    ax.set_title(f"{molecule} Direct Absorption — {band}")
    ax.grid(True, alpha=0.3)

    info = (
        f"Peak absorbance: {absorbance.max():.4e}\n"
        f"T = {temperature_K:.0f} K, P = {pressure_atm:.2f} atm, "
        f"L = {path_length_m:.1f} m, [{molecule}] = {concentration_ppm:.1f} ppm"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    return fig


def plot_wms(molecule, concentration_ppm, temperature_K, pressure_atm, path_length_m,
             mod_depth_cm1, mod_freq_hz):
    lines, nu_start, nu_end = _get_lines_and_range(molecule)
    center = (nu_start + nu_end) / 2.0
    scan_range = (nu_end - nu_start) * 0.8
    scan_rate = 10.0

    cfg = WMSConfig(
        modulation_frequency_Hz=mod_freq_hz,
        modulation_depth_cm1=mod_depth_cm1,
        sampling_rate_Hz=mod_freq_hz * 20,
        duration_s=0.2,
        center_wavenumber_cm1=center,
        scan_range_cm1=scan_range,
        scan_rate_Hz=scan_rate,
    )

    def absorbance_fn(nu_arr):
        alpha = absorption_coefficient(
            nu_arr, lines,
            mole_fraction=concentration_ppm * 1e-6,
            temperature_K=temperature_K,
            pressure_atm=pressure_atm,
        )
        return alpha * path_length_m * 100.0

    result = simulate_wms(cfg, absorbance_fn, harmonics=(1, 2))
    t = result["t_s"]
    n = len(t)
    mid = n // 4
    end = 3 * n // 4

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    band = MOLECULES[molecule]["band"]

    axes[0].plot(t[mid:end] * 1000, result["r_1f"][mid:end], color="#dc2626", linewidth=1)
    axes[0].set_ylabel("1f amplitude")
    axes[0].set_title(f"{molecule} WMS — {band}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[mid:end] * 1000, result["r_2f"][mid:end], color="#2563eb", linewidth=1)
    axes[1].set_ylabel("2f amplitude")
    axes[1].grid(True, alpha=0.3)

    if "ratio_2f1f" in result:
        axes[2].plot(t[mid:end] * 1000, result["ratio_2f1f"][mid:end], color="#16a34a", linewidth=1)
        axes[2].set_ylabel("2f/1f ratio")
    axes[2].set_xlabel("Time (ms)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_multispecies(mol1, conc1, mol2, conc2, temperature_K, pressure_atm, path_length_m):
    nu1, abs1 = _make_absorbance_spectrum(mol1, conc1, temperature_K, pressure_atm, path_length_m)
    nu2, abs2 = _make_absorbance_spectrum(mol2, conc2, temperature_K, pressure_atm, path_length_m)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(nu1, abs1, color="#2563eb", linewidth=1.5, label=f"{mol1} ({conc1:.1f} ppm)")
    axes[0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_ylabel("Absorbance")
    axes[0].set_title(f"{mol1} — {MOLECULES[mol1]['band']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nu2, abs2, color="#dc2626", linewidth=1.5, label=f"{mol2} ({conc2:.1f} ppm)")
    axes[1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[1].set_ylabel("Absorbance")
    axes[1].set_title(f"{mol2} — {MOLECULES[mol2]['band']}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_crds(concentration_ppm, mirror_r, cavity_length_cm, path_length_m,
              temperature_K, pressure_atm):
    lines = demo_ch4_2nu3()
    nu = np.linspace(6046.0, 6048.0, 200)
    alpha = absorption_coefficient(
        nu, lines,
        mole_fraction=concentration_ppm * 1e-6,
        temperature_K=temperature_K,
        pressure_atm=pressure_atm,
    )
    tau = np.array([ring_down_time(cavity_length_cm, mirror_r, float(a)) for a in alpha])
    tau0 = empty_cavity_tau(cavity_length_cm, mirror_r)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(nu, tau * 1e6, color="#7c3aed", linewidth=1.5)
    axes[0].axhline(tau0 * 1e6, color="#ef4444", linestyle="--", label=f"empty cavity: {tau0*1e6:.1f} us")
    axes[0].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_ylabel("Ring-down time (us)")
    axes[0].set_title("CRDS — CH4 Ring-Down Time Spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    t_trace = np.linspace(0, 5 * tau0, 1000)
    peak_idx = int(np.argmax(alpha))
    decay_on = np.exp(-t_trace / tau[peak_idx])
    decay_off = np.exp(-t_trace / tau0)
    axes[1].plot(t_trace * 1e6, decay_on, color="#7c3aed", label="on-line (peak)")
    axes[1].plot(t_trace * 1e6, decay_off, color="#ef4444", linestyle="--", label="off-line")
    axes[1].set_xlabel("Time (us)")
    axes[1].set_ylabel("Intensity (arb.)")
    axes[1].set_title("Ring-Down Decay")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_ftir(concentration_ppm, max_opd_cm, apodization, path_length_m,
              temperature_K, pressure_atm):
    lines = demo_ch4_2nu3()
    result = simulate_ftir_spectrum(
        lines=lines, molecule="CH4",
        concentration_ppm=concentration_ppm,
        temperature_K=temperature_K, pressure_atm=pressure_atm,
        path_length_m=path_length_m, max_opd_cm=max_opd_cm,
        wavenumber_start_cm1=6045.0, wavenumber_end_cm1=6049.0,
        n_hires_points=10000, n_output_points=500,
        apod_function=apodization,
    )
    res = spectral_resolution_cm1(max_opd_cm)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result["nu_cm1"], result["spectrum_hires"], color="#0891b2", linewidth=1.5)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Transmittance")
    ax.set_title(f"FTIR — CH4 High-Resolution Spectrum (res = {res:.3f} cm$^{{-1}}$)")
    ax.grid(True, alpha=0.3)
    info = (
        f"Apodization: {apodization}\n"
        f"Max OPD: {max_opd_cm:.1f} cm, [{concentration_ppm:.0f}] ppm CH4"
    )
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    return fig


def plot_doas(concentration_ppm, path_length_m, poly_order, rayleigh, mie_tau):
    wavelength = np.linspace(300.0, 360.0, 500)
    sigma = simulate_doas_cross_section(
        wavelength, center_nm=330.0, peak_cross_section_cm2=6e-19,
        n_features=5, feature_width_nm=0.8,
    )
    result = simulate_doas_spectrum(
        wavelength_nm=wavelength, target_sigma_cm2=sigma,
        target_concentration_ppm=concentration_ppm,
        temperature_K=296.0, pressure_atm=1.0,
        path_length_m=path_length_m, poly_order=poly_order,
        rayleigh=rayleigh, mie_tau_ref=mie_tau,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(wavelength, result["od_total"], color="#ea580c", linewidth=1.5)
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Optical Density")
    axes[0].set_title(f"DOAS — SO2 Total OD ({concentration_ppm:.2f} ppm, {path_length_m:.0f} m)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(wavelength, result["doas_spectrum"], color="#2563eb", linewidth=1.5)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Differential OD")
    axes[1].set_title("Differential Optical Density (after polynomial high-pass)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


molecule_names = list(MOLECULES.keys())

with gr.Blocks(title="SPEKTRAN Interactive Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # SPEKTRAN Interactive Demo
        **Try optical gas sensing simulation in 30 seconds — 5 modalities.**

        SPEKTRAN is an open-source platform generating physically rigorous
        synthetic training data for ML-based gas sensing. This demo runs the
        forward physics engine in real time — no data download or install needed.
        Supports TDLAS (DA + WMS), CRDS, FTIR, and DOAS.

        [GitHub](https://github.com/spektran/spektran) |
        [Paper](https://doi.org/10.5281/zenodo.21790394) |
        [Dataset](https://huggingface.co/datasets/spektran/spektran-ch4-v0) |
        [PyPI](https://pypi.org/project/spektran/)
        """
    )

    with gr.Tab("Direct Absorption"):
        gr.Markdown("Simulate a clean Beer-Lambert absorption spectrum for any of 10 target molecules.")
        with gr.Row():
            with gr.Column(scale=1):
                da_mol = gr.Dropdown(molecule_names, value="CH4", label="Molecule")
                da_conc = gr.Slider(1, 10000, value=100, step=1, label="Concentration (ppm)")
                da_temp = gr.Slider(200, 1500, value=296, step=1, label="Temperature (K)")
                da_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Pressure (atm)")
                da_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1, label="Path length (m)")
                da_btn = gr.Button("Simulate", variant="primary")
            with gr.Column(scale=2):
                da_plot = gr.Plot(label="Absorption Spectrum")
        da_inputs = [da_mol, da_conc, da_temp, da_pres, da_path]
        da_btn.click(plot_da, inputs=da_inputs, outputs=da_plot)
        for inp in da_inputs:
            inp.change(plot_da, inputs=da_inputs, outputs=da_plot)

    with gr.Tab("WMS (Wavelength Modulation)"):
        gr.Markdown("Simulate wavelength-modulation spectroscopy: laser modulation + lock-in demodulation → 1f/2f signals.")
        with gr.Row():
            with gr.Column(scale=1):
                wms_mol = gr.Dropdown(molecule_names, value="CH4", label="Molecule")
                wms_conc = gr.Slider(1, 10000, value=100, step=1, label="Concentration (ppm)")
                wms_temp = gr.Slider(200, 1500, value=296, step=1, label="Temperature (K)")
                wms_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Pressure (atm)")
                wms_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1, label="Path length (m)")
                wms_mod = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Modulation depth (cm-1)")
                wms_freq = gr.Slider(1000, 100000, value=10000, step=1000, label="Modulation frequency (Hz)")
                wms_btn = gr.Button("Simulate", variant="primary")
            with gr.Column(scale=2):
                wms_plot = gr.Plot(label="WMS Demodulated Signals")
        wms_inputs = [wms_mol, wms_conc, wms_temp, wms_pres, wms_path, wms_mod, wms_freq]
        wms_btn.click(plot_wms, inputs=wms_inputs, outputs=wms_plot)

    with gr.Tab("Multi-species Comparison"):
        gr.Markdown("Compare absorption spectra of two molecules side by side.")
        with gr.Row():
            with gr.Column(scale=1):
                ms_mol1 = gr.Dropdown(molecule_names, value="CH4", label="Molecule 1")
                ms_conc1 = gr.Slider(1, 10000, value=100, step=1, label="Concentration 1 (ppm)")
                ms_mol2 = gr.Dropdown(molecule_names, value="H2O", label="Molecule 2")
                ms_conc2 = gr.Slider(1, 50000, value=10000, step=10, label="Concentration 2 (ppm)")
                ms_temp = gr.Slider(200, 1500, value=296, step=1, label="Temperature (K)")
                ms_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Pressure (atm)")
                ms_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1, label="Path length (m)")
                ms_btn = gr.Button("Simulate", variant="primary")
            with gr.Column(scale=2):
                ms_plot = gr.Plot(label="Multi-species Spectra")
        ms_inputs = [ms_mol1, ms_conc1, ms_mol2, ms_conc2, ms_temp, ms_pres, ms_path]
        ms_btn.click(plot_multispecies, inputs=ms_inputs, outputs=ms_plot)

    with gr.Tab("CRDS (Cavity Ring-Down)"):
        gr.Markdown("Simulate cavity ring-down spectroscopy: absorption coefficient from ring-down time decay.")
        with gr.Row():
            with gr.Column(scale=1):
                crds_conc = gr.Slider(1, 5000, value=100, step=1, label="CH4 Concentration (ppm)")
                crds_r = gr.Slider(0.9999, 0.999999, value=0.99995, step=0.000001, label="Mirror Reflectivity")
                crds_len = gr.Slider(10, 100, value=50, step=1, label="Cavity Length (cm)")
                crds_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1, label="Gas Path Length (m)")
                crds_temp = gr.Slider(200, 500, value=296, step=1, label="Temperature (K)")
                crds_pres = gr.Slider(0.01, 2.0, value=1.0, step=0.01, label="Pressure (atm)")
                crds_btn = gr.Button("Simulate", variant="primary")
            with gr.Column(scale=2):
                crds_plot = gr.Plot(label="CRDS Ring-Down Spectrum")
        crds_inputs = [crds_conc, crds_r, crds_len, crds_path, crds_temp, crds_pres]
        crds_btn.click(plot_crds, inputs=crds_inputs, outputs=crds_plot)

    with gr.Tab("FTIR (Fourier Transform)"):
        gr.Markdown("Simulate Fourier transform infrared spectroscopy: interferogram -> apodized FFT -> broadband spectrum.")
        with gr.Row():
            with gr.Column(scale=1):
                ftir_conc = gr.Slider(10, 5000, value=500, step=10, label="CH4 Concentration (ppm)")
                ftir_opd = gr.Slider(0.5, 50.0, value=10.0, step=0.5, label="Max OPD (cm)")
                ftir_apod = gr.Dropdown(
                    ["boxcar", "triangular", "happ_genzel", "norton_beer_medium", "norton_beer_strong"],
                    value="happ_genzel", label="Apodization"
                )
                ftir_path = gr.Slider(0.1, 50.0, value=10.0, step=0.1, label="Path Length (m)")
                ftir_temp = gr.Slider(200, 500, value=296, step=1, label="Temperature (K)")
                ftir_pres = gr.Slider(0.01, 2.0, value=1.0, step=0.01, label="Pressure (atm)")
                ftir_btn = gr.Button("Simulate", variant="primary")
            with gr.Column(scale=2):
                ftir_plot = gr.Plot(label="FTIR Spectrum")
        ftir_inputs = [ftir_conc, ftir_opd, ftir_apod, ftir_path, ftir_temp, ftir_pres]
        ftir_btn.click(plot_ftir, inputs=ftir_inputs, outputs=ftir_plot)

    with gr.Tab("DOAS (Differential OAS)"):
        gr.Markdown("Simulate differential optical absorption spectroscopy: UV/Vis Beer-Lambert + polynomial high-pass filter.")
        with gr.Row():
            with gr.Column(scale=1):
                doas_conc = gr.Slider(0.01, 100.0, value=1.0, step=0.01, label="SO2 Concentration (ppm)")
                doas_path = gr.Slider(10, 5000, value=500, step=10, label="Path Length (m)")
                doas_poly = gr.Slider(1, 9, value=5, step=1, label="Polynomial Order")
                doas_ray = gr.Checkbox(value=True, label="Include Rayleigh Scattering")
                doas_mie = gr.Slider(0.0, 2.0, value=0.3, step=0.01, label="Mie Aerosol Optical Depth")
                doas_btn = gr.Button("Simulate", variant="primary")
            with gr.Column(scale=2):
                doas_plot = gr.Plot(label="DOAS Differential OD")
        doas_inputs = [doas_conc, doas_path, doas_poly, doas_ray, doas_mie]
        doas_btn.click(plot_doas, inputs=doas_inputs, outputs=doas_plot)

    gr.Markdown(
        """
        ---
        **Note:** This demo uses approximate built-in line lists (TDLAS/CRDS/FTIR)
        and synthetic cross sections (DOAS) for offline operation.
        Official benchmark data uses HITRAN-fetched parameters.
        SPEKTRAN v0.6.0 | Apache-2.0 (code) | CC BY 4.0 (data)
        """
    )

if __name__ == "__main__":
    demo.launch()
