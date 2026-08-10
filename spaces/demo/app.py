"""SPEKTRAN Interactive Demo — try TDLAS simulation in 30 seconds.

Hosted on Hugging Face Spaces: https://huggingface.co/spaces/spektran/spektran-demo
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


molecule_names = list(MOLECULES.keys())

with gr.Blocks(title="SPEKTRAN Interactive Demo") as demo:
    gr.Markdown(
        """
        # SPEKTRAN Interactive Demo
        **Try TDLAS gas-sensing simulation in 30 seconds.**

        SPEKTRAN is an open-source platform generating physically rigorous
        synthetic training data for ML-based gas sensing. This demo runs the
        forward physics engine in real time — no data download or install needed.

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

    gr.Markdown(
        """
        ---
        **Note:** This demo uses approximate built-in line lists for offline operation.
        Official benchmark data uses HITRAN-fetched parameters.
        SPEKTRAN v0.5.0 | Apache-2.0 (code) | CC BY 4.0 (data)
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
