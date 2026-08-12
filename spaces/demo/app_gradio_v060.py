"""SPEKTRAN Interactive Demo v2.0

Fully redesigned with:
  - i18n (English / 中文) via client-side JS
  - Dark / Light theme toggle (system-default)
  - Multi-format export (CSV, JSON; PNG/SVG via Plotly toolbar)
  - AI-era glassmorphism UI with animated mesh gradient background
  - Interactive Plotly charts across all 5 modalities (6 tabs)
"""

import io
import json
import os
import tempfile

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
from spektran.physics.crds import empty_cavity_tau, ring_down_time
from spektran.physics.ftir import simulate_ftir_spectrum, spectral_resolution_cm1
from spektran.physics.doas import simulate_doas_cross_section, simulate_doas_spectrum


# ═══════════════════════════════════════════════════════════════════════════
# Molecule database
# ═══════════════════════════════════════════════════════════════════════════

MOLECULES = {
    "CH4": {"lines_fn": demo_ch4_2nu3, "nu_start": 6045.5, "nu_end": 6048.5,
            "band": "2ν₃ R-branch", "wl": "1653 nm", "color": "#00f0ff"},
    "H2O": {"lines_fn": demo_h2o, "nu_start": 7184.0, "nu_end": 7191.0,
            "band": "combination", "wl": "1392 nm", "color": "#3b82f6"},
    "CO2": {"lines_fn": demo_co2, "nu_start": 4976.5, "nu_end": 4979.5,
            "band": "combination", "wl": "2009 nm", "color": "#a855f7"},
    "CO":  {"lines_fn": demo_co, "nu_start": 2168.0, "nu_end": 2174.0,
            "band": "v=1-0 R", "wl": "4604 nm", "color": "#f97316"},
    "NH3": {"lines_fn": demo_nh3, "nu_start": 6547.0, "nu_end": 6550.0,
            "band": "ν₁+ν₃", "wl": "1527 nm", "color": "#22c55e"},
    "NO":  {"lines_fn": demo_no, "nu_start": 1899.5, "nu_end": 1901.0,
            "band": "v=1-0 R", "wl": "5263 nm", "color": "#ef4444"},
    "NO2": {"lines_fn": demo_no2, "nu_start": 6323.5, "nu_end": 6326.0,
            "band": "2ν₃ overtone", "wl": "1581 nm", "color": "#eab308"},
    "SO2": {"lines_fn": demo_so2, "nu_start": 2499.5, "nu_end": 2501.5,
            "band": "ν₃ fund.", "wl": "4000 nm", "color": "#ec4899"},
    "HCl": {"lines_fn": demo_hcl, "nu_start": 2885.0, "nu_end": 2887.0,
            "band": "v=1-0 R", "wl": "3465 nm", "color": "#14b8a6"},
    "HF":  {"lines_fn": demo_hf, "nu_start": 4137.5, "nu_end": 4140.0,
            "band": "v=1-0 R", "wl": "2416 nm", "color": "#8b5cf6"},
}


# ═══════════════════════════════════════════════════════════════════════════
# Theme-aware Plotly helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pcol(theme="dark"):
    if theme == "light":
        return dict(
            bg="rgba(255,255,255,0)", grid="rgba(0,0,0,0.06)",
            axis="rgba(0,0,0,0.12)", font="#334155",
            hover_bg="#ffffff", hover_border="#3b82f6", hover_font="#0f172a",
            legend_bg="rgba(255,255,255,0.85)", ann_bg="rgba(255,255,255,0.9)",
            ann_font="#0f172a", title_font="#0f172a",
            modebar_color="#94a3b8", modebar_active="#3b82f6",
        )
    return dict(
        bg="rgba(10,14,26,0)", grid="rgba(100,150,255,0.08)",
        axis="rgba(150,180,220,0.25)", font="#94a3b8",
        hover_bg="#1e293b", hover_border="#3b82f6", hover_font="#e2e8f0",
        legend_bg="rgba(10,14,26,0.7)", ann_bg="rgba(10,14,26,0.8)",
        ann_font="#e2e8f0", title_font="#e2e8f0",
        modebar_color="#475569", modebar_active="#3b82f6",
    )


def _apply_layout(fig, title="", xlab="", ylab="", height=460, theme="dark"):
    c = _pcol(theme)
    fig.update_layout(
        paper_bgcolor=c["bg"], plot_bgcolor=c["bg"],
        font=dict(family="'JetBrains Mono','SF Mono',Consolas,monospace",
                  color=c["font"], size=12),
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"],
                   linecolor=c["axis"], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"],
                   linecolor=c["axis"], tickfont=dict(size=11)),
        hoverlabel=dict(bgcolor=c["hover_bg"], bordercolor=c["hover_border"],
                        font=dict(color=c["hover_font"], size=12)),
        legend=dict(bgcolor=c["legend_bg"], bordercolor="rgba(100,150,255,0.15)",
                    borderwidth=1, font=dict(size=11)),
        title=dict(text=title, font=dict(size=15, color=c["title_font"]),
                   x=0.02, xanchor="left"),
        xaxis_title=xlab, yaxis_title=ylab, height=height,
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=c["modebar_color"],
                     activecolor=c["modebar_active"]),
    )
    return fig


def _apply_sub_layout(fig, title="", height=580, theme="dark"):
    c = _pcol(theme)
    fig.update_layout(
        paper_bgcolor=c["bg"], plot_bgcolor=c["bg"],
        font=dict(family="'JetBrains Mono','SF Mono',Consolas,monospace",
                  color=c["font"], size=12),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(bgcolor=c["hover_bg"], bordercolor=c["hover_border"],
                        font=dict(color=c["hover_font"], size=12)),
        legend=dict(bgcolor=c["legend_bg"], bordercolor="rgba(100,150,255,0.15)",
                    borderwidth=1, font=dict(size=11)),
        title=dict(text=title, font=dict(size=15, color=c["title_font"]), x=0.02),
        height=height,
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=c["modebar_color"],
                     activecolor=c["modebar_active"]),
    )
    for ax in fig.layout:
        if ax.startswith(("xaxis", "yaxis")):
            fig.layout[ax].update(gridcolor=c["grid"], linecolor=c["axis"])
    for ann in fig.layout.annotations:
        if not ann.font or not ann.font.color:
            ann.font = dict(size=12, color=c["font"])
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Core physics helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_lines_and_range(molecule):
    cfg = MOLECULES[molecule]
    return cfg["lines_fn"](), cfg["nu_start"], cfg["nu_end"]


def _make_absorbance(molecule, conc, temp, pres, path_m, n=2000):
    lines, nu0, nu1 = _get_lines_and_range(molecule)
    nu = np.linspace(nu0, nu1, n)
    alpha = absorption_coefficient(
        nu, lines, mole_fraction=conc * 1e-6,
        temperature_K=temp, pressure_atm=pres,
    )
    return nu, alpha * path_m * 100.0


def _hex_to_rgba(hex_color, alpha=0.08):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ═══════════════════════════════════════════════════════════════════════════
# Simulation functions — each returns (fig, stats_html, data_dict)
# ═══════════════════════════════════════════════════════════════════════════

def plot_da(molecule, conc, temp, pres, path_m, theme_state):
    theme = theme_state or "dark"
    nu, absorbance = _make_absorbance(molecule, conc, temp, pres, path_m)
    c = MOLECULES[molecule]["color"]
    band = MOLECULES[molecule]["band"]
    wl = MOLECULES[molecule]["wl"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nu, y=absorbance, mode="lines",
        line=dict(color=c, width=2.2),
        fill="tozeroy", fillcolor=_hex_to_rgba(c),
        hovertemplate="<b>%{x:.2f}</b> cm⁻¹<br>Abs: %{y:.4e}<extra></extra>",
        name=molecule,
    ))
    peak = float(absorbance.max())
    peak_nu = float(nu[np.argmax(absorbance)])
    pc = _pcol(theme)
    fig.add_annotation(
        x=peak_nu, y=peak, text=f"peak {peak:.3e}",
        showarrow=True, arrowhead=2, arrowcolor=c, arrowwidth=1.5,
        font=dict(size=11, color=c), bgcolor=pc["ann_bg"],
        bordercolor=c, borderwidth=1, borderpad=4, ax=0, ay=-40,
    )
    _apply_layout(fig, f"<b>{molecule}</b> — {band} ({wl})",
                  "Wavenumber (cm⁻¹)", "Absorbance", theme=theme)

    stats = (f"<span style='color:{c}'>Peak: {peak:.3e}</span> &nbsp;│&nbsp; "
             f"T={temp:.0f} K &nbsp;│&nbsp; P={pres:.2f} atm &nbsp;│&nbsp; "
             f"L={path_m:.1f} m &nbsp;│&nbsp; [{molecule}]={conc:.0f} ppm")

    data = {"type": "da", "molecule": molecule,
            "params": {"concentration_ppm": conc, "temperature_K": temp,
                       "pressure_atm": pres, "path_length_m": path_m},
            "columns": ["wavenumber_cm1", "absorbance"],
            "wavenumber_cm1": nu.tolist(), "absorbance": absorbance.tolist()}
    return fig, stats, data


def plot_wms(molecule, conc, temp, pres, path_m, mod_depth, mod_freq, theme_state):
    theme = theme_state or "dark"
    lines, nu0, nu1 = _get_lines_and_range(molecule)
    center = (nu0 + nu1) / 2.0
    scan_range = (nu1 - nu0) * 0.8

    cfg = WMSConfig(
        modulation_frequency_Hz=mod_freq, modulation_depth_cm1=mod_depth,
        sampling_rate_Hz=mod_freq * 20, duration_s=0.2,
        center_wavenumber_cm1=center, scan_range_cm1=scan_range, scan_rate_Hz=10.0,
    )

    def abs_fn(nu_arr):
        alpha = absorption_coefficient(
            nu_arr, lines, mole_fraction=conc * 1e-6,
            temperature_K=temp, pressure_atm=pres,
        )
        return alpha * path_m * 100.0

    result = simulate_wms(cfg, abs_fn, harmonics=(1, 2))
    t = result["t_s"]
    n = len(t)
    mid, end = n // 4, 3 * n // 4
    t_ms = t[mid:end] * 1000

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("1f Signal", "2f Signal", "2f/1f Ratio"))
    fig.add_trace(go.Scatter(
        x=t_ms, y=result["r_1f"][mid:end], mode="lines",
        line=dict(color="#ef4444", width=1.8), name="1f",
        hovertemplate="t=%{x:.2f} ms<br>1f=%{y:.4e}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_ms, y=result["r_2f"][mid:end], mode="lines",
        line=dict(color="#3b82f6", width=1.8), name="2f",
        hovertemplate="t=%{x:.2f} ms<br>2f=%{y:.4e}<extra></extra>",
    ), row=2, col=1)
    if "ratio_2f1f" in result:
        fig.add_trace(go.Scatter(
            x=t_ms, y=result["ratio_2f1f"][mid:end], mode="lines",
            line=dict(color="#22c55e", width=1.8), name="2f/1f",
            hovertemplate="t=%{x:.2f} ms<br>ratio=%{y:.4f}<extra></extra>",
        ), row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
    band = MOLECULES[molecule]["band"]
    _apply_sub_layout(fig, f"<b>{molecule}</b> WMS — {band}", theme=theme)

    stats = (f"mod: {mod_depth:.2f} cm⁻¹ @ {mod_freq/1000:.0f} kHz &nbsp;│&nbsp; "
             f"[{molecule}]={conc:.0f} ppm")

    data = {"type": "wms", "molecule": molecule,
            "params": {"concentration_ppm": conc, "temperature_K": temp,
                       "pressure_atm": pres, "path_length_m": path_m,
                       "mod_depth_cm1": mod_depth, "mod_freq_Hz": mod_freq},
            "columns": ["time_ms", "signal_1f", "signal_2f"],
            "time_ms": t_ms.tolist(),
            "signal_1f": result["r_1f"][mid:end].tolist(),
            "signal_2f": result["r_2f"][mid:end].tolist()}
    return fig, stats, data


def plot_crds(conc, mirror_r, cavity_cm, path_m, temp, pres, theme_state):
    theme = theme_state or "dark"
    lines = demo_ch4_2nu3()
    nu = np.linspace(6046.0, 6048.0, 300)
    alpha = absorption_coefficient(
        nu, lines, mole_fraction=conc * 1e-6,
        temperature_K=temp, pressure_atm=pres,
    )
    tau = np.array([ring_down_time(cavity_cm, mirror_r, float(a)) for a in alpha])
    tau0 = empty_cavity_tau(cavity_cm, mirror_r)

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=("τ Ring-Down Spectrum", "Decay Traces"))
    fig.add_trace(go.Scatter(
        x=nu, y=tau * 1e6, mode="lines",
        line=dict(color="#a855f7", width=2.2),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.06)",
        name="τ(ν)", hovertemplate="%{x:.2f} cm⁻¹<br>τ=%{y:.2f} µs<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=tau0 * 1e6, line=dict(color="#ef4444", dash="dash", width=1.5),
                  annotation_text=f"empty cavity: {tau0*1e6:.1f} µs",
                  annotation_font=dict(color="#ef4444", size=11), row=1, col=1)

    t_trace = np.linspace(0, 5 * tau0, 500)
    peak_idx = int(np.argmax(alpha))
    fig.add_trace(go.Scatter(
        x=t_trace * 1e6, y=np.exp(-t_trace / tau[peak_idx]), mode="lines",
        line=dict(color="#a855f7", width=2), name="on-line (peak)",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_trace * 1e6, y=np.exp(-t_trace / tau0), mode="lines",
        line=dict(color="#ef4444", dash="dash", width=1.5), name="off-line",
    ), row=2, col=1)

    fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=1, col=1)
    fig.update_yaxes(title_text="τ (µs)", row=1, col=1)
    fig.update_xaxes(title_text="Time (µs)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity (arb.)", row=2, col=1)
    _apply_sub_layout(fig, "<b>CRDS</b> — CH₄ Cavity Ring-Down", theme=theme)

    stats = (f"<span style='color:#a855f7'>τₘᵢₙ: {tau.min()*1e6:.2f} µs</span> &nbsp;│&nbsp; "
             f"τ₀: {tau0*1e6:.2f} µs &nbsp;│&nbsp; R={mirror_r:.6f} &nbsp;│&nbsp; "
             f"[{conc:.0f}] ppm")

    data = {"type": "crds", "molecule": "CH4",
            "params": {"concentration_ppm": conc, "mirror_reflectivity": mirror_r,
                       "cavity_cm": cavity_cm, "path_length_m": path_m,
                       "temperature_K": temp, "pressure_atm": pres},
            "columns": ["wavenumber_cm1", "tau_us"],
            "wavenumber_cm1": nu.tolist(), "tau_us": (tau * 1e6).tolist()}
    return fig, stats, data


def plot_ftir(conc, max_opd, apod, path_m, temp, pres, theme_state):
    theme = theme_state or "dark"
    lines = demo_ch4_2nu3()
    result = simulate_ftir_spectrum(
        lines=lines, molecule="CH4",
        concentration_ppm=conc, temperature_K=temp, pressure_atm=pres,
        path_length_m=path_m, max_opd_cm=max_opd,
        wavenumber_start_cm1=6045.0, wavenumber_end_cm1=6049.0,
        n_hires_points=4000, n_output_points=500,
        apod_function=apod,
    )
    res = spectral_resolution_cm1(max_opd)
    nu = result["nu_cm1"]
    spec = result["spectrum"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nu, y=spec, mode="lines",
        line=dict(color="#0891b2", width=2.2),
        fill="tozeroy", fillcolor="rgba(8,145,178,0.06)",
        hovertemplate="%{x:.2f} cm⁻¹<br>T=%{y:.5f}<extra></extra>",
        name="transmittance",
    ))
    _apply_layout(fig, f"<b>FTIR</b> — CH₄ Transmittance (Δν = {res:.3f} cm⁻¹)",
                  "Wavenumber (cm⁻¹)", "Transmittance", theme=theme)

    stats = (f"<span style='color:#0891b2'>res: {res:.3f} cm⁻¹</span> &nbsp;│&nbsp; "
             f"apod: {apod} &nbsp;│&nbsp; OPD: {max_opd:.1f} cm &nbsp;│&nbsp; "
             f"[{conc:.0f}] ppm")

    data = {"type": "ftir", "molecule": "CH4",
            "params": {"concentration_ppm": conc, "max_opd_cm": max_opd,
                       "apodization": apod, "path_length_m": path_m,
                       "temperature_K": temp, "pressure_atm": pres,
                       "resolution_cm1": res},
            "columns": ["wavenumber_cm1", "transmittance"],
            "wavenumber_cm1": nu.tolist(), "transmittance": spec.tolist()}
    return fig, stats, data


def plot_doas(conc, path_m, poly_order, rayleigh, mie_tau, theme_state):
    theme = theme_state or "dark"
    wl = np.linspace(300.0, 360.0, 500)
    sigma = simulate_doas_cross_section(
        wl, center_nm=330.0, peak_cross_section_cm2=6e-19,
        n_features=5, feature_width_nm=0.8,
    )
    result = simulate_doas_spectrum(
        wavelength_nm=wl, target_sigma_cm2=sigma,
        target_concentration_ppm=conc, temperature_K=296.0, pressure_atm=1.0,
        path_length_m=path_m, poly_order=poly_order,
        rayleigh=rayleigh, mie_tau_ref=mie_tau,
    )

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=("Total Optical Density",
                                        "Differential OD (polynomial high-pass)"))
    fig.add_trace(go.Scatter(
        x=wl, y=result["od_total"], mode="lines",
        line=dict(color="#ea580c", width=2), name="total OD",
        hovertemplate="%{x:.1f} nm<br>OD=%{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=wl, y=result["doas_spectrum"], mode="lines",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
        name="diff. OD",
        hovertemplate="%{x:.1f} nm<br>ΔOD=%{y:.4e}<extra></extra>",
    ), row=2, col=1)
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    _apply_sub_layout(fig, "<b>DOAS</b> — SO₂ Differential Optical Absorption",
                      theme=theme)

    stats = (f"SO₂: {conc:.2f} ppm &nbsp;│&nbsp; path: {path_m:.0f} m &nbsp;│&nbsp; "
             f"poly: {poly_order} &nbsp;│&nbsp; Mie τ: {mie_tau:.2f}")

    data = {"type": "doas", "molecule": "SO2",
            "params": {"concentration_ppm": conc, "path_length_m": path_m,
                       "poly_order": poly_order, "rayleigh": rayleigh,
                       "mie_tau_ref": mie_tau},
            "columns": ["wavelength_nm", "od_total", "doas_spectrum"],
            "wavelength_nm": wl.tolist(),
            "od_total": result["od_total"].tolist(),
            "doas_spectrum": result["doas_spectrum"].tolist()}
    return fig, stats, data


def plot_compare(mol1, c1, mol2, c2, temp, pres, path_m, theme_state):
    theme = theme_state or "dark"
    nu1, abs1 = _make_absorbance(mol1, c1, temp, pres, path_m)
    nu2, abs2 = _make_absorbance(mol2, c2, temp, pres, path_m)
    c1c, c2c = MOLECULES[mol1]["color"], MOLECULES[mol2]["color"]

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=(
                            f"{mol1} — {MOLECULES[mol1]['band']} ({MOLECULES[mol1]['wl']})",
                            f"{mol2} — {MOLECULES[mol2]['band']} ({MOLECULES[mol2]['wl']})",
                        ))
    fig.add_trace(go.Scatter(
        x=nu1, y=abs1, mode="lines", line=dict(color=c1c, width=2.2),
        fill="tozeroy", fillcolor=_hex_to_rgba(c1c),
        name=f"{mol1} ({c1:.0f} ppm)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=nu2, y=abs2, mode="lines", line=dict(color=c2c, width=2.2),
        fill="tozeroy", fillcolor=_hex_to_rgba(c2c),
        name=f"{mol2} ({c2:.0f} ppm)",
    ), row=2, col=1)
    for i in (1, 2):
        fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=i, col=1)
        fig.update_yaxes(title_text="Absorbance", row=i, col=1)
    _apply_sub_layout(fig, "<b>Multi-Species</b> Comparison", theme=theme)

    stats = (f"<span style='color:{c1c}'>{mol1}: {abs1.max():.3e}</span> &nbsp;│&nbsp; "
             f"<span style='color:{c2c}'>{mol2}: {abs2.max():.3e}</span>")

    data = {"type": "compare", "molecules": [mol1, mol2],
            "params": {"conc1_ppm": c1, "conc2_ppm": c2,
                       "temperature_K": temp, "pressure_atm": pres,
                       "path_length_m": path_m},
            "columns": [f"nu1_cm1", f"abs1_{mol1}", f"nu2_cm1", f"abs2_{mol2}"],
            f"nu1_cm1": nu1.tolist(), f"abs1_{mol1}": abs1.tolist(),
            f"nu2_cm1": nu2.tolist(), f"abs2_{mol2}": abs2.tolist()}
    return fig, stats, data


# ═══════════════════════════════════════════════════════════════════════════
# Export functions
# ═══════════════════════════════════════════════════════════════════════════

def export_csv(data_state):
    if not data_state:
        return gr.update(visible=False)
    buf = io.StringIO()
    cols = data_state.get("columns", [])
    buf.write(",".join(cols) + "\n")
    arrays = [data_state.get(c, []) for c in cols]
    if arrays:
        for row in zip(*arrays):
            buf.write(",".join(f"{v}" for v in row) + "\n")
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w",
        prefix=f"spektran_{data_state.get('type', 'data')}_",
    )
    tmp.write(buf.getvalue())
    tmp.close()
    return gr.update(value=tmp.name, visible=True)


def export_json(data_state):
    if not data_state:
        return gr.update(visible=False)
    export = {
        "generator": "SPEKTRAN v0.6.0",
        "simulation_type": data_state.get("type", "unknown"),
        "molecule": data_state.get("molecule", data_state.get("molecules", "unknown")),
        "parameters": data_state.get("params", {}),
        "data": {c: data_state.get(c, []) for c in data_state.get("columns", [])},
    }
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w",
        prefix=f"spektran_{data_state.get('type', 'data')}_",
    )
    json.dump(export, tmp, indent=2)
    tmp.close()
    return gr.update(value=tmp.name, visible=True)


# ═══════════════════════════════════════════════════════════════════════════
# CSS — glassmorphism, mesh gradient, dual theme, smooth transitions
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Theme tokens ────────────────────────────────────────────────── */
[data-theme="dark"], :root {
    --sp-bg: #060a14;
    --sp-bg2: #0a0e1a;
    --sp-card: rgba(12,18,32,0.65);
    --sp-card-solid: #0f1525;
    --sp-input: #141b2d;
    --sp-text: #e2e8f0;
    --sp-text2: #94a3b8;
    --sp-muted: #475569;
    --sp-accent: #3b82f6;
    --sp-accent2: #a855f7;
    --sp-cyan: #00f0ff;
    --sp-border: rgba(59,130,246,0.12);
    --sp-border-h: rgba(59,130,246,0.35);
    --sp-glow: 0 0 30px rgba(59,130,246,0.08);
    --sp-glow-h: 0 0 40px rgba(59,130,246,0.15);
    --sp-shadow: 0 4px 24px rgba(0,0,0,0.25);
    --sp-glass: rgba(15,21,37,0.55);
    --sp-glass-b: rgba(59,130,246,0.1);
    --sp-mesh1: rgba(59,130,246,0.07);
    --sp-mesh2: rgba(168,85,247,0.05);
    --sp-mesh3: rgba(0,240,255,0.04);
    --sp-stats-bg: rgba(15,21,37,0.6);
    --sp-scrollbar: rgba(59,130,246,0.2);
    --sp-divider: linear-gradient(90deg, transparent, rgba(59,130,246,0.2), rgba(168,85,247,0.15), transparent);
}
[data-theme="light"] {
    --sp-bg: #f0f4f8;
    --sp-bg2: #f8fafc;
    --sp-card: rgba(255,255,255,0.65);
    --sp-card-solid: #ffffff;
    --sp-input: #f1f5f9;
    --sp-text: #0f172a;
    --sp-text2: #334155;
    --sp-muted: #64748b;
    --sp-accent: #2563eb;
    --sp-accent2: #7c3aed;
    --sp-cyan: #0891b2;
    --sp-border: rgba(0,0,0,0.08);
    --sp-border-h: rgba(59,130,246,0.25);
    --sp-glow: 0 1px 4px rgba(0,0,0,0.06);
    --sp-glow-h: 0 4px 20px rgba(59,130,246,0.1);
    --sp-shadow: 0 1px 8px rgba(0,0,0,0.08);
    --sp-glass: rgba(255,255,255,0.6);
    --sp-glass-b: rgba(0,0,0,0.06);
    --sp-mesh1: rgba(59,130,246,0.04);
    --sp-mesh2: rgba(168,85,247,0.03);
    --sp-mesh3: rgba(0,200,255,0.02);
    --sp-stats-bg: rgba(255,255,255,0.6);
    --sp-scrollbar: rgba(0,0,0,0.12);
    --sp-divider: linear-gradient(90deg, transparent, rgba(0,0,0,0.06), rgba(0,0,0,0.04), transparent);
}

/* ── Global ──────────────────────────────────────────────────────── */
* { transition: background-color 0.35s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease; }
.js-plotly-plot *, .plotly *, .plot-container * { transition: none !important; }
body, .gradio-container, .main, .contain {
    background: var(--sp-bg) !important;
    color: var(--sp-text) !important;
}
.gradio-container {
    background:
        radial-gradient(ellipse 120% 80% at 20% 10%, var(--sp-mesh1) 0%, transparent 60%),
        radial-gradient(ellipse 100% 100% at 80% 90%, var(--sp-mesh2) 0%, transparent 55%),
        radial-gradient(ellipse 80% 60% at 55% 50%, var(--sp-mesh3) 0%, transparent 50%),
        var(--sp-bg) !important;
    background-attachment: fixed !important;
}
footer { display: none !important; }
#sp-theme-val, #sp-lang-val { position: absolute !important; height: 0 !important; overflow: hidden !important; }

/* ── Animated mesh orbs ─────────────────────────────────────────── */
.sp-mesh-bg {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0; overflow: hidden;
}
.sp-mesh-bg .orb {
    position: absolute; border-radius: 50%; filter: blur(100px);
    animation: orbFloat 20s ease-in-out infinite alternate;
}
.sp-mesh-bg .orb:nth-child(1) {
    width: 500px; height: 500px; top: -10%; left: 10%;
    background: rgba(59,130,246,0.08);
    animation-duration: 22s;
}
.sp-mesh-bg .orb:nth-child(2) {
    width: 400px; height: 400px; top: 60%; right: -5%;
    background: rgba(168,85,247,0.06);
    animation-duration: 26s; animation-delay: -5s;
}
.sp-mesh-bg .orb:nth-child(3) {
    width: 350px; height: 350px; top: 30%; left: 50%;
    background: rgba(0,240,255,0.04);
    animation-duration: 30s; animation-delay: -10s;
}
[data-theme="light"] .sp-mesh-bg .orb:nth-child(1) { background: rgba(59,130,246,0.05); }
[data-theme="light"] .sp-mesh-bg .orb:nth-child(2) { background: rgba(168,85,247,0.04); }
[data-theme="light"] .sp-mesh-bg .orb:nth-child(3) { background: rgba(0,200,255,0.03); }
@keyframes orbFloat {
    0% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -40px) scale(1.05); }
    66% { transform: translate(-20px, 20px) scale(0.95); }
    100% { transform: translate(10px, -10px) scale(1.02); }
}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--sp-scrollbar); border-radius: 3px; }

/* ── Control bar (theme + lang toggles) ──────────────────────────── */
.sp-ctrl-row {
    position: sticky !important;
    top: 0;
    z-index: 100;
    display: flex !important;
    justify-content: center !important;
    gap: 8px !important;
    padding: 10px 12px !important;
    background: transparent !important;
    border: none !important;
}
.ctrl-btn {
    min-width: 0 !important;
    padding: 7px 18px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    background: var(--sp-glass) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--sp-glass-b) !important;
    border-radius: 24px !important;
    color: var(--sp-text2) !important;
    cursor: pointer !important;
    box-shadow: var(--sp-glow) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
.ctrl-btn:hover {
    border-color: var(--sp-border-h) !important;
    color: var(--sp-text) !important;
    box-shadow: var(--sp-glow-h) !important;
    transform: translateY(-2px) !important;
}

/* ── Entrance animation ─────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.hero-wrap { animation: fadeInUp 0.8s ease-out both; }
.tabs { animation: fadeInUp 0.8s ease-out 0.15s both; }

/* ── Hero ─────────────────────────────────────────────────────────── */
.hero-wrap {
    text-align: center;
    padding: 2rem 1rem 1.2rem;
    position: relative;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00f0ff 0%, #3b82f6 30%, #a855f7 60%, #ec4899 100%) !important;
    background-size: 200% 200% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0;
    line-height: 1.15;
    filter: drop-shadow(0 0 50px rgba(59,130,246,0.2));
    animation: heroGradient 8s ease-in-out infinite;
}
@keyframes heroGradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.hero-sub {
    font-size: 0.82rem;
    color: var(--sp-muted);
    margin-top: 0.5rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.hero-ver {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 1rem;
    color: #22c55e;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.hero-links {
    margin-top: 0.9rem;
    display: flex;
    gap: 1.4rem;
    justify-content: center;
    flex-wrap: wrap;
}
.hero-links a {
    color: var(--sp-muted) !important;
    text-decoration: none !important;
    font-size: 0.82rem;
    transition: color 0.25s, text-shadow 0.25s !important;
    position: relative;
}
.hero-links a:hover {
    color: var(--sp-cyan) !important;
    text-shadow: 0 0 14px rgba(0,240,255,0.35);
}
.hero-links a::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 0;
    height: 1.5px;
    background: linear-gradient(90deg, var(--sp-cyan), var(--sp-accent));
    transition: width 0.3s ease;
}
.hero-links a:hover::after { width: 100%; }

/* ── Stats bar ───────────────────────────────────────────────────── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    padding: 1rem 0 0.6rem;
    flex-wrap: wrap;
}
.stat-item {
    text-align: center;
    min-width: 80px;
    padding: 0.6rem 1rem;
    background: var(--sp-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--sp-border);
    border-radius: 12px;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.stat-item:hover {
    border-color: var(--sp-border-h);
    transform: translateY(-2px);
    box-shadow: var(--sp-glow-h);
}
.stat-num {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00f0ff, #3b82f6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    line-height: 1.2;
}
[data-theme="light"] .stat-num {
    background: linear-gradient(135deg, #0891b2, #2563eb) !important;
}
.stat-label {
    font-size: 0.65rem;
    color: var(--sp-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.15rem;
}

/* ── Divider ─────────────────────────────────────────────────────── */
.sp-divider {
    height: 1px;
    background: var(--sp-divider);
    margin: 0.5rem 0 0.8rem;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.tabs { border: none !important; }
.tab-nav {
    border: none !important;
    background: var(--sp-glass) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--sp-border) !important;
    gap: 2px !important;
}
button.selected {
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(168,85,247,0.12)) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-bottom: none !important;
    color: var(--sp-text) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 12px rgba(59,130,246,0.1) !important;
}
button.tab-nav-button {
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: var(--sp-muted) !important;
    border: 1px solid transparent !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    letter-spacing: 0.015em;
    padding: 8px 16px !important;
    border-radius: 8px !important;
}
button.tab-nav-button:hover {
    color: var(--sp-cyan) !important;
    background: rgba(0,240,255,0.04) !important;
}

/* ── Glass cards ──────────────────────────────────────────────────── */
.input-group, .block {
    background: var(--sp-card) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--sp-border) !important;
    border-radius: 14px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
.input-group:hover, .block:hover {
    border-color: var(--sp-border-h) !important;
    box-shadow: var(--sp-glow) !important;
}

/* ── Inputs ───────────────────────────────────────────────────────── */
input[type="range"] { accent-color: var(--sp-accent) !important; }
input[type="number"] {
    background: var(--sp-input) !important;
    color: var(--sp-text) !important;
    border: 1px solid var(--sp-border) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s !important;
}
input[type="number"]:focus {
    border-color: var(--sp-accent) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
    outline: none !important;
}
.wrap label span, .label-wrap span {
    font-weight: 500 !important;
    color: var(--sp-text2) !important;
}
.secondary-wrap, select, .wrap ul {
    background: var(--sp-input) !important;
    color: var(--sp-text) !important;
    border: 1px solid var(--sp-border) !important;
}

/* ── Stats readout ───────────────────────────────────────────────── */
.stats-readout {
    font-size: 0.78rem;
    color: var(--sp-muted);
    padding: 0.6rem 1rem;
    background: var(--sp-stats-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--sp-border);
    border-radius: 10px;
    text-align: center;
    min-height: 1.8rem;
}

/* ── Simulate button ─────────────────────────────────────────────── */
.primary {
    background: linear-gradient(135deg, #3b82f6, #6366f1, #8b5cf6) !important;
    background-size: 200% 200% !important;
    border: none !important;
    box-shadow: 0 0 24px rgba(59,130,246,0.3), inset 0 1px rgba(255,255,255,0.1) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em;
    animation: btnGradient 4s ease-in-out infinite !important;
}
@keyframes btnGradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.primary:hover {
    box-shadow: 0 0 40px rgba(59,130,246,0.45), 0 0 80px rgba(99,102,241,0.2),
                inset 0 1px rgba(255,255,255,0.15) !important;
    transform: translateY(-2px) scale(1.01) !important;
}
.primary:active {
    transform: translateY(0) scale(0.98) !important;
    box-shadow: 0 0 15px rgba(59,130,246,0.2) !important;
}

/* ── Export bar ───────────────────────────────────────────────────── */
.export-bar {
    display: flex !important;
    gap: 8px !important;
    align-items: center !important;
    padding: 0 !important;
}
.export-btn {
    min-width: 0 !important;
    padding: 6px 16px !important;
    font-size: 0.76rem !important;
    background: var(--sp-glass) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--sp-border) !important;
    border-radius: 8px !important;
    color: var(--sp-text2) !important;
    transition: all 0.25s !important;
}
.export-btn:hover {
    border-color: var(--sp-border-h) !important;
    color: #22c55e !important;
    box-shadow: 0 0 16px rgba(34,197,94,0.12) !important;
    transform: translateY(-1px) !important;
}

/* ── File download ───────────────────────────────────────────────── */
.export-file { max-height: 2.5rem !important; }
.export-file .file-preview { padding: 4px 8px !important; }

/* ── Plotly chart container ──────────────────────────────────────── */
.plotly .js-plotly-plot { border-radius: 12px; overflow: hidden; }
.plot-container {
    background: var(--sp-card-solid) !important;
    border-radius: 14px !important;
    border: 1px solid var(--sp-border) !important;
    box-shadow: var(--sp-glow) !important;
}
.plot-container:hover {
    box-shadow: var(--sp-glow-h) !important;
}
.gr-plot, .gr-plot > div { max-height: 600px !important; overflow: hidden !important; }

/* ── Footer ──────────────────────────────────────────────────────── */
.sp-footer {
    text-align: center;
    padding: 1.2rem;
    font-size: 0.72rem;
    color: var(--sp-muted);
    letter-spacing: 0.04em;
}
.sp-footer .sp-divider { margin: 0.5rem auto; max-width: 400px; }

/* ── i18n visibility ─────────────────────────────────────────────── */
[data-lang="zh"] .i18n-en { display: none !important; }
[data-lang="zh"] .i18n-zh { display: inline !important; }
[data-lang="en"] .i18n-zh, :root .i18n-zh { display: none !important; }
[data-lang="en"] .i18n-en, :root .i18n-en { display: inline !important; }

/* ── Loading shimmer ─────────────────────────────────────────────── */
.generating {
    position: relative;
    overflow: hidden;
}
.generating::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 200%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.06), rgba(168,85,247,0.04), transparent);
    animation: shimmer 2s ease infinite;
}
@keyframes shimmer {
    0% { transform: translateX(-50%); }
    100% { transform: translateX(50%); }
}

/* ── Responsive ──────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .hero-title { font-size: 2.2rem; }
    .stats-bar { gap: 0.4rem; }
    .stat-item { padding: 0.4rem 0.6rem; min-width: 60px; }
    .stat-num { font-size: 1.2rem; }
    .tab-nav { flex-wrap: wrap; }
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# JavaScript — theme toggle, i18n, system preference detection
# ═══════════════════════════════════════════════════════════════════════════

INIT_JS = """
() => {
    /* ── System preference detection ─────────────────────────────── */
    if (!document.documentElement.getAttribute('data-theme')) {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
    }
    if (!document.documentElement.getAttribute('data-lang')) {
        const navLang = (navigator.language || '').toLowerCase();
        document.documentElement.setAttribute('data-lang',
            navLang.startsWith('zh') ? 'zh' : 'en');
    }

    /* ── i18n translation map ────────────────────────────────────── */
    const T = {
        "Concentration (ppm)": "浓度 (ppm)",
        "Temperature (K)": "温度 (K)",
        "Pressure (atm)": "气压 (atm)",
        "Path length (m)": "光程 (m)",
        "Path (m)": "光程 (m)",
        "Simulate": "仿真",
        "Compare": "对比",
        "Molecule": "分子",
        "Molecule 1": "分子 1",
        "Molecule 2": "分子 2",
        "Conc. 1 (ppm)": "浓度 1 (ppm)",
        "Conc. 2 (ppm)": "浓度 2 (ppm)",
        "Mirror reflectivity": "镜面反射率",
        "Cavity (cm)": "腔长 (cm)",
        "Gas path (m)": "气体光程 (m)",
        "Max OPD (cm)": "最大光程差 (cm)",
        "Apodization": "切趾函数",
        "Poly order": "多项式阶数",
        "Rayleigh scattering": "瑞利散射",
        "Mod depth (cm⁻¹)": "调制深度 (cm⁻¹)",
        "Mod freq (Hz)": "调制频率 (Hz)",
        "Direct Absorption": "直接吸收",
        "Multi-Species": "多组分",
        "Absorption Spectrum": "吸收光谱",
        "WMS Signals": "WMS 信号",
        "Comparison": "多组分对比",
        "10 species with demo line lists": "包含10种气体演示谱线库",
        "Mie τ": "米氏消光 τ",
    };
    const Trev = {};
    Object.entries(T).forEach(([k, v]) => { Trev[v] = k; });

    window._spLang = document.documentElement.getAttribute('data-lang') || 'en';

    window.applyLang = function(lang) {
        window._spLang = lang;
        document.documentElement.setAttribute('data-lang', lang);
        const map = lang === 'zh' ? T : Trev;
        const selectors = [
            'span.svelte-1gfkn6j',
            '.label-wrap span',
            '.block label span',
            'button.tab-nav-button',
            '.gr-button span',
            '.wrap label span',
            '.info-msg span',
        ];
        document.querySelectorAll(selectors.join(',')).forEach(el => {
            const txt = el.textContent.trim();
            if (map[txt]) {
                if (!el.dataset.origText) el.dataset.origText = txt;
                el.textContent = map[txt];
            }
        });
        document.querySelectorAll('.i18n').forEach(el => {
            const val = el.getAttribute('data-' + lang);
            if (val) el.textContent = val;
        });
    };

    /* Re-apply after Gradio re-renders (skip Plotly mutations) */
    let _rafPending = false;
    const obs = new MutationObserver((mutations) => {
        if (window._spLang === 'en' || _rafPending) return;
        const isPlotly = mutations.every(m =>
            m.target.closest && m.target.closest('.js-plotly-plot, .plotly, .plot-container'));
        if (isPlotly) return;
        _rafPending = true;
        requestAnimationFrame(() => {
            window.applyLang(window._spLang);
            _rafPending = false;
        });
    });
    obs.observe(document.body, { childList: true, subtree: true });

    /* Apply initial lang if zh */
    setTimeout(() => {
        if (window._spLang === 'zh') window.applyLang('zh');
    }, 500);
}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Build the Gradio app
# ═══════════════════════════════════════════════════════════════════════════

molecule_names = list(MOLECULES.keys())

with gr.Blocks(
    title="SPEKTRAN Interactive Demo",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ),
    css=CUSTOM_CSS,
) as demo:

    theme_state = gr.State("dark")
    lang_state = gr.State("en")
    sim_data = gr.State(None)
    theme_bridge = gr.Textbox(visible=False, value="dark", elem_id="sp-theme-val")
    lang_bridge = gr.Textbox(visible=False, value="en", elem_id="sp-lang-val")

    # ── Control bar (theme + language) ────────────────────────────
    with gr.Row(elem_classes=["sp-ctrl-row"]):
        lang_btn = gr.Button("EN | 中", size="sm", elem_classes=["ctrl-btn"],
                             min_width=70, scale=0)
        theme_btn = gr.Button("🌙", size="sm", elem_classes=["ctrl-btn"],
                              min_width=44, scale=0)

    # ── Mesh background orbs ─────────────────────────────────────
    gr.HTML("""<div class="sp-mesh-bg"><div class="orb"></div><div class="orb"></div><div class="orb"></div></div>""")

    # ── Hero ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-wrap">
        <h1 class="hero-title">SPEKTRAN</h1>
        <p class="hero-sub">
            <span class="i18n i18n-en"
                  data-en="Optical Gas Sensing Simulation Engine"
                  data-zh="光学气体传感仿真引擎"
                  >Optical Gas Sensing Simulation Engine</span>
            <span class="i18n i18n-zh"
                  data-en="Optical Gas Sensing Simulation Engine"
                  data-zh="光学气体传感仿真引擎"
                  >光学气体传感仿真引擎</span>
            <span class="hero-ver">v0.6.0</span>
        </p>
        <div class="hero-links">
            <a href="https://github.com/spektran/spektran" target="_blank">GitHub</a>
            <a href="https://pypi.org/project/spektran/" target="_blank">PyPI</a>
            <a href="https://huggingface.co/datasets/spektran/spektran-ch4-v0" target="_blank">Dataset</a>
            <a href="https://spektran.github.io/spektran/" target="_blank">Docs</a>
            <a href="https://doi.org/10.5281/zenodo.21790394" target="_blank">DOI</a>
        </div>
        <div class="stats-bar">
            <div class="stat-item"><div class="stat-num">5</div>
                <div class="stat-label i18n" data-en="Modalities" data-zh="模态">Modalities</div></div>
            <div class="stat-item"><div class="stat-num">46</div>
                <div class="stat-label i18n" data-en="Instruments" data-zh="虚拟仪器">Instruments</div></div>
            <div class="stat-item"><div class="stat-num">25</div>
                <div class="stat-label i18n" data-en="Baselines" data-zh="基线模型">Baselines</div></div>
            <div class="stat-item"><div class="stat-num">10</div>
                <div class="stat-label i18n" data-en="Molecules" data-zh="气体分子">Molecules</div></div>
            <div class="stat-item"><div class="stat-num">9</div>
                <div class="stat-label i18n" data-en="Tasks" data-zh="基准任务">Tasks</div></div>
        </div>
        <div class="sp-divider"></div>
    </div>
    """)

    # ── Tabs ──────────────────────────────────────────────────────
    with gr.Tabs():

        # ─── Direct Absorption ────────────────────────────────────
        with gr.Tab("Direct Absorption"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    da_mol = gr.Dropdown(molecule_names, value="CH4", label="Molecule",
                                         info="10 species with demo line lists")
                    da_conc = gr.Slider(1, 10000, value=100, step=1,
                                        label="Concentration (ppm)")
                    da_temp = gr.Slider(200, 1500, value=296, step=1,
                                        label="Temperature (K)")
                    da_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01,
                                        label="Pressure (atm)")
                    da_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1,
                                        label="Path length (m)")
                    da_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    da_plot = gr.Plot(label="Absorption Spectrum")
                    da_stats = gr.HTML(elem_classes=["stats-readout"])
            da_in = [da_mol, da_conc, da_temp, da_pres, da_path, theme_state]
            da_out = [da_plot, da_stats, sim_data]
            da_btn.click(plot_da, da_in, da_out)
            for inp in [da_mol, da_conc, da_temp, da_pres, da_path]:
                inp.change(plot_da, da_in, da_out)

        # ─── WMS ─────────────────────────────────────────────────
        with gr.Tab("WMS"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    wms_mol = gr.Dropdown(molecule_names, value="CH4", label="Molecule")
                    wms_conc = gr.Slider(1, 10000, value=100, step=1,
                                         label="Concentration (ppm)")
                    wms_temp = gr.Slider(200, 1500, value=296, step=1,
                                         label="Temperature (K)")
                    wms_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01,
                                         label="Pressure (atm)")
                    wms_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1,
                                         label="Path length (m)")
                    wms_mod = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                        label="Mod depth (cm⁻¹)")
                    wms_freq = gr.Slider(1000, 100000, value=10000, step=1000,
                                         label="Mod freq (Hz)")
                    wms_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    wms_plot = gr.Plot(label="WMS Signals")
                    wms_stats = gr.HTML(elem_classes=["stats-readout"])
            wms_in = [wms_mol, wms_conc, wms_temp, wms_pres, wms_path,
                      wms_mod, wms_freq, theme_state]
            wms_btn.click(plot_wms, wms_in, [wms_plot, wms_stats, sim_data])

        # ─── Multi-Species ────────────────────────────────────────
        with gr.Tab("Multi-Species"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    ms_mol1 = gr.Dropdown(molecule_names, value="CH4", label="Molecule 1")
                    ms_conc1 = gr.Slider(1, 10000, value=100, step=1,
                                         label="Conc. 1 (ppm)")
                    ms_mol2 = gr.Dropdown(molecule_names, value="H2O", label="Molecule 2")
                    ms_conc2 = gr.Slider(1, 50000, value=10000, step=10,
                                         label="Conc. 2 (ppm)")
                    ms_temp = gr.Slider(200, 1500, value=296, step=1,
                                        label="Temperature (K)")
                    ms_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01,
                                        label="Pressure (atm)")
                    ms_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1,
                                        label="Path length (m)")
                    ms_btn = gr.Button("Compare", variant="primary", size="lg")
                with gr.Column(scale=3):
                    ms_plot = gr.Plot(label="Comparison")
                    ms_stats = gr.HTML(elem_classes=["stats-readout"])
            ms_in = [ms_mol1, ms_conc1, ms_mol2, ms_conc2, ms_temp, ms_pres,
                     ms_path, theme_state]
            ms_btn.click(plot_compare, ms_in, [ms_plot, ms_stats, sim_data])

        # ─── CRDS ────────────────────────────────────────────────
        with gr.Tab("CRDS"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    crds_conc = gr.Slider(1, 5000, value=100, step=1,
                                          label="CH₄ (ppm)")
                    crds_r = gr.Slider(0.9999, 0.999999, value=0.99995,
                                       step=0.000001, label="Mirror reflectivity")
                    crds_len = gr.Slider(10, 100, value=50, step=1,
                                         label="Cavity (cm)")
                    crds_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1,
                                          label="Gas path (m)")
                    crds_temp = gr.Slider(200, 500, value=296, step=1,
                                          label="Temperature (K)")
                    crds_pres = gr.Slider(0.01, 2.0, value=1.0, step=0.01,
                                          label="Pressure (atm)")
                    crds_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    crds_plot = gr.Plot(label="CRDS")
                    crds_stats = gr.HTML(elem_classes=["stats-readout"])
            crds_in = [crds_conc, crds_r, crds_len, crds_path, crds_temp,
                       crds_pres, theme_state]
            crds_btn.click(plot_crds, crds_in, [crds_plot, crds_stats, sim_data])

        # ─── FTIR ────────────────────────────────────────────────
        with gr.Tab("FTIR"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    ftir_conc = gr.Slider(10, 5000, value=500, step=10,
                                          label="CH₄ (ppm)")
                    ftir_opd = gr.Slider(0.5, 50.0, value=10.0, step=0.5,
                                         label="Max OPD (cm)")
                    ftir_apod = gr.Dropdown(
                        ["boxcar", "triangular", "happ_genzel",
                         "norton_beer_medium", "norton_beer_strong"],
                        value="happ_genzel", label="Apodization")
                    ftir_path = gr.Slider(0.1, 50.0, value=10.0, step=0.1,
                                          label="Path (m)")
                    ftir_temp = gr.Slider(200, 500, value=296, step=1,
                                          label="Temperature (K)")
                    ftir_pres = gr.Slider(0.01, 2.0, value=1.0, step=0.01,
                                          label="Pressure (atm)")
                    ftir_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    ftir_plot = gr.Plot(label="FTIR")
                    ftir_stats = gr.HTML(elem_classes=["stats-readout"])
            ftir_in = [ftir_conc, ftir_opd, ftir_apod, ftir_path, ftir_temp,
                       ftir_pres, theme_state]
            ftir_btn.click(plot_ftir, ftir_in, [ftir_plot, ftir_stats, sim_data])

        # ─── DOAS ────────────────────────────────────────────────
        with gr.Tab("DOAS"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    doas_conc = gr.Slider(0.01, 100.0, value=1.0, step=0.01,
                                          label="SO₂ (ppm)")
                    doas_path = gr.Slider(10, 5000, value=500, step=10,
                                          label="Path (m)")
                    doas_poly = gr.Slider(1, 9, value=5, step=1,
                                          label="Poly order")
                    doas_ray = gr.Checkbox(value=True, label="Rayleigh scattering")
                    doas_mie = gr.Slider(0.0, 2.0, value=0.3, step=0.01,
                                         label="Mie τ")
                    doas_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    doas_plot = gr.Plot(label="DOAS")
                    doas_stats = gr.HTML(elem_classes=["stats-readout"])
            doas_in = [doas_conc, doas_path, doas_poly, doas_ray, doas_mie,
                       theme_state]
            doas_btn.click(plot_doas, doas_in, [doas_plot, doas_stats, sim_data])

    # ── Export bar ────────────────────────────────────────────────
    with gr.Row(elem_classes=["export-bar"]):
        csv_btn = gr.Button("📄 CSV", size="sm", elem_classes=["export-btn"],
                            min_width=80)
        json_btn = gr.Button("📋 JSON", size="sm", elem_classes=["export-btn"],
                             min_width=80)
        gr.HTML("""<span style='font-size:0.72rem; color:var(--sp-muted);
                    font-family:JetBrains Mono,monospace'>
                    <span class="i18n" data-en="PNG/SVG via chart toolbar ↗"
                          data-zh="PNG/SVG 可通过图表工具栏导出 ↗">PNG/SVG via chart toolbar ↗</span>
                    </span>""")
        export_file = gr.File(visible=False, elem_classes=["export-file"])

    csv_btn.click(export_csv, [sim_data], [export_file])
    json_btn.click(export_json, [sim_data], [export_file])

    # ── Footer ────────────────────────────────────────────────────
    gr.HTML("""
    <div class="sp-footer">
        SPEKTRAN v0.6.0 &nbsp;│&nbsp; Apache-2.0 (code) &nbsp;│&nbsp; CC BY 4.0 (data)
        &nbsp;│&nbsp;
        <span class="i18n" data-en="Built for the ML + spectroscopy community"
              data-zh="为机器学习 + 光谱学社区而建">Built for the ML + spectroscopy community</span>
    </div>
    """)

    # ── Theme toggle ──────────────────────────────────────────────
    def _toggle_theme(current):
        new = "light" if current == "dark" else "dark"
        icon = "☀️" if new == "light" else "🌙"
        return new, gr.update(value=icon), new

    theme_btn.click(_toggle_theme, [theme_state],
                    [theme_state, theme_btn, theme_bridge])

    theme_bridge.change(
        fn=None, inputs=[theme_bridge],
        js="(theme) => { document.documentElement.setAttribute('data-theme', theme); }"
    )

    # ── Language toggle ───────────────────────────────────────────
    def _toggle_lang(current):
        new = "zh" if current == "en" else "en"
        label = "中文" if new == "zh" else "EN"
        return new, gr.update(value=label), new

    lang_btn.click(_toggle_lang, [lang_state],
                   [lang_state, lang_btn, lang_bridge])

    lang_bridge.change(
        fn=None, inputs=[lang_bridge],
        js="(lang) => { if(window.applyLang) window.applyLang(lang); }"
    )

    # ── Init JS on load ──────────────────────────────────────────
    demo.load(fn=None, js=INIT_JS)


if __name__ == "__main__":
    demo.launch()
