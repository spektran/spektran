"""SPEKTRAN Interactive Demo — 5-modality optical gas sensing simulation.

Hosted on Hugging Face Spaces. Plotly interactive charts, sci-fi theme.
"""

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

PLOT_BG = "rgba(10, 14, 26, 0)"
GRID_COLOR = "rgba(100, 150, 255, 0.08)"
AXIS_COLOR = "rgba(150, 180, 220, 0.4)"
FONT_COLOR = "#94a3b8"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="JetBrains Mono, SF Mono, Consolas, monospace", color=FONT_COLOR, size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
               linecolor=AXIS_COLOR, tickfont=dict(size=11)),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
               linecolor=AXIS_COLOR, tickfont=dict(size=11)),
    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#3b82f6",
                    font=dict(color="#e2e8f0", size=12)),
    legend=dict(bgcolor="rgba(10,14,26,0.7)", bordercolor="rgba(100,150,255,0.2)",
                borderwidth=1, font=dict(size=11)),
)


def _apply_layout(fig, title="", xlab="", ylab="", height=460):
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=15, color="#e2e8f0"), x=0.02, xanchor="left"),
        xaxis_title=xlab, yaxis_title=ylab, height=height,
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="#475569", activecolor="#3b82f6"),
    )
    return fig


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


def plot_da(molecule, conc, temp, pres, path_m):
    nu, absorbance = _make_absorbance(molecule, conc, temp, pres, path_m)
    c = MOLECULES[molecule]["color"]
    band = MOLECULES[molecule]["band"]
    wl = MOLECULES[molecule]["wl"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nu, y=absorbance, mode="lines",
        line=dict(color=c, width=2.2),
        fill="tozeroy", fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.08)",
        hovertemplate="<b>%{x:.2f}</b> cm⁻¹<br>Abs: %{y:.4e}<extra></extra>",
        name=molecule,
    ))
    peak = absorbance.max()
    peak_nu = nu[np.argmax(absorbance)]
    fig.add_annotation(
        x=peak_nu, y=peak, text=f"peak {peak:.3e}",
        showarrow=True, arrowhead=2, arrowcolor=c, arrowwidth=1.5,
        font=dict(size=11, color=c), bgcolor="rgba(10,14,26,0.8)",
        bordercolor=c, borderwidth=1, borderpad=4,
        ax=0, ay=-40,
    )

    title = f"<b>{molecule}</b> Direct Absorption — {band} ({wl})"
    stats = (f"<span style='color:{c}'>Peak: {peak:.3e}</span> &nbsp;│&nbsp; "
             f"T={temp:.0f} K &nbsp;│&nbsp; P={pres:.2f} atm &nbsp;│&nbsp; "
             f"L={path_m:.1f} m &nbsp;│&nbsp; [{molecule}]={conc:.0f} ppm")

    _apply_layout(fig, title, "Wavenumber (cm⁻¹)", "Absorbance (napierian)")
    return fig, stats


def plot_wms(molecule, conc, temp, pres, path_m, mod_depth, mod_freq):
    lines, nu0, nu1 = _get_lines_and_range(molecule)
    center = (nu0 + nu1) / 2.0
    scan_range = (nu1 - nu0) * 0.8
    c = MOLECULES[molecule]["color"]

    cfg = WMSConfig(
        modulation_frequency_Hz=mod_freq,
        modulation_depth_cm1=mod_depth,
        sampling_rate_Hz=mod_freq * 20,
        duration_s=0.2,
        center_wavenumber_cm1=center,
        scan_range_cm1=scan_range,
        scan_rate_Hz=10.0,
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
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, row=i, col=1)

    band = MOLECULES[molecule]["band"]
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        title=dict(text=f"<b>{molecule}</b> WMS — {band}",
                   font=dict(size=15, color="#e2e8f0"), x=0.02),
        height=580, showlegend=True,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#94a3b8")

    stats = (f"mod: {mod_depth:.2f} cm⁻¹ @ {mod_freq/1000:.0f} kHz &nbsp;│&nbsp; "
             f"[{molecule}]={conc:.0f} ppm")
    return fig, stats


def plot_crds(conc, mirror_r, cavity_cm, path_m, temp, pres):
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
        name="τ(v)", hovertemplate="%{x:.2f} cm⁻¹<br>τ=%{y:.2f} µs<extra></extra>",
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
    for i in (1, 2):
        fig.update_xaxes(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, row=i, col=1)

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        title=dict(text="<b>CRDS</b> — CH₄ Cavity Ring-Down",
                   font=dict(size=15, color="#e2e8f0"), x=0.02),
        height=580,
    )
    for ann in fig.layout.annotations:
        if hasattr(ann, "font") and ann.font:
            pass
        else:
            ann.font = dict(size=12, color="#94a3b8")

    stats = (f"<span style='color:#a855f7'>τₘᵢₙ: {tau.min()*1e6:.2f} µs</span> &nbsp;│&nbsp; "
             f"τ₀: {tau0*1e6:.2f} µs &nbsp;│&nbsp; "
             f"R={mirror_r:.6f} &nbsp;│&nbsp; [{conc:.0f}] ppm")
    return fig, stats


def plot_ftir(conc, max_opd, apod, path_m, temp, pres):
    lines = demo_ch4_2nu3()
    result = simulate_ftir_spectrum(
        lines=lines, molecule="CH4",
        concentration_ppm=conc, temperature_K=temp, pressure_atm=pres,
        path_length_m=path_m, max_opd_cm=max_opd,
        wavenumber_start_cm1=6045.0, wavenumber_end_cm1=6049.0,
        n_hires_points=10000, n_output_points=500,
        apod_function=apod,
    )
    res = spectral_resolution_cm1(max_opd)
    nu = result["nu_cm1"]
    spec = result["spectrum_hires"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nu, y=spec, mode="lines",
        line=dict(color="#0891b2", width=2.2),
        fill="tozeroy", fillcolor="rgba(8,145,178,0.06)",
        hovertemplate="%{x:.2f} cm⁻¹<br>T=%{y:.5f}<extra></extra>",
        name="transmittance",
    ))
    _apply_layout(fig,
                  f"<b>FTIR</b> — CH₄ Transmittance (Δν = {res:.3f} cm⁻¹)",
                  "Wavenumber (cm⁻¹)", "Transmittance")
    stats = (f"<span style='color:#0891b2'>res: {res:.3f} cm⁻¹</span> &nbsp;│&nbsp; "
             f"apod: {apod} &nbsp;│&nbsp; OPD: {max_opd:.1f} cm &nbsp;│&nbsp; "
             f"[{conc:.0f}] ppm")
    return fig, stats


def plot_doas(conc, path_m, poly_order, rayleigh, mie_tau):
    wl = np.linspace(300.0, 360.0, 500)
    sigma = simulate_doas_cross_section(
        wl, center_nm=330.0, peak_cross_section_cm2=6e-19,
        n_features=5, feature_width_nm=0.8,
    )
    result = simulate_doas_spectrum(
        wavelength_nm=wl, target_sigma_cm2=sigma,
        target_concentration_ppm=conc,
        temperature_K=296.0, pressure_atm=1.0,
        path_length_m=path_m, poly_order=poly_order,
        rayleigh=rayleigh, mie_tau_ref=mie_tau,
    )

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=("Total Optical Density", "Differential OD (after polynomial high-pass)"))
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
    for i in (1, 2):
        fig.update_xaxes(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, row=i, col=1)

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        title=dict(text="<b>DOAS</b> — SO₂ Differential Optical Absorption",
                   font=dict(size=15, color="#e2e8f0"), x=0.02),
        height=580,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#94a3b8")

    stats = (f"SO₂: {conc:.2f} ppm &nbsp;│&nbsp; "
             f"path: {path_m:.0f} m &nbsp;│&nbsp; "
             f"poly: {poly_order} &nbsp;│&nbsp; "
             f"Mie τ: {mie_tau:.2f}")
    return fig, stats


def plot_compare(mol1, c1, mol2, c2, temp, pres, path_m):
    nu1, abs1 = _make_absorbance(mol1, c1, temp, pres, path_m)
    nu2, abs2 = _make_absorbance(mol2, c2, temp, pres, path_m)
    c1c = MOLECULES[mol1]["color"]
    c2c = MOLECULES[mol2]["color"]

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12,
                        subplot_titles=(
                            f"{mol1} — {MOLECULES[mol1]['band']} ({MOLECULES[mol1]['wl']})",
                            f"{mol2} — {MOLECULES[mol2]['band']} ({MOLECULES[mol2]['wl']})",
                        ))
    fig.add_trace(go.Scatter(
        x=nu1, y=abs1, mode="lines", line=dict(color=c1c, width=2.2),
        fill="tozeroy",
        fillcolor=f"rgba({int(c1c[1:3],16)},{int(c1c[3:5],16)},{int(c1c[5:7],16)},0.06)",
        name=f"{mol1} ({c1:.0f} ppm)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=nu2, y=abs2, mode="lines", line=dict(color=c2c, width=2.2),
        fill="tozeroy",
        fillcolor=f"rgba({int(c2c[1:3],16)},{int(c2c[3:5],16)},{int(c2c[5:7],16)},0.06)",
        name=f"{mol2} ({c2:.0f} ppm)",
    ), row=2, col=1)
    for i in (1, 2):
        fig.update_xaxes(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR,
                         title_text="Wavenumber (cm⁻¹)", row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=AXIS_COLOR,
                         title_text="Absorbance", row=i, col=1)

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        title=dict(text="<b>Multi-Species</b> Comparison",
                   font=dict(size=15, color="#e2e8f0"), x=0.02),
        height=580,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="#94a3b8")

    stats = (f"<span style='color:{c1c}'>{mol1}: {abs1.max():.3e}</span> &nbsp;│&nbsp; "
             f"<span style='color:{c2c}'>{mol2}: {abs2.max():.3e}</span>")
    return fig, stats


# ─── custom CSS ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #0f1525;
    --bg-input: #141b2d;
    --border-glow: rgba(59, 130, 246, 0.25);
    --accent-cyan: #00f0ff;
    --accent-blue: #3b82f6;
    --accent-purple: #a855f7;
    --text-primary: #e2e8f0;
    --text-muted: #64748b;
}

/* Force dark background everywhere */
.gradio-container { background: var(--bg-primary) !important; }
.main, .contain { background: var(--bg-primary) !important; }
footer { display: none !important; }

/* Animated hero header */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg at 50% 50%,
        transparent 0deg, rgba(59,130,246,0.03) 60deg,
        transparent 120deg, rgba(168,85,247,0.03) 180deg,
        transparent 240deg, rgba(0,240,255,0.03) 300deg,
        transparent 360deg);
    animation: rotate-bg 20s linear infinite;
}
@keyframes rotate-bg {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00f0ff 0%, #3b82f6 40%, #a855f7 70%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    margin: 0;
    line-height: 1.2;
    text-shadow: 0 0 80px rgba(59,130,246,0.3);
}
.hero-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.5rem;
    position: relative;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.hero-version {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 1rem;
    color: #22c55e;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
.hero-links {
    margin-top: 1rem;
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    flex-wrap: wrap;
    position: relative;
}
.hero-links a {
    color: #64748b !important;
    text-decoration: none !important;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    transition: color 0.3s, text-shadow 0.3s;
}
.hero-links a:hover {
    color: #00f0ff !important;
    text-shadow: 0 0 12px rgba(0, 240, 255, 0.4);
}

/* Stats bar */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2rem;
    padding: 1rem 0;
    flex-wrap: wrap;
    position: relative;
}
.stat-item {
    text-align: center;
    min-width: 80px;
}
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00f0ff, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Inter', sans-serif;
}

/* Tab styling */
.tabs { border: none !important; }
.tab-nav { border: none !important; background: transparent !important; }
button.selected {
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(168,85,247,0.1)) !important;
    border-bottom: 2px solid #3b82f6 !important;
    color: #e2e8f0 !important;
}
button.tab-nav-button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    color: #64748b !important;
    border: none !important;
    transition: all 0.3s !important;
    letter-spacing: 0.02em;
}
button.tab-nav-button:hover {
    color: #00f0ff !important;
    background: rgba(0,240,255,0.05) !important;
}

/* Input cards with glow */
.input-group {
    background: var(--bg-card) !important;
    border: 1px solid rgba(59,130,246,0.1) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.input-group:hover {
    border-color: rgba(59,130,246,0.3) !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.05) !important;
}

/* Slider styling */
input[type="range"] {
    accent-color: #3b82f6 !important;
}
.wrap label span {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
}

/* Stats readout below chart */
.stats-readout {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    padding: 0.5rem 1rem;
    background: rgba(15, 21, 37, 0.6);
    border: 1px solid rgba(59,130,246,0.1);
    border-radius: 8px;
    text-align: center;
    backdrop-filter: blur(8px);
    min-height: 2rem;
}

/* Plotly container */
.plotly .js-plotly-plot { border-radius: 8px; overflow: hidden; }

/* Button glow */
.primary {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.3) !important;
    transition: box-shadow 0.3s, transform 0.2s !important;
}
.primary:hover {
    box-shadow: 0 0 30px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Section divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.2), rgba(168,85,247,0.2), transparent);
    margin: 0.5rem 0 1rem;
}

/* Dropdown */
.wrap .secondary-wrap { background: var(--bg-input) !important; }
"""


# ─── Build the app ─────────────────────────────────────────────────────────
molecule_names = list(MOLECULES.keys())

with gr.Blocks(
    title="SPEKTRAN Interactive Demo",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0a0e1a",
        body_background_fill_dark="#0a0e1a",
        block_background_fill="#0f1525",
        block_background_fill_dark="#0f1525",
        block_border_color="rgba(59,130,246,0.1)",
        block_border_color_dark="rgba(59,130,246,0.1)",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        input_background_fill="#141b2d",
        input_background_fill_dark="#141b2d",
        input_border_color="rgba(59,130,246,0.15)",
        input_border_color_dark="rgba(59,130,246,0.15)",
        button_primary_background_fill="linear-gradient(135deg, #3b82f6, #6366f1)",
        button_primary_background_fill_dark="linear-gradient(135deg, #3b82f6, #6366f1)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        slider_color="#3b82f6",
        slider_color_dark="#3b82f6",
    ),
    css=CUSTOM_CSS,
) as demo:

    # ── Hero ───────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-header">
        <h1 class="hero-title">SPEKTRAN</h1>
        <p class="hero-subtitle">
            Optical Gas Sensing Simulation Engine
            <span class="hero-version">v0.6.0</span>
        </p>
        <div class="hero-links">
            <a href="https://github.com/spektran/spektran" target="_blank">GitHub</a>
            <a href="https://pypi.org/project/spektran/" target="_blank">PyPI</a>
            <a href="https://huggingface.co/datasets/spektran/spektran-ch4-v0" target="_blank">Dataset</a>
            <a href="https://spektran.github.io/spektran/" target="_blank">Docs</a>
            <a href="https://doi.org/10.5281/zenodo.21790394" target="_blank">DOI</a>
        </div>
        <div class="stats-bar">
            <div class="stat-item"><div class="stat-num">5</div><div class="stat-label">Modalities</div></div>
            <div class="stat-item"><div class="stat-num">46</div><div class="stat-label">Instruments</div></div>
            <div class="stat-item"><div class="stat-num">25</div><div class="stat-label">Baselines</div></div>
            <div class="stat-item"><div class="stat-num">10</div><div class="stat-label">Molecules</div></div>
            <div class="stat-item"><div class="stat-num">9</div><div class="stat-label">Tasks</div></div>
        </div>
        <div class="section-divider"></div>
    </div>
    """)

    # ── Tabs ───────────────────────────────────────────────────────
    with gr.Tabs():

        # ─── Direct Absorption ─────────────────────────────────────
        with gr.Tab("Direct Absorption"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    da_mol = gr.Dropdown(molecule_names, value="CH4", label="Molecule",
                                         info="10 species with demo line lists")
                    da_conc = gr.Slider(1, 10000, value=100, step=1, label="Concentration (ppm)")
                    da_temp = gr.Slider(200, 1500, value=296, step=1, label="Temperature (K)")
                    da_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Pressure (atm)")
                    da_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1, label="Path length (m)")
                    da_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    da_plot = gr.Plot(label="Absorption Spectrum")
                    da_stats = gr.HTML(elem_classes=["stats-readout"])
            da_inputs = [da_mol, da_conc, da_temp, da_pres, da_path]
            da_outputs = [da_plot, da_stats]
            da_btn.click(plot_da, da_inputs, da_outputs)
            for inp in da_inputs:
                inp.change(plot_da, da_inputs, da_outputs)

        # ─── WMS ──────────────────────────────────────────────────
        with gr.Tab("WMS"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    wms_mol = gr.Dropdown(molecule_names, value="CH4", label="Molecule")
                    wms_conc = gr.Slider(1, 10000, value=100, step=1, label="Concentration (ppm)")
                    wms_temp = gr.Slider(200, 1500, value=296, step=1, label="Temperature (K)")
                    wms_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Pressure (atm)")
                    wms_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1, label="Path length (m)")
                    wms_mod = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Mod depth (cm⁻¹)")
                    wms_freq = gr.Slider(1000, 100000, value=10000, step=1000, label="Mod freq (Hz)")
                    wms_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    wms_plot = gr.Plot(label="WMS Signals")
                    wms_stats = gr.HTML(elem_classes=["stats-readout"])
            wms_inputs = [wms_mol, wms_conc, wms_temp, wms_pres, wms_path, wms_mod, wms_freq]
            wms_btn.click(plot_wms, wms_inputs, [wms_plot, wms_stats])

        # ─── Multi-Species ─────────────────────────────────────────
        with gr.Tab("Multi-Species"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    ms_mol1 = gr.Dropdown(molecule_names, value="CH4", label="Molecule 1")
                    ms_conc1 = gr.Slider(1, 10000, value=100, step=1, label="Conc. 1 (ppm)")
                    ms_mol2 = gr.Dropdown(molecule_names, value="H2O", label="Molecule 2")
                    ms_conc2 = gr.Slider(1, 50000, value=10000, step=10, label="Conc. 2 (ppm)")
                    ms_temp = gr.Slider(200, 1500, value=296, step=1, label="Temperature (K)")
                    ms_pres = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Pressure (atm)")
                    ms_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1, label="Path length (m)")
                    ms_btn = gr.Button("Compare", variant="primary", size="lg")
                with gr.Column(scale=3):
                    ms_plot = gr.Plot(label="Comparison")
                    ms_stats = gr.HTML(elem_classes=["stats-readout"])
            ms_inputs = [ms_mol1, ms_conc1, ms_mol2, ms_conc2, ms_temp, ms_pres, ms_path]
            ms_btn.click(plot_compare, ms_inputs, [ms_plot, ms_stats])

        # ─── CRDS ─────────────────────────────────────────────────
        with gr.Tab("CRDS"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    crds_conc = gr.Slider(1, 5000, value=100, step=1, label="CH₄ (ppm)")
                    crds_r = gr.Slider(0.9999, 0.999999, value=0.99995, step=0.000001,
                                       label="Mirror reflectivity")
                    crds_len = gr.Slider(10, 100, value=50, step=1, label="Cavity (cm)")
                    crds_path = gr.Slider(0.1, 100.0, value=10.0, step=0.1, label="Gas path (m)")
                    crds_temp = gr.Slider(200, 500, value=296, step=1, label="Temperature (K)")
                    crds_pres = gr.Slider(0.01, 2.0, value=1.0, step=0.01, label="Pressure (atm)")
                    crds_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    crds_plot = gr.Plot(label="CRDS")
                    crds_stats = gr.HTML(elem_classes=["stats-readout"])
            crds_inputs = [crds_conc, crds_r, crds_len, crds_path, crds_temp, crds_pres]
            crds_btn.click(plot_crds, crds_inputs, [crds_plot, crds_stats])

        # ─── FTIR ─────────────────────────────────────────────────
        with gr.Tab("FTIR"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    ftir_conc = gr.Slider(10, 5000, value=500, step=10, label="CH₄ (ppm)")
                    ftir_opd = gr.Slider(0.5, 50.0, value=10.0, step=0.5, label="Max OPD (cm)")
                    ftir_apod = gr.Dropdown(
                        ["boxcar", "triangular", "happ_genzel",
                         "norton_beer_medium", "norton_beer_strong"],
                        value="happ_genzel", label="Apodization")
                    ftir_path = gr.Slider(0.1, 50.0, value=10.0, step=0.1, label="Path (m)")
                    ftir_temp = gr.Slider(200, 500, value=296, step=1, label="Temperature (K)")
                    ftir_pres = gr.Slider(0.01, 2.0, value=1.0, step=0.01, label="Pressure (atm)")
                    ftir_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    ftir_plot = gr.Plot(label="FTIR")
                    ftir_stats = gr.HTML(elem_classes=["stats-readout"])
            ftir_inputs = [ftir_conc, ftir_opd, ftir_apod, ftir_path, ftir_temp, ftir_pres]
            ftir_btn.click(plot_ftir, ftir_inputs, [ftir_plot, ftir_stats])

        # ─── DOAS ─────────────────────────────────────────────────
        with gr.Tab("DOAS"):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    doas_conc = gr.Slider(0.01, 100.0, value=1.0, step=0.01, label="SO₂ (ppm)")
                    doas_path = gr.Slider(10, 5000, value=500, step=10, label="Path (m)")
                    doas_poly = gr.Slider(1, 9, value=5, step=1, label="Poly order")
                    doas_ray = gr.Checkbox(value=True, label="Rayleigh scattering")
                    doas_mie = gr.Slider(0.0, 2.0, value=0.3, step=0.01, label="Mie τ")
                    doas_btn = gr.Button("Simulate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    doas_plot = gr.Plot(label="DOAS")
                    doas_stats = gr.HTML(elem_classes=["stats-readout"])
            doas_inputs = [doas_conc, doas_path, doas_poly, doas_ray, doas_mie]
            doas_btn.click(plot_doas, doas_inputs, [doas_plot, doas_stats])

    gr.HTML("""
    <div style="text-align:center; padding:1rem; font-size:0.75rem; color:#334155;
                font-family:'JetBrains Mono',monospace; letter-spacing:0.05em;">
        SPEKTRAN v0.6.0 &nbsp;│&nbsp; Apache-2.0 (code) &nbsp;│&nbsp; CC BY 4.0 (data)
        &nbsp;│&nbsp; Built for the ML + spectroscopy community
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
