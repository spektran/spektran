# SPEKTRAN

<div style="text-align: center; margin: 1em 0 2em;">
<strong style="font-size: 1.2em;">The MNIST of Gas Sensing — AI Agent Ready</strong><br>
<em>Simulate, train, and evaluate optical gas sensing ML models via natural language</em>
</div>

[:material-play-circle: Try the Demo](https://huggingface.co/spaces/spektran/spektran-demo){ .md-button .md-button--primary }
[:material-download: Install](quickstart.md){ .md-button }
[:material-trophy: Leaderboard](leaderboard.md){ .md-button }
[:material-robot: Agent Interface](https://github.com/spektran/spektran/blob/main/AGENTS.md){ .md-button }

---

## What's Inside

<div class="grid cards" markdown>

-   :material-molecule:{ .lg .middle } **10 Molecules**

    ---

    CH4, H2O, CO2, CO, NH3, NO, NO2, SO2, HCl, HF —
    HITRAN line-by-line physics with TIPS partition functions

-   :material-sine-wave:{ .lg .middle } **Advanced Line Shapes**

    ---

    Voigt profile and Hartmann-Tran Profile (HTP) with
    speed-dependent broadening, Dicke narrowing, and correlation

-   :material-chart-timeline-variant:{ .lg .middle } **WMS Chain**

    ---

    1f–4f lock-in demodulation, 2f/1f calibration-free ratio,
    etalon fringes in the time-domain chain

-   :material-cog-outline:{ .lg .middle } **14+ Virtual Instruments**

    ---

    Laser scan nonlinearity, thermal chirp, RIN, etalon fringes,
    window contamination, beam wander, ADC quantization

-   :material-flask-outline:{ .lg .middle } **2 Modalities**

    ---

    TDLAS (direct absorption + wavelength modulation) and
    NDIR (Planck source + bandpass filter)

-   :material-trophy-outline:{ .lg .middle } **9 Benchmark Tasks**

    ---

    Concentration regression, denoising, cross-instrument,
    WMS, drift, OOD, cross-modality, multi-species, temperature

-   :material-robot:{ .lg .middle } **AI Agent Ready**

    ---

    Every CLI command outputs JSON. Discovery, training, and
    evaluation via natural language with any AI coding agent

</div>

---

## Quick Start

=== "Hugging Face (zero install)"

    ```python
    from datasets import load_dataset
    ds = load_dataset("spektran/spektran-ch4-v0")
    ```

=== "pip install"

    ```bash
    pip install spektran
    ```

    ```python
    from spektran.physics import simulate_absorbance

    nu, absorbance = simulate_absorbance(
        molecule="CH4", concentration_ppm=100.0,
        temperature_K=296.0, pressure_atm=1.0,
        path_length_m=10.0,
        wavenumber_start_cm1=6046.0, wavenumber_end_cm1=6048.0,
    )
    ```

=== "CLI"

    ```bash
    spektran generate configs/datasets/ch4-t1-train-v0.yaml --out data
    spektran benchmark --task T1-concentration \
        --truth data/test.h5 --predictions preds.csv
    ```

=== "AI Agent"

    Tell your AI coding agent (Claude Code, Cursor, etc.):

    > "Train the ridge baseline on T1 and show me the scores"

    The agent reads [`AGENTS.md`](https://github.com/spektran/spektran/blob/main/AGENTS.md) and runs:

    ```bash
    spektran info --json                      # Bootstrap
    spektran train --baseline ridge --json    # Auto-generates data + trains
    # → {"scores": {"T1": {"mae_ppm": 2.84, "mape_pct": 29.87}}}
    ```

---

## Key Results

!!! tip "The flagship finding"
    Model complexity correlates with instrument overfitting.
    Ridge degrades **1.31x**, Transformer **1.46x**, CNN **1.82x**
    on held-out instruments — can you build a model that breaks this pattern?

| Model | T1 MAE (ppm) | T3 Degradation |
|:------|:------------:|:--------------:|
| Ridge regression | **2.84** | **1.31x** |
| Patchified Transformer | 7.39 | 1.46x |
| 1D CNN | 15.58 | 1.82x |

[View full leaderboard :material-arrow-right:](leaderboard.md)

---

## Links

| | |
|:--|:--|
| :fontawesome-brands-github: [GitHub](https://github.com/spektran/spektran) | :material-database: [Dataset](https://huggingface.co/datasets/spektran/spektran-ch4-v0) |
| :fontawesome-brands-python: [PyPI](https://pypi.org/project/spektran/) | :material-weight-lifter: [Pre-trained Baselines](https://huggingface.co/spektran/spektran-baselines-v0) |
| :material-play-circle: [Interactive Demo](https://huggingface.co/spaces/spektran/spektran-demo) | :material-file-document: [CITATION.cff](https://github.com/spektran/spektran/blob/main/CITATION.cff) |
