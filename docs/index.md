# SPEKTRAN

Open-source **simulation engine**, **data standard**, and **ML benchmark
suite** for optical gas sensing — currently shipping two modalities:
**TDLAS** (tunable diode laser absorption spectroscopy) and **NDIR**
(non-dispersive infrared).

## What's inside

- **10 target molecules**: CH4, H2O, CO2, CO, NH3, NO, NO2, SO2, HCl, HF
  with HITRAN line-by-line physics and TIPS partition-function polynomials
- **2 modalities**: TDLAS (direct absorption + wavelength modulation) and
  NDIR (Planck source + bandpass filter)
- **Advanced line shapes**: Voigt profile and Hartmann-Tran Profile (HTP)
  with speed-dependent broadening/shifting and Dicke narrowing
- **WMS chain**: 1f–4f lock-in demodulation, 2f/1f calibration-free ratio,
  etalon fringes in the time-domain chain
- **Realistic instrument noise**: laser scan nonlinearity, thermal chirp,
  RAM, etalon fringes, 1/f noise, baseline drift, window contamination,
  beam wander, laser RIN, TIA bandwidth, temperature-dependent detector noise,
  ADC quantization — 14+ virtual instrument configurations
- **9 benchmark tasks** (T1–T9): concentration regression, spectral denoising,
  cross-instrument generalization, WMS concentration, drift compensation,
  OOD instrument detection, cross-modality transfer, multi-species regression,
  temperature regression
- **12+ baseline models**: Ridge, 1D CNN, Patchified Transformer, 1D U-Net,
  TCN, plus classical baselines per task
- **Full reproducibility**: every record carries generator version, seed, and
  every sampled noise parameter; same config = bit-identical data
- **Isotopologue handling**: per-line isotopologue ID, natural abundance
  lookup, and configurable line-wing cutoff

Licenses: code Apache-2.0, data & schema CC BY 4.0.

## Try it now

- [Interactive demo](https://huggingface.co/spaces/spektran/spektran-demo) — no installation, runs in your browser
- [Quickstart guide](quickstart.md) — install and simulate your first spectrum in 5 minutes
- [Leaderboard](leaderboard.md) — current benchmark standings
