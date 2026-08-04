# SPEKTRAN

Open-source **simulation engine**, **data standard**, and **ML benchmark
suite** for tunable diode laser absorption spectroscopy (TDLAS).

- Physically rigorous forward model (HITRAN line-by-line, Voigt, direct
  absorption + wavelength modulation with lock-in demodulation)
- Multi-species support: CH4, H2O, CO2, CO with Beer-Lambert superposition
- Higher-harmonic WMS: 1f through 4f demodulation
- TIPS partition-function polynomial for accurate temperature scaling
- A modular instrument-noise chain: laser scan nonlinearity, RAM, etalon
  fringes, 1/f noise, baseline drift, ADC quantization
- Fully reproducible: every record carries generator version, seed, and every
  sampled noise parameter; same config = bit-identical data
- 6 standardized tasks: concentration regression, spectral denoising,
  cross-instrument generalization, WMS concentration, drift compensation,
  and OOD instrument detection

Licenses: code Apache-2.0, data & schema CC BY 4.0.

Start with the [Quickstart](quickstart.md).
