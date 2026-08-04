# SPEKTRAN

Open-source **simulation engine**, **data standard**, and **ML benchmark
suite** for tunable diode laser absorption spectroscopy (TDLAS).

- Physically rigorous forward model (HITRAN line-by-line, Voigt, direct
  absorption + wavelength modulation with lock-in demodulation)
- A modular instrument-noise chain: laser scan nonlinearity, RAM, etalon
  fringes, 1/f noise, baseline drift, ADC quantization
- Fully reproducible: every record carries generator version, seed, and every
  sampled noise parameter; same config = bit-identical data
- Standardized tasks: concentration regression, spectral denoising, and the
  flagship **cross-instrument generalization** track

Licenses: code Apache-2.0, data & schema CC BY 4.0.

Start with the [Quickstart](quickstart.md).
