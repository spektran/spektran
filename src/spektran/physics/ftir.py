"""FTIR forward model: interferogram generation and spectral retrieval.

Fourier Transform Infrared Spectroscopy records an interferogram — the
modulated signal from a Michelson interferometer — then recovers the
spectrum via FFT.  The instrument line shape (ILS) is determined by
the maximum optical path difference (OPD) and the apodization function.

Forward model chain::

    HITRAN line data
      -> High-resolution absorption spectrum alpha(nu)
        -> Inverse FFT -> ideal interferogram
          -> Truncate at max OPD -> introduces ILS (sinc)
            -> Apodize -> modified ILS sidelobes
              -> Add noise (detector, phase, source)
                -> FFT -> recovered spectrum with ILS broadening

For ML benchmarks, the input signal is the recovered FTIR spectrum
(after FFT), not the raw interferogram, since this is what real FTIR
users work with for quantitative analysis.

References:
- P.R. Griffiths and J.A. de Haseth, "Fourier Transform Infrared
  Spectrometry", 2nd ed., Wiley (2007), doi:10.1002/047010631X
- S.P. Davis et al., "Fourier Transform Spectrometry", Academic Press
  (2001), doi:10.1016/B978-0-12-042510-5.X5000-2
- D. Wunch et al., "The Total Carbon Column Observing Network",
  Phil. Trans. R. Soc. A 369 (2011) 2087,
  doi:10.1098/rsta.2010.0240
"""

from __future__ import annotations

import numpy as np

from .absorption import absorption_coefficient
from .constants import C_CM_PER_S
from .hitran import LineList


def apodization_function(
    opd_cm: np.ndarray,
    max_opd_cm: float,
    function: str = "norton_beer_medium",
) -> np.ndarray:
    """Compute apodization window for interferogram truncation.

    Norton & Beer, JOSA 66 (1976) 259, doi:10.1364/JOSA.66.000259

    Supported functions:
    - ``"boxcar"``: no apodization (1 everywhere within max OPD)
    - ``"triangular"``: linear taper to zero at max OPD
    - ``"happ_genzel"``: 0.54 + 0.46*cos(pi*x/L)
    - ``"norton_beer_medium"``: NB medium, good sidelobe suppression
    - ``"norton_beer_strong"``: NB strong, maximum sidelobe suppression
    """
    x = np.abs(opd_cm) / max_opd_cm
    x = np.clip(x, 0.0, 1.0)

    if function == "boxcar":
        return np.ones_like(x)
    if function == "triangular":
        return 1.0 - x
    if function == "happ_genzel":
        return 0.54 + 0.46 * np.cos(np.pi * x)
    if function == "norton_beer_medium":
        return 0.348093 - 0.087221 * (1 - x**2) + 0.703128 * (1 - x**2) ** 2
    if function == "norton_beer_strong":
        return (
            0.045335
            + 0.0 * (1 - x**2)
            + 0.554883 * (1 - x**2) ** 2
            + 0.399782 * (1 - x**2) ** 3
        )
    raise ValueError(f"Unknown apodization: {function!r}")


def spectral_resolution_cm1(max_opd_cm: float) -> float:
    """Spectral resolution delta_nu = 1 / (2 * max_OPD).

    Griffiths & de Haseth (2007) eq. 2.6.
    """
    return 1.0 / (2.0 * max_opd_cm)


def generate_interferogram(
    nu_cm1: np.ndarray,
    spectrum: np.ndarray,
    max_opd_cm: float,
    n_opd_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate interferogram from a spectrum via inverse DFT.

    The interferogram is the cosine transform of the spectrum:
    I(delta) = integral S(nu) * cos(2*pi*nu*delta) dnu

    Returns (opd_cm, interferogram) arrays.
    """
    dnu = nu_cm1[1] - nu_cm1[0]
    opd_max = 1.0 / (2.0 * dnu) if max_opd_cm == 0 else max_opd_cm
    opd = np.linspace(-opd_max, opd_max, n_opd_points)

    igram = np.zeros(n_opd_points)
    for i, delta in enumerate(opd):
        igram[i] = np.sum(spectrum * np.cos(2.0 * np.pi * nu_cm1 * delta)) * dnu

    return opd, igram


def interferogram_to_spectrum(
    opd_cm: np.ndarray,
    interferogram: np.ndarray,
    nu_out_cm1: np.ndarray | None = None,
    apod_function: str = "norton_beer_medium",
) -> tuple[np.ndarray, np.ndarray]:
    """Recover spectrum from interferogram via cosine transform.

    Applies apodization before the transform. Returns (nu_cm1, spectrum).
    """
    max_opd = np.max(np.abs(opd_cm))
    apod = apodization_function(opd_cm, max_opd, apod_function)
    apodized = interferogram * apod

    d_opd = opd_cm[1] - opd_cm[0]
    n = len(opd_cm)

    fft_result = np.fft.rfft(apodized)
    freqs = np.fft.rfftfreq(n, d=np.abs(d_opd))

    spectrum = np.abs(fft_result) * np.abs(d_opd) * 2.0

    if nu_out_cm1 is not None:
        spectrum = np.interp(nu_out_cm1, freqs, spectrum)
        return nu_out_cm1, spectrum
    return freqs, spectrum


def simulate_ftir_spectrum(
    lines: LineList,
    molecule: str,
    concentration_ppm: float,
    temperature_K: float,
    pressure_atm: float,
    path_length_m: float,
    max_opd_cm: float = 45.0,
    wavenumber_start_cm1: float = 6000.0,
    wavenumber_end_cm1: float = 6100.0,
    n_hires_points: int = 10000,
    n_output_points: int = 500,
    apod_function: str = "norton_beer_medium",
    zero_fill_factor: int = 2,
    interferents: list[dict] | None = None,
) -> dict:
    """Simulate a clean FTIR absorption spectrum.

    Full forward chain: HITRAN -> high-res spectrum -> interferogram ->
    apodize -> FFT -> ILS-broadened recovered spectrum.

    Returns dict with keys: ``nu_cm1``, ``spectrum``, ``spectrum_hires``,
    ``transmittance``, ``absorbance``, ``resolution_cm1``,
    ``concentration_ppm``.
    """
    nu_hires = np.linspace(wavenumber_start_cm1, wavenumber_end_cm1, n_hires_points)

    mole_fraction = concentration_ppm * 1e-6
    alpha = absorption_coefficient(
        nu_hires, lines, mole_fraction, temperature_K, pressure_atm,
    )

    if interferents:
        for interf in interferents:
            alpha_i = absorption_coefficient(
                nu_hires,
                interf["lines"],
                interf["concentration_ppm"] * 1e-6,
                temperature_K,
                pressure_atm,
            )
            alpha = alpha + alpha_i

    path_cm = path_length_m * 100.0
    absorbance_hires = alpha * path_cm
    transmittance_hires = np.exp(-absorbance_hires)

    source = np.ones_like(nu_hires)
    signal_hires = source * transmittance_hires

    n_igram = n_output_points * zero_fill_factor * 2
    opd, igram = generate_interferogram(
        nu_hires, signal_hires, max_opd_cm, n_opd_points=n_igram,
    )

    nu_out = np.linspace(wavenumber_start_cm1, wavenumber_end_cm1, n_output_points)
    _, spectrum_recovered = interferogram_to_spectrum(
        opd, igram, nu_out_cm1=nu_out, apod_function=apod_function,
    )

    spectrum_recovered = np.maximum(spectrum_recovered, 1e-15)
    recovered_absorbance = -np.log(spectrum_recovered / np.max(spectrum_recovered))

    resolution = spectral_resolution_cm1(max_opd_cm)

    return {
        "nu_cm1": nu_out,
        "spectrum": spectrum_recovered,
        "spectrum_hires": signal_hires,
        "nu_hires_cm1": nu_hires,
        "transmittance": spectrum_recovered / np.max(spectrum_recovered),
        "absorbance": recovered_absorbance,
        "resolution_cm1": resolution,
        "max_opd_cm": max_opd_cm,
        "apodization": apod_function,
        "concentration_ppm": concentration_ppm,
    }
