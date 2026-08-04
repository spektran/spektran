"""TIPS total internal partition sums: accurate Q(T) for temperature scaling.

Replaces the power-law approximation in :func:`absorption.default_q_ratio`
with a 7th-order polynomial fit to the total internal partition sum Q(T),
for the principal isotopologue of each supported molecule (CH4, H2O, CO2,
CO). The target curve approximates the TIPS-2017 tables (R.R. Gamache et
al., "Total internal partition sums for the HITRAN2016 database", JQSRT 203
(2017) 70, doi:10.1016/j.jqsrt.2017.03.045):

- A rigid-rotor-harmonic-oscillator (RRHO) partition function built from
  standard published rotational constants and fundamental vibrational
  wavenumbers for each molecule (independent-mode harmonic approximation),
- normalized so Q(296 K) matches the standard HITRAN reference values
  (Gordon et al., JQSRT 277 (2022) 107949): CH4 590.48, H2O 174.58,
  CO2 286.09, CO 107.42,
- then least-squares fit (relative-error weighted) to a degree-7 polynomial
  in T over 70-3000 K.

This RRHO-anchored fit is an offline approximation to the published table
(no network / hapi access is required at runtime); it is not a literal
transcription of the TIPS-2017 numbers. It agrees with the RRHO ground
truth to < 0.4% over 200-2000 K and captures the vibrational-mode growth of
Q(T) that the old power law omits entirely (see
:func:`absorption.default_q_ratio`). Production pipelines with network
access should prefer hapi's ``partitionSum()`` where higher-order
(anharmonic, centrifugal-distortion) corrections matter.

Valid range: 70-3000 K (extrapolation outside this range is not prevented
but accuracy degrades). At T = T_ref the ratio Q(T_ref)/Q(T) is exactly 1.0
by construction, independent of polynomial rounding.

An independent reference implementation (separately fit coefficients, plain
Python evaluation instead of numpy.polyval, different fitting grid) lives in
``tests/reference_impl/ref_tips.py`` for Gate G3 cross-validation.
"""

from __future__ import annotations

import numpy as np

from .constants import T_REF_K

# 7th-order polynomial coefficients (ascending powers of T: a_0 .. a_7), so
#     Q(T) = sum_i a_i * T^i,   valid 70-3000 K
# for the principal isotopologue of each molecule. See module docstring for
# derivation. Independently re-derived (different molecular-constant
# precision, different fitting grid) in tests/reference_impl/ref_tips.py --
# the two coefficient sets are deliberately NOT identical, only numerically
# close (< 0.5% agreement on the resulting ratio, see tests/test_tips.py).
_Q_COEFFS: dict[str, np.ndarray] = {
    "CH4": np.array(
        [
            -1.8218e01,
            8.9888e-01,
            4.8630e-03,
            -5.1131e-06,
            5.5179e-09,
            3.2311e-12,
            -2.7103e-15,
            1.3156e-18,
        ]
    ),
    "H2O": np.array(
        [
            -4.4462e00,
            2.4705e-01,
            1.5538e-03,
            -1.5720e-06,
            1.5312e-09,
            -6.9319e-13,
            1.6994e-16,
            -1.6697e-20,
        ]
    ),
    "CO2": np.array(
        [
            -2.4127e00,
            9.8188e-01,
            -1.0320e-03,
            3.9991e-06,
            -2.4297e-09,
            1.6839e-12,
            -4.0420e-16,
            4.2199e-20,
        ]
    ),
    "CO": np.array(
        [
            2.4537e-01,
            3.6270e-01,
            4.2678e-06,
            -5.1328e-08,
            1.1419e-10,
            -6.5920e-14,
            1.6928e-17,
            -1.6565e-21,
        ]
    ),
}


def tips_q_total(molecule: str, temperature_K: float) -> float:
    """Total internal partition sum Q(T) for the principal isotopologue.

    Raises ``KeyError`` for molecules outside {CH4, H2O, CO2, CO}.
    """
    coeffs = _Q_COEFFS[molecule]
    return float(np.polyval(coeffs[::-1], temperature_K))


def tips_q_ratio(molecule: str, temperature_K: float, T_ref_K: float = T_REF_K) -> float:
    """Partition-sum ratio Q(T_ref)/Q(T); drop-in replacement for default_q_ratio.

    Exactly 1.0 when ``temperature_K == T_ref_K`` (short-circuits before
    touching the polynomial, so fit rounding can never perturb the
    reference-temperature identity that HITRAN-comparison tests pin on).
    """
    if temperature_K == T_ref_K:
        return 1.0
    return tips_q_total(molecule, T_ref_K) / tips_q_total(molecule, temperature_K)
