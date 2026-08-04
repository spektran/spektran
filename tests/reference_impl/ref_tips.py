"""REFERENCE implementation of the TIPS partition-function polynomial — Gate G3.

Independent 7th-order polynomial fit to the total internal partition sum
Q(T), approximating the TIPS-2017 tables (R.R. Gamache et al., JQSRT 203
(2017) 70, doi:10.1016/j.jqsrt.2017.03.045) for the principal isotopologue of
each supported molecule.

Transcribed and fit independently from ``src/spektran/physics/tips.py``:
different molecular-constant precision and a different temperature sampling
grid went into the least-squares fit, and evaluation here uses a plain
Python sum over powers of T rather than the main implementation's
``numpy.polyval``. No code or numeric constants are shared with the main
implementation or imported from ``spektran``.

MUST NOT import from or share code with ``spektran`` (see plan §9, G3).
"""

from __future__ import annotations

_T_REF = 296.0

# 7th-order polynomial coefficients (ascending powers of T: a_0 .. a_7), so
#     Q(T) = sum_i a_i * T^i,   valid 70-3000 K
# for the principal isotopologue of each molecule. Independently transcribed
# from a separate least-squares fit (own rounding, own grid) -- these values
# are deliberately NOT identical to spektran.physics.tips._Q_COEFFS, though
# both approximate the same underlying TIPS-2017 Q(T) curves and must agree
# to within 0.5% (see tests/test_tips.py).
POLY_COEFFS: dict[str, list[float]] = {
    "CH4": [
        -1.8010e01,
        8.9523e-01,
        4.8839e-03,
        -5.1663e-06,
        5.5850e-09,
        3.1921e-12,
        -2.7000e-15,
        1.3153e-18,
    ],
    "H2O": [
        -4.4154e00,
        2.4657e-01,
        1.5561e-03,
        -1.5770e-06,
        1.5366e-09,
        -6.9606e-13,
        1.7070e-16,
        -1.6775e-20,
    ],
    "CO2": [
        -2.4247e00,
        9.8206e-01,
        -1.0331e-03,
        4.0016e-06,
        -2.4315e-09,
        1.6849e-12,
        -4.0442e-16,
        4.2217e-20,
    ],
    "CO": [
        2.4776e-01,
        3.6267e-01,
        4.3804e-06,
        -5.1518e-08,
        1.1435e-10,
        -6.5991e-14,
        1.6943e-17,
        -1.6578e-21,
    ],
    "NH3": [
        -5.4997e01,
        2.6903e00,
        1.3559e-02,
        -1.3579e-05,
        2.1854e-08,
        -9.6344e-12,
        3.2801e-15,
        -2.7334e-19,
    ],
    "NO": [
        -2.3852e00,
        3.9054e00,
        -1.6054e-04,
        -1.4174e-07,
        1.0457e-09,
        -6.8718e-13,
        1.8832e-16,
        -1.9250e-20,
    ],
    "NO2": [
        -4.7720e02,
        2.2296e01,
        9.0743e-02,
        -6.0988e-05,
        1.5949e-07,
        -6.6432e-11,
        1.9498e-14,
        -2.1269e-18,
    ],
    "SO2": [
        -1.8809e02,
        9.5985e00,
        3.6502e-02,
        1.5907e-06,
        6.2697e-08,
        -1.6421e-11,
        5.5856e-15,
        -6.2765e-19,
    ],
    "HCl": [
        6.8260e-01,
        1.6972e00,
        1.1194e-04,
        -3.2151e-07,
        3.9159e-10,
        -1.7866e-13,
        3.8833e-17,
        -3.3312e-21,
    ],
    "HF": [
        2.5465e-02,
        1.3955e-01,
        3.1233e-06,
        -6.7049e-09,
        5.0282e-12,
        -2.9382e-16,
        -3.3289e-19,
        5.8772e-23,
    ],
}


def ref_q_total(molecule: str, temperature_K: float) -> float:
    """Reference total internal partition sum Q(T) via polynomial evaluation.

    Raises ``KeyError`` for molecules outside {CH4, H2O, CO2, CO, NH3, NO,
    NO2, SO2, HCl, HF}.
    """
    coeffs = POLY_COEFFS[molecule]
    return sum(c * temperature_K**i for i, c in enumerate(coeffs))


def ref_q_ratio(molecule: str, temperature_K: float, T_ref_K: float = _T_REF) -> float:
    """Reference Q(T_ref)/Q(T), for cross-validation against ``tips_q_ratio``."""
    if temperature_K == T_ref_K:
        return 1.0
    return ref_q_total(molecule, T_ref_K) / ref_q_total(molecule, temperature_K)
