"""TIPS total internal partition sums: accurate Q(T) for temperature scaling.

Thin wrapper around hapi's ``PYTIPS2021`` — the official HITRAN partition-sum
tables (R.R. Gamache et al., "Total internal partition sums for the
HITRAN2020 database", JQSRT 271 (2021) 107713,
doi:10.1016/j.jqsrt.2021.107713).

Supports the principal isotopologue of each of the 10 molecules shipped
with SPEKTRAN: CH4, H2O, CO2, CO, NH3, NO, NO2, SO2, HCl, HF.

Valid range: 1-5000 K (hapi TIPS-2021 tabulation range).
"""

from __future__ import annotations

from hapi import PYTIPS2021

from .constants import T_REF_K

# HITRAN molecule number → principal isotopologue (always 1)
_MOL_TO_HITRAN: dict[str, int] = {
    "CH4": 6, "H2O": 1, "CO2": 2, "CO": 5, "NH3": 11,
    "NO": 8, "NO2": 10, "SO2": 9, "HCl": 15, "HF": 14,
}


def tips_q_total(molecule: str, temperature_K: float) -> float:
    """Total internal partition sum Q(T) for the principal isotopologue.

    Delegates to hapi's ``PYTIPS2021``.

    Raises ``KeyError`` for molecules outside {CH4, H2O, CO2, CO, NH3, NO,
    NO2, SO2, HCl, HF}.
    """
    mol_id = _MOL_TO_HITRAN[molecule]
    return float(PYTIPS2021(mol_id, 1, temperature_K))


def tips_q_ratio(molecule: str, temperature_K: float, T_ref_K: float = T_REF_K) -> float:
    """Partition-sum ratio Q(T_ref)/Q(T); drop-in replacement for default_q_ratio.

    Exactly 1.0 when ``temperature_K == T_ref_K`` (short-circuits to avoid
    numerical noise at the reference temperature).
    """
    if temperature_K == T_ref_K:
        return 1.0
    return tips_q_total(molecule, T_ref_K) / tips_q_total(molecule, temperature_K)
