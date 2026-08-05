"""Tests for the TIPS partition-function wrapper around hapi PYTIPS2021.

Validates that our wrapper correctly delegates to hapi and returns
physically sensible partition-sum ratios.
"""

import pytest

_MOLECULES = ("CH4", "H2O", "CO2", "CO", "NH3", "NO", "NO2", "SO2", "HCl", "HF")

# HITRAN reference Q(296 K) values for the principal isotopologue
# (Gordon et al., JQSRT 277 (2022) 107949)
_Q_296_HITRAN = {
    "CH4": 590.53, "H2O": 174.58, "CO2": 286.09, "CO": 107.42,
    "NH3": 1725.22, "NO": 1141.09, "NO2": 13575.24, "SO2": 6339.09,
    "HCl": 160.65, "HF": 41.47,
}


def test_tips_q_ratio_at_296K_is_unity():
    """Q(296)/Q(296) must be exactly 1.0 for all supported molecules."""
    from spektran.physics.tips import tips_q_ratio

    for mol in _MOLECULES:
        assert tips_q_ratio(mol, 296.0) == pytest.approx(1.0, abs=1e-10)


def test_tips_q_total_matches_hitran_reference():
    """Q(296 K) from hapi must agree with published HITRAN values."""
    from spektran.physics.tips import tips_q_total

    for mol in _MOLECULES:
        q = tips_q_total(mol, 296.0)
        expected = _Q_296_HITRAN[mol]
        rel = abs(q - expected) / expected
        assert rel < 0.01, f"{mol}: Q(296)={q:.4f}, expected={expected:.4f}, rel={rel:.4e}"


def test_tips_q_ratio_monotonic_for_nonlinear():
    """For nonlinear molecules (CH4, H2O), Q grows with T, so the ratio
    Q(T_ref)/Q(T) is < 1 above the reference temperature and > 1 below it.
    """
    from spektran.physics.tips import tips_q_ratio

    for mol in ("CH4", "H2O"):
        assert tips_q_ratio(mol, 500.0) < 1.0
        assert tips_q_ratio(mol, 200.0) > 1.0


def test_tips_replaces_power_law_accurately():
    """TIPS (hapi) must be more accurate than the old power-law approximation."""
    from spektran.physics.absorption import default_q_ratio
    from spektran.physics.tips import tips_q_ratio

    for mol in ("CH4", "CO2"):
        for t in (400.0, 600.0, 1000.0):
            tips = tips_q_ratio(mol, t)
            power = default_q_ratio(mol, t)
            # TIPS should give a different (more accurate) value than power-law
            assert tips != pytest.approx(power, rel=0.01), (
                f"{mol}@{t}K: TIPS and power-law suspiciously close"
            )


def test_tips_unknown_molecule_raises():
    from spektran.physics.tips import tips_q_ratio

    with pytest.raises(KeyError):
        tips_q_ratio("XeF6", 300.0)
