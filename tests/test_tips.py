"""Tests for the TIPS partition-function polynomial (Gate G3 territory).

Cross-validates ``spektran.physics.tips`` against the independent
``tests/reference_impl/ref_tips.py`` implementation (separately fit
coefficients, separate evaluation code path).
"""

import numpy as np
import pytest

_MOLECULES = ("CH4", "H2O", "CO2", "CO")


def test_tips_q_ratio_at_296K_is_unity():
    """Q(296)/Q(296) must be exactly 1.0 for all supported molecules."""
    from spektran.physics.tips import tips_q_ratio

    for mol in _MOLECULES:
        assert tips_q_ratio(mol, 296.0) == pytest.approx(1.0, abs=1e-10)


def test_tips_q_ratio_monotonic_for_nonlinear():
    """For nonlinear molecules (CH4, H2O), Q grows with T, so the ratio

    Q(T_ref)/Q(T) is < 1 above the reference temperature and > 1 below it.
    """
    from spektran.physics.tips import tips_q_ratio

    for mol in ("CH4", "H2O"):
        assert tips_q_ratio(mol, 500.0) < 1.0
        assert tips_q_ratio(mol, 200.0) > 1.0


def test_tips_cross_validation_vs_reference():
    """Main and reference implementations agree to < 0.5% over 200-2000 K."""
    from spektran.physics.tips import tips_q_ratio
    from tests.reference_impl.ref_tips import ref_q_ratio

    temps = np.linspace(200.0, 2000.0, 50)
    for mol in _MOLECULES:
        for t in temps:
            main_val = tips_q_ratio(mol, float(t))
            ref_val = ref_q_ratio(mol, float(t))
            rel = abs(main_val - ref_val) / max(abs(ref_val), 1e-30)
            assert rel < 0.005, f"{mol} at {t} K: main={main_val}, ref={ref_val}, rel={rel}"


def test_tips_replaces_power_law_accurately():
    """TIPS must be more accurate than the old power-law at non-reference temps."""
    from spektran.physics.absorption import default_q_ratio
    from spektran.physics.tips import tips_q_ratio
    from tests.reference_impl.ref_tips import ref_q_ratio

    for mol in ("CH4", "CO2"):
        for t in (400.0, 600.0, 1000.0):
            ref = ref_q_ratio(mol, t)
            tips = tips_q_ratio(mol, t)
            power = default_q_ratio(mol, t)
            assert abs(tips - ref) < abs(power - ref), (
                f"{mol}@{t}K: TIPS err={abs(tips - ref):.6f}, "
                f"power-law err={abs(power - ref):.6f}"
            )


def test_tips_unknown_molecule_raises():
    from spektran.physics.tips import tips_q_ratio

    with pytest.raises(KeyError):
        tips_q_ratio("XeF6", 300.0)
