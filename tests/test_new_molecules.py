"""Tests for NH3, NO, NO2, SO2, HCl, HF physics."""
import numpy as np
import pytest

from spektran.physics.absorption import absorption_coefficient
from spektran.physics.hitran import (
    demo_hcl,
    demo_hf,
    demo_nh3,
    demo_no,
    demo_no2,
    demo_so2,
)
from spektran.physics.tips import tips_q_ratio, tips_q_total

NEW_MOLECULES = ["NH3", "NO", "NO2", "SO2", "HCl", "HF"]
DEMO_FNS = {
    "NH3": demo_nh3, "NO": demo_no, "NO2": demo_no2,
    "SO2": demo_so2, "HCl": demo_hcl, "HF": demo_hf,
}

# HITRAN reference Q(296K) values (tolerance +-20% for RRHO approximation)
Q_296_REF = {
    "NH3": 1725.22, "NO": 1142.47, "NO2": 13577.33,
    "SO2": 6340.07, "HCl": 507.14, "HF": 41.47,
}


@pytest.mark.parametrize("mol", NEW_MOLECULES)
def test_demo_lines_valid(mol):
    ll = DEMO_FNS[mol]()
    assert len(ll) >= 2
    assert ll.molecule == mol
    assert ll.molar_mass_amu > 0


@pytest.mark.parametrize("mol", NEW_MOLECULES)
def test_tips_q_ratio_ref_temp(mol):
    assert tips_q_ratio(mol, 296.0) == 1.0


@pytest.mark.parametrize("mol", NEW_MOLECULES)
def test_tips_q_total_near_hitran(mol):
    q = tips_q_total(mol, 296.0)
    ref = Q_296_REF[mol]
    rel_err = abs(q - ref) / ref
    assert rel_err < 0.20, f"{mol}: Q(296)={q:.2f}, ref={ref:.2f}, err={rel_err:.1%}"


@pytest.mark.parametrize("mol", NEW_MOLECULES)
def test_absorption_nonzero(mol):
    ll = DEMO_FNS[mol]()
    nu = np.linspace(ll.nu0_cm1.min() - 1, ll.nu0_cm1.max() + 1, 500)
    alpha = absorption_coefficient(nu, ll, 100e-6, 296.0, 1.0)
    assert np.max(alpha) > 0


@pytest.mark.parametrize("mol", NEW_MOLECULES)
def test_tips_cross_validation(mol):
    """Main and reference TIPS implementations agree within 0.5%."""
    from tests.reference_impl.ref_tips import ref_q_ratio
    for T in [200.0, 296.0, 500.0, 1000.0, 2000.0]:
        main = tips_q_ratio(mol, T)
        ref = ref_q_ratio(mol, T)
        if T == 296.0:
            assert main == ref == 1.0
        else:
            rel = abs(main - ref) / max(abs(ref), 1e-12)
            assert rel < 0.005, f"{mol} T={T}: main={main:.6f} ref={ref:.6f} err={rel:.4f}"
