"""Test that HITRAN production line data loads and has expected properties."""
import pytest


@pytest.mark.hitran_online
def test_ch4_hitran_line_count():
    from spektran.physics.hitran import fetch_lines
    ll = fetch_lines("CH4", 6045.0, 6049.0)
    assert len(ll) >= 10, f"Expected >=10 CH4 lines in 6045-6049, got {len(ll)}"
    assert ll.molecule == "CH4"


@pytest.mark.hitran_online
def test_h2o_hitran_line_count():
    from spektran.physics.hitran import fetch_lines
    ll = fetch_lines("H2O", 7183.0, 7192.0)
    assert len(ll) >= 5, f"Expected >=5 H2O lines in 7183-7192, got {len(ll)}"


@pytest.mark.hitran_online
def test_co2_hitran_line_count():
    from spektran.physics.hitran import fetch_lines
    ll = fetch_lines("CO2", 4976.0, 4980.0)
    assert len(ll) >= 3, f"Expected >=3 CO2 lines in 4976-4980, got {len(ll)}"


@pytest.mark.hitran_online
def test_co_hitran_line_count():
    from spektran.physics.hitran import fetch_lines
    ll = fetch_lines("CO", 2168.0, 2174.0)
    assert len(ll) >= 2, f"Expected >=2 CO lines in 2168-2174, got {len(ll)}"


@pytest.mark.hitran_online
def test_hitran_lines_have_positive_intensities():
    from spektran.physics.hitran import fetch_lines
    ll = fetch_lines("CH4", 6045.0, 6049.0)
    assert all(s > 0 for s in ll.sw_cm_per_molec)
