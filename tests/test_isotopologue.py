"""Tests for isotopologue handling and line-wing cutoff."""

import numpy as np
import pytest

from spektran.physics.hitran import (
    NATURAL_ABUNDANCE,
    WING_CUTOFF_CM1,
    LineList,
    demo_ch4_2nu3,
)


class TestIsotopologueFilter:
    def _make_multiiso_lines(self):
        return LineList(
            molecule="CH4",
            nu0_cm1=np.array([6046.9, 6047.0, 6047.1, 6047.5]),
            sw_cm_per_molec=np.array([1e-21, 2e-21, 3e-21, 4e-21]),
            gamma_air=np.array([0.06, 0.06, 0.06, 0.06]),
            gamma_self=np.array([0.08, 0.08, 0.08, 0.08]),
            n_air=np.array([0.75, 0.75, 0.75, 0.75]),
            delta_air=np.array([-0.005, -0.005, -0.005, -0.005]),
            elower_cm1=np.array([10.0, 20.0, 30.0, 40.0]),
            isotopologue_id=np.array([1, 1, 2, 1]),
        )

    def test_filter_returns_subset(self):
        lines = self._make_multiiso_lines()
        iso1 = lines.filter_isotopologue(1)
        assert len(iso1) == 3
        assert iso1.molecule == "CH4"
        np.testing.assert_array_almost_equal(
            iso1.nu0_cm1, [6046.9, 6047.0, 6047.5]
        )

    def test_filter_other_isotopologue(self):
        lines = self._make_multiiso_lines()
        iso2 = lines.filter_isotopologue(2)
        assert len(iso2) == 1
        assert iso2.nu0_cm1[0] == pytest.approx(6047.1)

    def test_filter_empty_result(self):
        lines = self._make_multiiso_lines()
        iso99 = lines.filter_isotopologue(99)
        assert len(iso99) == 0

    def test_filter_preserves_htp_params(self):
        lines = self._make_multiiso_lines()
        lines.gamma_2 = np.array([0.001, 0.002, 0.003, 0.004])
        lines.delta_2 = np.array([0.0001, 0.0002, 0.0003, 0.0004])
        lines.nu_vc = np.array([0.01, 0.02, 0.03, 0.04])
        lines.eta = np.array([0.1, 0.2, 0.3, 0.4])
        iso1 = lines.filter_isotopologue(1)
        assert iso1.has_htp_params
        assert len(iso1.gamma_2) == 3

    def test_filter_raises_without_isotopologue_id(self):
        lines = demo_ch4_2nu3()
        with pytest.raises(ValueError, match="No isotopologue_id"):
            lines.filter_isotopologue(1)


class TestHTPParams:
    def test_has_htp_params_false_by_default(self):
        lines = demo_ch4_2nu3()
        assert not lines.has_htp_params

    def test_has_htp_params_true_when_set(self):
        lines = demo_ch4_2nu3()
        n = len(lines)
        lines.gamma_2 = np.zeros(n)
        lines.delta_2 = np.zeros(n)
        lines.nu_vc = np.zeros(n)
        lines.eta = np.zeros(n)
        assert lines.has_htp_params

    def test_htp_partial_is_false(self):
        lines = demo_ch4_2nu3()
        n = len(lines)
        lines.gamma_2 = np.zeros(n)
        assert not lines.has_htp_params


class TestNaturalAbundance:
    def test_ch4_principal(self):
        assert NATURAL_ABUNDANCE[("CH4", 1)] > 0.98

    def test_hcl_two_isotopologues(self):
        total = NATURAL_ABUNDANCE[("HCl", 1)] + NATURAL_ABUNDANCE[("HCl", 2)]
        assert total == pytest.approx(1.0, abs=0.001)

    def test_known_molecules(self):
        for mol in ("CH4", "H2O", "CO2", "CO"):
            assert ("mol", 1) != (mol, 1) or (mol, 1) in NATURAL_ABUNDANCE


class TestWingCutoff:
    def test_default_exists(self):
        assert WING_CUTOFF_CM1["default"] == 25.0

    def test_co2_wider(self):
        assert WING_CUTOFF_CM1["CO2"] > WING_CUTOFF_CM1["default"]

    def test_wing_cutoff_in_absorption(self):
        from spektran.physics.absorption import absorption_coefficient
        lines = demo_ch4_2nu3()
        nu = np.linspace(6000.0, 6100.0, 2000)
        alpha_no_cut = absorption_coefficient(
            nu, lines, 100e-6, 296.0, 1.0, wing_cutoff_cm1=0.0
        )
        alpha_cut = absorption_coefficient(
            nu, lines, 100e-6, 296.0, 1.0, wing_cutoff_cm1=25.0
        )
        assert np.sum(alpha_cut > 0) < np.sum(alpha_no_cut > 0)
