"""Physics-correctness tests for the Hartmann-Tran Profile (HTP).

The main implementation wraps hapi's ``pcqsdhc``.  Cross-validation (Gate G3)
compares against the independent 2D speed-quadrature reference for the SDV
sub-case.  Tests use physically realistic parameter ratios
(gamma_2/gamma_0 ~ 0.05-0.15, typical of molecular spectroscopy).
"""

from __future__ import annotations

import numpy as np
import pytest

from spektran.physics import voigt_profile
from spektran.physics.lineshape_htp import htp_profile
from tests.reference_impl.ref_lineshape_htp import ref_sdv_quadrature


def _wide_grid(nu0: float, width: float, half_span: float = 2000.0, n: int = 4_000_001):
    return np.linspace(nu0 - half_span * width, nu0 + half_span * width, n)


# ---------------------------------------------------------------------------
# 1. Voigt degenerate limit
# ---------------------------------------------------------------------------

class TestVoigtDegenerateLimit:
    """HTP(gamma_2=delta_2=nu_vc=eta=0) must match voigt_profile."""

    @pytest.mark.parametrize(
        "doppler_hwhm, gamma_0",
        [
            (0.01, 0.05),
            (0.005, 0.002),
            (0.02, 0.10),
            (0.008, 0.008),
        ],
    )
    def test_matches_voigt(self, doppler_hwhm: float, gamma_0: float):
        nu0 = 6047.0
        nu = np.linspace(nu0 - 0.5, nu0 + 0.5, 2001)

        htp = htp_profile(nu, nu0, doppler_hwhm, gamma_0, delta_0=0.0)
        voigt = voigt_profile(nu, nu0, doppler_hwhm, gamma_0)

        # hapi uses Humlicek CPF vs scipy's Poppe-Wijers wofz — 1e-4 agreement
        np.testing.assert_allclose(htp, voigt, rtol=2e-4)

    def test_with_pressure_shift(self):
        """With nonzero delta_0, HTP equals a Voigt shifted by delta_0."""
        nu0 = 6047.0
        delta_0 = -0.008
        doppler_hwhm = 0.01
        gamma_0 = 0.05
        nu = np.linspace(nu0 - 0.5, nu0 + 0.5, 2001)

        htp = htp_profile(nu, nu0, doppler_hwhm, gamma_0, delta_0=delta_0)
        voigt_shifted = voigt_profile(nu, nu0 + delta_0, doppler_hwhm, gamma_0)

        np.testing.assert_allclose(htp, voigt_shifted, rtol=2e-4)


# ---------------------------------------------------------------------------
# 2. Area normalization (physically realistic parameters only)
# ---------------------------------------------------------------------------

class TestAreaNormalization:
    """Numerical integral of HTP must equal 1.0 for physical parameter ratios."""

    @pytest.mark.parametrize(
        "gamma_2, delta_2, nu_vc, eta",
        [
            (0.0, 0.0, 0.0, 0.0),
            (0.005, 0.002, 0.0, 0.0),
            (0.0, 0.0, 0.01, 0.0),
            (0.005, 0.002, 0.01, 0.0),
            (0.005, 0.002, 0.01, 0.1),
        ],
    )
    def test_area_is_one(self, gamma_2, delta_2, nu_vc, eta):
        nu0 = 6047.0
        doppler_hwhm = 0.01
        gamma_0 = 0.05
        delta_0 = -0.005
        width = max(doppler_hwhm, gamma_0, 0.01)
        nu = _wide_grid(nu0, width)

        phi = htp_profile(
            nu, nu0, doppler_hwhm, gamma_0, delta_0,
            gamma_2=gamma_2, delta_2=delta_2, nu_vc=nu_vc, eta=eta,
        )
        area = np.trapezoid(phi, nu)

        assert abs(area - 1.0) < 1e-3, f"area = {area}"


# ---------------------------------------------------------------------------
# 3. G3 cross-validation: hapi vs independent speed quadrature (SDV)
# ---------------------------------------------------------------------------

class TestG3CrossValidation:
    """Gate G3: hapi pcqsdhc vs 2D speed-quadrature reference (SDV case)."""

    @pytest.mark.parametrize(
        "gamma_2, delta_2",
        [
            (0.005, 0.002),
            (0.008, 0.003),
            (0.003, 0.0),
            (0.0, 0.003),
        ],
    )
    def test_sdv_vs_speed_quadrature(self, gamma_2, delta_2):
        """SDV case: hapi wofz-based vs 2D speed-quadrature reference."""
        nu0 = 6047.0
        doppler_hwhm = 0.01
        gamma_0 = 0.05
        delta_0 = -0.005

        offsets = np.array([0.0, 0.005, 0.02, 0.05, 0.15, -0.01, -0.05])
        nu_points = nu0 + offsets

        main = htp_profile(
            nu_points, nu0, doppler_hwhm, gamma_0, delta_0,
            gamma_2=gamma_2, delta_2=delta_2, nu_vc=0.0, eta=0.0,
        )

        max_rel = 0.0
        for i, nu_i in enumerate(nu_points):
            ref = ref_sdv_quadrature(
                nu_i, nu0, doppler_hwhm, gamma_0, delta_0,
                gamma_2=gamma_2, delta_2=delta_2,
                n_hermite=120, n_laguerre=100,
            )
            if ref > 0:
                rel = abs(main[i] - ref) / ref
                max_rel = max(max_rel, rel)

        assert max_rel < 1e-4, (
            f"SDV speed-quadrature max relative deviation {max_rel:.2e}"
        )


# ---------------------------------------------------------------------------
# 4. Speed-dependent broadening: narrowing effect
# ---------------------------------------------------------------------------

class TestSpeedDependentNarrowing:
    """With gamma_2 > 0, the profile should be narrower than the Voigt."""

    def test_fwhm_narrower_than_voigt(self):
        nu0 = 6047.0
        doppler_hwhm = 0.01
        gamma_0 = 0.05
        delta_0 = 0.0
        gamma_2 = 0.008
        nu = np.linspace(nu0 - 0.3, nu0 + 0.3, 10001)

        voigt = voigt_profile(nu, nu0, doppler_hwhm, gamma_0)
        sdv = htp_profile(
            nu, nu0, doppler_hwhm, gamma_0, delta_0,
            gamma_2=gamma_2,
        )

        voigt_half = voigt.max() / 2.0
        sdv_half = sdv.max() / 2.0

        voigt_fwhm = nu[voigt > voigt_half][-1] - nu[voigt > voigt_half][0]
        sdv_fwhm = nu[sdv > sdv_half][-1] - nu[sdv > sdv_half][0]

        assert sdv_fwhm < voigt_fwhm, (
            f"SDV FWHM {sdv_fwhm:.6f} not narrower than Voigt FWHM {voigt_fwhm:.6f}"
        )

    def test_peak_higher_than_voigt(self):
        """Narrower profile with unit area must have a higher peak."""
        nu0 = 6047.0
        doppler_hwhm = 0.01
        gamma_0 = 0.05
        nu = np.linspace(nu0 - 0.3, nu0 + 0.3, 10001)

        voigt = voigt_profile(nu, nu0, doppler_hwhm, gamma_0)
        sdv = htp_profile(
            nu, nu0, doppler_hwhm, gamma_0, delta_0=0.0,
            gamma_2=0.008,
        )

        assert sdv.max() > voigt.max()


# ---------------------------------------------------------------------------
# 5. Pure Dicke narrowing (nu_vc > 0, no SD)
# ---------------------------------------------------------------------------

class TestDickeNarrowing:
    """With nu_vc > 0, velocity-changing collisions narrow the Doppler core."""

    def test_fwhm_narrower_than_voigt(self):
        nu0 = 6047.0
        doppler_hwhm = 0.01
        gamma_0 = 0.005
        delta_0 = 0.0
        nu_vc = 0.02
        nu = np.linspace(nu0 - 0.1, nu0 + 0.1, 20001)

        voigt = voigt_profile(nu, nu0, doppler_hwhm, gamma_0)
        dicke = htp_profile(
            nu, nu0, doppler_hwhm, gamma_0, delta_0,
            nu_vc=nu_vc,
        )

        voigt_half = voigt.max() / 2.0
        dicke_half = dicke.max() / 2.0

        voigt_fwhm = nu[voigt > voigt_half][-1] - nu[voigt > voigt_half][0]
        dicke_fwhm = nu[dicke > dicke_half][-1] - nu[dicke > dicke_half][0]

        assert dicke_fwhm < voigt_fwhm, (
            f"Dicke FWHM {dicke_fwhm:.6f} not narrower than Voigt {voigt_fwhm:.6f}"
        )

    def test_peak_higher_than_voigt(self):
        """Narrower profile with unit area must have a higher peak."""
        nu0 = 6047.0
        doppler_hwhm = 0.01
        gamma_0 = 0.005
        nu = np.linspace(nu0 - 0.1, nu0 + 0.1, 20001)

        voigt = voigt_profile(nu, nu0, doppler_hwhm, gamma_0)
        dicke = htp_profile(
            nu, nu0, doppler_hwhm, gamma_0, delta_0=0.0,
            nu_vc=0.02,
        )

        assert dicke.max() > voigt.max()


# ---------------------------------------------------------------------------
# 6. Symmetry (positivity not guaranteed for extreme/full-HTP parameters)
# ---------------------------------------------------------------------------

class TestProfileProperties:
    """Basic physical properties of the HTP."""

    def test_positive_for_physical_params(self):
        """Profile should be positive for physically reasonable parameters."""
        nu0 = 6047.0
        nu = np.linspace(nu0 - 1.0, nu0 + 1.0, 5001)
        phi = htp_profile(
            nu, nu0, 0.01, 0.05, -0.005,
            gamma_2=0.005, delta_2=0.002, nu_vc=0.01, eta=0.1,
        )
        assert np.all(phi >= -1e-10), f"Profile has negative values, min={np.min(phi):.4e}"

    def test_symmetric_when_no_shift(self):
        """When delta_0 = delta_2 = 0, profile is symmetric about nu0."""
        nu0 = 6047.0
        offsets = np.linspace(0.001, 0.3, 200)
        phi_plus = htp_profile(
            nu0 + offsets, nu0, 0.01, 0.05, 0.0,
            gamma_2=0.005, delta_2=0.0, nu_vc=0.01, eta=0.0,
        )
        phi_minus = htp_profile(
            nu0 - offsets, nu0, 0.01, 0.05, 0.0,
            gamma_2=0.005, delta_2=0.0, nu_vc=0.01, eta=0.0,
        )
        np.testing.assert_allclose(phi_plus, phi_minus, rtol=1e-10)
