"""REFERENCE implementation of the Voigt profile — Gate G3 cross-validation.

Deliberately independent algorithm path: adaptive numerical quadrature of the
Voigt definition integral

    K(x, y) = (y / pi) * Integral exp(-t^2) / (y^2 + (x - t)^2) dt
    phi_V(nu) = sqrt(ln2 / pi) / alpha_D * K(x, y)
    x = sqrt(ln2) * (nu - nu0) / alpha_D,  y = sqrt(ln2) * gamma_L / alpha_D

(e.g. B.H. Armstrong, JQSRT 7 (1967) 61, doi:10.1016/0022-4073(67)90057-X).

MUST NOT import from or share code with ``spektran`` (see plan §9, G3).
The main implementation uses the Faddeeva function instead (scipy.special.wofz).
"""

from __future__ import annotations

import math

from scipy.integrate import quad

_SQRT_LN2 = math.sqrt(math.log(2.0))

# exp(-t^2) < 2e-29 beyond |t| = 8.1; truncation error is negligible relative
# to the smallest integral values probed in the G3 parameter ranges.
_T_LIMIT = 8.5


def voigt_K(x: float, y: float) -> float:
    """Voigt function K(x, y) by adaptive quadrature."""
    if y <= 0.0:
        raise ValueError("y must be > 0 for the quadrature reference")

    def integrand(t: float) -> float:
        d = x - t
        return math.exp(-t * t) / (y * y + d * d)

    # Break the interval at the Lorentzian peak (t = x) so the adaptive rule
    # resolves the near-singular region when y is small.
    pts = sorted({-_T_LIMIT, _T_LIMIT, *(p for p in (x - y, x, x + y) if -_T_LIMIT < p < _T_LIMIT)})
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        val, _ = quad(integrand, a, b, limit=400, epsabs=1e-16, epsrel=1e-12)
        total += val
    return y / math.pi * total


def voigt_profile_ref(
    nu_cm1: float, nu0_cm1: float, doppler_hwhm: float, lorentz_hwhm: float
) -> float:
    """Area-normalized Voigt profile value [cm] at a single wavenumber."""
    x = _SQRT_LN2 * (nu_cm1 - nu0_cm1) / doppler_hwhm
    y = _SQRT_LN2 * lorentz_hwhm / doppler_hwhm
    return _SQRT_LN2 / (doppler_hwhm * math.sqrt(math.pi)) * voigt_K(x, y)
