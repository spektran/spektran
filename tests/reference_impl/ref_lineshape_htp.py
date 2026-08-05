"""REFERENCE implementation of HTP sub-cases — Gate G3 cross-validation.

Provides a completely independent algorithm for the speed-dependent Voigt
(SDV) profile via 2D Gauss-Hermite x Gauss-Laguerre quadrature over the
Maxwell-Boltzmann speed distribution.  No Faddeeva function involved —
this is structurally independent of the main hapi-based implementation.

For the full HTP (with velocity-changing collisions and correlation),
the main code delegates to hapi's ``pcqsdhc`` — the official reference
implementation by the paper's authors.  No independent reimplementation
is attempted; see the "no reinventing the wheel" principle in CLAUDE.md.

MUST NOT import from or share code with ``spektran``.

References:
- H. Ngo et al., JQSRT 129 (2013) 89, doi:10.1016/j.jqsrt.2013.05.034
"""

from __future__ import annotations

import math

import numpy as np

_SQRT_LN2 = math.sqrt(math.log(2.0))
_SQRT_PI = math.sqrt(math.pi)


def ref_sdv_quadrature(
    nu_cm1: float,
    nu0_cm1: float,
    doppler_hwhm: float,
    gamma_0: float,
    delta_0: float,
    gamma_2: float,
    delta_2: float,
    n_hermite: int = 100,
    n_laguerre: int = 80,
) -> float:
    """SDV profile via 2D speed quadrature — completely independent algorithm.

    Integrates the speed-dependent Lorentzian over the Maxwell-Boltzmann
    distribution using Gauss-Hermite (line-of-sight velocity v_z) and
    Gauss-Laguerre (transverse kinetic energy v_perp^2) quadrature.

    Valid only for the SDV case: nu_vc = 0, eta = 0.

    The dimensionless velocity coordinates are x = v_z / v_p (Hermite) and
    s = v_perp^2 / v_p^2 (Laguerre), where v_p = sqrt(2kT/m) is the most
    probable speed. The dimensionless speed squared is u^2 = x^2 + s.
    """
    sigma = doppler_hwhm / _SQRT_LN2
    dnu = nu_cm1 - nu0_cm1

    x_nodes, x_weights = np.polynomial.hermite.hermgauss(n_hermite)
    s_nodes, s_weights = np.polynomial.laguerre.laggauss(n_laguerre)

    total = 0.0
    for xj, wj in zip(x_nodes, x_weights):
        for sk, wk in zip(s_nodes, s_weights):
            u2 = xj * xj + sk
            gamma_u = gamma_0 + gamma_2 * (u2 - 1.5)
            delta_u = delta_0 + delta_2 * (u2 - 1.5)

            if gamma_u <= 0.0:
                continue

            d = dnu - sigma * xj - delta_u
            total += wj * wk * gamma_u / (d * d + gamma_u * gamma_u)

    return total / (math.pi * _SQRT_PI)
