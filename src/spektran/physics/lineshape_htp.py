"""Hartmann-Tran Profile (HTP): the HITRAN2016+ recommended beyond-Voigt line shape.

Thin wrapper around hapi's ``pcqsdhc`` — the official reference implementation
by H. Tran, N.H. Ngo, and J.-M. Hartmann.

References:
- H. Tran, N.H. Ngo, J.-M. Hartmann, JQSRT 129 (2013) 199,
  doi:10.1016/j.jqsrt.2013.06.015
- H. Ngo, D. Lisak, H. Tran, J.-M. Hartmann, JQSRT 129 (2013) 89,
  doi:10.1016/j.jqsrt.2013.05.034
- J. Tennyson et al., Pure Appl. Chem. 86 (2014) 1931,
  doi:10.1515/pac-2014-0208

An independent reference implementation (2D speed quadrature over the
Maxwell-Boltzmann distribution) lives in
``tests/reference_impl/ref_lineshape_htp.py`` (Gate G3).
"""

from __future__ import annotations

import numpy as np
from hapi import pcqsdhc


def htp_profile(
    nu_cm1: np.ndarray,
    nu0_cm1: float,
    doppler_hwhm: float,
    gamma_0: float,
    delta_0: float,
    gamma_2: float = 0.0,
    delta_2: float = 0.0,
    nu_vc: float = 0.0,
    eta: float = 0.0,
) -> np.ndarray:
    """Hartmann-Tran Profile [cm].

    IUPAC/HITRAN2016+ recommended beyond-Voigt line shape.  Delegates to
    hapi's ``pcqsdhc`` (Tran, Ngo, Hartmann, JQSRT 129 (2013) 199).

    Reduces to Voigt when gamma_2 = delta_2 = nu_vc = eta = 0.

    Parameters
    ----------
    nu_cm1 : array
        Wavenumber grid [cm-1].
    nu0_cm1 : float
        Line center [cm-1].
    doppler_hwhm : float
        Doppler (Gaussian) half-width at half-maximum [cm-1].
    gamma_0 : float
        Pressure-broadened Lorentzian HWHM [cm-1].
    delta_0 : float
        Pressure-induced line shift [cm-1].
    gamma_2 : float
        Speed-dependent broadening parameter [cm-1].
    delta_2 : float
        Speed-dependent shifting parameter [cm-1].
    nu_vc : float
        Velocity-changing collision frequency [cm-1].
    eta : float
        Correlation parameter between velocity- and state-changing
        collisions [dimensionless, 0 <= eta <= 1].

    Returns
    -------
    phi : array
        Profile values [cm].  Area-normalized to unity for physically
        reasonable parameter combinations (gamma_2/gamma_0 << 1).
    """
    # pcqsdhc(sg0, GamD, Gam0, Gam2, Shift0, Shift2, anuVC, eta, sg)
    real_part, _ = pcqsdhc(
        nu0_cm1, doppler_hwhm, gamma_0, gamma_2,
        delta_0, delta_2, nu_vc, eta,
        np.asarray(nu_cm1, dtype=np.float64),
    )
    return np.asarray(real_part, dtype=np.float64)
