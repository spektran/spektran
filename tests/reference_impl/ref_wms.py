"""REFERENCE implementation for WMS harmonics — Gate G3 (WMS part).

Independent algorithm path: the lock-in output of a periodically modulated
transmission signal equals its Fourier cosine/sine coefficients. For
nu(t) = nu_c + a*cos(theta), theta = omega*t, and transmission tau(nu),
the k-th cosine harmonic of s(t) = I0(t)*tau(nu(t)) is computed directly by
numerical quadrature over one modulation period:

    C_k = (2/2pi) * Integral_0^2pi s(theta) cos(k*theta) dtheta   (k >= 1)
    S_k = (2/2pi) * Integral_0^2pi s(theta) sin(k*theta) dtheta

matching the main implementation's lock-in normalization (X of A*cos(k w t)
equals A). For the transmission-only part this reduces to the classical
harmonic coefficients H_k(nu_c, a) of the WMS literature.

Analytic cross-check: for an optically thin Lorentzian line, the harmonic
coefficients have Arndt's closed form (R. Arndt, "Analytical line shapes for
Lorentzian signals broadened by modulation", J. Appl. Phys. 36 (1965) 2522,
doi:10.1063/1.1714333; see also J. Reid & D. Labrie, "Second-harmonic
detection with tunable diode lasers — comparison of experiment and theory",
Appl. Phys. B 26 (1981) 203, doi:10.1007/BF00692448).

MUST NOT import from or share code with ``opensensorsim``.
"""

from __future__ import annotations

import math

from scipy.integrate import quad


def harmonic_coefficients(
    signal_of_theta,
    k: int,
) -> tuple[float, float]:
    """(cosine, sine) Fourier coefficients of a 2pi-periodic signal, k >= 1."""

    def c_int(theta: float) -> float:
        return signal_of_theta(theta) * math.cos(k * theta)

    def s_int(theta: float) -> float:
        return signal_of_theta(theta) * math.sin(k * theta)

    c, _ = quad(c_int, 0.0, 2.0 * math.pi, limit=400, epsabs=1e-13, epsrel=1e-11)
    s, _ = quad(s_int, 0.0, 2.0 * math.pi, limit=400, epsabs=1e-13, epsrel=1e-11)
    return c / math.pi, s / math.pi


def wms_harmonic_ref(
    absorbance_of_nu,
    nu_center_cm1: float,
    depth_cm1: float,
    harmonic: int,
    im_i0_rel: float = 0.0,
    im_i2_rel: float = 0.0,
    fm_im_phase1_rad: float = 0.0,
    fm_im_phase2_rad: float = 0.0,
    mean_intensity: float = 1.0,
) -> tuple[float, float]:
    """Reference (X, Y) lock-in output at ``harmonic`` for a fixed line center.

    theta convention: nu(theta) = nu_c + a*cos(theta), intensity
    I0(theta) = Ibar*(1 + i0*cos(theta+psi1) + i2*cos(2*theta+psi2)),
    matching the main implementation with lockin_phase = 0.
    """

    def signal(theta: float) -> float:
        nu = nu_center_cm1 + depth_cm1 * math.cos(theta)
        i0 = mean_intensity * (
            1.0
            + im_i0_rel * math.cos(theta + fm_im_phase1_rad)
            + im_i2_rel * math.cos(2.0 * theta + fm_im_phase2_rad)
        )
        return i0 * math.exp(-absorbance_of_nu(nu))

    return harmonic_coefficients(signal, harmonic)


def arndt_lorentzian_h2_peak(peak_absorbance: float, m: float) -> float:
    """Analytic 2f X-magnitude at line center, optically thin Lorentzian.

    The Lorentzian L(x) = 1/(1+x^2) under modulation x = x0 + m*cos(theta)
    has closed-form harmonic coefficients (Arndt 1965, doi:10.1063/1.1714333;
    Reid & Labrie 1981, doi:10.1007/BF00692448). At line center (x0 = 0),
    with the C_k = (1/pi) * Int_0^2pi L cos(k theta) dtheta normalization:

        H2_L(0, m) = (4/m^2) * [ 1 - (2 + m^2) / (2*sqrt(1 + m^2)) ]   (< 0)

    (small-m limit: H2_L -> -m^2/2, i.e. (m/2)^2 * d^2L/dx^2 at 0.)

    For optically thin transmission tau ~ 1 - A_peak*L(x(theta)), the 2f
    cosine coefficient of tau is -A_peak*H2_L(0, m) > 0. This function
    returns that positive transmission value:

        X2f_peak = A_peak * (2/m^2) * [ (2 + m^2)/sqrt(1 + m^2) - 2 ]

    with A_peak the peak napierian absorbance and m = a / HWHM.
    """
    m2 = m * m
    return peak_absorbance * (2.0 / m2) * ((2.0 + m2) / math.sqrt(1.0 + m2) - 2.0)
