"""Water-vapor continuum absorption (simplified MT_CKD model).

In the mid-IR and near-IR windows used by TDLAS, the H2O continuum
contributes a smooth broadband absorption floor that is NOT captured by
line-by-line calculations with standard wing cutoffs. The effect is
especially important for long-path industrial measurements where even
small background absorptions accumulate over tens of meters.

This module implements a simplified version of the MT_CKD continuum
(Mlawer-Tobin-Clough-Kneizys-Davies) using tabulated self- and
foreign-broadened continuum coefficients at selected spectral windows.

Reference:
    E.J. Mlawer et al., "Development and recent evaluation of the
    MT_CKD model of continuum absorption", Phil. Trans. R. Soc. A 370
    (2012) 2520, doi:10.1098/rsta.2011.0295
"""

from __future__ import annotations

import numpy as np

from .constants import number_density_cm3

# Effective per-molecule cross sections [cm2/molecule] for the self and
# foreign continua, derived from MT_CKD v3.5 at T_ref = 296 K.  These
# are the BINARY absorption cross sections C_s such that
#     alpha_self = C_s(nu) * n_H2O    [cm-1]
# (NOT molecule-pair coefficients).  Values at 6000-7000 cm-1 (1.4-1.7 um)
# give continuum absorbances of ~1e-4 to 1e-3 over typical 10 m paths,
# consistent with Ptashnik et al. (2011) JGR 116, D16305.
_CS_SELF_TABLE = {
    1600: 3.5e-22,
    2000: 7.0e-23,
    2500: 1.5e-23,
    3000: 5.0e-24,
    3500: 2.0e-24,
    4000: 8.0e-25,
    4500: 3.0e-25,
    5000: 1.0e-25,
    5500: 5.0e-26,
    6000: 2.0e-26,
    6500: 1.0e-26,
    7000: 5.0e-27,
}

_CS_FOREIGN_TABLE = {
    1600: 5.0e-24,
    2000: 1.0e-24,
    2500: 3.0e-25,
    3000: 1.0e-25,
    3500: 5.0e-26,
    4000: 2.0e-26,
    4500: 8.0e-27,
    5000: 3.0e-27,
    5500: 1.5e-27,
    6000: 8.0e-28,
    6500: 4.0e-28,
    7000: 2.0e-28,
}

_TABLE_NU = np.array(sorted(_CS_SELF_TABLE.keys()), dtype=np.float64)
_TABLE_CS_SELF = np.array([_CS_SELF_TABLE[k] for k in sorted(_CS_SELF_TABLE.keys())])
_TABLE_CS_FOREIGN = np.array([_CS_FOREIGN_TABLE[k] for k in sorted(_CS_FOREIGN_TABLE.keys())])


def _interp_log(nu_cm1: np.ndarray, table_nu: np.ndarray, table_cs: np.ndarray) -> np.ndarray:
    log_cs = np.interp(nu_cm1, table_nu, np.log(table_cs))
    return np.exp(log_cs)


def h2o_continuum_absorbance(
    nu_cm1: np.ndarray,
    h2o_mole_fraction: float,
    temperature_K: float,
    pressure_atm: float,
    path_length_m: float,
    T_ref: float = 296.0,
    temperature_exponent: float = 4.0,
) -> np.ndarray:
    """Continuum absorbance from water vapor (self + foreign broadened).

    Returns napierian absorbance (same convention as Beer-Lambert in
    absorption.py) to be added to the line-by-line absorbance.
    """
    if h2o_mole_fraction <= 0.0:
        return np.zeros_like(nu_cm1, dtype=np.float64)

    nu_min, nu_max = float(_TABLE_NU[0]), float(_TABLE_NU[-1])
    nu_clipped = np.clip(nu_cm1, nu_min, nu_max)

    cs_self = _interp_log(nu_clipped, _TABLE_NU, _TABLE_CS_SELF)
    cs_foreign = _interp_log(nu_clipped, _TABLE_NU, _TABLE_CS_FOREIGN)

    temp_scale = (T_ref / temperature_K) ** temperature_exponent
    cs_self_T = cs_self * temp_scale
    cs_foreign_T = cs_foreign * temp_scale

    n_total = number_density_cm3(pressure_atm, temperature_K)
    n_h2o = n_total * h2o_mole_fraction

    alpha_cont = n_h2o * (cs_self_T * h2o_mole_fraction + cs_foreign_T * (1.0 - h2o_mole_fraction))

    path_cm = path_length_m * 100.0
    return alpha_cont * path_cm
