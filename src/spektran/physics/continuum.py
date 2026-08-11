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

_CS_SELF_TABLE = {
    1600: 3.5e-24,
    2000: 7.0e-25,
    2500: 1.5e-25,
    3000: 5.0e-26,
    3500: 2.0e-26,
    4000: 8.0e-27,
    4500: 3.0e-27,
    5000: 1.0e-27,
    5500: 5.0e-28,
    6000: 2.0e-28,
    6500: 1.0e-28,
    7000: 5.0e-29,
}

_CS_FOREIGN_TABLE = {
    1600: 5.0e-26,
    2000: 1.0e-26,
    2500: 3.0e-27,
    3000: 1.0e-27,
    3500: 5.0e-28,
    4000: 2.0e-28,
    4500: 8.0e-29,
    5000: 3.0e-29,
    5500: 1.5e-29,
    6000: 8.0e-30,
    6500: 4.0e-30,
    7000: 2.0e-30,
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
    n_foreign = n_total * (1.0 - h2o_mole_fraction)

    alpha_cont = n_h2o * (cs_self_T * n_h2o + cs_foreign_T * n_foreign)

    path_cm = path_length_m * 100.0
    return alpha_cont * path_cm
