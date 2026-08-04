"""HITRAN line-parameter access: hapi wrapper, local cache, LineList container.

Line data are fetched through the official HITRAN API client ``hapi``
(R.V. Kochanov et al., JQSRT 177 (2016) 15, doi:10.1016/j.jqsrt.2016.03.005)
and cached under ``.hitran_cache/``. The HITRAN edition and fetch date are
recorded so that every record's provenance can pin the exact line data used.

Offline use: ``LineList`` can be constructed directly from arrays, and
``demo_ch4_2nu3()`` returns an APPROXIMATE built-in line list for examples and
unit tests that must not touch the network. Official dataset generation always
uses hapi-fetched data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# HITRAN molecule numbers (Gordon et al., JQSRT 277 (2022) 107949,
# doi:10.1016/j.jqsrt.2021.107949)
MOLECULE_IDS = {
    "H2O": 1,
    "CO2": 2,
    "O3": 3,
    "N2O": 4,
    "CO": 5,
    "CH4": 6,
    "O2": 7,
    "NH3": 11,
}

# Molar mass of the principal isotopologue [amu] (HITRAN isotopologue metadata)
PRINCIPAL_ISO_MASS_AMU = {
    "H2O": 18.010565,
    "CO2": 43.989830,
    "N2O": 44.001062,
    "CO": 27.994915,
    "CH4": 16.031300,
    "O2": 31.989830,
    "NH3": 17.026549,
}

DEFAULT_CACHE_DIR = ".hitran_cache"


@dataclass
class LineList:
    """Container for HITRAN line-by-line parameters of one molecule.

    All arrays share the same length (one entry per transition). Units follow
    the HITRAN native convention (Rothman et al., JQSRT 110 (2009) 533):

    - nu0_cm1: vacuum transition wavenumber [cm-1]
    - sw_cm_per_molec: line intensity at 296 K [cm-1 / (molecule cm-2)]
    - gamma_air / gamma_self: pressure HWHM [cm-1/atm] at 296 K
    - n_air: temperature exponent of gamma_air [-]
    - delta_air: air pressure-induced line shift [cm-1/atm]
    - elower_cm1: lower-state energy [cm-1]
    """

    molecule: str
    nu0_cm1: np.ndarray
    sw_cm_per_molec: np.ndarray
    gamma_air: np.ndarray
    gamma_self: np.ndarray
    n_air: np.ndarray
    delta_air: np.ndarray
    elower_cm1: np.ndarray
    molar_mass_amu: float = 0.0
    source: str = "unspecified"
    hitran_data_version: str = "unspecified"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = [
            self.nu0_cm1,
            self.sw_cm_per_molec,
            self.gamma_air,
            self.gamma_self,
            self.n_air,
            self.delta_air,
            self.elower_cm1,
        ]
        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("All LineList arrays must have equal length")
        if self.molar_mass_amu <= 0.0:
            mass = PRINCIPAL_ISO_MASS_AMU.get(self.molecule)
            if mass is None:
                raise ValueError(
                    f"Unknown molar mass for {self.molecule!r}; pass molar_mass_amu"
                )
            self.molar_mass_amu = mass

    def __len__(self) -> int:
        return len(self.nu0_cm1)


def fetch_lines(
    molecule: str,
    wavenumber_start_cm1: float,
    wavenumber_end_cm1: float,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> LineList:
    """Fetch line parameters from HITRAN via hapi, with a local file cache.

    Requires the ``hitran-api`` package and (on first call for a region)
    network access to hitran.org. Subsequent calls hit the cache.
    """
    try:
        import hapi
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "hitran-api is required for HITRAN fetching: pip install 'opengasspec[hitran]'"
        ) from exc

    mol_id = MOLECULE_IDS.get(molecule)
    if mol_id is None:
        raise ValueError(f"Unsupported molecule {molecule!r}; known: {sorted(MOLECULE_IDS)}")

    os.makedirs(cache_dir, exist_ok=True)
    hapi.db_begin(cache_dir)
    table = f"{molecule}_{wavenumber_start_cm1:.2f}_{wavenumber_end_cm1:.2f}"
    if table not in hapi.getTableList():
        hapi.fetch(table, mol_id, 1, wavenumber_start_cm1, wavenumber_end_cm1)

    nu0, sw, g_air, g_self, n_air, d_air, elower = hapi.getColumns(
        table, ["nu", "sw", "gamma_air", "gamma_self", "n_air", "delta_air", "elower"]
    )
    return LineList(
        molecule=molecule,
        nu0_cm1=np.asarray(nu0, dtype=np.float64),
        sw_cm_per_molec=np.asarray(sw, dtype=np.float64),
        gamma_air=np.asarray(g_air, dtype=np.float64),
        gamma_self=np.asarray(g_self, dtype=np.float64),
        n_air=np.asarray(n_air, dtype=np.float64),
        delta_air=np.asarray(d_air, dtype=np.float64),
        elower_cm1=np.asarray(elower, dtype=np.float64),
        source="hapi",
        hitran_data_version=f"HITRAN via hapi, table {table}",
    )


def demo_ch4_2nu3() -> LineList:
    """Built-in APPROXIMATE CH4 line list near 6046-6048 cm-1 (2nu3 band, ~1653 nm).

    For offline examples and unit tests ONLY. Parameter values are
    representative of the 2nu3 R-branch region used in methane sensing but are
    NOT authoritative HITRAN data — official dataset generation must use
    :func:`fetch_lines`. Tests of physics correctness (normalization,
    Beer-Lambert linearity, lineshape limits) are invariant to the exact
    parameter values.
    """
    return LineList(
        molecule="CH4",
        nu0_cm1=np.array([6046.9647, 6046.4180, 6047.5100]),
        sw_cm_per_molec=np.array([1.2e-21, 4.0e-22, 2.5e-22]),
        gamma_air=np.array([0.060, 0.062, 0.058]),
        gamma_self=np.array([0.075, 0.078, 0.073]),
        n_air=np.array([0.72, 0.70, 0.74]),
        delta_air=np.array([-0.008, -0.007, -0.009]),
        elower_cm1=np.array([62.88, 104.77, 157.13]),
        source="builtin-demo (approximate values, not for production)",
        hitran_data_version="n/a (demo)",
    )
