"""Physical constants in CGS / spectroscopic units.

Values follow CODATA 2018 (exact where SI-defined).
Internal unit convention (see project plan §11): wavenumbers in cm-1,
pressure in atm, temperature in K, lengths in cm unless suffixed otherwise.
"""

# Speed of light [cm/s] (exact)
C_CM_PER_S = 2.99792458e10

# Boltzmann constant [erg/K] (exact: 1.380649e-23 J/K)
K_ERG_PER_K = 1.380649e-16

# Planck constant [erg*s] (exact: 6.62607015e-34 J*s)
H_ERG_S = 6.62607015e-27

# Second radiation constant c2 = h*c/k [cm*K]
C2_CM_K = H_ERG_S * C_CM_PER_S / K_ERG_PER_K  # = 1.4387768775039337

# Avogadro constant [1/mol] (exact)
N_AVOGADRO = 6.02214076e23

# Atomic mass unit [g] (CODATA 2018)
AMU_G = 1.66053906660e-24

# Standard atmosphere [dyn/cm^2] (exact: 101325 Pa)
ATM_DYN_PER_CM2 = 1.01325e6

# HITRAN reference temperature [K]
T_REF_K = 296.0


def number_density_cm3(pressure_atm: float, temperature_K: float) -> float:
    """Ideal-gas number density n = P / (k_B * T) in molecules/cm^3."""
    return pressure_atm * ATM_DYN_PER_CM2 / (K_ERG_PER_K * temperature_K)
