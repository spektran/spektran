"""Multi-species Beer-Lambert superposition: CH4 target with H2O interferent."""

import numpy as np

from spektran.physics import absorption_coefficient
from spektran.physics.hitran import demo_ch4_2nu3, demo_h2o

if __name__ == "__main__":
    nu = np.linspace(6046.0, 6048.0, 2000)
    path_cm = 10.0 * 100.0
    ch4_ppm = 100.0
    h2o_ppm = 10000.0

    alpha_ch4 = absorption_coefficient(
        nu, demo_ch4_2nu3(), mole_fraction=ch4_ppm * 1e-6, temperature_K=296.0, pressure_atm=1.0,
    )
    # demo_h2o's built-in lines sit in a different band (1.4 um), so they
    # contribute little here -- this still exercises the superposition path.
    alpha_h2o = absorption_coefficient(
        nu, demo_h2o(), mole_fraction=h2o_ppm * 1e-6, temperature_K=296.0, pressure_atm=1.0,
    )

    absorbance_ch4 = alpha_ch4 * path_cm
    absorbance_h2o = alpha_h2o * path_cm
    absorbance_total = absorbance_ch4 + absorbance_h2o

    print(f"CH4 ({ch4_ppm:.0f} ppm) peak absorbance: {absorbance_ch4.max():.4e}")
    print(f"H2O ({h2o_ppm:.0f} ppm) peak absorbance: {absorbance_h2o.max():.4e}")
    print(f"combined peak absorbance: {absorbance_total.max():.4e}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        plt.semilogy(nu, absorbance_ch4, label=f"CH4 ({ch4_ppm:.0f} ppm)")
        plt.semilogy(nu, absorbance_h2o, label=f"H2O ({h2o_ppm:.0f} ppm)")
        plt.semilogy(nu, absorbance_total, "--", label="combined")
        plt.xlabel("wavenumber [cm-1]")
        plt.ylabel("absorbance")
        plt.legend()
        plt.show()
