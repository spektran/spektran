"""Simulate a WMS 2f signal for CH4 and report the line-center peak height."""

import numpy as np

from spektran.physics import simulate_absorbance
from spektran.physics.wms import WMSConfig, simulate_wms

if __name__ == "__main__":
    nu_grid, absorbance_grid = simulate_absorbance(
        molecule="CH4", concentration_ppm=100.0, temperature_K=296.0, pressure_atm=1.0,
        path_length_m=10.0, wavenumber_start_cm1=6046.0, wavenumber_end_cm1=6048.0, n_points=5000,
    )

    def absorbance_of_nu(nu):
        return np.interp(nu, nu_grid, absorbance_grid)

    f_m, scan_rate = 10_000.0, 25.0
    cfg = WMSConfig(
        center_wavenumber_cm1=6046.9647,
        modulation_depth_cm1=0.06,
        modulation_frequency_Hz=f_m,
        scan_range_cm1=0.4,
        scan_rate_Hz=scan_rate,
        duration_s=1.0 / scan_rate,
        sampling_rate_Hz=f_m * 50.0,
    )
    out = simulate_wms(cfg, absorbance_of_nu, harmonics=(1, 2))

    n = len(out["r_2f"])
    settled = slice(n // 10, n - n // 10)  # skip lock-in filter edge transients
    r2f = out["r_2f"][settled]
    peak_idx = int(r2f.argmax())
    print(f"2f peak height: {r2f[peak_idx]:.4e} at {out['nu_cm1'][settled][peak_idx]:.4f} cm-1")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        plt.plot(out["t_s"][settled], r2f)
        plt.xlabel("time [s]")
        plt.ylabel("R_2f")
        plt.title("WMS 2f signal, CH4 100 ppm")
        plt.show()
