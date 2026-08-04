"""HITRAN/hapi reference comparison (plan §8: deviation < 0.1% at 296 K, 1 atm).

Marked ``hitran_online``: requires the hitran-api package and network access on
first run (line data are cached afterwards). Excluded from hermetic PR CI.
"""

import numpy as np
import pytest

pytest.importorskip("hapi")

from opensensorsim.physics import fetch_lines  # noqa: E402
from opensensorsim.physics.absorption import absorption_coefficient  # noqa: E402

pytestmark = pytest.mark.hitran_online


class TestAgainstHapi:
    def test_ch4_absorption_coefficient_296K(self, tmp_path):
        import hapi

        cache = str(tmp_path / "hitran")
        lines = fetch_lines("CH4", 6045.0, 6049.0, cache_dir=cache)
        assert len(lines) > 0

        nu_grid = np.linspace(6046.0, 6048.0, 2001)
        # hapi reference: Voigt absorption coefficient, air-diluted CH4
        x = 1e-4  # 100 ppm
        nu_h, coef_h = hapi.absorptionCoefficient_Voigt(
            SourceTables=f"CH4_{6045.0:.2f}_{6049.0:.2f}",
            Environment={"T": 296.0, "p": 1.0},
            Diluent={"air": 1.0 - x, "self": x},
            WavenumberGrid=nu_grid,
            HITRAN_units=True,  # cm^2/molecule (cross-section per molecule)
        )
        # Our alpha [cm-1] = n_absorber * cross_section => compare shapes scaled
        from opensensorsim.physics.constants import number_density_cm3

        alpha_ours = absorption_coefficient(nu_grid, lines, x, 296.0, 1.0)
        alpha_hapi = coef_h * number_density_cm3(1.0, 296.0) * x

        peak = alpha_hapi.max()
        mask = alpha_hapi > 0.01 * peak  # compare where signal is meaningful
        rel = np.abs(alpha_ours[mask] - alpha_hapi[mask]) / alpha_hapi[mask]
        assert rel.max() < 1e-3, f"max deviation vs hapi {rel.max():.2e} exceeds 0.1%"
