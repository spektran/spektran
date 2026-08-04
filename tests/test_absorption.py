"""Physics-correctness tests for the Beer-Lambert forward model (plan §8)."""

import numpy as np
import pytest

from spektran.physics import demo_ch4_2nu3, line_strength_at_T, simulate_absorbance
from spektran.physics.absorption import absorption_coefficient, default_q_ratio
from spektran.physics.constants import number_density_cm3
from tests.reference_impl.ref_absorption import absorbance_ref

RNG_SEED = 20260805


class TestBeerLambertLinearity:
    """Optically thin: absorbance linear in concentration, R^2 > 0.9999."""

    def test_concentration_linearity(self):
        concs = np.linspace(1.0, 200.0, 25)  # ppm, optically thin at 10 m
        peaks = []
        for c in concs:
            _, a = simulate_absorbance(concentration_ppm=float(c))
            peaks.append(a.max())
        peaks = np.asarray(peaks)
        slope, intercept = np.polyfit(concs, peaks, 1)
        fitted = slope * concs + intercept
        ss_res = np.sum((peaks - fitted) ** 2)
        ss_tot = np.sum((peaks - peaks.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.9999

    def test_doubling(self):
        # Not bit-exact: mole fraction feeds the self-broadening term
        # (gamma_self * x * P), so the Lorentz width shifts by
        # ~ x * (gamma_self - gamma_air) * P ~ 1e-5 relative at ppm levels.
        # Linearity holds to that physically-expected order.
        _, a1 = simulate_absorbance(concentration_ppm=50.0)
        _, a2 = simulate_absorbance(concentration_ppm=100.0)
        assert np.allclose(a2, 2.0 * a1, rtol=1e-4)

    def test_path_length_doubling(self):
        _, a1 = simulate_absorbance(path_length_m=5.0)
        _, a2 = simulate_absorbance(path_length_m=10.0)
        assert np.allclose(a2, 2.0 * a1, rtol=1e-12)


class TestLineStrength:
    def test_identity_at_reference_temperature(self):
        lines = demo_ch4_2nu3()
        s = line_strength_at_T(
            lines.sw_cm_per_molec, lines.nu0_cm1, lines.elower_cm1, 296.0, 1.0
        )
        assert np.allclose(s, lines.sw_cm_per_molec, rtol=1e-14)

    def test_number_density_at_stp_like_conditions(self):
        # 1 atm, 296 K -> ~2.479e19 cm-3 (ideal gas)
        assert number_density_cm3(1.0, 296.0) == pytest.approx(2.4790e19, rel=1e-3)


class TestReproducibility:
    """Plan §8: same config + same version -> bit-identical output."""

    def test_bit_identical(self):
        nu1, a1 = simulate_absorbance()
        nu2, a2 = simulate_absorbance()
        assert nu1.tobytes() == nu2.tobytes()
        assert a1.tobytes() == a2.tobytes()


class TestAbsorptionCrossValidation:
    """Gate G3: full forward chain, main vs reference, 1000 random points < 0.1%."""

    def test_1000_random_points(self):
        rng = np.random.default_rng(RNG_SEED)
        lines = demo_ch4_2nu3()
        n = 1000
        max_rel = 0.0
        for _ in range(n):
            j = int(rng.integers(0, len(lines)))
            T = rng.uniform(250.0, 350.0)
            P = 10.0 ** rng.uniform(-1.0, 0.3)  # 0.1 .. 2 atm
            x = 10.0 ** rng.uniform(-6.0, -3.0)  # 1 ppm .. 1000 ppm
            L_cm = rng.uniform(10.0, 5000.0)
            offset = rng.uniform(-0.3, 0.3)
            nu = lines.nu0_cm1[j] + offset
            q = default_q_ratio("CH4", T)

            # Main implementation: single-line LineList slice
            from spektran.physics.hitran import LineList

            single = LineList(
                molecule="CH4",
                nu0_cm1=lines.nu0_cm1[j : j + 1],
                sw_cm_per_molec=lines.sw_cm_per_molec[j : j + 1],
                gamma_air=lines.gamma_air[j : j + 1],
                gamma_self=lines.gamma_self[j : j + 1],
                n_air=lines.n_air[j : j + 1],
                delta_air=lines.delta_air[j : j + 1],
                elower_cm1=lines.elower_cm1[j : j + 1],
            )
            # Pin q_ratio explicitly: this test cross-validates the Voigt/HITRAN
            # line-shape formula, not the partition-function model, so it must
            # not depend on whichever q_ratio absorption_coefficient defaults to.
            alpha = absorption_coefficient(
                np.array([nu]), single, x, T, P, q_ratio=default_q_ratio
            )[0]
            main = alpha * L_cm

            ref = absorbance_ref(
                nu,
                nu0_cm1=float(lines.nu0_cm1[j]),
                sw_ref=float(lines.sw_cm_per_molec[j]),
                gamma_air=float(lines.gamma_air[j]),
                gamma_self=float(lines.gamma_self[j]),
                n_air=float(lines.n_air[j]),
                delta_air=float(lines.delta_air[j]),
                elower_cm1=float(lines.elower_cm1[j]),
                molar_mass_amu=lines.molar_mass_amu,
                mole_fraction=x,
                temperature_K=T,
                pressure_atm=P,
                path_length_cm=L_cm,
                q_ratio_value=q,
            )
            rel = abs(main - ref) / abs(ref)
            max_rel = max(max_rel, rel)
        assert max_rel < 1e-3, f"max relative deviation {max_rel:.2e} exceeds 0.1%"


class TestInputValidation:
    def test_mole_fraction_bounds(self):
        lines = demo_ch4_2nu3()
        with pytest.raises(ValueError):
            absorption_coefficient(np.array([6047.0]), lines, 1.5, 296.0, 1.0)

    def test_non_ch4_without_lines_raises(self):
        with pytest.raises(ValueError):
            simulate_absorbance(molecule="CO2")
