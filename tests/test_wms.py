"""Physics-correctness tests for the WMS chain (plan §8: analytic comparison
< 1% in the optically thin, small-modulation limit; §9 G3 WMS part)."""

import numpy as np
import pytest

from spektran.physics.lineshape import lorentz_profile
from spektran.physics.wms import WMSConfig, simulate_wms
from tests.reference_impl.ref_wms import arndt_lorentzian_h2_peak, wms_harmonic_ref

RNG_SEED = 20260807

# Shared toy line: Lorentzian at 6047 cm-1
NU0 = 6047.0
GAMMA_L = 0.05  # HWHM [cm-1]


def make_absorbance(peak: float):
    """Absorbance function with given peak value (Lorentzian)."""
    scale = peak / (1.0 / (np.pi * GAMMA_L))  # peak of area-normalized profile

    def absorbance(nu):
        return scale * lorentz_profile(np.asarray(nu, dtype=float), NU0, GAMMA_L)

    return absorbance


def settled_mean(x: np.ndarray, frac: float = 0.3) -> float:
    """Mean of the central portion, discarding filter edge transients."""
    n = len(x)
    lo, hi = int(n * frac), int(n * (1.0 - frac))
    return float(np.mean(x[lo:hi]))


def settled_peak(x: np.ndarray, frac: float = 0.3) -> float:
    """Max magnitude of the central portion, discarding filter edge transients."""
    n = len(x)
    lo, hi = int(n * frac), int(n * (1.0 - frac))
    return float(np.max(np.abs(x[lo:hi])))


class TestLockinBasics:
    def test_pure_tone_demodulates_to_amplitude(self):
        fs, fm, dur = 2e6, 1e4, 0.02
        t = np.arange(int(fs * dur)) / fs
        sig = 0.7 * np.cos(2 * np.pi * 2 * fm * t)  # pure 2f tone
        from spektran.physics.wms import lockin_demodulate

        x, y = lockin_demodulate(sig, t, fm, harmonic=2, sampling_rate_Hz=fs)
        assert settled_mean(x) == pytest.approx(0.7, rel=1e-6)
        assert abs(settled_mean(y)) < 1e-6

    def test_phase_rotation(self):
        fs, fm, dur = 2e6, 1e4, 0.02
        t = np.arange(int(fs * dur)) / fs
        sig = 0.5 * np.cos(2 * np.pi * 2 * fm * t + 0.4)
        from spektran.physics.wms import lockin_demodulate

        x, y = lockin_demodulate(sig, t, fm, harmonic=2, sampling_rate_Hz=fs)
        # X = A cos(phase_sig), Y = A sin(phase_sig) with this convention
        assert settled_mean(x) == pytest.approx(0.5 * np.cos(0.4), rel=1e-5)
        assert settled_mean(y) == pytest.approx(0.5 * np.sin(0.4), rel=1e-5)


class TestArndtAnalytic:
    """Plan §8 red line: optically thin + small m, 2f peak vs published
    analytic formula, deviation < 1%."""

    @pytest.mark.parametrize("m", [0.3, 0.5, 1.0, 2.0, 2.2])
    def test_2f_peak_matches_arndt(self, m):
        peak_absorbance = 1e-3  # optically thin
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=m * GAMMA_L,
            sampling_rate_Hz=2e6,
            duration_s=0.02,
            center_wavenumber_cm1=NU0,  # sit at line center
        )
        out = simulate_wms(cfg, make_absorbance(peak_absorbance))
        x2f = settled_mean(out["x_2f"])
        analytic = arndt_lorentzian_h2_peak(peak_absorbance, m)
        assert x2f == pytest.approx(analytic, rel=0.01), (
            f"m={m}: time-domain {x2f:.6e} vs Arndt {analytic:.6e}"
        )


class TestWMSCrossValidation:
    """G3 WMS part: time-domain lock-in vs Fourier-quadrature reference,
    random parameters, < 1% (full chain threshold, plan §9)."""

    def test_random_points(self):
        rng = np.random.default_rng(RNG_SEED)
        n_points = 60  # time-domain sim is expensive; gate script runs more
        worst = 0.0
        for _ in range(n_points):
            m = rng.uniform(0.3, 3.0)
            peak = 10.0 ** rng.uniform(-3.0, -0.5)  # thin to moderately thick
            offset = rng.uniform(-2.0, 2.0) * GAMMA_L
            i0 = rng.uniform(0.0, 0.4)
            psi1 = rng.uniform(-np.pi, np.pi)
            harmonic = int(rng.integers(1, 3))
            cfg = WMSConfig(
                modulation_frequency_Hz=1e4,
                modulation_depth_cm1=m * GAMMA_L,
                sampling_rate_Hz=2e6,
                duration_s=0.02,
                center_wavenumber_cm1=NU0 + offset,
                im_i0_rel=i0,
                fm_im_phase1_rad=psi1,
            )
            absorb = make_absorbance(peak)
            out = simulate_wms(cfg, absorb, harmonics=(harmonic,))
            x_main = settled_mean(out[f"x_{harmonic}f"])
            y_main = settled_mean(out[f"y_{harmonic}f"])
            x_ref, y_ref = wms_harmonic_ref(
                lambda nu: float(absorb(np.array([nu]))[0]),
                NU0 + offset,
                m * GAMMA_L,
                harmonic,
                im_i0_rel=i0,
                fm_im_phase1_rad=psi1,
            )
            r_main = float(np.hypot(x_main, y_main))
            r_ref = float(np.hypot(x_ref, y_ref))
            scale = max(abs(r_ref), 1e-12)
            rel = abs(r_main - r_ref) / scale
            worst = max(worst, rel)
        assert worst < 0.01, f"max relative deviation {worst:.3e} exceeds 1%"


class TestDeterminism:
    def test_bit_identical(self):
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=0.05,
            sampling_rate_Hz=1e6,
            duration_s=0.01,
            center_wavenumber_cm1=NU0,
        )
        a = simulate_wms(cfg, make_absorbance(1e-2))
        b = simulate_wms(cfg, make_absorbance(1e-2))
        assert a["x_2f"].tobytes() == b["x_2f"].tobytes()


def test_3f_4f_demodulation_returns_signals():
    """simulate_wms with harmonics=(1,2,3,4) returns all four harmonic outputs."""
    import numpy as np

    from spektran.physics.wms import WMSConfig, simulate_wms

    cfg = WMSConfig(
        modulation_frequency_Hz=10000.0,
        modulation_depth_cm1=0.05,
        sampling_rate_Hz=500000.0,
        duration_s=0.01,
        center_wavenumber_cm1=6047.0,
    )
    result = simulate_wms(cfg, lambda nu: np.zeros_like(nu), harmonics=(1, 2, 3, 4))
    for h in (1, 2, 3, 4):
        assert f"x_{h}f" in result, f"missing x_{h}f"
        assert f"y_{h}f" in result, f"missing y_{h}f"
        assert f"r_{h}f" in result, f"missing r_{h}f"


def test_3f_4f_with_absorbing_gas():
    """3f/4f harmonics are nonzero for absorbing gas and smaller than 2f."""
    from spektran.physics.absorption import absorption_coefficient
    from spektran.physics.hitran import demo_ch4_2nu3
    from spektran.physics.wms import WMSConfig, simulate_wms

    lines = demo_ch4_2nu3()
    cfg = WMSConfig(
        modulation_frequency_Hz=10000.0,
        modulation_depth_cm1=0.03,
        sampling_rate_Hz=1000000.0,
        # 0.02 s = 20 time constants of the default lowpass (cutoff = f_m/10 =
        # 1 kHz), long enough for the zero-phase filter to settle away from
        # the edges; peaks are read via settled_peak() for the same reason.
        duration_s=0.02,
        center_wavenumber_cm1=6046.9647,
    )

    def abs_fn(nu):
        return absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0) * 1000.0

    result = simulate_wms(cfg, abs_fn, harmonics=(1, 2, 3, 4))
    r2f_peak = settled_peak(result["r_2f"])
    for h in (3, 4):
        r_peak = settled_peak(result[f"r_{h}f"])
        assert r_peak > 0, f"{h}f peak should be nonzero"
        assert r_peak < r2f_peak, f"{h}f peak should be smaller than 2f"


class TestWMS2f1fRatio:
    """Tests for the calibration-free WMS 2f/1f ratio (Rieker et al. 2009)."""

    def test_ratio_present_in_simulate_wms(self):
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=0.05,
            sampling_rate_Hz=1e6,
            duration_s=0.01,
            center_wavenumber_cm1=NU0,
        )
        out = simulate_wms(cfg, make_absorbance(1e-2), harmonics=(1, 2))
        assert "ratio_2f1f" in out
        assert out["ratio_2f1f"].shape == out["r_2f"].shape

    def test_ratio_not_present_without_both_harmonics(self):
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=0.05,
            sampling_rate_Hz=1e6,
            duration_s=0.01,
            center_wavenumber_cm1=NU0,
        )
        out = simulate_wms(cfg, make_absorbance(1e-2), harmonics=(2,))
        assert "ratio_2f1f" not in out

    def test_ratio_nonnegative(self):
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=0.05,
            sampling_rate_Hz=1e6,
            duration_s=0.01,
            center_wavenumber_cm1=NU0,
        )
        out = simulate_wms(cfg, make_absorbance(5e-2), harmonics=(1, 2))
        assert np.all(out["ratio_2f1f"] >= 0.0)

    def test_ratio_invariant_to_intensity(self):
        """2f/1f should be approximately invariant to mean laser intensity."""
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=0.05,
            sampling_rate_Hz=2e6,
            duration_s=0.02,
            center_wavenumber_cm1=NU0,
        )
        abs_fn = make_absorbance(1e-2)
        out1 = simulate_wms(cfg, abs_fn, mean_intensity=1.0, harmonics=(1, 2))
        out2 = simulate_wms(cfg, abs_fn, mean_intensity=0.5, harmonics=(1, 2))
        r1 = settled_mean(out1["ratio_2f1f"])
        r2 = settled_mean(out2["ratio_2f1f"])
        assert r1 == pytest.approx(r2, rel=0.01)

    def test_wms_2f_1f_ratio_function(self):
        from spektran.physics.wms import wms_2f_1f_ratio
        r2 = np.array([0.5, 0.3, 0.1, 0.0])
        r1 = np.array([1.0, 0.6, 0.5, 0.0])
        ratio = wms_2f_1f_ratio(r2, r1)
        assert ratio[0] == pytest.approx(0.5)
        assert ratio[1] == pytest.approx(0.5)
        assert ratio[2] == pytest.approx(0.2)
        assert ratio[3] == pytest.approx(0.0, abs=1e-8)


class TestEtalonInWMS:
    """Tests for etalon fringe interaction in the WMS chain."""

    def test_etalon_modifies_wms_output(self):
        """Etalon fringes should alter the demodulated WMS signal."""
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=0.05,
            sampling_rate_Hz=2e6,
            duration_s=0.02,
            center_wavenumber_cm1=NU0,
        )
        abs_fn = make_absorbance(1e-2)
        out_no_et = simulate_wms(cfg, abs_fn, harmonics=(1, 2))

        def etalon_fn(nu):
            return 1.0 + 0.01 * np.cos(2.0 * np.pi * nu / 0.5)

        out_et = simulate_wms(cfg, abs_fn, harmonics=(1, 2),
                              etalon_transmission_of_nu=etalon_fn)
        diff = np.max(np.abs(out_et["x_2f"] - out_no_et["x_2f"]))
        assert diff > 0, "etalon should change the 2f signal"

    def test_no_etalon_matches_default(self):
        """No etalon transmission function should give same result as None."""
        cfg = WMSConfig(
            modulation_frequency_Hz=1e4,
            modulation_depth_cm1=0.05,
            sampling_rate_Hz=1e6,
            duration_s=0.01,
            center_wavenumber_cm1=NU0,
        )
        abs_fn = make_absorbance(1e-2)
        out1 = simulate_wms(cfg, abs_fn, harmonics=(1, 2),
                            etalon_transmission_of_nu=None)
        out2 = simulate_wms(cfg, abs_fn, harmonics=(1, 2))
        np.testing.assert_array_equal(out1["x_2f"], out2["x_2f"])
