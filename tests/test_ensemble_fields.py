"""Phase 2 tests — field statistics, lon-wrap periodicity, AR(1) coupling."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.fields import (  # noqa: E402
    apply_ar1_horizon_coupling,
    empirical_variogram_from_field,
    generate_correlated_field,
    radial_power_spectrum,
)


def test_unit_variance_and_zero_mean():
    f = generate_correlated_field(256, 256, [3.0, 40.0], [0.6, 0.3], nugget=0.1,
                                  rng=np.random.default_rng(0), wrap_lon=False)
    assert abs(f.mean()) < 1e-5
    assert abs(f.std() - 1.0) < 1e-5


def test_longitude_seam_is_continuous_when_wrapping():
    f = generate_correlated_field(128, 512, [30.0], [1.0], nugget=0.0,
                                  rng=np.random.default_rng(1), wrap_lon=True)
    seam = np.mean(np.abs(f[:, -1] - f[:, 0]))
    interior = np.mean(np.abs(np.diff(f, axis=1)))
    assert seam < 3 * interior, "the +/-180 seam must look like any other column boundary"


def test_no_wrap_leaves_the_row_axis_unlinked():
    """Latitude is not periodic: the top and bottom rows must not be correlated."""
    corrs = []
    for s in range(6):
        f = generate_correlated_field(128, 128, [40.0], [1.0], nugget=0.0,
                                      rng=np.random.default_rng(s), wrap_lon=False)
        corrs.append(np.corrcoef(f[0], f[-1])[0, 1])
    assert abs(np.mean(corrs)) < 0.3


def test_reproducible_from_seed():
    a = generate_correlated_field(64, 64, [5.0], [1.0], 0.0, rng=np.random.default_rng(7),
                                  wrap_lon=False)
    b = generate_correlated_field(64, 64, [5.0], [1.0], 0.0, rng=np.random.default_rng(7),
                                  wrap_lon=False)
    assert np.allclose(a, b)


def test_longer_range_means_flatter_variogram_at_short_lags():
    short = generate_correlated_field(512, 512, [3.0], [1.0], 0.0,
                                      rng=np.random.default_rng(2), wrap_lon=False)
    long = generate_correlated_field(512, 512, [80.0], [1.0], 0.0,
                                     rng=np.random.default_rng(2), wrap_lon=False)
    _, gs, _ = empirical_variogram_from_field(short, max_lag_px=20, seed=0)
    _, gl, _ = empirical_variogram_from_field(long, max_lag_px=20, seed=0)
    assert gl.mean() < gs.mean()


def test_ar1_coupling_reproduces_the_target_correlation():
    rng = np.random.default_rng(0)
    z = {h: generate_correlated_field(256, 256, [15.0], [1.0], 0.0, rng=rng, wrap_lon=False)
         for h in (5, 10, 15, 20)}
    rho = {10: 0.9, 15: 0.8, 20: 0.7}
    out = apply_ar1_horizon_coupling(z, rho)
    for prev, h in ((5, 10), (10, 15), (15, 20)):
        c = np.corrcoef(out[prev].ravel(), out[h].ravel())[0, 1]
        assert abs(c - rho[h]) < 0.05, (prev, h, c)


def test_ar1_preserves_unit_variance():
    rng = np.random.default_rng(0)
    z = {h: generate_correlated_field(128, 128, [10.0], [1.0], 0.0, rng=rng, wrap_lon=False)
         for h in (5, 10)}
    out = apply_ar1_horizon_coupling(z, {10: 0.85})
    assert abs(out[10].std() - 1.0) < 0.05


def test_radial_spectrum_peaks_at_low_wavenumber_for_smooth_fields():
    f = generate_correlated_field(256, 256, [50.0], [1.0], 0.0,
                                  rng=np.random.default_rng(4), wrap_lon=False)
    k, P, _ = radial_power_spectrum(f)
    assert P[0] > P[-1] * 10


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a GPU")
def test_gpu_and_cpu_agree_statistically():
    g = generate_correlated_field(512, 512, [6.0, 60.0], [0.5, 0.4], 0.1,
                                  device="cuda:0", seed=3, wrap_lon=False)
    c = generate_correlated_field(512, 512, [6.0, 60.0], [0.5, 0.4], 0.1,
                                  rng=np.random.default_rng(3), wrap_lon=False)
    _, gg, _ = empirical_variogram_from_field(g, max_lag_px=120, seed=0)
    _, gc, _ = empirical_variogram_from_field(c, max_lag_px=120, seed=0)
    assert np.allclose(gg, gc, rtol=0.25, atol=0.05)




def test_spectral_fit_recovers_a_planted_texture():
    """Fitting the spectrum must reproduce the scale distribution the variogram misses."""
    from src.ensemble.fields import fit_spectral_mixture

    truth = generate_correlated_field(512, 512, [3.0, 40.0], [0.5, 0.2], nugget=0.3,
                                      rng=np.random.default_rng(0), wrap_lon=False,
                                      kernel="matern", nu=0.5)
    fit = fit_spectral_mixture(truth, kernel="matern", nu=0.5)
    sim = generate_correlated_field(512, 512, fit["ranges_px"], fit["weights"], fit["nugget"],
                                    rng=np.random.default_rng(1), wrap_lon=False,
                                    kernel="matern", nu=fit["nu"])

    def shares(f):
        k, _, total = radial_power_spectrum(f)
        total = total / total.sum()
        return np.array([total[(k >= lo) & (k < hi)].sum()
                         for lo, hi in [(0, 0.02), (0.02, 0.1), (0.1, 0.35), (0.35, 0.71)]])

    a, b = shares(truth), shares(sim)
    assert np.all(np.abs(b - a) < 0.06), (a, b)


def test_matern_is_rougher_than_gaussian_at_the_same_ranges():
    """The Gaussian spectrum dies as exp(-k^2) and cannot make fine texture; Matern's
    power-law tail can, and the roughness is ordered by nu."""
    args = dict(ranges_px=[4.0, 40.0], weights=[0.5, 0.5], nugget=0.0, wrap_lon=False)

    def fine_power(**kw):
        f = generate_correlated_field(512, 512, rng=np.random.default_rng(0), **args, **kw)
        k, _, total = radial_power_spectrum(f)
        total = total / total.sum()
        return total[k >= 0.1].sum()          # everything finer than ~10 px

    gauss = fine_power(kernel="gaussian")
    rough = fine_power(kernel="matern", nu=0.5)
    smooth = fine_power(kernel="matern", nu=1.5)
    # nu = 0.5 is decisively rougher; nu = 1.5 is already close enough to Gaussian that
    # the two are within a percent of each other, so only the large gap is asserted.
    assert rough > 1.5 * gauss, (gauss, rough)
    assert rough > 1.5 * smooth, (smooth, rough)
    assert abs(smooth - gauss) < 0.1 * gauss, (gauss, smooth)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
