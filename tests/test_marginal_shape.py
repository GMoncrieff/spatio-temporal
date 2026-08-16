"""The empirical marginal shape: does it fix the body without moving the gates?

The two-piece normal pins the 2.5/50/97.5 quantiles and fills between them with Gaussian
mass. The measured residual is far spikier than that — 19-25% of pixels beyond 0.5 sigma
where a Gaussian puts 62%, kurtosis 900-20000 — so members scatter ~3.5x too much moderate
change. `fit_residual_shape` replaces the shape with the residual's own.

Two things have to hold at once, and the tests are split accordingly:

  * the hard gates (T5.1 median == central, T5.2 tails == published bounds) survive
    *exactly*, because they are what the whole ensemble is built to guarantee;
  * the body actually gets spikier, or the change was pointless.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.copula import (  # noqa: E402
    apply_shape_torch,
    invert_shape,
    read_shape_artifact,
    shape_bounds,
    stack_shapes,
    Z975,
    apply_shape,
    fit_marginal_two_piece_normal,
    fit_residual_shape,
    marginal_from_z,
    marginal_ppf,
)


def spiky_residual(n=400_000, seed=0, spike_frac=0.9, spike_scale=0.02):
    """A residual with the measured geometry: a dominant spike plus a broad tail."""
    rng = np.random.default_rng(seed)
    n_spike = int(n * spike_frac)
    spike = rng.normal(0.0, spike_scale, n_spike)
    tail = rng.normal(0.0, 1.0, n - n_spike)
    e = np.concatenate([spike, tail])
    # Normalize so the 2.5/97.5 quantiles sit at -/+ Z975, i.e. the two-piece normal's own
    # units, which is what fit_residual_shape expects.
    lo, hi = np.quantile(e, [0.025, 0.975])
    e = np.where(e < 0, e / abs(lo) * Z975, e / hi * Z975)
    return e


def params_for(n, lower=-1.0, central=0.0, upper=2.0, shape=None):
    p = fit_marginal_two_piece_normal(np.full(n, lower), np.full(n, central),
                                      np.full(n, upper))
    if shape is not None:
        p["shape"] = shape
    return p


# ---------------------------------------------------------------------------- hard gates
def test_median_is_exactly_the_central_forecast():
    shape = fit_residual_shape(spiky_residual())
    p = params_for(5, shape=shape)
    got = marginal_ppf(np.full(5, 0.5), p, clip=None)
    assert np.allclose(got, 0.0, atol=1e-12), got


def test_tails_are_exactly_the_published_bounds():
    """Exactly, not approximately.

    The quantile function is steep near the bounds, so interpolating between knots there
    leaves a *bias* — it does not shrink with member count, and an M=400 run made T5.2
    worse rather than better because the tolerance tightened while the bias did not. The
    gate quantiles are therefore pinned as knots.
    """
    shape = fit_residual_shape(spiky_residual())
    p = params_for(3, lower=-1.0, central=0.0, upper=2.0, shape=shape)
    lo = marginal_ppf(np.full(3, 0.025), p, clip=None)
    hi = marginal_ppf(np.full(3, 0.975), p, clip=None)
    assert np.allclose(lo, -1.0, atol=1e-9), lo
    assert np.allclose(hi, 2.0, atol=1e-9), hi


def test_mapping_is_monotone_so_member_ranks_are_preserved():
    """The copula's spatial structure is carried entirely by the rank of z."""
    shape = fit_residual_shape(spiky_residual())
    z = np.linspace(-4, 4, 4001)
    out = apply_shape(z, shape)
    assert np.all(np.diff(out) >= -1e-9), "shape mapping is not monotone"
    assert apply_shape(np.array([0.0]), shape)[0] == pytest.approx(0.0, abs=1e-12)


def test_extreme_tail_is_extended_not_truncated():
    """Beyond the 95% bound the mapping must keep going, or T6's far tail collapses."""
    shape = fit_residual_shape(spiky_residual())
    far = apply_shape(np.array([-6.0, 6.0]), shape)
    near = apply_shape(np.array([-Z975, Z975]), shape)
    assert far[0] < near[0]
    assert far[1] > near[1]


# ------------------------------------------------------------------------------ the body
def test_body_becomes_spikier_which_is_the_whole_point():
    e = spiky_residual()
    shape = fit_residual_shape(e)
    z = norm.ppf(np.linspace(1e-6, 1 - 1e-6, 200_000))

    gauss = z
    shaped = apply_shape(z, shape)

    def beyond(a, k):
        return float((np.abs(a) > k * Z975).mean())

    # A Gaussian puts far more mass at moderate deviations than the measured residual does.
    assert beyond(shaped, 0.25) < 0.6 * beyond(gauss, 0.25), (
        beyond(shaped, 0.25), beyond(gauss, 0.25))
    # ... while the 95% point itself is unmoved, which is what keeps T5.2 intact.
    assert beyond(shaped, 1.0) == pytest.approx(beyond(gauss, 1.0), abs=0.02)


def test_shape_recovers_the_identity_for_a_gaussian_residual():
    """If the residual really were Gaussian, the change must be a no-op."""
    rng = np.random.default_rng(1)
    e = rng.normal(0.0, 1.0, 400_000)
    lo, hi = np.quantile(e, [0.025, 0.975])
    e = np.where(e < 0, e / abs(lo) * Z975, e / hi * Z975)
    shape = fit_residual_shape(e)
    z = np.linspace(-1.9, 1.9, 999)
    assert np.allclose(apply_shape(z, shape), z, atol=0.08)


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    from src.ensemble.copula import apply_shape_torch

    shape = fit_residual_shape(spiky_residual())
    z = np.linspace(-3.5, 3.5, 1001)
    a = apply_shape(z, shape)
    b = apply_shape_torch(torch.tensor(z, dtype=torch.float64), shape,
                          device=torch.device("cpu"), dtype=torch.float64).numpy()
    assert np.allclose(a, b, atol=1e-6), np.abs(a - b).max()


def test_no_shape_reproduces_the_two_piece_normal_exactly():
    """Existing behaviour must be untouched when no shape is supplied."""
    p = params_for(7, shape=None)
    z = np.linspace(-3, 3, 7)
    assert np.allclose(marginal_from_z(z, p, clip=None),
                       np.where(z < 0, 1.0 / Z975, 2.0 / Z975) * z)


def test_thin_sample_returns_none_rather_than_a_bad_shape():
    assert fit_residual_shape(np.random.default_rng(0).normal(size=100)) is None


# ------------------------------------------------------- Monte-Carlo tolerance correctness
def test_shape_slope_is_one_for_the_two_piece_normal():
    """The gate must score the incumbent family exactly as before."""
    from src.ensemble.copula import shape_slope

    z = np.linspace(-3, 3, 11)
    assert np.allclose(shape_slope(None, z), 1.0)


def test_shape_slope_is_large_at_the_bounds_and_small_at_the_median():
    """This is the factor T5's tolerance was missing.

    The standard error of a sample p-quantile is sqrt(p(1-p)/M)/f(x_p); converting from
    normal-score units to value units costs dx/dz, which is sigma only when the shape map is
    the identity. A spiky marginal is steep at the bounds and flat at the median, so a
    Gaussian-derived tolerance is simultaneously too tight there and too loose here.
    """
    from src.ensemble.copula import shape_slope

    shape = fit_residual_shape(spiky_residual())
    lo, med, hi = shape_slope(shape, np.array([-Z975, 0.0, Z975]))
    assert lo > 1.2, lo
    assert hi > 1.2, hi
    assert med < 0.9, med


def test_shape_slope_recovers_unity_for_a_gaussian_residual():
    rng = np.random.default_rng(2)
    e = rng.normal(0.0, 1.0, 400_000)
    lo_q, hi_q = np.quantile(e, [0.025, 0.975])
    e = np.where(e < 0, e / abs(lo_q) * Z975, e / hi_q * Z975)
    from src.ensemble.copula import shape_slope

    s = shape_slope(fit_residual_shape(e), np.array([-Z975, 0.0, Z975]))
    assert np.allclose(s, 1.0, atol=0.25), s


def test_invert_shape_round_trips():
    """The T3 diagnostics need z back, not S(z)."""
    from src.ensemble.copula import invert_shape

    shape = fit_residual_shape(spiky_residual())
    z = np.linspace(-3.0, 3.0, 601)
    assert np.allclose(invert_shape(apply_shape(z, shape), shape), z, atol=2e-2)
    # And with no shape it must be exactly the identity.
    assert np.allclose(invert_shape(z, None), z)


# --------------------------------------------------------------------------------------
# The tail bound, and the per-band shape it exists for
# --------------------------------------------------------------------------------------
def test_default_bound_is_bit_identical_to_the_two_piece_tail():
    """The regression guard for every existing artifact and every existing scorecard row.

    u_bound became a knob so the far field's real tail could survive. The default has to
    stay exactly what it was, or every number this project has recorded moves underneath
    it. Equality, not allclose: the continuation is written as a shift of z precisely so
    the default offset is 0.0 and this holds bit for bit.
    """
    shape = fit_residual_shape(spiky_residual())
    z = np.random.default_rng(3).normal(size=200_000) * 1.7
    old = np.where(np.abs(z) > Z975, z,
                   np.interp(norm.cdf(z), shape["u"], shape["q"]))
    assert np.array_equal(apply_shape(z, shape), old)
    assert shape_bounds(shape)[2] == 0.0  # offset_hi
    assert shape_bounds(shape)[3] == 0.0  # offset_lo


@pytest.mark.parametrize("u_bound", [0.975, 0.99, 0.999])
def test_bound_keeps_the_map_monotone_continuous_and_invertible(u_bound):
    shape = fit_residual_shape(spiky_residual(), u_bound=u_bound)
    zb = float(norm.ppf(u_bound))
    # The bound and its mirror must be knots, for the same reason the gates are: the
    # mapping changes definition there, and interpolating across it is a standing bias.
    u = np.asarray(shape["u"])
    assert u[np.searchsorted(u, u_bound)] == u_bound
    assert np.diff(u).min() > 1e-12, "near-duplicate knots divide by ~0 on the GPU path"

    z = np.sort(np.concatenate([np.linspace(-6.0, 6.0, 100_001), [zb, -zb]]))
    s = apply_shape(z, shape)
    assert np.all(np.diff(s) > 0), "strict monotonicity is what keeps the copula's ranks"
    eps = 1e-7
    assert abs(apply_shape(zb + eps, shape) - apply_shape(zb - eps, shape)) < 1e-5
    assert np.allclose(invert_shape(s, shape), z, atol=1e-9)


def test_gates_stay_exact_at_every_bound():
    for u_bound in (0.975, 0.99, 0.999):
        shape = fit_residual_shape(spiky_residual(), u_bound=u_bound)
        u, q = np.asarray(shape["u"]), np.asarray(shape["q"])
        for anchor, target in ((0.025, -Z975), (0.5, 0.0), (0.975, Z975)):
            i = int(np.searchsorted(u, anchor))
            assert u[i] == anchor and q[i] == target


def test_raising_the_bound_only_extends_the_tail():
    """It must not touch the body — that is the division of labour with the width level."""
    e = spiky_residual()
    lo = fit_residual_shape(e, u_bound=0.975)
    hi = fit_residual_shape(e, u_bound=0.999)
    body = np.linspace(-1.5, 1.5, 2001)
    assert np.allclose(apply_shape(body, lo), apply_shape(body, hi), atol=1e-12)
    assert apply_shape(3.0, hi) > apply_shape(3.0, lo)


def test_stacked_shapes_match_the_per_band_numpy_path():
    torch = pytest.importorskip("torch")
    e = spiky_residual()
    shapes = [fit_residual_shape(e * (1.0 + 0.4 * k), u_bound=0.99) for k in range(3)]
    shapes.append(None)  # a band too sparse to fit
    stacked = stack_shapes(shapes)
    rng = np.random.default_rng(5)
    z = rng.normal(size=50_000) * 1.6
    band = rng.integers(0, len(shapes), size=z.size)

    got = apply_shape_torch(torch.as_tensor(z, dtype=torch.float64), stacked,
                            band=torch.as_tensor(band), dtype=torch.float64).numpy()
    for k, s in enumerate(shapes):
        m = band == k
        want = z[m] if s is None else apply_shape(z[m], s)
        assert np.allclose(got[m], want, atol=1e-10)
    # An unfitted band must be the two-piece normal exactly, not an interpolated ppf.
    assert np.array_equal(got[band == 3], z[band == 3])


def test_read_shape_artifact_handles_all_three_layouts(tmp_path):
    import json

    shape = fit_residual_shape(spiky_residual())
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"by_horizon": {"20": shape}}))
    table, banded = read_shape_artifact(flat, 6)
    assert not banded and len(table) == 6 and table[(20, 3)] == shape

    banded_path = tmp_path / "banded.json"
    other = fit_residual_shape(spiky_residual(seed=1), u_bound=0.999)
    banded_path.write_text(json.dumps(
        {"by_horizon": {"20": {"pooled": shape, "by_band": {"0": other}}}}))
    table, banded = read_shape_artifact(banded_path, 6)
    assert banded
    assert table[(20, 0)]["u_bound"] == 0.999
    # Bands with no fit of their own inherit pooled, never the identity.
    assert table[(20, 5)]["u_bound"] == 0.975


def test_the_two_tail_bounds_are_independent():
    """The upper tail is too thin and the lower already too hot; one knob cannot serve both.

    Raising both together is what took T6.2/T6.3/T6.5/T8.4 down while T8.1 improved, so the
    asymmetry is the whole point of carrying two bounds rather than one.
    """
    e = spiky_residual()
    sym = fit_residual_shape(e, u_bound=0.999)
    asym = fit_residual_shape(e, u_bound=0.999, u_bound_lo=0.025)
    z_hi, z_lo, off_hi, off_lo = shape_bounds(asym)
    assert off_hi == shape_bounds(sym)[2], "upper side must be untouched by the lower bound"
    assert off_lo == 0.0, "lower bound at 0.025 is the two-piece normal's own tail"
    # Upper tail extended, lower tail exactly the two-piece normal's.
    assert apply_shape(3.5, asym) > apply_shape(3.5, fit_residual_shape(e))
    assert apply_shape(-3.5, asym) == -3.5
    assert apply_shape(-3.5, sym) < -3.5
    z = np.sort(np.concatenate([np.linspace(-6, 6, 100_001), [z_hi, z_lo]]))
    assert np.all(np.diff(apply_shape(z, asym)) > 0)
    assert np.allclose(invert_shape(apply_shape(z, asym), asym), z, atol=1e-9)
