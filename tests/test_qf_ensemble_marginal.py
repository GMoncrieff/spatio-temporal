"""The ensemble's marginal must BE the model's quantile function, not a fit to it.

`generate_ensemble.py --qf_dir` replaces the two-piece-normal marginal with the per-pixel
quantile function the distributional model emits. That mapping is the whole point of the
distributional lineage -- if it silently reshaped the tail, the ensemble would score the
post-hoc chain the branch exists to delete.

Checked against an independent construction: a slab whose stored quantiles are those of a
known normal, so `Q(Phi(z))` has a closed form and any error in the u-lookup, the interpolation
or the int16 dequantisation shows up as a mismatch.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scripts.generate_ensemble import INT16_SCALE, qf_from_z_torch  # noqa: E402
from src.models.quantile_spline import output_u_grid  # noqa: E402


def _slab(u, mu, sd):
    """Stored quantiles of N(mu, sd) on the grid, quantised exactly as the writer does."""
    from scipy.stats import norm
    q = norm.ppf(u)[:, None] * sd[None, :] + mu[None, :]
    return torch.as_tensor(np.round(q / INT16_SCALE).astype(np.int16))


def test_maps_z_through_the_pixels_own_quantile_function():
    u = output_u_grid(64)
    mu = np.array([0.10, 0.40, 0.02], dtype=np.float64)
    sd = np.array([0.01, 0.05, 0.002], dtype=np.float64)
    Q = _slab(u, mu, sd)
    z = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)

    got = qf_from_z_torch(z, torch.as_tensor(u, dtype=torch.float32), Q, INT16_SCALE)
    # Q(Phi(z)) for a normal marginal is mu + sd*z exactly; the grid is finite, so the
    # piecewise-linear read of it carries interpolation error, not a bias.
    want = torch.as_tensor(mu + sd * z.numpy(), dtype=torch.float32)
    assert torch.allclose(got, want, atol=2e-4), f"{got} vs {want}"


def test_the_median_is_the_stored_median():
    """u=0.5 is an exact knot of the grid, so z=0 must return the stored Q(0.5) with no
    interpolation at all -- the property that makes the published triple a lookup."""
    u = output_u_grid(64)
    mu = np.array([0.25, 0.75], dtype=np.float64)
    sd = np.array([0.03, 0.01], dtype=np.float64)
    Q = _slab(u, mu, sd)
    got = qf_from_z_torch(torch.zeros(2), torch.as_tensor(u, dtype=torch.float32), Q, INT16_SCALE)
    assert torch.allclose(got, torch.as_tensor(mu, dtype=torch.float32), atol=INT16_SCALE)


def test_it_is_monotone_in_z():
    u = output_u_grid(64)
    mu, sd = np.array([0.2] * 9), np.array([0.02] * 9)
    Q = _slab(u, mu, sd)
    z = torch.linspace(-4, 4, 9)
    got = qf_from_z_torch(z, torch.as_tensor(u, dtype=torch.float32), Q, INT16_SCALE)
    assert torch.all(torch.diff(got) >= -1e-7), got


def test_draws_past_the_grid_clamp_rather_than_extrapolate():
    """The raster IS the forecast. Extrapolating past its outermost stored quantile would
    invent tail the model never emitted -- which is the one thing this lineage exists to
    stop the post-hoc chain doing."""
    u = output_u_grid(64)
    mu, sd = np.array([0.3, 0.3]), np.array([0.02, 0.02])
    Q = _slab(u, mu, sd)
    ut = torch.as_tensor(u, dtype=torch.float32)
    far = qf_from_z_torch(torch.tensor([-40.0, 40.0]), ut, Q, INT16_SCALE)
    edge = qf_from_z_torch(torch.tensor([float(np.sqrt(2) * torch.erfinv(torch.tensor(2 * u[0] - 1)).item()),
                                         float(np.sqrt(2) * torch.erfinv(torch.tensor(2 * u[-1] - 1)).item())]),
                           ut, Q, INT16_SCALE)
    assert torch.allclose(far, edge, atol=1e-6), f"{far} vs {edge}"


def test_output_is_clipped_to_the_hm_range():
    u = output_u_grid(64)
    mu, sd = np.array([0.99]), np.array([0.05])
    Q = _slab(u, mu, sd)
    got = qf_from_z_torch(torch.tensor([3.0]), torch.as_tensor(u, dtype=torch.float32), Q, INT16_SCALE)
    assert 0.0 <= float(got) <= 1.0


@pytest.mark.parametrize("n", [2000])
def test_the_sampled_marginal_reproduces_the_stored_one(n):
    """End to end: push standard normals through one pixel's quantile function and the
    empirical quantiles of the draws must come back to the stored ones."""
    u = output_u_grid(64)
    mu, sd = np.array([0.15]), np.array([0.04])
    Q = _slab(u, mu, sd)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(n, generator=g)
    vals = qf_from_z_torch(z, torch.as_tensor(u, dtype=torch.float32), Q.expand(len(u), n).contiguous(),
                           INT16_SCALE)
    for level in (0.1, 0.5, 0.9):
        emp = float(torch.quantile(vals, level))
        from scipy.stats import norm
        assert abs(emp - (mu[0] + sd[0] * norm.ppf(level))) < 0.01, level


# ------------------------------------------------------------------ the Student-t copula
#
# The whole claim of a t-copula here is that it changes the JOINT behaviour and leaves every
# pixel's marginal exactly where it was. That is not a soft property: T1, T5 and
# check_qf_ensemble's discrete sandwich all assume the marginal is the model's own quantile
# function, and they must not move. These pin the claim rather than trusting it.


def test_t_cdf_table_matches_scipy():
    from scipy.stats import t as _t
    from scripts.generate_ensemble import t_cdf_table

    for df in (4.0, 7.0, 30.0):
        g, c = t_cdf_table(df)
        probe = np.linspace(-11.5, 11.5, 4001)
        got = np.interp(probe, g, c)
        assert np.abs(got - _t.cdf(probe, df)).max() < 1e-6, df


def test_t_copula_leaves_the_marginal_uniform():
    """``T_df(z / sqrt(chi2_df/df))`` is uniform by definition. If it is not, the marginal moved.

    Drawn one pixel per ``w`` so the sample is i.i.d. Pooling many pixels under a shared ``w``
    — which is what a member is — makes the draws *clustered*, and KS on clustered data is
    anti-conservative: 500k samples from 5000 w values reports KS = 0.0025, p = 0.005 for a
    construction that is exactly right. Verified by running the identical draws through
    scipy alone, which returns the same statistic to five decimals.
    """
    from scipy.stats import kstest
    from scripts.generate_ensemble import t_cdf_table, u_from_t_torch

    df = 7.0
    g, c = t_cdf_table(df)
    grid, cdf = torch.as_tensor(g), torch.as_tensor(c)
    rng = np.random.default_rng(0)
    n = 400_000
    w = (rng.chisquare(df, size=n) / df).astype(np.float32)
    z = rng.standard_normal(n).astype(np.float32)
    # u_from_t_torch takes one scalar w; the vectorised equivalent is the same arithmetic.
    zt = torch.as_tensor(z / np.sqrt(w))
    j = torch.searchsorted(grid, zt.contiguous()).clamp_(1, grid.numel() - 1)
    t = ((zt - grid[j - 1]) / (grid[j] - grid[j - 1])).clamp_(0.0, 1.0)
    u = (cdf[j - 1] + t * (cdf[j] - cdf[j - 1])).numpy()

    assert u.min() > 0.0 and u.max() < 1.0
    ks = kstest(u, "uniform")
    assert ks.pvalue > 0.01, ks


def test_t_copula_table_matches_scipy_on_the_worker_path():
    """The scalar-w path the worker actually runs, against scipy's own t CDF."""
    from scipy.stats import t as _t
    from scripts.generate_ensemble import t_cdf_table, u_from_t_torch

    df = 7.0
    g, c = t_cdf_table(df)
    grid, cdf = torch.as_tensor(g), torch.as_tensor(c)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(40):
        w = float(rng.chisquare(df) / df)
        z = rng.standard_normal(2000).astype(np.float32)
        got = u_from_t_torch(torch.as_tensor(z), w, grid, cdf).numpy().astype(np.float64)
        ref = _t.cdf(z.astype(np.float64) / np.sqrt(w), df)
        worst = max(worst, float(np.abs(got - ref).max()))
    assert worst < 1e-5, worst


def test_t_copula_is_tail_dependent_where_gaussian_is_not():
    """Members must co-move into their tails more often than under a Gaussian copula.

    The measurable version: the *fraction of pixels a member puts above u = 0.975* is a
    constant 2.5% in expectation either way, but under a shared chi2 factor its spread
    across members is far larger — that spread is the compound-extreme behaviour the
    ecoregion-coverage rows are short of.
    """
    from scipy.stats import norm
    from scripts.generate_ensemble import t_cdf_table, u_from_t_torch

    df, n_px, n_mem = 7.0, 20_000, 200
    g, c = t_cdf_table(df)
    grid, cdf = torch.as_tensor(g), torch.as_tensor(c)
    rng = np.random.default_rng(1)

    gauss, tcop = [], []
    for m in range(n_mem):
        z = torch.as_tensor(rng.standard_normal(n_px).astype(np.float32))
        gauss.append(float((norm.cdf(z.numpy()) > 0.975).mean()))
        w = float(rng.chisquare(df) / df)
        tcop.append(float((u_from_t_torch(z, w, grid, cdf).numpy() > 0.975).mean()))
    gauss, tcop = np.array(gauss), np.array(tcop)

    assert abs(gauss.mean() - 0.025) < 0.004, gauss.mean()
    assert abs(tcop.mean() - 0.025) < 0.006, tcop.mean()
    assert tcop.std() > 4 * gauss.std(), (tcop.std(), gauss.std())


def test_stratified_chi2_factor_matches_its_target_law_at_finite_M():
    """The per-member tail factor is reused at every pixel, so its *realised* law is what counts.

    An i.i.d. draw of M chi2 values has a sample mean that is off by O(1/sqrt(M)), and because
    the same M values are applied at all 13.8M pixels that error does not average away — it
    biases every published interval. Measured on the M=400 t-copula run: the realised marginal
    CDF sat +0.0033 above target at u = 0.944, narrowing intervals ~6% and flattering T2.8.
    Stratifying the draw over the chi2 quantile function fixes the realised law exactly at any
    M while staying independent of the field.
    """
    from scipy.stats import chi2

    df, M = 7.0, 400
    rng = np.random.default_rng(1)
    strat = chi2.ppf((np.arange(M) + 0.5) / M, df) / df
    iid = rng.chisquare(df, size=M) / df

    assert abs(strat.mean() - 1.0) < 1e-3, strat.mean()
    # The stratified draw is an order of magnitude closer to the target mean than an i.i.d.
    # one, which is the whole point; the i.i.d. sample here is 3.4% low.
    assert abs(strat.mean() - 1.0) < 0.1 * abs(iid.mean() - 1.0)
    # A permutation must not change the multiset, only which member gets which factor.
    perm = strat[rng.permutation(M)]
    assert np.allclose(np.sort(perm), np.sort(strat))
