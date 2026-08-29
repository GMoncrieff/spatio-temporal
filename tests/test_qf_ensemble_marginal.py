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
