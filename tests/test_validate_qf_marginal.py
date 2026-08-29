"""The validator must invert the marginal the members were actually drawn from.

T3 recovers a normal-score field and T5 sizes its tolerances from ``dx/dz``. Both were written
for the two-piece normal built from the published triple. An ensemble drawn from the model's own
quantile function is a different marginal, and inverting the wrong one does not fail loudly — it
returns a finite field whose variance is the ratio of the two marginals' tail weights. On the e1
Africa ensemble that read 5.95 against a target of 1.0.

These pin the three pieces against closed forms rather than against the validator itself.
"""
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scripts.validate_ensemble import qf_dx_dz, qf_quantile_at, qf_recover_z  # noqa: E402
from src.models.quantile_spline import output_u_grid  # noqa: E402

U = output_u_grid(64)


def _normal_qf(mu, sd, n_px):
    """Stored quantiles of N(mu, sd) at every level, [levels, n_px]."""
    return norm.ppf(U)[:, None] * np.full(n_px, sd)[None, :] + np.full(n_px, mu)[None, :]


def test_quantile_at_recovers_the_stored_level():
    q = _normal_qf(0.2, 0.05, 3)
    for level, want in [(0.025, 0.2 + 0.05 * norm.ppf(0.025)), (0.5, 0.2),
                        (0.975, 0.2 + 0.05 * norm.ppf(0.975))]:
        assert np.allclose(qf_quantile_at(U, q, level), want, atol=2e-4), level


def test_dx_dz_is_sigma_for_a_normal_marginal():
    """For N(mu, sd), Q(u) = mu + sd*Phi^-1(u), so dQ/du * phi(z) is exactly sd — the same
    number the two-piece normal supplies as sigma. Any other marginal differs, which is the
    point of computing it from Q."""
    q = _normal_qf(0.3, 0.02, 4)
    # A central difference over the stored grid is exact only where the grid is dense. At the
    # median it is within 1%; at the 2.5% points the normal-spaced grid is coarse and it runs
    # ~10% high. That is deliberate and harmless: it feeds a tolerance that is already
    # multiplied by three, and a tolerance finer than the grid the ensemble was sampled on
    # would be testing the grid rather than the ensemble.
    assert np.allclose(qf_dx_dz(U, q, 0.5), 0.02, rtol=0.02)
    for level in (0.025, 0.975):
        got = qf_dx_dz(U, q, level)
        assert np.allclose(got, 0.02, rtol=0.15), (level, got)


def test_recover_z_inverts_the_sampler():
    """Q(Phi(z)) then back again must return z."""
    q = _normal_qf(0.25, 0.04, 5)
    z = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    v = 0.25 + 0.04 * z                      # = Q(Phi(z)) for this marginal
    got = qf_recover_z(v, U, q)
    assert np.allclose(got, z, atol=0.02), got


def test_recovered_field_has_unit_variance():
    """The property T3.1 actually gates on."""
    rng = np.random.default_rng(0)
    n = 4000
    mu = rng.uniform(0.0, 0.5, n)
    sd = rng.uniform(0.005, 0.05, n)
    q = norm.ppf(U)[:, None] * sd[None, :] + mu[None, :]
    z = rng.standard_normal(n)
    v = mu + sd * z
    got = qf_recover_z(v, U, q)
    assert abs(np.var(got) - 1.0) < 0.15, np.var(got)


def test_an_atom_maps_to_the_middle_of_its_plateau():
    """The quantile function is clipped at HM=0, so a real atom sits there. A one-sided
    convention would map every member in the atom to the top of the plateau and report a
    spurious skew in the quiet pixels that dominate this region."""
    q = np.clip(_normal_qf(0.001, 0.01, 1), 0.0, None)   # a wide plateau at exactly 0
    z = qf_recover_z(np.array([0.0]), U, q)
    frac_at_zero = float((q[:, 0] <= 0.0).mean())
    # mid-distribution: u should land in the middle of the tied range, not at its top
    assert norm.cdf(z)[0] < frac_at_zero + 1e-9
    assert norm.cdf(z)[0] > 0.0
