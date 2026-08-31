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


# --------------------------------------------------------------- T4.2's population spread
#
# T4.2 asks whether the ensemble gets more uncertain further out. That is a property of the
# marginal's *law*, not of any finite sample, so it is computed without members. It was
# computed by integrating a two-piece normal against N(0,1) even when the members came from
# a quantile function — the same defect these tests were written for one stage over, at the
# one site that had no qf branch.


def _sampler_law(u, q, z):
    """What ``generate_ensemble.qf_from_z_torch`` does, in numpy, for reference.

    Linear interpolation between the two stored levels bracketing ``Phi(z)``, clamped to the
    outermost stored quantile beyond the grid. Shares no code with the closed form.
    """
    uu = norm.cdf(z)
    j = np.clip(np.searchsorted(u, uu), 1, u.size - 1)
    t = np.clip((uu - u[j - 1]) / (u[j] - u[j - 1]), 0.0, 1.0)
    return q[j - 1] + t * (q[j] - q[j - 1])


def test_population_spread_matches_a_draw_from_the_same_law():
    """The closed form must agree with a dense sample of the law it claims to integrate."""
    from scripts.validate_ensemble import qf_population_spread

    rng = np.random.default_rng(0)
    q = _normal_qf(0.2, 0.05, 3)
    q[:, 1] = norm.ppf(U) * 0.2 + 0.5          # a wider pixel
    q[:, 2] = np.clip(norm.ppf(U) * 0.1, 0, 1)  # a pixel with a real atom at 0

    z = rng.standard_normal(400_000)
    got = qf_population_spread(U, q)
    ref = np.array([_sampler_law(U, q[:, i], z).std() for i in range(q.shape[1])])
    assert np.allclose(got, ref, rtol=0.01), f"{got} vs {ref}"


def test_population_spread_recovers_a_normal_sigma():
    """A normal marginal on the stored grid must read back its own sigma."""
    from scripts.validate_ensemble import qf_population_spread

    for sd in (0.01, 0.05, 0.2):
        got = float(qf_population_spread(U, _normal_qf(0.5, sd, 1))[0])
        # The grid truncates at u = 1e-4 (losing tail variance) and joins the levels with
        # chords (adding some back on a convex tail). The net is +0.08%, and it is scale-free
        # for a location-scale family, so it is a property of the forecast as stored rather
        # than of this estimator.
        assert 0.99 <= got / sd <= 1.01, f"sd={sd}: {got / sd}"


def test_population_spread_of_a_degenerate_marginal_is_zero():
    from scripts.validate_ensemble import qf_population_spread

    q = np.full((U.size, 4), 0.31)
    assert np.allclose(qf_population_spread(U, q), 0.0, atol=1e-12)


# ------------------------------------------------- the PIT kernel Phase 2 now shares with T3
#
# `scripts/validate_ensemble.qf_recover_z` recovers a *member's* normal score; Phase 2's
# `--fit_space pit` recovers the *observation's* from the same quantile function. They are the
# same map and there is one implementation, in `src.ensemble.residuals.qf_normal_score`. These
# pin the property that makes the copula's latent well posed at all.


def test_the_validator_and_phase2_share_one_kernel():
    """Two copies of the mid-distribution tie convention would drift. There is one."""
    from src.ensemble.residuals import qf_normal_score

    rng = np.random.default_rng(3)
    q = _normal_qf(0.4, 0.08, 50)
    v = rng.uniform(q[0], q[-1])
    assert np.array_equal(qf_recover_z(v, U, q), qf_normal_score(v, U, q))


def _sampler_law_per_pixel(u, q, z):
    """``_sampler_law`` when every pixel has its own quantile function. ``q`` is [levels, n]."""
    uu = norm.cdf(z)
    j = np.clip(np.searchsorted(u, uu), 1, u.size - 1)
    t = np.clip((uu - u[j - 1]) / (u[j] - u[j - 1]), 0.0, 1.0)
    ar = np.arange(z.size)
    return q[j - 1, ar] + t * (q[j, ar] - q[j - 1, ar])


def test_pit_of_a_draw_from_the_forecast_is_standard_normal():
    """If the observation is drawn from the forecast, its PIT normal score is N(0,1).

    This is the whole justification for fitting the dependence model in PIT space: the
    latent the copula samples has this law by construction. The width-standardised residual
    only has it when the marginal is symmetric, which this one deliberately is not.
    """
    from src.ensemble.residuals import qf_normal_score

    rng = np.random.default_rng(11)
    n = 40_000
    sd = rng.uniform(0.002, 0.05, n)
    loc = rng.uniform(0.0, 0.06, n)
    # Bounded below at 0, which is where this project's atom lives.
    q = np.clip(norm.ppf(U)[:, None] * sd[None, :] + loc[None, :], 0.0, 1.0)
    y = _sampler_law_per_pixel(U, q, rng.standard_normal(n))
    z = qf_normal_score(y, U, q)
    z = z[np.isfinite(z)]
    assert abs(z.mean()) < 0.03, z.mean()
    assert 0.9 < z.var() < 1.1, z.var()


def test_the_width_form_inherits_a_skew_the_pit_form_does_not():
    """The two fit spaces are not the same field, and the difference is the marginal's skew.

    Phase 2 fitted ``(y - central)/sigma`` with sigma from the interval half-widths, which
    reads three of the sixty-four stored levels and assumes the rest is symmetric. On a
    right-skewed marginal that leaves a location error in the field whose spectrum is being
    fitted; on the real e1 residual it is a 0.62-sigma region-wide mean at h=20.
    """
    from src.ensemble.copula import Z975
    from src.ensemble.residuals import qf_normal_score

    rng = np.random.default_rng(5)
    n = 40_000
    zl = norm.ppf(U)
    # A short lower half-width and a long upper one: HM's actual shape near the floor.
    q = np.clip(0.05 + np.where(zl < 0, 0.004, 0.06)[:, None] * zl[:, None], 0.0, 1.0) \
        * np.ones((1, n))
    y = _sampler_law_per_pixel(U, q, rng.standard_normal(n))

    cen = q[U.size // 2]
    sigma = np.maximum(np.where(y >= cen, q[-1] - cen, cen - q[0]), 1e-6) / Z975
    width_form = (y - cen) / sigma
    pit_form = qf_normal_score(y, U, q)

    assert abs(pit_form.mean()) < abs(width_form.mean()), \
        f"pit {pit_form.mean():+.4f} vs width {width_form.mean():+.4f}"
    assert 0.9 < pit_form.var() < 1.1, pit_form.var()
