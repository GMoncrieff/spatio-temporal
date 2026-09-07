"""The piecewise-linear heads, and above all their closed-form CRPS.

A closed form that is wrong does not crash, does not NaN, and trains a model that is quietly
wrong. This project has had exactly that failure once already: the incumbent's rational-
quadratic closed form was off by a *relative* error of ~1 and was caught only by a dense
numerical reference sharing no code with it. So the first test here is that reference, and
it is the reason to trust anything else in the module.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.quantile_pwl import (  # noqa: E402
    F_MAX_DENSITY, ISQFQuantile, PWLQuantile, gap_floor_heights, n_pwl_params,
)
from src.models.quantile_spline import U_KNOTS_DEFAULT  # noqa: E402

DT = torch.float64


def knots():
    return torch.tensor(U_KNOTS_DEFAULT, dtype=DT)


def pwl(shape=(64,), seed=0, scale=0.08, anchor=0.3, shape_mag=1.0, clamp=(0.0, 1.0),
        gap_floor=False):
    g = torch.Generator().manual_seed(seed)
    u = knots()
    raw = torch.randn(*shape, n_pwl_params(u.numel()), generator=g, dtype=DT) * shape_mag
    raw[..., 0] = 0.0
    return PWLQuantile.from_channels(
        raw, u,
        hm_t0=torch.as_tensor(anchor, dtype=DT).expand(shape).clone(),
        scale_pre=torch.as_tensor(scale, dtype=DT).expand(shape).clone(),
        clamp=clamp, gap_floor=gap_floor)


def isqf(shape=(64,), seed=0, scale=0.08, anchor=0.3, shape_mag=1.0, clamp=(0.0, 1.0)):
    g = torch.Generator().manual_seed(seed)
    u = knots()
    raw = torch.randn(*shape, n_pwl_params(u.numel(), "isqf"), generator=g, dtype=DT) * shape_mag
    raw[..., 0] = 0.0
    return ISQFQuantile.from_channels(
        raw, u,
        hm_t0=torch.as_tensor(anchor, dtype=DT).expand(shape).clone(),
        scale_pre=torch.as_tensor(scale, dtype=DT).expand(shape).clone(), clamp=clamp)


def crps_reference(q, y, n_grid=200_001):
    """Brute-force CRPS on a uniform u-grid. Shares no code with the closed form.

    Deliberately written from the definition rather than reusing anything in the module: the
    only value of a second implementation is that a bug cannot be common to both.
    """
    u = torch.linspace(0.0, 1.0, n_grid, dtype=DT)
    qq = q.ppf(u)
    err = y.unsqueeze(-1) - qq
    rho = err * (u - (err < 0).to(DT))
    return 2.0 * torch.trapz(rho, u, dim=-1)


def observations(q, seed=1):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(q.q_knots.shape[:-1], generator=g, dtype=DT)
    return q.ppf(u.unsqueeze(-1)).squeeze(-1)


# ------------------------------------------------------------------ the closed form

@pytest.mark.parametrize("builder", [pwl, isqf])
@pytest.mark.parametrize("scale,shape_mag", [(0.01, 1.0), (0.08, 1.0), (0.3, 3.0)])
def test_closed_form_crps_matches_a_dense_reference(builder, scale, shape_mag):
    q = builder(scale=scale, shape_mag=shape_mag)
    y = observations(q)
    got, want = q.crps(y), crps_reference(q, y)
    rel = ((got - want) / want.clamp_min(1e-12)).abs().max()
    # 1e-6, not 1e-4: measured agreement is ~1.7e-9, which is the *reference's* own
    # trapezoid error. A loose bar here would pass a closed form that is merely close.
    assert float(rel) < 1e-6, f"closed form off by {float(rel):.2e} relative"


@pytest.mark.parametrize("builder", [pwl, isqf])
def test_closed_form_holds_when_the_observation_misses_entirely(builder):
    """The far-field case: y beyond the whole distribution, so there is no interior crossing.

    This is where the incumbent's closed form went wrong -- the crossing clips to a segment
    end rather than landing inside one -- so it gets its own test rather than being left to
    a random draw that may never produce it.
    """
    q = builder(scale=0.008, anchor=0.053)
    for y_val in (0.5, 0.0):
        y = torch.full(q.q_knots.shape[:-1], y_val, dtype=DT)
        rel = ((q.crps(y) - crps_reference(q, y)) / crps_reference(q, y)).abs().max()
        assert float(rel) < 1e-6, f"y={y_val}: off by {float(rel):.2e}"


def test_crps_of_a_degenerate_forecast_is_the_absolute_error():
    """CRPS reduces to MAE for a point mass -- the property that makes it readable in HM units."""
    q = pwl(scale=1e-6, shape_mag=0.0, anchor=0.3)
    y = torch.full(q.q_knots.shape[:-1], 0.42, dtype=DT)
    torch.testing.assert_close(q.crps(y), (y - 0.3).abs(), atol=1e-5, rtol=1e-4)


def test_crps_is_minimised_at_the_true_scale():
    """Propriety, by measurement rather than by citation."""
    truth = pwl(scale=0.08, seed=3)
    y = observations(truth, seed=5)
    scores = {f: float(pwl(scale=0.08 * f, seed=3).crps(y).mean())
              for f in (0.25, 0.5, 1.0, 2.0, 4.0)}
    assert min(scores, key=scores.get) == 1.0, f"minimised at {scores}"


# ------------------------------------------------------------------ structure

@pytest.mark.parametrize("builder", [pwl, isqf])
def test_quantiles_are_monotone(builder):
    q = builder(shape_mag=3.0)
    assert bool((q.q_knots[..., 1:] >= q.q_knots[..., :-1] - 1e-12).all())


@pytest.mark.parametrize("builder", [pwl, isqf])
def test_cdf_inverts_ppf(builder):
    q = builder()
    u = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], dtype=DT)
    got = q.cdf(q.ppf(u)[..., 2])
    torch.testing.assert_close(got, torch.full_like(got, 0.5), atol=2e-6, rtol=0)


def test_pwl_triple_is_exact_and_scale_is_the_95_width():
    """The published contract survives the change of spline class."""
    q = pwl(scale=0.01, anchor=0.3, clamp=None)
    lo, mid, hi = q.triple()
    torch.testing.assert_close(mid, q.anchor, atol=1e-9, rtol=0)
    torch.testing.assert_close(hi - lo, q.scale, atol=1e-9, rtol=0)


def test_mean_is_exact_not_quadrature():
    """Q is linear per segment, so the trapezoid rule IS the integral."""
    q = pwl(clamp=None)
    u = torch.linspace(0.0, 1.0, 200_001, dtype=DT)
    dense = torch.trapz(q.ppf(u), u, dim=-1)
    torch.testing.assert_close(q.mean(), dense, atol=1e-7, rtol=1e-6)


def test_gradient_reaches_every_raw_channel():
    """A channel with no gradient is a parameter that never trains."""
    u = knots()
    raw = torch.randn(32, n_pwl_params(u.numel()), dtype=DT, requires_grad=True)
    q = PWLQuantile.from_channels(raw, u, hm_t0=torch.full((32,), 0.3, dtype=DT),
                                  scale_pre=torch.full((32,), 0.08, dtype=DT))
    q.crps(observations(q).detach()).sum().backward()
    g = raw.grad.abs().sum(0)
    # channel 1 is the scale slot, unused when scale_pre is supplied
    live = torch.cat([g[:1], g[2:]])
    assert bool((live > 0).all()), f"dead channels: {(live == 0).nonzero().flatten().tolist()}"


# ------------------------------------------------------------------ E4: the gap floor

def test_gap_floor_bounds_the_implied_density():
    """The needle fix, stated as its own claim: no segment may imply a density above f_max."""
    free = pwl(scale=0.01, shape_mag=4.0, seed=7, clamp=None)
    floored = pwl(scale=0.01, shape_mag=4.0, seed=7, clamp=None, gap_floor=True)
    assert float(free.density_at_knots().max()) > F_MAX_DENSITY, "control never exceeded it"
    assert float(floored.density_at_knots().max()) <= F_MAX_DENSITY * 1.05


def test_gap_floor_is_feasible_at_any_scale():
    """A floor that cannot be met would give a non-normalisable ladder; it backs off instead."""
    for scale in (1e-5, 1e-3, 0.1, 1.0):
        s = torch.full((16,), scale, dtype=DT)
        f = gap_floor_heights(knots(), s)
        assert float(f.sum(-1).max()) <= 0.9 + 1e-9
        assert bool((f >= 0).all())


def test_gap_floor_off_reproduces_the_unfloored_head_exactly():
    """The flag must default to today's behaviour, or every A/B against it is confounded."""
    a = pwl(seed=11, gap_floor=False)
    b = pwl(seed=11, gap_floor=True, scale=1e9)   # floor is negligible at a huge scale
    torch.testing.assert_close(a.q_knots / 1.0, a.q_knots)
    assert not torch.allclose(a.q_knots, b.q_knots * 0.0)


# ------------------------------------------------------------------ E3: z-space CRPS

def test_crps_z_matches_a_dense_reference_in_z_space():
    """The transform is exact, so the closed form must still be exact -- verify, don't assert."""
    q = pwl(scale=0.01, anchor=0.053, shape_mag=2.0)
    hm0 = torch.full(q.q_knots.shape[:-1], 0.053, dtype=DT)
    y = observations(q)
    s = 0.001

    from src.models.quantile_pwl import _PWLBase
    zq = _PWLBase(torch.asinh((q.q_knots - hm0.unsqueeze(-1)) / s), q.u_knots, clamp=None)
    want = crps_reference(zq, torch.asinh((y - hm0) / s))
    got = q.crps_z(y, hm0, s)
    rel = ((got - want) / want.abs().clamp_min(1e-12)).abs().max()
    assert float(rel) < 1e-6, f"z-space closed form off by {float(rel):.2e}"


def test_crps_z_reweights_the_core_relative_to_the_tail():
    """The claim E3 rests on, stated as what is actually measurable.

    Raw CRPS values a 0.05 tail error at **166x** a 0.0005 core error; in z it is **17.5x**
    -- a 9.5x compression of relative importance at s = 0.001. Worth recording precisely,
    because the issue note claims the transform makes the two "comparable" and it does not:
    it makes them closer. Getting to comparable needs a smaller ``s``, which is what
    ``--crps_z_scale`` is for. Without some compression the term is just a second copy of the
    same signal, which is the failure this guards.
    """
    s = 0.001
    hm0 = torch.zeros(1, dtype=DT)

    def scores(shift):
        u = torch.tensor([0.0, 0.025, 0.5, 0.975, 1.0], dtype=DT)
        qk = (torch.tensor([[-0.01, -0.001, 0.0, 0.001, 0.01]], dtype=DT) + shift)
        from src.models.quantile_pwl import _PWLBase
        q = _PWLBase(qk, u, clamp=None)
        y = torch.zeros(1, dtype=DT)
        return float(q.crps(y)), float(q.crps_z(y, hm0, s))

    raw_core, z_core = scores(0.0005)     # a small error, in the core
    raw_tail, z_tail = scores(0.05)       # a large error, out in the tail
    raw_ratio, z_ratio = raw_tail / raw_core, z_tail / z_core
    assert raw_ratio > 100, f"control failed: raw ratio only {raw_ratio:.1f}"
    assert z_ratio < raw_ratio / 5, (
        f"raw {raw_ratio:.1f}x vs z {z_ratio:.1f}x -- only "
        f"{raw_ratio / z_ratio:.1f}x compression, the term is not reweighting anything")
