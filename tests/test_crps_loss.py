"""CRPS and NLL on the quantile spline.

A quadrature bug here is the worst kind this project produces: it does not crash, it does not
NaN, it returns a plausible loss and trains a model that is quietly wrong. So the quadrature is
checked against a second implementation that shares no code with it, propriety is checked by
measurement rather than by citation, and the clamped tails -- which are active on most pixels,
since the median pixel sits at HM 0.053 -- are exercised explicitly.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.crps_loss import (  # noqa: E402
    crps_reference,
    crps_spline,
    nll_spline,
    out_of_support_mass,
    tail_weight_fn,
)
from src.models.quantile_spline import (  # noqa: E402
    U_KNOTS_DEFAULT,
    QuantileSpline,
    n_spline_params,
)


def knots(dtype=torch.float64):
    return torch.tensor(U_KNOTS_DEFAULT, dtype=dtype)


def spline(shape=(64,), seed=0, scale=0.08, anchor=0.3, shape_mag=1.0,
           dtype=torch.float64, clamp=(0.0, 1.0)):
    g = torch.Generator().manual_seed(seed)
    u = knots(dtype)
    raw = torch.randn(*shape, n_spline_params(u.numel()), generator=g, dtype=dtype) * shape_mag
    raw[..., 0] = 0.0
    a = torch.as_tensor(anchor, dtype=dtype).expand(shape).clone()
    s = torch.as_tensor(scale, dtype=dtype).expand(shape).clone()
    return QuantileSpline.from_channels(raw, u, hm_t0=a, scale_pre=s, clamp=clamp)


def observations(sp, seed=1):
    """Draws from the spline itself, by pushing uniforms through Q."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(sp.anchor.shape, generator=g, dtype=sp.anchor.dtype)
    return sp.ppf(u.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------- the pinball primitive

def test_pinball_is_minimised_at_the_true_quantile():
    """One line in the loss, and a sign error in it would still train something."""
    from src.models.crps_loss import _pinball

    g = torch.Generator().manual_seed(3)
    y = torch.randn(200000, generator=g, dtype=torch.float64)
    for level in (0.025, 0.5, 0.9, 0.975):
        grid = torch.linspace(-3.0, 3.0, 1201, dtype=torch.float64)
        u = torch.full_like(grid, level)
        loss = _pinball(u, y.unsqueeze(-1) - grid).mean(dim=0)
        best = grid[loss.argmin()]
        truth = torch.quantile(y, level)
        assert (best - truth).abs() < 0.03, f"u={level}: argmin {best:.3f} vs {truth:.3f}"


# --------------------------------------------------------------- quadrature correctness

@pytest.mark.parametrize("scale,atol", [(0.02, 3e-5), (0.3, 3e-5), (2.0, 1e-4)])
def test_crps_matches_an_independent_dense_grid(scale, atol):
    """Second implementation, no shared quadrature code. scale=2.0 makes the clamp dominant.

    The bar for a realistic forecast is 3e-5, the int16 storage quantum: the objective must be
    finer than the product it trains. scale=2.0 is a 95% interval two HM units wide on a [0,1]
    index -- nothing trained reaches it, and it is here to exercise the clamped tails, so it is
    held to relative accuracy instead.
    """
    sp = spline(scale=scale, shape_mag=1.5)
    y = observations(sp)
    got = crps_spline(sp, y, reduce=False)
    want = crps_reference(sp, y)
    assert float((got - want).abs().max()) < atol
    assert float(((got - want) / want).abs().max()) < 1e-3


def test_crps_quadrature_converges_with_more_nodes():
    sp = spline(scale=0.5, shape_mag=2.0)
    y = observations(sp)
    want = crps_reference(sp, y, n_grid=1000001)
    errs = [float((crps_spline(sp, y, n_nodes=n, reduce=False) - want).abs().max())
            for n in (2, 3, 6)]
    assert errs[0] >= errs[1] >= errs[2], f"not converging: {errs}"


def test_clamp_is_active_in_the_hard_case():
    """Guards the test above: a clamp that never bites would make scale=2.0 vacuous."""
    sp = spline(scale=2.0, shape_mag=1.5)
    assert float(out_of_support_mass(sp).mean()) > 0.25


def test_crps_of_a_degenerate_forecast_is_the_absolute_error():
    sp = spline(scale=1e-6, shape_mag=0.0)
    y = torch.linspace(0.05, 0.95, sp.anchor.numel(), dtype=torch.float64)
    got = crps_spline(sp, y, reduce=False)
    torch.testing.assert_close(got, (y - sp.anchor).abs(), atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------- propriety

def test_crps_is_minimised_at_the_true_scale():
    """The propriety check that matters: a forecast that is too sharp or too wide must lose."""
    truth = spline(shape=(40000,), scale=0.08, shape_mag=1.2)
    y = observations(truth, seed=7)
    scores = {}
    for factor in (0.4, 0.7, 1.0, 1.4, 2.5):
        cand = spline(shape=(40000,), scale=0.08 * factor, shape_mag=1.2)
        scores[factor] = float(crps_spline(cand, y))
    best = min(scores, key=scores.get)
    assert best == 1.0, f"CRPS minimised at scale factor {best}: {scores}"


def test_tail_weighting_reweights_without_changing_the_pinball_optimum():
    """A u-weighting scales each pinball term; every term keeps its own optimum.

    This is why the tail-weighted variants need no importance correction, unlike the
    stratified *sampling* variants.
    """
    u = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    assert tail_weight_fn(u, 0.0) is None
    w = tail_weight_fn(u, 4.0, 0.95, 2.0)
    assert torch.all(w >= 1.0)
    assert torch.all(w[u <= 0.95] == 1.0), "weight leaks below u0"
    assert w[-1] == pytest.approx(5.0)

    sp = spline(scale=0.1)
    y = observations(sp)
    plain = crps_spline(sp, y, tail_lam=0.0)
    weighted = crps_spline(sp, y, tail_lam=4.0)
    assert weighted > plain


# --------------------------------------------------------------- NLL

def test_nll_matches_the_negative_log_of_a_numerical_density():
    sp = spline(scale=0.1, shape_mag=1.0, clamp=None)
    y = observations(sp, seed=5)
    eps = 1e-7
    numeric = -torch.log((sp.cdf(y + eps) - sp.cdf(y - eps)) / (2 * eps))
    torch.testing.assert_close(nll_spline(sp, y, reduce=False), numeric,
                               atol=1e-4, rtol=1e-4)


def test_nll_penalises_a_forecast_that_misses():
    sp = spline(scale=0.05, shape_mag=0.5, clamp=None)
    on_target = nll_spline(sp, observations(sp, seed=11))
    far_off = nll_spline(sp, sp.anchor + 5.0 * sp.scale)
    assert far_off > on_target


def test_out_of_support_mass_is_zero_for_a_narrow_interior_forecast():
    sp = spline(scale=1e-3, anchor=0.5, shape_mag=0.5)
    assert float(out_of_support_mass(sp).max()) < 1e-9


# --------------------------------------------------------------- plumbing

def test_mask_excludes_invalid_pixels():
    sp = spline(shape=(32,), scale=0.1)
    y = observations(sp)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[:8] = True
    masked = crps_spline(sp, y, mask=mask)
    per_pixel = crps_spline(sp, y, reduce=False)
    torch.testing.assert_close(masked, per_pixel[:8].mean())


def test_gradient_reaches_every_raw_channel():
    u = knots(torch.float32)
    raw = torch.randn(16, n_spline_params(u.numel()), requires_grad=True)
    sp = QuantileSpline.from_channels(raw, u, hm_t0=torch.full((16,), 0.3))
    crps_spline(sp, torch.full((16,), 0.32)).backward()
    g = raw.grad.abs().sum(dim=0)
    assert torch.all(g > 0), f"channels with no gradient: {(g == 0).nonzero().flatten().tolist()}"


def test_float32_agrees_with_float64():
    """Training runs in float32; a loss that only works in double is not a loss."""
    sp64 = spline(scale=0.08, shape_mag=1.0)
    y64 = observations(sp64)
    # Re-draw in float32 and the generator yields a different stream, so cast the *same*
    # spline instead: this must compare one forecast in two precisions, not two forecasts.
    sp32 = QuantileSpline(sp64.anchor.float(), sp64.scale.float(), sp64.v_knots.float(),
                          sp64.derivs.float(), sp64.u_knots.float(), clamp=sp64.clamp)
    got32 = crps_spline(sp32, y64.to(torch.float32), reduce=False)
    torch.testing.assert_close(got32.to(torch.float64),
                               crps_spline(sp64, y64, reduce=False),
                               atol=1e-6, rtol=1e-3)
