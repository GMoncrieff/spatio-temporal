"""The monotone quantile spline, checked on synthetic tensors.

Measurement bugs have outnumbered model bugs throughout this project, and a quantile function
is the worst possible place for one: a subtly wrong ``Q`` produces plausible numbers at every
downstream stage and announces itself nowhere. So every structural claim the head makes is
asserted directly, against adversarial parameters rather than tame ones.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.quantile_spline import (  # noqa: E402
    U_KNOTS_DEFAULT,
    QuantileSpline,
    segment_nodes,
    knot_index,
    n_spline_params,
)

SHAPE = (3, 5, 7)


def knots(dtype=torch.float64):
    return torch.tensor(U_KNOTS_DEFAULT, dtype=dtype)


def make(seed=0, magnitude=1.0, dtype=torch.float64, learn_slopes=True, clamp=(0.0, 1.0),
         realistic_loc=False):
    """A batch of splines from random head channels.

    ``realistic_loc`` pins the anchor into [0, 1] and the 95% width into [1e-3, 0.3], which is
    the range a trained model actually occupies, leaving only the *shape* channels adversarial.
    """
    g = torch.Generator().manual_seed(seed)
    u = knots(dtype)
    p = n_spline_params(u.numel(), learn_slopes)
    raw = torch.randn(*SHAPE, p, generator=g, dtype=dtype) * magnitude
    hm0 = torch.rand(*SHAPE, generator=g, dtype=dtype) * 0.6
    scale_pre = None
    if realistic_loc:
        raw[..., 0] = 0.0
        scale_pre = 1e-3 + 0.3 * torch.rand(*SHAPE, generator=g, dtype=dtype)
    return QuantileSpline.from_channels(raw, u, learn_slopes=learn_slopes, hm_t0=hm0,
                                        scale_pre=scale_pre, clamp=clamp)


def test_param_count_matches_the_knot_grid():
    assert n_spline_params(15, True) == 29
    assert n_spline_params(15, False) == 16


def test_knot_index_rejects_a_near_miss():
    # A gate quantile that is *nearly* a knot is a bias that no number of members removes.
    assert knot_index(U_KNOTS_DEFAULT, 0.975) == 11
    with pytest.raises(ValueError):
        knot_index(U_KNOTS_DEFAULT, 0.976)


@pytest.mark.parametrize("magnitude", [0.5, 3.0, 12.0])
def test_shape_is_strictly_increasing_in_u(magnitude):
    """Monotonicity is the whole contract: it is what makes lower <= central <= upper hold.

    Asserted on the *shape* rather than on ``Q``, with the anchor and scale held at realistic
    values. At ``magnitude=12`` the raw scale channel reaches softplus(-17) = 3e-8, and a
    95% interval of 3e-8 added to an anchor of order 1 is below float64 resolution -- so
    ``Q`` comes back numerically constant while the spline underneath is perfectly monotone.
    That is what ``MIN_SCALE`` exists to prevent, and it is tested separately below.
    """
    s = make(magnitude=magnitude, clamp=None, realistic_loc=True)
    u = torch.linspace(0.0, 1.0, 501, dtype=torch.float64)
    q = s.ppf(u, clamp=False)
    assert torch.all(q.diff(dim=-1) > 0), "quantile function is not strictly increasing"


def test_scale_is_floored_so_q_never_collapses_to_a_constant():
    u = knots()
    raw = torch.zeros(2, n_spline_params(u.numel()), dtype=torch.float64)
    raw[:, 1] = -40.0                      # softplus(-40) underflows to 0
    s = QuantileSpline.from_channels(raw, u, clamp=None)
    assert torch.all(s.scale >= 1e-6)
    lower, _, upper = s.triple()
    assert torch.all(upper > lower)


def test_triple_is_a_lookup_and_agrees_with_the_spline():
    s = make(clamp=None)
    lower, central, upper = s.triple()
    direct = s.ppf(torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64), clamp=False)
    torch.testing.assert_close(lower, direct[..., 0])
    torch.testing.assert_close(central, direct[..., 1])
    torch.testing.assert_close(upper, direct[..., 2])


def test_anchor_is_the_median_and_scale_is_the_95pc_width():
    s = make(magnitude=4.0, clamp=None)
    lower, central, upper = s.triple()
    torch.testing.assert_close(central, s.anchor)
    torch.testing.assert_close(upper - lower, s.scale)


def test_clamp_keeps_every_quantile_inside_the_physical_range():
    s = make(magnitude=8.0)
    u = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    q = s.ppf(u)
    assert q.min() >= 0.0 and q.max() <= 1.0


def test_cdf_inverts_ppf():
    s = make(magnitude=2.0, clamp=None)
    u = torch.tensor([0.003, 0.02, 0.11, 0.5, 0.87, 0.98, 0.9993], dtype=torch.float64)
    q = s.ppf(u, clamp=False)
    for j in range(u.numel()):
        back = s.cdf(q[..., j])
        torch.testing.assert_close(back, u[j].expand_as(back), atol=1e-9, rtol=0)


def test_pdf_matches_a_finite_difference_of_ppf():
    """log dQ/du is used by NLL; a sign or a missing normaliser would be invisible in training."""
    s = make(magnitude=1.5, clamp=None)
    u = torch.full(SHAPE, 0.63, dtype=torch.float64)
    eps = 1e-6
    q_hi = s.ppf((u + eps).unsqueeze(-1), clamp=False).squeeze(-1)
    q_lo = s.ppf((u - eps).unsqueeze(-1), clamp=False).squeeze(-1)
    numeric = torch.log((q_hi - q_lo) / (2 * eps))
    torch.testing.assert_close(s.log_dq_du(u), numeric, atol=1e-6, rtol=1e-6)


def test_fritsch_slopes_are_monotone_too():
    s = make(magnitude=5.0, learn_slopes=False, clamp=None)
    u = torch.linspace(0.0, 1.0, 401, dtype=torch.float64)
    assert torch.all(s.ppf(u, clamp=False).diff(dim=-1) > 0)


def _dense_mean(s, n_grid=400001):
    u = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
    return torch.trapz(s.ppf(u), u, dim=-1)


def test_mean_converges_to_a_dense_riemann_integral():
    """E[Q] is the published central forecast, so its quadrature error is a raster artifact."""
    s = make(magnitude=2.0)
    reference = _dense_mean(s)
    errs = [float((s.mean(n_nodes=n) - reference).abs().max()) for n in (4, 8, 16)]
    assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
    assert errs[2] < 1e-6, f"still {errs[2]:.2e} at 16 nodes"


def test_mean_is_exact_below_the_storage_quantum_at_realistic_parameters():
    s = make(magnitude=1.0, realistic_loc=True)
    err = (s.mean() - _dense_mean(s)).abs().max()
    assert err < 3e-5, f"E[Q] off by {float(err):.2e}, above the int16 quantum"


def test_segment_weights_integrate_the_requested_interval():
    k = knots()
    a = torch.tensor([0.0, 0.13, 0.4])
    b = torch.tensor([1.0, 0.9931, 0.4])
    _, w = segment_nodes(k, 4, a.to(torch.float64), b.to(torch.float64))
    torch.testing.assert_close(w.sum(dim=-1), (b - a).to(torch.float64))


def test_segment_nodes_stay_inside_the_requested_interval():
    k = knots()
    a = torch.tensor([0.22], dtype=torch.float64)
    b = torch.tensor([0.61], dtype=torch.float64)
    u, _ = segment_nodes(k, 3, a, b)
    assert u.min() >= a.item() - 1e-12 and u.max() <= b.item() + 1e-12


def test_tail_knots_can_reach_far_beyond_the_95pc_interval():
    """The reason the abscissae are fixed: Q(0.999) must be able to sit many half-widths out.

    This is the far-field defect restated as a capability test. With +0.05 sitting ~12
    half-widths out in the remote band, a head that cannot express a tail reach of that order
    cannot fix it however it is trained.
    """
    u = knots()
    n_bins = u.numel() - 1
    raw = torch.zeros(1, n_spline_params(u.numel()), dtype=torch.float64)
    raw[0, 1] = 3.0                                    # scale
    raw[0, 2 + n_bins - 2] = 9.0                       # the mass into the [0.99, 0.999] bin
    s = QuantileSpline.from_channels(raw, u, clamp=None)
    lo, mid, hi = s.triple()
    reach = (s.ppf(torch.tensor([0.999], dtype=torch.float64), clamp=False)[..., 0] - mid) / (hi - mid)
    assert reach.item() > 12.0, f"tail reach only {reach.item():.1f} half-widths"


def test_gradients_reach_every_parameter_group():
    u = knots(torch.float32)
    p = n_spline_params(u.numel())
    raw = torch.randn(4, p, requires_grad=True)
    s = QuantileSpline.from_channels(raw, u, clamp=None)
    s.ppf(torch.tensor([0.01, 0.5, 0.99])).sum().backward()
    g = raw.grad.abs().sum(dim=0)
    assert g[0] > 0 and g[1] > 0, "anchor or scale gets no gradient"
    assert (g[2:2 + u.numel() - 1] > 0).all(), "some bin height gets no gradient"
    assert (g[2 + u.numel() - 1:] > 0).any(), "no internal slope gets any gradient"


def test_float32_is_accurate_enough_for_the_gate_quantiles():
    """Training runs in float32; the triple must still be exact there."""
    s32 = make(magnitude=3.0, dtype=torch.float32, clamp=None)
    lower, central, upper = s32.triple()
    direct = s32.ppf(torch.tensor([0.025, 0.5, 0.975]), clamp=False)
    torch.testing.assert_close(lower, direct[..., 0], atol=1e-6, rtol=0)
    torch.testing.assert_close(upper, direct[..., 2], atol=1e-6, rtol=0)
