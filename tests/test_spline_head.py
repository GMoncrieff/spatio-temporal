"""The distributional head's contract, on synthetic tensors.

The head makes four structural claims, and each one is the sort that a training run would not
reveal for twenty minutes and a scorecard would not reveal at all:

  * the first 12 output channels still carry ``(lower, central, upper) x 4`` in the historical
    order, so every existing reader is unchanged;
  * those channels are *derived from* the spline rather than predicted beside it, so the
    published triple and the quantile function cannot drift apart;
  * the 95% width is non-decreasing in lead time (T4.2), by construction;
  * every quantile lies inside HM's physical range, which requires the normalisation to
    survive a checkpoint round trip.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.quantile_spline import n_spline_params, splines_from_output  # noqa: E402
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

B, T, C_D, C_S, H, W = 2, 3, 4, 2, 24, 24
HID = 8
HM_MEAN, HM_STD = 0.06, 0.11


def model(head_family='spline', **kw):
    m = SpatioTemporalPredictor(
        hidden_dim=HID, num_layers=1, kernel_size=3,
        num_static_channels=C_S, num_dynamic_channels=C_D,
        use_location_encoder=False, central_residual=True,
        monotone_quantile_width=True, head_family=head_family, **kw)
    if head_family == 'spline':
        m.set_norm_stats(HM_MEAN, HM_STD)
    return m


def inputs(hm_level=0.3):
    """Dynamic inputs whose HM channel sits at a realistic normalized level."""
    dyn = torch.randn(B, T, C_D, H, W)
    dyn[:, :, 0] = (hm_level - HM_MEAN) / HM_STD
    return dyn, torch.randn(B, C_S, H, W)


def test_triple_head_is_untouched():
    out = model('triple')(*inputs())
    assert out.shape == (B, 12, H, W)


def test_spline_head_appends_its_parameters_after_the_triple():
    m = model()
    out = m(*inputs())
    assert out.shape == (B, 12 + 4 * n_spline_params(m.spline_u_knots.numel()), H, W)


def test_first_twelve_channels_are_derived_from_the_spline():
    """The triple must be a *view* of the quantile function, not a parallel prediction."""
    m = model()
    out = m(*inputs())
    splines = splines_from_output(out, 4, m.spline_u_knots, clamp=m.spline_clamp())
    for h in range(4):
        lower, _, upper = splines[h].triple()
        torch.testing.assert_close(out[:, 3 * h], lower)
        torch.testing.assert_close(out[:, 3 * h + 2], upper)
        torch.testing.assert_close(out[:, 3 * h + 1], splines[h].mean(
            n_nodes=m.spline_mean_nodes))


def test_central_channel_is_the_mean_not_the_median():
    """Rule 12: the mean is RMSE-optimal and this residual is skewed, so they differ."""
    m = model()
    out = m(*inputs())
    splines = splines_from_output(out, 4, m.spline_u_knots, clamp=m.spline_clamp())
    median = splines[3].anchor
    assert not torch.allclose(out[:, 10], median, atol=1e-7), \
        "central channel is indistinguishable from the median"


def test_interval_brackets_and_widens_with_lead_time():
    m = model()
    out = m(*inputs())
    splines = splines_from_output(out, 4, m.spline_u_knots, clamp=m.spline_clamp())
    widths = torch.stack([s.scale for s in splines])
    assert torch.all(widths.diff(dim=0) >= 0), "95% width shrinks with lead time"
    for h in range(4):
        assert torch.all(out[:, 3 * h] <= out[:, 3 * h + 2])


def test_every_quantile_stays_inside_the_physical_range():
    m = model()
    out = m(*inputs())
    splines = splines_from_output(out, 4, m.spline_u_knots, clamp=m.spline_clamp())
    u = torch.linspace(0.0, 1.0, 129)
    lo, hi = m.spline_clamp()
    for s in splines:
        q = s.ppf(u)
        assert q.min() >= lo - 1e-6 and q.max() <= hi + 1e-6


def test_forward_refuses_to_guess_the_normalisation():
    """An unclamped spline trains happily and emits impossible HM. Fail loudly instead."""
    m = SpatioTemporalPredictor(
        hidden_dim=HID, num_layers=1, num_static_channels=C_S,
        num_dynamic_channels=C_D, use_location_encoder=False, head_family='spline')
    with pytest.raises(RuntimeError, match="set_norm_stats"):
        m(*inputs())


def test_norm_stats_survive_a_state_dict_round_trip():
    m = model()
    fresh = SpatioTemporalPredictor(
        hidden_dim=HID, num_layers=1, num_static_channels=C_S,
        num_dynamic_channels=C_D, use_location_encoder=False,
        central_residual=True, monotone_quantile_width=True, head_family='spline')
    fresh.load_state_dict(m.state_dict())
    assert fresh.spline_clamp() == m.spline_clamp()


def test_fresh_model_starts_at_persistence():
    """The zero-initialised central head means "nothing happens" costs the model nothing.

    E[Q] is not exactly HM_t0 -- the clamp truncates the distribution asymmetrically near the
    physical bounds -- so this asserts the interval is centred there, which is the property
    the residual skip actually buys.
    """
    hm = 0.3
    dyn, stat = inputs(hm_level=hm)
    m = model()
    out = m(dyn, stat)
    splines = splines_from_output(out, 4, m.spline_u_knots, clamp=m.spline_clamp())
    median_hm = splines[0].anchor * HM_STD + HM_MEAN
    torch.testing.assert_close(median_hm, torch.full_like(median_hm, hm),
                               atol=1e-5, rtol=0)


def test_fritsch_slopes_shrink_the_parameter_block():
    m = model(spline_learn_slopes=False)
    out = m(*inputs())
    n_knots = m.spline_u_knots.numel()
    assert out.shape[1] == 12 + 4 * n_spline_params(n_knots, False)
    assert n_spline_params(n_knots, False) < n_spline_params(n_knots, True)
    splines = splines_from_output(out, 4, m.spline_u_knots, learn_slopes=False,
                                  clamp=m.spline_clamp())
    assert torch.all(splines[0].ppf(torch.linspace(0, 1, 65)).diff(dim=-1) >= 0)
