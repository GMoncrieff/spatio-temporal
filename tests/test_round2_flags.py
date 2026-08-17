"""Round-2 flags: substantial architecture and objective changes.

Same two jobs as `test_model_phase_flags.py` — every default must be inert so the control is
genuinely the round-1 winner, and every flag's structural claim is asserted here rather than
inferred from a training curve.

It also pins the algebraic fact that removed an experiment from the slate before it was run:
the Winkler interval score is a positive multiple of the pinball sum, so it is the same
objective and cannot move the optimum.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.histogram_loss import compute_histogram, soft_histogram  # noqa: E402
from src.models.lightning_module import SpatioTemporalLightningModule  # noqa: E402
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

B, T, C_D, C_S, H, W = 2, 3, 4, 2, 24, 24
HID, N_CTX = 8, 8
BASE = dict(quantile_context_channels=N_CTX, central_context_channels=N_CTX,
            central_residual=True, monotone_quantile_width=True)


def build(seed=0, **kwargs):
    torch.manual_seed(seed)
    return SpatioTemporalPredictor(
        hidden_dim=HID, kernel_size=3, num_layers=2,
        num_static_channels=C_S, num_dynamic_channels=C_D,
        use_location_encoder=False, locenc_out_channels=0, **kwargs)


def inputs(seed=1):
    torch.manual_seed(seed)
    return (torch.randn(B, T, C_D, H, W), torch.randn(B, C_S, H, W),
            torch.randn(B, N_CTX, H, W))


# ---------------------------------------------------------------------------------------
# Defaults stay inert
# ---------------------------------------------------------------------------------------

def test_round2_defaults_change_nothing():
    d, s, ctx = inputs()
    a = build(**BASE)
    b = build(**BASE, head_hidden_layers=1, width_head_mode='per_horizon',
              central_target_transform='none')
    with torch.no_grad():
        assert torch.equal(a(d, s, quantile_context=ctx), b(d, s, quantile_context=ctx))


def test_lightning_round2_defaults():
    m = SpatioTemporalLightningModule(hidden_dim=HID, num_layers=1,
                                      num_static_channels=C_S, num_dynamic_channels=C_D,
                                      use_location_encoder=False, locenc_out_channels=0,
                                      histogram_weight=0.0)
    assert m.quantile_loss == 'pinball'
    assert m.histogram_soft is False
    assert m.model.head_hidden_layers == 1
    assert m.model.width_head_mode == 'per_horizon'
    assert m.model.central_target_transform == 'none'


# ---------------------------------------------------------------------------------------
# The Winkler score is not a new objective
# ---------------------------------------------------------------------------------------

def test_interval_score_is_a_multiple_of_the_pinball_sum():
    """Interval score = (u-l) + (2/a)(l-y)1[y<l] + (2/a)(y-u)1[y>u].

    Expanding the two pinball losses at a/2 and 1-a/2 gives exactly a/2 times that, for
    every position of y relative to the interval. So optimising one optimises the other and
    'train on the interval score instead' is the pinball objective at a different learning
    rate. Asserted rather than argued, because it deleted an experiment from the slate.
    """
    alpha = 0.05
    ql, qu = alpha / 2, 1 - alpha / 2
    y = torch.tensor([-3.0, -1.0, 0.0, 0.5, 1.0, 4.0])
    lo, up = torch.full_like(y, -1.0), torch.full_like(y, 1.0)

    def pinball(pred, target, q):
        e = target - pred
        return torch.where(e >= 0, q * e, (q - 1) * e)

    pin = pinball(lo, y, ql) + pinball(up, y, qu)
    iscore = ((up - lo)
              + (2 / alpha) * (lo - y) * (y < lo)
              + (2 / alpha) * (y - up) * (y > up))
    assert torch.allclose(iscore, pin * (2 / alpha), atol=1e-6)


# ---------------------------------------------------------------------------------------
# Structural claims
# ---------------------------------------------------------------------------------------

def test_head_depth_adds_parameters_and_keeps_the_output_shape():
    d, s, ctx = inputs()
    shallow, deep = build(**BASE), build(**BASE, head_hidden_layers=3)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(deep) > n(shallow)
    with torch.no_grad():
        assert deep(d, s, quantile_context=ctx).shape == shallow(d, s, quantile_context=ctx).shape


@pytest.mark.parametrize("mode", ["joint", "power", "power_plus"])
def test_alternative_width_heads_stay_positive_and_monotone(mode):
    d, s, ctx = inputs()
    m = build(**BASE, width_head_mode=mode)
    with torch.no_grad():
        p = m(d, s, quantile_context=ctx)
    for h in range(4):
        lo, ce, up = p[:, 3 * h], p[:, 3 * h + 1], p[:, 3 * h + 2]
        assert bool((up >= ce).all()) and bool((ce >= lo).all()), f"{mode} h={h}"
    for h in range(3):
        w_now = p[:, 3 * h + 2] - p[:, 3 * h + 1]
        w_next = p[:, 3 * (h + 1) + 2] - p[:, 3 * (h + 1) + 1]
        assert bool((w_next >= w_now - 1e-6).all()), f"{mode} not monotone at h={h}"


def test_power_width_starts_linear_in_lead_time():
    """gamma = 1 at init, so w(h) is proportional to h — exactly what four equal cumulative
    increments give. The two modes therefore start at the same place."""
    d, s, ctx = inputs()
    m = build(**BASE, width_head_mode='power', initial_width_normalized=0.065)
    with torch.no_grad():
        p = m(d, s, quantile_context=ctx)
    w = [float((p[:, 3 * h + 2] - p[:, 3 * h + 1]).mean()) for h in range(4)]
    assert w[0] == pytest.approx(0.065, rel=1e-4)
    for h in range(4):
        assert w[h] == pytest.approx(0.065 * (h + 1), rel=1e-3)


def test_joint_and_power_use_one_module_per_side():
    a = build(**BASE)
    for mode in ("joint", "power"):
        m = build(**BASE, width_head_mode=mode)
        assert len(m.lower_heads) == 1 and len(m.upper_heads) == 1
    assert len(a.lower_heads) == 4


def test_asinh_transform_still_starts_at_persistence():
    d, s, ctx = inputs()
    m = build(**BASE, central_target_transform='asinh')
    with torch.no_grad():
        p = m(d, s, quantile_context=ctx)
    hm_t0 = d[:, -1, 0:1]
    for h in range(4):
        assert torch.allclose(p[:, 3 * h + 1:3 * h + 2], hm_t0, atol=1e-6)


def test_soft_histogram_carries_a_gradient_and_tracks_the_hard_one():
    bins = torch.tensor([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
    x = (torch.randn(2, 32, 32) * 0.08).requires_grad_(True)
    mask = torch.ones(2, 32, 32, dtype=torch.bool)
    _, p_soft = soft_histogram(x, bins, mask=mask)
    assert p_soft.requires_grad
    p_soft.sum().backward()
    assert x.grad is not None
    with torch.no_grad():
        _, p_hard = compute_histogram(x, bins, mask=mask)
    # Soft binning must approximate the thing it replaces, not invent a different histogram.
    assert float((p_soft.detach() - p_hard).abs().max()) < 0.05


def test_soft_histogram_survives_nan_pixels():
    """The predictions carry NaN at invalid pixels. The hard version masks by indexing so
    they never reach the arithmetic; the soft one is dense, and NaN * 0 is still NaN.
    Getting this wrong made every prediction NaN after 35 minutes of training."""
    bins = torch.tensor([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
    x = (torch.randn(2, 16, 16) * 0.05)
    mask = torch.ones(2, 16, 16, dtype=torch.bool)
    mask[:, :4] = False
    x[:, :4] = float('nan')
    x = x.requires_grad_(True)
    counts, props = soft_histogram(x, bins, mask=mask)
    assert torch.isfinite(counts).all() and torch.isfinite(props).all()
    props.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_nll_quantile_loss_prefers_the_width_that_matches_the_spread():
    """The log score must be minimised near the scale that generated the data, or it is not
    fitting a density."""
    m = SpatioTemporalLightningModule(hidden_dim=HID, num_layers=1,
                                      num_static_channels=C_S, num_dynamic_channels=C_D,
                                      use_location_encoder=False, locenc_out_channels=0,
                                      histogram_weight=0.0, quantile_loss='nll')
    torch.manual_seed(3)
    sigma = 0.05
    y = torch.randn(1, 1, 64, 64) * sigma
    c = torch.zeros_like(y)
    mask = torch.ones_like(y, dtype=torch.bool)
    losses = {}
    for k in (0.25, 0.5, 1.0, 2.0, 4.0):
        w = torch.full_like(y, k * sigma * m.Z975)
        lo_, up_ = m._two_piece_nll(c - w, c, c + w, y, mask)
        losses[k] = float(lo_ + up_)
    best = min(losses, key=losses.get)
    assert best == 1.0, losses
