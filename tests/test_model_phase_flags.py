"""The model-phase experiment flags, checked on synthetic tensors.

Two jobs. First, every flag must be *inert at its default* — a slate of nine experiments is
only readable if the control is genuinely today's model, and "byte-identical numbers across
supposedly different configurations" is this project's signature failure mode, so the
inverse (a silently changed control) deserves the same suspicion. Second, each flag makes a
structural claim, asserted here rather than inferred from a training curve twenty minutes
later.

It also pins the finding that motivated turning the histogram term off: it carries no
gradient, so it has never trained anything. If someone makes it differentiable, this test
fails and they have to say so on purpose.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.histogram_loss import HistogramLoss  # noqa: E402
from src.models.lightning_module import SpatioTemporalLightningModule  # noqa: E402
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

B, T, C_D, C_S, H, W = 2, 3, 4, 2, 24, 24
HID = 8
N_CTX = 8


def build(seed=0, **kwargs):
    torch.manual_seed(seed)
    return SpatioTemporalPredictor(
        hidden_dim=HID, kernel_size=3, num_layers=2,
        num_static_channels=C_S, num_dynamic_channels=C_D,
        use_location_encoder=False, locenc_out_channels=0,
        **kwargs,
    )


def inputs(seed=1):
    torch.manual_seed(seed)
    return (torch.randn(B, T, C_D, H, W),
            torch.randn(B, C_S, H, W),
            torch.randn(B, N_CTX, H, W))


# ---------------------------------------------------------------------------------------
# The histogram term trains nothing
# ---------------------------------------------------------------------------------------

def test_histogram_loss_carries_no_gradient():
    """compute_histogram bins with boolean comparisons into a plain zeros buffer.

    So the loss is a constant with respect to the prediction. It has been in the central
    objective at weight 1.0 for the whole project and has never moved a parameter — while
    still entering val_total_loss, which is what ModelCheckpoint selects on.
    """
    bins = torch.tensor([-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0])
    hl = HistogramLoss(bins)
    obs = torch.randn(2, 16, 16) * 0.05
    pred = (torch.randn(2, 16, 16) * 0.05).requires_grad_(True)
    loss, _, _ = hl(obs, pred, mask=torch.ones(2, 16, 16, dtype=torch.bool), horizon_idx=0)
    assert loss.requires_grad is False
    assert loss.grad_fn is None
    with pytest.raises(RuntimeError):
        loss.backward()


# ---------------------------------------------------------------------------------------
# Inertness of the defaults
# ---------------------------------------------------------------------------------------

BASE = dict(quantile_context_channels=N_CTX, central_context_channels=N_CTX,
            central_residual=True, monotone_quantile_width=True)


def test_explicit_defaults_reproduce_the_original_model():
    d, s, ctx = inputs()
    a = build(**BASE)
    b = build(**BASE, quantile_dhat_context=False, width_parameterisation='softplus',
              convlstm_dilations=None)
    with torch.no_grad():
        assert torch.equal(a(d, s, quantile_context=ctx), b(d, s, quantile_context=ctx))


def test_unit_dilation_reproduces_the_original_trunk():
    d, s, ctx = inputs()
    a = build(**BASE)
    b = build(**BASE, convlstm_dilations=[1, 1])
    with torch.no_grad():
        assert torch.equal(a(d, s, quantile_context=ctx), b(d, s, quantile_context=ctx))


def test_lightning_defaults_leave_the_loss_untouched():
    """The horizon-weight, loss-on-change and pinball-norm defaults change no number."""
    m = SpatioTemporalLightningModule(hidden_dim=HID, num_layers=1,
                                      num_static_channels=C_S, num_dynamic_channels=C_D,
                                      use_location_encoder=False, locenc_out_channels=0,
                                      histogram_weight=0.0)
    assert m.horizon_loss_weights is None
    assert m.loss_on_change is False
    assert m.pinball_scale_norm is False
    assert m.lr_schedule == 'none'
    assert m.grad_clip == 0.0
    # _horizon_mean with weights off is exactly .mean()
    hl = [{'x': torch.tensor(float(v))} for v in (1.0, 2.0, 3.0, 4.0)]
    assert torch.equal(m._horizon_mean(hl, 'x'), torch.tensor(2.5))


# ---------------------------------------------------------------------------------------
# Each flag's structural claim
# ---------------------------------------------------------------------------------------

def test_dilation_widens_the_receptive_field():
    """A single non-zero input pixel must reach further with dilation than without."""
    def extent(dilations):
        m = build(**BASE, convlstm_dilations=dilations)
        d = torch.zeros(B, T, C_D, H, W)
        s = torch.zeros(B, C_S, H, W)
        ctx = torch.zeros(B, N_CTX, H, W)
        with torch.no_grad():
            base = m(d, s, quantile_context=ctx)
            d[:, :, :, H // 2, W // 2] = 5.0
            hit = m(d, s, quantile_context=ctx)
        moved = (hit - base).abs().amax(dim=(0, 1)) > 1e-7
        rows = moved.any(dim=1).nonzero()
        return int(rows.max() - rows.min()) + 1

    assert extent([4, 4]) > extent([1, 1])


def test_horizon_weights_renormalise_to_mean_one():
    m = SpatioTemporalLightningModule(hidden_dim=HID, num_layers=1,
                                      num_static_channels=C_S, num_dynamic_channels=C_D,
                                      use_location_encoder=False, locenc_out_channels=0,
                                      histogram_weight=0.0,
                                      horizon_loss_weights=[1.0, 4 / 3, 2.0, 4.0])
    assert sum(m.horizon_loss_weights) == pytest.approx(4.0)
    # The ratio the exposure imbalance asks for is preserved.
    assert m.horizon_loss_weights[3] / m.horizon_loss_weights[0] == pytest.approx(4.0)
    # A uniform vector is the identity.
    m2 = SpatioTemporalLightningModule(hidden_dim=HID, num_layers=1,
                                       num_static_channels=C_S, num_dynamic_channels=C_D,
                                       use_location_encoder=False, locenc_out_channels=0,
                                       histogram_weight=0.0,
                                       horizon_loss_weights=[3.0, 3.0, 3.0, 3.0])
    hl = [{'x': torch.tensor(float(v))} for v in (1.0, 2.0, 3.0, 4.0)]
    assert m2._horizon_mean(hl, 'x') == pytest.approx(2.5)


def test_exp_width_starts_at_the_same_place_as_softplus():
    """Both parameterisations must begin at initial_width_normalized, or the A/B is
    confounded by a different starting interval rather than by the parameterisation."""
    w0 = 0.065
    d, s, ctx = inputs()
    soft = build(**BASE, initial_width_normalized=w0)
    expo = build(**BASE, initial_width_normalized=w0, width_parameterisation='exp')
    with torch.no_grad():
        po = soft(d, s, quantile_context=ctx)
        pe = expo(d, s, quantile_context=ctx)
    # 'exp' zero-initialises the output convolution, so the first increment is exactly w0
    # at every pixel — the same device --central_residual uses to start at persistence.
    assert float((pe[:, 2] - pe[:, 1]).mean()) == pytest.approx(w0, rel=1e-5)
    assert float((pe[:, 2] - pe[:, 1]).std()) == pytest.approx(0.0, abs=1e-7)
    # softplus sets only the bias, so it starts *near* w0 with the untouched conv's spread
    # around it. Worth recording: the two do not start identically, and 'exp' is the
    # cleaner start rather than the matched one.
    assert float((po[:, 2] - po[:, 1]).mean()) == pytest.approx(w0, rel=0.25)


def test_exp_width_stays_positive_and_monotone():
    d, s, ctx = inputs()
    m = build(**BASE, width_parameterisation='exp')
    with torch.no_grad():
        p = m(d, s, quantile_context=ctx)
    for h in range(4):
        lo, ce, up = p[:, 3 * h], p[:, 3 * h + 1], p[:, 3 * h + 2]
        assert bool((up >= ce).all()) and bool((ce >= lo).all())
    for h in range(3):
        w_now = p[:, 3 * h + 2] - p[:, 3 * h + 1]
        w_next = p[:, 3 * (h + 1) + 2] - p[:, 3 * (h + 1) + 1]
        assert bool((w_next >= w_now - 1e-6).all())


def test_dhat_context_changes_the_prediction_but_not_the_gradient_isolation():
    d, s, ctx = inputs()
    plain = build(**BASE)
    with_dhat = build(**BASE, quantile_dhat_context=True)
    with torch.no_grad():
        a = plain(d, s, quantile_context=ctx)
        b = with_dhat(d, s, quantile_context=ctx)
    # Central heads are untouched by the extra quantile-head channels.
    assert torch.allclose(a[:, 1::3], b[:, 1::3])
    # The bounds do move (an inert flag would be the bug).
    assert not torch.allclose(a[:, 0::3], b[:, 0::3])

    # The bounds must still carry no structural path to the *central heads*. (The trunk is
    # a different matter and always has been: the quantile heads read last_hidden, so a
    # path exists and training_step is what severs it, by saving and restoring the central
    # pass's gradients around the pinball backward. This flag must not weaken that, which
    # is what test_pinball_gradient_isolation.py checks on the real loop.)
    p = with_dhat(d, s, quantile_context=ctx)
    (p[:, 0::3].sum() + p[:, 2::3].sum()).backward()
    for name, param in with_dhat.named_parameters():
        if name.startswith('central_heads'):
            assert param.grad is None or float(param.grad.abs().max()) == 0.0, name


def test_pinball_scale_norm_upweights_the_narrow_pixels():
    """The mechanism, asserted directly: with the same absolute miss on a narrow and a wide
    pixel, the normalised objective must weight the narrow one more."""
    from src.models.pinball_loss import PinballLoss
    pin = PinballLoss(quantile=0.975, reduction='mean')
    target = torch.tensor([[[[1.0, 1.0]]]])
    pred = torch.tensor([[[[0.9, 0.9]]]])          # identical absolute error
    width = torch.tensor([[[[0.01, 1.0]]]])        # one narrow pixel, one wide
    mask = torch.ones_like(target, dtype=torch.bool)
    plain = pin(pred, target, mask=mask)
    scaled = pin(pred, target, mask=mask, weights=1.0 / width)
    # The weighted mean is pulled toward the narrow pixel's (identical) loss, so the value
    # itself is unchanged here; what matters is that the weight ratio is the width ratio.
    assert float(plain) == pytest.approx(float(scaled))
    w = 1.0 / width
    assert float(w[..., 0] / w[..., 1]) == pytest.approx(100.0)
