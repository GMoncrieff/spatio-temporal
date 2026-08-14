"""The three central/quantile head parameterisation flags, checked on synthetic tensors.

Each flag makes a structural claim. These assert the claim directly rather than trusting
the training loss to reveal a broken one twenty minutes later:

  * ``central_residual``   — a fresh model reproduces persistence exactly, and the head
    still receives gradient despite the zero-initialised output convolution.
  * ``monotone_quantile_width`` — spread is non-decreasing in horizon and the interval
    brackets the central forecast at every pixel, for arbitrary inputs.
  * gradient isolation — anchoring the interval to the central forecast must not give the
    pinball loss a path into the trunk or the central heads.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

B, T, C_D, C_S, H, W = 2, 3, 4, 2, 24, 24
HID = 8
N_CTX = 8


def build(**kwargs):
    torch.manual_seed(0)
    return SpatioTemporalPredictor(
        hidden_dim=HID, kernel_size=3, num_layers=1,
        num_static_channels=C_S, num_dynamic_channels=C_D,
        use_location_encoder=False, locenc_out_channels=0,
        **kwargs,
    )


def inputs():
    torch.manual_seed(1)
    return (torch.randn(B, T, C_D, H, W), torch.randn(B, C_S, H, W),
            torch.randn(B, N_CTX, H, W))


def central_channels(pred):
    return pred[:, 1::3]


def test_central_residual_starts_at_exact_persistence():
    model = build(central_residual=True)
    dyn, stat, _ = inputs()
    with torch.no_grad():
        pred = model(dyn, stat)
    hm_t0 = dyn[:, -1, 0:1]
    for h in range(4):
        assert torch.allclose(pred[:, 3 * h + 1:3 * h + 2], hm_t0, atol=1e-6), (
            f"horizon {h} does not start at persistence")


def test_central_residual_output_conv_still_learns():
    """A zero-initialised output conv must still get gradient, or the branch is dead."""
    model = build(central_residual=True)
    dyn, stat, _ = inputs()
    pred = model(dyn, stat)
    central_channels(pred).sum().backward()
    for h_idx, head in enumerate(model.central_heads):
        g = head[-1].weight.grad
        assert g is not None and g.abs().sum() > 0, f"central head {h_idx} output conv is dead"


def test_absolute_parameterisation_is_unchanged_by_default():
    """The default path must not start at persistence — otherwise the flag did nothing."""
    model = build()
    dyn, stat, _ = inputs()
    with torch.no_grad():
        pred = model(dyn, stat)
    hm_t0 = dyn[:, -1, 0:1]
    assert not torch.allclose(pred[:, 1:2], hm_t0, atol=1e-6)


def test_monotone_width_is_non_decreasing_and_brackets_central():
    model = build(quantile_context_channels=N_CTX, monotone_quantile_width=True)
    dyn, stat, ctx = inputs()
    with torch.no_grad():
        pred = model(dyn, stat, quantile_context=ctx)
    lo = pred[:, 0::3]
    ce = pred[:, 1::3]
    up = pred[:, 2::3]
    assert bool((lo <= ce).all()), "lower exceeds central somewhere"
    assert bool((ce <= up).all()), "central exceeds upper somewhere"
    w = up - lo
    for h in range(3):
        assert bool((w[:, h + 1] >= w[:, h] - 1e-7).all()), (
            f"width shrinks from horizon {h} to {h + 1}")


def test_monotone_width_initial_scale_is_sane():
    """softplus(0) would start intervals ~0.11 HM wide; the bias must fix that."""
    model = build(quantile_context_channels=N_CTX, monotone_quantile_width=True,
                  initial_width_normalized=0.065)
    dyn, stat, ctx = inputs()
    with torch.no_grad():
        pred = model(dyn, stat, quantile_context=ctx)
    first_half_width = (pred[:, 2:3] - pred[:, 1:2]).mean().item()
    assert 0.01 < first_half_width < 0.25, first_half_width


def test_anchoring_adds_no_gradient_path_into_the_central_heads():
    """Anchoring the interval to the central forecast must not couple them.

    The trunk is *not* asserted here: the quantile heads have always read trunk features,
    so a quantile-loss gradient reaching the trunk predates this change and is neutralised
    in ``training_step`` by saving and restoring the central-loss gradients. What is new is
    ``upper = central.detach() + w``, and the detach is what this asserts.
    """
    model = build(quantile_context_channels=N_CTX, monotone_quantile_width=True)
    dyn, stat, ctx = inputs()
    pred = model(dyn, stat, quantile_context=ctx)
    (pred[:, 0::3].sum() + pred[:, 2::3].sum()).backward()
    for h_idx, head in enumerate(model.central_heads):
        for name, p in head.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                f"central head {h_idx}.{name} got quantile grad")


def test_central_loss_does_not_reach_the_quantile_heads():
    """The reverse direction: the central objective must leave the interval heads alone."""
    model = build(quantile_context_channels=N_CTX, monotone_quantile_width=True,
                  central_residual=True)
    dyn, stat, ctx = inputs()
    pred = model(dyn, stat, quantile_context=ctx)
    central_channels(pred).sum().backward()
    for heads, tag in ((model.lower_heads, "lower"), (model.upper_heads, "upper")):
        for h_idx, head in enumerate(heads):
            for name, p in head.named_parameters():
                assert p.grad is None or p.grad.abs().sum() == 0, (
                    f"{tag} head {h_idx}.{name} got central grad")


def test_central_context_changes_the_central_prediction():
    model = build(central_context_channels=N_CTX)
    dyn, stat, ctx = inputs()
    with torch.no_grad():
        a = central_channels(model(dyn, stat, quantile_context=ctx))
        b = central_channels(model(dyn, stat, quantile_context=torch.zeros_like(ctx)))
    assert not torch.allclose(a, b), "central heads ignore the context they were given"


def test_missing_context_is_tolerated():
    """Inference paths without a context raster must not crash, they must see zeros."""
    model = build(central_context_channels=N_CTX, quantile_context_channels=N_CTX)
    dyn, stat, ctx = inputs()
    with torch.no_grad():
        pred = model(dyn, stat, quantile_context=None)
    assert pred.shape == (B, 12, H, W)
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize("residual", [False, True])
def test_output_contract_is_12_channels(residual):
    model = build(central_residual=residual, quantile_context_channels=N_CTX,
                  monotone_quantile_width=True)
    dyn, stat, ctx = inputs()
    with torch.no_grad():
        pred = model(dyn, stat, quantile_context=ctx)
    assert pred.shape == (B, 12, H, W)
