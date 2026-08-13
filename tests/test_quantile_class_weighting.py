"""Tests for the class-weighted pinball loss and head-only retraining.

The defect being fixed: the upper quantile head decays only 1.9x with distance from past
change while the observed probability of change falls to exactly zero, because a pooled
pinball loss is dominated by a far field that contributes almost no signal about its own
tail.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.change_weights import (  # noqa: E402
    class_balanced_weights,
    distance_band,
    past_change_from_inputs,
)
from src.models.pinball_loss import PinballLoss  # noqa: E402


def test_distance_band_orders_by_proximity_to_past_change():
    past = torch.zeros(1, 1, 64, 64)
    past[0, 0, 32, 32] = 0.5                     # one pixel of past change
    band = distance_band(past, radii=(1, 3, 10, 30))
    assert band[0, 0, 32, 32].item() == 0        # on the seed
    assert band[0, 0, 32, 34].item() == 1        # 2 px away -> within radius 3
    assert band[0, 0, 32, 40].item() == 2        # 8 px -> within radius 10
    assert band[0, 0, 0, 0].item() == 4          # corner, beyond every radius
    # Monotone: never closer-banded as you move away.
    row = band[0, 0, 32, 32:].numpy()
    assert np.all(np.diff(row) >= 0)


def test_weights_balance_the_rare_near_field_against_the_far_field():
    """The far field dominates by count; weighting must equalise their contributions."""
    past = torch.zeros(1, 1, 128, 128)
    past[0, 0, 60:68, 60:68] = 0.5               # a small patch of past change
    mask = torch.ones(1, 1, 128, 128, dtype=torch.bool)
    w = class_balanced_weights(past, mask)

    band = distance_band(past)
    near = (band == 0)
    far = (band == 4)
    assert near.sum() > 0 and far.sum() > 0
    assert far.sum() > near.sum(), "the far field must be the majority here"
    # Total weight per class is what the loss sees; it should be far more even than counts.
    near_total, far_total = w[near].sum().item(), w[far].sum().item()
    count_ratio = far.sum().item() / near.sum().item()
    weight_ratio = far_total / near_total
    assert weight_ratio < count_ratio / 3, (weight_ratio, count_ratio)
    assert abs(w[mask].mean().item() - 1.0) < 1e-4, "mean weight must stay 1"


def test_weighted_pinball_matches_unweighted_when_weights_are_uniform():
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    mask = torch.ones_like(pred, dtype=torch.bool)
    loss = PinballLoss(quantile=0.975)
    a = loss(pred, target, mask=mask)
    b = loss(pred, target, mask=mask, weights=torch.ones_like(pred))
    assert torch.allclose(a, b, atol=1e-7)


def test_weighting_moves_the_fitted_quantile_toward_the_rare_class():
    """A single shared prediction fits the pooled quantile; weighting moves it.

    With the rare class rarer than the tail being fit (1% vs a 2.5% tail), the pooled
    97.5th percentile sits inside the majority spike and the rare class is invisible.
    Balancing the classes makes it visible again.
    """
    torch.manual_seed(0)
    n_far, n_near = 9900, 100
    target = torch.cat([torch.zeros(n_far), torch.full((n_near,), 1.0)])
    weights = torch.cat([torch.ones(n_far), torch.full((n_near,), float(n_far / n_near))])
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = PinballLoss(quantile=0.975)

    def fit(w):
        p = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.05)
        for _ in range(800):
            opt.zero_grad()
            loss(p.expand_as(target), target, mask=mask, weights=w).backward()
            opt.step()
        return float(p.detach())

    pooled = fit(None)
    balanced = fit(weights)
    assert pooled < 0.5, pooled          # majority spike swallows the 1% class
    assert balanced > 0.5, balanced      # balanced, the classes weigh equally


def test_quantile_context_sees_beyond_the_trunk_receptive_field():
    """The head's new input must answer 'is there past change within 100 px', which the
    ~10 px trunk receptive field cannot."""
    from src.models.change_weights import CONTEXT_RADII, quantile_context

    past = torch.zeros(1, 1, 256, 256)
    past[0, 0, 128, 128] = 0.5
    ctx = quantile_context(past, hm_now=torch.zeros_like(past))
    assert ctx.shape[1] == len(CONTEXT_RADII) + 2

    # A pixel 60 px away is outside every radius below 100 and inside the 100 px one.
    far = ctx[0, :, 128, 188]
    for i, r in enumerate(CONTEXT_RADII):
        assert (far[i] > 0) == (r >= 60), (r, float(far[i]))
    # A pixel 200 px away sees nothing at all.
    assert ctx[0, : len(CONTEXT_RADII), 128, 255].sum() == 0


def test_past_change_recovers_raw_units_from_normalised_inputs():
    hm_std = 0.1535
    raw_then, raw_now = 0.10, 0.17
    dyn = torch.zeros(1, 3, 2, 4, 4)
    dyn[:, 0, 0] = raw_then / hm_std             # normalised (mean cancels in the difference)
    dyn[:, -1, 0] = raw_now / hm_std
    got = past_change_from_inputs(dyn, hm_std=hm_std)
    assert torch.allclose(got, torch.full_like(got, raw_now - raw_then), atol=1e-5)


def test_head_only_mode_leaves_the_trunk_and_central_heads_untouched():
    """freeze_trunk must make the published central forecast unchanged by construction."""
    from src.models.lightning_module import SpatioTemporalLightningModule

    torch.manual_seed(0)
    model = SpatioTemporalLightningModule(
        hidden_dim=4, num_static_channels=1, num_dynamic_channels=1, num_layers=1,
        use_location_encoder=False, histogram_weight=0.0, ssim_weight=0.0,
        laplacian_weight=0.0, freeze_trunk=True, quantile_class_weighting='distance',
    )
    model.hm_std = 0.15
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    B, H, W = 2, 32, 32
    batch = {
        'input_dynamic': torch.randn(B, 3, 1, H, W),
        'input_static': torch.randn(B, 1, H, W),
        **{f'target_{h}yr': torch.rand(B, H, W) for h in (5, 10, 15, 20)},
    }
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    model.optimizers = lambda: opt          # stand in for Lightning's accessor
    model.manual_backward = lambda loss, **kw: loss.backward(**kw)
    model.log = lambda *a, **kw: None
    model.trainer = None
    model.training_step(batch, 0)

    moved = {n for n, p in model.named_parameters()
             if not torch.equal(before[n], p.detach())}
    assert moved, "the quantile heads must have moved"
    assert all(("lower_heads" in n or "upper_heads" in n) for n in moved), sorted(moved)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
