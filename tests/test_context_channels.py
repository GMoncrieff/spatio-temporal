"""The context channel budget, which is about to stop being a constant.

``N_CONTEXT_CHANNELS`` was ``len(CONTEXT_RADII) + 3`` = 8, read at two places to size the head
input convolutions. Round 2 makes the radii configurable and adds neighbourhood-HM channels, so
the count becomes a function of flags — and a head sized for one count fed a tensor of another
is the kind of mistake that produces a shape error at best and a silently zeroed covariate at
worst. Both are pinned here.

Every default must reproduce today's eight channels exactly, or the round-1 baseline stops being
reproducible and the stability gate has nothing to compare against.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.change_weights import (  # noqa: E402
    CONTEXT_RADII,
    N_CONTEXT_CHANNELS,
    context_channel_count,
    quantile_context_from_distance,
)

B, H, W = 2, 8, 8


def fields():
    dist = torch.rand(B, 1, H, W) * 200.0
    past = torch.rand(B, 1, H, W) * 0.2
    hm = torch.rand(B, 1, H, W)
    return dist, past, hm


def test_the_default_is_still_eight_channels():
    assert context_channel_count() == 8
    assert N_CONTEXT_CHANNELS == 8
    dist, past, hm = fields()
    assert quantile_context_from_distance(dist, past, hm_now=hm).shape[1] == 8


def test_default_output_is_unchanged_channel_for_channel():
    """The occupancy / log-distance / past-change / hm ordering is a wire format."""
    dist, past, hm = fields()
    out = quantile_context_from_distance(dist, past, hm_now=hm)
    for i, r in enumerate(CONTEXT_RADII):
        torch.testing.assert_close(out[:, i:i + 1], (dist <= float(r)).to(dist.dtype))
    torch.testing.assert_close(out[:, 5:6], torch.log1p(dist.clamp(min=0.0)) / 10.0)
    torch.testing.assert_close(out[:, 6:7], past)
    torch.testing.assert_close(out[:, 7:8], hm)


@pytest.mark.parametrize("radii,stats,hm_radii,expected", [
    ((1, 3, 10, 30, 100), (), (), 8),
    ((3, 30, 100), (), (), 6),
    ((3, 30, 100), ("mean", "max"), (3, 30, 100), 12),      # the round-2 covariate
    ((3, 30, 100), ("mean",), (3, 30, 100), 9),             # E1a, mean only
    ((1, 3, 10, 30, 100), ("mean", "max"), (3, 30, 100), 14),
])
def test_channel_count_matches_the_tensor_it_describes(radii, stats, hm_radii, expected):
    assert context_channel_count(radii, stats, hm_radii) == expected
    dist, past, hm = fields()
    hm_ctx = torch.rand(B, len(stats) * len(hm_radii), H, W) if stats else None
    out = quantile_context_from_distance(dist, past, hm_now=hm, radii=radii,
                                         hm_context=hm_ctx)
    assert out.shape[1] == expected


def test_hm_channels_are_appended_verbatim_and_last():
    """The dataloader selects the bands; the builder must not reorder or rescale them."""
    dist, past, hm = fields()
    hm_ctx = torch.rand(B, 6, H, W)
    out = quantile_context_from_distance(dist, past, hm_now=hm, radii=(3, 30, 100),
                                         hm_context=hm_ctx)
    torch.testing.assert_close(out[:, 6:12], hm_ctx)


def test_a_mismatched_hm_context_width_is_refused():
    """Silently accepting the wrong width is how a covariate gets read off by one band."""
    dist, past, hm = fields()
    with pytest.raises(ValueError, match="hm_context"):
        quantile_context_from_distance(dist, past, hm_now=hm, radii=(3, 30, 100),
                                       hm_context=torch.rand(B, 5, H, W),
                                       hm_stats=("mean", "max"), hm_radii=(3, 30, 100))


def test_missing_hm_context_when_expected_is_refused():
    dist, past, hm = fields()
    with pytest.raises(ValueError, match="hm_context"):
        quantile_context_from_distance(dist, past, hm_now=hm, radii=(3, 30, 100),
                                       hm_context=None,
                                       hm_stats=("mean", "max"), hm_radii=(3, 30, 100))


# ------------------------------------------------------------------ the model-side guard

def test_a_missing_context_raises_instead_of_being_zeroed():
    """The silent-degradation path this round removes.

    ``_with_context`` used to substitute zeros when no context tensor arrived, so a mis-wired
    run trained on a zeroed covariate and looked like a real result. With the channel count now
    variable that is a live hazard rather than a latent one.
    """
    from src.models.spatiotemporal_predictor import SpatioTemporalPredictor

    m = SpatioTemporalPredictor(
        hidden_dim=8, num_layers=1, num_static_channels=2, num_dynamic_channels=4,
        use_location_encoder=False, central_residual=True,
        quantile_context_channels=8, central_context_channels=8)
    with pytest.raises(RuntimeError, match="context"):
        m(torch.randn(2, 3, 4, 16, 16), torch.randn(2, 2, 16, 16))


def test_a_wrong_width_context_raises():
    from src.models.spatiotemporal_predictor import SpatioTemporalPredictor

    m = SpatioTemporalPredictor(
        hidden_dim=8, num_layers=1, num_static_channels=2, num_dynamic_channels=4,
        use_location_encoder=False, central_residual=True,
        quantile_context_channels=12, central_context_channels=12)
    with pytest.raises(RuntimeError, match="12"):
        m(torch.randn(2, 3, 4, 16, 16), torch.randn(2, 2, 16, 16),
          quantile_context=torch.randn(2, 8, 16, 16))


def test_an_eight_channel_checkpoint_is_rejected_by_a_twelve_channel_model():
    """The discriminating control. A check that accepts everything checks nothing."""
    from src.models.spatiotemporal_predictor import SpatioTemporalPredictor

    def build(n):
        return SpatioTemporalPredictor(
            hidden_dim=8, num_layers=1, num_static_channels=2, num_dynamic_channels=4,
            use_location_encoder=False, central_residual=True,
            quantile_context_channels=n, central_context_channels=n)

    old, new = build(8), build(12)
    with pytest.raises(RuntimeError):
        new.load_state_dict(old.state_dict())
    # ...and the matching one loads cleanly, or the test above would pass for the wrong reason.
    build(8).load_state_dict(old.state_dict())
