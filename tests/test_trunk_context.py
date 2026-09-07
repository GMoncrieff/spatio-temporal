"""The b1 baseline modification: the neighbourhood context reaches the ConvLSTM.

Three things are checked, and the third is the one that matters. A context tensor that is
*accepted* but never influences the trunk's output would pass a shape test and read
downstream as "trunk injection did nothing" -- a null that is really a plumbing bug. Rule:
prove the check fires on a control.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

B, T, H, W, C_CTX = 2, 3, 32, 32, 4


def build(trunk_ctx, **kw):
    return SpatioTemporalPredictor(
        hidden_dim=8, num_static_channels=2, num_dynamic_channels=1,
        use_location_encoder=False, trunk_context_channels=trunk_ctx,
        central_residual=True, **kw)


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, T, 1, H, W, generator=g),
            torch.rand(B, 2, H, W, generator=g),
            torch.rand(B, C_CTX, H, W, generator=g))


def test_trunk_input_width_grows_by_the_context():
    """The cell convolution sees ``input_dim + hidden_dim``; only the first term may move."""
    plain = build(0).convlstm.cell_list[0].conv.in_channels        # 1 dyn + 2 static + 8 hidden
    with_ctx = build(C_CTX).convlstm.cell_list[0].conv.in_channels
    assert plain == 1 + 2 + 8
    assert with_ctx - plain == C_CTX


def test_trunk_context_changes_the_prediction():
    """The control the shape test cannot give: a different context must move the output.

    Both forwards share the model and the dynamic/static inputs, so anything that moves is
    the context moving through the trunk.
    """
    torch.manual_seed(0)
    model = build(C_CTX).eval()
    dyn, stat, ctx = inputs()
    with torch.no_grad():
        a = model(dyn, stat, quantile_context=ctx)
        b = model(dyn, stat, quantile_context=torch.rand_like(ctx))
    assert not torch.allclose(a, b), "the trunk ignored its context channels"


def test_missing_or_miscounted_context_is_refused():
    """Refuse rather than substitute zeros: a zeroed covariate trains happily and reads
    downstream as a real result."""
    model = build(C_CTX)
    dyn, stat, ctx = inputs()
    with pytest.raises(RuntimeError, match="trunk context channels"):
        model(dyn, stat, quantile_context=None)
    with pytest.raises(RuntimeError, match="channels, this model expects"):
        model(dyn, stat, quantile_context=ctx[:, :2])


def test_head_context_still_works_alongside_trunk_context():
    """b1 adds trunk injection; it does not remove the head's copy. Both at once must run."""
    model = build(C_CTX, quantile_context_channels=C_CTX, central_context_channels=C_CTX)
    dyn, stat, ctx = inputs()
    out = model(dyn, stat, quantile_context=ctx)
    assert out.shape == (B, 12, H, W)
