"""The neighbourhood context is an ordinary covariate: it enters the trunk and nothing else.

There is no flag for this. ``--trunk_context``, ``--central_context`` and
``--quantile_context`` were removed once the arrangement was settled, so what these tests pin
is the *wiring*, which is the only thing left that could silently drift.

Four things are checked, and the last two are the ones that matter. A context tensor that is
accepted but never influences the trunk's output would pass a shape test and read downstream
as "the covariate does nothing" -- a null that is really a plumbing bug. And a head whose
input width still grew with the context would mean the model was both arrangements at once.
Rule: prove the check fires on a control.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

B, T, H, W, C_CTX = 2, 3, 32, 32, 4
HIDDEN = 8


def build(ctx_channels, **kw):
    return SpatioTemporalPredictor(
        hidden_dim=HIDDEN, num_static_channels=2, num_dynamic_channels=1,
        use_location_encoder=False, context_channels=ctx_channels,
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
    assert plain == 1 + 2 + HIDDEN
    assert with_ctx - plain == C_CTX


def test_no_head_input_widens_with_the_context():
    """The control on the other side: the heads must not receive it a second time.

    e1 fed the heads and never the trunk. If a head's first conv still grew with the
    covariate, the model would carry both arrangements and no comparison could attribute
    either -- and nothing else in the suite would notice, because the shapes stay valid.
    """
    plain, with_ctx = build(0), build(C_CTX)
    for attr in ("central_heads", "lower_heads", "upper_heads"):
        a = getattr(plain, attr)[0][0].in_channels
        b = getattr(with_ctx, attr)[0][0].in_channels
        assert a == b, f"{attr} widened by the context: {a} -> {b}"
    assert with_ctx.central_heads[0][0].in_channels == HIDDEN


def test_trunk_context_changes_the_prediction():
    """The control the shape test cannot give: a different context must move the output.

    Both forwards share the model and the dynamic/static inputs, so anything that moves is
    the context moving through the trunk.
    """
    torch.manual_seed(0)
    model = build(C_CTX).eval()
    dyn, stat, ctx = inputs()
    with torch.no_grad():
        a = model(dyn, stat, context=ctx)
        b = model(dyn, stat, context=torch.rand_like(ctx))
    assert not torch.allclose(a, b), "the trunk ignored its context channels"


def test_missing_or_miscounted_context_is_refused():
    """Refuse rather than substitute zeros: a zeroed covariate trains happily and reads
    downstream as a real result."""
    model = build(C_CTX)
    dyn, stat, ctx = inputs()
    with pytest.raises(RuntimeError, match="context channels but none"):
        model(dyn, stat, context=None)
    with pytest.raises(RuntimeError, match="channels, this model expects"):
        model(dyn, stat, context=ctx[:, :2])
