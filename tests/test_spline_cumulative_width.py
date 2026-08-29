"""`--spline_cumulative_width False` must actually let spread shrink with lead time.

The spline head accumulates non-negative width increments across horizons, so the 95% width
cannot shrink as lead time grows -- T4.2 holds by construction rather than by a later pass.
That is the only monotonicity the model *imposes*. Q(u) increasing in u is a different thing
entirely: it is structural to the rational-quadratic spline (softmax'd bin heights floored at
MIN_BIN_HEIGHT, derivatives floored at MIN_DERIVATIVE) and this flag does not touch it.

These tests pin both halves, because an ablation that silently failed to ablate would read as
"the constraint does not matter" -- the same shape as a lever whose flag never engaged.
"""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402
from src.models.quantile_spline import knot_preset  # noqa: E402


def _model(cumulative):
    m = SpatioTemporalPredictor(
        hidden_dim=8, kernel_size=3, num_layers=1,
        num_static_channels=2, num_dynamic_channels=2,
        use_location_encoder=False, head_family="spline",
        spline_u_knots=torch.tensor(knot_preset("default14"), dtype=torch.float32),
        spline_cumulative_width=cumulative,
    )
    # the spline head refuses to run without the HM normalisation
    m.set_norm_stats(0.085463, 0.153474)
    return m


def _widths(model, seed=0):
    """95% half-width per horizon for one forward pass, [B, n_horizons]."""
    torch.manual_seed(seed)
    B, H, W, T = 2, 16, 16, 3
    dyn = torch.rand(B, T, 2, H, W)
    stat = torch.rand(B, 2, H, W)
    model.eval()
    with torch.no_grad():
        out = model(dyn, stat)
    # channel layout per horizon: [lower, central, upper] x 4 in the first 12 channels
    w = []
    for h in range(4):
        lo, up = out[:, h * 3 + 0], out[:, h * 3 + 2]
        w.append((up - lo).mean(dim=(1, 2)))
    return torch.stack(w, dim=1)


def test_default_is_non_decreasing_in_lead_time():
    m = _model(cumulative=True)
    w = _widths(m)
    d = w[:, 1:] - w[:, :-1]
    assert (d >= -1e-6).all(), (
        f"default must not let width shrink with lead time; got steps {d.tolist()}")


def test_the_flag_is_wired_through_to_the_module():
    assert _model(cumulative=True).spline_cumulative_width is True
    assert _model(cumulative=False).spline_cumulative_width is False


def test_disabling_it_changes_the_widths():
    """The ablation must actually ablate -- identical output would mean the flag does nothing."""
    a = _model(cumulative=True)
    b = _model(cumulative=False)
    b.load_state_dict(a.state_dict())          # same weights, only the constraint differs
    wa, wb = _widths(a), _widths(b)
    assert not torch.allclose(wa, wb), "the flag changed nothing; the ablation is inert"
    # horizon 0 has nothing to accumulate yet, so it must be untouched
    assert torch.allclose(wa[:, 0], wb[:, 0], atol=1e-6), \
        "the first horizon has no accumulation and must be identical"


def test_quantile_monotonicity_survives_either_way():
    """Q(u) increasing in u is structural and must hold with the constraint off."""
    from src.models.quantile_spline import splines_from_output
    for cumulative in (True, False):
        m = _model(cumulative=cumulative)
        torch.manual_seed(3)
        with torch.no_grad():
            out = m(torch.rand(1, 3, 2, 16, 16), torch.rand(1, 2, 16, 16))
        sp = splines_from_output(out, 4, m.spline_u_knots,
                                 learn_slopes=m.spline_learn_slopes, clamp=(0.0, 1.0))
        u = torch.linspace(0.001, 0.999, 64)
        q = sp[3].ppf(u, clamp=False)
        d = q[..., 1:] - q[..., :-1]
        assert (d >= -1e-6).all(), (
            f"cumulative={cumulative}: Q(u) must stay increasing in u regardless")
