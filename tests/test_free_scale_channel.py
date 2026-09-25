"""--free_scale removes the scale channel, and the gap floor binds where it says it does.

Two changes landed together on 2026-09-16 and each is pinned here against a control, because
a flag that is accepted and inert reads downstream as "the experiment was a null" rather than
"the experiment never ran".

1. **The channel is gone, not merely unread.** Until now --free_scale stopped the decoder
   reading channel 1 while the width head still emitted it: E1b/E2a ran at 16 params and
   E1c/E1d at 18 where section 4.1 advertised 15 and 17, with one dead channel per horizon
   taking no gradient from the quantile path. Simplicity is a scoring criterion in this
   project, so the count has to be the real one.

2. **The density floor had its units inverted**, in both spellings. See
   ``density_floor_width``. The tell was on the scorecard: E4's ``over_f_max_frac_ref_5`` is
   0.7464 on the arm whose purpose is to make a density above ``f_max`` impossible.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.quantile_pwl import (  # noqa: E402
    F_MAX_DENSITY,
    ISQFQuantile,
    PWLQuantile,
    density_floor_width,
    n_pwl_params,
    shape_offset,
)
from src.models.quantile_spline import knot_preset  # noqa: E402
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

HM_STD = 0.15347392857074738


def _u(name="default14"):
    return torch.tensor(knot_preset(name), dtype=torch.float32)


def _model(knots="default14", **kw):
    torch.manual_seed(0)
    m = SpatioTemporalPredictor(
        hidden_dim=8, kernel_size=3, num_layers=1,
        num_static_channels=2, num_dynamic_channels=2,
        use_location_encoder=False, spline_u_knots=_u(knots), **kw)
    m.set_norm_stats(0.08546292036771774, HM_STD)
    return m


def _max_density(q_knots, u):
    """Largest implied density in ABSOLUTE HM, which is the unit f_max is quoted in."""
    dq = (q_knots[..., 1:] - q_knots[..., :-1]) * HM_STD
    return ((u[1:] - u[:-1]) / dq.clamp_min(1e-30)).max().item()


# --------------------------------------------------------------------------------------
# 1. the channel is not allocated
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("family,tails,free,want", [
    ("pwl", False, False, 16),
    ("pwl", False, True, 15),
    ("isqf", False, False, 16),
    ("isqf", True, False, 18),
    ("isqf", False, True, 15),
    ("isqf", True, True, 17),
])
def test_free_scale_drops_exactly_one_parameter(family, tails, free, want):
    """The counts section 4.1 advertises, from the function every caller sizes off."""
    assert n_pwl_params(15, family=family, tails=tails, free_scale=free) == want


@pytest.mark.parametrize("family,tails,free,want", [
    ("pwl", False, False, 16), ("pwl", False, True, 15),
    ("isqf", True, False, 18), ("isqf", True, True, 17),
])
def test_the_model_emits_the_advertised_channel_count(family, tails, free, want):
    """The control is the same arm with the flag off: 16 -> 15 and 18 -> 17, not 16 -> 16."""
    m = _model(head_family=family, isqf_tails=tails, free_scale=free)
    assert m.n_spline_params == want
    # The head that emitted the dead channel is not built at all.
    assert len(m.width_heads) == (0 if free else m.num_horizons)
    assert m.shape_heads[0][-1].out_channels == want - shape_offset(free)
    with torch.no_grad():
        out = m(torch.rand(2, 3, 2, 16, 16), torch.rand(2, 2, 16, 16))
    assert out.shape[1] == 12 + m.num_horizons * want


@pytest.mark.parametrize("family,tails", [("pwl", False), ("isqf", True)])
def test_the_free_scale_head_still_decodes_to_a_usable_quantile_function(family, tails):
    """Removing a channel is only correct if the shape channels did not shift under it."""
    m = _model(head_family=family, isqf_tails=tails, free_scale=True).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 2, 16, 16), torch.rand(2, 2, 16, 16))
    qfs = m._decode(out, m.spline_clamp(), n_triple=12)
    assert len(qfs) == m.num_horizons
    q = qfs[0].ppf(torch.tensor([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]))
    assert torch.isfinite(q).all()
    assert bool((q[..., 1:] >= q[..., :-1] - 1e-6).all())


def test_free_scale_on_the_rational_quadratic_head_is_refused_not_ignored():
    """It was accepted and inert before 2026-09-16, which is the worst of the three states."""
    with pytest.raises(ValueError, match="pwl and isqf"):
        _model(head_family="spline", free_scale=True)


def test_the_two_spellings_of_free_scale_cannot_disagree():
    """free_scale=True with a scale channel is a caller bug, not a branch to take."""
    u = _u()
    raw = torch.randn(8, 2 + u.numel() - 1)
    with pytest.raises(ValueError, match="channel layout"):
        ISQFQuantile.from_channels(raw, u, scale_pre=torch.ones(8), free_scale=True)


# --------------------------------------------------------------------------------------
# 2. the gap floor binds, on both parameterisations
# --------------------------------------------------------------------------------------

def test_density_floor_width_multiplies_by_hm_std():
    """The inverted form was 1/hm_std^2 out; pin the algebra, not just the behaviour."""
    u = _u()
    w = density_floor_width(u, f_max=F_MAX_DENSITY, hm_std=HM_STD)
    dp = u[1:] - u[:-1]
    # density in HM is dp / (w * hm_std); it must equal f_max exactly, not f_max/hm_std.
    assert torch.allclose(dp / (w * HM_STD), torch.full_like(dp, F_MAX_DENSITY), rtol=1e-5)


@pytest.mark.parametrize("cls,kw", [(PWLQuantile, {}),
                                    (ISQFQuantile, dict(tails=False, space="logit"))])
def test_free_scale_gap_floor_caps_the_density_and_the_control_does_not(cls, kw):
    """E2iii / E1iii. Proving the control fences is the half that makes this a test."""
    u = _u()
    torch.manual_seed(7)
    raw = torch.randn(64, 1 + u.numel() - 1) * 0.02
    raw[:, 1 + 3:1 + 8] = 1e-6          # collapse the core: a picket fence by construction
    got = []
    for gap_floor in (False, True):
        o = cls.from_channels(raw, u, scale_pre=None, free_scale=True, clamp=None,
                              gap_floor=gap_floor, hm_std=HM_STD, **kw)
        got.append(_max_density(o.q_knots, u))
    assert got[0] > F_MAX_DENSITY, "control did not fence, so the test proves nothing"
    assert got[1] <= F_MAX_DENSITY * 1.001, f"floor did not bind: {got[1]}"


def test_injected_scale_gap_floor_caps_the_density_too():
    """E4's own path. It capped at ~7020 against a stated 578 until the units were fixed."""
    u = _u()
    torch.manual_seed(7)
    raw = torch.randn(64, 2 + u.numel() - 1) * 0.02
    raw[:, 2 + 3:2 + 8] = -12.0         # softmax -> near-zero heights in the core
    scale = torch.full((64,), 0.02 / HM_STD)
    got = []
    for gap_floor in (False, True):
        o = PWLQuantile.from_channels(raw, u, scale_pre=scale, clamp=None,
                                      gap_floor=gap_floor, hm_std=HM_STD)
        got.append(_max_density(o.q_knots, u))
    assert got[0] > F_MAX_DENSITY, "control did not fence, so the test proves nothing"
    # Conservative rather than exact: this path approximates v_span by 1 on purpose.
    assert got[1] <= F_MAX_DENSITY * 1.001, f"floor did not bind: {got[1]}"


@pytest.mark.parametrize("family,tails", [("pwl", False), ("isqf", True)])
def test_gap_floor_composes_with_free_scale_through_the_model(family, tails):
    """The combination used to raise. It is E2iii's and E1iii's whole mechanism."""
    m = _model(head_family=family, isqf_tails=tails, free_scale=True,
               spline_gap_floor=True).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 2, 16, 16), torch.rand(2, 2, 16, 16))
    q = m._decode(out, m.spline_clamp(), n_triple=12)[0].ppf(
        torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99]))
    assert torch.isfinite(q).all()
    assert bool((q[..., 1:] >= q[..., :-1] - 1e-6).all())


@pytest.mark.parametrize("knots", ["skew14", "skew11"])
def test_the_new_knot_presets_carry_free_scale_and_the_floor(knots):
    """E2i/E2ii/E1i/E1ii change the grid under a head whose channel count follows it."""
    m = _model(knots=knots, head_family="pwl", free_scale=True, spline_gap_floor=True)
    n_knots = len(knot_preset(knots))
    assert m.n_spline_params == n_pwl_params(n_knots, family="pwl", free_scale=True)
    with torch.no_grad():
        out = m(torch.rand(2, 3, 2, 16, 16), torch.rand(2, 2, 16, 16))
    assert out.shape[1] == 12 + m.num_horizons * m.n_spline_params
    q = m._decode(out, m.spline_clamp(), n_triple=12)[0].ppf(
        torch.tensor([0.025, 0.5, 0.975]))
    assert torch.isfinite(q).all()


# --------------------------------------------------------------------------------------
# 3. E1v: the tails on the other family
# --------------------------------------------------------------------------------------

def test_pwl_takes_the_tails_and_they_act_only_in_the_tails():
    """E1v is E1d's tails and transform on E2a's anchor, so the ONE difference left between
    the two heads -- whether the free channel is Q(0.5) or Q(0.0) -- is isolated.

    The interior half is the real assertion, exactly as it is for E1a on isqf: if the tails
    leaked inward, E1v would differ from E2a in two places at once and the A/B would be
    unattributable.
    """
    base = _model(head_family="pwl", free_scale=True).eval()
    tail = _model(head_family="pwl", free_scale=True,
                  isqf_tails=True, isqf_space="neglog").eval()
    assert base.n_spline_params == 15 and tail.n_spline_params == 17
    x, ll = torch.rand(2, 3, 2, 16, 16), torch.rand(2, 2, 16, 16)
    with torch.no_grad():
        qb = base._decode(base(x, ll), base.spline_clamp(), n_triple=12)[0]
        qt = tail._decode(tail(x, ll), tail.spline_clamp(), n_triple=12)[0]
    assert getattr(qt, "_has_tails", False) and not getattr(qb, "_has_tails", False)
    inner = torch.tensor([0.001, 0.005, 0.025, 0.5, 0.975, 0.99, 0.999])
    assert torch.allclose(qb.ppf(inner), qt.ppf(inner), atol=1e-6), \
        "the tails leaked into the spline interior"
    far = torch.tensor([0.99995])
    assert not torch.allclose(qb.ppf(far), qt.ppf(far), atol=1e-9), \
        "the tails changed nothing beyond the outer knot -- accepted and inert"


@pytest.mark.parametrize("space", ["logit", "neglog"])
def test_e1v_and_e1d_differ_only_by_where_persistence_enters(space):
    """Same increments, same unit, same span floor: the ladders are one object up to a
    constant shift. pwl pins Q(0.5) to the location channel, isqf pins Q(0.0) to it, so with
    the zero-init persistence skip E1v starts centred on persistence and E1d starts with its
    whole median above it. Asserting the widths match is what makes that claim checkable.
    """
    u = _u()
    torch.manual_seed(3)
    raw = torch.cat([torch.zeros(256, 1), torch.randn(256, u.numel() - 1) * 0.5,
                     torch.randn(256, 2)], dim=-1)
    hm_t0 = torch.full((256,), 0.30)
    # A real clamp, not None: it is the support the tail transform maps onto. All the
    # values here sit near 0.30, well inside it, so it never binds on this data.
    kw = dict(hm_t0=hm_t0, scale_pre=None, free_scale=True, clamp=(0.0, 1.0),
              hm_std=HM_STD, tails=True, space=space)
    p = PWLQuantile.from_channels(raw, u, **kw)
    i = ISQFQuantile.from_channels(raw, u, **kw)
    lo_p, mid_p, hi_p = p.triple()
    lo_i, mid_i, hi_i = i.triple()
    assert torch.allclose(hi_p - lo_p, hi_i - lo_i, atol=1e-6), "not the same ladder"
    assert torch.allclose(mid_p, hm_t0, atol=1e-6), "pwl did not anchor the median"
    assert (mid_i > hm_t0).all(), "isqf did not anchor the bottom knot"


def test_learned_tails_without_a_clamp_say_why():
    """Prove the guard fires: it replaced a TypeError four frames down on None[0]."""
    u = _u()
    raw = torch.randn(8, 2 + u.numel() - 1 + 2)
    with pytest.raises(ValueError, match="need a clamp"):
        PWLQuantile.from_channels(raw, u, scale_pre=None, free_scale=True, clamp=None,
                                  hm_std=HM_STD, tails=True, space="neglog")
