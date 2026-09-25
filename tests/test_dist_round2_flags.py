"""Round-2 head and loss flags: knot presets, two-sided tail weighting, shape-head capacity.

Each exists to attack one measured defect (docs/dist_scorecard.md), and each defaults to
round-1 behaviour exactly, because the stability gate compares against round-1 replicates and a
default that shifted would silently invalidate them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.crps_loss import crps_spline, tail_weight_fn  # noqa: E402
from src.models.quantile_spline import (  # noqa: E402
    KNOT_PRESETS,
    U_KNOTS_DEFAULT,
    QuantileSpline,
    knot_preset,
    n_spline_params,
)
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402


# ------------------------------------------------------------------ knot presets

def test_default_preset_is_the_round_one_grid():
    assert knot_preset("default14") == U_KNOTS_DEFAULT


@pytest.mark.parametrize("name", sorted(KNOT_PRESETS))
def test_every_preset_is_usable_as_a_quantile_grid(name):
    """0.025 / 0.5 / 0.975 must be exact knots: the published triple is a lookup, not an
    interpolation, and interpolating near a gate quantile leaves a bias no amount of
    ensemble members removes."""
    u = np.asarray(knot_preset(name), dtype=np.float64)
    assert np.all(np.diff(u) > 0), f"{name} is not strictly increasing"
    assert u[0] == 0.0 and u[-1] == 1.0
    for gate in (0.025, 0.5, 0.975):
        assert np.isclose(u, gate).any(), f"{name} lost the {gate} gate"


def test_body_dense_adds_resolution_where_the_mass_is():
    """The defect it targets: three knots between u=0.10 and u=0.90 while 53% of pixels move
    by less than 0.001 over twenty years, and cov50 reads 0.42 at h=5 and 0.63 at h=20."""
    base = np.asarray(knot_preset("default14"))
    dense = np.asarray(knot_preset("body_dense"))
    body = lambda u: ((u > 0.10) & (u < 0.90)).sum()
    assert body(dense) > body(base) + 3
    assert set(base.tolist()).issubset(set(dense.tolist()))


def test_deep_lower_reaches_further_into_the_tail():
    deep = np.asarray(knot_preset("deep_lower"))
    assert deep[1] < np.asarray(knot_preset("default14"))[1]
    assert deep[-2] > np.asarray(knot_preset("default14"))[-2]


@pytest.mark.parametrize("name", sorted(KNOT_PRESETS))
def test_a_spline_on_every_preset_is_monotone(name):
    u = torch.tensor(knot_preset(name), dtype=torch.float64)
    raw = torch.randn(16, n_spline_params(u.numel()), dtype=torch.float64,
                      generator=torch.Generator().manual_seed(0)) * 2.0
    sp = QuantileSpline.from_channels(raw, u, hm_t0=torch.full((16,), 0.3, dtype=torch.float64),
                                      scale_pre=torch.full((16,), 0.05, dtype=torch.float64),
                                      clamp=None)
    q = sp.ppf(torch.linspace(0, 1, 257, dtype=torch.float64), clamp=False)
    assert torch.all(q.diff(dim=-1) > 0)


# ------------------------------------------------------------------ two-sided tail weight

def test_lower_weight_defaults_off_and_changes_nothing():
    u = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    assert tail_weight_fn(u, 0.0) is None
    assert tail_weight_fn(u, 0.0, lam_lo=0.0) is None


def test_lower_weight_is_the_mirror_of_the_upper():
    u = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    hi = tail_weight_fn(u, 4.0, u0=0.95, p=2.0)
    lo = tail_weight_fn(u, 0.0, lam_lo=4.0, u0_lo=0.05, p=2.0)
    torch.testing.assert_close(lo, torch.flip(hi, dims=[0]))


def test_lower_weight_leaks_nowhere_above_its_cutoff():
    u = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    w = tail_weight_fn(u, 0.0, lam_lo=4.0, u0_lo=0.05)
    assert torch.all(w[u >= 0.05] == 1.0)
    assert w[0] == pytest.approx(5.0)


def test_both_sides_compose_additively():
    u = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    both = tail_weight_fn(u, 2.0, lam_lo=3.0)
    assert both[0] == pytest.approx(4.0)      # 1 + 3 on the lower side
    assert both[-1] == pytest.approx(3.0)     # 1 + 2 on the upper side
    assert torch.all(both >= 1.0)


def test_lower_weighting_raises_the_loss_without_touching_the_optimum():
    """A u-weighting scales each pinball term; every term keeps its own minimiser, which is
    why this needs no importance correction where stratified sampling does."""
    u = torch.tensor(U_KNOTS_DEFAULT, dtype=torch.float64)
    raw = torch.randn(32, n_spline_params(u.numel()), dtype=torch.float64,
                      generator=torch.Generator().manual_seed(1))
    sp = QuantileSpline.from_channels(raw, u, hm_t0=torch.full((32,), 0.3, dtype=torch.float64))
    y = torch.full((32,), 0.31, dtype=torch.float64)
    assert float(crps_spline(sp, y, tail_lam_lo=3.0)) > float(crps_spline(sp, y))


# ------------------------------------------------------------------ shape head capacity

B, T, C_D, C_S, HH, WW = 2, 3, 4, 2, 16, 16


def model(**kw):
    m = SpatioTemporalPredictor(
        hidden_dim=16, num_layers=1, num_static_channels=C_S, num_dynamic_channels=C_D,
        use_location_encoder=False, central_residual=True, monotone_quantile_width=True,
        head_family="spline", **kw)
    m.set_norm_stats(0.06, 0.11)
    return m


def n_params(mod):
    return sum(p.numel() for p in mod.parameters())


def test_shape_head_flags_default_to_round_one():
    base = model()
    assert n_params(base.shape_heads) == n_params(model(shape_head_hidden_layers=1,
                                                        shape_head_width=0).shape_heads)


def test_a_deeper_shape_head_has_more_parameters_and_nothing_else_moves():
    base, deep = model(), model(shape_head_hidden_layers=2)
    assert n_params(deep.shape_heads) > n_params(base.shape_heads)
    assert n_params(deep.central_heads) == n_params(base.central_heads)
    assert n_params(deep.width_heads) == n_params(base.width_heads)


def test_a_wider_shape_head_has_more_parameters_and_nothing_else_moves():
    base, wide = model(), model(shape_head_width=16)
    assert n_params(wide.shape_heads) > n_params(base.shape_heads)
    assert n_params(wide.central_heads) == n_params(base.central_heads)


@pytest.mark.parametrize("kw", [{}, {"shape_head_hidden_layers": 2}, {"shape_head_width": 16}])
def test_the_output_contract_is_unchanged_by_head_capacity(kw):
    m = model(**kw)
    out = m(torch.randn(B, T, C_D, HH, WW), torch.randn(B, C_S, HH, WW))
    assert out.shape == (B, 12 + 4 * n_spline_params(m.spline_u_knots.numel()), HH, WW)
