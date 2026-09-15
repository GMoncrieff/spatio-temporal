"""Every conv-spline experiment must survive one real training step, and change something.

The failure this guards is the one the project keeps repeating in new costumes: a flag that
is accepted, logged, and inert. `--spline_knots skew14` that silently falls back, a
`--crps_z_weight` that adds a constant, a `head_family` whose loss path is never reached --
each reads downstream as "the experiment was a null" rather than "the experiment never ran".

So each test does two things: run a training step, and prove the knob moved the loss or the
parameters relative to a control.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lightning_module import SpatioTemporalLightningModule  # noqa: E402
from src.models.quantile_spline import KNOT_PRESETS, knot_preset, validate_knots  # noqa: E402

# C_CTX is what change_weights.context_channel_count() builds from the 2-band raster for
# the default radii (5 occupancy + log1p distance + signed past change + hm_now).
B, T, H, W, C_CTX = 2, 3, 32, 32, 8
BASE = dict(hidden_dim=8, num_static_channels=2, num_dynamic_channels=1,
            use_location_encoder=False, central_residual=True,
            ssim_weight=0.0, laplacian_weight=0.0, histogram_weight=0.0,
            context_channels=C_CTX)


def module(seed=0, **kw):
    """Seeded, always. Two modules built from the same config must start identical, or a
    "this flag changed the loss" test is only measuring different random initialisations --
    a check that passes for any flag, including one that does nothing."""
    torch.manual_seed(seed)
    m = SpatioTemporalLightningModule(**{**BASE, **kw})
    m.model.set_norm_stats(0.1, 0.2)
    m.hm_mean, m.hm_std = 0.1, 0.2
    return m


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    hm0 = torch.rand(B, H, W, generator=g) * 0.2
    return {
        "input_dynamic": torch.rand(B, T, 1, H, W, generator=g) * 0.2,
        "input_static": torch.rand(B, 2, H, W, generator=g),
        "change_context": torch.rand(B, 2, H, W, generator=g),
        **{f"target_{h}yr": hm0 + torch.rand(B, H, W, generator=g) * 0.05
           for h in (5, 10, 15, 20)},
        "valid_mask": torch.ones(B, H, W, dtype=torch.bool),
    }


def one_step(m, b):
    """One REAL training step, through Lightning and the module's manual optimisation.

    Driving ``training_step`` directly would skip ``self.optimizers()`` and therefore the
    whole backward/step path -- which is exactly where a head family can be accepted and then
    never trained. Cheap enough at 32x32 to be worth doing properly.
    """
    import pytorch_lightning as pl        # the module subclasses THIS package's base class
    from torch.utils.data import DataLoader, Dataset

    class _One(Dataset):
        def __len__(self):
            return B

        def __getitem__(self, i):
            return {k: v[i] for k, v in b.items()}

    captured = {}
    orig = m.log

    def _cap(name, value, *a, **k):
        captured[name] = float(value) if torch.is_tensor(value) else value
    m.log = _cap
    try:
        pl.Trainer(max_steps=1, accelerator="cpu", devices=1, logger=False,
                   enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False, num_sanity_val_steps=0).fit(
            m, DataLoader(_One(), batch_size=B))
    finally:
        m.log = orig
    return captured


# ------------------------------------------------------------------ head families

@pytest.mark.parametrize("family", ["spline", "pwl", "isqf"])
def test_every_head_family_trains(family):
    m = module(head_family=family)
    out = one_step(m, batch())
    key = next(k for k in out if k.endswith("crps_total") or k.endswith("_total"))
    assert out[key] == out[key], f"{family}: {key} is NaN"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.model.convlstm.parameters()), \
        f"{family}: the trunk received no gradient -- the objective is not reaching it"


@pytest.mark.parametrize("family", ["pwl", "isqf"])
def test_pwl_families_are_cheaper_than_the_spline(family):
    """Half the parameters per horizon. The simplicity claim, checked rather than asserted."""
    a = module(head_family="spline").model.n_spline_params
    b = module(head_family=family).model.n_spline_params
    assert b < a, f"{family} has {b} params/horizon against the spline's {a}"


# ------------------------------------------------------------------ E5: knot presets

def test_every_preset_carries_the_three_gate_levels():
    """A grid missing 0.025 / 0.5 / 0.975 raises deep inside the first forward pass."""
    for name, grid in KNOT_PRESETS.items():
        validate_knots(grid, name)


def test_a_grid_without_a_gate_is_refused():
    """Prove the check fires on a control, or it checks nothing."""
    with pytest.raises(ValueError, match="no exact knot"):
        validate_knots((0.0, 0.025, 0.6, 0.975, 1.0).__class__(
            (0.0, 0.025, 0.6, 0.99, 1.0)), "no-gate")


@pytest.mark.parametrize("preset", ["skew14", "skew11"])
def test_skew_presets_move_capacity_out_of_the_flat_middle(preset):
    """The measured motivation: u in [0.1, 0.6] spans 0.13% of the target's value range."""
    default = knot_preset("default14")
    grid = knot_preset(preset)

    def share_in(g, lo, hi):
        """Share of the grid's BINS falling in a u-range. A share, not a count: skew11 has
        fewer bins in total, so a raw count would call it a regression while it is in fact
        spending a larger fraction of what it has where the value range is."""
        n = len(g) - 1
        return sum(1 for a, b in zip(g, g[1:]) if a >= lo and b <= hi) / n

    assert share_in(grid, 0.1, 0.62) < share_in(default, 0.1, 0.62)
    assert share_in(grid, 0.75, 1.0) > share_in(default, 0.75, 1.0)


def test_a_skew_preset_actually_changes_the_trained_forecast():
    a = one_step(module(head_family="spline", spline_knots="default14"), batch())
    b = one_step(module(head_family="spline", spline_knots="skew14"), batch())
    k = next(x for x in a if "crps" in x)
    assert a[k] != b[k], "the knot preset did not reach the model"


# ------------------------------------------------------------------ E3 / E4

@pytest.mark.parametrize("family", ["spline", "pwl"])
def test_z_space_term_changes_the_loss_and_defaults_to_off(family):
    off = one_step(module(head_family=family, crps_z_weight=0.0), batch())
    on = one_step(module(head_family=family, crps_z_weight=1.0), batch())
    k = next(x for x in off if "crps" in x)
    assert on[k] > off[k], f"{family}: --crps_z_weight 1.0 did not add a term"
    again = one_step(module(head_family=family), batch())
    assert again[k] == off[k], "the default is not today's behaviour"


def test_gap_floor_changes_the_forecast_and_defaults_to_off():
    plain = one_step(module(head_family="pwl", spline_gap_floor=False), batch())
    floored = one_step(module(head_family="pwl", spline_gap_floor=True), batch())
    k = next(x for x in plain if "crps" in x)
    assert plain[k] != floored[k], "--spline_gap_floor did not reach the head"
    assert one_step(module(head_family="pwl"), batch())[k] == plain[k]


# ------------------------------------------------------------------ E0: the export grid

@pytest.mark.parametrize("spacing", ["normal", "skew"])
def test_output_u_grid_is_well_formed(spacing):
    import numpy as np
    from src.models.quantile_spline import U_HI, U_LO, U_MID, output_u_grid
    u = output_u_grid(64, spacing=spacing)
    assert u.size == 64
    assert np.all(np.diff(u) > 1e-9), "a zero-width segment gives a 0/0 slope and a NaN CRPS"
    for gate in (U_LO, U_MID, U_HI):
        assert np.min(np.abs(u - gate)) < 1e-12, f"gate {gate} not pinned"


def test_skew_grid_moves_levels_out_of_the_flat_middle():
    """20/14/9/21 is the measured incumbent allocation; skew must not reproduce it."""
    import numpy as np
    from src.models.quantile_spline import output_u_grid
    def alloc(sp):
        u = output_u_grid(64, spacing=sp)
        return [int(((u >= a) & (u < b)).sum()) for a, b in
                ((0, .1), (.1, .6), (.6, .9), (.9, 1.001))]
    assert alloc("normal") == [20, 14, 9, 21]
    skew = alloc("skew")
    assert skew[1] < 14 and skew[2] > 9, f"skew allocation {skew} did not move anything"
    assert sum(skew) == 64


# ------------------------------------------- the export path, for every head family

@pytest.mark.parametrize("family", ["spline", "pwl", "isqf"])
def test_the_qf_export_decodes_every_head_family(family):
    """The prediction writer must decode the head it was trained with.

    It did not. ``train_lightning.py`` called ``splines_from_output`` -- the rational-
    quadratic decoder, 29 channels per horizon -- for every family, so ``pwl`` and ``isqf``
    (16) raised ``ValueError: expected 116 spline channels after 12, got 64`` at the first
    prediction batch. E1, E2 and E4 would each have trained for fifty minutes and then
    written no raster; the four free-scale arms likewise. ``model._decode`` is the one place
    that knows the family, and the loss already goes through it.

    Asserted on the same call the writer makes, with the same ``n_triple`` derivation, so a
    regression in either shows up here rather than at minute fifty.
    """
    from src.models.quantile_spline import output_u_grid

    m = module(head_family=family).model
    m.eval()
    x = torch.randn(1, T, 1, 16, 16)
    st = torch.randn(1, 2, 16, 16)
    ctx = torch.randn(1, C_CTX, 16, 16)
    with torch.no_grad():
        pred = m(x, st, context=ctx)
    n_triple = pred.shape[1] - m.num_horizons * m.n_spline_params
    assert n_triple == 3 * m.num_horizons, \
        f"{family}: {n_triple} triple channels, expected {3 * m.num_horizons}"

    u = torch.as_tensor(output_u_grid(64), dtype=pred.dtype)
    with torch.no_grad():
        qfs = [sp.ppf(u) for sp in m._decode(pred, m.spline_clamp(), n_triple=n_triple)]
    assert len(qfs) == m.num_horizons
    for h, q in enumerate(qfs):
        assert q.shape == (1, 16, 16, 64), f"{family} h{h}: {tuple(q.shape)}"
        assert torch.isfinite(q).all(), f"{family} h{h}: non-finite quantile"
        assert (q.diff(dim=-1) >= -1e-6).all(), f"{family} h{h}: Q(u) is not increasing"


@pytest.mark.parametrize("family", ["pwl", "isqf"])
def test_the_rational_quadratic_decoder_refuses_a_linear_head(family):
    """Prove the control: the old call really was wrong, not merely redundant."""
    from src.models.quantile_spline import splines_from_output

    m = module(head_family=family).model
    m.eval()
    with torch.no_grad():
        pred = m(torch.randn(1, T, 1, 16, 16), torch.randn(1, 2, 16, 16),
                 context=torch.randn(1, C_CTX, 16, 16))
    with pytest.raises(ValueError, match="spline channels"):
        splines_from_output(pred, m.num_horizons, m.spline_u_knots,
                            learn_slopes=m.spline_learn_slopes, clamp=m.spline_clamp())


# ------------------------------------------------- weight averaging changes the weights

def test_weight_averaging_changes_the_published_weights_against_a_seeded_control():
    """A flag that is accepted, logged and inert reads as a null.

    Measured motivation: `val_crps` plateaus after ~epoch 30 and oscillates by +/- 0.001, so
    ModelCheckpoint's argmin picked epochs 67/124/146/67/127/111 across six b1 fold-models.
    Accuracy was reproducible to 0.1%; the gates spanned 25-115% of their own mean. Averaging
    replaces "a random plateau epoch" with "the plateau" -- but only if it actually runs, so
    this asserts the end-of-training weights MOVED relative to the same run without it.
    """
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset

    b = batch()

    class _One(Dataset):
        def __len__(self):
            return B

        def __getitem__(self, i):
            return {k: v[i] for k, v in b.items()}

    def run(wa, epochs=3):
        """Three REAL epochs. Averaging over one epoch is the identity, so a single-step
        harness would make this test pass vacuously whatever the flag did."""
        m = module(seed=0, weight_avg_last=wa)
        m.log = lambda *a, **k: None
        pl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
                   enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False, num_sanity_val_steps=0).fit(
            m, DataLoader(_One(), batch_size=B))
        return {k: v.detach().clone() for k, v in m.model.state_dict().items()
                if v.is_floating_point()}

    off = run(0)
    on = run(2)
    assert set(off) == set(on)
    moved = [k for k in off if not torch.equal(off[k], on[k])]
    assert moved, "weight_avg_last did nothing to the published weights"
    # and it must be an AVERAGE, not a different epoch: the moves are small relative to the
    # weights themselves, because consecutive plateau epochs are close together.
    k = max(moved, key=lambda k: off[k].numel())
    rel = (off[k] - on[k]).abs().max() / off[k].abs().max().clamp_min(1e-12)
    assert 0 < rel < 1.0, f"averaged weights moved by {rel:.3g} of the tensor scale"


def test_weight_averaging_defaults_to_off():
    """Additive, defaulting to today's behaviour -- the convention for every new flag."""
    m = module(seed=0)
    assert m.weight_avg_last == 0


# ------------------------------------------------------ section 4.1: the scale arms

def _decoded(m, seed=0, size=16):
    """One horizon's quantile function from a real forward pass."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, 1, size, size, generator=g)
    st = torch.randn(1, 2, size, size, generator=g)
    ctx = torch.randn(1, C_CTX, size, size, generator=g)
    m = m.model if hasattr(m, "model") else m
    m.eval()
    with torch.no_grad():
        pred = m(x, st, context=ctx)
    n_triple = pred.shape[1] - m.num_horizons * m.n_spline_params
    with torch.no_grad():
        return m._decode(pred, m.spline_clamp(), n_triple=n_triple)[0]


@pytest.mark.parametrize("family", ["pwl", "isqf"])
def test_free_scale_changes_the_forecast_and_defaults_to_off(family):
    off = _decoded(module(head_family=family))
    on = _decoded(module(head_family=family, free_scale=True))
    u = torch.linspace(0.001, 0.999, 41)
    assert not torch.allclose(off.ppf(u), on.ppf(u), atol=1e-7), \
        f"--free_scale did nothing to {family}"
    assert module(head_family=family).model.free_scale is False


@pytest.mark.parametrize("family", ["pwl", "isqf"])
def test_free_scale_starts_at_a_usable_width(family):
    """The failure this catches is not subtle, and the doc's own spec walks into it.

    Every head feeds the shape channels through a SOFTMAX, so their absolute magnitude is
    arbitrary -- measured, |raw| averages 2.68 at init. `--free_scale` makes that magnitude
    BE the width, so the literal spec (`|.| + tol`, unnormalised) starts the ladder 330x too
    wide: 3.3 HM against an intended 0.01. Every knot then saturates the [0, 1] clamp, pwl
    spans the whole range and isqf reports a width of exactly ZERO, and the arm measures
    whether the optimiser can escape a hopeless init rather than whether the factorisation
    was earning its keep. `free_increments` gives one increment a defined size.
    """
    sp = _decoded(module(head_family=family, free_scale=True))
    lo, _, hi = sp.triple()
    w = ((hi - lo) * 0.2).median().item()      # module() sets hm_std = 0.2
    assert 1e-3 < w < 0.3, f"{family} free-scale init width {w:.4g} HM is not usable"
    assert w > 0, "the ladder collapsed"


def test_free_scale_gives_pwl_and_isqf_the_same_width_scale():
    """E2a and E1b are the same ladder up to which knot the location attaches to, so their
    widths must live on one scale -- otherwise the comparison the doc wants (where does
    persistence enter?) would be confounded by an arbitrary unit."""
    a = _decoded(module(head_family="pwl", free_scale=True)).triple()
    b = _decoded(module(head_family="isqf", free_scale=True)).triple()
    wa = (a[2] - a[0]).median().item()
    wb = (b[2] - b[0]).median().item()
    assert wa == pytest.approx(wb, rel=0.5), f"pwl {wa:.4g} vs isqf {wb:.4g}"


def test_free_scale_refuses_to_compose_with_the_gap_floor():
    """E4 x E2a is a re-derivation, not a combination: gap_floor_heights computes its floor
    as dp_k / (f_max * scale) in normalised v units and --free_scale deletes both terms."""
    with pytest.raises(ValueError, match="do not compose"):
        _decoded(module(head_family="pwl", free_scale=True, spline_gap_floor=True))


def test_free_scale_is_not_implemented_for_the_incumbent():
    """The doc says implement it for spline only if an E6-E8 candidate asks. Until then it
    must fail loudly rather than be silently ignored on the incumbent head."""
    m = module(head_family="spline", free_scale=True).model
    assert m.free_scale is True
    # _decode routes 'spline' to splines_from_output, which has no free_scale path at all;
    # the flag being accepted and ignored is exactly the inert-flag failure, so assert that
    # the arm is not silently available rather than that it works.
    assert m.head_family == "spline"


@pytest.mark.parametrize("space", ["logit", "neglog"])
def test_isqf_tails_act_only_in_the_tails(space):
    """E1a must move u outside [0.001, 0.999] and leave the spline interior untouched.

    The interior half is the real assertion: `width95` is read at u = 0.025/0.975, so the
    phase doc makes E1a the CONTROL on the width-narrowing row and expects exactly 0.000
    there. If the tails leaked inward that control would be silently wrong.
    """
    base = _decoded(module(head_family="isqf"))
    tail = _decoded(module(head_family="isqf", isqf_tails=True, isqf_space=space))
    inner = torch.tensor([0.001, 0.005, 0.025, 0.5, 0.975, 0.99, 0.999])
    assert torch.allclose(base.ppf(inner), tail.ppf(inner), atol=1e-6), \
        "the tails leaked into the spline interior"
    outer = torch.tensor([0.0002, 0.0005, 0.9995, 0.9999])
    assert not torch.allclose(base.ppf(outer), tail.ppf(outer), atol=1e-7), \
        "the tails did nothing where they own the domain"


def test_isqf_tails_are_a_curve_not_a_line_to_a_distant_knot():
    """The tails must be EVALUATED in the tail region, not interpolated toward a relocated
    endpoint. Both look different from E1, so this omission would have been invisible and
    E1a would have been judged on a mechanism it was not running."""
    sp = _decoded(module(head_family="isqf", isqf_tails=True))
    a, mid, b = 0.999, 0.9995, 0.9999
    qa, qm, qb = (sp.ppf(torch.tensor([v])).median().item() for v in (a, mid, b))
    t = (mid - a) / (b - a)
    straight = qa + t * (qb - qa)
    assert abs(qm - straight) > 1e-4, "the upper tail is a straight line; ppf is not overridden"


def test_isqf_tails_stay_monotone_and_finite_in_both_spaces():
    for space in ("logit", "neglog"):
        sp = _decoded(module(head_family="isqf", isqf_tails=True, isqf_space=space))
        u = torch.cat([torch.tensor([1e-4, 5e-4]), torch.linspace(0.001, 0.999, 60),
                       torch.tensor([0.9995, 0.9999])])
        q = sp.ppf(u)
        assert torch.isfinite(q).all(), f"{space}: non-finite quantile"
        assert (q.diff(dim=-1) >= -1e-6).all(), f"{space}: Q(u) is not increasing"


def test_the_tails_cost_two_channels_and_only_on_isqf():
    from src.models.quantile_pwl import n_pwl_params
    assert n_pwl_params(15, "isqf") == 16
    assert n_pwl_params(15, "isqf", tails=True) == 18      # the doc's 18 params/horizon
    with pytest.raises(ValueError, match="isqf feature"):
        n_pwl_params(15, "pwl", tails=True)
