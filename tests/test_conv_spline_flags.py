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
            quantile_context_channels=C_CTX, central_context_channels=C_CTX)


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
