"""e9 and e10 are the head's parameter count, so the log must say which head trained.

Two levers land this round -- `--spline_slopes fritsch` (e9) and `--spline_knots lean9`
(e10) -- and each one is invisible in every other line of the log: same loss, same channels,
same epoch count. A run whose flag silently failed to engage would score as "the lever does
nothing", which this project has already mistaken for a finding.

So the banner line and the shell check that greps it are tested against each other here.
Writing a check against text nobody emits is not hypothetical: `verify_context_channels`
grepped lowercase against a capitalised banner, failed on every *correct* run, and killed a
slate under `set -e`. A check must also be proven to FIRE, which is what the reject cases are.
"""
import os
import subprocess
import sys
import types

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
# train_lightning.py imports its siblings bare (``from torchgeo_dataloader import ...``), so
# scripts/ has to be importable as a top-level directory as well as a package.
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from scripts.train_lightning import _spline_head_banner  # noqa: E402
from src.models.quantile_spline import KNOT_PRESETS, knot_preset  # noqa: E402
from src.models.spatiotemporal_predictor import SpatioTemporalPredictor  # noqa: E402

REPO = _REPO


def _args(head_family="spline", spline_knots="default14", spline_slopes="learned"):
    return types.SimpleNamespace(head_family=head_family, spline_knots=spline_knots,
                                 spline_slopes=spline_slopes)


def test_lean9_is_a_strict_subset_of_the_default_grid():
    """A preset that MOVED a knot would confound "less resolution" with "different
    resolution". lean9 only removes: 0.001, 0.05, 0.25, 0.75, 0.95, 0.999."""
    lean = set(knot_preset("lean9"))
    base = set(KNOT_PRESETS["default14"])
    assert lean < base
    assert base - lean == {0.001, 0.05, 0.25, 0.75, 0.95, 0.999}
    assert len(knot_preset("lean9")) == 9
    assert np.all(np.diff(np.asarray(knot_preset("lean9"))) > 0)


@pytest.mark.parametrize("knots,slopes,expect", [
    ("default14", "learned", "knots default14 (n=15, bins=14), slopes learned, 29 params/horizon"),
    ("lean9", "learned", "knots lean9 (n=9, bins=8), slopes learned, 17 params/horizon"),
    ("lean9", "fritsch", "knots lean9 (n=9, bins=8), slopes fritsch, 10 params/horizon"),
    ("default14", "fritsch", "knots default14 (n=15, bins=14), slopes fritsch, 16 params/horizon"),
])
def test_the_banner_reports_the_head_that_will_train(knots, slopes, expect):
    line = _spline_head_banner(_args(spline_knots=knots, spline_slopes=slopes))
    assert line.startswith("Spline head:")
    assert expect in line


def test_no_banner_for_the_triple_head():
    assert _spline_head_banner(_args(head_family="triple")) is None


def _verify(tmp_path, log_text, flags):
    log = tmp_path / "fold.log"
    log.write_text(log_text)
    script = (f'source "{REPO}/scripts/dist_base_args.sh"; '
              f'verify_spline_head "{log}" run "{flags}"')
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _log(knots, slopes):
    return "LOSS WEIGHTS\n" + _spline_head_banner(_args(spline_knots=knots,
                                                        spline_slopes=slopes)) + "\n"


def test_verifier_accepts_the_head_that_was_asked_for(tmp_path):
    r = _verify(tmp_path, _log("lean9", "fritsch"),
                "--spline_knots lean9 --spline_slopes fritsch")
    assert r.returncode == 0, r.stderr
    assert "✓" in r.stdout


@pytest.mark.parametrize("log_knots,log_slopes,flags", [
    ("default14", "learned", "--spline_slopes fritsch"),      # slope mode did not engage
    ("default14", "learned", "--spline_knots lean9"),         # preset did not engage
    ("default14", "fritsch", "--spline_knots lean9 --spline_slopes fritsch"),
])
def test_verifier_fires_when_the_lever_did_not_engage(tmp_path, log_knots, log_slopes, flags):
    r = _verify(tmp_path, _log(log_knots, log_slopes), flags)
    assert r.returncode == 1
    assert "FATAL" in r.stderr


def test_verifier_fires_on_a_log_with_no_banner_at_all(tmp_path):
    r = _verify(tmp_path, "LOSS WEIGHTS\nSSIM weight: 0.0\n", "--spline_slopes fritsch")
    assert r.returncode == 1 and "no 'Spline head:' line" in r.stderr


def test_verifier_is_silent_for_a_run_that_sets_neither(tmp_path):
    """af_nomono and af_rad pass through this function. A guard that fired on them would kill
    the queue under `set -e` -- the exact way the covariate check once killed a slate."""
    r = _verify(tmp_path, "LOSS WEIGHTS\n", "--weight_avg_last 20 --context_radii 3,30,100")
    assert r.returncode == 0 and r.stdout == ""


def _built(knots, slopes):
    """A tiny predictor on the grid under test, forward + backward through CRPS."""
    import torch
    from src.models.crps_loss import crps_spline
    from src.models.quantile_spline import splines_from_output

    torch.manual_seed(0)
    m = SpatioTemporalPredictor(
        hidden_dim=8, kernel_size=3, num_layers=1,
        num_static_channels=2, num_dynamic_channels=2,
        use_location_encoder=False, head_family="spline",
        spline_u_knots=torch.tensor(knot_preset(knots), dtype=torch.float32),
        spline_learn_slopes=(slopes == "learned"),
    )
    m.set_norm_stats(0.085463, 0.153474)
    out = m(torch.rand(2, 3, 2, 16, 16), torch.rand(2, 2, 16, 16))
    splines = splines_from_output(out, 4, m.spline_u_knots,
                                  learn_slopes=m.spline_learn_slopes)
    loss = crps_spline(splines[0], torch.rand(2, 16, 16) * 0.2)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    return m, loss, grads


@pytest.mark.parametrize("knots,slopes", [("lean9", "learned"), ("lean9", "fritsch"),
                                          ("default14", "fritsch")])
def test_the_new_heads_build_and_backpropagate(knots, slopes):
    """`deep_lower` could not train at all -- knots 1e-4 apart drove the boundary derivatives
    non-finite on the first step. Coarsening is the opposite move, but "it should be fine"
    is what that preset had going for it too, so both e9 and e10 are proven to take a step
    before 3 x 50 min of GPU time is spent on them."""
    import torch
    m, loss, grads = _built(knots, slopes)
    assert torch.isfinite(loss), f"{knots}/{slopes} gave a non-finite CRPS"
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), \
        f"{knots}/{slopes} produced non-finite gradients"


def test_fritsch_removes_the_slope_parameters_from_the_head():
    """The point of e9: 13 learned derivatives per horizon go away, and the output channel
    count is what proves it rather than the flag being stored."""
    learned = _built("default14", "learned")[0]
    fritsch = _built("default14", "fritsch")[0]
    assert learned.n_spline_params - fritsch.n_spline_params == 13


# --------------------------------------------- the banner must tell the families apart

def _banner(**kw):
    """Call _spline_head_banner without importing train_lightning's __main__ block."""
    import types
    src = open(os.path.join(REPO, "scripts", "train_lightning.py")).read()
    ns = {}
    exec("import sys\nsys.path.insert(0, %r)\n" % REPO
         + src[src.index("def _spline_head_banner"):src.index("def _experiment_kwargs")], ns)
    a = types.SimpleNamespace(**{**dict(head_family="spline", spline_knots="default14",
                                        spline_slopes="learned", isqf_tails=False,
                                        isqf_space="logit", free_scale=False,
                                        spline_cumulative_width=True), **kw})
    return ns["_spline_head_banner"](a)


def test_the_head_banner_distinguishes_every_family_on_the_slate():
    """Until 2026-09-15 it printed "Spline head ... 29 params/horizon" for EVERY family.

    It hardcoded the word and computed the count with `n_spline_params`, the
    rational-quadratic formula, whatever --head_family said -- so E1, E2, E4 and all four
    scale arms logged a 29-parameter spline while running a 16- or 18-parameter head. The one
    line a log reader checks to see which head ran could not tell them. Same
    one-formula-for-three-families conflation that left the prediction writer unable to
    decode pwl or isqf at all.
    """
    lines = {
        "spline": _banner(),
        "isqf": _banner(head_family="isqf"),
        "pwl": _banner(head_family="pwl"),
    }
    assert len(set(lines.values())) == 3, f"the banner cannot tell the families apart: {lines}"
    for fam, line in lines.items():
        assert f"family {fam}" in line, f"{fam}: {line}"
    assert "29 params/horizon" in lines["spline"]
    assert "16 params/horizon" in lines["isqf"]
    assert "16 params/horizon" in lines["pwl"]


def test_the_head_banner_carries_the_scale_arm_flags():
    """An arm whose only delta is a flag must be distinguishable in the log by that flag."""
    assert "tails neglog" in _banner(head_family="isqf", isqf_tails=True, isqf_space="neglog")
    assert "tails logit" in _banner(head_family="isqf", isqf_tails=True)
    assert "18 params/horizon" in _banner(head_family="isqf", isqf_tails=True)
    assert "free scale" in _banner(head_family="isqf", free_scale=True)
    assert "non-cumulative width" in _banner(spline_cumulative_width=False)
    # E1c is both, and must say so
    e1c = _banner(head_family="isqf", isqf_tails=True, free_scale=True)
    assert "tails logit" in e1c and "free scale" in e1c


def test_the_head_banner_prefers_the_constructed_module_over_the_flags():
    """Read the count off the module, never recompute it from args -- the same reason the
    context line reads its channel count off the trunk. A head built differently from what
    the flags imply must show the head, not the flags."""
    import types
    src = open(os.path.join(REPO, "scripts", "train_lightning.py")).read()
    ns = {}
    exec("import sys\nsys.path.insert(0, %r)\n" % REPO
         + src[src.index("def _spline_head_banner"):src.index("def _experiment_kwargs")], ns)
    a = types.SimpleNamespace(head_family="isqf", spline_knots="default14",
                              spline_slopes="learned", isqf_tails=False,
                              isqf_space="logit", free_scale=False,
                              spline_cumulative_width=True)
    fake = types.SimpleNamespace(n_spline_params=99)
    assert "99 params/horizon" in ns["_spline_head_banner"](a, fake)
