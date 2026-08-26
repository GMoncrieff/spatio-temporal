"""A diverged run must fail loudly instead of scoring normally.

This is the guard, not the bug fix. The bug (a `sqrt` derivative singularity in
`QuantileSpline.cdf`) is fixed separately; this exists because *any* future divergence was
silent, and silence is what let it corrupt a whole round of results:

`ModelCheckpoint` compares the monitored metric with `<`, and every comparison against NaN is
false, so a NaN epoch is never selected. A run that dies at epoch 3 therefore keeps its epoch-2
checkpoint, trains 147 more worthless epochs, exits 0, and produces a scorecard that looks like
any other. Round 1's `d2` was published that way -- a 3-epoch model compared against 150-epoch
baselines -- and the spread that comparison produced is what motivated round 2's whole design.

Unlike `test_spline_cdf_gradient.py`, these tests DO discriminate: they fail if the guard is
removed.
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lightning_module import SpatioTemporalLightningModule  # noqa: E402


class _Trainer:
    """Minimal stand-in: the hooks under test read only max_epochs and current_epoch."""

    def __init__(self, max_epochs=10, current_epoch=0):
        self.max_epochs = max_epochs
        self.current_epoch = current_epoch


def _module(**kw):
    m = SpatioTemporalLightningModule(
        hidden_dim=4, num_layers=1, num_static_channels=1, num_dynamic_channels=1,
        use_location_encoder=False, **kw)
    m.trainer = _Trainer()
    return m


def _poison(module):
    with torch.no_grad():
        p = next(module.parameters())
        p[(0,) * p.dim()] = float("nan")


def test_epoch_end_raises_on_nonfinite_weights():
    m = _module()
    m.on_train_epoch_end()          # healthy: silent
    _poison(m)
    with pytest.raises(RuntimeError, match="NON-FINITE WEIGHTS"):
        m.on_train_epoch_end()


def test_epoch_end_names_the_epoch_and_the_escape_hatch():
    m = _module()
    _poison(m)
    with pytest.raises(RuntimeError) as e:
        m.on_train_epoch_end()
    msg = str(e.value)
    assert "epoch" in msg.lower()
    assert "--abort_on_nonfinite" in msg, "the error must say how to restore the old behaviour"


def test_opting_out_restores_the_old_silent_behaviour():
    """The flag is additive: False must reproduce exactly what happened before."""
    m = _module(abort_on_nonfinite=False)
    _poison(m)
    m.on_train_epoch_end()          # must not raise


def test_weight_averaging_refuses_a_poisoned_accumulator():
    """One NaN epoch inside the window makes the averaged weights NaN, and prediction uses them."""
    m = _module(weight_avg_last=3)
    m.trainer = _Trainer(max_epochs=3)
    m.abort_on_nonfinite = False     # let the poisoned epoch accumulate
    for e in range(2):
        m.trainer.current_epoch = e
        m.on_train_epoch_end()
    _poison(m)
    m.trainer.current_epoch = 2
    m.on_train_epoch_end()
    assert m._wa_count >= 1
    m.abort_on_nonfinite = True
    with pytest.raises(RuntimeError, match="non-finite"):
        m.on_train_end()


def test_weight_averaging_still_works_when_healthy():
    m = _module(weight_avg_last=3)
    m.trainer = _Trainer(max_epochs=3)
    for e in range(3):
        m.trainer.current_epoch = e
        m.on_train_epoch_end()
    m.on_train_end()
    assert all(torch.isfinite(p).all() for p in m.parameters())
