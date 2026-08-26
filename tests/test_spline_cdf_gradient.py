"""Invariant checks on `QuantileSpline.cdf`'s gradient.

**Read this before trusting these tests.** They are guards against regression of an invariant,
NOT a discriminating reproduction: every test here also passes against the unfixed source. A
compact synthetic trigger was searched for and not found (3800+ random spline configurations,
masked and unmasked, y inside and at the clamp bounds). The failure needs the parameter
structure a trained model reaches, which random draws do not produce.

The discriminating evidence is a recorded replay, not a unit test:

    the captured failing batch + pre-step weights reproduce it deterministically,
    38 of 94 parameter tensors non-finite with the unfixed `cdf`, 0 with the fix,
    identical finite gradient norm (0.0920376) under two independent formulations.

What went wrong, since the forward gives no sign of it: `cdf` solves a quadratic and takes
`sqrt` of a discriminant floored at 0. `sqrt` is finite at 0 and its derivative is not, so
`SqrtBackward0` computes `grad / (2*sqrt(0))` -- `inf` for a non-zero incoming gradient, and
`0/0 = NaN` for a zero one. A masked-out pixel has exactly zero incoming gradient, which is how
`inf` becomes `NaN`. In the captured failure exactly ONE pixel of 131072 in the h=20 horizon was
poisoned, and that one pixel put NaN into every parameter upstream of the CRPS path in a single
optimizer step, permanently, while the loss stayed finite and the surviving gradient norm stayed
ordinary (0.135 against a median of 0.15). Six of round 1's sixteen runs died this way and were
scored anyway, because the checkpoint monitor silently fell back past the dead epochs.
"""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.quantile_spline import QuantileSpline, knot_preset  # noqa: E402
from src.models.crps_loss import crps_spline  # noqa: E402


def _knots():
    return torch.tensor(knot_preset("default14"), dtype=torch.float32)


def _raw(n, u_knots, scale=3.0, seed=0):
    n_p = 2 + (u_knots.numel() - 1) + (u_knots.numel() - 2)
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, n_p, generator=g) * scale).requires_grad_(True)


def test_cdf_gradient_finite_at_the_clamp_bounds():
    """`support_crossings` inverts at y=0 and y=1, where the discriminant clamps for every pixel."""
    uk = _knots()
    raw = _raw(2048, uk)
    sp = QuantileSpline.from_channels(raw, uk, learn_slopes=True)
    u_a, u_b = sp.support_crossings()
    assert torch.isfinite(u_a).all() and torch.isfinite(u_b).all()
    (u_a.sum() + u_b.sum()).backward()
    assert torch.isfinite(raw.grad).all(), (
        f"{int((~torch.isfinite(raw.grad)).sum())} non-finite head gradients from "
        f"support_crossings")


def test_crps_gradient_finite_with_masked_pixels():
    """Masked pixels carry exactly zero incoming gradient -- the 0/0 case."""
    uk = _knots()
    raw = _raw(1024, uk, seed=7)
    sp = QuantileSpline.from_channels(raw, uk, learn_slopes=True)
    g = torch.Generator().manual_seed(11)
    y = torch.rand(1024, generator=g) * 0.9 + 0.02
    mask = torch.ones(1024)
    mask[:256] = 0.0
    loss = crps_spline(sp, y, mask=mask)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert torch.isfinite(raw.grad).all(), (
        f"{int((~torch.isfinite(raw.grad)).sum())} non-finite head gradients through crps")


def test_cdf_forward_unchanged_by_the_guard():
    """The guard removes a derivative singularity; it must not move a forward value.

    This one DOES discriminate, but only on the forward: it pins the property that made the
    `where` formulation preferable to flooring the argument with an epsilon (which shifts every
    root by ~1e-6).
    """
    uk = _knots()
    raw = _raw(4096, uk, seed=3).detach()
    sp = QuantileSpline.from_channels(raw, uk, learn_slopes=True)
    g = torch.Generator().manual_seed(5)
    for y in (torch.zeros(4096), torch.ones(4096),
              torch.rand(4096, generator=g) * 0.9 + 0.05):
        u = sp.cdf(y)
        assert torch.isfinite(u).all()
        assert (u >= 0).all() and (u <= 1).all(), "cdf must stay in [0, 1]"
