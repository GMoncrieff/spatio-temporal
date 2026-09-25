"""Objectives for the distributional head: CRPS (primary) and NLL (secondary).

CRPS is written in its quantile form, as the integral of the pinball loss over ``u``::

    CRPS(Q, y) = 2 * int_0^1 rho_u( y - Q(u) ) du

which is what makes it the right objective for *this* residual rather than merely a
defensible one. ``docs/background/model_phase.md`` §6.3 measured Gaussian NLL on this data and
it inflated the fitted widths by up to 7x: the residual's kurtosis runs 928-20366, the log
score's quadratic term is dominated by a handful of extreme pixels, and ``log sigma`` resists
only logarithmically. Pinball has bounded influence per pixel and cannot be dragged that way,
and CRPS is an integral of pinball losses, so it inherits the bound.

**Tail weighting is safe in a way sample reweighting is not.** ``int w(u) rho_u(y - Q(u)) du``
is a positive combination of pinball losses, and each one is minimised at the true ``u``-th
quantile *independently of* ``w``. So a u-weighting moves where the optimiser spends its
effort without moving the target, and needs no importance correction. Reweighting *samples*
does move the target, which is why that knob lives in the dataloader with a correction beside
it.

The quadrature is split at three points, all available in closed form: the two support
crossings ``u_a``, ``u_b`` where the clamp to ``[0, 1]`` takes over, and ``u* = Q^-1(y)``
where the pinball kink sits. Outside ``[u_a, u_b]`` the quantile function is constant, so
those two pieces are integrated without evaluating the spline at all.
"""

from __future__ import annotations

import torch

from .quantile_spline import flat_piece_nodes, segment_nodes


def tail_weight_fn(u, lam: float, u0: float = 0.95, p: float = 2.0,
                   lam_lo: float = 0.0, u0_lo: float = 0.05):
    """``w(u) = 1 + lam*((u-u0)/(1-u0))_+^p + lam_lo*((u0_lo-u)/u0_lo)_+^p``.

    Both weights zero gives ordinary CRPS. The lower side is the exact mirror of the upper and
    exists for a measured reason: `P(u<0.001)` reads 0.0076 at h=20 against a nominal 0.001, so
    the model is blindsided by declines while being braced for growth that does not come.
    """
    if lam == 0.0 and lam_lo == 0.0:
        return None
    w = torch.ones_like(u)
    if lam:
        w = w + lam * ((u - u0).clamp_min(0.0) / (1.0 - u0)).pow(p)
    if lam_lo:
        w = w + lam_lo * ((u0_lo - u).clamp_min(0.0) / u0_lo).pow(p)
    return w


def _masked_mean(x, mask):
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype).reshape(x.shape)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _pinball(u, err):
    """``rho_u(e) = e * (u - 1{e < 0})``."""
    return err * (u - (err < 0).to(err.dtype))


def crps_spline(spline, y, mask=None, n_nodes: int = 6,
                tail_lam: float = 0.0, tail_u0: float = 0.95, tail_p: float = 2.0,
                tail_lam_lo: float = 0.0, tail_u0_lo: float = 0.05,
                reduce: bool = True):
    """Quadrature CRPS of a :class:`QuantileSpline` against observations ``y``.

    ``y`` broadcasts against the spline's pixel shape.

    Six nodes per bin, not three. Measured against :func:`crps_reference` on adversarial shape
    parameters the worst-pixel error is 5.0e-4 at two nodes, 1.1e-4 at three and 8.0e-6 at six
    -- and 3e-5 is the int16 storage quantum, so three nodes would make the objective itself
    coarser than the product it trains. ``--crps_nodes`` lowers it if memory ever demands.
    """
    def weight(u, w):
        tw = tail_weight_fn(u, tail_lam, tail_u0, tail_p, tail_lam_lo, tail_u0_lo)
        return w if tw is None else w * tw

    u_a, u_b = spline.support_crossings()
    u_star = spline.cdf(y).maximum(u_a).minimum(u_b)

    u_lo, w_lo = segment_nodes(spline.u_knots, n_nodes, u_a, u_star)
    u_hi, w_hi = segment_nodes(spline.u_knots, n_nodes, u_star, u_b)
    u = torch.cat([u_lo, u_hi], dim=-1)
    w = weight(u, torch.cat([w_lo, w_hi], dim=-1))
    total = (_pinball(u, y.unsqueeze(-1) - spline.ppf(u, clamp=False)) * w).sum(dim=-1)

    if spline.clamp is not None:
        lo, hi = spline.clamp
        # Q is constant on both flat pieces, so the integrand is a low-degree polynomial in u
        # and the spline is never evaluated.
        uf, wf = flat_piece_nodes(torch.zeros_like(u_a), u_a)
        total = total + ((y - lo).unsqueeze(-1) * uf * weight(uf, wf)).sum(dim=-1)
        uf, wf = flat_piece_nodes(u_b, torch.ones_like(u_b))
        total = total + ((hi - y).unsqueeze(-1) * (1.0 - uf) * weight(uf, wf)).sum(dim=-1)

    per_pixel = 2.0 * total
    return _masked_mean(per_pixel, mask) if reduce else per_pixel


def nll_spline(spline, y, mask=None, reduce: bool = True):
    """``-log f(y) = log (dQ/du)|_{u*}``.

    Evaluated on the **unclamped** spline. The clamp to ``[0, 1]`` creates flat regions, and a
    flat region in a quantile function is an atom that a log-density cannot represent. The
    data audit says no observation sits at either bound (HM is never exactly 0 or 1 on this
    extent), so nothing real is lost -- but the mass the spline puts outside ``[0, 1]`` is a
    diagnostic worth logging, which :func:`out_of_support_mass` returns.
    """
    per_pixel = spline.log_dq_du(spline.cdf(y))
    return _masked_mean(per_pixel, mask) if reduce else per_pixel


def out_of_support_mass(spline):
    """Probability the unclamped spline places outside ``[0, 1]`` -- an NLL-run diagnostic."""
    if spline.clamp is None:
        return torch.zeros_like(spline.anchor)
    u_a, u_b = spline.support_crossings()
    return u_a + (1.0 - u_b)


def crps_reference(spline, y, n_grid: int = 200001):
    """Brute-force CRPS on a uniform u-grid. Slow, for tests only.

    A second implementation of the same quantity sharing no quadrature code with
    :func:`crps_spline`: agreement is evidence, disagreement localises the bug. This project
    has been saved more than once by keeping a closed form and a sampler side by side.
    """
    u = torch.linspace(0.0, 1.0, n_grid, dtype=spline.v_knots.dtype,
                       device=spline.v_knots.device)
    q = spline.ppf(u)
    return 2.0 * torch.trapz(_pinball(u, y.unsqueeze(-1) - q), u, dim=-1)


# ---------------------------------------------------------------------------- E3: z-space

def asinh_z(x, s: float):
    """``z = asinh(x / s)`` -- linear near zero, logarithmic far from it."""
    return torch.asinh(x / s)


def crps_zspace(spline, y, hm_t0, s: float, mask=None, n_nodes: int = 6, reduce: bool = True,
                **kw):
    """CRPS of the *transformed* forecast, ``z = asinh((Q - hm_t0) / s)`` against ``z(y)``.

    The incentive problem this fixes, stated plainly. CRPS in raw HM units is dominated by
    the tail: the persistence core spans ~0.0036 HM against a range above 1.2, so placing
    the core badly costs almost nothing in the loss. 68% of land sits in that core, and the
    head is very likely picket-fencing there precisely because nothing ever penalised it.

    In ``z`` space an error of 0.0005 near the core weighs about the same as an error of
    0.05 in the tail, which is the relative importance actually wanted. Summed with the raw
    term (``--crps_z_weight``) rather than replacing it, so the tail is not abandoned to fix
    the body.

    CRPS is **not** invariant to a monotone transform, which is the whole point -- but it
    stays proper in the transformed space, because ``asinh`` is strictly increasing and
    therefore the ``z``-quantiles of the pushed-forward forecast are the transforms of the
    original quantiles. Minimising it still means "get the whole distribution right", scored
    on a different ruler.

    Implemented by evaluating ``Q`` at the same split nodes and transforming *there*, so the
    transform costs one ``asinh`` per node and no second quadrature apparatus.
    """
    u_a, u_b = spline.support_crossings()
    y_z = asinh_z(y - hm_t0, s)

    def q_z(u):
        return asinh_z(spline.ppf(u, clamp=True) - hm_t0.unsqueeze(-1), s)

    # The crossing in z-space is the crossing in raw space -- asinh is monotone -- so the
    # kink is still at u* = Q^-1(y) and the same split applies unchanged.
    u_star = spline.cdf(y).maximum(u_a).minimum(u_b)
    u_lo, w_lo = segment_nodes(spline.u_knots, n_nodes, u_a, u_star)
    u_hi, w_hi = segment_nodes(spline.u_knots, n_nodes, u_star, u_b)
    u = torch.cat([u_lo, u_hi], dim=-1)
    w = torch.cat([w_lo, w_hi], dim=-1)
    total = (_pinball(u, y_z.unsqueeze(-1) - q_z(u)) * w).sum(dim=-1)

    if spline.clamp is not None:
        lo, hi = spline.clamp
        lo_z = asinh_z(lo - hm_t0, s)
        hi_z = asinh_z(hi - hm_t0, s)
        uf, wf = flat_piece_nodes(torch.zeros_like(u_a), u_a)
        total = total + ((y_z - lo_z).unsqueeze(-1) * uf * wf).sum(dim=-1)
        uf, wf = flat_piece_nodes(u_b, torch.ones_like(u_b))
        total = total + ((hi_z - y_z).unsqueeze(-1) * (1.0 - uf) * wf).sum(dim=-1)

    per_pixel = 2.0 * total
    return _masked_mean(per_pixel, mask) if reduce else per_pixel
