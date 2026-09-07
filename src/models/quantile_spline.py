"""A monotone rational-quadratic quantile function, per pixel and per horizon.

This replaces the ``(lower, central, upper)`` triple with the whole conditional quantile
function ``Q(u)``. Everything the product publishes -- the central forecast, the 2.5/97.5
bounds, exceedance probabilities, and the ensemble members -- is read off this one object,
so the post-hoc width calibration and empirical marginal reshaping have nothing left to do.

Three properties are structural rather than enforced afterwards.

``Q`` is monotone in ``u``
    The rational-quadratic spline of Durkan et al. (2019) is monotone for any positive bin
    heights and positive knot derivatives, which is what the softmax and softplus below
    guarantee. So ``lower <= central <= upper`` cannot be violated and no sorting pass is
    needed.

``Q(0.5) = anchor`` and ``Q(0.975) - Q(0.025) = scale``
    ``0.025``, ``0.5`` and ``0.975`` are exact knots, so the normalisation that pins them is
    a lookup rather than an interpolation. That matters: interpolating near a gate quantile
    leaves a bias that does not shrink with more ensemble members, which is why
    ``copula.fit_residual_shape`` pins the same three points today. It also means a caller
    who wants only the published triple never has to evaluate the spline.

The knots are **fixed and tail-dense**, not learned
    A standard rational-quadratic spline learns its abscissae by softmax. With ~14 bins it
    will not place a boundary anywhere near ``u = 0.999``, and the far-field defect this
    phase exists to address lives at exactly that depth: ``docs/global_scorecard.md`` records
    ``P(dHM > 0.05)`` beyond 100 px at 0.034 of observed because +0.05 sits ~12 half-widths
    out. Fixing the abscissae is what buys the tail resolution; the learned heights then
    decide how much probability mass goes there.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

# 15 knots / 14 bins. Dense at both ends, with 0.025 / 0.5 / 0.975 present exactly so the
# published triple is a lookup. The upper side carries more knots than the lower because the
# product's tail question is one-directional: new development, not reversion.
U_KNOTS_DEFAULT = (
    0.0, 0.001, 0.005, 0.025, 0.05, 0.10, 0.25, 0.50,
    0.75, 0.90, 0.95, 0.975, 0.99, 0.999, 1.0,
)

U_LO, U_MID, U_HI = 0.025, 0.5, 0.975

# Named knot grids. `default14` is round 1's, and every other preset is a subset or a superset
# of it -- never a re-placement -- so a preset can only add or remove resolution and no knot
# ever moves. That is what makes a preset A/B readable: the difference is the knots themselves.
#
# `body_dense` corrects a choice round 1 got wrong. The default grid has exactly three knots
# between u = 0.10 and u = 0.90, while 53% of pixels move by less than 0.001 over twenty years:
# the model must represent a spike holding half its mass with three knots while spending six on
# the tails. `cov50` is the worst-calibrated coverage level on the card, 0.42 at h=5 and 0.63 at
# h=20. The grid was chosen to buy far-field tail resolution and under-resourced the body.
#
# `deep_lower` targets the other named defect: `P(u<0.001)` reads 0.0076 at h=20 against a
# nominal 0.001, so observations fall below the deepest quantile the grid can express.
#
# `lean9` goes the other way, and tests the premise the whole grid rests on. Tail knots create
# *capacity*, not information: ordinary CRPS integrates over u, so the deepest intervals
# contribute little to the population score while the head still has to estimate them. It drops
# 0.001, 0.05, 0.25, 0.75, 0.95 and 0.999 -- 14 bins to 8, 29 parameters per horizon to 17 with
# learned slopes -- and is a strict SUBSET of `default14`, so nothing moves and the comparison
# is only "was that resolution load-bearing". Round 1 refused this experiment (as D9) because a
# null would have meant "under-powered" against southern Africa's 40-89% h=20 bands; the Africa
# floor is ~5x tighter at n=6, so a null is readable now.
KNOT_PRESETS = {
    "default14": U_KNOTS_DEFAULT,
    "lean9": (0.0, 0.005, 0.025, 0.10, 0.5, 0.90, 0.975, 0.99, 1.0),
    "body_dense": tuple(sorted(set(U_KNOTS_DEFAULT) | {0.35, 0.45, 0.55, 0.65})),
    "deep_lower": tuple(sorted(set(U_KNOTS_DEFAULT) | {0.0001, 0.9999})),
    # E5. The default grid is symmetric -- dense at BOTH ends -- and the target is not. On
    # 10.3M land pixels the observed change spends 0.13% of its value range on u in
    # [0.1, 0.6] and 60% on the top decile alone, so the default spends five bins where
    # nothing varies and the picket fence is what a strictly-positive-derivative constraint
    # does when asked for a flat segment. This grid is the observed decile structure:
    # the lower tail keeps its knots (u 0-0.1 genuinely spans 35% of the range, with real
    # decreases to -0.43), the flat middle drops to three, and the upper region above 0.85
    # -- which carries the +0.017 substantive-change mode near u = 0.78 and the whole upper
    # tail -- gets the rest. Same 14 bins, so it is a pure re-placement at equal capacity
    # and an A/B against `default14` reads as "were the knots in the wrong places".
    # 0.25 is what goes; the three gate levels below cannot, so the flat middle keeps one
    # knot (0.5) rather than none.
    "skew14": (0.0, 0.001, 0.005, 0.025, 0.05, 0.10, 0.50,
               0.75, 0.85, 0.92, 0.96, 0.975, 0.99, 0.999, 1.0),
    # The same idea with the body collapsed as far as it will go and the capacity moved into
    # u > 0.75. Fewer bins than default14 (11), so it also tests whether the body needed any
    # resolution at all -- the strong form of the claim.
    "skew11": (0.0, 0.001, 0.01, 0.025, 0.10, 0.50, 0.85, 0.95, 0.975, 0.99, 0.999, 1.0),
}


def validate_knots(u, name="<grid>"):
    """Refuse a grid that cannot support the published contract, at definition time.

    ``U_LO``, ``U_MID`` and ``U_HI`` must be *exact* knots: the triple is a lookup and the
    normalisation that pins it is a gather, so a grid missing one raises deep inside
    ``QuantileSpline.__init__`` at the first forward pass instead of here. The first draft of
    the ``skew14`` preset dropped 0.5 and 0.975 while re-placing the middle, which is exactly
    the mistake this catches -- put the check where the cause is one line away.
    """
    arr = list(u)
    if arr != sorted(set(arr)):
        raise ValueError(f"knot grid {name} is not strictly increasing: {arr}")
    if arr[0] != 0.0 or arr[-1] != 1.0:
        raise ValueError(f"knot grid {name} must span [0, 1]: {arr}")
    for gate in (U_LO, U_MID, U_HI):
        if not any(abs(k - gate) < 1e-9 for k in arr):
            raise ValueError(
                f"knot grid {name} has no exact knot at u={gate}; the published triple is a "
                f"lookup at 0.025 / 0.5 / 0.975 and interpolating near a gate leaves a bias "
                f"that does not shrink with more data")
    return tuple(arr)


def knot_preset(name):
    """Look up a named knot grid, refusing an unknown name rather than falling back."""
    try:
        return validate_knots(KNOT_PRESETS[name], name)
    except KeyError:
        raise ValueError(
            f"unknown spline knot preset {name!r}; have {sorted(KNOT_PRESETS)}") from None

MIN_BIN_HEIGHT = 1e-4
MIN_DERIVATIVE = 1e-3
# A 95% interval narrower than this is not a forecast, it is a rounding error: members are
# stored as int16 x 1/32767, so 3e-5 is the smallest width the product can represent at all.
# Without a floor, CRPS can drive the width to zero on the quiet majority of pixels and the
# log-density then diverges -- and `anchor + scale * g(u)` silently returns a constant, which
# reads downstream as a perfectly confident forecast rather than as a degenerate one.
MIN_SCALE = 1e-6


def knot_index(u_knots, u: float) -> int:
    """Index of an exact knot; raises if ``u`` is not one, because a near miss is a bias."""
    # 1e-6, not 1e-12: the knot grid is held in float32 during training, where 0.025 is
    # 0.02500000037. The nearest non-knot a caller could plausibly pass is 1e-3 away, so this
    # still catches a near miss while tolerating the storage dtype.
    arr = np.asarray(u_knots, dtype=np.float64)
    hits = np.flatnonzero(np.abs(arr - u) < 1e-6)
    if hits.size != 1:
        raise ValueError(f"u={u} is not an exact knot of {arr.tolist()}")
    return int(hits[0])


def n_spline_params(n_knots: int, learn_slopes: bool = True) -> int:
    """Channels one horizon's spline needs: anchor, scale, bin heights, internal slopes."""
    n_bins = n_knots - 1
    n_internal = n_knots - 2
    return 2 + n_bins + (n_internal if learn_slopes else 0)


class QuantileSpline:
    """A batch of monotone quantile functions sharing one fixed grid of u-knots.

    All parameter tensors are shaped ``(...)`` over pixels (typically ``[B, H, W]``) with the
    knot axis last. Evaluations broadcast a trailing u-axis, returning ``(..., n_u)``.
    """

    def __init__(self, anchor, scale, v_knots, derivs, u_knots, clamp=(0.0, 1.0)):
        self.anchor = anchor              # (...)      Q(0.5)
        self.scale = scale                # (...)      Q(0.975) - Q(0.025), positive
        self.v_knots = v_knots            # (..., K)   monotone 0 -> 1
        self.derivs = derivs              # (..., K)   dv/du at each knot, positive
        self.u_knots = u_knots            # (K,)       fixed, shared
        self.clamp = clamp

        i_lo = knot_index(u_knots.detach().cpu().numpy(), U_LO)
        i_mid = knot_index(u_knots.detach().cpu().numpy(), U_MID)
        i_hi = knot_index(u_knots.detach().cpu().numpy(), U_HI)
        self._v_lo = v_knots[..., i_lo]
        self._v_mid = v_knots[..., i_mid]
        self._v_hi = v_knots[..., i_hi]
        # Positive because every bin height is positive; this is the normaliser that makes
        # `scale` the 95% width.
        self._v_span = self._v_hi - self._v_lo

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_raw(cls, anchor, scale, heights_raw, derivs_raw, u_knots,
                 min_bin_height=MIN_BIN_HEIGHT, min_derivative=MIN_DERIVATIVE,
                 min_scale=MIN_SCALE, clamp=(0.0, 1.0)):
        """Build from unconstrained head outputs.

        ``heights_raw`` is ``(..., n_bins)``; ``derivs_raw`` is ``(..., n_knots - 2)`` for the
        internal knots, or ``None`` to derive every slope from the secants (Fritsch-Carlson),
        which is monotonicity-preserving and costs no parameters.
        """
        n_bins = u_knots.numel() - 1
        if heights_raw.shape[-1] != n_bins:
            raise ValueError(f"expected {n_bins} bin heights, got {heights_raw.shape[-1]}")

        h = F.softmax(heights_raw, dim=-1)
        h = min_bin_height + (1.0 - n_bins * min_bin_height) * h
        v_knots = torch.cat(
            [torch.zeros_like(h[..., :1]), torch.cumsum(h, dim=-1)], dim=-1
        )
        # cumsum drifts by ~1e-7 in float32; the last knot must be exactly 1 or the
        # normalisation below inherits the drift at every pixel.
        v_knots = v_knots / v_knots[..., -1:].clamp_min(1e-12)

        widths = (u_knots[1:] - u_knots[:-1]).to(h.dtype)
        secants = h / widths                                    # (..., n_bins)

        if derivs_raw is None:
            derivs = cls._fritsch_carlson(secants)
        else:
            n_internal = u_knots.numel() - 2
            if derivs_raw.shape[-1] != n_internal:
                raise ValueError(
                    f"expected {n_internal} internal slopes, got {derivs_raw.shape[-1]}"
                )
            inner = F.softplus(derivs_raw) + min_derivative
            # The two boundary slopes are pinned to their own bin's secant rather than
            # learned. The tail reach is carried by the *height* of the outermost bin -- the
            # mass it holds -- so a free boundary slope would add parameters without adding
            # expressiveness, and an unlucky one makes the end bins badly conditioned.
            derivs = torch.cat([secants[..., :1], inner, secants[..., -1:]], dim=-1)

        return cls(anchor, scale.clamp_min(min_scale), v_knots, derivs, u_knots, clamp=clamp)

    @staticmethod
    def _fritsch_carlson(secants):
        """Harmonic mean of adjacent secants -- the monotone-preserving slope choice."""
        s0, s1 = secants[..., :-1], secants[..., 1:]
        inner = 2.0 / (1.0 / s0.clamp_min(1e-12) + 1.0 / s1.clamp_min(1e-12))
        return torch.cat([secants[..., :1], inner, secants[..., -1:]], dim=-1)

    @classmethod
    def from_channels(cls, raw, u_knots, learn_slopes=True, hm_t0=None,
                      scale_pre=None, **kw):
        """Decode ``(..., P)`` head channels: [anchor, scale, heights..., slopes...].

        ``hm_t0`` adds the persistence baseline to the anchor (the residual skip). ``scale_pre``
        overrides the raw scale channel, which is how the horizon-cumulative width is injected.
        """
        n_bins = u_knots.numel() - 1
        n_internal = u_knots.numel() - 2
        anchor = raw[..., 0]
        if hm_t0 is not None:
            anchor = anchor + hm_t0
        scale = F.softplus(raw[..., 1]) if scale_pre is None else scale_pre
        heights = raw[..., 2:2 + n_bins]
        slopes = raw[..., 2 + n_bins:2 + n_bins + n_internal] if learn_slopes else None
        return cls.from_raw(anchor, scale, heights, slopes, u_knots, **kw)

    # ---------------------------------------------------------------- evaluation

    def _bin_of_u(self, u):
        """Bin index for each u, as ``(..., n_u)`` long."""
        idx = torch.searchsorted(self.u_knots, u.contiguous(), right=True) - 1
        return idx.clamp(0, self.u_knots.numel() - 2)

    def _gather(self, tensor, idx):
        """Gather along the knot axis with a trailing u-axis, broadcasting the batch dims."""
        if idx.dim() == tensor.dim() - 1 + 1 and idx.shape[:-1] == tensor.shape[:-1]:
            return torch.gather(tensor, -1, idx)
        # u is shared across pixels: expand it to the batch shape first
        shape = tensor.shape[:-1] + idx.shape[-1:]
        return torch.gather(tensor, -1, idx.expand(shape))

    def _v_and_slope(self, u, need_slope=True):
        """Forward rational-quadratic map: ``v(u)`` and optionally ``dv/du``."""
        k = self._bin_of_u(u)
        uk = self.u_knots[k]
        w = self.u_knots[k + 1] - uk
        vk = self._gather(self.v_knots, k)
        vk1 = self._gather(self.v_knots, k + 1)
        dk = self._gather(self.derivs, k)
        dk1 = self._gather(self.derivs, k + 1)
        h = vk1 - vk
        s = h / w

        xi = ((u - uk) / w).clamp(0.0, 1.0)
        xi1 = 1.0 - xi
        common = dk1 + dk - 2.0 * s
        denom = s + common * xi * xi1
        v = vk + h * (s * xi * xi + dk * xi * xi1) / denom
        if not need_slope:
            return v, None
        num = s * s * (dk1 * xi * xi + 2.0 * s * xi * xi1 + dk * xi1 * xi1)
        return v, num / (denom * denom)

    def _g(self, v):
        return (v - self._v_mid.unsqueeze(-1)) / self._v_span.unsqueeze(-1)

    def ppf(self, u, clamp=True):
        """``Q(u)``. ``u`` may be a shared 1-D grid or a full ``(..., n_u)`` tensor."""
        u = torch.as_tensor(u, dtype=self.v_knots.dtype, device=self.v_knots.device)
        if u.dim() == 1:
            u = u.expand(self.anchor.shape + u.shape)
        v, _ = self._v_and_slope(u, need_slope=False)
        q = self.anchor.unsqueeze(-1) + self.scale.unsqueeze(-1) * self._g(v)
        return q.clamp(*self.clamp) if (clamp and self.clamp is not None) else q

    def triple(self):
        """``(lower, central, upper)`` without touching the spline -- all three are knots."""
        span = self._v_span
        lower = self.anchor + self.scale * (self._v_lo - self._v_mid) / span
        upper = self.anchor + self.scale * (self._v_hi - self._v_mid) / span
        if self.clamp is not None:
            lo, hi = self.clamp
            return lower.clamp(lo, hi), self.anchor.clamp(lo, hi), upper.clamp(lo, hi)
        return lower, self.anchor, upper

    def cdf(self, y):
        """``u* = Q^-1(y)``, on the **unclamped** spline. Also the PIT value.

        The unclamped inverse is the right one even when the clamp is active: the clamp only
        flattens ``Q`` where it would leave ``[0, 1]``, and any ``y`` strictly inside that
        range is crossed at exactly the same ``u``. Observed HM is never 0 or 1 (measured:
        min 0.00029, max 0.950 on southern Africa), so the degenerate case does not arise.
        """
        g = (y - self.anchor) / self.scale
        v = g * self._v_span + self._v_mid

        k = (torch.searchsorted(
                self.v_knots.contiguous(), v.unsqueeze(-1).contiguous(), right=True
             ) - 1).clamp(0, self.u_knots.numel() - 2).squeeze(-1)
        uk = self.u_knots[k]
        w = self.u_knots[k + 1] - uk
        vk = torch.gather(self.v_knots, -1, k.unsqueeze(-1)).squeeze(-1)
        vk1 = torch.gather(self.v_knots, -1, (k + 1).unsqueeze(-1)).squeeze(-1)
        dk = torch.gather(self.derivs, -1, k.unsqueeze(-1)).squeeze(-1)
        dk1 = torch.gather(self.derivs, -1, (k + 1).unsqueeze(-1)).squeeze(-1)
        h = vk1 - vk
        s = h / w

        yy = (v - vk).clamp(0.0, None)
        common = dk1 + dk - 2.0 * s
        a = h * (s - dk) + yy * common
        b = h * dk - yy * common
        c = -s * yy
        disc = (b * b - 4.0 * a * c).clamp_min(0.0)
        # `sqrt` is finite at 0 but its derivative is not, and `disc` is exactly 0 for whole
        # batches: `support_crossings` inverts the spline at the clamp bounds, which lie outside
        # the spline's range, so `b*b - 4ac` is negative for every pixel there and clamps to 0.
        # `SqrtBackward0` then computes `grad / (2*sqrt(0))` -- `inf` for a non-zero incoming
        # gradient, and `0/0 = NaN` for a zero one. A single such pixel puts NaN into every
        # parameter upstream of the CRPS path in one optimizer step, permanently, while the loss
        # stays finite and the surviving gradient norm stays ordinary, so nothing looks wrong.
        # Evaluating `sqrt` away from the singular point and selecting afterwards keeps the
        # forward bit-identical and gives the clamped pixels exactly zero gradient, which is what
        # a clamped-away root means.
        _live = disc > 0
        _root = torch.sqrt(torch.where(_live, disc, torch.ones_like(disc)))
        xi = (2.0 * c / (-b - torch.where(_live, _root, torch.zeros_like(_root)))).clamp(0.0, 1.0)
        u = uk + xi * w
        # Outside the spline's own range the crossing is the boundary, not a solved root.
        u = torch.where(v <= 0.0, torch.zeros_like(u), u)
        return torch.where(v >= 1.0, torch.ones_like(u), u)

    def log_dq_du(self, u):
        """``log dQ/du`` at ``u`` -- the negative log-density at ``Q(u)``."""
        u = u.unsqueeze(-1) if u.dim() == self.anchor.dim() else u
        _, dv = self._v_and_slope(u, need_slope=True)
        return (torch.log(self.scale.unsqueeze(-1).clamp_min(1e-12))
                + torch.log(dv.clamp_min(1e-12))
                - torch.log(self._v_span.unsqueeze(-1).clamp_min(1e-12))).squeeze(-1)

    def support_crossings(self):
        """``(u_a, u_b)``: where ``Q`` enters and leaves the physical range.

        Below ``u_a`` and above ``u_b`` the clamped quantile function is *constant*. Every
        integral over ``u`` has to know this, because a quadrature panel straddling one of
        those corners is integrating a function with a kink in it and loses most of its
        accuracy there. The clamp is not a rare edge case on this data: the median pixel sits
        at HM 0.053, so the lower corner is active over a large fraction of the map.
        """
        if self.clamp is None:
            zero = torch.zeros_like(self.anchor)
            return zero, zero + 1.0
        lo, hi = self.clamp
        return (self.cdf(torch.full_like(self.anchor, lo)),
                self.cdf(torch.full_like(self.anchor, hi)))

    def mean(self, n_nodes=8):
        """``E[Q] = int_0^1 Q(u) du`` -- the published central forecast.

        This, not the median, is what gets published: the mean is the RMSE-optimal point
        estimate and this residual is strongly right-skewed, so the two differ materially.
        The two flat pieces outside ``[u_a, u_b]`` contribute in closed form. Eight nodes per
        bin, not four: the spline is a *rational* function and a steep bin converges slowly,
        and this number has to be right to well under the int16 storage quantum of 3e-5 or the
        published central raster carries a quadrature artifact.
        """
        u_a, u_b = self.support_crossings()
        u, w = segment_nodes(self.u_knots, n_nodes, u_a, u_b)
        interior = (self.ppf(u, clamp=False) * w).sum(dim=-1)
        if self.clamp is None:
            return interior
        lo, hi = self.clamp
        return lo * u_a + interior + hi * (1.0 - u_b)


def _gauss_legendre(n_nodes, dtype, device):
    t, wt = np.polynomial.legendre.leggauss(n_nodes)
    return (torch.as_tensor((t + 1.0) / 2.0, dtype=dtype, device=device),
            torch.as_tensor(wt / 2.0, dtype=dtype, device=device))


def segment_nodes(u_knots, n_nodes, a, b):
    """Gauss-Legendre nodes over every fixed bin, clipped to ``[a, b]``.

    ``a`` and ``b`` are per-pixel tensors, so this is how an integral over a *data-dependent*
    sub-interval stays vectorised: a bin entirely outside ``[a, b]`` collapses to zero width
    and therefore zero weight, with no branch and no ragged tensor. Weights sum to ``b - a``.

    Splitting on the bins matters because the spline is only C1 at its knots; splitting on
    ``a`` and ``b`` matters because those are where the integrand's own kinks live. Both
    together are the difference between an exact rule and one that is quietly wrong by ~1%
    on exactly the pixels the tail questions are about.
    """
    dtype, device = u_knots.dtype, u_knots.device
    t, wt = _gauss_legendre(n_nodes, dtype, device)
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-1)
    lo = torch.minimum(torch.maximum(u_knots[:-1], a), b)
    hi = torch.minimum(torch.maximum(u_knots[1:], a), b)
    width = (hi - lo).unsqueeze(-1)
    u = (lo.unsqueeze(-1) + width * t).flatten(-2)
    w = (width * wt).flatten(-2)
    return u, w


def flat_piece_nodes(a, b, n_nodes=8):
    """Nodes on a single interval where ``Q`` is constant -- no spline evaluation needed.

    Used for the two clamped tails in the CRPS integral. The integrand there is a low-degree
    polynomial in ``u`` times a constant, so a fixed 8-point rule is exact for any tail
    weight worth using, and it costs nothing because ``Q`` is never touched.
    """
    t, wt = _gauss_legendre(n_nodes, a.dtype, a.device)
    a = a.unsqueeze(-1)
    width = (b.unsqueeze(-1) - a)
    return a + width * t, width * wt


# E0. How the published u-grid spends its 64 levels. ``normal`` is the incumbent's, uniform
# in the normal score and therefore SYMMETRIC about the median; the target is not. Measured on
# 10.3M land pixels, that grid spends 22% of its levels on 0.3% of the value range, packing
# consecutive levels within a few int16 quantisation steps of each other -- which is what
# renders as a picket fence -- while the segment carrying the +0.017 substantive-change mode
# near u = 0.78 gets nine.
#
# ``skew`` is an explicit budget per u-segment instead of a formula, so what it spends where is
# readable at a glance and arguable on its own terms. It moves eight levels out of the flat
# middle and into u 0.6-0.9, which the decile table says is under-resolved by ~2x relative to
# its share of the variation. This is export-only: the same trained spline, evaluated at
# different u. Nothing about any model changes and it is reversible by re-running the export.
U_GRID_SEGMENTS = (0.0, 0.1, 0.6, 0.9, 1.0)
U_GRID_BUDGETS = {
    #             [0,0.1)  [0.1,0.6)  [0.6,0.9)  [0.9,1]      share of value range:
    "normal": None,                             # 35.3%   0.3%   4.5%   59.9%
    "skew": (20, 6, 20, 18),
}


def output_u_grid(n=64, trunc=1e-4, tol=1e-9, spacing="normal", budget=None):
    """The fixed u-grid the quantile-function raster is written on.

    ``normal`` spaces uniformly in the normal score, so resolution concentrates at both ends
    equally. ``skew`` instead spends an explicit per-segment budget (:data:`U_GRID_BUDGETS`),
    spacing uniformly in the normal score *within* each segment.

    Either way the three gate quantiles are pinned so ``qf[0.025]`` and ``qf[0.975]``
    reproduce the published bounds *exactly* -- including after the prediction writer's
    overlap blending, which is a weighted average and therefore commutes with reading a level
    off the grid.

    The levels are guaranteed **strictly increasing by more than ``tol``**, and there are
    exactly ``n`` of them. Both matter: the normal-spaced set already contains a point
    indistinguishable from 0.5, and merging the gate in beside it produced a zero-width
    segment whose slope is 0/0. Downstream that came back as a NaN CRPS for every pixel -- a
    metric failure that looks exactly like a model failure.

    Defined here and nowhere else.
    """
    from scipy.stats import norm as _norm

    gates = np.array([U_LO, U_MID, U_HI])
    if spacing == "normal":
        cand = _norm.cdf(np.linspace(_norm.ppf(trunc), _norm.ppf(1.0 - trunc), n))
    elif spacing == "skew":
        b = tuple(budget if budget is not None else U_GRID_BUDGETS["skew"])
        if len(b) != len(U_GRID_SEGMENTS) - 1:
            raise ValueError(f"budget needs {len(U_GRID_SEGMENTS) - 1} entries, got {len(b)}")
        if sum(b) != n:
            raise ValueError(f"budget sums to {sum(b)}, not n={n}")
        parts = []
        for (lo, hi), k in zip(zip(U_GRID_SEGMENTS, U_GRID_SEGMENTS[1:]), b):
            z0 = _norm.ppf(max(lo, trunc))
            z1 = _norm.ppf(min(hi, 1.0 - trunc))
            parts.append(_norm.cdf(np.linspace(z0, z1, k)))
        cand = np.concatenate(parts)
    else:
        raise ValueError(f"unknown u-grid spacing {spacing!r}; have 'normal', 'skew'")

    cand = np.unique(cand)
    cand = cand[np.min(np.abs(cand[:, None] - gates[None, :]), axis=1) > tol]
    u = np.unique(np.concatenate([cand, gates]))
    # Trim by dropping the tightest non-gate gaps first, so the grid stays well conditioned.
    while u.size > n:
        gaps = np.diff(u)
        order = np.argsort(gaps)
        for k in order:
            drop = k if not np.any(np.abs(u[k] - gates) <= tol) else k + 1
            if np.any(np.abs(u[drop] - gates) <= tol):
                continue
            u = np.delete(u, drop)
            break
        else:                                                    # pragma: no cover
            raise ValueError(f"cannot trim the u grid to {n} levels without dropping a gate")
    if u.size != n or np.any(np.diff(u) <= tol):
        raise ValueError(f"u grid is {u.size} levels with min gap {np.diff(u).min():.2e}")
    return np.clip(u, 0.0, 1.0)


def rebuild(anchor, scale, v_knots, derivs, u_knots, clamp):
    """Reconstruct a spline from its own tensors.

    Exists so the expensive evaluations can go through ``torch.utils.checkpoint``, which
    takes a function of tensors and cannot be handed an object. The spline's quadrature holds
    ~280 node-evaluations per horizon in the autograd graph, which measured **22.4 GB of a
    24.6 GB card** at the production batch size -- 91% of the device, with an OOM waiting for
    the first unlucky chip mix twenty minutes into a fold. Recomputing in backward instead of
    storing costs about a third more compute for the memory, and the arithmetic is identical.
    """
    return QuantileSpline(anchor, scale, v_knots, derivs, u_knots, clamp=clamp)


def normal_height_bias(u_knots, trunc=1e-4):
    """Output-conv bias that starts the shape head at an exactly Gaussian marginal.

    ``softmax`` is shift-invariant, so biasing the head with ``log h`` makes its zero-weight
    output reproduce the bin heights ``h``. Taking ``h`` from the standard normal quantile
    function means the model begins where the incumbent's two-piece normal begins -- a
    ``Q(0.999)`` reach of 1.577 half-widths, the Gaussian value -- and every departure from
    Gaussianity afterwards is something training paid for. A uniform init would instead start
    at a shape nothing chose, and its tail reach would be an artifact of the knot spacing.
    """
    from scipy.stats import norm as _norm

    u = np.clip(np.asarray(u_knots, dtype=np.float64), trunc, 1.0 - trunc)
    z = _norm.ppf(u)
    v = (z - z[0]) / (z[-1] - z[0])
    return torch.as_tensor(np.log(np.diff(v)), dtype=torch.float32)


def splines_from_output(pred, n_horizons, u_knots, learn_slopes=True, clamp=(0.0, 1.0),
                        n_triple=12):
    """Decode the trailing spline block of a model output ``[B, n_triple + n_h * P, H, W]``.

    One decoder, used by the loss, the prediction writer and the scorer alike. This project
    has been bitten repeatedly by the same quantity being defined in several places (``Z975``
    lives in four files); the channel layout is defined here and nowhere else.

    Layout per horizon: ``[anchor, scale, heights..., slopes...]``, with ``anchor`` and
    ``scale`` already decoded and the rest raw.
    """
    p = n_spline_params(u_knots.numel(), learn_slopes)
    block = pred[:, n_triple:].movedim(1, -1)                    # [B, H, W, n_h * P]
    if block.shape[-1] != n_horizons * p:
        raise ValueError(
            f"expected {n_horizons * p} spline channels after {n_triple}, "
            f"got {block.shape[-1]}"
        )
    out = []
    for h in range(n_horizons):
        b = block[..., h * p:(h + 1) * p]
        out.append(QuantileSpline.from_channels(
            b, u_knots, learn_slopes=learn_slopes, scale_pre=b[..., 1], clamp=clamp))
    return out
