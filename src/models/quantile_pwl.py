"""Piecewise-linear quantile functions: the ISQF head (E1) and the PL spline head (E2).

Both emit a monotone ``Q(u)`` like :mod:`quantile_spline`, and both are C0 rather than C1 --
straight segments meeting at corners instead of rational-quadratic pieces meeting smoothly.
That single choice is what buys the thing this module exists for: **CRPS in closed form.**

Integrating the pinball loss against a linear ``Q`` gives a polynomial in ``u``, so there is
an exact expression and no quadrature at all. Against the incumbent's rational-quadratic
spline there is none, and the measured price of that was 168 spline evaluations per horizon,
22.4 GB of a 24.6 GB card before gradient checkpointing, and a split-quadrature apparatus
whose kink term is worth 15-230x in gradient accuracy (measured 2026-09-07). Whether C1 was
ever load-bearing has never been tested; E2 is that test.

What C0 costs: the density ``1/(dQ/du)`` becomes piecewise *constant*, so it jumps at every
knot. CRPS does not care. The NLL option and any density render do.

Two heads, one base class:

``PWLQuantile`` (E2)
    The incumbent's own construction with linear pieces substituted for rational-quadratic
    ones. Same fixed u-knots, same ``anchor``/``scale`` factorisation, same softmax heights,
    same clamp -- so an E2-vs-baseline A/B isolates the spline class and nothing else.

``ISQFQuantile`` (E1)
    Park et al. (2022), adapted for bounded support. Fixed outer quantile levels; learned
    values at them via cumulative positive increments; a linear spline between them. The one
    departure from the paper we keep is the **cumulative-in-horizon scale**; the one we drop
    is the exponential tails, which are unbounded and HM is not.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

MIN_BIN_HEIGHT = 1e-4
MIN_SCALE = 1e-6
# The --free_scale floor, and NOT a second spelling of MIN_SCALE. With the anchor/scale
# factorisation removed there is no `scale` channel to bound, so the same physical statement
# -- "a 95% interval narrower than this is not a forecast" -- has to be made about the SPAN
# of the ladder instead. The number is 3e-5 rather than MIN_SCALE's 1e-6 because that is what
# quantile_spline.py:130's own comment justifies and has always justified: the published
# product could not represent anything narrower while it was int16 x 1/32767. Inheriting the
# 1e-6 silently is how a floor stops binding (docs/conv_spline_phase.md section 4.1).
#
# Measured on b1, for the record: NO pixel is near either number -- width95's 1st percentile
# is 4.6e-3, 150x above this floor -- so this is insurance, not a live constraint. It is
# here because CRPS can drive the width to zero on the quiet majority of pixels once the
# factorisation stops holding it up, and `anchor + (v - v_mid)` then silently returns a
# constant, which reads downstream as a perfectly confident forecast rather than a degenerate
# one. In ABSOLUTE HM; callers working in normalized units must divide by hm_std.
MIN_FREE_SPAN = 3e-5

# The unit of a free-scale increment, and the reason --free_scale needs one at all.
#
# Every existing head feeds the shape channels through a SOFTMAX, so only their relative
# values matter and their absolute magnitude is arbitrary -- measured at init, |raw| averages
# 2.68. --free_scale makes that absolute magnitude BE the 95% width, so `|raw| + tol` starts
# the ladder 330x too wide (8 bins x 2.68 = 21.5 normalized = 3.3 HM against an intended
# 0.065 / 0.010 HM). At that width every knot saturates the [0, 1] clamp: pwl spans the whole
# range and isqf piles its whole ladder onto the upper bound and reports a width of ZERO.
# Both arms would then be measuring whether the optimiser can recover from a hopeless start.
#
# So one increment of raw is given a defined size: INITIAL_INC = initial_width_normalized
# divided by the number of bins spanning the 95% interval. This is a UNIT, not a
# normalisation -- no pixel's width is tied to another's and nothing is rescaled to a target
# span, so the width stays emergent, which is the whole point of the arm. It is the same
# thing `initial_width_normalized` already does for the incumbent's width head.
INITIAL_WIDTH_NORMALIZED = 0.065
# Smallest quantile gap the observable can justify, as a *density* ceiling. The core of the
# HM-change distribution has a robust sigma of 0.00069, so a kernel that narrow supports a
# density of at most ~578; the incumbent emits needles reaching 677. A gap floor of
# ``dp_k / F_MAX_DENSITY`` makes a picket fence structurally impossible rather than merely
# discouraged -- see ``gap_floor_heights``. Expressed in absolute HM; callers working in
# normalized units must scale it.
F_MAX_DENSITY = 578.0


def _cumulative_v(heights_raw, min_bin_height=MIN_BIN_HEIGHT, floor=None):
    """Softmax bin heights -> a monotone ladder ``v`` running exactly 0 -> 1.

    Shared with :class:`~src.models.quantile_spline.QuantileSpline` so the two head families
    normalise identically and an A/B between them is about the interpolation, not the
    parameterisation of the heights.
    """
    n_bins = heights_raw.shape[-1]
    h = F.softmax(heights_raw, dim=-1)
    h = min_bin_height + (1.0 - n_bins * min_bin_height) * h
    if floor is not None:
        # Reserve the floor, then distribute what is left. Keeps the sum at 1 and makes every
        # bin at least ``floor`` wide, which is the gap floor of E4.
        room = (1.0 - floor.sum(-1, keepdim=True)).clamp_min(1e-6)
        h = floor + room * (h / h.sum(-1, keepdim=True))
    v = torch.cat([torch.zeros_like(h[..., :1]), torch.cumsum(h, dim=-1)], dim=-1)
    # cumsum drifts ~1e-7 in float32; the last knot must be exactly 1 or the normalisation
    # below inherits the drift at every pixel.
    return v / v[..., -1:].clamp_min(1e-12)


def free_increments(inc_raw, u_knots, tol=1e-4, u_lo=0.025, u_hi=0.975,
                    width0=INITIAL_WIDTH_NORMALIZED):
    """``--free_scale``'s positive increments, in units where one of them means something.

    The paper's non-crossing construction is ``|.| + tol`` and that is kept exactly; what is
    added is the unit. See :data:`INITIAL_WIDTH_NORMALIZED` for why a unit is needed at all --
    without it the ladder starts 330x too wide and saturates the clamp.

    One definition, used by both free-scale families, because two spellings of one quantity
    is how E1b and E2a would end up on different width scales while claiming to differ only
    in where persistence enters.
    """
    from .quantile_spline import knot_index
    arr = np.asarray(u_knots.detach().cpu().numpy(), dtype=np.float64)
    n95 = max(knot_index(arr, u_hi) - knot_index(arr, u_lo), 1)
    return (inc_raw.abs() + tol) * (float(width0) / n95)


def _floor_free_span(inc, u_knots, hm_std=1.0, u_lo=0.025, u_hi=0.975):
    """Enforce :data:`MIN_FREE_SPAN` on the 95% span of a free-scale ladder.

    ``MIN_SCALE`` bounded the ``scale`` CHANNEL, and --free_scale deletes that channel, so
    the same physical statement has to be made about the span instead -- at construction,
    because without it CRPS can drive the width to zero on the quiet majority of pixels and
    ``anchor + (v - v_mid)`` then silently returns a constant, which reads downstream as a
    perfectly confident forecast rather than a degenerate one.
    """
    from .quantile_spline import knot_index
    arr = np.asarray(u_knots.detach().cpu().numpy(), dtype=np.float64)
    i_lo, i_hi = knot_index(arr, u_lo), knot_index(arr, u_hi)
    ladder = torch.cat([torch.zeros_like(inc[..., :1]), inc], dim=-1).cumsum(-1)
    span = (ladder[..., i_hi] - ladder[..., i_lo]).unsqueeze(-1)
    floor = MIN_FREE_SPAN / float(hm_std)
    return torch.where(span < floor, inc * (floor / span.clamp_min(1e-12)), inc)


def gap_floor_heights(u_knots, scale, f_max=F_MAX_DENSITY, hm_std=1.0):
    """Per-bin minimum height ``dp_k / (f_max * scale)`` in normalised ``v`` units (E4).

    The physical statement is "no segment may imply a density above ``f_max``". A bin holding
    probability ``dp_k`` over a value width ``w_k`` implies density ``dp_k / w_k``, and
    ``w_k = scale * h_k / v_span``; approximating ``v_span`` by 1 (it is 0.53 at Gaussian
    init, so this is conservative by ~2x) gives the floor above. Returns zeros where the
    floor would exceed what the ladder can hold, so it can never make the heights infeasible.
    """
    dp = (u_knots[1:] - u_knots[:-1]).to(scale.dtype)
    denom = (f_max / float(hm_std)) * scale.unsqueeze(-1).clamp_min(1e-9)
    floor = dp / denom
    total = floor.sum(-1, keepdim=True)
    # If the floors alone would exceed 1 the constraint is infeasible for this scale; back off
    # proportionally rather than producing a non-monotone or non-normalisable ladder.
    return torch.where(total > 0.9, floor * (0.9 / total), floor)


class _PWLBase:
    """A batch of monotone piecewise-linear quantile functions on a shared u-grid.

    ``u_knots`` is ``(K,)``; ``q_knots`` is ``(..., K)`` and non-decreasing. Everything below
    -- ppf, cdf, mean, and the closed-form CRPS -- is defined by those two alone, so the two
    subclasses differ only in how they build ``q_knots``.
    """

    def __init__(self, q_knots, u_knots, clamp=(0.0, 1.0)):
        self.q_knots = q_knots
        self.u_knots = u_knots
        self.clamp = clamp
        if clamp is not None:
            self.q_knots = q_knots.clamp(*clamp)

    # ------------------------------------------------------------------ evaluation

    def _seg(self, u):
        """Segment index for each u, as ``(..., n_u)`` long."""
        idx = torch.searchsorted(self.u_knots, u.contiguous(), right=True) - 1
        return idx.clamp(0, self.u_knots.numel() - 2)

    def ppf(self, u):
        """``Q(u)`` by linear interpolation between knots."""
        u = torch.as_tensor(u, dtype=self.q_knots.dtype, device=self.q_knots.device)
        if u.dim() == 1:
            u = u.expand(self.q_knots.shape[:-1] + u.shape)
        k = self._seg(u)
        uk, uk1 = self.u_knots[k], self.u_knots[k + 1]
        qk = torch.gather(self.q_knots, -1, k)
        qk1 = torch.gather(self.q_knots, -1, k + 1)
        t = ((u - uk) / (uk1 - uk)).clamp(0.0, 1.0)
        return qk + t * (qk1 - qk)

    def cdf(self, y):
        """``u* = Q^-1(y)`` -- the PIT value. Flat segments resolve to their left endpoint."""
        yy = y.unsqueeze(-1)
        k = (torch.searchsorted(self.q_knots.contiguous(), yy.contiguous(), right=True) - 1
             ).clamp(0, self.u_knots.numel() - 2).squeeze(-1)
        qk = torch.gather(self.q_knots, -1, k.unsqueeze(-1)).squeeze(-1)
        qk1 = torch.gather(self.q_knots, -1, (k + 1).unsqueeze(-1)).squeeze(-1)
        uk, uk1 = self.u_knots[k], self.u_knots[k + 1]
        span = (qk1 - qk)
        t = torch.where(span > 1e-12, (y - qk) / span.clamp_min(1e-12), torch.zeros_like(y))
        u = uk + t.clamp(0.0, 1.0) * (uk1 - uk)
        u = torch.where(y <= self.q_knots[..., 0], torch.zeros_like(u), u)
        return torch.where(y >= self.q_knots[..., -1], torch.ones_like(u), u)

    def triple(self, lo=0.025, mid=0.5, hi=0.975):
        """``(lower, central, upper)``. Exact knots are a gather, not an interpolation."""
        levels = torch.tensor([lo, mid, hi], dtype=self.q_knots.dtype,
                              device=self.q_knots.device)
        q = self.ppf(levels)
        return q[..., 0], q[..., 1], q[..., 2]

    def mean(self):
        """``E[Q] = int_0^1 Q(u) du`` -- exact, since Q is linear on each segment.

        The trapezoid rule is not an approximation here; it is the integral.
        """
        w = (self.u_knots[1:] - self.u_knots[:-1]).to(self.q_knots.dtype)
        mid = 0.5 * (self.q_knots[..., 1:] + self.q_knots[..., :-1])
        return (mid * w).sum(-1)

    def density_at_knots(self):
        """``dp_k / dq_k`` per segment -- the implied density, and the needle diagnostic.

        A segment whose value width has collapsed reports a huge density here. That is the
        quantity the E4 gap floor bounds and the scorecard's ``max_density`` gate reads.
        """
        dp = (self.u_knots[1:] - self.u_knots[:-1]).to(self.q_knots.dtype)
        dq = (self.q_knots[..., 1:] - self.q_knots[..., :-1]).clamp_min(1e-12)
        return dp / dq

    # ------------------------------------------------------------------ closed-form CRPS

    def crps(self, y):
        """``2 * int_0^1 rho_u(y - Q(u)) du``, in closed form. No quadrature, no nodes.

        On one segment ``[a, b]`` with ``Q`` linear, the pinball integrand is piecewise
        quadratic in ``u`` with at most one kink -- at ``u*``, where ``Q`` crosses ``y``.
        Both pieces integrate exactly, and the crossing is a root of a *linear* function, so
        it has no branch to get wrong. (The incumbent's closed form failed on exactly this
        point: with a rational-quadratic ``Q`` the crossing clips to a segment end far more
        often than it lands inside one, and the second piece's error at its own origin is not
        zero. Here that cannot arise -- the clip is the same linear formula.)

        Checked against a dense numerical reference in ``tests/test_quantile_pwl.py``; the
        two share no code, which is the only way this project has ever caught a CRPS bug.
        """
        if float(self.u_knots[0]) != 0.0 or float(self.u_knots[-1]) != 1.0:
            raise ValueError(
                "the closed form assumes the u-grid spans [0, 1]; a grid that stops short "
                "needs two flat-tail terms whose sign depends on which side of the end knot "
                "y falls, and getting that wrong is silent. Add the endpoints instead.")

        a = self.u_knots[:-1]
        w = (self.u_knots[1:] - a).to(self.q_knots.dtype)
        qa = self.q_knots[..., :-1]
        qb = self.q_knots[..., 1:]
        yy = y.unsqueeze(-1)

        # e(u) = y - Q(u) is linear on the segment, and *decreasing* in t because Q is
        # non-decreasing. So the positive-e region is always [0, t*] and the negative-e
        # region always [t*, 1] -- there is no "which branch comes first" to get wrong, and
        # clamping t* to [0, 1] already covers a segment that never crosses at all.
        ea, eb = yy - qa, yy - qb
        denom = ea - eb                                  # >= 0
        safe = denom.abs() > 1e-12
        t_star = torch.where(
            safe,
            ea / torch.where(safe, denom, torch.ones_like(denom)),
            # A flat segment: e is constant, so the whole piece is positive (t*=1) or
            # negative (t*=0) with no crossing inside it.
            torch.where(ea >= 0, torch.ones_like(ea), torch.zeros_like(ea)),
        ).clamp(0.0, 1.0)

        # rho_u(e) = e*(u - 1{e<0}). With u and e both linear in t the integrand is a
        # quadratic, so Simpson's rule is exact (it is exact through cubics) -- this is a
        # closed form written as a three-point evaluation, not a quadrature approximation.
        def _piece(t0, t1, ind):
            def integrand(t):
                return (ea + t * (eb - ea)) * ((a + t * w) - ind)
            tm = 0.5 * (t0 + t1)
            return (t1 - t0) * w / 6.0 * (integrand(t0) + 4.0 * integrand(tm) + integrand(t1))

        zero, one = torch.zeros_like(t_star), torch.ones_like(t_star)
        return 2.0 * (_piece(zero, t_star, 0.0) + _piece(t_star, one, 1.0)).sum(-1)

    def crps_z(self, y, hm_t0, s: float):
        """CRPS scored in ``z = asinh((. - hm_t0) / s)`` -- the E3 core-weighting term.

        ``asinh`` of a linear function is not linear, so strictly the transformed quantile
        function is curved. But the transform of a piecewise-linear ``Q`` is naturally read
        as the piecewise-linear interpolant *of the transformed knots*, which is the same
        object the model would have emitted had it been trained in z from the start -- and
        under that reading the closed form carries over unchanged, exactly, by transforming
        the knots and calling it again. No quadrature enters, which is the whole reason the
        C0 families exist.
        """
        z = type(self).__new__(_PWLBase)
        _PWLBase.__init__(z, torch.asinh((self.q_knots - hm_t0.unsqueeze(-1)) / s),
                          self.u_knots, clamp=None)
        return z.crps(torch.asinh((y - hm_t0) / s))


class PWLQuantile(_PWLBase):
    """E2: the incumbent's construction with linear pieces (``anchor + scale * g(v)``).

    Identical to :class:`QuantileSpline` in every respect except the interpolation, so the
    published triple is still an exact lookup, ``scale`` is still the 95% width, the shape is
    still scale-free, and the clamp is still the physical range.
    """

    def __init__(self, anchor, scale, v_knots, u_knots, clamp=(0.0, 1.0),
                 u_lo=0.025, u_mid=0.5, u_hi=0.975):
        from .quantile_spline import knot_index
        arr = u_knots.detach().cpu().numpy()
        i_lo, i_mid, i_hi = (knot_index(arr, u_lo), knot_index(arr, u_mid),
                             knot_index(arr, u_hi))
        v_mid = v_knots[..., i_mid]
        v_span = v_knots[..., i_hi] - v_knots[..., i_lo]
        g = (v_knots - v_mid.unsqueeze(-1)) / v_span.unsqueeze(-1)
        super().__init__(anchor.unsqueeze(-1) + scale.unsqueeze(-1) * g, u_knots, clamp=clamp)
        self.anchor, self.scale = anchor, scale

    @classmethod
    def from_channels(cls, raw, u_knots, hm_t0=None, scale_pre=None, clamp=(0.0, 1.0),
                      gap_floor=False, hm_std=1.0, free_scale=False, tol=1e-4, **kw):
        """Decode ``[anchor, scale, heights...]``. Slopes are not a parameter here.

        ``free_scale`` (E2a) removes the anchor/scale factorisation: unnormalised positive
        increments carry the width directly, there is no ``scale`` channel, and ``Q(u)`` is
        ``anchor + (v - v_mid)`` with the ``/ v_span`` division dropped. Note this is NOT
        "strip the scale channel": ``_cumulative_v`` softmaxes and then divides by
        ``v[..., -1:]``, so the ladder spans exactly 0 -> 1 twice over and removing ``scale``
        naively would hand every pixel a span of 1.0 -- a broken model rather than a free one.
        """
        n_bins = u_knots.numel() - 1
        anchor = raw[..., 0] if hm_t0 is None else raw[..., 0] + hm_t0
        if free_scale:
            if gap_floor:
                # gap_floor_heights derives its floor in NORMALISED v units from
                # dp_k / (f_max * scale), and --free_scale deletes both terms. Restating it on
                # the increments in absolute HM is the simpler form of the same physical
                # claim, but it is a re-derivation, not a combination -- so refuse rather than
                # silently apply a floor computed from a quantity that no longer exists.
                raise ValueError(
                    "--spline_gap_floor and --free_scale do not compose: the floor is "
                    "dp_k / (f_max * scale) in normalised v units and free_scale removes "
                    "both scale and the normalisation. Restate it as inc_k >= dp_k / f_max "
                    "in absolute HM first (docs/conv_spline_phase.md section 4.1).")
            return cls.from_free_increments(anchor, raw[..., 2:2 + n_bins], u_knots,
                                            clamp=clamp, tol=tol, hm_std=hm_std, **kw)
        scale = (F.softplus(raw[..., 1]) if scale_pre is None else scale_pre).clamp_min(MIN_SCALE)
        floor = gap_floor_heights(u_knots, scale, hm_std=hm_std) if gap_floor else None
        v = _cumulative_v(raw[..., 2:2 + n_bins], floor=floor)
        return cls(anchor, scale, v, u_knots, clamp=clamp, **kw)

    @classmethod
    def from_free_increments(cls, anchor, inc_raw, u_knots, clamp=(0.0, 1.0), tol=1e-4,
                             hm_std=1.0, u_lo=0.025, u_mid=0.5, u_hi=0.975):
        """E2a's ladder: ``anchor + (v - v_mid)`` from unnormalised positive increments.

        The 95% width is whatever the increments imply, per horizon, with no cross-horizon
        accumulation -- emergent rather than injected. ``anchor`` stays at ``Q(0.5)`` and
        still receives ``hm_t0`` through the zero-init persistence skip, and the u-knots stay
        fixed, so 0.025 / 0.5 / 0.975 remain exact knots and ``triple()`` stays a gather.
        """
        from .quantile_spline import knot_index
        arr = np.asarray(u_knots.detach().cpu().numpy(), dtype=np.float64)
        i_lo, i_mid, i_hi = (knot_index(arr, u_lo), knot_index(arr, u_mid),
                             knot_index(arr, u_hi))
        inc = free_increments(inc_raw, u_knots, tol=tol, u_lo=u_lo, u_hi=u_hi)
        inc = _floor_free_span(inc, u_knots, hm_std=hm_std, u_lo=u_lo, u_hi=u_hi)
        v = torch.cat([torch.zeros_like(inc[..., :1]), inc], dim=-1).cumsum(dim=-1)
        q = anchor.unsqueeze(-1) + (v - v[..., i_mid].unsqueeze(-1))
        obj = _PWLBase.__new__(cls)
        _PWLBase.__init__(obj, q, u_knots, clamp=clamp)
        obj.anchor = anchor
        # No scale channel exists under --free_scale. Expose the EMERGENT 95% width under a
        # different name: anything still reading `.scale` is reading a quantity this head does
        # not have, and should fail loudly rather than pick up a plausible number.
        obj.free_span = (q[..., i_hi] - q[..., i_lo])
        return obj


class ISQFQuantile(_PWLBase):
    """E1: Incremental Spline Quantile Function (Park et al. 2022), bounded.

    The paper's shape, with two deliberate departures:

    **Dropped -- the exponential tails.** ``q(a) = a_l log a + b_l`` is unbounded on both
    sides and HM lives on [0, 1], so the mechanism is physically wrong here even though it is
    the paper's headline contribution. The outermost knots sit at u = 0.001 / 0.999 and the
    clamp does the rest.

    **Kept -- the cumulative-in-horizon scale.** The paper's Seq2Seq horizons carry
    independent parameters. Over 5-20 years the 95% width must not be free to shrink with
    lead time, so ``scale`` accumulates exactly as it does for the incumbent head.

    Location and scale therefore come from a free first knot plus an external scale, which is
    the paper's arrangement rather than the incumbent's anchor/scale factorisation. That is
    the point of running E1 as a separate head instead of a flag: it tests the whole
    parameterisation, not one piece of it.
    """

    def __init__(self, q0, increments, u_knots, clamp=(0.0, 1.0)):
        q = torch.cat([q0.unsqueeze(-1), increments], dim=-1).cumsum(dim=-1)
        super().__init__(q, u_knots, clamp=clamp)
        self.q0 = q0

    @classmethod
    def from_channels(cls, raw, u_knots, hm_t0=None, scale_pre=None, clamp=(0.0, 1.0),
                      tol=1e-4, tails=False, space="logit", hm_std=1.0, **_):
        """Decode ``[q0, scale, increments...]`` -- the same channel contract as every head.

        One layout for all three families (``[location, scale, shape...]``) rather than one
        per family. The alternative saves one channel for ISQF and buys an off-by-one that
        only shows up as a wrong-shaped distribution, which is not a trade worth taking.

        ``q0`` is the free first knot (the location), anchored on persistence through the
        same zero-init residual skip the incumbent uses. The increments are ``|.| + tol``,
        the paper's own non-crossing construction, then rescaled so ``scale_pre`` is the 95%
        width and the horizon-cumulative constraint applies to the whole ladder at once.
        """
        n_inc = u_knots.numel() - 1
        q0 = raw[..., 0] if hm_t0 is None else raw[..., 0] + hm_t0
        if scale_pre is None:
            # --free_scale (E1b). The renormalisation below is what this arm removes, so the
            # increments carry the width directly -- in the same unit E2a uses, or the two
            # arms would sit on different width scales while claiming to differ only in
            # where persistence enters the ladder.
            inc = free_increments(raw[..., 2:2 + n_inc], u_knots, tol=tol)
            inc = _floor_free_span(inc, u_knots, hm_std=hm_std)
        else:
            inc = raw[..., 2:2 + n_inc].abs() + tol
        if scale_pre is not None:
            # Normalise the ladder to unit 95% span first, so `scale_pre` means the same
            # quantity here as it does for the incumbent head and the two are comparable.
            from .quantile_spline import U_HI, U_LO, knot_index
            arr = np.asarray(u_knots.detach().cpu().numpy(), dtype=np.float64)
            # knot_index, not argmin: a near miss is a bias, and argmin never says so.
            i_lo, i_hi = knot_index(arr, U_LO), knot_index(arr, U_HI)
            ladder = torch.cat([torch.zeros_like(inc[..., :1]), inc], dim=-1).cumsum(-1)
            span = (ladder[..., i_hi] - ladder[..., i_lo]).clamp_min(1e-9)
            inc = inc * (scale_pre / span).unsqueeze(-1)
        obj = cls(q0, inc, u_knots, clamp=clamp)
        obj._has_tails = bool(tails)
        if tails:
            obj._attach_tails(u_knots, raw, space, clamp)
        return obj

    def _attach_tails(self, u_knots, raw, space, clamp):
        """E1a: learned exponential tail rates REPLACING the outermost bins.

        They cannot attach *beyond* the outer knots. ``validate_knots`` requires the grid to
        span [0, 1] exactly, so there is no u out there and a tail placed at one would cover
        an empty set -- accepted, logged, and inert. Instead the tails own
        ``[u[0], u[1]]`` and ``[u[-2], u[-1]]`` (0.001 of probability each on every preset
        this repo has), anchored at the 0.001 / 0.999 knots, which is also the paper's shape:
        their spline covers [alpha, 1-alpha] and the tails cover the rest.

        The rates are trainable and live in a transformed space where an unbounded tail is
        admissible -- ``logit(HM)`` (symmetric) or ``-log(1-HM)`` (unbounded above only).
        Mapping back through the inverse respects [0, 1] structurally, which is what makes
        the clamp redundant here rather than load-bearing.
        """
        arr = np.asarray(u_knots.detach().cpu().numpy(), dtype=np.float64)
        lo, hi = float(clamp[0]), float(clamp[1])
        eps = 1e-6
        def fwd(x):
            z = ((x - lo) / max(hi - lo, 1e-12)).clamp(eps, 1.0 - eps)
            return torch.log(z / (1.0 - z)) if space == "logit" else -torch.log1p(-z)
        def inv(z):
            y = torch.sigmoid(z) if space == "logit" else (1.0 - torch.exp(-z.clamp_min(0.0)))
            return lo + y * (hi - lo)
        # Two trainable rates, taken from the two channels the ladder does not use. beta > 0
        # via softplus: a tail that can flatten or invert is not a tail.
        self.beta_l = F.softplus(raw[..., -2]) + 1e-3
        self.beta_r = F.softplus(raw[..., -1]) + 1e-3
        self.tail_space, self._fwd, self._inv = space, fwd, inv
        self.u1, self.un = float(arr[1]), float(arr[-2])
        # Anchored at the knot the tail replaces, so Q is continuous there by construction.
        self._z1 = fwd(self.q_knots[..., 1])
        self._zn = fwd(self.q_knots[..., -2])
        # The endpoint knots are still written, because the exported raster and every
        # consumer reading q_knots directly needs a finite monotone grid. But ``ppf`` below
        # evaluates the tail wherever it is ASKED instead of interpolating to these -- the
        # difference between a learned exponential tail and a straight line to a distant
        # point. u = 0 and u = 1 are exactly where the tail diverges, so the stored endpoint
        # is taken one part in a thousand inside.
        u0e = max(float(arr[0]), self.u1 * 1e-3)
        uNe = min(float(arr[-1]), 1.0 - (1.0 - self.un) * 1e-3)
        q_lo = self._tail_lo(torch.as_tensor(u0e, dtype=self.q_knots.dtype))
        q_hi = self._tail_hi(torch.as_tensor(uNe, dtype=self.q_knots.dtype))
        new_q = torch.cat([q_lo.unsqueeze(-1), self.q_knots[..., 1:-1],
                           q_hi.unsqueeze(-1)], dim=-1)
        # cummax, not an assert: the two seams are the only places monotonicity is not
        # structural, and a tail that crossed its own anchor would read downstream as a
        # collapsed segment -- i.e. as a fence, the very thing E1a is judged on.
        self.q_knots = torch.cummax(new_q, dim=-1).values

    def _align(self, u, z, beta):
        """Broadcast the per-pixel anchor and rate against a trailing u axis.

        ``ppf`` asks for ``(..., n_u)`` while the anchors are ``(...)`` per pixel; the
        endpoint call in ``_attach_tails`` passes a scalar. One helper so the two call sites
        cannot disagree about which one needs the extra axis.
        """
        while z.dim() < u.dim():
            z, beta = z.unsqueeze(-1), beta.unsqueeze(-1)
        return z, beta

    def _tail_lo(self, u):
        """``inv(z(u1) - beta_l * log(u1 / u))`` -- the lower tail, for ``u <= u1``."""
        z, beta = self._align(u, self._z1, self.beta_l)
        lg = torch.log(torch.as_tensor(self.u1, dtype=u.dtype, device=u.device)
                       / u.clamp_min(1e-12))
        return self._inv(z - beta * lg)

    def _tail_hi(self, u):
        """``inv(z(un) + beta_r * log((1-un) / (1-u)))`` -- the upper tail, for ``u >= un``."""
        z, beta = self._align(u, self._zn, self.beta_r)
        lg = torch.log(torch.as_tensor(1.0 - self.un, dtype=u.dtype, device=u.device)
                       / (1.0 - u).clamp_min(1e-12))
        return self._inv(z + beta * lg)

    def ppf(self, u):
        """``Q(u)``, using the exponential tails wherever they own the domain.

        Without this the tails would exist only as two relocated endpoint knots, and every u
        between 0.999 and 1 -- which is precisely where the exported raster samples the far
        tail -- would come from a straight line to a distant point. The numbers move either
        way, so the omission would have been invisible: E1a would have been judged on a
        mechanism it was not running.
        """
        if not getattr(self, "_has_tails", False):
            return super().ppf(u)
        u = torch.as_tensor(u, dtype=self.q_knots.dtype, device=self.q_knots.device)
        if u.dim() == 1:
            u = u.expand(self.q_knots.shape[:-1] + u.shape)
        out = super().ppf(u)
        # The tail is anchored so that it MEETS its knot exactly -- but `inv(fwd(q))` does not
        # round-trip to the last bit when q sits on the transform's own clamp, which showed up
        # as a -3e-06 backward step right at the seam. Tiny (4.6e-07 HM, below the int16
        # quantum) and entirely cosmetic in value, but Q must be non-decreasing: the scorer's
        # qf reader refuses a non-increasing grid rather than passing NaN downstream, so a
        # seam like this would fail a real run rather than shade a number. Bounding each tail
        # by its own anchor is exact and costs one min/max.
        if bool((u < self.u1).any()):
            anchor = self.q_knots[..., 1].unsqueeze(-1) if u.dim() > self.q_knots.dim() - 1 \
                else self.q_knots[..., 1]
            out = torch.where(u < self.u1,
                              torch.minimum(self._tail_lo(u.clamp_max(self.u1)), anchor), out)
        if bool((u > self.un).any()):
            anchor = self.q_knots[..., -2].unsqueeze(-1) if u.dim() > self.q_knots.dim() - 1 \
                else self.q_knots[..., -2]
            out = torch.where(u > self.un,
                              torch.maximum(self._tail_hi(u.clamp_min(self.un)), anchor), out)
        return out if self.clamp is None else out.clamp(*self.clamp)


def n_pwl_params(n_knots: int, family: str = "pwl", tails: bool = False) -> int:
    """Channels one horizon needs: ``[location, scale, shape...]``, shape being one per bin.

    Identical for both families by construction -- see ``ISQFQuantile.from_channels``.
    ``tails`` (E1a) appends two more for the trainable rates beta_L and beta_R, which is why
    the phase doc counts E1a at 18 params/horizon against E1's 16. They are appended rather
    than carved out of the ladder so that E1 and E1a differ by the tails and nothing else.
    """
    if family not in ("pwl", "isqf"):
        raise ValueError(f"unknown piecewise-linear family {family!r}")
    if tails and family != "isqf":
        raise ValueError("learned tails are an isqf feature; --isqf_tails with "
                         f"head_family={family!r} would allocate two channels nothing reads")
    return 2 + (n_knots - 1) + (2 if tails else 0)
