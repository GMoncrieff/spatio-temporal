"""Phase 3 — Gaussian copula coupling of correlated fields to the per-pixel quantiles.

The marginal at each pixel is a **median-spliced two-piece normal** fitted in closed form
from the published triple (lower, central, upper):

    F(x) = Phi((x − central)/sigma_L)   for x <  central
           Phi((x − central)/sigma_R)   for x >= central

    sigma_L = (central − lower)/z_0.975,  sigma_R = (upper − central)/z_0.975

so that by construction ``ppf(0.5) = central``, ``ppf(0.025) = lower`` and
``ppf(0.975) = upper`` — the three constraints the ensemble must reproduce exactly. A
per-pixel optimizer is out of the question at 684M pixels, hence the closed form; the
median splice (rather than the classical mode-spliced split normal) is what keeps the
median equal to the central forecast, which is the hard constraint here. The density has a
step at the median when sigma_L != sigma_R; that is the accepted cost of preserving all
three quantiles exactly with a two-parameter family.

Because u = Phi(z), the sampling step collapses to a multiplication and never needs Phi at
all: ``member = central + (z < 0 ? sigma_L : sigma_R) * z``.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import norm

Z975 = float(norm.ppf(0.975))  # 1.959963985
INT16_SENTINEL = -32768
DEFAULT_SCALE = 1.0 / 32767.0  # HM is bounded on [0, 1]; the observed max reaches 1.0
# Grid edge for the empirical marginal shape. Past this the mapping continues with
# unit slope in z rather than clamping, so the rare extreme residuals stay representable.
EDGE_U = 1e-4


def fit_marginal_two_piece_normal(lower, central, upper, min_sigma: float = 1e-6):
    """Closed-form marginal parameters from a per-pixel quantile triple."""
    lower = np.asarray(lower, dtype=np.float32)
    central = np.asarray(central, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    sigma_l = np.maximum((central - lower) / Z975, min_sigma)
    sigma_r = np.maximum((upper - central) / Z975, min_sigma)
    return {"loc": central, "scale_left": sigma_l.astype(np.float32),
            "scale_right": sigma_r.astype(np.float32)}


def marginal_ppf(u, params, clip=(0.0, 1.0)):
    """Vectorized inverse CDF of the median-spliced two-piece normal."""
    u = np.asarray(u, dtype=np.float64)
    z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return marginal_from_z(z, params, clip=clip)


def marginal_from_z(z, params, clip=(0.0, 1.0)):
    """Same as :func:`marginal_ppf` but taking the normal score directly (u = Phi(z)).

    With a ``shape`` in ``params`` the two-piece *normal* becomes a two-piece *empirical*:
    the normal score is remapped through the residual's own standardized quantile function
    before being scaled. See :func:`fit_residual_shape` for why.
    """
    loc = params["loc"]
    sl = params["scale_left"]
    sr = params["scale_right"]
    shape = params.get("shape")
    zz = z if shape is None else apply_shape(z, shape)
    scale = np.where(zz < 0, sl, sr)
    out = loc + scale * zz
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


# --------------------------------------------------------------------------------------
# Empirical marginal shape
# --------------------------------------------------------------------------------------
def _dedupe_knots(u, tol: float = 1e-12):
    """Sort and drop knots that are indistinguishable from their neighbour.

    ``np.unique`` is not enough: the default bound's mirror is ``1.0 - 0.975 ==
    0.025000000000000022``, which is a *different float* from the pinned ``0.025`` and so
    survives as a second knot 2e-17 away. Two consequences, both bad — the anchor pinning
    writes to one of the pair and the bound reads the other, and the GPU interpolator gets a
    bucket of width 2e-17 to divide by.
    """
    u = np.unique(np.asarray(u, dtype=np.float64))
    return u[np.concatenate([[True], np.diff(u) > tol])]


def fit_residual_shape(standardized_residual, n_knots: int = 512, min_count: int = 10_000,
                       u_bound: float = 0.975, u_bound_lo=None, extra_knots=()):
    """The residual's own distribution shape, in units of its 95% half-width.

    The two-piece normal fixes the 2.5/50/97.5 quantiles and then fills everything between
    them with Gaussian mass. Measured on this model's hindcast residuals, that is badly
    wrong in the body: where a Gaussian puts 62% of its mass beyond 0.5 sigma, the residual
    puts 19-25%, with a kurtosis of 900-20000 against the Gaussian's 3. The consequence is
    an ensemble whose members scatter 3.5x too much moderate change over ground the
    observation leaves flat — measured as T6.1 (3.6x), T8.1 (up to 6.9x), T6.5 and T7.3,
    which are four scorecard rows for one defect.

    The fix keeps every hard gate and replaces only the shape. Each tail is stored as a
    monotone quantile function normalized so that ``S(0.5) = 0`` and ``|S(0.025)| =
    |S(0.975)| = z_{0.975}``. Pushing the normal score through it therefore leaves the
    median exactly at the central forecast (T5.1) and the 2.5/97.5 percentiles exactly at
    the published bounds (T5.2), while the mass between them follows the data.

    Parameters
    ----------
    standardized_residual
        ``(Y - central) / sigma`` with ``sigma`` the published 95% half-width divided by
        ``z_{0.975}`` — i.e. the residual in the same units the two-piece normal works in,
        so a perfectly Gaussian residual returns the identity.
    u_bound, u_bound_lo
        Where the fitted shape stops and the unit-slope continuation begins, upper and
        lower side. ``u_bound_lo`` defaults to ``1 - u_bound``, i.e. symmetric. Both are
        stored on the returned shape so the mapping travels with it. The two sides are
        separate because this residual's tails fail in opposite directions — see
        :func:`shape_bounds`.
    extra_knots
        Additional u values to pin as exact knots. Used when several shapes must share one
        u-grid — see :func:`stack_shapes`.
    """
    e = np.asarray(standardized_residual, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size < min_count:
        return None

    # Centre on the median so u = 0.5 maps to exactly 0, which is T5.1.
    ec = e - float(np.median(e))
    lo_ref = abs(float(np.quantile(ec, 0.025)))
    hi_ref = float(np.quantile(ec, 0.975))
    if not (lo_ref > 0 and hi_ref > 0):
        return None

    # Store the quantile function on a u-grid, each side normalized by its own 95% point so
    # u = 0.025 / 0.975 map to exactly -/+ Z975 and the published bounds are reproduced
    # whatever the shape between them. A Gaussian residual therefore returns the identity.
    # Odd knot count so u = 0.5 lands exactly on a knot; the centre is then pinned to zero
    # rather than left to interpolation, because T5.1 is an exact-equality gate and a
    # 1e-5 offset there is a real (if tiny) violation of "median == central forecast".
    # The three gate quantiles must be *knots*, not interpolated between them. A uniform
    # grid does not contain 0.025/0.5/0.975, and because the quantile function is steep near
    # the bounds the interpolation error there is a systematic bias, not noise — it does not
    # shrink with member count. Caught by an M=400 run in which T5.2 got *worse* rather than
    # better, which is only possible against an MC-scaled tolerance if a bias is present.
    #
    # u_bound and its mirror join the anchor set for the same reason: apply_shape switches
    # to the unit-slope continuation there, and a join that falls between knots is a bias
    # in exactly the tail the band conditioning exists to get right.
    ub_lo = float(1.0 - u_bound if u_bound_lo is None else u_bound_lo)
    u = np.linspace(EDGE_U, 1.0 - EDGE_U, int(n_knots))
    anchors = [0.025, 0.5, 0.975, float(u_bound), ub_lo]
    u = _dedupe_knots(np.concatenate([u, anchors,
                                      np.asarray(extra_knots, dtype=np.float64)]))
    q = np.quantile(ec, u)
    vals = np.where(q < 0, q / lo_ref * Z975, q / hi_ref * Z975)
    for anchor, target in ((0.025, -Z975), (0.5, 0.0), (0.975, Z975)):
        vals[int(np.searchsorted(u, anchor))] = target
    vals = np.maximum.accumulate(vals)  # quantile functions are monotone; keep it exact
    return {"u": u.tolist(), "q": vals.tolist(), "n": int(e.size),
            "u_bound": float(u_bound), "u_bound_lo": ub_lo}


def shape_bounds(shape):
    """``(z_hi, z_lo, offset_hi, offset_lo)`` — the unit-slope continuation past the fit.

    Above ``z_hi`` and below ``z_lo`` the mapping is ``z + offset``, one offset per side,
    so these four numbers are the whole continuation.

    **The two sides are independent**, because the two tails of this residual fail in
    opposite directions. The upper tail is far too thin — the far field cannot produce the
    rare distant change the observation contains — while the lower tail is already 3-28x
    too hot. Raising both bounds together buys the upper tail and wrecks the lower one:
    measured at h=20, ``P(Δ < −0.15)`` goes from 1e-6 to 1.6e-3 against an observed
    **0.000000 in every band**, taking T6.2, T6.3, T6.5 and T8.4 with it.

    Written as a *shift of z* rather than as ``q(u_bound) + (z - z_bound)`` deliberately.
    The two are equal in exact arithmetic, but at the default bound ``q(0.975) == Z975 ==
    z_bound`` makes the offset exactly ``0.0``, so the continuation returns ``z`` bit for
    bit. The algebraically identical form leaves 1e-15 of float error, which is enough to
    make "the default is unchanged" untestable by equality.
    """
    ub = float(shape.get("u_bound", 0.975))
    ub_lo = float(shape.get("u_bound_lo", 1.0 - ub))
    ug = np.asarray(shape["u"], dtype=np.float64)
    qv = np.asarray(shape["q"], dtype=np.float64)
    # The lower z is taken by reflection rather than as ppf(ub_lo) directly: scipy's ppf is
    # not exactly antisymmetric (they differ by 4e-16), and the anchors were pinned to
    # +/-Z975 == +/-ppf(0.975). Reflecting makes the default offset exactly 0.0 instead of
    # 4e-16, which is again the difference between "unchanged" and "nearly unchanged".
    z_hi, z_lo = float(norm.ppf(ub)), -float(norm.ppf(1.0 - ub_lo))
    return z_hi, z_lo, _q_at(ug, qv, ub) - z_hi, _q_at(ug, qv, ub_lo) - z_lo


def _q_at(ug, qv, u):
    """The shape's value at ``u``, reading it as a knot when it is one.

    Both bounds are pinned as exact knots by :func:`fit_residual_shape`, so the right
    operation is a lookup and not an interpolation. The snap is load-bearing rather than
    defensive: ``1.0 - 0.975`` is ``0.025000000000000022``, not the ``0.025`` that was
    pinned, so interpolating the mirror bound lands between two knots and leaves ~1e-15 in
    the offset. That is small, but it is the difference between the default reproducing the
    two-piece normal exactly and merely nearly.
    """
    i = int(np.abs(ug - u).argmin())
    return float(qv[i]) if abs(ug[i] - u) < 1e-9 else float(np.interp(u, ug, qv))


def apply_shape(z, shape):
    """Remap a normal score through the residual's standardized quantile function.

    Strictly monotone in ``z``, so the ordering of members at a pixel is unchanged — the
    copula's rank structure, and therefore every spatial property of the ensemble, is
    untouched. Only the values attached to those ranks move.

    Beyond ``u_bound`` the fitted quantile function is abandoned for a **unit-slope
    continuation** from the last fitted value.

    The empirical shape is only trustworthy where enough residual sits behind it. Past that
    the standardized residual is dominated by pixels whose *width* is near-degenerate — the
    ratio explodes because the denominator collapsed, not because the error was large — and
    taking a pooled fit at face value maps z = 3 to 5.4 half-widths, putting rare members
    five to eight times outside the published interval. Measured cost of getting this wrong:
    T8.2 crossed its gate, T7.3 went 1.28 -> 1.66, T4.2 0.997 -> 0.79.

    ``u_bound`` is a knob rather than a constant because where that trust runs out depends
    on the band. Fitted per distance band, ``q(0.999)`` runs from 5.99 in the 0-1 px band to
    26.4 beyond 30 px: the far field's tail is real signal, not a collapsed denominator, and
    truncating it at 0.975 is what left the ensemble unable to produce the rare distant
    change the observation contains (predicted P(delta>0.05) 0.00012 against 0.0024
    observed, *worse* than the two-piece normal it replaced).

    At the default ``u_bound = 0.975`` this is exactly the previous behaviour: the shape
    pins ``q(0.975) = Z975`` and ``q(0.025) = -Z975``, so the continuation reduces to
    handing the normal score straight back, which is the two-piece normal's own tail.
    """
    z = np.asarray(z, dtype=np.float64)
    ug = np.asarray(shape["u"], dtype=np.float64)
    qv = np.asarray(shape["q"], dtype=np.float64)
    out = np.interp(norm.cdf(z), ug, qv)

    # Continuous at both joins by construction, and unit slope keeps it monotone.
    z_hi, z_lo, off_hi, off_lo = shape_bounds(shape)
    out = np.where(z > z_hi, z + off_hi, out)
    return np.where(z < z_lo, z + off_lo, out)


def invert_shape(s, shape):
    """Inverse of :func:`apply_shape` — recover the normal score from a shaped one.

    The T3 diagnostics work on the member's *normal score*, recovered by undoing the
    marginal. Undoing only the two-piece normal's scaling leaves ``S(z)`` rather than ``z``,
    and since ``S`` is a strongly nonlinear monotone map that measures the variogram and
    spectrum of a distorted field. Monotone, so the inverse is well defined.
    """
    if shape is None:
        return np.asarray(s, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    ug = np.asarray(shape["u"], dtype=np.float64)
    qv = np.asarray(shape["q"], dtype=np.float64)
    u = np.interp(s, qv, ug)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    # Outside the bound apply_shape is a unit-slope shift, so its inverse is the shift back.
    z_hi, z_lo, off_hi, off_lo = shape_bounds(shape)
    z = np.where(s > z_hi + off_hi, s - off_hi, z)
    return np.where(s < z_lo + off_lo, s - off_lo, z)


def shape_slope(shape, z, h: float = 1e-3):
    """d(shape)/dz at ``z`` — the factor a Monte-Carlo tolerance is missing.

    The standard error of a sample p-quantile is ``sqrt(p(1-p)/M) / f(x_p)``. Written in
    normal-score units that is ``sqrt(p(1-p)/M) / phi(z_p)``, and converting to value units
    costs a factor ``dx/dz``. For the two-piece normal ``x = loc + sigma*z`` so that factor
    is just ``sigma`` — which is what the T5 gates assume. For any other marginal it is
    ``sigma * S'(z_p)``, and omitting ``S'`` makes the tolerance wrong by exactly that
    amount: too tight where the marginal is steep (near the bounds of a spiky shape), too
    loose where it is flat.

    Returns 1.0 for ``shape=None``, so the two-piece normal is unaffected.
    """
    if shape is None:
        return np.ones_like(np.asarray(z, dtype=np.float64))
    z = np.asarray(z, dtype=np.float64)
    return (apply_shape(z + h, shape) - apply_shape(z - h, shape)) / (2.0 * h)


def copula_sample_member(z_field, marginal_params, clip=(0.0, 1.0)):
    """One ensemble member: push a correlated normal field through the pixel marginals.

    ``z = 0`` maps to ``u = 0.5`` maps to exactly the central forecast, so the ensemble
    median reproduces the frozen checkpoint's forecast (T5.1).
    """
    return marginal_from_z(np.asarray(z_field), marginal_params, clip=clip)


def marginal_from_z_torch(z, loc, sl, sr, clip=(0.0, 1.0), shape=None, band=None):
    """Torch version, for keeping the whole member on GPU."""
    import torch

    zz = z if shape is None else apply_shape_torch(z, shape, device=z.device,
                                                   dtype=z.dtype, band=band)
    scale = torch.where(zz < 0, sl, sr)
    out = loc + scale * zz
    if clip is not None:
        out = torch.clamp(out, clip[0], clip[1])
    return out


def apply_shape_torch(z, shape, device=None, dtype=None, band=None):
    """GPU counterpart of :func:`apply_shape`; same mapping, same guarantees.

    ``shape`` is either one shape dict, or a stack from :func:`stack_shapes` together with
    a per-element ``band`` index. The stacked form shares one u-grid across bands, so the
    bucket search is done once and only the gathered ``q`` rows differ.
    """
    import torch

    stacked = "q_stack" in shape
    ug = torch.as_tensor(shape["u"], device=device, dtype=dtype)
    n = ug.numel()

    # Normal CDF without scipy, so this stays on the GPU.
    u = 0.5 * (1.0 + torch.erf(z / float(np.sqrt(2.0))))

    # The grid is *not* uniform — the gate quantiles and the tail bound are inserted as
    # exact knots — so the bucket has to be searched rather than computed. Getting this
    # wrong silently mis-maps every member on the GPU path while the numpy path stays
    # correct.
    i0 = torch.clamp(torch.searchsorted(ug, u.contiguous(), right=True) - 1, 0, n - 2)
    du = ug[i0 + 1] - ug[i0]
    frac = torch.clamp((u - ug[i0]) / torch.clamp(du, min=1e-30), 0.0, 1.0)

    if not stacked:
        qv = torch.as_tensor(shape["q"], device=device, dtype=dtype)
        q0, q1 = qv[i0], qv[i0 + 1]
        # Precomputed when the caller has already moved the grids to the device, since
        # shape_bounds works in numpy and cannot read a GPU tensor.
        z_hi, z_lo, off_hi, off_lo = (
            (shape["z_hi"], shape["z_lo"], shape["off_hi"], shape["off_lo"])
            if "z_hi" in shape else shape_bounds(shape))
    else:
        b = band.to(torch.long)
        qs = torch.as_tensor(shape["q_stack"], device=device, dtype=dtype)
        q0 = qs[b, i0]
        q1 = qs[b, i0 + 1]
        pick = lambda k: torch.as_tensor(shape[k], device=device, dtype=dtype)[b]
        z_hi, z_lo = pick("z_hi"), pick("z_lo")
        off_hi, off_lo = pick("off_hi"), pick("off_lo")

    out = q0 + frac * (q1 - q0)

    # Past the bound, continue with unit slope from the fitted value — see apply_shape.
    out = torch.where(z > z_hi, z + off_hi, out)
    return torch.where(z < z_lo, z + off_lo, out)


def read_shape_artifact(path, n_bands):
    """Read a marginal-shape JSON into ``({(horizon, band): shape}, banded)``.

    The one reader for all three artifact layouts, because three scripts consume it and
    this project has already paid once for the same definition living in eight places.

      * ``{"by_horizon": {h: {u, q, n}}}`` — the original, one shape per horizon;
      * ``{"by_horizon": {h: {"pooled": ..., "by_band": {b: ...}}}}`` — per band;
      * a per-band artifact with some bands missing, which inherit that horizon's pooled
        shape. Inheriting pooled rather than the identity is deliberate: "too sparse to fit"
        must not mean "this band alone reverts to the two-piece normal".
    """
    blob = json.load(open(path)).get("by_horizon", {})
    out, banded = {}, False
    for k, v in blob.items():
        if not v:
            continue
        h = int(k)
        if "u" in v:
            for b in range(n_bands):
                out[(h, b)] = v
            continue
        pooled = v.get("pooled")
        by_band = v.get("by_band", {})
        banded = banded or bool(by_band)
        for b in range(n_bands):
            s = by_band.get(str(b), pooled)
            if s is not None:
                out[(h, b)] = s
    return out, banded


def stack_shapes(shapes):
    """Put several shapes on one shared u-grid so the GPU can gather rows from a matrix.

    ``shapes`` is a sequence indexed by band. Entries may be ``None`` (no fit for that
    band); they are filled with the identity, which is the two-piece normal. The union of
    every input grid is used, and since each shape is piecewise linear in ``u`` and every
    grid it needs is already one of its own knots, evaluating on the union is exact — the
    gate quantiles stay exact knots rather than becoming interpolated.
    """
    grids = [np.asarray(s["u"], dtype=np.float64) for s in shapes if s is not None]
    if not grids:
        return None
    ug = _dedupe_knots(np.concatenate(grids))
    ident = norm.ppf(np.clip(ug, 1e-12, 1 - 1e-12))
    q_stack, z_hi, z_lo, off_hi, off_lo = [], [], [], [], []
    for s in shapes:
        if s is None:
            # Zero bounds put every z on the continuation, which at zero offset is the
            # identity exactly — rather than the interpolated normal ppf that q_stack would
            # give. "No fit for this band" has to mean "two-piece normal", bit for bit.
            q_stack.append(ident)
            z_hi.append(0.0)
            z_lo.append(0.0)
            off_hi.append(0.0)
            off_lo.append(0.0)
            continue
        q_stack.append(np.interp(ug, np.asarray(s["u"], dtype=np.float64),
                                 np.asarray(s["q"], dtype=np.float64)))
        zh, zl, oh, ol = shape_bounds(s)
        z_hi.append(zh)
        z_lo.append(zl)
        off_hi.append(oh)
        off_lo.append(ol)
    return {"u": ug, "q_stack": np.asarray(q_stack),
            "z_hi": np.asarray(z_hi), "z_lo": np.asarray(z_lo),
            "off_hi": np.asarray(off_hi), "off_lo": np.asarray(off_lo)}


# --------------------------------------------------------------------------------------
# int16 quantization
# --------------------------------------------------------------------------------------
def quantize(values, scale: float = DEFAULT_SCALE, offset: float = 0.0, valid=None):
    """Quantize HM values to int16 with an explicit nodata sentinel."""
    v = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(v)
    q = np.rint((np.where(finite, v, 0.0) - offset) / scale)
    q = np.clip(q, INT16_SENTINEL + 1, 32767).astype(np.int16)
    keep = finite if valid is None else (valid & finite)
    return np.where(keep, q, np.int16(INT16_SENTINEL))


def dequantize(q, scale: float = DEFAULT_SCALE, offset: float = 0.0):
    q = np.asarray(q)
    out = q.astype(np.float32) * scale + offset
    return np.where(q == INT16_SENTINEL, np.nan, out)


def quantization_error_bound(scale: float = DEFAULT_SCALE):
    """Max round-trip error of :func:`quantize` (used by the T5.1 gate)."""
    return scale / 2.0
