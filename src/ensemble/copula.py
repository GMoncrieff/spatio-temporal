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
def fit_residual_shape(standardized_residual, n_knots: int = 512, min_count: int = 10_000):
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
    n_knots = int(n_knots) | 1
    u = np.linspace(EDGE_U, 1.0 - EDGE_U, n_knots)
    q = np.quantile(ec, u)
    vals = np.where(q < 0, q / lo_ref * Z975, q / hi_ref * Z975)
    vals[n_knots // 2] = 0.0
    vals = np.maximum.accumulate(vals)  # quantile functions are monotone; keep it exact
    return {"u": u.tolist(), "q": vals.tolist(), "n": int(e.size)}


def apply_shape(z, shape):
    """Remap a normal score through the residual's standardized quantile function.

    Strictly monotone in ``z``, so the ordering of members at a pixel is unchanged — the
    copula's rank structure, and therefore every spatial property of the ensemble, is
    untouched. Only the values attached to those ranks move.
    """
    z = np.asarray(z, dtype=np.float64)
    ug = np.asarray(shape["u"], dtype=np.float64)
    qv = np.asarray(shape["q"], dtype=np.float64)
    out = np.interp(norm.cdf(z), ug, qv)

    # Outside the published 95% bound, revert to the two-piece normal's unit slope.
    #
    # The empirical shape is only trustworthy in the body. Beyond the bound the
    # standardized residual is dominated by pixels whose *width* is near-degenerate — the
    # ratio explodes because the denominator collapsed, not because the error was large —
    # and taking that at face value maps z = 3 to 5.4 half-widths, putting rare members
    # five to eight times outside the published interval. Measured cost of getting this
    # wrong: T8.2 crossed its gate, T7.3 went 1.28 -> 1.66, T4.2 0.997 -> 0.79.
    #
    # Replacing the body and keeping the tail is also the honest division of labour: the
    # bounds are what the quantile heads and the recalibration are calibrated to state, and
    # the shape's job is only to say how mass is distributed between them.
    # The shape maps +/-Z975 to exactly +/-Z975, so handing the tail straight back is
    # continuous at the join as well as being the two-piece normal's own behaviour.
    return np.where(np.abs(z) > Z975, z, out)


def copula_sample_member(z_field, marginal_params, clip=(0.0, 1.0)):
    """One ensemble member: push a correlated normal field through the pixel marginals.

    ``z = 0`` maps to ``u = 0.5`` maps to exactly the central forecast, so the ensemble
    median reproduces the frozen checkpoint's forecast (T5.1).
    """
    return marginal_from_z(np.asarray(z_field), marginal_params, clip=clip)


def marginal_from_z_torch(z, loc, sl, sr, clip=(0.0, 1.0), shape=None):
    """Torch version, for keeping the whole member on GPU."""
    import torch

    zz = z if shape is None else apply_shape_torch(z, shape, device=z.device, dtype=z.dtype)
    scale = torch.where(zz < 0, sl, sr)
    out = loc + scale * zz
    if clip is not None:
        out = torch.clamp(out, clip[0], clip[1])
    return out


def apply_shape_torch(z, shape, device=None, dtype=None):
    """GPU counterpart of :func:`apply_shape`; same mapping, same guarantees."""
    import torch

    ug = torch.as_tensor(shape["u"], device=device, dtype=dtype)
    qv = torch.as_tensor(shape["q"], device=device, dtype=dtype)
    n = ug.numel()
    u0, u1 = float(ug[0]), float(ug[-1])

    # Normal CDF without scipy, so this stays on the GPU.
    u = 0.5 * (1.0 + torch.erf(z / float(np.sqrt(2.0))))

    # The grid is a uniform ramp on [u0, u1], so the bucket index is exact arithmetic.
    pos = torch.clamp((u - u0) / (u1 - u0), 0.0, 1.0) * (n - 1)
    i0 = torch.clamp(pos.floor().long(), 0, n - 2)
    frac = pos - i0.to(pos.dtype)
    out = qv[i0] + frac * (qv[i0 + 1] - qv[i0])

    # Outside the published bound, hand the normal score straight back — see apply_shape.
    return torch.where(z.abs() > Z975, z, out)


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
