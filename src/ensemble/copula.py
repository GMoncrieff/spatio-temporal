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
    """Same as :func:`marginal_ppf` but taking the normal score directly (u = Phi(z))."""
    loc = params["loc"]
    sl = params["scale_left"]
    sr = params["scale_right"]
    scale = np.where(z < 0, sl, sr)
    out = loc + scale * z
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


def copula_sample_member(z_field, marginal_params, clip=(0.0, 1.0)):
    """One ensemble member: push a correlated normal field through the pixel marginals.

    ``z = 0`` maps to ``u = 0.5`` maps to exactly the central forecast, so the ensemble
    median reproduces the frozen checkpoint's forecast (T5.1).
    """
    return marginal_from_z(np.asarray(z_field), marginal_params, clip=clip)


def marginal_from_z_torch(z, loc, sl, sr, clip=(0.0, 1.0)):
    """Torch version, for keeping the whole member on GPU."""
    import torch

    scale = torch.where(z < 0, sl, sr)
    out = loc + scale * z
    if clip is not None:
        out = torch.clamp(out, clip[0], clip[1])
    return out


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
