"""The two gates this phase is judged on: the picket fence, and PIT structure.

Both are computed from an exported quantile-function raster -- ``[n_levels, n_px]``, monotone
along axis 0 -- so they apply to every head family without knowing which one produced it.

**Why every fence statistic is reported twice.** A needle is a pair of adjacent quantile
levels whose *values* have collapsed together, and "adjacent" is a property of the grid the
raster was written on. So a metric read off that grid moves when the grid moves -- which
makes E0 (re-spacing the published levels, no retraining) look like it fixed the model when
all it did was stop over-sampling a flat stretch. Both readings are wanted, and they answer
different questions:

``*_export``
    on the raster's own grid: what a consumer of the published product actually sees. E0
    moves this **by construction**, and that is the point of E0.

``*_ref``
    on :data:`REF_U`, a fixed dense grid this module owns and never changes. The raster is a
    piecewise-linear quantile function, so it can be evaluated anywhere; subdividing a
    segment leaves its implied density unchanged, which is what makes this reading a property
    of the *forecast distribution* rather than of the export. E2, E4 and E5 must move this to
    have done anything. E0 should barely move it -- and if it does, the export grid was
    destroying information, which is itself worth knowing.

Reporting one alone is how a rendering fix gets recorded as a model fix.
"""

from __future__ import annotations

import numpy as np

# int16 x this is the published storage quantum, so a gap below it is not representable at all.
INT16_SCALE = 3.0518509e-05

# A gap at or below three quantisation steps (~9.2e-5) is a needle. The measured incumbent has
# 4.5% of gaps here and only 2.0% within a single step, so the fence is mostly wider than the
# export resolution -- int16 storage is not the cause and raising this threshold would not
# make the finding go away.
NEEDLE_EPS = 3 * INT16_SCALE

# The largest density the observable can justify. The core of the HM-change distribution has a
# robust sigma (1.4826 x MAD) of 0.00069, and a kernel that narrow supports a density of at
# most ~578. The incumbent reaches 677. A sharp mode is correct here; exceeding what the noise
# can resolve is not.
F_MAX_DENSITY = 578.0


def _ref_grid(n=256, trunc=1e-4):
    from scipy.stats import norm
    return norm.cdf(np.linspace(norm.ppf(trunc), norm.ppf(1.0 - trunc), n))


#: Frozen. Defined here, spelled out nowhere else, and **never changed** -- every ``*_ref``
#: number across every experiment is comparable only for as long as this is the same grid.
REF_U = _ref_grid()


def interp_to(u_src, q, u_dst):
    """Evaluate a piecewise-linear quantile function at new levels.

    ``q`` is ``[n_src, n_px]``; returns ``[n_dst, n_px]``. Exact where ``u_dst`` subdivides a
    source segment, which is the case that matters: subdividing leaves ``dp/dq`` unchanged, so
    the reference-grid density is the source-grid density regardless of how the source spent
    its levels.
    """
    u_src = np.asarray(u_src, dtype=np.float64)
    u_dst = np.asarray(u_dst, dtype=np.float64)
    j = np.clip(np.searchsorted(u_src, u_dst), 1, u_src.size - 1)
    t = ((u_dst - u_src[j - 1]) / (u_src[j] - u_src[j - 1]))[:, None]
    return q[j - 1] + t * (q[j] - q[j - 1])


def segment_stats(u, q):
    """``(dp, dq)`` per segment: ``dp`` is ``[n_seg]``, ``dq`` is ``[n_seg, n_px]``."""
    return np.diff(np.asarray(u, dtype=np.float64)), np.diff(q, axis=0)


def needle_mass(u, q, eps: float = NEEDLE_EPS):
    """Per-pixel probability mass sitting in collapsed segments. ``[n_px]``.

    The incumbent reads 4.5% at the median pixel and 21% at the 90th percentile. The tell that
    this is a monotonicity artefact rather than a real spike is that 4.5% of gaps are within
    three quantisation steps while only 0.7% are exactly zero: levels collapsing *toward* each
    other without collapsing *onto* each other is what a strictly-positive-derivative
    constraint produces when the target wants a flat segment.
    """
    dp, dq = segment_stats(u, q)
    return np.where(np.isfinite(dq) & (dq <= eps), dp[:, None], 0.0).sum(axis=0)


def implied_density(u, q):
    """``dp/dq`` per segment, ``[n_seg, n_px]``. The density the forecast actually claims."""
    dp, dq = segment_stats(u, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dp[:, None] / dq


def fence_stats(u, q, label: str, eps: float = NEEDLE_EPS, f_max: float = F_MAX_DENSITY):
    """The fence gate, as a dict of ``{stat}_{label}`` keys.

    ``needle_mass_median`` / ``_p90`` target ~0; ``max_density_p99`` targets <= ``f_max``.
    ``gap_frac_zero`` against ``gap_frac_needle`` is the diagnostic that distinguishes a
    monotonicity artefact from a genuine atom, so both are reported rather than just the one.
    """
    nm = needle_mass(u, q, eps)
    dens = implied_density(u, q)
    dp, dq = segment_stats(u, q)
    finite = np.isfinite(dq)
    n = max(int(finite.sum()), 1)
    px_max = np.nanmax(np.where(finite, dens, -np.inf), axis=0)
    px_max = px_max[np.isfinite(px_max)]
    return {
        f"needle_mass_median_{label}": float(np.median(nm)),
        f"needle_mass_p90_{label}": float(np.percentile(nm, 90)),
        f"needle_mass_mean_{label}": float(nm.mean()),
        f"gap_frac_needle_{label}": float((finite & (dq <= eps)).sum() / n),
        f"gap_frac_zero_{label}": float((finite & (dq <= 0)).sum() / n),
        f"max_density_p50_{label}": float(np.median(px_max)) if px_max.size else np.nan,
        f"max_density_p99_{label}": float(np.percentile(px_max, 99)) if px_max.size else np.nan,
        f"over_f_max_frac_{label}": (float((px_max > f_max).mean()) if px_max.size else np.nan),
    }


def fence_gate(u, q, eps: float = NEEDLE_EPS, f_max: float = F_MAX_DENSITY):
    """Both readings at once -- see the module docstring for why one alone is not enough."""
    out = fence_stats(u, q, "export", eps, f_max)
    out.update(fence_stats(REF_U, interp_to(u, q, REF_U), "ref", eps, f_max))
    return out


# --------------------------------------------------------------------------- PIT structure

def pit_structure(pit_values, bins=(20, 40, 60)):
    """Is the PIT histogram's roughness real, or is it what noise looks like at these bins?

    Under a calibrated forecast the count in each of ``B`` bins is multinomial, so the RMS
    deviation from uniform -- expressed in units of its own standard error -- has expectation
    ~1 **at every** ``B``. Raw RMS instead grows as ``sqrt(B)``, which is why comparing raw
    numbers across bin counts says nothing.

    The incumbent's measured growth was 0.577 -> 0.735 -> 0.750 at 20/40/60 bins against a
    noise prediction of x1.41 and x1.22 (observed x1.27, x1.02). Slower-than-noise growth with
    peak locations stable across resolutions is the signature of real structure -- a fixed
    number of genuine features being resolved -- rather than of sampling error.

    Two statistics, and the first is the primary one. ``pit_rms_se_B`` detects *any*
    departure from uniform and is ~1 under the null at every ``B``; measured on synthetic
    spikes it reads 30-140, so nothing real hides from it. ``pit_growth_vs_noise`` is a
    secondary discriminator for whether the structure has resolvable *width*: measured, it
    runs 0.97 for delta-narrow features (sigma 0.004), 0.80 at sigma 0.01, 0.67 at 0.02 and
    0.61 at 0.04. **A feature narrower than the finest bin therefore grows like noise in this
    statistic even though it is real** -- so read growth only after ``rms_se`` has established
    that there is structure at all, never as the test for whether there is any.

    Returns per-bin-count RMS in standard-error units (``rms_se_B``, target ~1), the raw RMS
    (``rms_B``, comparable only within one ``B``), and the location of the largest peak.
    """
    p = np.asarray(pit_values, dtype=np.float64)
    p = p[np.isfinite(p)]
    n = p.size
    out = {"pit_n": int(n)}
    if n < 100:
        return out
    for b in bins:
        counts, edges = np.histogram(p, bins=b, range=(0.0, 1.0))
        expected = n / b
        dev = (counts - expected) / expected                     # relative deviation
        # SE of one bin's relative deviation under the uniform null.
        se = np.sqrt((1.0 - 1.0 / b) / expected)
        out[f"pit_rms_{b}"] = float(np.sqrt((dev ** 2).mean()))
        out[f"pit_rms_se_{b}"] = float(np.sqrt((dev ** 2).mean()) / se)
        k = int(np.argmax(np.abs(dev)))
        out[f"pit_peak_u_{b}"] = float(0.5 * (edges[k] + edges[k + 1]))
    # Growth against the sqrt(B) noise prediction. ~1 means "indistinguishable from noise";
    # well below 1 means a fixed set of real features, resolved rather than created.
    a, c = bins[0], bins[-1]
    pred = np.sqrt(c / a)
    obs = out[f"pit_rms_{c}"] / max(out[f"pit_rms_{a}"], 1e-12)
    out["pit_growth_vs_noise"] = float(obs / pred)
    return out


def zero_leak(u, q, y, hm_t0, lo: float = -0.05, hi: float = 0.005):
    """Predicted-over-observed mass in the bin straddling zero change.

    The incumbent puts 1.70x too much probability on ``[-0.05, 0)`` and correspondingly too
    little on ``[0, 0.005)``: about 15% of cells displaced one bin left, *across zero*. HM does
    not meaningfully decrease, so that mass is in a region where change does not occur -- the
    fence straddles the persistence anchor and half of it lands on the wrong side.

    Target 1.0. Reported for both sub-bins because a ratio alone cannot say which way it moved.
    """
    def predicted(a, b):
        from_ = _pit_at(u, q, hm_t0 + a)
        to_ = _pit_at(u, q, hm_t0 + b)
        return float(np.nanmean(to_ - from_))

    def observed(a, b):
        d = y - hm_t0
        return float(np.nanmean((d >= a) & (d < b)))

    out = {}
    for name, (a, b) in (("neg", (lo, 0.0)), ("pos", (0.0, hi))):
        p, o = predicted(a, b), observed(a, b)
        out[f"zero_leak_{name}_pred"] = p
        out[f"zero_leak_{name}_obs"] = o
        out[f"zero_leak_{name}_ratio"] = float(p / o) if o > 0 else np.nan
    return out


def _pit_at(u, q, y):
    """``Q^-1(y)`` by linear interpolation -- the same convention the scorer's ``pit`` uses."""
    n = q.shape[0]
    idx = np.clip((q < y[None, :]).sum(axis=0), 1, n - 1)
    ar = np.arange(y.size)
    q0, q1 = q[idx - 1, ar], q[idx, ar]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.clip(np.where(q1 > q0, (y - q0) / (q1 - q0), 0.0), 0.0, 1.0)
    out = u[idx - 1] + w * (u[idx] - u[idx - 1])
    out = np.where(y <= q[0], 0.0, out)
    return np.where(y >= q[-1], 1.0, out)
