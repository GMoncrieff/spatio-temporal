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

# The int16 storage quantum of the OLD export convention, kept because NEEDLE_EPS was derived
# from it and every measured number in docs/conv_spline_phase.md is on that scale. It is no
# longer the export resolution: --predict_qf_dtype float32 removes the floor entirely.
INT16_SCALE = 3.0518509e-05

# A gap at or below ~9.2e-5 in HM is a needle. **This is a physical width, not a storage one**,
# and the distinction became load-bearing on 2026-09-14. It was introduced as "three int16
# quantisation steps" on the argument that e1's fence was mostly WIDER than the export could
# resolve -- 4.5% of gaps within it against 2.0% within a single step -- so storage was not the
# cause. b1 broke that argument rather than inheriting it: 47% of its gaps fall within this
# threshold and 34% are EXACTLY ZERO, i.e. adjacent levels exported to the same int16 code, so
# on b1 the int16 export WAS a cause. The threshold keeps its value (9.155e-05, about 0.13 of
# the core's robust sigma of 0.00069) so every number stays comparable across the change; what
# it no longer means is "the smallest gap the raster can hold".
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


#: Pixels per block in :func:`fence_per_pixel`. The intermediates are ``[n_seg, block]`` and
#: the reference reading has 255 segments, so a whole Africa fold at once builds a 13.6 GB
#: float64 density array -- measured, and the largest single term in the scorer's 59.1 GB
#: peak. Every quantity here is pixel-independent, so blocking is exact.
GATE_PX_BLOCK = 1_000_000


def _fence_block(u, q, eps, clamp):
    """One block's worth of :func:`fence_per_pixel`."""
    dp, dq = segment_stats(u, q)
    finite = np.isfinite(dq)
    needle = finite & (dq <= eps)
    zero = finite & (dq <= 0)
    # A segment lying flat ON the support boundary is HM hitting 0 or 1, not a fence. HM
    # cannot be negative, so probability piling up at the floor is the physically correct
    # thing for the model to do -- and measured on b1_s42, EVERY pixel whose two lowest
    # levels were identical had Q = 0.0 exactly there. Counting that as a collapsed segment
    # put a large constant into the degeneracy statistic on every arm, which is a column that
    # cannot then discriminate between them (rule 8's shape, one level down).
    lo, hi = clamp
    at_clamp = zero & ((q[1:] <= lo) | (q[:-1] >= hi))
    real_zero = zero & ~at_clamp
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = dp[:, None] / dq
    # The boundary atom is excluded from the density too: an infinite density AT the clamp is
    # a statement about the support, and reporting it as the pixel's max density would hide
    # whatever the interior is doing.
    usable = finite & ~at_clamp
    return {
        "needle_mass": np.where(needle & ~at_clamp, dp[:, None], 0.0).sum(axis=0),
        "px_max_density": np.nanmax(np.where(usable, dens, -np.inf), axis=0),
        "n_finite": finite.sum(axis=0).astype(np.int32),
        "n_needle": (needle & ~at_clamp).sum(axis=0).astype(np.int32),
        "n_zero": real_zero.sum(axis=0).astype(np.int32),
        "n_clamp": at_clamp.sum(axis=0).astype(np.int32),
    }


def fence_per_pixel(u, q, eps: float = NEEDLE_EPS, u_interp=None,
                    block: int = GATE_PX_BLOCK, clamp=(0.0, 1.0)):
    """The fence gate's per-pixel ingredients, before any reduction. All ``[n_px]``.

    ``fence_stats`` used to read the whole ``[n_levels, n_px]`` quantile function for every
    stratum it was asked about. The scorer cannot do that: it streams the raster in
    horizontal bands and frees each one, while a stratum is a mask over the whole region --
    so the gate has to be built from quantities that *survive the band*. Splitting it here
    rather than in the scorer keeps one definition of the statistic (``fence_stats`` is now
    this plus :func:`fence_reduce`), which is the only way the two cannot disagree.

    The split is exact, not an approximation. ``needle_mass`` and the per-pixel maximum
    density are already per-pixel; the two gap fractions are ratios of counts over
    ``(segment, pixel)`` pairs, and a sum of per-pixel counts is that same total.

    ``u_interp`` evaluates the quantile function on another grid first -- the ``_ref``
    reading -- **inside the block loop**, so ``[len(u_interp), n_px]`` is never materialised
    for the whole raster. That array is why the scorer peaked at 59.1 GB on a single Africa
    fold; bounding it here rather than in the caller means the reference reading costs the
    same whether it is asked for one stratum or the whole region.
    """
    n_px = q.shape[1]
    if n_px == 0:
        z64, z32 = np.zeros(0), np.zeros(0, dtype=np.int32)
        return {"needle_mass": z64, "px_max_density": z64, "n_finite": z32,
                "n_needle": z32, "n_zero": z32, "n_clamp": z32}
    step = int(block) if block and block > 0 else n_px
    outs = []
    for a in range(0, n_px, step):
        qb = q[:, a:a + step]
        ub = u
        if u_interp is not None:
            qb, ub = interp_to(u, qb, u_interp), u_interp
        outs.append(_fence_block(ub, qb, eps, clamp))
    if len(outs) == 1:
        return outs[0]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def fence_reduce(px, label: str, f_max: float = F_MAX_DENSITY):
    """Reduce :func:`fence_per_pixel` over a set of pixels -- see :func:`fence_stats`.

    **A pixel with a zero-width segment has infinite implied density, and that is the
    strongest possible evidence of a needle rather than a reason to ignore it.** This
    function used to drop them (``px_max[np.isfinite(px_max)]``) before taking the median,
    the 99th percentile and the over-``f_max`` fraction -- so the density gate was computed
    over the pixels that had fenced *least*, and the worst ones left no trace. On b1_s42,
    where 34% of adjacent levels exported to the same code, that is most of the mass of the
    problem. Rule: a gate written for one failure mode keeps passing after the mode inverts.

    So ``over_f_max_frac`` now counts every pixel, degenerate ones included, and
    ``px_degenerate_frac`` reports them in their own right. The two percentiles stay on the
    finite subset, because a percentile of ``inf`` is ``inf`` and says nothing about how
    sharp the rest are -- but they are no longer readable without the degenerate fraction
    beside them, which is the point.

    **A segment flat on the support boundary is not counted as either.** HM cannot leave
    [0, 1], so probability piling up at the floor is correct behaviour, and on b1_s42 it was
    the whole of the apparent degeneracy: every pixel whose two lowest levels were identical
    had ``Q = 0.0`` exactly, and the clamp contributed 0.3% of the needle mass against the
    core's 73.3%. It gets ``gap_frac_clamp`` and ``px_clamp_frac`` of its own so a column
    that is a large constant on every arm stops sitting in the middle of the fence gate.
    """
    nm = px["needle_mass"]
    n = max(int(px["n_finite"].sum()), 1)
    all_max = px["px_max_density"]
    ok = np.isfinite(all_max)
    finite_max = all_max[ok]
    n_px = max(all_max.size, 1)
    return {
        f"needle_mass_median_{label}": float(np.median(nm)),
        f"needle_mass_p90_{label}": float(np.percentile(nm, 90)),
        f"needle_mass_mean_{label}": float(nm.mean()),
        f"gap_frac_needle_{label}": float(int(px["n_needle"].sum()) / n),
        f"gap_frac_zero_{label}": float(int(px["n_zero"].sum()) / n),
        # The support boundary, reported in its own right rather than mixed into the fence.
        f"gap_frac_clamp_{label}": float(int(px["n_clamp"].sum()) / n),
        f"px_clamp_frac_{label}": float((px["n_clamp"] > 0).mean()) if n_px else np.nan,
        f"max_density_p50_{label}": float(np.median(finite_max)) if finite_max.size else np.nan,
        f"max_density_p99_{label}": (float(np.percentile(finite_max, 99))
                                     if finite_max.size else np.nan),
        # Over all pixels: a genuinely degenerate one is over any ceiling. A pixel that is
        # degenerate ONLY at the clamp is not counted here -- its interior is measurable and
        # ``px_max_density`` holds it.
        f"over_f_max_frac_{label}": float(((all_max > f_max) | ~ok).sum() / n_px),
        f"px_degenerate_frac_{label}": float((~ok).sum() / n_px),
    }


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
    return fence_reduce(fence_per_pixel(u, q, eps), label, f_max)


def fence_gate(u, q, eps: float = NEEDLE_EPS, f_max: float = F_MAX_DENSITY):
    """Both readings at once -- see the module docstring for why one alone is not enough."""
    out = fence_reduce(fence_per_pixel(u, q, eps), "export", f_max)
    out.update(fence_reduce(fence_per_pixel(u, q, eps, u_interp=REF_U), "ref", f_max))
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
    return zero_leak_reduce(zero_leak_per_pixel(u, q, y, hm_t0, lo, hi))


def zero_leak_per_pixel(u, q, y, hm_t0, lo: float = -0.05, hi: float = 0.005):
    """The leak's per-pixel ingredients, so a banded scorer can reduce it over a stratum.

    Same reason as :func:`fence_per_pixel`: the quantile function does not survive the band
    it was read in, and both the predicted mass and the observed indicator are per-pixel
    quantities whose stratum value is a mean. One definition, reduced in two places.
    """
    p_lo = _pit_at(u, q, hm_t0 + lo)
    p_mid = _pit_at(u, q, hm_t0)
    p_hi = _pit_at(u, q, hm_t0 + hi)
    d = y - hm_t0
    return {
        "zero_leak_neg_pred_px": p_mid - p_lo,
        "zero_leak_pos_pred_px": p_hi - p_mid,
        "zero_leak_neg_obs_px": ((d >= lo) & (d < 0.0)).astype(np.float64),
        "zero_leak_pos_obs_px": ((d >= 0.0) & (d < hi)).astype(np.float64),
    }


def zero_leak_reduce(px):
    """Reduce :func:`zero_leak_per_pixel` over a set of pixels. Target 1.0."""
    out = {}
    for name in ("neg", "pos"):
        p = float(np.nanmean(px[f"zero_leak_{name}_pred_px"]))
        o = float(np.nanmean(px[f"zero_leak_{name}_obs_px"]))
        out[f"zero_leak_{name}_pred"] = p
        out[f"zero_leak_{name}_obs"] = o
        out[f"zero_leak_{name}_ratio"] = float(p / o) if o > 0 else np.nan
    return out


# --------------------------------------------------------------- both gates, streamed

#: Prefixes ``gate_per_pixel`` writes, so a caller can pick its keys out of a larger dict
#: without spelling any of them a second time.
GATE_PX_PREFIXES = ("fence_export_", "fence_ref_", "zero_leak_")


def gate_per_pixel(u, q, y, hm_t0, eps: float = NEEDLE_EPS,
                   block: int = GATE_PX_BLOCK):
    """Every per-pixel quantity both gates reduce over. ``q`` is ``[n_levels, n_px]``.

    Computed where the quantile function is alive -- inside the scorer's band loop -- and
    reduced later over whatever stratum mask is asked for. The PIT structure statistic is not
    here because the scorer already carries the PIT per pixel.
    """
    out = {f"fence_export_{k}": v
           for k, v in fence_per_pixel(u, q, eps, block=block).items()}
    ref = fence_per_pixel(u, q, eps, u_interp=REF_U, block=block)
    out.update({f"fence_ref_{k}": v for k, v in ref.items()})
    out.update(zero_leak_per_pixel(u, q, y, hm_t0))
    return out


def gate_reduce(px, pit_values, f_max: float = F_MAX_DENSITY):
    """Both gates over one stratum, from :func:`gate_per_pixel` sliced to that stratum.

    The counterpart of ``fence_gate`` + ``pit_structure`` + ``zero_leak``, and identical to
    them by construction: each is now a reduction of the same per-pixel quantities.
    """
    out = {}
    for label in ("export", "ref"):
        pre = f"fence_{label}_"
        out.update(fence_reduce({k[len(pre):]: v for k, v in px.items()
                                 if k.startswith(pre)}, label, f_max))
    out.update(pit_structure(pit_values))
    out.update(zero_leak_reduce(px))
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
