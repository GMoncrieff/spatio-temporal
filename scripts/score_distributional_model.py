#!/usr/bin/env python3
"""Score a distributional model from its stitched rasters, without generating an ensemble.

The fast iteration instrument for the spline head, the counterpart of
``score_model_experiment.py`` (which is left untouched and still worth running: its ``k_up``
table is an independent width check on the emitted triple, and two instruments that share no
code is how a bug gets localised rather than believed).

What it measures, and why each row is here:

``crps_skill``
    ``1 - CRPS / MAE_persistence``. **Scored against persistence, not zero.** The median
    20-year HM change is 1e-4, so a pooled error can look unremarkable while the forecast
    loses to "nothing will change" -- which is exactly what happened to the central field
    once already. A point forecast at HM_t0 has CRPS = |y - HM_t0|, so persistence is the
    denominator with no extra machinery.

PIT and its tails
    ``u* = Q^-1(y)`` is uniform if and only if the whole predictive distribution is right,
    which makes it a strictly stronger test than any coverage number. The four tail rows
    exist because a pooled PIT statistic is structurally blind to a defect living in a
    thousandth of the pixels, and the far-field defect this phase exists to address is
    exactly that shape.

exceedance calibration
    ``P(dHM > 0.05)`` and friends, read straight off the quantile function as ``1 - CDF``.
    This replaces ``predict_change_rates.py`` outright: there is no marginal to assume any
    more, so the closed form is the definition rather than a second approximation to it.

``tail_reach``
    ``(Q(0.999) - Q(0.5)) / (Q(0.975) - Q(0.5))``, per distance band. **Reported, never
    gated on southern Africa**: the >100 px band contains *zero* pixels here, so a far-field
    ratio-to-observed cannot be computed at all and band 4 (30-100 px, observed
    P(dHM>0.05) = 0.00225) is the furthest thing that can be measured.

The Gaussian value of ``tail_reach`` is 1.577. The incumbent's problem is that its remote-band
intervals need a reach near 12 to put +0.05 inside them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.strata import (  # noqa: E402
    DHAT_BINS, DHAT_LABELS, DIST_LABELS, HM_BINS, HM_LABELS, distance_band,
)
from src.strata import OBS_MAG_BINS as _OBS_MAG_BINS  # noqa: E402
from src.strata import OBS_MAG_LABELS  # noqa: E402
from src.qf_diagnostics import fence_gate, pit_structure, zero_leak  # noqa: E402
from diagnose_central_field import (  # noqa: E402
    HORIZONS, MAX_OBSERVED_YEAR, WINDOWS, HM_DIR, _read, _read_like,
)
from score_model_experiment import central_stats  # noqa: E402

REGION_ROOT = Path("data/ensemble/region/southern_africa")
INT16_SCALE = 1.0 / 32767.0
# The three exceedance questions the product is actually asked, in HM units.
THRESHOLDS = ((0.01, "hi"), (0.05, "hi"), (-0.01, "lo"))
COVERAGE_LEVELS = (0.50, 0.80, 0.95, 0.99)
# Imported, not re-declared: this stratum lived in two files with two different conventions
# (signed here, absolute in src/strata.py), which is rule 9 forming again in a new place.
OBS_MAG_BINS = np.array(_OBS_MAG_BINS)


# ----------------------------------------------------------------------------- the raster

def qf_levels(path: str):
    """The ``u`` grid of a quantile-function raster, validated, without reading any pixels."""
    with rasterio.open(path) as src:
        tags = src.tags()
        if "u_levels" not in tags:
            raise SystemExit(f"{path} carries no u_levels tag; not a quantile-function raster")
        u = np.array([float(v) for v in tags["u_levels"].split(",")], dtype=np.float64)
        n_bands, nod = src.count, src.nodata
    if n_bands != u.size:
        raise SystemExit(f"{path}: {n_bands} bands against {u.size} u levels")
    # A zero-width segment makes the piecewise-linear slope 0/0, and CRPS comes back NaN for
    # every pixel -- a metric failure indistinguishable from a model failure at a glance. Fail
    # here instead, where the cause is one line away.
    gaps = np.diff(u)
    if gaps.min() <= 0:
        raise SystemExit(
            f"{path}: u levels are not strictly increasing (min gap {gaps.min():.3e} at "
            f"index {int(gaps.argmin())}, u={u[int(gaps.argmin())]!r}). Check the writer's "
            f"tag precision and output_u_grid's dedupe tolerance.")
    return u, nod


def read_qf(path: str, row_off: int = 0, n_rows: int | None = None):
    """``(u_levels, Q)`` with ``Q`` as ``[n_levels, n_rows, W]`` float32, NaN outside data.

    ``row_off``/``n_rows`` read one horizontal band instead of the whole raster. Reading it
    whole is 64 x 684.4 Mpx x 4 B = 175 GB on the global grid, and the nodata replacement
    used to double that. Africa's grid is 63.1 Mpx, where the same call is 16 GB and
    invisible -- a working set regional scale never exercised. The scale is applied in place
    for the same reason.
    """
    u, nod = qf_levels(path)
    with rasterio.open(path) as src:
        win = None if n_rows is None else Window(0, row_off, src.width, n_rows)
        q = src.read(window=win).astype(np.float32)
    if nod is not None:
        q[q == nod] = np.nan
    q *= np.float32(INT16_SCALE)
    return u, q


# ------------------------------------------------------------------- quantile-function maths

def _bracket(q, y):
    """Index of the first level at or above ``y``, and the interpolation weight.

    ``q`` is ``[n_levels, n_px]`` and monotone along axis 0; ``y`` is ``[n_px]``. Counting
    levels below ``y`` is O(levels x pixels) but fully vectorised, which beats a per-pixel
    ``searchsorted`` by two orders of magnitude at these sizes.
    """
    n = q.shape[0]
    idx = np.clip((q < y[None, :]).sum(axis=0), 1, n - 1)
    ar = np.arange(y.size)
    q0, q1 = q[idx - 1, ar], q[idx, ar]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(q1 > q0, (y - q0) / (q1 - q0), 0.0)
    return idx, np.clip(w, 0.0, 1.0)


def pit(u, q, y):
    """``u* = Q^-1(y)`` by linear interpolation, clamped to the represented range."""
    idx, w = _bracket(q, y)
    out = u[idx - 1] + w * (u[idx] - u[idx - 1])
    out = np.where(y <= q[0], 0.0, out)
    return np.where(y >= q[-1], 1.0, out)


def quantile_at(u, q, level: float):
    """``Q(level)`` for a level that need not be one of the stored ones."""
    j = int(np.clip(np.searchsorted(u, level), 1, u.size - 1))
    t = (level - u[j - 1]) / (u[j] - u[j - 1])
    return q[j - 1] + t * (q[j] - q[j - 1])


def crps_piecewise_linear(u, q, y, chunk: int = 250_000):
    """Exact CRPS of the piecewise-linear quantile function the raster represents.

    Not a quadrature. On a segment where ``Q`` is linear the pinball integrand is a quadratic
    in ``u``, so the integral has a closed form, and the one segment ``y`` crosses is split at
    the crossing so each piece keeps a single pinball branch. Exactness matters here because
    CRPS is the ranking metric of the whole phase: a systematic quadrature bias would reorder
    configurations while looking perfectly plausible.

    ``F(a0, c, m, L) = a0*c*L + (c - a0*m)*L^2/2 - m*L^3/3`` integrates ``(a0 + t)(c - m t)``
    over ``t`` in ``[0, L]``; the two pinball branches differ only in ``a0`` against ``a0 - 1``.
    The branch is chosen from the sign of the error at each piece's *midpoint* rather than at
    its origin -- the origin is degenerate exactly when the crossing is clipped to a segment
    end, which is the common case, not the rare one.

    The two flat tails beyond the stored grid (``u < u[0]`` and ``u > u[-1]``) contribute in
    closed form too. They are only 2e-4 of the probability, but they are the part of the
    distribution the far-field question is about, so they are not dropped.
    """
    n_px = y.size
    out = np.empty(n_px, dtype=np.float64)
    a_all = u[:-1]
    L_all = (u[1:] - u[:-1])[:, None]
    u0, ul = float(u[0]), float(u[-1])

    def F(a0, c, m, L):
        return a0 * c * L + (c - a0 * m) * L ** 2 / 2.0 - m * L ** 3 / 3.0

    for sl in range(0, n_px, chunk):
        e = slice(sl, min(sl + chunk, n_px))
        qq = q[:, e].astype(np.float64)
        yy = y[e].astype(np.float64)
        qa, qb = qq[:-1], qq[1:]
        L = np.broadcast_to(L_all, qa.shape)
        a = np.broadcast_to(a_all[:, None], qa.shape)
        m = (qb - qa) / L
        c = yy[None, :] - qa

        with np.errstate(invalid="ignore", divide="ignore"):
            t_star = np.where(m != 0, c / m, np.inf)
        t_star = np.clip(np.nan_to_num(t_star, nan=np.inf, posinf=np.inf, neginf=-np.inf),
                         0.0, L)

        # Piece 1 spans [0, t*]; piece 2 spans [t*, L] with its own origin error.
        L2 = L - t_star
        c2 = c - m * t_star
        tot = F(np.where(c - m * t_star / 2.0 >= 0, a, a - 1.0), c, m, t_star)
        a2 = a + t_star
        tot = tot + F(np.where(c2 - m * L2 / 2.0 >= 0, a2, a2 - 1.0), c2, m, L2)
        total = tot.sum(axis=0)

        # Flat tails outside the stored grid.
        e_lo = yy - qq[0]
        total = total + np.where(e_lo >= 0, e_lo * u0 ** 2 / 2.0,
                                 e_lo * (u0 ** 2 / 2.0 - u0))
        e_hi = yy - qq[-1]
        total = total + np.where(e_hi >= 0, e_hi * (1.0 - ul ** 2) / 2.0,
                                 -e_hi * (1.0 - ul) ** 2 / 2.0)
        out[e] = 2.0 * total
    return out


# ------------------------------------------------------------------------------- one row

def _read_band(path, row_off, n_rows):
    """``_read`` restricted to a horizontal band of the raster's own grid."""
    with rasterio.open(path) as src:
        arr = src.read(1, window=Window(0, row_off, src.width, n_rows)).astype(np.float64)
        nod = src.nodata
    if nod is not None and np.isfinite(nod):
        arr = np.where(arr == nod, np.nan, arr)
    return arr


def _read_like_band(path, ref, row_off, n_rows, band: int = 1):
    """``_read_like`` restricted to a band: a global raster on the reference grid's rows."""
    with rasterio.open(path) as src:
        col_off = int(round((ref["transform"].c - src.transform.c) / src.transform.a))
        r_off = int(round((ref["transform"].f - src.transform.f) / src.transform.e))
        arr = src.read(band, window=Window(col_off, r_off + row_off, ref["width"], n_rows),
                       boundless=True, fill_value=np.nan).astype(np.float64)
        nod = src.nodata
    if nod is not None and np.isfinite(nod):
        arr = np.where(arr == nod, np.nan, arr)
    return np.where(arr < -1e6, np.nan, arr)


def _cat(parts, key):
    return np.concatenate([p[key] for p in parts]) if len(parts) > 1 else parts[0][key]


def load_row(row, fold_sel, row_chunk: int = 0):
    """Compacted per-pixel arrays, the per-pixel scores, and the qf-vs-triple consistency.

    Streamed by horizontal band. The quantile-function raster is 64 bands, so holding one
    window-year whole is 175 GB on the 17111 x 40000 global grid against 16 GB on Africa --
    the third working set in this project that regional scale could not see. Everything the
    scorer computes per pixel is pixel-independent, and every stratum is a mean over a
    boolean mask, so banding changes nothing but the peak.

    ``consistency`` is folded in here because it needs the quantile function, which does not
    survive the band it was read in: the two max rows reduce with ``max`` and the outside-
    interval rate is a count-weighted mean, both exact.
    """
    with rasterio.open(row["path_central"]) as src:
        H, W = src.height, src.width
        ref = {"transform": src.transform, "width": W, "height": H}
    u = qf_levels(row["path_qf"])[0]
    step = row_chunk if row_chunk and row_chunk > 0 else H

    parts, pps = [], []
    lo_max = up_max = 0.0
    n_out = n_tot = 0
    for r0 in range(0, H, step):
        nr = min(step, H - r0)
        central = _read_band(row["path_central"], r0, nr)
        lower = _read_band(row["path_lower"], r0, nr)
        upper = _read_band(row["path_upper"], r0, nr)
        observed = _read_like_band(row["path_observed"], ref, r0, nr)
        hm_t0 = _read_like_band(row["path_baseline"], ref, r0, nr)
        dist = _read_like_band(row["path_context"], ref, r0, nr, band=2)
        _, qf = read_qf(row["path_qf"], r0, nr)

        ok = (np.isfinite(central) & np.isfinite(observed) & np.isfinite(hm_t0)
              & np.isfinite(lower) & np.isfinite(upper) & np.isfinite(dist)
              & np.isfinite(qf).all(axis=0))
        if fold_sel is not None:
            ok = ok & fold_sel[r0:r0 + nr]
        if not ok.any():
            del qf
            continue
        cell = dict(
            qf=qf[:, ok], central=central[ok], observed=observed[ok], hm_t0=hm_t0[ok],
            lower=lower[ok], upper=upper[ok], band=distance_band(dist[ok]),
            dhat_idx=np.digitize((central - hm_t0)[ok], DHAT_BINS[1:-1]),
            hm_idx=np.digitize(hm_t0[ok], HM_BINS[1:-1]),
            obs_idx=np.digitize((observed - hm_t0)[ok], OBS_MAG_BINS[1:-1]),
        )
        del qf, central, lower, upper, observed, hm_t0, dist
        c = consistency(u, cell)
        lo_max = max(lo_max, c["qf_vs_lower_max"])
        up_max = max(up_max, c["qf_vs_upper_max"])
        n_here = cell["central"].size
        n_out += c["central_outside_interval"] * n_here
        n_tot += n_here
        pps.append(per_pixel(u, cell))
        del cell["qf"]
        parts.append(cell)

    if not parts:
        empty = np.zeros(0)
        return u, {k: empty for k in ("central", "observed", "hm_t0", "lower", "upper",
                                      "band", "dhat_idx", "hm_idx", "obs_idx")}, {}, {}
    cell = {k: _cat(parts, k) for k in parts[0]}
    pp = {k: _cat(pps, k) for k in pps[0]}
    cons = {"qf_vs_lower_max": lo_max, "qf_vs_upper_max": up_max,
            "central_outside_interval": n_out / n_tot if n_tot else np.nan}
    return u, cell, pp, cons


def consistency(u, cell):
    """The qf and the published triple must describe the same forecast.

    If they do not, nothing downstream is trustworthy: the scorecard reads the triple, the
    ensemble reads the qf, and they would be scoring two different models. 0.025 and 0.975
    are levels of the grid, so this is an equality, and the only slack is int16 rounding at
    3.05e-5.
    """
    i_lo = int(np.argmin(np.abs(u - 0.025)))
    i_hi = int(np.argmin(np.abs(u - 0.975)))
    return {
        "qf_vs_lower_max": float(np.max(np.abs(cell["qf"][i_lo] - cell["lower"]))),
        "qf_vs_upper_max": float(np.max(np.abs(cell["qf"][i_hi] - cell["upper"]))),
        "central_outside_interval": float(np.mean(
            (cell["central"] < cell["lower"]) | (cell["central"] > cell["upper"]))),
    }


def per_pixel(u, cell):
    """Every per-pixel quantity, computed **once** for the whole raster.

    Each stratum is then a mean over a boolean mask. Recomputing CRPS inside every stratum
    instead costs a factor of the number of strata for an identical answer, and the strata
    partition the pixels, so there is nothing to gain from it.
    """
    qf, y, hm0 = cell["qf"], cell["observed"], cell["hm_t0"]
    out = {
        "crps": crps_piecewise_linear(u, qf, y),
        "mae_persistence": np.abs(y - hm0),
        "pit": pit(u, qf, y),
    }
    for lev in COVERAGE_LEVELS:
        lo = quantile_at(u, qf, (1.0 - lev) / 2.0)
        hi = quantile_at(u, qf, (1.0 + lev) / 2.0)
        out[f"cov{int(lev * 100)}"] = (y >= lo) & (y <= hi)
        out[f"width{int(lev * 100)}"] = hi - lo
    med = quantile_at(u, qf, 0.5)
    hw = quantile_at(u, qf, 0.975) - med
    reach = quantile_at(u, qf, 0.999) - med
    with np.errstate(invalid="ignore", divide="ignore"):
        out["tail_reach"] = np.where(hw > 1e-9, reach / hw, np.nan)
    out["halfwidth_p999"] = reach
    for thr, side in THRESHOLDS:
        key = f"p{'gt' if side == 'hi' else 'lt'}{abs(thr):g}".replace(".", "")
        u_thr = pit(u, qf, hm0 + thr)
        if side == "hi":
            out[f"{key}_pred"] = 1.0 - u_thr
            out[f"{key}_obs"] = (y - hm0) > thr
        else:
            out[f"{key}_pred"] = u_thr
            out[f"{key}_obs"] = (y - hm0) < thr
    return out


def gate_stats(u, cell, sel):
    """The two conv-spline gates over one stratum: the picket fence and PIT structure.

    These are why the phase exists, so they sit beside CRPS rather than in a side script --
    a diagnostic that has to be remembered is one that stops being run.
    """
    if sel.sum() < 100:
        return {}
    q = cell["qf"][:, sel]
    out = fence_gate(u, q)
    out.update(pit_structure(pit(u, q, cell["observed"][sel])))
    out.update(zero_leak(u, q, cell["observed"][sel], cell["hm_t0"][sel]))
    return out


def dist_stats(pp, sel):
    """Aggregate the per-pixel quantities over one stratum."""
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    crps = pp["crps"][sel]
    maep = pp["mae_persistence"][sel]
    p = np.sort(pp["pit"][sel])
    out = {
        "n": n,
        "crps": float(crps.mean()),
        "crps_persistence": float(maep.mean()),
        "crps_skill": float(1.0 - crps.mean() / maep.mean()) if maep.mean() > 0 else np.nan,
        "pit_mean": float(p.mean()),
        # KS against uniform: the sharpest single summary of whether the whole distribution
        # is right, not merely whether two of its percentiles are.
        "pit_ks": float(np.max(np.abs(p - (np.arange(n) + 0.5) / n))),
        "pit_gt_0975": float(np.mean(p > 0.975)),
        "pit_gt_0999": float(np.mean(p > 0.999)),
        "pit_lt_0025": float(np.mean(p < 0.025)),
        "pit_lt_0001": float(np.mean(p < 0.001)),
        "tail_reach_median": float(np.nanmedian(pp["tail_reach"][sel])),
        "tail_reach_p90": float(np.nanpercentile(pp["tail_reach"][sel], 90)),
        "halfwidth_p999_median": float(np.median(pp["halfwidth_p999"][sel])),
    }
    for lev in COVERAGE_LEVELS:
        k = int(lev * 100)
        out[f"cov{k}"] = float(pp[f"cov{k}"][sel].mean())
        out[f"width{k}"] = float(pp[f"width{k}"][sel].mean())
    for thr, side in THRESHOLDS:
        key = f"p{'gt' if side == 'hi' else 'lt'}{abs(thr):g}".replace(".", "")
        out[f"{key}_pred"] = float(pp[f"{key}_pred"][sel].mean())
        out[f"{key}_obs"] = float(pp[f"{key}_obs"][sel].mean())
    return out


def strata(cell):
    """``(stratum, bin, selection)`` triples: pooled first, then the four conditioning axes."""
    n = cell["observed"].size
    yield "pooled", "all", np.ones(n, dtype=bool)
    for name, idx, labels in (("distance", cell["band"], DIST_LABELS),
                              ("dhat", cell["dhat_idx"], DHAT_LABELS),
                              ("hm_t0", cell["hm_idx"], HM_LABELS),
                              ("obs_change", cell["obs_idx"], OBS_MAG_LABELS)):
        for k, label in enumerate(labels):
            sel = idx == k
            if sel.any():
                yield name, label, sel


def exceedance_error(df):
    """``mean |log10(pred / obs)|`` over the distance bands -- the placement summary.

    The same statistic ``predict_change_rates.py`` reports, so the two remain comparable
    across the change of model. Cells with no observed events are dropped rather than
    counted as an infinite error: a ratio whose denominator is zero is a statement about the
    denominator (southern Africa's >100 px band has no pixels at all).
    """
    sub = df[(df["stratum"] == "distance")]
    vals = []
    for _, r in sub.iterrows():
        for key in ("pgt001", "pgt005", "plt001"):
            o, p_ = r.get(f"{key}_obs"), r.get(f"{key}_pred")
            if o and o > 0 and p_ and p_ > 0:
                vals.append(abs(np.log10(p_ / o)))
    return float(np.mean(vals)) if vals else np.nan


def rows_from_stitched(stitched_dir: Path, context_pattern: str) -> list[dict]:
    out = []
    for window in WINDOWS:
        base = window[-1]
        for h in HORIZONS:
            year = base + h
            if year > MAX_OBSERVED_YEAR:
                continue
            paths = {q: stitched_dir / f"w{base}_prediction_{year}_{q}.tif"
                     for q in ("central", "lower", "upper", "qf")}
            if not all(p.exists() for p in paths.values()):
                continue
            out.append({
                "window": "-".join(str(y) for y in window), "base_year": base,
                "target_year": year, "horizon": h,
                **{f"path_{q}": str(p) for q, p in paths.items()},
                "path_observed": str(HM_DIR / f"HM_{year}_AA_1000.tiff"),
                "path_baseline": str(HM_DIR / f"HM_{base}_AA_1000.tiff"),
                "path_context": context_pattern.format(year=base),
            })
    if not out:
        raise SystemExit(
            f"no stitched triple+qf under {stitched_dir}. A run without "
            f"--head_family spline writes no *_qf.tif, and this scorer needs one.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stitched_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--folds", default=None,
                    help="Comma-separated fold ids; only those pixels are scored.")
    ap.add_argument("--fold_mask", default=str(REGION_ROOT / "fold_mask.tif"))
    ap.add_argument("--context_pattern",
                    default="data/raw/hm_global/change_context_w{year}_1000.tif")
    ap.add_argument("--subsample_blocks", type=int, default=0,
                   help="Score only N randomly chosen 128 px blocks of the grid instead of "
                        "every pixel. 0 (the default) is today's behaviour. This exists to "
                        "answer one question: can a cheap chip-sample screen stand in for the "
                        "full-raster score? Blocks rather than random pixels, because a "
                        "screen forward-passes contiguous chips and neighbouring pixels are "
                        "strongly correlated -- random pixels would flatter the estimate.")
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--row_chunk", type=int, default=0,
                    help="Read and score the rasters in bands of this many rows instead of "
                         "whole. 0 (the default) is today's behaviour. The quantile-function "
                         "raster is 64 bands, so one global window-year read whole is 175 GB "
                         "against 16 GB on Africa. Every per-pixel quantity is pixel-"
                         "independent and every stratum is a mean over a boolean mask, so "
                         "this changes the peak and nothing else.")
    ap.add_argument("--min_count", type=int, default=2000,
                    help="Strata thinner than this are written out but kept out of the "
                         "headline summary, where a 40-pixel cell would otherwise swing it.")
    args = ap.parse_args(argv)

    stitched = Path(args.stitched_dir)
    out_dir = Path(args.out_dir) if args.out_dir else stitched.parent / "score"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = rows_from_stitched(stitched, args.context_pattern)
    fold_sel = None
    if args.folds:
        want = [int(f) for f in str(args.folds).split(",")]
        with rasterio.open(rows[0]["path_central"]) as src:
            ref = {"transform": src.transform, "width": src.width, "height": src.height}
        fold_sel = np.isin(_read_like(args.fold_mask, ref), want)
        print(f"Scoring folds {want}: {int(fold_sel.sum()):,} px of the grid")
        # A fold mask from the wrong region silently scores a subset and reports a number that
        # looks fine. Measured: the default REGION_ROOT mask covers only southern Africa, so an
        # Africa run scored 680,594 of its 2,474,010 predicted pixels -- every "Africa" figure
        # would have been a southern-Africa figure wearing an Africa label. The stitcher writes
        # only held-out-fold pixels, so on a correct pairing essentially every finite pixel is
        # selected; a large shortfall means the mask does not match the rasters.
        with rasterio.open(rows[0]["path_central"]) as _src:
            _finite = np.isfinite(_src.read(1, masked=True).filled(np.nan))
        _cov = float((fold_sel & _finite).sum()) / max(int(_finite.sum()), 1)
        if _cov < 0.5:
            raise SystemExit(
                f"FATAL: --fold_mask {args.fold_mask} selects only {_cov:.1%} of the "
                f"{int(_finite.sum()):,} finite pixels in {rows[0]['path_central']}. "
                f"That mask does not cover these rasters -- pass the one prediction used "
                f"(the distributional drivers pass $FOLD_MASK).")
        print(f"  fold mask covers {_cov:.1%} of the finite pixels")

    if args.subsample_blocks > 0:
        with rasterio.open(rows[0]["path_central"]) as src:
            H, W = src.height, src.width
        B = 128
        nby, nbx = (H + B - 1) // B, (W + B - 1) // B
        keep = np.zeros((H, W), dtype=bool)
        # Only blocks that actually carry scorable pixels are candidates, or the sample is
        # mostly ocean and the effective size is a fiction.
        base = fold_sel if fold_sel is not None else np.ones((H, W), dtype=bool)
        cand = [(by, bx) for by in range(nby) for bx in range(nbx)
                if base[by * B:(by + 1) * B, bx * B:(bx + 1) * B].any()]
        rng = np.random.default_rng(args.subsample_seed)
        pick = rng.permutation(len(cand))[:args.subsample_blocks]
        for i in pick:
            by, bx = cand[i]
            keep[by * B:(by + 1) * B, bx * B:(bx + 1) * B] = True
        fold_sel = keep if fold_sel is None else (fold_sel & keep)
        print(f"Subsample: {len(pick)} of {len(cand)} candidate {B}px blocks, "
              f"{int(fold_sel.sum()):,} px scored")

    dist_recs, central_recs, consist_recs = [], [], []
    for row in rows:
        u, cell, pp, cons = load_row(row, fold_sel, args.row_chunk)
        n = cell["observed"].size
        print(f"  w{row['base_year']} h={row['horizon']:2d}  {n:,} px")
        if n == 0:
            continue
        consist_recs.append({"label": args.label, "window": row["window"],
                             "horizon": row["horizon"], "n": n, **cons})
        resid = cell["observed"] - cell["central"]
        obs_change = cell["observed"] - cell["hm_t0"]
        pred_change = cell["central"] - cell["hm_t0"]
        covered = (cell["observed"] >= cell["lower"]) & (cell["observed"] <= cell["upper"])
        for stratum, label, sel in strata(cell):
            base = {"label": args.label, "window": row["window"],
                    "horizon": row["horizon"], "stratum": stratum, "bin": label}
            # The gates are per-stratum but expensive (they re-read the qf), so they are
            # computed for the pooled row and the distance bands only -- the axes the phase
            # is judged on. Everything else stays on the cheap per-pixel path.
            gates = (gate_stats(u, cell, sel)
                     if stratum in ("pooled", "obs_change") else {})
            dist_recs.append({**base, **dist_stats(pp, sel), **gates})
            central_recs.append({**base, **central_stats(
                resid[sel], obs_change[sel], pred_change[sel], covered[sel])})

    dist_df = pd.DataFrame(dist_recs)
    central_df = pd.DataFrame(central_recs)
    consist_df = pd.DataFrame(consist_recs)
    dist_df.to_csv(out_dir / f"dist_{args.label}.csv", index=False)
    central_df.to_csv(out_dir / f"central_{args.label}.csv", index=False)
    consist_df.to_csv(out_dir / f"consistency_{args.label}.csv", index=False)

    def pooled(df, horizon=None):
        sub = df[df["stratum"] == "pooled"]
        if horizon is not None:
            sub = sub[sub["horizon"] == horizon]
        return sub

    def wmean(df, col):
        sub = df.dropna(subset=[col])
        return float(np.average(sub[col], weights=sub["n"])) if len(sub) else np.nan

    summary = {"label": args.label, "stitched_dir": str(stitched), "folds": args.folds}
    for h in HORIZONS:
        p = pooled(dist_df, h)
        if not len(p):
            continue
        summary[f"crps{h}"] = wmean(p, "crps")
        summary[f"crps_skill{h}"] = wmean(p, "crps_skill")
        summary[f"pit_ks{h}"] = wmean(p, "pit_ks")
        summary[f"pit_gt_0999_{h}"] = wmean(p, "pit_gt_0999")
        summary[f"cov95_{h}"] = wmean(p, "cov95")
        summary[f"tail_reach{h}"] = wmean(p, "tail_reach_median")
        # The two conv-spline gates, pooled. Both fence readings, because reporting one is
        # how an export-grid re-spacing gets recorded as a model fix.
        for k in ("needle_mass_median_export", "needle_mass_p90_export",
                  "max_density_p99_export", "over_f_max_frac_export",
                  "needle_mass_median_ref", "needle_mass_p90_ref",
                  "max_density_p99_ref", "over_f_max_frac_ref",
                  "pit_rms_se_20", "pit_rms_se_60", "pit_growth_vs_noise",
                  "pit_mean", "zero_leak_neg_ratio"):
            if k in p.columns:
                summary[f"{k}_{h}"] = wmean(p, k)
        c = pooled(central_df, h)
        summary[f"rmse{h}"] = wmean(c, "rmse")
        summary[f"skill{h}"] = wmean(c, "skill")
    big = dist_df[dist_df["n"] >= args.min_count]
    summary["exceedance_abs_log10"] = exceedance_error(big)
    summary["central_outside_interval"] = float(consist_df["central_outside_interval"].max())
    summary["qf_vs_triple_max"] = float(max(consist_df["qf_vs_lower_max"].max(),
                                            consist_df["qf_vs_upper_max"].max()))
    with open(out_dir / f"summary_{args.label}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"{args.label}: pooled by horizon")
    print("=" * 78)
    print(f"{'h':>3} {'n':>12} {'CRPS':>10} {'skill':>8} {'PIT KS':>8} "
          f"{'P(u>.999)':>10} {'cov95':>7} {'reach':>7} {'RMSE':>9} {'c.skill':>8}")
    for h in HORIZONS:
        if f"crps{h}" not in summary:
            continue
        p = pooled(dist_df, h)
        print(f"{h:>3} {int(p['n'].sum()):>12,} {summary[f'crps{h}']:>10.6f} "
              f"{summary[f'crps_skill{h}']:>8.4f} {summary[f'pit_ks{h}']:>8.4f} "
              f"{summary[f'pit_gt_0999_{h}']:>10.5f} {summary[f'cov95_{h}']:>7.4f} "
              f"{summary[f'tail_reach{h}']:>7.2f} {summary[f'rmse{h}']:>9.6f} "
              f"{summary[f'skill{h}']:>8.4f}")
    print("\n" + "-" * 78)
    print("conv-spline gates: the picket fence and PIT structure")
    print("-" * 78)
    print(f"{'h':>3} {'needle p50':>11} {'needle p90':>11} {'maxdens p99':>12} "
          f"{'>f_max':>8} | {'ref p50':>9} {'ref maxd':>9} | {'PITrms20':>9} {'PITrms60':>9} "
          f"{'growth':>7} {'PITmean':>8} {'0leak':>7}")
    for h in HORIZONS:
        if f"needle_mass_median_export_{h}" not in summary:
            continue
        g = lambda k: summary.get(f"{k}_{h}", float("nan"))
        print(f"{h:>3} {g('needle_mass_median_export'):>11.4f} "
              f"{g('needle_mass_p90_export'):>11.4f} {g('max_density_p99_export'):>12.1f} "
              f"{g('over_f_max_frac_export'):>8.3f} | {g('needle_mass_median_ref'):>9.4f} "
              f"{g('max_density_p99_ref'):>9.1f} | {g('pit_rms_se_20'):>9.2f} "
              f"{g('pit_rms_se_60'):>9.2f} {g('pit_growth_vs_noise'):>7.3f} "
              f"{g('pit_mean'):>8.4f} {g('zero_leak_neg_ratio'):>7.2f}")
    print("targets: needle ~0, maxdens <= 578, >f_max ~0, PITrms ~1, PITmean 0.50, 0leak 1.00")

    print(f"\nexceedance mean|log10(pred/obs)| over distance bands: "
          f"{summary['exceedance_abs_log10']:.4f}")
    print(f"qf vs published triple, max |diff|: {summary['qf_vs_triple_max']:.2e} "
          f"(int16 quantum is 3.05e-05)")
    print(f"E[Q] outside its own 95% interval: "
          f"{summary['central_outside_interval']:.5f} of pixels")

    band = dist_df[(dist_df["stratum"] == "distance") & (dist_df["horizon"] == HORIZONS[-1])]
    if len(band):
        agg = band.groupby("bin").apply(
            lambda g: pd.Series({
                "n": g["n"].sum(),
                "P(>0.05) pred": np.average(g["pgt005_pred"], weights=g["n"]),
                "P(>0.05) obs": np.average(g["pgt005_obs"], weights=g["n"]),
                "reach": np.average(g["tail_reach_median"], weights=g["n"]),
                "crps_skill": np.average(g["crps_skill"], weights=g["n"]),
            }), include_groups=False)
        agg = agg.reindex([b for b in DIST_LABELS if b in agg.index])
        print(f"\nby distance band at h={HORIZONS[-1]}:")
        print(agg.to_string(float_format=lambda v: f"{v:.5f}"))
        print("  (southern Africa has NO pixels beyond 100 px: that row is absent, not zero)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
