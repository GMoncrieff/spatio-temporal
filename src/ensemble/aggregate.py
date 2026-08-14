"""Phase 4 — region statistics, ensemble summaries, and proper scoring rules.

The point of the ensemble is that an aggregate statistic is computed *per member first* and
summarized afterwards. Averaging the pixelwise bounds instead is exactly the mistake the
project exists to fix: independent per-pixel noise cancels under aggregation, so
propagating pixel intervals into an aggregate understates its uncertainty badly.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window

from .copula import INT16_SENTINEL
from .validate import _stripe_rows


# --------------------------------------------------------------------------------------
# Store access
# --------------------------------------------------------------------------------------
def open_ensemble(path):
    import zarr

    z = zarr.open(str(path), mode="r")
    attrs = dict(z.attrs)
    return z, attrs


def dequantize_block(q, attrs):
    scale = float(attrs.get("scale", 1.0 / 32767.0))
    offset = float(attrs.get("offset", 0.0))
    out = q.astype(np.float32) * scale + offset
    return np.where(q == INT16_SENTINEL, np.nan, out)


def member_slice(store, attrs, member, horizon_idx, window=None):
    if window is None:
        q = store[member, horizon_idx]
    else:
        r0, r1, c0, c1 = window
        q = store[member, horizon_idx, r0:r1, c0:c1]
    return dequantize_block(np.asarray(q), attrs)


# --------------------------------------------------------------------------------------
# Region / zonal statistics
# --------------------------------------------------------------------------------------
def aggregate_region_statistic(zarr_store, region_mask, horizon, statistic_fn=None, threshold=None,
                               attrs=None, block_rows: int = 2048):
    """Value of a region statistic for every member.

    ``statistic_fn`` receives the member's masked pixel values and returns a scalar; the
    default is the mean, and passing ``threshold`` switches to area-above-threshold (the
    fraction of the region above it), which is strictly more sensitive to spatial structure
    than a mean and is what T2.3 scores.
    """
    store, at = (zarr_store, attrs) if attrs is not None else open_ensemble(zarr_store)
    M = store.shape[0]
    mask = np.asarray(region_mask, dtype=bool)
    out = np.full(M, np.nan)
    for m in range(M):
        num = 0.0
        den = 0
        vals_acc = []
        for r0 in range(0, mask.shape[0], block_rows):
            r1 = min(r0 + block_rows, mask.shape[0])
            sub = mask[r0:r1]
            if not sub.any():
                continue
            v = dequantize_block(np.asarray(store[m, horizon, r0:r1]), at)
            v = v[sub]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            if statistic_fn is not None:
                vals_acc.append(v)
            elif threshold is not None:
                num += float((v > threshold).sum())
                den += v.size
            else:
                num += float(v.sum())
                den += v.size
        if statistic_fn is not None:
            out[m] = statistic_fn(np.concatenate(vals_acc)) if vals_acc else np.nan
        elif den > 0:
            out[m] = num / den
    return out


def zonal_member_stats(zarr_store, horizon_idx, zone_raster, attrs=None, thresholds=(0.1, 0.3),
                       block_rows: int = 1024, max_zone=None):
    """Per-member zonal means and area-above-threshold, in one pass per member.

    Returns ``{"zone_ids", "n_px", "mean": (M, n_zones), "area{t}": (M, n_zones)}``.
    """
    store, at = (zarr_store, attrs) if attrs is not None else open_ensemble(zarr_store)
    M, _, H, W = store.shape
    with rasterio.open(zone_raster) as zsrc:
        z_t = zsrc.transform
        max_zone = max_zone or (65535 if zsrc.dtypes[0] == "uint16" else 4096)
    n_zones = int(max_zone) + 1

    cnt = np.zeros(n_zones, dtype=np.int64)
    means = np.zeros((M, n_zones))
    areas = {t: np.zeros((M, n_zones)) for t in thresholds}

    with rasterio.open(zone_raster) as zsrc:
        for r0 in range(0, H, block_rows):
            rr = min(block_rows, H - r0)
            zn = zsrc.read(1, window=Window(0, r0, W, rr), boundless=True, fill_value=0)
            if not (zn > 0).any():
                continue
            for m in range(M):
                v = dequantize_block(np.asarray(store[m, horizon_idx, r0:r0 + rr]), at)
                ok = np.isfinite(v) & (zn > 0)
                if not ok.any():
                    continue
                zf = zn[ok].astype(np.int64)
                if m == 0:
                    cnt += np.bincount(zf, minlength=n_zones)
                means[m] += np.bincount(zf, weights=v[ok], minlength=n_zones)
                for t in thresholds:
                    areas[t][m] += np.bincount(zf, weights=(v[ok] > t).astype(float), minlength=n_zones)

    keep = cnt > 0
    ids = np.nonzero(keep)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"zone_ids": ids, "n_px": cnt[keep], "mean": means[:, keep] / cnt[keep]}
        for t in thresholds:
            out[f"area{t}"] = areas[t][:, keep] / cnt[keep]
    return out


def zonal_observed(observed_path, zone_raster, reference_profile, thresholds=(0.1, 0.3),
                   block_rows: int = 1024, mask_path=None):
    """Observed zonal means / areas on exactly the same pixels the ensemble covers."""
    with rasterio.open(zone_raster) as zsrc:
        n_zones = (65535 if zsrc.dtypes[0] == "uint16" else 4096) + 1
    cnt = np.zeros(n_zones, dtype=np.int64)
    s_mean = np.zeros(n_zones)
    s_area = {t: np.zeros(n_zones) for t in thresholds}
    H, W = reference_profile["height"], reference_profile["width"]
    p_t = reference_profile["transform"]
    msrc = rasterio.open(mask_path) if mask_path else None
    with rasterio.open(observed_path) as osrc, rasterio.open(zone_raster) as zsrc:
        o_t, z_t = osrc.transform, zsrc.transform
        o_off = (int(round((p_t.f - o_t.f) / o_t.e)), int(round((p_t.c - o_t.c) / o_t.a)))
        z_off = (int(round((p_t.f - z_t.f) / z_t.e)), int(round((p_t.c - z_t.c) / z_t.a)))
        for r0 in range(0, H, block_rows):
            rr = min(block_rows, H - r0)
            ob = osrc.read(1, window=Window(o_off[1], o_off[0] + r0, W, rr),
                           boundless=True, fill_value=np.nan).astype(np.float64)
            zn = zsrc.read(1, window=Window(z_off[1], z_off[0] + r0, W, rr),
                           boundless=True, fill_value=0)
            ob = np.where(ob < 0, np.nan, ob)
            ok = np.isfinite(ob) & (zn > 0)
            if msrc is not None:
                ok &= np.isfinite(msrc.read(1, window=Window(0, r0, W, rr)).astype(np.float32))
            if not ok.any():
                continue
            zf = zn[ok].astype(np.int64)
            cnt += np.bincount(zf, minlength=n_zones)
            s_mean += np.bincount(zf, weights=ob[ok], minlength=n_zones)
            for t in thresholds:
                s_area[t] += np.bincount(zf, weights=(ob[ok] > t).astype(float), minlength=n_zones)
    if msrc is not None:
        msrc.close()
    keep = cnt > 0
    ids = np.nonzero(keep)[0]
    out = {"zone_ids": ids, "n_px": cnt[keep], "mean": s_mean[keep] / cnt[keep]}
    for t in thresholds:
        out[f"area{t}"] = s_area[t][keep] / cnt[keep]
    return out


def block_member_stats(zarr_store, horizon_idx, block_size, attrs=None, min_valid_frac=0.5,
                       stripe_blocks: int = 8):
    """Block means per member: ``(M, n_block_rows, n_block_cols)`` plus a validity mask."""
    store, at = (zarr_store, attrs) if attrs is not None else open_ensemble(zarr_store)
    M, _, H, W = store.shape
    B = int(block_size)
    n_bi, n_bj = H // B, W // B
    sums = np.zeros((M, n_bi, n_bj), dtype=np.float64)
    cnt = np.zeros((n_bi, n_bj), dtype=np.int64)
    stripe = _stripe_rows(B, n_bj, stripe_blocks)
    for r0 in range(0, n_bi * B, stripe):
        rr = min(stripe, n_bi * B - r0)
        for m in range(M):
            v = dequantize_block(np.asarray(store[m, horizon_idx, r0:r0 + rr, :n_bj * B]), at)
            ok = np.isfinite(v)
            sums[m, r0 // B: r0 // B + rr // B] += np.where(ok, v, 0.0).reshape(
                rr // B, B, n_bj, B).sum(axis=(1, 3))
            if m == 0:
                cnt[r0 // B: r0 // B + rr // B] += ok.reshape(rr // B, B, n_bj, B).sum(axis=(1, 3))
    valid = cnt >= max(1, int(min_valid_frac * B * B))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / np.maximum(cnt, 1)
    return means, valid, cnt


def block_member_stats_multi(zarr_store, horizon_idx, block_sizes, attrs=None,
                             min_valid_frac=0.5, stripe_blocks: int = 8):
    """Block means per member at several nested scales, in **one** pass over the members.

    Reading a 50-member global ensemble costs ~68 GB per pass, so doing it once per block
    size is the dominant cost of T2.1. When the sizes are nested (10, 100, 1000) the coarse
    sums are just aggregates of the fine ones, and only the finest scale needs the data.
    """
    sizes = sorted(int(b) for b in block_sizes)
    base = sizes[0]
    for b in sizes[1:]:
        if b % base:
            raise ValueError(f"block sizes must be multiples of the smallest ({base}): {sizes}")

    means, valid, cnt = block_member_stats(
        zarr_store, horizon_idx, base, attrs=attrs, min_valid_frac=min_valid_frac,
        stripe_blocks=stripe_blocks)
    sums = means * cnt[None, :, :]
    out = {base: (means, valid, cnt)}
    for b in sizes[1:]:
        f = b // base
        n_bi, n_bj = sums.shape[1] // f, sums.shape[2] // f
        s = sums[:, : n_bi * f, : n_bj * f].reshape(sums.shape[0], n_bi, f, n_bj, f).sum(axis=(2, 4))
        c = cnt[: n_bi * f, : n_bj * f].reshape(n_bi, f, n_bj, f).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[b] = (s / np.maximum(c, 1), c >= max(1, int(min_valid_frac * b * b)), c)
    return out


def block_observed_multi(observed_path, reference_profile, block_sizes, min_valid_frac=0.5,
                         stripe_blocks: int = 8, mask_path=None):
    """Observed block means at nested scales, aggregated from the finest."""
    sizes = sorted(int(b) for b in block_sizes)
    base = sizes[0]
    mean, valid, cnt = block_observed(observed_path, reference_profile, base,
                                      min_valid_frac=min_valid_frac, mask_path=mask_path,
                                      stripe_blocks=stripe_blocks, return_counts=True)
    out = {base: (mean, valid)}
    sums = np.nan_to_num(mean) * cnt
    for b in sizes[1:]:
        f = b // base
        n_bi, n_bj = sums.shape[0] // f, sums.shape[1] // f
        s = sums[: n_bi * f, : n_bj * f].reshape(n_bi, f, n_bj, f).sum(axis=(1, 3))
        c = cnt[: n_bi * f, : n_bj * f].reshape(n_bi, f, n_bj, f).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[b] = (s / np.maximum(c, 1), c > 0)
    return out


def block_observed(observed_path, reference_profile, block_size, min_valid_frac=0.5,
                   stripe_blocks: int = 8, return_counts: bool = False, mask_path=None):
    """Observed block means.

    ``mask_path`` restricts the average to the pixels the ensemble actually covers. Without
    it the observed mean is taken over a *different* pixel set than the member means —
    coastal and prediction-gap pixels enter one and not the other — and the two aggregates
    are then not comparable at all, which shows up as a spurious coverage collapse.
    """
    B = int(block_size)
    H, W = reference_profile["height"], reference_profile["width"]
    p_t = reference_profile["transform"]
    n_bi, n_bj = H // B, W // B
    sums = np.zeros((n_bi, n_bj))
    cnt = np.zeros((n_bi, n_bj), dtype=np.int64)
    msrc = rasterio.open(mask_path) if mask_path else None
    with rasterio.open(observed_path) as osrc:
        o_t = osrc.transform
        o_off = (int(round((p_t.f - o_t.f) / o_t.e)), int(round((p_t.c - o_t.c) / o_t.a)))
        stripe = _stripe_rows(B, n_bj, stripe_blocks)
        for r0 in range(0, n_bi * B, stripe):
            rr = min(stripe, n_bi * B - r0)
            ob = osrc.read(1, window=Window(o_off[1], o_off[0] + r0, n_bj * B, rr),
                           boundless=True, fill_value=np.nan).astype(np.float64)
            ob = np.where(ob < 0, np.nan, ob)
            ok = np.isfinite(ob)
            if msrc is not None:
                mk = msrc.read(1, window=Window(0, r0, n_bj * B, rr)).astype(np.float32)
                ok &= np.isfinite(mk)
            sums[r0 // B: r0 // B + rr // B] += np.where(ok, ob, 0.0).reshape(
                rr // B, B, n_bj, B).sum(axis=(1, 3))
            cnt[r0 // B: r0 // B + rr // B] += ok.reshape(rr // B, B, n_bj, B).sum(axis=(1, 3))
    if msrc is not None:
        msrc.close()
    valid = cnt >= max(1, int(min_valid_frac * B * B))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / np.maximum(cnt, 1)
    return (mean, valid, cnt) if return_counts else (mean, valid)


# --------------------------------------------------------------------------------------
# Summaries and scores
# --------------------------------------------------------------------------------------
def summarize_ensemble(stat_values, qs=(2.5, 97.5)):
    v = np.asarray(stat_values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": np.nan, "p2_5": np.nan, "p97_5": np.nan, "n": 0}
    lo, hi = np.percentile(v, qs)
    return {"median": float(np.median(v)), "p2_5": float(lo), "p97_5": float(hi),
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def coverage_from_members(member_stats, observed, qs=(2.5, 97.5)):
    """Fraction of units whose observed value lies inside the ensemble interval."""
    member_stats = np.asarray(member_stats, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    lo = np.nanpercentile(member_stats, qs[0], axis=0)
    hi = np.nanpercentile(member_stats, qs[1], axis=0)
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(observed)
    covered = (observed >= lo) & (observed <= hi) & ok
    return {
        "n": int(ok.sum()), "n_covered": int(covered.sum()),
        "coverage": float(covered.sum() / max(ok.sum(), 1)),
        "mean_width": float(np.nanmean((hi - lo)[ok])) if ok.any() else np.nan,
        "frac_below": float(np.sum((observed < lo) & ok) / max(ok.sum(), 1)),
        "frac_above": float(np.sum((observed > hi) & ok) / max(ok.sum(), 1)),
    }


# Thresholds for the change-sign realism target (T6). Negative HM change is real but
# rare, and large negative change is ~30x rarer than the equivalent increase at +20yr.
CHANGE_THRESHOLDS = (-0.15, -0.05, -0.01, -0.001, 0.001, 0.01, 0.05, 0.15)


def change_distribution(zarr_store, horizon_idx, baseline_hm_path, reference_profile,
                        attrs=None, thresholds=CHANGE_THRESHOLDS, members=None,
                        block_rows: int = 512, observed_path=None):
    """Tail fractions of ``member − HM_t0`` (and of the observation), for T6.

    Scores whether the *members themselves* are plausible, which no coverage target does:
    an interval can cover perfectly while being made of fields that collapse HM in places
    the real world never does.
    """
    store, at = (zarr_store, attrs) if attrs is not None else open_ensemble(zarr_store)
    M, _, H, W = store.shape
    members = list(range(M)) if members is None else list(members)
    p_t = reference_profile["transform"]

    counts = np.zeros(len(thresholds), dtype=np.int64)
    obs_counts = np.zeros(len(thresholds), dtype=np.int64)
    n_tot = 0
    n_obs = 0
    quant_sample = []
    obs_sample = []
    rng = np.random.default_rng(0)

    with rasterio.open(baseline_hm_path) as bsrc:
        b_t = bsrc.transform
        b_off = (int(round((p_t.f - b_t.f) / b_t.e)), int(round((p_t.c - b_t.c) / b_t.a)))
        osrc = rasterio.open(observed_path) if observed_path else None
        if osrc is not None:
            o_t = osrc.transform
            o_off = (int(round((p_t.f - o_t.f) / o_t.e)), int(round((p_t.c - o_t.c) / o_t.a)))
        try:
            for r0 in range(0, H, block_rows):
                rr = min(block_rows, H - r0)
                hm0 = bsrc.read(1, window=Window(b_off[1], b_off[0] + r0, W, rr),
                                boundless=True, fill_value=np.nan).astype(np.float32)
                hm0 = np.where(hm0 < 0, np.nan, hm0)
                if not np.isfinite(hm0).any():
                    continue
                if osrc is not None:
                    ob = osrc.read(1, window=Window(o_off[1], o_off[0] + r0, W, rr),
                                   boundless=True, fill_value=np.nan).astype(np.float32)
                    ob = np.where(ob < 0, np.nan, ob)
                    d_ob = ob - hm0
                    fin = np.isfinite(d_ob)
                    n_obs += int(fin.sum())
                    for k, t in enumerate(thresholds):
                        obs_counts[k] += int((d_ob[fin] < t).sum() if t < 0 else (d_ob[fin] > t).sum())
                    if fin.any() and len(obs_sample) < 40:
                        v = d_ob[fin]
                        obs_sample.append(v[rng.integers(0, v.size, min(v.size, 200_000))])
                for m in members:
                    v = dequantize_block(np.asarray(store[m, horizon_idx, r0:r0 + rr]), at)
                    d = v - hm0
                    fin = np.isfinite(d)
                    n_tot += int(fin.sum())
                    dv = d[fin]
                    for k, t in enumerate(thresholds):
                        counts[k] += int((dv < t).sum() if t < 0 else (dv > t).sum())
                    if len(quant_sample) < 40 and dv.size:
                        quant_sample.append(dv[rng.integers(0, dv.size, min(dv.size, 200_000))])
        finally:
            if osrc is not None:
                osrc.close()

    out = {"n_member_px": n_tot, "n_observed_px": n_obs,
           "thresholds": list(thresholds),
           "member_frac": (counts / max(n_tot, 1)).tolist(),
           "observed_frac": (obs_counts / max(n_obs, 1)).tolist() if n_obs else None}
    if quant_sample:
        s = np.concatenate(quant_sample)
        out["member_q01"], out["member_q05"] = float(np.quantile(s, 0.01)), float(np.quantile(s, 0.05))
    if obs_sample:
        s = np.concatenate(obs_sample)
        out["observed_q01"], out["observed_q05"] = float(np.quantile(s, 0.01)), float(np.quantile(s, 0.05))
    return out


def interval_score(lower, upper, observed, alpha: float = 0.05):
    """Winkler interval score; lower is better.

    ``(u−l) + (2/α)(l−y)·1{y<l} + (2/α)(y−u)·1{y>u}``

    A proper scoring rule, and the reason coverage is never scored alone here: widening is
    only rewarded when it buys back more miscoverage penalty than it costs in width, so an
    interval cannot win by being enormous.
    """
    l = np.asarray(lower, dtype=np.float64)
    u = np.asarray(upper, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    ok = np.isfinite(l) & np.isfinite(u) & np.isfinite(y)
    if not ok.any():
        return {"interval_score": np.nan, "width_term": np.nan, "penalty_term": np.nan, "n": 0}
    l, u, y = l[ok], u[ok], y[ok]
    width = u - l
    penalty = (2.0 / alpha) * (np.maximum(l - y, 0.0) + np.maximum(y - u, 0.0))
    return {
        "interval_score": float(np.mean(width + penalty)),
        "width_term": float(np.mean(width)),
        "penalty_term": float(np.mean(penalty)),
        "n": int(ok.sum()),
    }


def interval_score_from_members(member_stats, observed, alpha: float = 0.05, qs=(2.5, 97.5)):
    m = np.asarray(member_stats, dtype=np.float64)
    lo = np.nanpercentile(m, qs[0], axis=0)
    hi = np.nanpercentile(m, qs[1], axis=0)
    return interval_score(lo, hi, observed, alpha=alpha)


def crps_from_members(member_stats, observed):
    """CRPS estimated from a finite ensemble (fair/unbiased form); lower is better.

    ``CRPS = mean|X_i − y| − 1/(2M(M−1)) * sum_ij |X_i − X_j|``
    """
    X = np.asarray(member_stats, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=0)
    if not ok.any():
        return np.nan
    X, y = X[:, ok], y[ok]
    M = X.shape[0]
    term1 = np.mean(np.abs(X - y[None, :]), axis=0)
    Xs = np.sort(X, axis=0)
    # sum_ij |Xi - Xj| via the sorted-order identity, O(M log M) instead of O(M^2)
    w = (2 * np.arange(1, M + 1) - M - 1).astype(np.float64)[:, None]
    pair = 2.0 * (w * Xs).sum(axis=0)
    term2 = pair / (2.0 * M * max(M - 1, 1))
    return float(np.mean(term1 - term2))


def spread_skill_ratio(member_stats, observed):
    """Ensemble sd vs RMSE of the ensemble mean (T7.3). 1.0 means correctly dispersed."""
    X = np.asarray(member_stats, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=0)
    if ok.sum() < 2:
        return {"spread": np.nan, "rmse": np.nan, "ratio": np.nan, "n": int(ok.sum())}
    X, y = X[:, ok], y[ok]
    M = X.shape[0]
    spread = float(np.sqrt(np.mean(X.var(axis=0, ddof=1) * (M + 1) / M)))
    rmse = float(np.sqrt(np.mean((X.mean(axis=0) - y) ** 2)))
    return {"spread": spread, "rmse": rmse,
            "ratio": float(spread / rmse) if rmse > 0 else np.nan, "n": int(ok.sum())}


def member_diversity(member_fields):
    """Mean pairwise correlation between members (T7.2); 1.0 means they are identical."""
    X = np.asarray(member_fields, dtype=np.float64)
    X = X.reshape(X.shape[0], -1)
    ok = np.isfinite(X).all(axis=0)
    X = X[:, ok]
    if X.shape[1] < 10 or X.shape[0] < 2:
        return {"mean_pairwise_corr": np.nan, "min": np.nan, "max": np.nan}
    C = np.corrcoef(X)
    iu = np.triu_indices_from(C, k=1)
    v = C[iu]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean_pairwise_corr": np.nan, "min": np.nan, "max": np.nan}
    return {"mean_pairwise_corr": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def rank_histogram(member_stats, observed, n_bins=None):
    """Rank of the observation among the members (M members -> M+1 bins).

    Ties are broken randomly, which is the standard treatment and keeps a discrete
    ensemble from producing spurious spikes at the extreme bins.
    """
    member_stats = np.asarray(member_stats, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    M = member_stats.shape[0]
    n_bins = n_bins or (M + 1)
    ok = np.isfinite(observed) & np.isfinite(member_stats).all(axis=0)
    if not ok.any():
        return np.zeros(n_bins, dtype=int)
    rng = np.random.default_rng(0)
    below = (member_stats[:, ok] < observed[ok]).sum(axis=0)
    ties = (member_stats[:, ok] == observed[ok]).sum(axis=0)
    ranks = below + (rng.random(below.shape) * (ties + 1)).astype(int)
    return np.bincount(np.clip(ranks, 0, n_bins - 1), minlength=n_bins)


def rank_histogram_test(hist):
    """Chi-square goodness of fit against uniformity, plus a reliability index."""
    from scipy.stats import chisquare

    hist = np.asarray(hist, dtype=float)
    n = hist.sum()
    if n < 10 or (hist > 0).sum() < 2:
        return {"chi2": np.nan, "p_value": np.nan, "reliability_index": np.nan, "n": int(n)}
    expected = np.full_like(hist, n / hist.size)
    chi2, p = chisquare(hist, expected)
    ri = float(np.sum(np.abs(hist / n - 1.0 / hist.size)))
    return {"chi2": float(chi2), "p_value": float(p), "reliability_index": ri, "n": int(n)}


def energy_score(members, observation):
    """Energy score (multivariate CRPS generalization); lower is better."""
    X = np.asarray(members, dtype=np.float64)  # (M, D)
    y = np.asarray(observation, dtype=np.float64)
    M = X.shape[0]
    term1 = np.mean(np.linalg.norm(X - y[None, :], axis=1))
    diff = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    term2 = diff.sum() / (2.0 * M * M)
    return float(term1 - term2)


def variogram_score(members, observation, pairs, p: float = 0.5, weights=None):
    """Variogram score of order p (Scheuerer & Hamill); lower is better.

    Sensitive to the *correlation* structure rather than the marginals — which is exactly
    what separates the copula ensemble from an independent-pixel ensemble with identical
    marginals.

    Reported as a weighted **mean** over pairs, not a sum. A sum makes the number depend on
    how many of the requested pairs survived the distance filter, which varies between runs
    and between weighting schemes; the mean is comparable across both. Ratios between two
    ensembles scored on identical pairs are unaffected either way.
    """
    X = np.asarray(members, dtype=np.float64)
    y = np.asarray(observation, dtype=np.float64)
    i, j = pairs
    if len(i) == 0:
        return float("nan")
    obs_term = np.abs(y[i] - y[j]) ** p
    ens_term = np.mean(np.abs(X[:, i] - X[:, j]) ** p, axis=0)
    sq = (obs_term - ens_term) ** 2
    if weights is None:
        return float(np.mean(sq))
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return float(np.mean(sq))
    return float(np.sum(w * sq) / total)


def informative_pair_fraction(spread, pairs, rel_floor: float = 0.05):
    """Share of pairs where the ensemble has enough spread to have a structure at all.

    A pair whose two endpoints both carry near-zero spread contributes the same quantity to
    a correlated ensemble and to an independent one — for both, the members collapse onto
    the central forecast and the variogram term reduces to |central_i − central_j|^p. Such
    pairs cancel in the ratio while still diluting it, so a variogram-score comparison is
    only as informative as this fraction is large.

    ``rel_floor`` is expressed relative to a high quantile of the non-zero spread, so the
    threshold follows the ensemble rather than being an absolute HM number that stops
    meaning the same thing when the marginals tighten. It has to be a *high* quantile: in
    exactly the regime this function exists to detect, the degenerate background is the
    large majority, so any central statistic — median, even the 90th percentile — sits
    inside the dead part and the threshold collapses to "everything counts". The 99th
    percentile tracks the live scale as long as the live region is more than ~1% of the
    map, and the alternative (the bare maximum) would be at the mercy of one pixel.
    """
    s = np.asarray(spread, dtype=np.float64)
    i, j = pairs
    if len(i) == 0:
        return float("nan")
    pos = s[np.isfinite(s) & (s > 0)]
    if pos.size == 0:
        return 0.0
    thresh = rel_floor * float(np.quantile(pos, 0.99))
    live = np.isfinite(s) & (s > thresh)
    return float(np.mean(live[i] & live[j]))


def sample_pairs(n_points, n_pairs, rng=None, coords=None, max_dist=None):
    rng = rng or np.random.default_rng(0)
    i = rng.integers(0, n_points, n_pairs)
    j = rng.integers(0, n_points, n_pairs)
    keep = i != j
    if coords is not None and max_dist is not None:
        d = np.linalg.norm(coords[i] - coords[j], axis=1)
        keep &= d <= max_dist
    return i[keep], j[keep]
