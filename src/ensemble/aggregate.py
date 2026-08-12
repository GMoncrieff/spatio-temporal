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
                   block_rows: int = 1024):
    """Observed zonal means / areas on exactly the same pixels."""
    with rasterio.open(zone_raster) as zsrc:
        n_zones = (65535 if zsrc.dtypes[0] == "uint16" else 4096) + 1
    cnt = np.zeros(n_zones, dtype=np.int64)
    s_mean = np.zeros(n_zones)
    s_area = {t: np.zeros(n_zones) for t in thresholds}
    H, W = reference_profile["height"], reference_profile["width"]
    p_t = reference_profile["transform"]
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
            if not ok.any():
                continue
            zf = zn[ok].astype(np.int64)
            cnt += np.bincount(zf, minlength=n_zones)
            s_mean += np.bincount(zf, weights=ob[ok], minlength=n_zones)
            for t in thresholds:
                s_area[t] += np.bincount(zf, weights=(ob[ok] > t).astype(float), minlength=n_zones)
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
    max_rows = max(B, int(4e8 / max(n_bj * B * 8, 1)) // B * B)
    stripe = min(max(1, stripe_blocks) * B, max_rows)
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


def block_observed(observed_path, reference_profile, block_size, min_valid_frac=0.5,
                   stripe_blocks: int = 8):
    B = int(block_size)
    H, W = reference_profile["height"], reference_profile["width"]
    p_t = reference_profile["transform"]
    n_bi, n_bj = H // B, W // B
    sums = np.zeros((n_bi, n_bj))
    cnt = np.zeros((n_bi, n_bj), dtype=np.int64)
    with rasterio.open(observed_path) as osrc:
        o_t = osrc.transform
        o_off = (int(round((p_t.f - o_t.f) / o_t.e)), int(round((p_t.c - o_t.c) / o_t.a)))
        max_rows = max(B, int(4e8 / max(n_bj * B * 8, 1)) // B * B)
        stripe = min(max(1, stripe_blocks) * B, max_rows)
        for r0 in range(0, n_bi * B, stripe):
            rr = min(stripe, n_bi * B - r0)
            ob = osrc.read(1, window=Window(o_off[1], o_off[0] + r0, n_bj * B, rr),
                           boundless=True, fill_value=np.nan).astype(np.float64)
            ob = np.where(ob < 0, np.nan, ob)
            ok = np.isfinite(ob)
            sums[r0 // B: r0 // B + rr // B] += np.where(ok, ob, 0.0).reshape(
                rr // B, B, n_bj, B).sum(axis=(1, 3))
            cnt[r0 // B: r0 // B + rr // B] += ok.reshape(rr // B, B, n_bj, B).sum(axis=(1, 3))
    valid = cnt >= max(1, int(min_valid_frac * B * B))
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / np.maximum(cnt, 1), valid


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
    """
    X = np.asarray(members, dtype=np.float64)
    y = np.asarray(observation, dtype=np.float64)
    i, j = pairs
    obs_term = np.abs(y[i] - y[j]) ** p
    ens_term = np.mean(np.abs(X[:, i] - X[:, j]) ** p, axis=0)
    w = 1.0 if weights is None else np.asarray(weights)
    return float(np.sum(w * (obs_term - ens_term) ** 2))


def sample_pairs(n_points, n_pairs, rng=None, coords=None, max_dist=None):
    rng = rng or np.random.default_rng(0)
    i = rng.integers(0, n_points, n_pairs)
    j = rng.integers(0, n_points, n_pairs)
    keep = i != j
    if coords is not None and max_dist is not None:
        d = np.linalg.norm(coords[i] - coords[j], axis=1)
        keep &= d <= max_dist
    return i[keep], j[keep]
