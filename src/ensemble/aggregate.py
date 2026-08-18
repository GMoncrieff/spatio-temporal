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
ARRAY_NAME = "members"

# Read-only sessions are pinned here for the life of the process. The zarr array holds the
# session's store, but nothing holds the session, and letting it be collected pulls the
# store out from under an array that is still being read.
_OPEN_SESSIONS = []


def open_ensemble(path, branch: str = "main"):
    """``(array, attrs)`` for an ensemble stored as an icechunk repository.

    Ensembles are icechunk rather than a bare zarr directory because the write is
    transactional: two GPU workers write disjoint member blocks into a forked session and
    the snapshot only exists once both have been merged and committed. The previous
    plain-zarr store had no such boundary, so a run killed part-way — which is exactly how
    the Africa scorecard ended — left a directory that looked like a complete ensemble and
    read back as sentinel.
    """
    import icechunk
    import zarr

    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path)))
    session = repo.readonly_session(branch)
    _OPEN_SESSIONS.append(session)
    root = zarr.open_group(session.store, mode="r")
    arr = root[ARRAY_NAME]
    return arr, dict(arr.attrs)


def dequantize_block(q, attrs):
    scale = float(attrs.get("scale", 1.0 / 32767.0))
    offset = float(attrs.get("offset", 0.0))
    out = q.astype(np.float32) * scale + offset
    return np.where(q == INT16_SENTINEL, np.nan, out)


def members_at_points(store, attrs, horizon_idx, flat_idx, H, W, tile: int = 512):
    """``(M, len(flat_idx))`` member values at scattered flat pixel indices.

    The obvious spelling — ``member_slice(store, attrs, m, hi).ravel()[idx]`` per member —
    reads the entire horizon once for every member to keep ~1500 points, which at M=400 is
    the whole store pulled through memory for a handful of pixels, and the T3 sample is
    drawn twice besides. The points come from a dozen 512 px patches, so reading the tiles
    that actually contain them costs three orders of magnitude less and returns exactly the
    same values.
    """
    M = store.shape[0]
    flat_idx = np.asarray(flat_idx)
    rr, cc = np.unravel_index(flat_idx, (H, W))
    out = np.empty((M, flat_idx.size), dtype=np.float32)
    key = (rr // tile).astype(np.int64) * (W // tile + 2) + (cc // tile)
    for k in np.unique(key):
        sel = key == k
        r0 = int(rr[sel].min()) // tile * tile
        c0 = int(cc[sel].min()) // tile * tile
        rh, cwid = min(tile, H - r0), min(tile, W - c0)
        blk = dequantize_block(
            np.asarray(store[:, horizon_idx, r0:r0 + rh, c0:c0 + cwid]), attrs)
        out[:, sel] = blk[:, rr[sel] - r0, cc[sel] - c0]
        del blk
    return out


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


def _compact_zones(zone_raster, H, W, block_rows=1024):
    """``(remap, ids)`` mapping raw zone ids onto 0..n-1 over the ids actually present.

    The raster is uint16, so indexing by raw id makes every per-member table 65,536 wide
    for the ~164 ecoregions that exist. That is a 400x over-allocation at M=400 and it grows
    with the member count, which is exactly the shape of thing this pass is not allowed to
    do any more.
    """
    present = np.zeros(65536, dtype=bool)
    with rasterio.open(zone_raster) as zsrc:
        for r0 in range(0, H, block_rows):
            rr = min(block_rows, H - r0)
            zn = zsrc.read(1, window=Window(0, r0, W, rr), boundless=True, fill_value=0)
            u = np.unique(zn)
            present[u[u > 0]] = True
    ids = np.nonzero(present)[0].astype(np.int64)
    remap = np.zeros(65536, dtype=np.int64)
    remap[ids] = np.arange(ids.size)
    return remap, ids


def zonal_member_stats(zarr_store, horizon_idx, zone_raster, attrs=None, thresholds=(0.1, 0.3),
                       block_rows: int = 1024, budget_bytes=8e9, member_block: int = 10):
    """Per-member zonal means and area-above-threshold, in one streaming pass.

    Returns ``{"zone_ids", "n_px", "mean": (M, n_zones), "area{t}": (M, n_zones)}`` where
    ``n_zones`` counts only the zones present in the raster.

    Members are read a chunk-block at a time rather than one at a time: the store is chunked
    ``(10, 1, 1024, 1024)``, so a single-member read decompresses a ten-member chunk and
    throws nine tenths of it away.
    """
    store, at = (zarr_store, attrs) if attrs is not None else open_ensemble(zarr_store)
    M, _, H, W = store.shape
    remap, ids = _compact_zones(zone_raster, H, W, block_rows)
    n_zones = max(ids.size, 1)

    cnt = np.zeros(n_zones, dtype=np.int64)
    means = np.zeros((M, n_zones))
    areas = {t: np.zeros((M, n_zones)) for t in thresholds}

    # Rows per read, and members per read, both sized so the working set follows the budget
    # rather than the raster width or the member count.
    rows = max(1, min(block_rows, int(budget_bytes / max(member_block * W * 8.0, 1))))
    rows = min(rows, H)

    with rasterio.open(zone_raster) as zsrc:
        for r0 in range(0, H, rows):
            rr = min(rows, H - r0)
            zn = zsrc.read(1, window=Window(0, r0, W, rr), boundless=True, fill_value=0)
            zpos = zn > 0
            if not zpos.any():
                continue
            zc = remap[zn]
            for m0 in range(0, M, member_block):
                m1 = min(m0 + member_block, M)
                blk = dequantize_block(
                    np.asarray(store[m0:m1, horizon_idx, r0:r0 + rr]), at)
                for i, m in enumerate(range(m0, m1)):
                    v = blk[i]
                    ok = np.isfinite(v) & zpos
                    if not ok.any():
                        continue
                    zf = zc[ok]
                    if m == 0:
                        cnt += np.bincount(zf, minlength=n_zones)
                    means[m] += np.bincount(zf, weights=v[ok], minlength=n_zones)
                    for t in thresholds:
                        areas[t][m] += np.bincount(zf, weights=(v[ok] > t).astype(float),
                                                   minlength=n_zones)
                del blk

    keep = cnt > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"zone_ids": ids[keep], "n_px": cnt[keep], "mean": means[:, keep] / cnt[keep]}
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


# --------------------------------------------------------------------------------------
# Streaming block scores
# --------------------------------------------------------------------------------------
# Bytes per member-pixel held while a tile is being *read*: the int16 chunk data, the
# float32 dequantized copy, the float64 base block sums, and the finite mask.
READ_BYTES_PER_PX = 15.0
# Bytes per member-block held while a batch of blocks is being *scored*: the compacted
# means, nanpercentile's partition copy, and CRPS's sort and difference arrays.
SCORE_BYTES_PER_BLOCK = 40.0


def score_tile_shape(n_members, H, W, align, budget_bytes=8e9):
    """Tile ``(rows, cols)`` whose read working set fits ``budget_bytes``.

    Both dimensions come back as multiples of ``align`` (the coarsest block size), so a
    block never straddles a tile and no cross-tile carry-over is needed. The final row band
    and column tile are allowed to be short: at the coarse scales their blocks are
    incomplete and get dropped, which is exactly what ``n_bi = H // B`` does today, and at
    the base scale they are still whole blocks and are kept.

    The tile is kept as square as the budget allows. Store chunks are 1024 px and block
    sizes are powers of ten, so a tile can never line up with the chunk grid; what it can do
    is be large enough that the partial chunks around its edge stop mattering, and for a
    fixed area a square has the least edge.
    """
    budget_px = max(float(budget_bytes) / (n_members * READ_BYTES_PER_PX), float(align) ** 2)
    side = max(align, int(budget_px ** 0.5) // align * align)
    rows = min(H, side)
    cols = min(W, max(align, int(budget_px // max(rows, 1)) // align * align))
    return int(rows), int(cols)


def score_batch_blocks(n_members, budget_bytes=8e9):
    """How many blocks to score at once, so the tile can grow without the workspace doing so.

    The read and the arithmetic want different sizes: reads want a big tile because the
    store is chunked, while ``nanpercentile`` and CRPS allocate several copies of whatever
    they are handed. Since every statistic here is additive over blocks, the two can be
    decoupled — score the tile's blocks in batches and fold each into the accumulators.
    """
    return max(1024, int(float(budget_bytes) / (n_members * SCORE_BYTES_PER_BLOCK)))


class _ScaleScore:
    """Scalar accumulators for one aggregation scale.

    Every T2 statistic is a count or a mean over blocks, so none of them needs the blocks
    kept: coverage, the interval score and CRPS all reduce to a running sum plus a count.
    That is what makes the pass independent of raster size — the per-scale state here is
    nine numbers, whatever the grid.
    """

    __slots__ = ("B", "n", "n_covered", "n_below", "n_above", "sum_width", "sum_is",
                 "sum_is_width", "sum_is_penalty", "n_is", "sum_crps", "n_crps")

    def __init__(self, B):
        self.B = int(B)
        self.n = self.n_covered = self.n_below = self.n_above = 0
        self.sum_width = 0.0
        self.sum_is = self.sum_is_width = self.sum_is_penalty = 0.0
        self.n_is = 0
        self.sum_crps = 0.0
        self.n_crps = 0

    def add(self, member_means, observed, alpha=0.05, qs=(2.5, 97.5)):
        """Fold one batch of blocks in, reproducing the whole-array functions exactly."""
        if member_means.size == 0 or member_means.shape[1] == 0:
            return
        m = np.asarray(member_means, dtype=np.float64)
        y = np.asarray(observed, dtype=np.float64)
        # ``nanpercentile`` on a 2-D array has no C fast path: numpy falls back to
        # ``apply_along_axis``, a Python-level loop over the columns. At 63.1M blocks and
        # 400 members that is the difference between minutes and most of a day, and it buys
        # nothing here — the block means are sums of finite pixels over a non-empty count,
        # so they cannot be NaN. Asking for both quantiles in one call also partitions once
        # instead of twice. The whole-array branch is kept for the case the invariant does
        # not hold, so this is a speed path and not a change of definition.
        if np.isfinite(m).all():
            lo, hi = np.percentile(m, list(qs), axis=0)
        else:
            lo = np.nanpercentile(m, qs[0], axis=0)
            hi = np.nanpercentile(m, qs[1], axis=0)
        ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
        if ok.any():
            self.n += int(ok.sum())
            self.n_covered += int(((y >= lo) & (y <= hi) & ok).sum())
            self.n_below += int(((y < lo) & ok).sum())
            self.n_above += int(((y > hi) & ok).sum())
            self.sum_width += float((hi - lo)[ok].sum())
            width = (hi - lo)[ok]
            penalty = (2.0 / alpha) * (np.maximum(lo[ok] - y[ok], 0.0)
                                       + np.maximum(y[ok] - hi[ok], 0.0))
            self.sum_is += float((width + penalty).sum())
            self.sum_is_width += float(width.sum())
            self.sum_is_penalty += float(penalty.sum())
            self.n_is += int(ok.sum())

        # CRPS keeps its own mask: it needs every member finite, not just the two quantiles.
        okc = np.isfinite(y) & np.isfinite(m).all(axis=0)
        if okc.any():
            X, yy = m[:, okc], y[okc]
            M = X.shape[0]
            term1 = np.mean(np.abs(X - yy[None, :]), axis=0)
            Xs = np.sort(X, axis=0)
            w = (2 * np.arange(1, M + 1) - M - 1).astype(np.float64)[:, None]
            term2 = 2.0 * (w * Xs).sum(axis=0) / (2.0 * M * max(M - 1, 1))
            self.sum_crps += float((term1 - term2).sum())
            self.n_crps += int(okc.sum())

    def result(self):
        n = max(self.n, 1)
        return {
            "n": self.n, "n_covered": self.n_covered,
            "coverage": self.n_covered / n,
            "mean_width": (self.sum_width / self.n) if self.n else np.nan,
            "frac_below": self.n_below / n, "frac_above": self.n_above / n,
            "interval_score": (self.sum_is / self.n_is) if self.n_is else np.nan,
            "width_term": (self.sum_is_width / self.n_is) if self.n_is else np.nan,
            "penalty_term": (self.sum_is_penalty / self.n_is) if self.n_is else np.nan,
            "crps": (self.sum_crps / self.n_crps) if self.n_crps else np.nan,
        }


def _block_reduce(a, nbi, nbj, B, axis0_is_member):
    """Sum ``a`` over ``B x B`` blocks, dropping the incomplete right/bottom margin."""
    if axis0_is_member:
        s = a[:, : nbi * B, : nbj * B]
        return s.reshape(s.shape[0], nbi, B, nbj, B).sum(axis=(2, 4))
    s = a[: nbi * B, : nbj * B]
    return s.reshape(nbi, B, nbj, B).sum(axis=(1, 3))


def block_score_streaming(zarr_store, horizon_idx, block_sizes, observed_path,
                          reference_profile, attrs=None, mask_path=None,
                          min_valid_frac=0.5, budget_bytes=8e9, alpha=0.05,
                          qs=(2.5, 97.5)):
    """T2.1/T2.7/T2.8 statistics at several nested scales in one bounded-memory pass.

    Replaces ``block_member_stats_multi`` + ``block_observed_multi`` + the three
    ``*_from_members`` calls that followed them. Those built ``(M, n_bi, n_bj)`` float64
    arrays: at ``--block_sizes 1,10,100`` the base scale's block grid *is* the pixel grid,
    so on Africa at M=100 that is 50.5 GB for the sums and another 50.5 GB for the means —
    which is what the OOM was. Nothing here is proportional to ``M x H x W``; the tile is
    sized from ``budget_bytes`` and shrinks as the member count grows.

    The block sets, the masking rule and the arithmetic are deliberately identical to the
    functions being replaced, including the detail that the observed validity threshold is
    ``min_valid_frac`` at the base scale and ``count > 0`` at the coarser ones.
    """
    store, at = (zarr_store, attrs) if attrs is not None else open_ensemble(zarr_store)
    M, _, H, W = store.shape
    sizes = sorted(int(b) for b in set(block_sizes))
    base = sizes[0]
    for b in sizes[1:]:
        if b % base:
            raise ValueError(f"block sizes must be multiples of the smallest ({base}): {sizes}")
    bmax = sizes[-1]
    acc = {B: _ScaleScore(B) for B in sizes}
    min_px = {B: max(1, int(min_valid_frac * B * B)) for B in sizes}

    p_t = reference_profile["transform"]
    # The read arrays are still live while a batch is scored, so the two claims are
    # concurrent and the budget is split between them rather than granted twice.
    tile_h, tile_w = score_tile_shape(M, H, W, bmax, budget_bytes * 0.5)
    batch = score_batch_blocks(M, budget_bytes * 0.5)

    osrc = rasterio.open(observed_path)
    msrc = rasterio.open(mask_path) if mask_path else None
    try:
        o_t = osrc.transform
        o_off = (int(round((p_t.f - o_t.f) / o_t.e)), int(round((p_t.c - o_t.c) / o_t.a)))
        for r0 in range(0, H, tile_h):
            rr = min(tile_h, H - r0)
            for c0 in range(0, W, tile_w):
                cw = min(tile_w, W - c0)
                nbi_b, nbj_b = rr // base, cw // base
                if nbi_b == 0 or nbj_b == 0:
                    continue

                v = dequantize_block(
                    np.asarray(store[:, horizon_idx, r0:r0 + rr, c0:c0 + cw]), at)
                finite = np.isfinite(v)
                np.copyto(v, np.float32(0.0), where=~finite)
                # float32 reduce then widen, which is what block_member_stats does when it
                # adds a float32 stripe sum into its float64 accumulator. Reducing the
                # coarse scales from a float32 base instead costs ~1e-8 relative — enough
                # to miss an exact-match check against the path being replaced.
                msum = _block_reduce(v, nbi_b, nbj_b, base, True).astype(np.float64)
                mcnt = _block_reduce(finite[0], nbi_b, nbj_b, base, False).astype(np.int64)
                del v, finite

                ob = osrc.read(1, window=Window(o_off[1] + c0, o_off[0] + r0, cw, rr),
                               boundless=True, fill_value=np.nan).astype(np.float64)
                ob = np.where(ob < 0, np.nan, ob)
                ook = np.isfinite(ob)
                if msrc is not None:
                    mk = msrc.read(1, window=Window(c0, r0, cw, rr)).astype(np.float32)
                    ook &= np.isfinite(mk)
                osum = _block_reduce(np.where(ook, ob, 0.0), nbi_b, nbj_b, base, False)
                ocnt = _block_reduce(ook, nbi_b, nbj_b, base, False).astype(np.int64)
                del ob, ook

                for B in sizes:
                    f = B // base
                    nbi, nbj = nbi_b // f, nbj_b // f
                    if nbi == 0 or nbj == 0:
                        continue
                    if f == 1:
                        s, c, os_, oc = msum, mcnt, osum, ocnt
                        obs_ok = oc >= min_px[B]
                    else:
                        s = _block_reduce(msum, nbi, nbj, f, True)
                        c = _block_reduce(mcnt, nbi, nbj, f, False)
                        os_ = _block_reduce(osum, nbi, nbj, f, False)
                        oc = _block_reduce(ocnt, nbi, nbj, f, False)
                        # block_observed_multi aggregates the base counts and asks only for
                        # a non-empty block at the coarse scales; keep that rule.
                        obs_ok = oc > 0
                    ok = (c >= min_px[B]) & obs_ok
                    if not ok.any():
                        continue
                    sel = np.flatnonzero(ok.ravel())
                    s2 = s.reshape(s.shape[0], -1)
                    c2, os2, oc2 = c.ravel(), os_.ravel(), oc.ravel()
                    for k0 in range(0, sel.size, batch):
                        j = sel[k0:k0 + batch]
                        with np.errstate(invalid="ignore", divide="ignore"):
                            mm = s2[:, j].astype(np.float64) / np.maximum(c2[j], 1)
                            om = os2[j] / np.maximum(oc2[j], 1)
                        acc[B].add(mm, om, alpha=alpha, qs=qs)
    finally:
        osrc.close()
        if msrc is not None:
            msrc.close()

    return {B: acc[B].result() for B in sizes}


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
                # Members come back as one contiguous block: they are consecutive and share
                # a member-chunk, so reading them one at a time decompresses that chunk once
                # per member.
                m_lo, m_hi = members[0], members[-1] + 1
                blk = dequantize_block(
                    np.asarray(store[m_lo:m_hi, horizon_idx, r0:r0 + rr]), at)
                for m in members:
                    v = blk[m - m_lo]
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


class PairCorrAccumulator:
    """Mean pairwise member correlation (T7.2) from co-moments instead of a stacked field.

    ``member_diversity`` needs every member's whole window resident to call ``np.corrcoef``;
    on the "global" render window that is four full-resolution rasters, 11 GB on the global
    grid. A correlation is a function of six sums per pair, so the same number falls out of
    a streaming pass. Pixels are counted only where *every* member is finite, which is the
    mask ``member_diversity`` applies.
    """

    def __init__(self, n_members):
        n = int(n_members)
        self.n = 0
        self.s = np.zeros(n)
        self.ss = np.zeros(n)
        self.sxy = np.zeros((n, n))

    def add(self, X):
        X = np.asarray(X, dtype=np.float64).reshape(np.shape(X)[0], -1)
        ok = np.isfinite(X).all(axis=0)
        if not ok.any():
            return
        Xo = X[:, ok]
        self.n += Xo.shape[1]
        self.s += Xo.sum(axis=1)
        self.ss += (Xo * Xo).sum(axis=1)
        self.sxy += Xo @ Xo.T

    def result(self):
        n = self.n
        k = self.s.size
        if n < 10 or k < 2:
            return {"mean_pairwise_corr": np.nan, "min": np.nan, "max": np.nan}
        mean = self.s / n
        var = self.ss / n - mean ** 2
        cov = self.sxy / n - np.outer(mean, mean)
        sd = np.sqrt(np.maximum(var, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            C = cov / np.outer(sd, sd)
        iu = np.triu_indices(k, k=1)
        v = C[iu]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"mean_pairwise_corr": np.nan, "min": np.nan, "max": np.nan}
        return {"mean_pairwise_corr": float(v.mean()), "min": float(v.min()),
                "max": float(v.max())}


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


def energy_score(members, observation, block: int = 32):
    """Energy score (multivariate CRPS generalization); lower is better.

    The pairwise term is accumulated a few rows at a time. Written the obvious way it
    materialises ``(M, M, D)`` — 1.9 GB at M=400 on a 1500-point sample and 7.7 GB at
    M=800 — which is the one allocation in the scorecard that scales with the *square* of
    the member count and answers to no memory budget.
    """
    X = np.asarray(members, dtype=np.float64)  # (M, D)
    y = np.asarray(observation, dtype=np.float64)
    M = X.shape[0]
    term1 = np.mean(np.linalg.norm(X - y[None, :], axis=1))
    total = 0.0
    for i0 in range(0, M, max(1, block)):
        d = np.linalg.norm(X[i0:i0 + block, None, :] - X[None, :, :], axis=2)
        total += float(d.sum())
    term2 = total / (2.0 * M * M)
    return float(term1 - term2)


def variogram_score(members, observation, pairs, p: float = 0.5, weights=None,
                    pair_block: int = 20000):
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
    # ``X[:, i]`` is (M, n_pairs) and four of those are live at once inside the expression;
    # at M=400 with 200k requested pairs that is gigabytes for a scalar. Chunked over pairs
    # it is bounded by ``pair_block`` instead, and the per-pair values are unchanged.
    ens_term = np.empty(len(i), dtype=np.float64)
    for k0 in range(0, len(i), pair_block):
        k1 = min(k0 + pair_block, len(i))
        ens_term[k0:k1] = np.mean(np.abs(X[:, i[k0:k1]] - X[:, j[k0:k1]]) ** p, axis=0)
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
