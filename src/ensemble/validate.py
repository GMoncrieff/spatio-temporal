"""Phase 1 — calibration diagnostics.

Two questions, two functions:

``compute_block_coverage`` / ``compute_zonal_coverage``
    Does the interval still cover once you *aggregate*? Independent per-pixel noise cancels
    under averaging, so naive propagation of pixel intervals collapses toward 0 coverage at
    large scales. That collapse is the motivating figure for the whole ensemble layer.

``compute_class_conditional_coverage``
    Does the interval cover *within* subpopulations? The repo's only coverage number is a
    single pooled scalar per horizon, dominated by a static majority (only 2.7% of pixels
    move more than 0.1 HM over 20 years), so it is nearly blind to the high-change pixels
    the forecast is actually used for.

Sample sizes are reported as ``n_eff`` — the number of distinct 128 px chips contributing —
not raw pixel counts. Residuals are spatially correlated far beyond 1 km, so pixel counts
overstate the information content by orders of magnitude.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

CHIP_SIZE = 128

# Bin edges fixed from the measured HM change distribution (2000->2020, 18.9M px), so that
# no bin is degenerate. The top two Delta-hat bins are small but are the point of the audit.
DHAT_BINS = [-np.inf, -0.01, 0.001, 0.01, 0.05, 0.15, np.inf]
DHAT_LABELS = ["<=-0.01", "(-0.01,0.001]", "(0.001,0.01]", "(0.01,0.05]", "(0.05,0.15]", ">0.15"]
HM_BINS = [0.0, 0.01, 0.1, 0.3, 0.6, 1.0001]
HM_LABELS = ["[0,0.01)", "[0.01,0.1)", "[0.1,0.3)", "[0.3,0.6)", "[0.6,1]"]

# Distance (px ~ km) to the nearest pixel that changed by >0.01 in the previous decade.
# Measured on southern Africa, P(future change > 0.05) runs 0.177 / 0.061 / 0.023 / 0.0078
# / 0.0024 / 0.0000 across these bands — a 70x gradient ending in an exact zero. It is
# computable from the input years alone, so it is admissible as a prediction-time stratum.
DIST_BINS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf]
DIST_LABELS = ["0-1", "1-3", "3-10", "10-30", "30-100", ">100"]


def distance_band(dist):
    """Band index of a distance-to-past-change raster. The one definition; use it.

    ``right=True`` is load-bearing rather than cosmetic. The distance comes from an exact
    Euclidean transform, so ``dist == 1.0`` and ``dist == 3.0`` are not measure-zero events
    but two of the most populated values on the raster, and the convention decides which
    side of a band edge they fall on. Scored the other way the 0-1 px band's observed
    P(change > 0.05) reads 0.222 rather than 0.177 — a 26% difference in the number the
    marginal is being fitted to reproduce. Half the call sites in this repository used each
    convention, so the primary per-member judge and the T8 clustering stage were scoring
    different pixels as "near".
    """
    return np.digitize(np.asarray(dist), DIST_BINS[1:-1], right=True).astype(np.int8)


N_MARGINAL_CLASS = len(HM_LABELS) * len(DIST_LABELS)


def marginal_class(dist, hm0):
    """Joint (HM level x distance band) index, for a marginal conditioned on both.

    The two axes fail differently and cannot substitute for each other. Distance decides
    whether a tail is representable at all — beyond 100 px it is not, and the bound has to
    come back down or remote stable country stops being exactly zero (T8.2/T8.3). HM level
    decides how much change the land can absorb: near-pristine ground at 3-10 px produces
    2.8x too much change under a bound tuned on distance alone, and 1.07x under one tuned on
    the pair. Applying either bound across the other axis overshoots — at 10-30 px the same
    correction takes pristine land to 0.32x.

    Flattened to one integer so the existing gather path is unchanged: ``stack_shapes`` and
    the GPU row-gather are already generic over the number of classes, so widening from 6 to
    30 costs nothing but the index.
    """
    d = distance_band(dist).astype(np.int16)
    # NaN digitizes to len(bins); clip so a nodata pixel lands in a real class rather than
    # indexing past the end of the stacked grids, which on the GPU is a device-side assert.
    h = np.clip(np.digitize(np.asarray(hm0), HM_BINS[1:-1]), 0, len(HM_LABELS) - 1)
    return (h.astype(np.int16) * len(DIST_LABELS) + d).astype(np.int16)


# --------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------
def wilson_interval(k, n, z: float = 1.96):
    """Wilson score interval for a binomial rate. Vectorized; NaN when n == 0."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1.0 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        lo = np.where(n > 0, centre - half, np.nan)
        hi = np.where(n > 0, centre + half, np.nan)
    return lo, hi


def _stripe_rows(block, n_block_cols, stripe_blocks=8, target_rows=1024, budget_bytes=4e8):
    """Rows to read per pass: a whole number of blocks, big enough to amortize IO and
    small enough to stay inside a memory budget.

    Both ends matter at the global grid: with ``block=1`` a naive ``stripe_blocks * block``
    reads 8 rows at a time (2000+ passes over a 40000-wide raster), while with
    ``block=1000`` it would pull 8000 x 40000 pixels into memory at once.
    """
    B = int(block)
    max_rows = max(B, int(budget_bytes / max(n_block_cols * B * 8, 1)) // B * B)
    target = max(int(stripe_blocks) * B, int(target_rows))
    return max(B, min(max_rows, max(B, target // B * B)))


def biome_lut(lookup_csv, max_eco_id: int | None = None):
    """Array mapping ECO_ID -> BIOME_NUM (0 = unknown/ocean), plus the realm equivalent."""
    lut = pd.read_csv(lookup_csv)
    n = int(max_eco_id or lut["ECO_ID"].max()) + 1
    biome = np.zeros(n, dtype=np.int16)
    realm_codes = {r: i + 1 for i, r in enumerate(sorted(lut["REALM"].dropna().unique()))}
    realm = np.zeros(n, dtype=np.int16)
    for _, row in lut.iterrows():
        i = int(row["ECO_ID"])
        if 0 < i < n:
            biome[i] = int(row["BIOME_NUM"]) if np.isfinite(row["BIOME_NUM"]) else 0
            realm[i] = realm_codes.get(row["REALM"], 0)
    return biome, realm, realm_codes


# --------------------------------------------------------------------------------------
# 1b. Coverage vs aggregation scale
# --------------------------------------------------------------------------------------
def compute_block_coverage(
    pred_lower_path,
    pred_upper_path,
    observed_path,
    block_sizes=(1, 10, 100, 1000),
    min_valid_frac: float = 0.5,
    mask_path=None,
    stripe_blocks: int = 8,
    pred_central_path=None,
):
    """Coverage of block-mean HM at several aggregation scales, two ways.

    ``kind="mean-of-bounds"`` — the block mean of the published lower/upper rasters. This
    is *perfectly dependent* propagation: it keeps the full pixel-scale width while the
    observation's error averages down, so it can only over-cover as blocks grow.

    ``kind="independent"`` — the same pixels propagated as if the per-pixel errors were
    independent, ``half_width = sqrt(sum(hw_i^2)) / n``. This is the baseline that
    collapses toward zero coverage at large scales, and the contrast between the two is
    the motivating figure for the ensemble: the truth is neither, because real errors are
    spatially correlated but not perfectly so.

    Block size is in pixels (~1 km each).
    """
    rows = []
    with rasterio.open(pred_lower_path) as lo_src:
        H, W = lo_src.height, lo_src.width
        lo_transform = lo_src.transform

    for B in block_sizes:
        n_bi, n_bj = H // B, W // B
        if n_bi == 0 or n_bj == 0:
            continue
        sum_lo = np.zeros((n_bi, n_bj), dtype=np.float64)
        sum_hi = np.zeros((n_bi, n_bj), dtype=np.float64)
        sum_ob = np.zeros((n_bi, n_bj), dtype=np.float64)
        sum_cen = np.zeros((n_bi, n_bj), dtype=np.float64)
        sum_hw2 = np.zeros((n_bi, n_bj), dtype=np.float64)
        cnt = np.zeros((n_bi, n_bj), dtype=np.int64)

        stripe = _stripe_rows(B, n_bj, stripe_blocks)
        srcs = {
            "lo": rasterio.open(pred_lower_path),
            "hi": rasterio.open(pred_upper_path),
            "ob": rasterio.open(observed_path),
        }
        cen_src = rasterio.open(pred_central_path) if pred_central_path else None
        # Observed rasters are global; predictions may be a sub-window.
        ob_t = srcs["ob"].transform
        col_off = int(round((lo_transform.c - ob_t.c) / ob_t.a))
        row_off = int(round((lo_transform.f - ob_t.f) / ob_t.e))
        try:
            for r0 in range(0, n_bi * B, stripe):
                rr = min(stripe, n_bi * B - r0)
                win = Window(0, r0, n_bj * B, rr)
                lo = srcs["lo"].read(1, window=win).astype(np.float64)
                hi = srcs["hi"].read(1, window=win).astype(np.float64)
                ob = srcs["ob"].read(
                    1, window=Window(col_off, row_off + r0, n_bj * B, rr),
                    boundless=True, fill_value=np.nan,
                ).astype(np.float64)
                ob = np.where(ob < 0, np.nan, ob)
                valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(ob)
                if mask_path is not None:
                    with rasterio.open(mask_path) as msrc:
                        mm = msrc.read(1, window=Window(col_off, row_off + r0, n_bj * B, rr),
                                       boundless=True, fill_value=0)
                    valid &= mm > 0

                def blocksum(a):
                    a = np.where(valid, a, 0.0)
                    return a.reshape(rr // B, B, n_bj, B).sum(axis=(1, 3))

                b0 = r0 // B
                sum_lo[b0:b0 + rr // B] += blocksum(lo)
                sum_hi[b0:b0 + rr // B] += blocksum(hi)
                sum_ob[b0:b0 + rr // B] += blocksum(ob)
                cen = 0.5 * (lo + hi) if cen_src is None else cen_src.read(1, window=win).astype(np.float64)
                sum_cen[b0:b0 + rr // B] += blocksum(cen)
                sum_hw2[b0:b0 + rr // B] += blocksum((0.5 * (hi - lo)) ** 2)
                cnt[b0:b0 + rr // B] += valid.reshape(rr // B, B, n_bj, B).sum(axis=(1, 3))
        finally:
            for s in srcs.values():
                s.close()
            if cen_src is not None:
                cen_src.close()

        ok = cnt >= max(1, int(min_valid_frac * B * B))
        n = int(ok.sum())
        if n == 0:
            rows.append({"scale_px": B, "n_blocks": 0, "coverage": np.nan})
            continue
        with np.errstate(invalid="ignore"):
            m_lo = sum_lo[ok] / cnt[ok]
            m_hi = sum_hi[ok] / cnt[ok]
            m_ob = sum_ob[ok] / cnt[ok]
            m_cen = sum_cen[ok] / cnt[ok]
            # Independent propagation: sd of the block mean of n independent errors.
            hw_ind = np.sqrt(sum_hw2[ok]) / cnt[ok]

        for kind, lo_b, hi_b in (
            ("mean-of-bounds", m_lo, m_hi),
            ("independent", m_cen - hw_ind, m_cen + hw_ind),
        ):
            covered = (m_ob >= lo_b) & (m_ob <= hi_b)
            k = int(covered.sum())
            wlo, whi = wilson_interval(k, n)
            rows.append({
                "scale_px": B,
                "scale_km": B,
                "kind": kind,
                "n_blocks": n,
                "n_covered": k,
                "coverage": k / n,
                "wilson_lo": float(wlo),
                "wilson_hi": float(whi),
                "mean_width": float(np.mean(hi_b - lo_b)),
                "frac_below_lower": float(np.mean(m_ob < lo_b)),
                "frac_above_upper": float(np.mean(m_ob > hi_b)),
            })
    return pd.DataFrame(rows)


def compute_zonal_coverage(
    pred_lower_path,
    pred_upper_path,
    observed_path,
    zone_raster_path,
    lookup_csv=None,
    thresholds=(0.1, 0.3),
    min_valid_px: int = 100,
    block_rows: int = 1024,
):
    """Zonal (ecoregion) coverage of the mean and of area-above-threshold.

    One streaming pass with ``np.bincount`` over the ECO_ID raster — far cheaper than 846
    separate ``rasterio.mask`` calls at this grid size. Biome/realm roll-ups are a join on
    the lookup, guaranteeing they reconcile with the ecoregion numbers pixel for pixel.
    """
    with rasterio.open(zone_raster_path) as zsrc:
        max_zone = 65535 if zsrc.dtypes[0] == "uint16" else int(zsrc.read(1, out_shape=(1, 1)).max())
        z_transform = zsrc.transform
    n_zones = max_zone + 1

    acc = {
        "cnt": np.zeros(n_zones, dtype=np.int64),
        "sum_lo": np.zeros(n_zones), "sum_hi": np.zeros(n_zones), "sum_ob": np.zeros(n_zones),
    }
    for t in thresholds:
        for k in ("lo", "hi", "ob"):
            acc[f"thr{t}_{k}"] = np.zeros(n_zones)

    srcs = {
        "lo": rasterio.open(pred_lower_path),
        "hi": rasterio.open(pred_upper_path),
        "ob": rasterio.open(observed_path),
        "zone": rasterio.open(zone_raster_path),
    }
    H, W = srcs["lo"].height, srcs["lo"].width
    p_t = srcs["lo"].transform
    ob_t, z_t = srcs["ob"].transform, srcs["zone"].transform
    ob_off = (int(round((p_t.f - ob_t.f) / ob_t.e)), int(round((p_t.c - ob_t.c) / ob_t.a)))
    z_off = (int(round((p_t.f - z_t.f) / z_t.e)), int(round((p_t.c - z_t.c) / z_t.a)))
    try:
        for r0 in range(0, H, block_rows):
            rr = min(block_rows, H - r0)
            lo = srcs["lo"].read(1, window=Window(0, r0, W, rr)).astype(np.float64)
            hi = srcs["hi"].read(1, window=Window(0, r0, W, rr)).astype(np.float64)
            ob = srcs["ob"].read(1, window=Window(ob_off[1], ob_off[0] + r0, W, rr),
                                 boundless=True, fill_value=np.nan).astype(np.float64)
            zn = srcs["zone"].read(1, window=Window(z_off[1], z_off[0] + r0, W, rr),
                                   boundless=True, fill_value=0)
            ob = np.where(ob < 0, np.nan, ob)
            valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(ob) & (zn > 0)
            if not valid.any():
                continue
            zf = zn[valid].astype(np.int64)
            acc["cnt"] += np.bincount(zf, minlength=n_zones)
            acc["sum_lo"] += np.bincount(zf, weights=lo[valid], minlength=n_zones)
            acc["sum_hi"] += np.bincount(zf, weights=hi[valid], minlength=n_zones)
            acc["sum_ob"] += np.bincount(zf, weights=ob[valid], minlength=n_zones)
            for t in thresholds:
                # Area above threshold: for the interval bound, the *lower* bound on area
                # comes from the lower raster and vice versa (monotone functional).
                acc[f"thr{t}_lo"] += np.bincount(zf, weights=(lo[valid] > t).astype(float), minlength=n_zones)
                acc[f"thr{t}_hi"] += np.bincount(zf, weights=(hi[valid] > t).astype(float), minlength=n_zones)
                acc[f"thr{t}_ob"] += np.bincount(zf, weights=(ob[valid] > t).astype(float), minlength=n_zones)
    finally:
        for s in srcs.values():
            s.close()

    keep = acc["cnt"] >= min_valid_px
    zone_ids = np.nonzero(keep)[0]
    df = pd.DataFrame({"zone_id": zone_ids, "n_px": acc["cnt"][keep]})
    for k in ("lo", "hi", "ob"):
        df[f"mean_{k}"] = acc[f"sum_{k}"][keep] / acc["cnt"][keep]
    df["covered_mean"] = (df["mean_ob"] >= df["mean_lo"]) & (df["mean_ob"] <= df["mean_hi"])
    for t in thresholds:
        for k in ("lo", "hi", "ob"):
            df[f"area{t}_{k}"] = acc[f"thr{t}_{k}"][keep] / acc["cnt"][keep]
        df[f"covered_area{t}"] = (
            (df[f"area{t}_ob"] >= df[f"area{t}_lo"]) & (df[f"area{t}_ob"] <= df[f"area{t}_hi"])
        )

    if lookup_csv is not None:
        lut = pd.read_csv(lookup_csv)[["ECO_ID", "BIOME_NUM", "BIOME_NAME", "REALM"]]
        df = df.merge(lut, left_on="zone_id", right_on="ECO_ID", how="left")
    return df


def rollup_zonal(df: pd.DataFrame, group: str, thresholds=(0.1, 0.3)):
    """Aggregate ecoregion rows into coarser polygons (biome / realm).

    Re-derives the coarse polygon's *pixel* means from the ecoregion sums, rather than
    averaging ecoregion coverages — a biome's coverage is a property of the biome-mean, and
    both must come from exactly the same pixels or the numbers will not reconcile.
    """
    if df.empty or group not in df:
        return pd.DataFrame()
    work = df.copy()
    cols = ["mean_lo", "mean_hi", "mean_ob"] + [
        f"area{t}_{k}" for t in thresholds for k in ("lo", "hi", "ob") if f"area{t}_{k}" in work
    ]
    for c in cols:
        work[f"_sum_{c}"] = work[c] * work["n_px"]
    agg = work.groupby(group, dropna=True).agg(
        n_px=("n_px", "sum"), n_zones=("zone_id", "count"),
        **{f"_sum_{c}": (f"_sum_{c}", "sum") for c in cols},
    ).reset_index()
    for c in cols:
        agg[c] = agg[f"_sum_{c}"] / agg["n_px"]
        agg.drop(columns=[f"_sum_{c}"], inplace=True)
    agg["covered_mean"] = (agg["mean_ob"] >= agg["mean_lo"]) & (agg["mean_ob"] <= agg["mean_hi"])
    for t in thresholds:
        if f"area{t}_ob" in agg:
            agg[f"covered_area{t}"] = (
                (agg[f"area{t}_ob"] >= agg[f"area{t}_lo"]) & (agg[f"area{t}_ob"] <= agg[f"area{t}_hi"])
            )
    return agg


def summarize_zonal(df, group=None, cov_col="covered_mean"):
    """Coverage point estimate + Wilson CI, optionally grouped (biome / realm)."""
    def _one(sub):
        n, k = len(sub), int(sub[cov_col].sum())
        lo, hi = wilson_interval(k, n)
        return pd.Series({"n_zones": n, "n_covered": k, "coverage": k / n if n else np.nan,
                          "wilson_lo": float(lo), "wilson_hi": float(hi)})

    if group is None:
        return _one(df).to_frame().T
    return df.groupby(group).apply(_one, include_groups=False).reset_index()


# --------------------------------------------------------------------------------------
# 1d. Class-conditional coverage audit
# --------------------------------------------------------------------------------------
@dataclass
class _Cell:
    n: int = 0
    covered: int = 0
    below: int = 0
    above: int = 0
    chips: set = field(default_factory=set)


def compute_class_conditional_coverage(
    manifest,
    ecoregion_raster=None,
    lookup_csv=None,
    block_rows: int = 1024,
    alpha: float = 0.05,
    scale_factors=None,
):
    """Coverage per (horizon, Delta-hat bin, HM-level bin, biome) cell.

    Two miscoverage counters are kept, not one: with a right-skewed change distribution the
    upper tail is expected to fail far harder than the lower, and a single coverage number
    would hide which side is broken.

    ``scale_factors`` optionally applies a recalibration (a callable taking
    ``(horizon, dhat_bin_idx, hm_bin_idx, biome)`` arrays and returning ``(s_up, s_lo)``)
    so the same audit can score recalibrated intervals without rewriting rasters.
    """
    df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
    biome_map = None
    if ecoregion_raster is not None and lookup_csv is not None:
        biome_map, _, _ = biome_lut(lookup_csv)

    cells = defaultdict(_Cell)
    for _, row in df.iterrows():
        horizon = int(row["horizon"])
        srcs = {
            "res": rasterio.open(row["path_res_native"]),
            "dhat": rasterio.open(row["path_dhat"]),
            "hm0": rasterio.open(row["path_hm_t0"]),
            "w_up": rasterio.open(row["path_w_up"]),
            "w_lo": rasterio.open(row["path_w_lo"]),
        }
        eco_src = rasterio.open(ecoregion_raster) if biome_map is not None else None
        dist_col = row.get("path_dist_past_change")
        dist_src = (rasterio.open(dist_col)
                    if isinstance(dist_col, str) and Path(dist_col).exists() else None)
        H, W = srcs["res"].height, srcs["res"].width
        p_t = srcs["res"].transform
        if eco_src is not None:
            e_t = eco_src.transform
            e_off = (int(round((p_t.f - e_t.f) / e_t.e)), int(round((p_t.c - e_t.c) / e_t.a)))
        # Absolute pixel offset of this raster on the global grid, for chip ids that are
        # comparable across windows.
        glob_row = int(round((p_t.f - 84.0) / p_t.e))
        glob_col = int(round((p_t.c + 180.0) / p_t.a))
        try:
            for r0 in range(0, H, block_rows):
                rr = min(block_rows, H - r0)
                win = Window(0, r0, W, rr)
                res = srcs["res"].read(1, window=win).astype(np.float64)
                dhat = srcs["dhat"].read(1, window=win).astype(np.float64)
                hm0 = srcs["hm0"].read(1, window=win).astype(np.float64)
                w_up = srcs["w_up"].read(1, window=win).astype(np.float64)
                w_lo = srcs["w_lo"].read(1, window=win).astype(np.float64)
                ok = np.isfinite(res) & np.isfinite(dhat) & np.isfinite(hm0) & \
                    np.isfinite(w_up) & np.isfinite(w_lo)
                if not ok.any():
                    continue
                if dist_src is not None:
                    dd = dist_src.read(1, window=Window(0, r0, W, rr)).astype(np.float64)
                    biome = distance_band(dd).astype(np.int16)
                elif eco_src is not None:
                    eco = eco_src.read(1, window=Window(e_off[1], e_off[0] + r0, W, rr),
                                       boundless=True, fill_value=0)
                    biome = biome_map[np.clip(eco, 0, len(biome_map) - 1)]
                else:
                    biome = np.zeros((rr, W), dtype=np.int16)

                rows_idx, cols_idx = np.nonzero(ok)
                chip_id = (((glob_row + r0 + rows_idx) // CHIP_SIZE).astype(np.int64) * 400000
                           + ((glob_col + cols_idx) // CHIP_SIZE))

                r = res[ok]
                wu = np.maximum(w_up[ok], 1e-9)
                wl = np.maximum(w_lo[ok], 1e-9)
                d_idx = np.digitize(dhat[ok], DHAT_BINS[1:-1])
                h_idx = np.digitize(hm0[ok], HM_BINS[1:-1])
                b = biome[ok]

                s_up = np.ones_like(r)
                s_lo = np.ones_like(r)
                if scale_factors is not None:
                    s_up, s_lo = scale_factors(horizon, d_idx, h_idx, b)

                above = r > s_up * wu
                below = -r > s_lo * wl
                covered = ~(above | below)

                # Group identical keys to avoid a Python loop over every pixel. The three
                # components are packed into one int64 first: np.unique(axis=0) lexsorts a
                # 2-D array, which is an order of magnitude slower at 40M pixels a block.
                key_arr = (d_idx.astype(np.int64) * 100000
                           + h_idx.astype(np.int64) * 10000
                           + b.astype(np.int64))
                uniq, inv = np.unique(key_arr, return_inverse=True)
                for u_i, packed in enumerate(uniq):
                    sel = inv == u_i
                    key = (int(packed) // 100000, (int(packed) // 10000) % 10, int(packed) % 10000)
                    cell = cells[(horizon, int(key[0]), int(key[1]), int(key[2]))]
                    cell.n += int(sel.sum())
                    cell.covered += int(covered[sel].sum())
                    cell.above += int(above[sel].sum())
                    cell.below += int(below[sel].sum())
                    cell.chips.update(np.unique(chip_id[sel]).tolist())
        finally:
            for s in srcs.values():
                s.close()
            if eco_src is not None:
                eco_src.close()
            if dist_src is not None:
                dist_src.close()

    rows = []
    for (horizon, d_i, h_i, b), c in cells.items():
        rows.append({
            "horizon": horizon,
            "dhat_bin": DHAT_LABELS[d_i], "dhat_bin_idx": d_i,
            "hm_bin": HM_LABELS[h_i], "hm_bin_idx": h_i,
            "biome": b,
            "n_px": c.n, "n_eff": len(c.chips),
            "n_covered": c.covered, "n_below_lower": c.below, "n_above_upper": c.above,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["coverage"] = out["n_covered"] / out["n_px"]
    out["frac_below"] = out["n_below_lower"] / out["n_px"]
    out["frac_above"] = out["n_above_upper"] / out["n_px"]
    lo, hi = wilson_interval(out["coverage"] * out["n_eff"], out["n_eff"])
    out["wilson_lo"], out["wilson_hi"] = lo, hi
    return out.sort_values(["horizon", "dhat_bin_idx", "hm_bin_idx", "biome"]).reset_index(drop=True)


def rollup_coverage(audit: pd.DataFrame, by=("horizon", "dhat_bin")):
    """Marginalize the finest audit cells up to a coarser stratification."""
    if audit.empty:
        return audit
    by = list(by)
    order = audit.groupby(by, observed=True)["dhat_bin_idx"].min() if "dhat_bin" in by else None
    g = audit.groupby(by, observed=True).agg(
        n_px=("n_px", "sum"),
        n_covered=("n_covered", "sum"),
        n_below_lower=("n_below_lower", "sum"),
        n_above_upper=("n_above_upper", "sum"),
        n_eff=("n_eff", "sum"),
    ).reset_index()
    g["coverage"] = g["n_covered"] / g["n_px"]
    g["frac_below"] = g["n_below_lower"] / g["n_px"]
    g["frac_above"] = g["n_above_upper"] / g["n_px"]
    lo, hi = wilson_interval(g["coverage"] * g["n_eff"], g["n_eff"])
    g["wilson_lo"], g["wilson_hi"] = lo, hi
    if order is not None:
        g = g.merge(order.rename("sort_idx"), on=by, how="left").sort_values(["horizon", "sort_idx"])
    return g.reset_index(drop=True)


def plot_coverage_heatmap(audit: pd.DataFrame, out_path, target: float = 0.95):
    """Horizon x Delta-hat-bin coverage heatmap annotated with n_eff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = rollup_coverage(audit, by=["horizon", "dhat_bin", "dhat_bin_idx"])
    horizons = sorted(g["horizon"].unique())
    labels = [l for l in DHAT_LABELS if l in set(g["dhat_bin"])]
    M = np.full((len(horizons), len(labels)), np.nan)
    N = np.zeros_like(M)
    for _, r in g.iterrows():
        i = horizons.index(r["horizon"])
        j = labels.index(r["dhat_bin"])
        M[i, j] = r["coverage"]
        N[i, j] = r["n_eff"]

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 1.1 * len(horizons) + 2.5))
    im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(horizons)), [f"+{h}yr" for h in horizons])
    for i in range(len(horizons)):
        for j in range(len(labels)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}\nn_eff={int(N[i, j])}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Class-conditional coverage of the 95% interval (target {target:.2f})")
    ax.set_xlabel("predicted change $\\hat\\Delta$ = central − HM$_{t_0}$")
    fig.colorbar(im, ax=ax, label="coverage")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_coverage_vs_scale(block_df, out_path, title="Coverage vs aggregation scale"):
    """One line per (propagation kind, horizon), so the two baselines are contrasted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = {"mean-of-bounds": dict(marker="o", ls="-"),
              "independent": dict(marker="s", ls="--"),
              "ensemble": dict(marker="^", ls="-")}
    if "kind" in block_df:
        groups = block_df.groupby(["kind", "horizon"] if "horizon" in block_df else ["kind"])
    elif "label" in block_df:
        groups = block_df.groupby("label")
    else:
        groups = [("", block_df)]
    for key, sub in groups:
        kind = key[0] if isinstance(key, tuple) else key
        sub = sub.groupby("scale_km", as_index=False)[["coverage", "wilson_lo", "wilson_hi"]].mean()
        # Averaging the Wilson bounds over horizons can put a bound fractionally the wrong
        # side of the averaged point estimate — most visibly at coverage 1.000, where the
        # interval is one-sided. Clip rather than let matplotlib reject the whole figure.
        lo_err = np.clip(sub["coverage"] - sub["wilson_lo"], 0.0, None)
        hi_err = np.clip(sub["wilson_hi"] - sub["coverage"], 0.0, None)
        ax.errorbar(
            sub["scale_km"], sub["coverage"], yerr=[lo_err, hi_err],
            capsize=3, label=str(key), **styles.get(str(kind), {"marker": "o"}),
        )
    ax.axhline(0.95, ls="--", c="k", lw=1, label="nominal 0.95")
    ax.set_xscale("log")
    ax.set_xlabel("block size (km)")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
