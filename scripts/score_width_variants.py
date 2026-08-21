#!/usr/bin/env python3
"""Which class axes does the half-width rescaling actually earn its place on?

Stage D of ``docs/improvement_plan.md``. The two recalibration layers are keyed on
different axes — conformal on ``horizon x dhat x HM x biome``, width factors on
``horizon x band x dhat x HM`` out to 10 px — so unifying them is a *measurement* of which
axis carries signal, not a refactor. This scores candidate width-factor fits against each
other on pixels none of them were fitted on.

It evaluates from the residual manifest directly rather than by rewriting rasters. A width
factor multiplies the published half-widths, so for a residual ``res`` with published
half-widths ``w_lo``/``w_up`` the rescaled interval is ``[-k_lo*w_lo, +k_up*w_up]`` about
the central forecast, and both coverage and the interval score follow in closed form. One
pass over the rasters scores every variant, which is what makes a real held-out comparison
affordable.

The interval score is the primary number because coverage alone can always be bought with
width; it is reported per horizon alongside its two components so a variant that wins by
inflating cannot hide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.copula import Z975  # noqa: E402
from src.ensemble.validate import (  # noqa: E402
    DHAT_BINS, DHAT_LABELS, DIST_LABELS, HM_BINS, HM_LABELS, distance_band,
)

ALPHA = 0.05


def load_factors(path):
    """``{(horizon, band, dhat, hm): (k_up, k_lo)}`` from fit_width_factors.py output."""
    if path is None:
        return None
    d = json.load(open(path))
    by_h = d.get("by_horizon", d)
    out = {}
    for h, bands in by_h.items():
        if not isinstance(bands, dict):
            continue
        for b, dhats in bands.items():
            for dd, hms in dhats.items():
                for j, kv in hms.items():
                    out[(int(h), int(b), int(dd), int(j))] = (float(kv[0]), float(kv[1]))
    return out


_TABLE_CACHE = {}


def factor_arrays(factors, h, band, d_idx, hm_idx, tag=""):
    """Per-pixel (k_up, k_lo), defaulting to 1.0 where the fit has no cell.

    Built once as a dense ``(axis, dhat, hm)`` table and read with a single gather. The
    obvious spelling — mask the whole raster once per fitted cell — is O(cells x pixels),
    which at 6,900 cells and 63M pixels is tens of thousands of full-raster passes and turns
    a two-minute comparison into two hours.
    """
    ones = np.ones(band.shape, dtype=np.float64)
    if factors is None:
        return ones, ones.copy()
    key = (tag, h)
    if key not in _TABLE_CACHE:
        rel = [(b, d, j, ku, kl) for (hh, b, d, j), (ku, kl) in factors.items() if hh == h]
        if not rel:
            _TABLE_CACHE[key] = None
        else:
            nb = max(r[0] for r in rel) + 1
            nd = max(r[1] for r in rel) + 1
            nj = max(r[2] for r in rel) + 1
            up = np.ones((nb, nd, nj))
            lo = np.ones((nb, nd, nj))
            for b, d, j, ku, kl in rel:
                up[b, d, j] = ku
                lo[b, d, j] = kl
            _TABLE_CACHE[key] = (up, lo)
    tab = _TABLE_CACHE[key]
    if tab is None:
        return ones, ones.copy()
    up, lo = tab
    nb, nd, nj = up.shape
    inb = (band >= 0) & (band < nb) & (d_idx < nd) & (hm_idx < nj)
    bi = np.clip(band, 0, nb - 1)
    di = np.clip(d_idx, 0, nd - 1)
    ji = np.clip(hm_idx, 0, nj - 1)
    return np.where(inb, up[bi, di, ji], 1.0), np.where(inb, lo[bi, di, ji], 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fold_mask", required=True)
    ap.add_argument("--score_folds", required=True,
                    help="Folds to score on — must be disjoint from every variant's fit set")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="label=path_to_width_factors.json ; use label=identity for no rescale")
    ap.add_argument("--ecoregion_raster",
                    default="data/ensemble/region/africa/ecoregion.tif")
    ap.add_argument("--lookup_csv", default="data/raw/hm_global/ecoregion_lookup.csv")
    ap.add_argument("--clip", type=float, default=8.0)
    ap.add_argument("--min_class_n", type=int, default=10_000)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args(argv)

    variants, axis_of = {}, {}
    for spec in args.variants:
        label, _, path = spec.partition("=")
        if path in ("", "identity"):
            variants[label], axis_of[label] = None, "band"
        else:
            variants[label] = load_factors(path)
            axis_of[label] = json.load(open(path)).get("class_axis", "band")
        n = 0 if variants[label] is None else len(variants[label])
        print(f"  variant {label}: {n} fitted cells, axis '{axis_of[label]}'")

    keep = {int(x) for x in args.score_folds.split(",") if x.strip() != ""}
    with rasterio.open(args.fold_mask) as s:
        score_mask = np.isin(s.read(1), list(keep))
    print(f"  scoring on folds {sorted(keep)}: {int(score_mask.sum()):,} px")

    man = pd.read_csv(args.manifest)
    rows = []
    for _, row in man.iterrows():
        h = int(row["horizon"])
        with rasterio.open(row["path_res_native"]) as s:
            res = s.read(1).astype(np.float64)
        with rasterio.open(row["path_w_up"]) as s:
            wu = s.read(1).astype(np.float64)
        with rasterio.open(row["path_w_lo"]) as s:
            wl = s.read(1).astype(np.float64)
        with rasterio.open(row["path_dhat"]) as s:
            dhat = s.read(1).astype(np.float64)
        with rasterio.open(row["path_hm_t0"]) as s:
            hm0 = s.read(1).astype(np.float64)
        with rasterio.open(row["path_dist_past_change"]) as s:
            dist_band = distance_band(s.read(1).astype(np.float64)).astype(np.int64)
        axes = {"band": dist_band}
        if any(a != "band" for a in axis_of.values()):
            if "biome" not in axes:
                from src.ensemble.validate import biome_lut
                bmap, _, _ = biome_lut(args.lookup_csv)
                with rasterio.open(args.ecoregion_raster) as s:
                    bio = bmap[np.clip(s.read(1), 0, len(bmap) - 1)].astype(np.int64)
                axes["biome"] = bio
                axes["both"] = bio * len(DIST_LABELS) + dist_band

        ok = (score_mask & np.isfinite(res) & np.isfinite(wu) & np.isfinite(wl)
              & np.isfinite(dhat) & np.isfinite(hm0) & (wu > 0) & (wl > 0))
        if not ok.any():
            continue
        d_idx = np.digitize(dhat, DHAT_BINS[1:-1])
        hm_idx = np.digitize(hm0, HM_BINS[1:-1])

        for label, factors in variants.items():
            band = axes[axis_of[label]]
            k_up, k_lo = factor_arrays(factors, h, band, d_idx, hm_idx, tag=label)
            up = k_up * wu
            lo = k_lo * wl
            covered = (res <= up) & (res >= -lo)
            width = up + lo
            # Interval score: width plus a 2/alpha penalty on each side's miss.
            pen = (2.0 / ALPHA) * (np.maximum(0.0, -lo - res) + np.maximum(0.0, res - up))
            rows.append({
                "horizon": h, "variant": label, "n": int(ok.sum()),
                "coverage": float(covered[ok].mean()),
                "mean_width": float(width[ok].mean()),
                "interval_score": float((width + pen)[ok].mean()),
                "penalty": float(pen[ok].mean()),
            })
            # Worst class deviation, on classes with enough pixels to mean anything.
            worst = 0.0
            for b in range(len(DIST_LABELS)):
                for j in range(len(HM_LABELS)):
                    m = ok & (dist_band == b) & (hm_idx == j)
                    n = int(m.sum())
                    if n < args.min_class_n:
                        continue
                    worst = max(worst, abs(float(covered[m].mean()) - 0.95))
            rows[-1]["worst_class_dev"] = worst

    df = pd.DataFrame(rows)
    agg = df.groupby(["variant", "horizon"]).agg(
        coverage=("coverage", "mean"), mean_width=("mean_width", "mean"),
        interval_score=("interval_score", "mean"), penalty=("penalty", "mean"),
        worst_class_dev=("worst_class_dev", "max")).reset_index()
    print("\n=== held-out, by horizon (interval score is the verdict; lower is better) ===")
    print(agg.round(6).to_string(index=False))
    print("\n=== pooled over horizons ===")
    pool = agg.groupby("variant").agg(
        coverage=("coverage", "mean"), mean_width=("mean_width", "mean"),
        interval_score=("interval_score", "mean"),
        worst_class_dev=("worst_class_dev", "max")).reset_index()
    pool = pool.sort_values("interval_score")
    print(pool.round(6).to_string(index=False))
    best = pool.iloc[0]["variant"]
    print(f"\n  best held-out interval score: {best}")
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
