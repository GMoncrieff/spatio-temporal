#!/usr/bin/env python3
"""Residuals for a single-model regional hindcast (no fold-CV stitching).

Used when iterating on the quantile heads: the heads are retrained on the region's *train*
split, so residuals on those pixels are in-sample and would flatter the calibration. Pixels
the heads trained on are therefore masked out, leaving the val/test/calib geography — the
same honesty the k=5 fold rotation buys for the global run, at regional speed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.residuals import HORIZONS, RankGaussianTransform, append_manifest, compute_residuals

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"
WINDOWS = [(1990, 1995, 2000), (1995, 2000, 2005), (2000, 2005, 2010), (2005, 2010, 2015)]
MAX_YEAR = 2020


def mask_training_pixels(paths, split_mask_path, reference_profile, keep_values=(2, 3, 4)):
    """NaN out pixels used to train the heads, in place."""
    p_t = reference_profile["transform"]
    H, W = reference_profile["height"], reference_profile["width"]
    with rasterio.open(split_mask_path) as s:
        s_t = s.transform
        off = (int(round((p_t.f - s_t.f) / s_t.e)), int(round((p_t.c - s_t.c) / s_t.a)))
        split = s.read(1, window=Window(off[1], off[0], W, H), boundless=True, fill_value=0)
    keep = np.isin(split, keep_values)
    n_dropped = 0
    for p in paths:
        with rasterio.open(p) as src:
            arr = src.read(1)
            profile = src.profile.copy()
        n_dropped = int((~keep & np.isfinite(arr)).sum())
        arr = np.where(keep, arr, np.nan).astype(np.float32)
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(arr, 1)
    return keep.sum(), n_dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", default="data/ensemble/region/southern_africa/newheads")
    ap.add_argument("--out_dir", default="data/ensemble/region/southern_africa/newheads_residuals")
    ap.add_argument("--covariate_dir", default="data/ensemble/region/southern_africa/covariates")
    ap.add_argument("--split_mask", default="data/raw/hm_global/split_mask_region_1000.tif")
    ap.add_argument("--transform_json", default="data/ensemble/rank_gaussian.json")
    ap.add_argument("--keep_splits", default="2,3,4",
                    help="Split values to evaluate on (default: everything except train). "
                         "Use 'all' when the rasters are already restricted to held-out "
                         "geography, as fold-CV predictions are.")
    ap.add_argument("--pred_suffix", default="_blended",
                    help="Filename suffix before .tif; fold-stitched rasters have none")
    args = ap.parse_args(argv)

    pred = Path(args.pred_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    if manifest.exists():
        manifest.unlink()

    tr = (RankGaussianTransform.from_json(args.transform_json)
          if Path(args.transform_json).exists() else None)
    keep_values = None if args.keep_splits.strip().lower() == "all" \
        else tuple(int(v) for v in args.keep_splits.split(","))

    for window in WINDOWS:
        base = window[-1]
        for h in HORIZONS:
            year = base + h
            if year > MAX_YEAR:
                continue
            trio = {q: pred / f"w{base}_prediction_{year}_{q}{args.pred_suffix}.tif"
                    for q in ("central", "lower", "upper")}
            if not all(p.exists() for p in trio.values()):
                continue
            tag = f"w{base}_h{h}"
            info = compute_residuals(
                observed_path=str(HM_DIR / f"HM_{year}_AA_1000.tiff"),
                central_path=str(trio["central"]), lower_path=str(trio["lower"]),
                upper_path=str(trio["upper"]),
                baseline_hm_path=str(HM_DIR / f"HM_{base}_AA_1000.tiff"),
                out_dir=str(out), tag=tag, transform=tr,
            )
            with rasterio.open(trio["central"]) as c:
                profile = c.profile.copy()
            if keep_values is None:
                dropped = 0
            else:
                kept, dropped = mask_training_pixels(
                    [v for k, v in info.items() if k.startswith("path_")],
                    args.split_mask, profile, keep_values=keep_values)
            ctx = Path(args.covariate_dir) / f"w{base}_dist_past_change.tif"
            row = {"window": "-".join(str(y) for y in window), "base_year": base,
                   "target_year": year, "horizon": h,
                   "path_central": str(trio["central"]), "path_lower": str(trio["lower"]),
                   "path_upper": str(trio["upper"]),
                   "path_observed": str(HM_DIR / f"HM_{year}_AA_1000.tiff"),
                   "path_dist_past_change": str(ctx) if ctx.exists() else "",
                   **info}
            append_manifest(manifest, row)
            print(f"  {tag}: {info['n_valid_px']:,} residual px, {dropped:,} training px dropped")

    print(f"\n✓ manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
