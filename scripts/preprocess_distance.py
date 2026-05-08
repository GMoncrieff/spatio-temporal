"""Precompute distance-to-HM-threshold rasters.

For each available HM_YEAR_AA_1000.tiff and each requested HM threshold T,
produces a `HM_YEAR_dist{T*100:02d}_1000.tiff` raster (e.g. dist10 for
threshold 0.10, dist40 for 0.40) with the same grid / projection as the
input. Each pixel holds the Euclidean distance (in pixels ≈ km on the
1 km grid) to the nearest pixel where HM > T.

Pixels that themselves satisfy HM > T have distance 0; pixels far from
any HM > T region have large distances. Ocean / no-data pixels (NaN in
the source) are propagated as NaN in the output.

Usage:
    python scripts/preprocess_distance.py --thresholds 0.1 0.4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from torchgeo_dataloader import _resolve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+",
                   default=[1990, 1995, 2000, 2005, 2010, 2015, 2020])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.4],
                   help="HM thresholds (each yields one distance raster).")
    p.add_argument("--out_dir", default="data/raw/hm_global",
                   help="Where to write the distance rasters.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def threshold_label(threshold: float) -> str:
    """Format a threshold like 0.10 → 'dist10', 0.40 → 'dist40'."""
    return f"dist{int(round(threshold * 100)):02d}"


def compute_distance(hm: np.ndarray, threshold: float) -> np.ndarray:
    """Per-pixel distance to the nearest pixel with HM > threshold.

    `scipy.ndimage.distance_transform_edt(arr)` measures the distance from
    each True/non-zero pixel to the nearest False/zero pixel. We want the
    inverse — distance from each pixel to the nearest *feature* pixel
    (HM > threshold) — so we feed `~feature_mask`.
    """
    nan_mask = ~np.isfinite(hm)
    hm_filled = np.where(nan_mask, 0.0, hm)
    feature_mask = hm_filled > threshold
    if feature_mask.sum() == 0:
        # No features exist at this threshold: distance is undefined → NaN
        out = np.full_like(hm_filled, np.nan, dtype=np.float32)
    else:
        out = distance_transform_edt(~feature_mask).astype(np.float32)
    out[nan_mask] = np.nan
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Thresholds: {args.thresholds} → labels: "
          f"{[threshold_label(t) for t in args.thresholds]}")
    total_start = time.time()

    for year in args.years:
        src_path = _resolve(f"HM_{year}_AA_1000.tiff")
        if not os.path.exists(src_path):
            print(f"⚠ skipping {year}: source not found at {src_path}")
            continue
        print(f"\n── {year} ── source: {src_path}")
        with rasterio.open(src_path) as src:
            hm = src.read(1, masked=True).filled(np.nan).astype(np.float32)
            profile = src.profile.copy()
        profile.update(dtype="float32", compress="deflate", predictor=2, tiled=True)

        for threshold in args.thresholds:
            label = threshold_label(threshold)
            out_path = out_dir / f"HM_{year}_{label}_1000.tiff"
            if out_path.exists() and not args.overwrite:
                print(f"  ✓ {out_path.name} already exists; skipping (use --overwrite)")
                continue

            t0 = time.time()
            n_features = int(((hm > threshold) & np.isfinite(hm)).sum())
            print(f"  computing distance to HM>{threshold} "
                  f"({n_features:,} feature pixels)...", flush=True)
            dist = compute_distance(hm, threshold)
            elapsed = time.time() - t0
            finite = np.isfinite(dist)
            print(f"    distance done in {elapsed:.1f}s; finite={finite.sum():,} / {dist.size:,}, "
                  f"min={float(np.nanmin(dist)):.1f}, max={float(np.nanmax(dist)):.1f}, "
                  f"mean={float(np.nanmean(dist)):.2f} pixels", flush=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(dist, 1)
            print(f"    wrote {out_path}", flush=True)

    print(f"\nTotal preprocessing time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
