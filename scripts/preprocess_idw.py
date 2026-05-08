"""Precompute inverse-distance-weighted HM rasters.

For each available HM_YEAR_AA_1000.tiff and each requested kernel radius,
produces a `HM_YEAR_idw{radius}_1000.tiff` raster with the same grid /
projection as the input. Each pixel holds a per-pixel IDW-weighted average
of HM in its neighborhood:

    IDW(p) = Σ_q w(d(p, q)) * HM(q)         where w(d) = 1 / (d^k + ε)
             ─────────────────────                kernel weights are
                Σ_q w(d(p, q))                    normalized to sum to 1

Self-distance (d=0) is excluded so a pixel's own HM does not contaminate
its neighborhood signal — the IDW value answers "how surrounded is this
pixel by modified land?", not "what is this pixel's own modification?".

NaN is treated as 0 inside the convolution (interpretation: ocean / no
data contributes no modification pressure), then the original NaN mask is
restored on the output so the IDW raster has the same valid-pixel mask
as the source.

Usage:
    python scripts/preprocess_idw.py --radii 15 51 --years 1990 1995 2000 2005 2010 2015 2020
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
from scipy.signal import fftconvolve

from torchgeo_dataloader import _resolve, _make_idw_kernels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+",
                   default=[1990, 1995, 2000, 2005, 2010, 2015, 2020])
    p.add_argument("--radii", type=int, nargs="+", default=[15, 51])
    p.add_argument("--decay", type=float, default=1.0)
    p.add_argument("--out_dir", default="data/raw/hm_global",
                   help="Where to write the IDW rasters. Defaults to the same dir "
                        "as the source HM_AA files.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def compute_idw(hm: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve HM with the IDW kernel; restore NaN mask of the input."""
    nan_mask = ~np.isfinite(hm)
    hm_filled = np.where(nan_mask, 0.0, hm).astype(np.float32)
    idw = fftconvolve(hm_filled, kernel, mode="same").astype(np.float32)
    idw[nan_mask] = np.nan
    return idw


def main():
    args = parse_args()
    kernels = _make_idw_kernels(tuple(args.radii), args.decay)
    print(f"Built {len(kernels)} kernels: "
          + ", ".join(f"radius={r} (size {k.shape[0]}×{k.shape[0]})"
                      for r, k in zip(args.radii, kernels)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

        for radius, kernel in zip(args.radii, kernels):
            out_path = out_dir / f"HM_{year}_idw{radius}_1000.tiff"
            if out_path.exists() and not args.overwrite:
                print(f"  ✓ {out_path.name} already exists; skipping (use --overwrite)")
                continue
            t0 = time.time()
            print(f"  computing IDW (radius={radius}, kernel {kernel.shape[0]}×{kernel.shape[0]})...",
                  flush=True)
            idw = compute_idw(hm, kernel)
            elapsed = time.time() - t0
            print(f"    convolution done in {elapsed:.1f}s; finite={np.isfinite(idw).sum():,} / {idw.size:,}, "
                  f"min={np.nanmin(idw):.4f}, max={np.nanmax(idw):.4f}, "
                  f"mean={np.nanmean(idw):.4f}",
                  flush=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(idw, 1)
            print(f"    wrote {out_path}", flush=True)

    print(f"\nTotal preprocessing time: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
