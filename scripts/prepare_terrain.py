#!/usr/bin/env python3
"""Slope and aspect from the project's own elevation raster, as three static covariates.

    python scripts/prepare_terrain.py            # writes the three rasters beside the DEM

Writes, beside ``hm_static_ele_1000.tiff`` and on its grid:

    hm_static_terrain_slope_1000.tiff    slope, degrees
    hm_static_terrain_aspsin_1000.tiff   sin(aspect)   aspect = downslope direction,
    hm_static_terrain_aspcos_1000.tiff   cos(aspect)   clockwise from north

**Why not the ``hm_static_ele_slope/asp_*`` rasters already in the directory.** They are not
derived from this DEM: against Horn slope computed here they correlate at only 0.51-0.89 and
run 0.9-1.6x steeper (median ratio, four test windows), consistent with a finer source DEM
aggregated to 1 km. Nothing records where they came from.

**Method.** Horn (1981) 3x3 finite differences. The grid is geographic (0.009 deg), so a pixel
is ~1.00 km wide at the equator and ~0.50 km at 60 deg; the gradient uses each row's true
metre spacing on the WGS84 ellipsoid (east: N(phi) cos(phi) dlambda, north: M(phi) dphi),
checked against pyproj's geodesic in tests/test_prepare_terrain.py. A slope taken in pixels
would halve every high-latitude slope.

**Aspect** is circular, so it is written as sin and cos. On flat ground it is undefined and
both are 0 -- the ocean, which this DEM fills with 0 m, is flat.

**Nodata.** The DEM declares none and fills every row south of 56 S with -32768
(``torchgeo_dataloader.STATIC_NODATA``). That fill is nodata here, and so is every pixel whose
3x3 window touches it; all three outputs declare NaN as nodata. The model's reader fills NaN
with 0 for these channels before standardising, exactly as it does for elevation.

**Edges.** The grid is exactly 360 deg wide, so longitude wraps: column 0's west neighbour
is the last column. The first and last rows replicate themselves (the grid stops at 84 N and
70 S, the south edge inside the fill anyway). Rows are streamed in blocks with a one-row halo;
the blocking is an identity (tested).
"""
import argparse
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.windows import Window

TERRAIN_NAMES = {
    "slope": "hm_static_terrain_slope_1000.tiff",
    "aspsin": "hm_static_terrain_aspsin_1000.tiff",
    "aspcos": "hm_static_terrain_aspcos_1000.tiff",
}
DESCRIPTIONS = {
    "slope": "slope, degrees (Horn 3x3, WGS84 metric spacing per row)",
    "aspsin": "sin(aspect); aspect = downslope direction clockwise from north; 0 on flat",
    "aspcos": "cos(aspect); aspect = downslope direction clockwise from north; 0 on flat",
}

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def pixel_spacing_m(lats_deg, xres_deg, yres_deg):
    """East-west and north-south size (m) of a pixel centred at each latitude, WGS84."""
    phi = np.radians(np.asarray(lats_deg, dtype=np.float64))
    w = 1.0 - _E2 * np.sin(phi) ** 2
    n = _A / np.sqrt(w)                      # prime-vertical radius of curvature
    m = _A * (1.0 - _E2) / w ** 1.5          # meridional radius of curvature
    return n * np.cos(phi) * np.radians(abs(xres_deg)), m * np.radians(abs(yres_deg))


def _horn(zp, lats, xres, yres):
    """Slope and aspect for the interior of ``zp``, which carries a one-pixel halo."""
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dx, dy = pixel_spacing_m(lats, xres, yres)
    gx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx[:, None])   # dz / d(east)
    # dy[:, None], not dy: on a square block a per-row vector broadcasts silently along the
    # COLUMNS instead, which the first (square) tests could not tell apart.
    gy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * dy[:, None])   # dz / d(north): row above is north
    mag = np.hypot(gx, gy)
    slope = np.degrees(np.arctan(mag))
    with np.errstate(invalid="ignore", divide="ignore"):
        # Downslope direction is -grad z; aspect is its bearing, so sin = east, cos = north.
        s = np.where(mag > 0, -gx / mag, 0.0)
        co = np.where(mag > 0, -gy / mag, 0.0)
    # Horn's stencil never reads the centre pixel, so a filled pixel ringed by valid ones
    # would get a finite slope unless the centre is masked explicitly.
    nan = np.isnan(mag) | np.isnan(zp[1:-1, 1:-1])
    slope[nan] = np.nan
    s[nan] = np.nan
    co[nan] = np.nan
    return slope.astype(np.float32), s.astype(np.float32), co.astype(np.float32)


def _pad_columns(zp, wrap):
    if wrap:
        return np.concatenate([zp[:, -1:], zp, zp[:, :1]], axis=1)
    return np.pad(zp, ((0, 0), (1, 1)), mode="edge")


def terrain(z, lats, xres, yres, wrap=True, fill=None):
    """Slope (deg), sin(aspect), cos(aspect) of a whole elevation array (edge rows replicate)."""
    z = np.asarray(z, dtype=np.float64)
    if fill is not None:
        z = np.where(z == fill, np.nan, z)
    zp = _pad_columns(np.pad(z, ((1, 1), (0, 0)), mode="edge"), wrap)
    return _horn(zp, np.asarray(lats, dtype=np.float64), xres, yres)


def write_terrain(ele_path, out_dir, block_rows=1024, fill=-32768.0, overwrite=False):
    """Stream the DEM in row blocks and write the three rasters; returns {key: path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.join(out_dir, n) for k, n in TERRAIN_NAMES.items()}
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite {existing} (overwrite=True)")
    with rasterio.open(ele_path) as src:
        H, W, t = src.height, src.width, src.transform
        xres, yres = t.a, t.e
        if t.b != 0 or t.d != 0:
            raise ValueError("rotated grids are not supported")
        wrap = abs(abs(xres) * W - 360.0) < abs(xres) / 2
        prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
                    crs=src.crs, transform=t, nodata=np.nan, tiled=True,
                    blockxsize=512, blockysize=512, compress="deflate", predictor=3,
                    BIGTIFF="IF_SAFER")
        dsts = {k: rasterio.open(p, "w", **prof) for k, p in paths.items()}
        try:
            for r0 in range(0, H, int(block_rows)):
                r1 = min(H, r0 + int(block_rows))
                a0, a1 = max(r0 - 1, 0), min(r1 + 1, H)
                z = src.read(1, window=Window(0, a0, W, a1 - a0)).astype(np.float64)
                if fill is not None:
                    z = np.where(z == fill, np.nan, z)
                if r0 == 0:
                    z = np.vstack([z[:1], z])
                if r1 == H:
                    z = np.vstack([z, z[-1:]])
                lats = t.f + (np.arange(r0, r1) + 0.5) * yres
                out = dict(zip(("slope", "aspsin", "aspcos"),
                               _horn(_pad_columns(z, wrap), lats, xres, yres)))
                for k, d in dsts.items():
                    d.write(out[k], 1, window=Window(0, r0, W, r1 - r0))
            for k, d in dsts.items():
                d.set_band_description(1, DESCRIPTIONS[k])
                d.update_tags(source=os.path.basename(ele_path), method="Horn 1981 3x3",
                              spacing="WGS84 per-row metric spacing", fill=str(fill),
                              longitude_wrap=str(wrap), nodata="NaN")
        finally:
            for d in dsts.values():
                d.close()
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ele", default="data/raw/hm_global/hm_static_ele_1000.tiff")
    ap.add_argument("--out_dir", default="data/raw/hm_global")
    ap.add_argument("--block_rows", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from torchgeo_dataloader import STATIC_NODATA
    fill = STATIC_NODATA.get(os.path.basename(args.ele))
    t0 = time.time()
    paths = write_terrain(args.ele, args.out_dir, args.block_rows, fill=fill,
                          overwrite=args.overwrite)
    for k, p in paths.items():
        print(f"wrote {p}")
    print(f"done in {(time.time() - t0) / 60:.1f} min (fill {fill} treated as nodata)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
