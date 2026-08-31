#!/usr/bin/env python3
"""Publish a recalibrated prediction directory as cloud-optimised GeoTIFFs.

The pipeline's working rasters are deflate-compressed tiled GeoTIFFs without overviews, which
is fine for the scripts that read them by window and useless for anything that wants to stream
a global raster over HTTP. This converts one window's twelve rasters (three quantiles x four
horizons) into COGs under a publication name, and then *proves* the conversion rather than
assuming it: a COG that silently dropped nodata or shifted its transform still opens fine.

Deliberately shells out to ``gdal_translate -of COG``. ``rio_cogeo`` is not installed and the
Python ``osgeo`` bindings are absent from this environment, but the GDAL CLI is present and its
COG driver builds the overviews and the header layout in one pass.

Source names are the pipeline's (``w{base}_prediction_{year}_{q}_recal.tif``); output names are
the product's (``{prefix}_{year}_{q}.tif``), so the deliverable does not leak the internal
suffix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

QUANTILES = ("central", "lower", "upper")
CREATION = [
    "-co", "COMPRESS=DEFLATE",
    "-co", "PREDICTOR=2",
    "-co", "BLOCKSIZE=512",
    "-co", "OVERVIEW_RESAMPLING=AVERAGE",
    "-co", "BIGTIFF=YES",
    "-co", "NUM_THREADS=ALL_CPUS",
]


def translate(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gdal_translate", "-of", "COG", *CREATION, str(src), str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"gdal_translate failed for {src}:\n{r.stderr.strip()[-2000:]}")


def verify(src: Path, dst: Path, n_windows: int = 12, seed: int = 42,
           min_px: int = 0) -> str:
    """Assert the COG is a COG, and that it carries the same numbers as its source.

    ``min_px`` guards against a vacuous pass. The value comparison only runs where the source
    is finite, and the global grid is 73% ocean, so a dozen random windows can all land on
    nodata and the check reports success having compared nothing. Regionally the sample was
    almost all land and the hole never showed.
    """
    info = subprocess.run(["gdalinfo", str(dst)], capture_output=True, text=True).stdout
    if "LAYOUT=COG" not in info:
        raise SystemExit(f"{dst} is not reported as LAYOUT=COG")
    if "Overviews:" not in info:
        raise SystemExit(f"{dst} has no overviews")

    with rasterio.open(src) as a, rasterio.open(dst) as b:
        if (a.width, a.height) != (b.width, b.height):
            raise SystemExit(f"{dst}: shape {b.width}x{b.height} != source {a.width}x{a.height}")
        if a.transform != b.transform:
            raise SystemExit(f"{dst}: transform differs from source")
        if a.crs != b.crs:
            raise SystemExit(f"{dst}: CRS differs from source")
        if not (a.nodata is None and b.nodata is None) and not (
            a.nodata == b.nodata or (np.isnan(a.nodata) and np.isnan(b.nodata))
        ):
            raise SystemExit(f"{dst}: nodata {b.nodata} != source {a.nodata}")
        # Full-raster equality would read 2.7 GB twice; sample instead, and sample where the
        # data is rather than uniformly, because most of this grid is ocean.
        rng = np.random.default_rng(seed)
        checked = 0
        for _ in range(n_windows):
            r0 = int(rng.integers(0, max(1, a.height - 1024)))
            c0 = int(rng.integers(0, max(1, a.width - 1024)))
            win = rasterio.windows.Window(c0, r0, 1024, 1024)
            xa, xb = a.read(1, window=win), b.read(1, window=win)
            if not np.array_equal(np.isnan(xa), np.isnan(xb)):
                raise SystemExit(f"{dst}: nodata mask differs at row {r0}, col {c0}")
            m = ~np.isnan(xa)
            if m.any() and not np.array_equal(xa[m], xb[m]):
                d = float(np.nanmax(np.abs(xa[m] - xb[m])))
                raise SystemExit(f"{dst}: values differ at row {r0}, col {c0} (max |diff| {d})")
            checked += int(m.sum())
    if checked < min_px:
        raise SystemExit(
            f"{dst}: only {checked:,} valid pixels were compared against a floor of "
            f"{min_px:,} -- the sample landed on nodata and verified nothing. Raise "
            f"--verify_windows.")
    ovr = info.split("Overviews:")[1].split("\n")[0].strip()
    return f"COG, overviews {ovr}, {checked:,} valid px verified identical"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", required=True, help="a recal_w* directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True,
                    help="publication basename, e.g. 'hm_hindcast_holdout'")
    ap.add_argument("--base_year", type=int, required=True)
    ap.add_argument("--years", required=True, help="comma-separated target years")
    ap.add_argument("--src_pattern",
                    default="w{base}_prediction_{year}_{q}_recal.tif")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify_windows", type=int, default=12,
                    help="1024 px windows sampled per raster for the value comparison")
    ap.add_argument("--min_verified_px", type=int, default=0,
                    help="Fail if fewer than this many valid pixels were actually compared. "
                         "The comparison skips nodata, and on a 73%%-ocean global grid a "
                         "random sample can verify nothing while reporting success.")
    args = ap.parse_args(argv)

    src_dir, out_dir = Path(args.src_dir), Path(args.out_dir)
    years = [int(y) for y in args.years.split(",")]
    written = []
    for year in years:
        for q in QUANTILES:
            src = src_dir / args.src_pattern.format(base=args.base_year, year=year, q=q)
            if not src.exists():
                raise SystemExit(f"missing source raster: {src}")
            dst = out_dir / f"{args.prefix}_{year}_{q}.tif"
            if dst.exists() and not args.overwrite:
                print(f"  {dst.name} exists, skipping (use --overwrite)")
            else:
                translate(src, dst)
            print(f"  {dst.name}  ({dst.stat().st_size / 1e9:.2f} GB)  "
                  f"{verify(src, dst, n_windows=args.verify_windows, min_px=args.min_verified_px)}")
            written.append(dst)

    print(f"\n✓ {len(written)} COGs -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
