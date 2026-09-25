#!/usr/bin/env python3
"""Build the global region root that the Phase 1-4 loop consumes.

``make_region_subset.py`` cannot do this job: it crops residual and stitched rasters out of
``data/ensemble/residuals/manifest.csv``, which points at a superseded model. The global run
needs only the three static assets, and two of them already exist at full extent.

The third, ``w{year}_dist_past_change.tif``, is **band 2 of**
``data/raw/hm_global/change_context_w{year}_1000.tif`` — same lag (10), same threshold (0.01),
same ``distance_transform_edt``, and computed on the full raster, which is what CLAUDE.md rule 8
requires. Verified: the Africa region root's ``w2000_dist_past_change.tif`` is pixel-identical
to that band cropped at row 5110, col 18000. So this script extracts rather than recomputes,
and ``--verify`` re-proves the identity for every year it writes.

Covariates are large (~1 GB each), so they are written to ``--covariate_dir`` (the HDD) and the
region root links to them. ecoregion and fold_mask are small and are linked straight to the
global sources, so there is exactly one copy of the fold assignment on the box.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"
DEFAULT_BASES = (2000, 2005, 2010, 2015, 2020)
DIST_BAND = 2  # 1 = past_change, 2 = distance to nearest past_change > 0.01
BLOCK_ROWS = 2048


def extract_distance(base_year: int, out_path: Path, block_rows: int = BLOCK_ROWS) -> Path:
    """Write band 2 of the global change-context raster as a standalone single-band GeoTIFF."""
    src_path = HM_DIR / f"change_context_w{base_year}_1000.tif"
    if not src_path.exists():
        raise SystemExit(f"missing global change context: {src_path}")

    with rasterio.open(src_path) as src:
        if src.count < DIST_BAND:
            raise SystemExit(f"{src_path} has {src.count} bands, need {DIST_BAND}")
        desc = src.descriptions[DIST_BAND - 1] or ""
        if "distance" not in desc.lower():
            raise SystemExit(
                f"{src_path} band {DIST_BAND} is described as {desc!r}, not a distance band"
            )
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate",
                       tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for r0 in range(0, src.height, block_rows):
                rr = min(block_rows, src.height - r0)
                win = Window(0, r0, src.width, rr)
                dst.write(src.read(DIST_BAND, window=win), 1, window=win)
            dst.set_band_description(1, desc)
    return out_path


def verify_against_africa(base_year: int, path: Path, africa_root: Path) -> str:
    """Re-prove that extraction equals the covariate the Africa scorecard was fitted on."""
    ref = africa_root / "covariates" / f"w{base_year}_dist_past_change.tif"
    if not ref.exists():
        return f"w{base_year}: no Africa reference, skipped"
    with rasterio.open(africa_root / "ecoregion.tif") as a:
        at, H, W = a.transform, a.height, a.width
    with rasterio.open(path) as g:
        gt = g.transform
        r0 = int(round((gt.f - at.f) / abs(gt.e)))
        c0 = int(round((at.c - gt.c) / gt.a))
        got = g.read(1, window=Window(c0, r0, W, H))
    with rasterio.open(ref) as s:
        want = s.read(1)
    if not np.array_equal(got, want):
        n = int((got != want).sum())
        raise SystemExit(
            f"w{base_year}: extraction differs from {ref} at {n:,} px "
            f"(max |diff| {np.nanmax(np.abs(got - want))})"
        )
    return f"w{base_year}: bit-identical to the Africa covariate at row {r0}, col {c0}"


def link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target.resolve())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", default="data/ensemble/region/global")
    ap.add_argument("--covariate_dir",
                    default="/mnt/hdd1/spatio-temporal/data/ensemble/region/global/covariates",
                    help="Where the ~1 GB distance rasters actually live (the HDD); the region "
                         "root links to it.")
    ap.add_argument("--base_years", default=",".join(str(b) for b in DEFAULT_BASES))
    ap.add_argument("--africa_root", default="data/ensemble/region/africa")
    ap.add_argument("--verify", action="store_true",
                    help="Assert each extracted raster matches the Africa covariate exactly")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out_root)
    cov = Path(args.covariate_dir)
    out.mkdir(parents=True, exist_ok=True)
    cov.mkdir(parents=True, exist_ok=True)

    for name, src in (("ecoregion", HM_DIR / "ecoregion_id_1000.tif"),
                      ("fold_mask", HM_DIR / "fold_mask_1000.tif")):
        if not src.exists():
            raise SystemExit(f"missing global source: {src}")
        link(src, out / f"{name}.tif")
        with rasterio.open(src) as s:
            print(f"  {name}.tif -> {src}  ({s.width} x {s.height}, {s.dtypes[0]})")

    checks = []
    for base in [int(b) for b in args.base_years.split(",")]:
        p = cov / f"w{base}_dist_past_change.tif"
        if p.exists() and not args.overwrite:
            print(f"  w{base}_dist_past_change.tif exists, skipping (use --overwrite)")
        else:
            extract_distance(base, p)
            print(f"  wrote {p}  ({p.stat().st_size / 1e9:.2f} GB)")
        if args.verify:
            checks.append(verify_against_africa(base, p, Path(args.africa_root)))

    link(cov, out / "covariates")
    print(f"  covariates -> {cov}")

    for line in checks:
        print(f"  verified {line}")

    print(f"\n✓ Global region root: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
