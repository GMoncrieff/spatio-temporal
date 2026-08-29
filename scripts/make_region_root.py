#!/usr/bin/env python3
"""Crop a region root out of the global assets.

``make_region_subset.py`` cannot build one any more: it crops the rasters named in
``data/ensemble/residuals/manifest.csv``, and that manifest belongs to a model whose artifacts
were deleted in the 2026-08-24 cleanup. ``make_global_region_root.py`` builds the global root but
only links assets at full extent.

This script does what is actually needed for regional iteration: take the three static assets the
Phase 1-4 loop reads and crop them to a GeoJSON bounding box on the shared 17111 x 40000 grid.

``dist_past_change`` is **cropped from band 2 of the global change-context raster, never
recomputed locally**. CLAUDE.md rule 8: the Euclidean distance transform must see the whole
raster. Recomputing it inside a window makes every pixel near the window edge measure the
distance to the nearest change *inside the window*, which is an artifact of framing. ``--verify``
quantifies exactly that difference rather than asserting the two agree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"
DEFAULT_BASES = (2000, 2005, 2010, 2015, 2020)
DIST_BAND = 2  # 1 = past_change, 2 = distance to nearest past_change > 0.01
PAST_CHANGE_LAG = 10
PAST_CHANGE_THRESHOLD = 0.01


def region_window(geojson: Path, reference: Path) -> Window:
    """Integer pixel window covering the GeoJSON bbox on the reference grid."""
    gj = json.loads(Path(geojson).read_text())
    xs, ys = [], []
    for feat in gj["features"]:
        for ring in feat["geometry"]["coordinates"]:
            for x, y in ring:
                xs.append(x)
                ys.append(y)
    with rasterio.open(reference) as src:
        win = from_bounds(min(xs), min(ys), max(xs), max(ys), transform=src.transform)
        # Floor the top-left and ceil the bottom-right -- an OUTER box, which is exactly what
        # train_lightning.py's prediction path does with the same GeoJSON. The previous
        # `round_offsets().round_lengths()` rounds to NEAREST, so whenever a region edge lands
        # short of a pixel centre the root comes out one row or column smaller than the
        # rasters it has to align with. Africa cropped to 8111 rows against the predictions'
        # 8112; southern Africa happened to round the same way, which is why this survived.
        # The cost is not a warning: it surfaced as an IndexError deep inside the coverage
        # audit (943 against 944), and would have mis-indexed the distance band in
        # generate_ensemble.py without any error at all.
        col_off = max(0, int(np.floor(win.col_off)))
        row_off = max(0, int(np.floor(win.row_off)))
        width = min(int(np.ceil(win.col_off + win.width)), src.width) - col_off
        height = min(int(np.ceil(win.row_off + win.height)), src.height) - row_off
    return Window(col_off, row_off, width, height)


def crop(src_path: Path, out_path: Path, win: Window, band: int = 1,
         dtype: str | None = None, nodata=None) -> Path:
    with rasterio.open(src_path) as src:
        data = src.read(band, window=win)
        profile = src.profile.copy()
        profile.update(
            count=1,
            height=int(win.height),
            width=int(win.width),
            transform=src.window_transform(win),
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        if dtype is not None:
            profile["dtype"] = dtype
            data = data.astype(dtype)
        if nodata is not None:
            profile["nodata"] = nodata
        desc = src.descriptions[band - 1] if src.descriptions else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        if desc:
            dst.set_band_description(1, desc)
    return out_path


def local_recompute(base_year: int, win: Window) -> np.ndarray:
    """The distance transform derived *inside* the window — the thing rule 8 forbids."""
    from scipy.ndimage import distance_transform_edt

    def rd(year):
        with rasterio.open(HM_DIR / f"HM_{year}_AA_1000.tiff") as s:
            a = s.read(1, window=win).astype(np.float32)
            return np.where(a == s.nodata, np.nan, a)

    past = rd(base_year) - rd(base_year - PAST_CHANGE_LAG)
    changed = np.nan_to_num(past, nan=0.0) > PAST_CHANGE_THRESHOLD
    return distance_transform_edt(~changed).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--region", default="config/region_to_predict_small.geojson")
    ap.add_argument("--name", default="southern_africa")
    ap.add_argument("--out_root", default="data/ensemble/region")
    ap.add_argument("--fold_mask", default=str(HM_DIR / "fold_mask_b4_1000.tif"),
                    help="Which global fold mask to crop. Defaults to the 512 px block mask the "
                         "production run used; the 128 px checkerboard holds geography out at "
                         "one residual correlation length (CLAUDE.md rule 19).")
    ap.add_argument("--base_years", default=",".join(str(b) for b in DEFAULT_BASES))
    ap.add_argument("--verify", action="store_true",
                    help="Report how far a locally recomputed distance transform departs from "
                         "the globally computed one, and the per-fold pixel counts.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    ref = HM_DIR / "HM_2020_AA_1000.tiff"
    win = region_window(Path(args.region), ref)
    root = Path(args.out_root) / args.name
    root.mkdir(parents=True, exist_ok=True)
    print(f"region {args.name}: window row {int(win.row_off)}, col {int(win.col_off)}, "
          f"{int(win.height)} x {int(win.width)} px")

    crop(HM_DIR / "ecoregion_id_1000.tif", root / "ecoregion.tif", win)
    print(f"  ecoregion.tif        <- ecoregion_id_1000.tif")
    crop(Path(args.fold_mask), root / "fold_mask.tif", win)
    print(f"  fold_mask.tif        <- {Path(args.fold_mask).name}")

    for base in [int(b) for b in args.base_years.split(",")]:
        out = root / "covariates" / f"w{base}_dist_past_change.tif"
        if out.exists() and not args.overwrite:
            print(f"  w{base}_dist_past_change.tif exists, skipping (use --overwrite)")
            continue
        src = HM_DIR / f"change_context_w{base}_1000.tif"
        with rasterio.open(src) as s:
            desc = (s.descriptions[DIST_BAND - 1] or "").lower()
        if "distance" not in desc:
            raise SystemExit(f"{src} band {DIST_BAND} is {desc!r}, not a distance band")
        crop(src, out, win, band=DIST_BAND, dtype="float32", nodata=np.nan)
        print(f"  {out.name} <- {src.name} band {DIST_BAND}")

    if args.verify:
        print("\nverification")
        with rasterio.open(ref) as s:
            hm = s.read(1, window=win)
            valid = np.isfinite(hm) & (hm != s.nodata)
        with rasterio.open(root / "fold_mask.tif") as s:
            fm = s.read(1)
        tot = int(valid.sum())
        print(f"  valid px {tot:,}")
        for k in range(1, 6):
            n = int(((fm == k) & valid).sum())
            print(f"    fold {k}: {n:8,d} px  ({n / tot:.3f})")
        for base in [int(b) for b in args.base_years.split(",")]:
            with rasterio.open(root / "covariates" / f"w{base}_dist_past_change.tif") as s:
                glob = s.read(1)
            loc = local_recompute(base, win)
            m = np.isfinite(glob) & valid
            diff = np.abs(glob[m] - loc[m])
            print(f"    w{base}: local recompute differs from the global transform at "
                  f"{int((diff > 0).sum()):,} / {int(m.sum()):,} px "
                  f"(max {diff.max():.1f} px). The global raster is the correct one.")

    print(f"\n✓ region root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
