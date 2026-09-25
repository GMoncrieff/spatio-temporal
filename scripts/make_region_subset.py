#!/usr/bin/env python3
"""Crop the global hindcast artifacts to a development region.

Development iterates on the southern-Africa subregion; only explicit instruction moves work
back to the global grid. The k=5 fold models are global and their predictions are
out-of-sample everywhere, so a *crop* of the finished global rasters is exactly what a
regional rerun of Phase 0 would have produced — minus several hours of retraining.

Also derives the covariates that the region needs and the global run never had:
``past_change`` (HM_t0 − HM_{t0−10}, available at prediction time) and ``dist_past_change``
(pixels to the nearest past-change pixel), which Phase 1.5 uses as a class dimension and
Phase 4 scores with T8.

Outputs mirror the global layout under ``data/ensemble/region/<name>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"
PAST_CHANGE_THRESHOLD = 0.01
# Bands in pixels (~km). The last one is where observed change is identically zero.
DIST_BANDS = [0, 1, 3, 10, 30, 100, np.inf]
DIST_LABELS = ["0-1", "1-3", "3-10", "10-30", "30-100", ">100"]


def region_window(geojson_path, reference_raster):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with open(geojson_path) as f:
        gj = json.load(f)
    geom = unary_union([shape(f["geometry"]) for f in gj["features"]])
    minx, miny, maxx, maxy = geom.bounds
    with rasterio.open(reference_raster) as ref:
        r_top, c_left = ref.index(minx, maxy)
        r_bot, c_right = ref.index(maxx, miny)
        r0, c0 = max(0, int(r_top)), max(0, int(c_left))
        r1 = min(ref.height, int(r_bot) + 1)
        c1 = min(ref.width, int(c_right) + 1)
    return r0, c0, r1 - r0, c1 - c0


def crop(src_path, out_path, win, dtype=None):
    with rasterio.open(src_path) as src:
        r0, c0, h, w = win
        data = src.read(1, window=Window(c0, r0, w, h))
        profile = src.profile.copy()
        profile.update(height=h, width=w,
                       transform=rasterio.windows.transform(Window(c0, r0, w, h), src.transform),
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        if dtype:
            profile.update(dtype=dtype)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"]), 1)
    return out_path


def derive_past_change(win, out_dir, base_year, lag=10, threshold=PAST_CHANGE_THRESHOLD):
    """past_change = HM_base − HM_{base−lag}, and distance to the nearest past-change pixel.

    Both come only from input-side years, so they are legitimate prediction-time strata.
    """
    r0, c0, h, w = win
    def read(year):
        with rasterio.open(HM_DIR / f"HM_{year}_AA_1000.tiff") as s:
            a = s.read(1, window=Window(c0, r0, w, h)).astype(np.float32)
            prof = s.profile.copy()
            prof.update(height=h, width=w, dtype="float32", nodata=np.nan, compress="deflate",
                        transform=rasterio.windows.transform(Window(c0, r0, w, h), s.transform))
        return np.where(a < 0, np.nan, a), prof

    now, prof = read(base_year)
    then, _ = read(base_year - lag)
    past = now - then
    seed = np.isfinite(past) & (past > threshold)
    dist = distance_transform_edt(~seed).astype(np.float32)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, arr in (("past_change", past), ("dist_past_change", dist)):
        p = out_dir / f"w{base_year}_{name}.tif"
        with rasterio.open(p, "w", **prof) as d:
            d.write(arr.astype(np.float32), 1)
        paths[name] = str(p)
    return paths, int(seed.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="config/region_to_predict_small.geojson")
    ap.add_argument("--name", default="southern_africa")
    ap.add_argument("--manifest", default="data/ensemble/residuals/manifest.csv")
    ap.add_argument("--stitched_dir", default="data/ensemble/hindcast/stitched")
    ap.add_argument("--out_root", default="data/ensemble/region")
    args = ap.parse_args(argv)

    out = Path(args.out_root) / args.name
    win = region_window(args.region, HM_DIR / "HM_2020_AA_1000.tiff")
    r0, c0, h, w = win
    print(f"Region '{args.name}': rows {r0}..{r0+h}, cols {c0}..{c0+w}  ({h} x {w} px)")

    man = pd.read_csv(args.manifest)
    rows = []
    for _, r in man.iterrows():
        newr = dict(r)
        for col in [c for c in man.columns if c.startswith("path_")]:
            src = Path(r[col])
            if not src.exists():
                continue
            dest = out / ("residuals" if "res" in src.stem or src.stem.endswith(
                ("dhat", "hm_t0", "w_up", "w_lo")) else "stitched") / src.name
            newr[col] = crop(src, dest, win)
        rows.append(newr)
        print(f"  cropped w{int(r['base_year'])}_h{int(r['horizon'])}")

    sub = pd.DataFrame(rows)
    # Covariates the global run never produced, derived per input window.
    for base in sorted(sub["base_year"].unique()):
        paths, n_seed = derive_past_change(win, out / "covariates", int(base))
        for k, v in paths.items():
            sub.loc[sub["base_year"] == base, f"path_{k}"] = v
        print(f"  w{int(base)}: past-change seed pixels {n_seed:,}")

    man_path = out / "manifest.csv"
    sub.to_csv(man_path, index=False)

    # Static rasters the downstream stages expect, cropped once.
    for name, src in (("ecoregion", HM_DIR / "ecoregion_id_1000.tif"),
                      ("fold_mask", HM_DIR / "fold_mask_1000.tif")):
        if src.exists():
            crop(src, out / f"{name}.tif", win)
            print(f"  cropped {name}")

    print(f"\n✓ Regional working set: {out}")
    print(f"  manifest: {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
