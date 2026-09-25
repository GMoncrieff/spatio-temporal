#!/usr/bin/env python3
"""
Phase 1a: Prepare RESOLVE Ecoregions2017 as a raster aligned to the project's HM grid.

Downloads (or reads a cached copy of) the RESOLVE Ecoregions2017 shapefile, rasterizes
``ECO_ID`` onto the same 17111 x 40000 EPSG:4326 grid used by the HM rasters, and writes
a lookup table so that biome (n=14) and realm (n=8) aggregations are a join rather than a
second rasterization.

Outputs
-------
data/raw/hm_global/ecoregion_id_1000.tif   uint16 ECO_ID (0 = nodata / ocean)
data/raw/hm_global/ecoregion_lookup.csv    ECO_ID, ECO_NAME, BIOME_NUM, BIOME_NAME, REALM
data/ensemble/diagnostics/ecoregion_rasterization_report.csv  per-ecoregion area check

Validation performed (per plan):
  * every non-zero ECO_ID in the raster appears in the lookup
  * count of valid HM pixels falling on ECO_ID == 0 (coastline mismatch), reported
  * rasterized area per ecoregion vs the shapefile geometry area
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

RESOLVE_URL = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
HM_DIR = Path("data/raw/hm_global")
REFERENCE_RASTER = HM_DIR / "HM_2020_AA_1000.tiff"
ECOREGION_RASTER = HM_DIR / "ecoregion_id_1000.tif"
ECOREGION_LOOKUP = HM_DIR / "ecoregion_lookup.csv"
EXTERNAL_DIR = Path("data/external/ecoregions")
REPORT_DIR = Path("data/ensemble/diagnostics")

# Equal-area projection for honest polygon areas (World Mollweide).
EQUAL_AREA_CRS = "ESRI:54009"


def download_resolve(dest_dir: Path, url: str = RESOLVE_URL) -> Path:
    """Download and unzip the RESOLVE Ecoregions2017 shapefile (cached)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    shp_candidates = sorted(dest_dir.rglob("*.shp"))
    if shp_candidates:
        print(f"Using cached shapefile: {shp_candidates[0]}")
        return shp_candidates[0]

    zip_path = dest_dir / "Ecoregions2017.zip"
    if not zip_path.exists():
        import urllib.request

        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"  saved {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    shp_candidates = sorted(dest_dir.rglob("*.shp"))
    if not shp_candidates:
        raise RuntimeError(f"No .shp found after extracting {zip_path}")
    return shp_candidates[0]


def normalize_columns(gdf):
    """RESOLVE ships slightly different column spellings between releases."""
    colmap = {c.upper(): c for c in gdf.columns}

    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    out = {
        "ECO_ID": pick("ECO_ID", "ECO_ID_U", "ECOID"),
        "ECO_NAME": pick("ECO_NAME", "ECONAME"),
        "BIOME_NUM": pick("BIOME_NUM", "BIOME"),
        "BIOME_NAME": pick("BIOME_NAME", "BIOMENAME"),
        "REALM": pick("REALM", "REALM_NAME"),
    }
    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise RuntimeError(f"Shapefile missing expected columns {missing}; has {list(gdf.columns)}")
    return out


def rasterize_ecoregions(shp_path: Path, block_rows: int = 2048, overwrite: bool = False):
    import geopandas as gpd

    print(f"Reading {shp_path} ...")
    gdf = gpd.read_file(shp_path)
    cols = normalize_columns(gdf)
    print(f"  {len(gdf)} polygons, CRS {gdf.crs}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Lookup table (one row per ECO_ID).
    lut = (
        gdf[[cols["ECO_ID"], cols["ECO_NAME"], cols["BIOME_NUM"], cols["BIOME_NAME"], cols["REALM"]]]
        .rename(
            columns={
                cols["ECO_ID"]: "ECO_ID",
                cols["ECO_NAME"]: "ECO_NAME",
                cols["BIOME_NUM"]: "BIOME_NUM",
                cols["BIOME_NAME"]: "BIOME_NAME",
                cols["REALM"]: "REALM",
            }
        )
        .drop_duplicates(subset="ECO_ID")
        .sort_values("ECO_ID")
    )
    lut["ECO_ID"] = lut["ECO_ID"].astype(int)
    # ECO_ID 0 is reserved as nodata; RESOLVE uses negative/large sentinels for rock & ice
    # and lakes, which we drop from the terrestrial-ecoregion set.
    lut = lut[(lut["ECO_ID"] > 0) & (lut["ECO_ID"] < 65535)]
    ECOREGION_LOOKUP.parent.mkdir(parents=True, exist_ok=True)
    lut.to_csv(ECOREGION_LOOKUP, index=False)
    print(f"✓ Lookup written: {ECOREGION_LOOKUP} ({len(lut)} ecoregions)")

    gdf = gdf[gdf[cols["ECO_ID"]].astype(int).isin(set(lut["ECO_ID"]))].copy()
    gdf["_eco_id"] = gdf[cols["ECO_ID"]].astype(int)

    # Equal-area geometry areas for the validation check.
    print("Computing equal-area polygon areas ...")
    areas = gdf.to_crs(EQUAL_AREA_CRS).area / 1e6  # km^2
    gdf["_geom_area_km2"] = areas.values
    shp_area = gdf.groupby("_eco_id")["_geom_area_km2"].sum()

    if ECOREGION_RASTER.exists() and not overwrite:
        print(f"Raster already exists, skipping rasterization: {ECOREGION_RASTER}")
    else:
        with rasterio.open(REFERENCE_RASTER) as ref:
            profile = ref.profile.copy()
            H, W = ref.height, ref.width
            transform = ref.transform
        profile.update(
            dtype="uint16",
            count=1,
            nodata=0,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
        )
        print(f"Rasterizing onto {H} x {W} grid in {block_rows}-row stripes ...")
        sindex = gdf.sindex
        with rasterio.open(ECOREGION_RASTER, "w", **profile) as dst:
            for r0 in range(0, H, block_rows):
                rows = min(block_rows, H - r0)
                win_transform = rasterio.windows.transform(Window(0, r0, W, rows), transform)
                # Bounds of this stripe in CRS units.
                west, north = win_transform * (0, 0)
                east, south = win_transform * (W, rows)
                hits = sorted(sindex.intersection((west, south, east, north)))
                if not hits:
                    dst.write(np.zeros((rows, W), dtype="uint16"), 1, window=Window(0, r0, W, rows))
                    continue
                sub = gdf.iloc[hits]
                shapes = ((geom, int(eid)) for geom, eid in zip(sub.geometry, sub["_eco_id"]))
                block = rio_features.rasterize(
                    shapes,
                    out_shape=(rows, W),
                    transform=win_transform,
                    fill=0,
                    dtype="uint16",
                    all_touched=False,
                )
                dst.write(block, 1, window=Window(0, r0, W, rows))
                pct = 100 * (r0 + rows) / H
                print(f"  rows {r0:6d}-{r0 + rows:6d} ({pct:5.1f}%) | {len(sub)} polygons in stripe")
        print(f"✓ Raster written: {ECOREGION_RASTER}")

    return lut, shp_area


def validate_rasterization(lut: pd.DataFrame, shp_area: pd.Series, block_rows: int = 2048):
    """Per-ecoregion pixel counts, area comparison, and HM-valid-on-nodata count."""
    print("\nValidating rasterization ...")
    max_id = int(lut["ECO_ID"].max())
    counts = np.zeros(max_id + 1, dtype=np.int64)
    area_km2 = np.zeros(max_id + 1, dtype=np.float64)
    hm_valid_on_zero = 0
    hm_valid_total = 0

    with rasterio.open(ECOREGION_RASTER) as eco_src, rasterio.open(REFERENCE_RASTER) as hm_src:
        H, W = eco_src.height, eco_src.width
        transform = eco_src.transform
        px_deg = abs(transform.a)
        for r0 in range(0, H, block_rows):
            rows = min(block_rows, H - r0)
            win = Window(0, r0, W, rows)
            eco = eco_src.read(1, window=win)
            hm = hm_src.read(1, window=win, masked=True).filled(np.nan)
            hm_valid = np.isfinite(hm) & (hm >= 0)
            hm_valid_total += int(hm_valid.sum())
            hm_valid_on_zero += int((hm_valid & (eco == 0)).sum())

            # Latitude of each row -> pixel area in km^2 (spherical approximation).
            lat = np.array([rasterio.transform.xy(transform, r0 + r, 0)[1] for r in range(rows)])
            km_per_deg = 111.32
            px_area = (px_deg * km_per_deg) ** 2 * np.cos(np.deg2rad(lat))  # [rows]

            flat = eco.ravel()
            counts += np.bincount(flat, minlength=max_id + 1)[: max_id + 1]
            weights = np.repeat(px_area, W)
            area_km2 += np.bincount(flat, weights=weights, minlength=max_id + 1)[: max_id + 1]

            if (r0 // block_rows) % 2 == 0:
                print(f"  scanned rows {r0}-{r0 + rows} ({100 * (r0 + rows) / H:5.1f}%)")

    ids_present = np.nonzero(counts)[0]
    ids_present = ids_present[ids_present > 0]
    unknown = sorted(set(ids_present.tolist()) - set(lut["ECO_ID"].astype(int).tolist()))
    if unknown:
        raise RuntimeError(f"Raster contains ECO_IDs absent from the lookup: {unknown[:20]}")

    report = lut.copy()
    report["n_pixels"] = report["ECO_ID"].map(lambda i: int(counts[i]))
    report["raster_area_km2"] = report["ECO_ID"].map(lambda i: float(area_km2[i]))
    report["shapefile_area_km2"] = report["ECO_ID"].map(shp_area.to_dict()).fillna(np.nan)
    report["area_ratio"] = report["raster_area_km2"] / report["shapefile_area_km2"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "ecoregion_rasterization_report.csv"
    report.to_csv(report_path, index=False)

    n_missing = int((report["n_pixels"] == 0).sum())
    finite = report["area_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    bad = report[(report["area_ratio"] < 0.9) | (report["area_ratio"] > 1.1)]

    print("\n" + "=" * 70)
    print("RASTERIZATION VALIDATION")
    print("=" * 70)
    print(f"Ecoregions in lookup:            {len(report)}")
    print(f"Ecoregions with 0 raster pixels: {n_missing}")
    print(f"Area ratio (raster/shapefile):   median {finite.median():.4f}, "
          f"p05 {finite.quantile(0.05):.4f}, p95 {finite.quantile(0.95):.4f}")
    print(f"Ecoregions outside +/-10% area:  {len(bad)}")
    print(f"Valid HM pixels total:           {hm_valid_total:,}")
    print(f"Valid HM pixels on ECO_ID == 0:  {hm_valid_on_zero:,} "
          f"({100 * hm_valid_on_zero / max(1, hm_valid_total):.2f}% — coastline mismatch)")
    print(f"Report: {report_path}")
    print("=" * 70)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapefile", type=str, default=None, help="Path to an existing Ecoregions2017.shp")
    ap.add_argument("--block_rows", type=int, default=2048)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_validation", action="store_true")
    args = ap.parse_args()

    if not REFERENCE_RASTER.exists():
        print(f"✗ Reference raster not found: {REFERENCE_RASTER}")
        return 1

    shp = Path(args.shapefile) if args.shapefile else download_resolve(EXTERNAL_DIR)
    lut, shp_area = rasterize_ecoregions(shp, block_rows=args.block_rows, overwrite=args.overwrite)
    if not args.skip_validation:
        validate_rasterization(lut, shp_area, block_rows=args.block_rows)
    print("\n✓ Phase 1a complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
