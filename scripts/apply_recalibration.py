#!/usr/bin/env python3
"""Phase 1.5c — apply the class-conditional rescale to published quantile rasters.

Reads ``scale_factors.csv`` plus the class covariates (predicted change, baseline HM level,
biome) and writes recalibrated lower/upper rasters. **Central rasters are copied byte for
byte, never regenerated**, so "the central forecast is unchanged" is structural rather than
something to verify afterwards.

Applies to the hindcast rasters (so Phase 4 can re-score them) and to the production
2025–2040 rasters (so the ensemble is built on recalibrated marginals).

Note on the assumption being made: ŝ is fit on 2000–2020 hindcast residuals and applied to
2025–2040 forecasts, i.e. the error structure is assumed stationary in time. That cannot be
validated directly — there is no future data — and it weakens the further out the forecast
runs. It is recorded in ``recal_manifest.json`` next to the outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.calibrate import ScaleFactorTable  # noqa: E402
from src.ensemble.validate import DHAT_BINS, DIST_BINS, HM_BINS, biome_lut  # noqa: E402

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"
ECO_RASTER = HM_DIR / "ecoregion_id_1000.tif"
ECO_LOOKUP = HM_DIR / "ecoregion_lookup.csv"


def recalibrate_one(
    central_path,
    lower_path,
    upper_path,
    baseline_hm_path,
    table: ScaleFactorTable,
    horizon: int,
    out_dir,
    out_stem,
    ecoregion_raster=None,
    lookup_csv=None,
    block_rows: int = 1024,
    dist_raster=None,
):
    """Write ``{stem}_lower_recal.tif`` / ``{stem}_upper_recal.tif`` and copy central."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The third class axis must be filled with whatever the factors were fit against.
    # When the fit used distance-to-past-change (the default now), reading a biome number
    # into that slot silently misses every cell and falls back to the coarse average.
    biome_map = None
    if dist_raster is None and ecoregion_raster is not None and lookup_csv is not None:
        biome_map, _, _ = biome_lut(lookup_csv)

    with rasterio.open(central_path) as c:
        profile = c.profile.copy()
        H, W = c.height, c.width
        p_t = c.transform
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", BIGTIFF="YES")

    central_out = out_dir / f"{out_stem}_central_recal.tif"
    shutil.copyfile(central_path, central_out)

    srcs = {
        "c": rasterio.open(central_path),
        "l": rasterio.open(lower_path),
        "u": rasterio.open(upper_path),
        "hm0": rasterio.open(baseline_hm_path),
    }
    eco_src = rasterio.open(ecoregion_raster) if biome_map is not None else None
    dist_src = rasterio.open(dist_raster) if dist_raster else None

    def _off(src):
        t = src.transform
        return int(round((p_t.f - t.f) / t.e)), int(round((p_t.c - t.c) / t.a))

    hm_off = _off(srcs["hm0"])
    e_off = _off(eco_src) if eco_src else None
    d_off = _off(dist_src) if dist_src else None

    stats = {"n_valid": 0, "sum_width_before": 0.0, "sum_width_after": 0.0,
             "n_monotonicity_fixed": 0, "n_clipped": 0}
    lo_out = out_dir / f"{out_stem}_lower_recal.tif"
    up_out = out_dir / f"{out_stem}_upper_recal.tif"
    try:
        with rasterio.open(lo_out, "w", **profile) as dlo, rasterio.open(up_out, "w", **profile) as dup:
            for r0 in range(0, H, block_rows):
                rr = min(block_rows, H - r0)
                win = Window(0, r0, W, rr)
                cen = srcs["c"].read(1, window=win).astype(np.float64)
                low = srcs["l"].read(1, window=win).astype(np.float64)
                upp = srcs["u"].read(1, window=win).astype(np.float64)
                hm0 = srcs["hm0"].read(1, window=Window(hm_off[1], hm_off[0] + r0, W, rr),
                                       boundless=True, fill_value=np.nan).astype(np.float64)
                hm0 = np.where(hm0 < 0, np.nan, hm0)
                valid = np.isfinite(cen) & np.isfinite(low) & np.isfinite(upp)

                if dist_src is not None:
                    dd = dist_src.read(1, window=Window(d_off[1], d_off[0] + r0, W, rr),
                                       boundless=True, fill_value=1e4).astype(np.float64)
                    biome = np.digitize(dd, DIST_BINS[1:-1]).astype(np.int64)
                elif eco_src is not None:
                    eco = eco_src.read(1, window=Window(e_off[1], e_off[0] + r0, W, rr),
                                       boundless=True, fill_value=0)
                    biome = biome_map[np.clip(eco, 0, len(biome_map) - 1)].astype(np.int64)
                else:
                    biome = np.zeros((rr, W), dtype=np.int64)

                dhat = np.where(np.isfinite(hm0), cen - hm0, 0.0)
                d_idx = np.digitize(dhat, DHAT_BINS[1:-1])
                h_idx = np.digitize(np.nan_to_num(hm0, nan=0.0), HM_BINS[1:-1])
                s_up, s_lo = table.lookup(horizon, d_idx, h_idx, biome)

                w_up = np.maximum(upp - cen, 0.0)
                w_lo = np.maximum(cen - low, 0.0)
                new_up = cen + s_up * w_up
                new_lo = cen - s_lo * w_lo

                # Guard 4: monotonicity and bounds are enforced here, not assumed.
                bad = new_lo > cen
                stats["n_monotonicity_fixed"] += int((bad & valid).sum())
                new_lo = np.minimum(new_lo, cen)
                new_up = np.maximum(new_up, cen)
                pre_clip = (new_lo < 0) | (new_up > 1)
                stats["n_clipped"] += int((pre_clip & valid).sum())
                new_lo = np.clip(new_lo, 0.0, 1.0)
                new_up = np.clip(new_up, 0.0, 1.0)

                new_lo = np.where(valid, new_lo, np.nan)
                new_up = np.where(valid, new_up, np.nan)
                stats["n_valid"] += int(valid.sum())
                stats["sum_width_before"] += float(np.nansum(np.where(valid, upp - low, 0.0)))
                stats["sum_width_after"] += float(np.nansum(np.where(valid, new_up - new_lo, 0.0)))

                dlo.write(new_lo.astype(np.float32), 1, window=win)
                dup.write(new_up.astype(np.float32), 1, window=win)
    finally:
        for s in srcs.values():
            s.close()
        if eco_src is not None:
            eco_src.close()
        if dist_src is not None:
            dist_src.close()

    width_ratio = (stats["sum_width_after"] / stats["sum_width_before"]
                   if stats["sum_width_before"] > 0 else np.nan)
    return {
        "central": str(central_out), "lower": str(lo_out), "upper": str(up_out),
        "horizon": horizon, "width_ratio": width_ratio, **stats,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factors", default="data/ensemble/calibration/scale_factors.csv")
    ap.add_argument("--targets", default="both", choices=["hindcast", "production", "both"])
    ap.add_argument("--hindcast_dir", default="data/ensemble/hindcast/stitched")
    ap.add_argument("--hindcast_suffix", default="",
                    help="Suffix before .tif on the hindcast rasters (e.g. '_blended')")
    ap.add_argument("--hindcast_out", default="data/ensemble/hindcast/recal")
    ap.add_argument("--production_dir", default="data/predictions")
    ap.add_argument("--production_out", default="data/predictions/recal")
    ap.add_argument("--production_years", default="2025,2030,2035,2040")
    ap.add_argument("--production_base_year", type=int, default=2020)
    ap.add_argument("--ecoregion_raster", default=str(ECO_RASTER))
    ap.add_argument("--lookup_csv", default=str(ECO_LOOKUP))
    ap.add_argument("--no_biome", action="store_true", help="Ignore the biome stratum")
    ap.add_argument("--dist_raster", default=None,
                    help="Distance-to-past-change raster; fills the third class axis when "
                         "the factors were fit against distance rather than biome")
    args = ap.parse_args(argv)

    table = ScaleFactorTable.from_csv(args.factors)
    eco = None if args.no_biome else args.ecoregion_raster
    lut = None if args.no_biome else args.lookup_csv
    results = []

    if args.targets in ("hindcast", "both"):
        hd = Path(args.hindcast_dir)
        suf = args.hindcast_suffix
        for cen in sorted(hd.glob(f"*_prediction_*_central{suf}.tif")):
            stem = cen.name.replace(f"_central{suf}.tif", "")
            base = int(stem.split("_")[0][1:])          # "w2000_prediction_2020" -> 2000
            target_year = int(stem.split("_")[-1])
            horizon = target_year - base
            low = hd / f"{stem}_lower{suf}.tif"
            upp = hd / f"{stem}_upper{suf}.tif"
            if not (low.exists() and upp.exists()):
                continue
            print(f"Recalibrating hindcast {stem} (h={horizon}) ...")
            results.append(recalibrate_one(
                cen, low, upp, HM_DIR / f"HM_{base}_AA_1000.tiff", table, horizon,
                args.hindcast_out, stem, ecoregion_raster=eco, lookup_csv=lut,
                dist_raster=args.dist_raster,
            ))

    if args.targets in ("production", "both"):
        pd_dir = Path(args.production_dir)
        base = args.production_base_year
        for year in [int(y) for y in args.production_years.split(",")]:
            cen = pd_dir / f"prediction_{year}_central_blended.tif"
            low = pd_dir / f"prediction_{year}_lower_blended.tif"
            upp = pd_dir / f"prediction_{year}_upper_blended.tif"
            if not (cen.exists() and low.exists() and upp.exists()):
                print(f"  ⚠ missing production rasters for {year}; skipping")
                continue
            horizon = year - base
            print(f"Recalibrating production {year} (h={horizon}) ...")
            results.append(recalibrate_one(
                cen, low, upp, HM_DIR / f"HM_{base}_AA_1000.tiff", table, horizon,
                args.production_out, f"prediction_{year}", ecoregion_raster=eco,
                lookup_csv=lut, dist_raster=args.dist_raster,
            ))

    out_manifest = Path(args.production_out if args.targets != "hindcast" else args.hindcast_out)
    out_manifest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "factors": str(args.factors),
        "stationarity_assumption": (
            "scale factors fit on 2000-2020 hindcast residuals and applied to later "
            "forecasts; assumes the error structure is stationary in time and cannot be "
            "validated directly"
        ),
        "outputs": results,
    }
    with open(out_manifest / "recal_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    print("\n" + "=" * 70)
    for r in results:
        print(f"  {Path(r['upper']).name}: width x{r['width_ratio']:.3f}, "
              f"{r['n_monotonicity_fixed']:,} monotonicity fixes, {r['n_clipped']:,} clipped")
    print(f"Manifest: {out_manifest / 'recal_manifest.json'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
