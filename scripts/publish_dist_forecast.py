#!/usr/bin/env python3
"""Average the fold models' forward rasters and publish them as cloud-optimised GeoTIFFs.

The forward product holds nothing out, so there is no mosaic to build and no seam: every fold
model predicts the whole extent and the folds are averaged at every pixel
(``stitch_fold_predictions(mode="mean")``, which is the mode written for exactly this case).

**What averaging costs, stated because it is easy to miss.** Averaging quantile functions
pointwise in ``u`` preserves monotonicity, so the averaged triple is still a valid forecast --
but the averaged interval is *narrower* than either fold's own, because it discards the
between-fold spread instead of adding it. On the hindcast that spread was 0.0383 mean pairwise
on the upper bound against 0.0053 on the central field, so the effect lands almost entirely on
the interval width and hardly at all on the centre. A single fold's rasters are kept beside the
mean for anyone who needs a model's own dispersion rather than a consensus.

    python scripts/publish_dist_forecast.py --pred_dir … --out_dir … --years 2025,2030,2035,2040
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ensemble.residuals import stitch_fold_predictions  # noqa: E402

QUANTILES = ("lower", "central", "upper")


def check(path: Path, years, quantiles) -> None:
    """Prove the COG rather than assume it: a conversion that dropped nodata or shifted the
    transform still opens fine, and the failure would only appear in a viewer."""
    with rasterio.open(path) as src:
        assert src.driver == "GTiff", src.driver
        if not src.overviews(1):
            raise SystemExit(f"{path}: no overviews — not a COG")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True, help="directory holding fold{N}_prediction_*.tif")
    ap.add_argument("--mean_dir", required=True, help="where the fold-averaged rasters go")
    ap.add_argument("--out_dir", required=True, help="where the COGs go")
    ap.add_argument("--years", default="2025,2030,2035,2040")
    ap.add_argument("--folds", default="1,2")
    ap.add_argument("--fold_mask", default="data/raw/hm_global/fold_mask_b4_1000.tif")
    ap.add_argument("--prefix", default="hm_forecast_e1")
    ap.add_argument("--base_year", type=int, default=2020)
    args = ap.parse_args(argv)

    pred_dir, mean_dir, out_dir = Path(args.pred_dir), Path(args.mean_dir), Path(args.out_dir)
    mean_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    years = [int(y) for y in args.years.split(",")]
    folds = [int(f) for f in args.folds.split(",")]

    print(f"averaging folds {folds} over {len(years)} years x {len(QUANTILES)} quantiles")
    for year in years:
        for q in QUANTILES:
            fold_paths = {}
            for f in folds:
                p = pred_dir / f"fold{f}_prediction_{year}_{q}.tif"
                if not p.exists():                      # the writer appends _blended for tiled runs
                    alt = pred_dir / f"fold{f}_prediction_{year}_{q}_blended.tif"
                    p = alt if alt.exists() else p
                if p.exists():
                    fold_paths[f] = str(p)
            if len(fold_paths) != len(folds):
                raise SystemExit(f"{year} {q}: found {sorted(fold_paths)} of folds {folds} "
                                 f"in {pred_dir} — refusing to publish a partial average")
            out = mean_dir / f"prediction_{year}_{q}.tif"
            info = stitch_fold_predictions(fold_paths, args.fold_mask, str(out), mode="mean")
            print(f"  ✓ {year} {q}: {info['n_valid_px']:,} px -> {out}")

    cmd = [sys.executable, "scripts/make_cogs.py",
           "--src_dir", str(mean_dir), "--out_dir", str(out_dir),
           "--prefix", args.prefix, "--base_year", str(args.base_year),
           "--years", args.years, "--src_pattern", "prediction_{year}_{q}.tif",
           "--overwrite"]
    print("\n" + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"make_cogs.py failed with {r.returncode}")

    print("\nverifying the published COGs carry overviews")
    for p in sorted(out_dir.glob("*.tif")):
        check(p, years, QUANTILES)
        with rasterio.open(p) as src:
            print(f"  ✓ {p.name}  {src.width}x{src.height}  {src.count} band(s)  "
                  f"overviews {src.overviews(1)}  nodata {src.nodata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
