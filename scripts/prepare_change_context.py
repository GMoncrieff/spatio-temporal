#!/usr/bin/env python3
"""Precompute past-change context rasters for the quantile heads.

Why this exists as a raster rather than a computation inside the model: the context has to
answer "is there past change within 100 px of here", and training chips are 128 px. Deriving
it from the chip means the 100 px radius saturates against the chip boundary (measured: with
one past-change pixel in a chip, the 100 px channel reads 0.752 mean — it has degenerated
into "is there any change in this chip"). Computed on the full raster instead, 100 px means
100 km of real geography, and training and inference see the same thing.

Two bands per input window, both derived only from input-side years:
  1. past_change      HM_base − HM_{base−lag}
  2. dist_past_change pixels to the nearest pixel with past_change > threshold

Occupancy at any radius is then ``dist <= r``, so the model needs no pooling at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent.parent))

HM_DIR = Path("data/raw/hm_global")
DEFAULT_BASES = (2000, 2005, 2010, 2015, 2020)
THRESHOLD = 0.01
LAG = 10


def build(base_year: int, lag: int = LAG, threshold: float = THRESHOLD, out_dir: Path = HM_DIR):
    now_p = HM_DIR / f"HM_{base_year}_AA_1000.tiff"
    then_p = HM_DIR / f"HM_{base_year - lag}_AA_1000.tiff"
    if not (now_p.exists() and then_p.exists()):
        print(f"  ⚠ missing inputs for base {base_year}; skipping")
        return None

    with rasterio.open(now_p) as s:
        now = s.read(1).astype(np.float32)
        profile = s.profile.copy()
    with rasterio.open(then_p) as s:
        then = s.read(1).astype(np.float32)
    now = np.where(now < 0, np.nan, now)
    then = np.where(then < 0, np.nan, then)

    past = now - then
    seed = np.isfinite(past) & (past > threshold)
    print(f"  base {base_year}: {int(seed.sum()):,} past-change seed pixels")
    dist = distance_transform_edt(~seed).astype(np.float32)

    profile.update(count=2, dtype="float32", nodata=np.nan, compress="deflate",
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
    out = Path(out_dir) / f"change_context_w{base_year}_1000.tif"
    with rasterio.open(out, "w", **profile) as d:
        d.write(np.nan_to_num(past, nan=0.0).astype(np.float32), 1)
        d.write(dist, 2)
        d.set_band_description(1, f"past_change HM_{base_year} - HM_{base_year - lag}")
        d.set_band_description(2, f"distance (px) to nearest past_change > {threshold}")
    print(f"    -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_years", default=",".join(str(b) for b in DEFAULT_BASES))
    ap.add_argument("--lag", type=int, default=LAG)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out_dir", default=str(HM_DIR))
    args = ap.parse_args(argv)

    print("Building past-change context rasters (full extent)")
    for b in [int(x) for x in args.base_years.split(",")]:
        build(b, lag=args.lag, threshold=args.threshold, out_dir=Path(args.out_dir))
    print("✓ done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
