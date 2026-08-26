#!/usr/bin/env python3
"""Precompute neighbourhood-HM context rasters.

The model has long-range information about *where change happened* — the past-change distance
band — and none at all about *where development is*. Its only HM-level input is the pixel's own
value, plus whatever the trunk's ~10 px receptive radius supplies. But development spreads from
development, not only from recently-changed ground, so "how much built-up land is within 30 km
of here" is a covariate the model has never had.

Six bands per input window: the **mean** and the **max** of HM within r = 3, 30 and 100 px.
Threshold-free by design — a mean at radius r says both how close development is and how much of
it there is, where a distance-to-cutoff says only the first and needs a cutoff nothing justifies.

**Computed on the full raster, never inside a chip.** A 201x201 window cannot be evaluated inside
a 128 px training chip. The measured precedent is the past-change context, where a 100 px radius
derived from a chip degenerated into "is there any change anywhere in this chip" — 0.752 mean
occupancy for a single changed pixel, an artifact of framing rather than a fact about geography.

Stored int16 x 1/32767. HM is bounded on [0, 1], so that is lossless to 3e-5 and halves the
five-year set from ~82 GB to ~41 GB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import maximum_filter1d, uniform_filter1d

HM_DIR = Path("data/raw/hm_global")
DEFAULT_BASES = (2000, 2005, 2010, 2015, 2020)
DEFAULT_RADII = (3, 30, 100)
DEFAULT_STATS = ("mean", "max")
HM_SCALE = 1.0 / 32767.0
NODATA = -32768
BLOCK_ROWS = 2048


def _mean2d(vals, mask, size):
    """NaN-aware neighbourhood mean: sum of finite values over count of finite values.

    Masking has to happen before the arithmetic, not after — NaN * 0 is still NaN, and a dense
    computation that masks afterwards comes back all-NaN. Both passes are separable, so this is
    two 1-D filters per array rather than a 2-D convolution.
    """
    def box(a):
        return uniform_filter1d(
            uniform_filter1d(a, size, axis=0, mode="nearest"),
            size, axis=1, mode="nearest")

    num = box(vals)
    den = box(mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


def _max2d(vals, mask, size):
    """NaN-aware neighbourhood max. A window with no finite pixel stays nodata."""
    filled = np.where(mask > 0, vals, -np.inf)
    out = maximum_filter1d(
        maximum_filter1d(filled, size, axis=0, mode="nearest"),
        size, axis=1, mode="nearest")
    return np.where(np.isfinite(out), out, np.nan)


def neighbourhood_stats(arr, radii, stats, block_rows: int = BLOCK_ROWS):
    """``{(stat, radius): array}`` over the whole array, streamed by row block.

    Each block is read with a halo of ``r`` rows so that every *output* row inside the block
    sees its full vertical footprint. The halo is ``r`` regardless of how small the block is,
    which is the case a naive implementation truncates. ``mode="nearest"`` therefore only ever
    engages at the true array edge, never at a block edge — asserted at four block sizes in
    tests/test_hm_context.py, because a wrong halo biases every pixel near a block boundary and
    produces a raster that otherwise looks entirely reasonable.
    """
    arr = np.asarray(arr, dtype=np.float32)
    H = arr.shape[0]
    radii = [int(r) for r in radii]
    out = {(s, r): np.empty_like(arr) for s in stats for r in radii}

    for r0 in range(0, H, block_rows):
        r1 = min(r0 + block_rows, H)
        for r in radii:
            lo, hi = max(0, r0 - r), min(H, r1 + r)
            chunk = arr[lo:hi]
            mask = np.isfinite(chunk).astype(np.float32)
            vals = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            size = 2 * r + 1
            keep = slice(r0 - lo, r0 - lo + (r1 - r0))
            if "mean" in stats:
                out[("mean", r)][r0:r1] = _mean2d(vals, mask, size)[keep]
            if "max" in stats:
                out[("max", r)][r0:r1] = _max2d(vals, mask, size)[keep]
    return out


def band_indices(tags, stats, radii):
    """1-based band numbers for the requested (stat, radius) pairs, from the raster's tags.

    One definition, used by both the dataloader and the prediction path. The band order is an
    implementation detail of this file, and this repo has paid for the same quantity being
    spelled out in several places (``Z975`` lives in four).
    """
    try:
        have_r = [int(v) for v in tags["radii"].split(",")]
        have_s = [v.strip() for v in tags["stats"].split(",")]
    except KeyError:
        raise ValueError(
            "hm_context raster carries no radii/stats tags; rebuild it with "
            "scripts/prepare_hm_context.py") from None
    idx = []
    for stat in stats:
        for r in radii:
            if stat not in have_s or int(r) not in have_r:
                raise ValueError(
                    f"hm_context raster has stats={have_s} radii={have_r}, "
                    f"cannot supply ({stat}, {r})")
            idx.append(have_s.index(stat) * len(have_r) + have_r.index(int(r)) + 1)
    return idx


def quantize(a):
    """int16 x 1/32767, nodata -32768. The ensemble's own storage convention."""
    # float32, not float64: the scaled values top out at 32767 and float32 is exact on
    # integers to 2^24, so rint is unaffected — and a float64 intermediate on the global grid
    # is a 5.5 GB spike for nothing.
    q = np.where(np.isfinite(a), np.rint(np.asarray(a, dtype=np.float32) / np.float32(HM_SCALE)),
                 NODATA)
    return np.clip(q, NODATA, 32767).astype(np.int16)


def build(base_year: int, radii=DEFAULT_RADII, stats=DEFAULT_STATS,
          out_dir: Path = HM_DIR, block_rows: int = BLOCK_ROWS) -> Path:
    src_path = HM_DIR / f"HM_{base_year}_AA_1000.tiff"
    if not src_path.exists():
        raise SystemExit(f"missing {src_path}")
    bands = [(s, r) for s in stats for r in radii]
    out_path = Path(out_dir) / f"hm_context_w{base_year}_1000.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=len(bands), dtype="int16", nodata=NODATA, compress="deflate",
                       tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
        hm = src.read(1).astype(np.float32)
        if src.nodata is not None:
            hm = np.where(hm == src.nodata, np.nan, hm)
        hm = np.where(np.isfinite(hm), np.clip(hm, 0.0, 1.0), np.nan)

    res = neighbourhood_stats(hm, radii, stats, block_rows=block_rows)
    del hm

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (s, r) in enumerate(bands, start=1):
            dst.write(quantize(res[(s, r)]), i)
            dst.set_band_description(i, f"{s}_HM_r{r}")
            del res[(s, r)]
        dst.update_tags(radii=",".join(str(r) for r in radii),
                        stats=",".join(stats), scale_factor=repr(HM_SCALE))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_years", default=",".join(str(b) for b in DEFAULT_BASES))
    ap.add_argument("--radii", default=",".join(str(r) for r in DEFAULT_RADII))
    ap.add_argument("--stats", default=",".join(DEFAULT_STATS))
    ap.add_argument("--out_dir", default=str(HM_DIR))
    ap.add_argument("--block_rows", type=int, default=BLOCK_ROWS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    radii = [int(r) for r in args.radii.split(",")]
    stats = tuple(s.strip() for s in args.stats.split(",") if s.strip())
    for base in [int(b) for b in args.base_years.split(",")]:
        p = Path(args.out_dir) / f"hm_context_w{base}_1000.tif"
        if p.exists() and not args.overwrite:
            print(f"  w{base}: exists, skipping (use --overwrite)")
            continue
        out = build(base, radii, stats, Path(args.out_dir), args.block_rows)
        with rasterio.open(out) as s:
            print(f"  ✓ {out.name}  {s.count} bands  {out.stat().st_size / 1e9:.2f} GB  "
                  f"[{', '.join(s.descriptions)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
