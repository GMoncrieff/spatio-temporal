#!/usr/bin/env python3
"""A written raster is not a finished raster. Count what is in it.

The failure this exists to stop, measured 2026-08-30: sixteen global forecast rasters were
truncated at the classic-TIFF 4 GiB ceiling and every unwritten row read back as **finite
zeros** -- plausible numbers, no exception, and an ``[ -f "$p" ]`` check passed all sixteen.
Had the chain not aborted for an unrelated reason those zeros would have gone into COGs, an
icechunk store and the published product, each step faithfully preserving them.

So the test is not existence but the valid-pixel count, and it is **two-sided**: a truncated
raster reads back with MORE valid pixels than it should, because the unwritten remainder is
not nodata. One-sided "at least N" would have passed every one of the sixteen.

The container is checked too. A TIFF whose version word is 42 is a classic TIFF and is
capped at 4 GiB however large the data wants to be; 43 is BigTIFF. GDAL's ``IF_NEEDED``
default cannot promote a *compressed* raster because it cannot predict the compressed size,
which is precisely how those sixteen came to exist.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import rasterio

#: Valid land pixels on the 17111 x 40000 global grid, from the validity mask.
GLOBAL_LAND_PX = 184_600_000


def tiff_version(path: Path) -> int:
    """42 = classic TIFF (4 GiB ceiling), 43 = BigTIFF. Read from the 8-byte header."""
    with open(path, "rb") as fh:
        head = fh.read(4)
    if len(head) < 4:
        raise ValueError(f"{path}: shorter than a TIFF header")
    endian = "<" if head[:2] == b"II" else ">"
    return struct.unpack(endian + "H", head[2:4])[0]


def count_valid(path: Path) -> tuple[int, int]:
    """Valid pixels in band 1, streamed by block. Returns (n_valid, n_bands).

    A float raster can carry a sentinel nodata AND be filled with NaN -- the forward
    product's triple did exactly that, declaring 3.4e38 and writing NaN. Testing only
    ``a != nodata`` then counts every NaN as valid (NaN != anything is true) and the whole
    684.4 M grid reads as data. Both conditions, always.
    """
    with rasterio.open(path) as src:
        nod = src.nodata
        is_float = np.issubdtype(np.dtype(src.dtypes[0]), np.floating)
        n = 0
        for _, win in src.block_windows(1):
            a = src.read(1, window=win)
            if is_float:
                ok = np.isfinite(a)
                if nod is not None and np.isfinite(nod):
                    ok &= a != nod
            else:
                ok = np.ones(a.shape, bool) if nod is None else (a != nod)
            n += int(ok.sum())
        return n, src.count


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--patterns", required=True,
                    help="Comma-separated globs, each of which must match at least one file")
    ap.add_argument("--expect_px", type=int, default=GLOBAL_LAND_PX,
                    help=f"Valid pixels a complete raster holds (default {GLOBAL_LAND_PX:,}, "
                         f"the global land total)")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="Fractional tolerance, applied in BOTH directions")
    ap.add_argument("--min_px", type=int, default=0,
                    help="Absolute lower bound, overriding --expect_px/--tol. Use when the "
                         "right answer is a RANGE rather than a number with a tolerance.")
    ap.add_argument("--max_px", type=int, default=0, help="Absolute upper bound, likewise")
    ap.add_argument("--expect_files", type=int, default=0,
                    help="Total files the patterns must match between them. 0 disables. "
                         "A pattern that matches nothing is always an error.")
    ap.add_argument("--require_bigtiff", default="",
                    help="Comma-separated substrings; any matched file whose name contains "
                         "one must be BigTIFF (version 43)")
    a = ap.parse_args(argv)

    root = Path(a.dir)
    big = [s for s in a.require_bigtiff.split(",") if s]
    bad, n_files = 0, 0
    if a.min_px or a.max_px:
        # An absolute range, because some expectations are not a number plus a wobble. A
        # fold's PREDICTION raster is one: prediction keeps every pixel of every 128 px tile
        # that overlaps the fold, not just the fold's own pixels, so it covers about 1.9x
        # its territory -- MEASURED 2026-09-17 on fold_mask_b4_1000, 67.4-71.6 M px against
        # an own-land share of 35.8-38.4 M. A tolerance around the share would have refused
        # every fold at the end of a nine-hour run. The ceiling is what does the work: a
        # raster truncated at the 4 GiB classic-TIFF limit reads its unwritten remainder
        # back as finite ZEROS, so it reports close to the whole 684.4 M grid.
        lo = float(a.min_px) if a.min_px else 0.0
        hi = float(a.max_px) if a.max_px else float("inf")
        ref = a.min_px or a.expect_px
    else:
        lo, hi = a.expect_px * (1 - a.tol), a.expect_px * (1 + a.tol)
        ref = a.expect_px
    for pat in a.patterns.split(","):
        hits = sorted(root.glob(pat))
        if not hits:
            print(f"FATAL: no file matches {pat!r} under {root}")
            bad += 1
            continue
        for p in hits:
            n_files += 1
            ver = tiff_version(p)
            n, nb = count_valid(p)
            flag = ""
            if not (lo <= n <= hi):
                flag += "  <<< INCOMPLETE OR OVERFULL"
                bad += 1
            if any(s in p.name for s in big) and ver != 43:
                flag += "  <<< NOT BIGTIFF"
                bad += 1
            print(f"  {p.name:52s} v{ver} {nb:>3d} band(s) {n:>12,} px "
                  f"{n / ref:6.1%}{flag}")

    if a.expect_files and n_files != a.expect_files:
        print(f"FATAL: matched {n_files} files, expected {a.expect_files}")
        bad += 1
    if bad:
        print(f"FATAL: {bad} problem(s); these rasters are not a finished product")
        return 1
    bounds = (f"[{lo:,.0f}, {hi:,.0f}] px" if (a.min_px or a.max_px)
              else f"{a.expect_px:,} px +/- {a.tol:.0%}")
    print(f"  ✓ {n_files} raster(s) complete ({bounds}, BigTIFF where required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
