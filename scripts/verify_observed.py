#!/usr/bin/env python
"""Verify the observed-HM icechunk store against its source rasters and the prediction stores.

    python scripts/verify_observed.py \
        --store .../products/observed/observed_hm.icechunk \
        --like  .../products/E2a/hindcast/hindcast_qf.icechunk .../products/E2a/forecast/forecast_qf.icechunk

A SECOND IMPLEMENTATION, on purpose. Nothing here imports the writer's reader or encoder:
missing is decided off the raster's own nodata tag, and the code is recomputed from the
prescription ``min(rint(clip(x,0,1)*65535), 65534)`` in float32. Agreement between two
spellings that share no code is the evidence; one spelling checked against itself is not.

EVERY PIXEL, NOT A SAMPLE. One year of the store is 1.4 GB of uint16 and the source 2.7 GB of
float32; the whole comparison is minutes. CLAUDE.md rule 35: count it instead of sampling it.

BUILT-IN CONTROL. On the first row band of the first year the naive encoding -- the
prediction exporter's ``isfinite`` test on the RAW array, where nodata is the finite 3.4e38 --
is evaluated too, and the run fails unless that control DOES disagree with the store. A check
that could not tell the two apart would pass a store that wrote the ocean as HM = 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

U16 = 65535
DIMS = ("year", "latitude", "longitude")


def open_group(path):
    import icechunk
    import zarr
    s = icechunk.Repository.open(icechunk.local_filesystem_storage(str(path))).readonly_session("main")
    return s, zarr.open_group(s.store, mode="r")


def prescribed(raw: np.ndarray, nodata) -> tuple[np.ndarray, np.ndarray]:
    """(codes, valid) from a raw float32 band, independent of the writer."""
    valid = np.isfinite(raw)
    if nodata is not None and np.isfinite(nodata):
        valid &= raw != np.float32(nodata)
    code = np.minimum(np.rint(np.clip(raw, np.float32(0), np.float32(1)) * np.float32(U16)),
                      np.float32(U16 - 1))
    return np.where(valid, code, np.float32(U16)).astype(np.uint16), valid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--like", nargs="+", required=True,
                    help="prediction stores the observed store must align with")
    ap.add_argument("--hm_dir", default="data/raw/hm_global")
    ap.add_argument("--row_chunk", type=int, default=1024)
    a = ap.parse_args(argv)

    bad: list[str] = []
    _, g = open_group(a.store)

    # ---- structure
    members = set(g.array_keys())
    if members != {"year", "latitude", "longitude", "hm", "crs"}:
        bad.append(f"members {sorted(members)}")
    hm = g["hm"]
    years = [int(y) for y in g["year"][:]]
    H, W = hm.shape[1:]
    print(f"  hm {hm.shape} {hm.dtype} chunks {hm.chunks} shards {hm.shards} "
          f"fill {hm.metadata.fill_value} dims {hm.metadata.dimension_names}")
    print(f"  years {years}")
    if hm.dtype != np.uint16:
        bad.append(f"dtype {hm.dtype}")
    if tuple(hm.metadata.dimension_names or ()) != DIMS:
        bad.append(f"hm dimension_names {hm.metadata.dimension_names}")
    for k in ("year", "latitude", "longitude"):
        if tuple(g[k].metadata.dimension_names or ()) != (k,):
            bad.append(f"{k} dimension_names {g[k].metadata.dimension_names}")
    if int(hm.metadata.fill_value) != U16:
        bad.append(f"fill {hm.metadata.fill_value}")

    # ---- alignment with every prediction store named
    for lp in a.like:
        _, p = open_group(lp)
        ph = p["hm"]
        tag = Path(lp).name
        checks = {
            "latitude": np.array_equal(g["latitude"][:], p["latitude"][:]),
            "longitude": np.array_equal(g["longitude"][:], p["longitude"][:]),
            "chunk footprint": tuple(hm.chunks[-2:]) == tuple(ph.chunks[-2:]),
            "shard footprint": tuple(hm.shards[-2:]) == tuple(ph.shards[-2:]),
            "crs attrs": dict(g["crs"].attrs) == dict(p["crs"].attrs),
            "encoding attrs": all(g["hm"].attrs.get(k) == ph.attrs.get(k) for k in
                                  ("scale_factor", "add_offset", "_FillValue", "valid_range")),
            "compressor": repr(hm.compressors) == repr(ph.compressors),
        }
        for name, ok in checks.items():
            if not ok:
                bad.append(f"{tag}: {name} differs")
        print(f"  aligned with {tag}: " + ", ".join(f"{k} {'ok' if v else 'DIFFERS'}"
                                                   for k, v in checks.items()))

    # ---- every pixel of every year, bitwise against the prescription
    control_fired = None
    for yi, year in enumerate(years):
        src = Path(a.hm_dir) / f"HM_{year}_AA_1000.tiff"
        mism = collide = phantom = n_src = n_store = 0
        with rasterio.open(src) as s:
            if (s.height, s.width) != (H, W):
                bad.append(f"{src.name}: grid {(s.height, s.width)} != store {(H, W)}")
                continue
            nod = s.nodata
            for r0 in range(0, H, a.row_chunk):
                nr = min(a.row_chunk, H - r0)
                raw = s.read(1, window=Window(0, r0, W, nr)).astype(np.float32)
                want, valid = prescribed(raw, nod)
                got = np.asarray(hm[yi, r0:r0 + nr, :])
                mism += int((got != want).sum())
                collide += int(((got == U16) & valid).sum())
                phantom += int(((got != U16) & ~valid).sum())
                n_src += int(valid.sum())
                n_store += int((got != U16).sum())
                if control_fired is None:
                    naive = np.rint(np.clip(raw, 0.0, 1.0) * U16)
                    np.minimum(naive, U16 - 1, out=naive)
                    naive = np.where(np.isfinite(raw), naive, float(U16)).astype(np.uint16)
                    control_fired = int((naive != got).sum())
                del raw, want, valid, got
        print(f"    {year}: {n_store:,} px in store, {n_src:,} valid in source | "
              f"code mismatches {mism:,} | fill on real data {collide:,} | "
              f"data where source is missing {phantom:,}", flush=True)
        if mism or collide or phantom or n_src != n_store:
            bad.append(f"{year}: mismatches {mism}, collisions {collide}, phantom {phantom}, "
                       f"count {n_store} vs {n_src}")

    print(f"  control (raw-array isfinite encoding, first band): disagrees with the store on "
          f"{control_fired:,} px")
    if not control_fired:
        bad.append("CONTROL DID NOT FIRE: the naive encoding agrees with the store, so this "
                   "check could not have caught ocean written as HM = 1")

    # ---- a consumer can open it
    try:
        import xarray as xr
        s, _ = open_group(a.store)
        ds = xr.open_zarr(s.store, consolidated=False)
        v = ds["hm"]
        print(f"  xarray: {dict(v.sizes)}  dtype after CF decode {v.dtype}")
        if v.dims != DIMS:
            bad.append(f"xarray dims {v.dims}")
    except Exception as e:  # noqa: BLE001
        bad.append(f"xarray could not open the store: {type(e).__name__}: {e}")

    if bad:
        print("\nFAILED:")
        for b in bad:
            print("  -", b)
        return 1
    print("  ✓ observed store verified: every pixel of every year, aligned with "
          f"{len(a.like)} prediction store(s), opens in xarray")
    return 0


if __name__ == "__main__":
    sys.exit(main())
