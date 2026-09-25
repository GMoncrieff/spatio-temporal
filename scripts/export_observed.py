#!/usr/bin/env python
"""Observed overall HM as an icechunk store laid out like the prediction stores.

    python scripts/export_observed.py \
        --out /mnt/hdd1/spatio-temporal/data/conv_spline/products/observed/observed_hm.icechunk \
        --like /mnt/hdd1/spatio-temporal/data/conv_spline/products/E2a/hindcast/hindcast_qf.icechunk

One array ``hm`` over (year, latitude, longitude), years 1990-2020 every five, read from
``HM_{year}_AA_1000.tiff`` -- the overall HM, the same file the scorer treats as "observed".
There is no percentile axis: an observation is a value, not a distribution.

WHAT IS THE SAME AS THE PREDICTION STORES, AND WHY IT IS COPIED RATHER THAN RESTATED

Grid, latitude/longitude, the ``crs`` variable, the uint16 encoding, the fill value, the
compressor, and the chunk and shard FOOTPRINT on the map are all read off an existing
prediction store (``--like``) or imported from ``export_products``. A second spelling of any of
them is how two stores meant to be read together come to disagree (CLAUDE.md rule 2). The
spatial footprint in particular is copied so every observed chunk covers exactly the
latitude/longitude window of a prediction chunk: comparing a prediction with its observation
reads the same tile of both.

Consequence worth knowing: a chunk here is 256 x 320 x 2 B = 160 KiB, not the prediction
stores' 10 MB, because what made those 10 MB was the 64-level percentile axis this store does
not have. Sharding keeps the file count to 544 shards per year regardless.

WHAT IS DELIBERATELY DIFFERENT

``dimension_names`` is declared on every array. The prediction stores do not declare it, and
``xarray.open_zarr`` refuses them with "Zarr object is missing the `dimension_names` metadata"
-- measured 2026-09-25 on the delivered E2a hindcast store. Copying that omission would make
this store unreadable by xarray for no reason.

THE NODATA TRAP

The HM rasters declare nodata = 3.4e38 and do NOT fill with NaN. 3.4e38 is finite, so the
prediction exporter's ``np.where(np.isfinite(q), ...)`` would clip every ocean pixel to 1.0 and
encode it 65534 -- "fully modified" -- with no error. The read therefore goes through the
scorer's own ``_read_like_band``, which maps the sentinel (and anything below -1e6) to NaN:
"observed" then means here exactly what it means in every scorecard.

ENCODING

Identical to ``export_products.write_icechunk``: ``min(rint(clip(x, 0, 1) * 65535), 65534)``,
65535 = missing, computed in the source's own float32 so a given HM value gets the same code
in this store as in a prediction store. ``_read_like_band`` returns float64; float32 ->
float64 -> float32 is exact, so casting back loses nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from export_products import U16_MAX, check_extent  # noqa: E402
from score_distributional_model import HM_DIR, _read_like_band  # noqa: E402

YEARS = (1990, 1995, 2000, 2005, 2010, 2015, 2020)


def source_path(year: int, hm_dir: Path = HM_DIR) -> Path:
    """The overall-HM raster for one year. The scorer's ``path_observed`` spelling."""
    return Path(hm_dir) / f"HM_{year}_AA_1000.tiff"


def encode_u16(x: np.ndarray) -> np.ndarray:
    """float32 HM with NaN where missing -> uint16 codes, the prediction stores' encoding.

    Written as the same expression ``write_icechunk`` evaluates, on the same dtype.
    """
    if x.dtype != np.float32:
        raise TypeError(f"encode in the source's float32, got {x.dtype}")
    code = np.rint(np.clip(x, 0.0, 1.0) * U16_MAX)
    np.minimum(code, U16_MAX - 1, out=code)
    return np.where(np.isfinite(x), code, float(U16_MAX)).astype("uint16")


def read_observed(path: Path, grid: dict, r0: int, nr: int) -> np.ndarray:
    """Rows ``[r0, r0+nr)`` as float32, NaN wherever the scorer would call it missing."""
    return _read_like_band(str(path), grid, r0, nr).astype(np.float32)


def like_layout(like: Path) -> dict:
    """Everything this store copies off a delivered prediction store."""
    import icechunk
    import zarr

    s = icechunk.Repository.open(icechunk.local_filesystem_storage(str(like))).readonly_session("main")
    g = zarr.open_group(s.store, mode="r")
    hm = g["hm"]
    if hm.ndim != 4:
        raise SystemExit(f"--like {like}: expected a 4-D prediction store, hm is {hm.shape}")
    return {
        "chunk_ll": tuple(hm.chunks[-2:]),
        "shard_ll": tuple(hm.shards[-2:]) if hm.shards else None,
        "fill": int(hm.metadata.fill_value),
        "hm_attrs": dict(hm.attrs),
        "compressors": hm.compressors,
        "latitude": np.asarray(g["latitude"][:]),
        "longitude": np.asarray(g["longitude"][:]),
        "coord_attrs": {k: dict(g[k].attrs) for k in ("latitude", "longitude")},
        "crs_attrs": dict(g["crs"].attrs),
        "shape_ll": tuple(hm.shape[-2:]),
    }


def write_store(out: Path, years, like: dict, grid: dict, row_chunk: int,
                hm_dir: Path, overwrite: bool) -> list[dict]:
    import icechunk
    import zarr

    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    H, W = grid["height"], grid["width"]
    if (H, W) != like["shape_ll"]:
        raise SystemExit(f"source grid {(H, W)} != prediction store grid {like['shape_ll']}")
    t = grid["transform"]
    lat = (t.f + (np.arange(H) + 0.5) * t.e).astype("float64")
    lon = (t.c + (np.arange(W) + 0.5) * t.a).astype("float64")
    # Computed the way write_icechunk computes them, then REQUIRED to equal what the
    # prediction store holds: alignment is asserted, not assumed.
    if not (np.array_equal(lat, like["latitude"]) and np.array_equal(lon, like["longitude"])):
        raise SystemExit("latitude/longitude differ from the --like store; the stores would not align")

    c_lat, c_lon = like["chunk_ll"]
    s_lat, s_lon = like["shard_ll"] or like["chunk_ll"]
    n_shards = len(years) * -(-H // s_lat) * -(-W // s_lon)
    print(f"  chunks (year=1, lat={c_lat}, lon={c_lon}) = {c_lat * c_lon * 2 / 1024:.0f} KiB")
    print(f"  shards (year=1, lat={s_lat}, lon={s_lon}) = {s_lat * s_lon * 2 / 1024 / 1024:.1f} MiB"
          f"  -> {n_shards:,} shard files")

    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(out)))
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="w")

    root.create_array("year", shape=(len(years),), dtype="int32", chunks=(len(years),),
                      dimension_names=("year",))[:] = np.asarray(years, "int32")
    root.create_array("latitude", shape=(H,), dtype="float64", chunks=(min(H, 65536),),
                      dimension_names=("latitude",))[:] = lat
    root.create_array("longitude", shape=(W,), dtype="float64", chunks=(min(W, 65536),),
                      dimension_names=("longitude",))[:] = lon

    hm = root.create_array(
        "hm", shape=(len(years), H, W), dtype="uint16",
        chunks=(1, c_lat, c_lon), shards=(1, s_lat, s_lon),
        fill_value=like["fill"], compressors=like["compressors"],
        dimension_names=("year", "latitude", "longitude"))

    attrs = dict(like["hm_attrs"])
    attrs["long_name"] = "Human Modification, observed (overall HM, AA)"
    hm.attrs.update(attrs)
    root["year"].attrs.update({"long_name": "observation year"})
    for k in ("latitude", "longitude"):
        root[k].attrs.update(like["coord_attrs"][k])

    crs = root.create_array("crs", shape=(), dtype="int32", chunks=())
    crs[()] = 0
    crs.attrs.update(like["crs_attrs"])
    root.attrs.update({
        "title": "HM observed",
        "mode": "observed",
        "years": list(map(int, years)),
        "source": "HM_{year}_AA_1000.tiff -- overall Human Modification",
        "source_dir": str(Path(hm_dir).resolve()),
        "Conventions": "CF-1.10",
        "north_bound": float(grid["bounds"].top), "south_bound": float(grid["bounds"].bottom),
    })

    stats = []
    for yi, year in enumerate(years):
        src = source_path(year, hm_dir)
        n = 0; lo = np.inf; hi = -np.inf; below = 0; above = 0; clamped = 0
        for r0 in range(0, H, row_chunk):
            nr = min(row_chunk, H - r0)
            x = read_observed(src, grid, r0, nr)
            fin = np.isfinite(x)
            if fin.any():
                v = x[fin]
                n += int(v.size); lo = min(lo, float(v.min())); hi = max(hi, float(v.max()))
                below += int((v < 0).sum()); above += int((v > 1).sum())
            enc = encode_u16(x)
            clamped += int((enc == U16_MAX - 1).sum())
            hm[yi, r0:r0 + nr, :] = enc
            del x, enc
        s = {"year": year, "valid_px": n, "min": lo, "max": hi,
             "below_0": below, "above_1": above, "at_max_code": clamped}
        stats.append(s)
        print(f"    {year}: {src.name}  {n:,} px  range [{lo:.6f}, {hi:.6f}]  "
              f"<0: {below:,}  >1: {above:,}  code 65534: {clamped:,}", flush=True)

    root.attrs["valid_px_per_year"] = {str(s["year"]): s["valid_px"] for s in stats}
    session.commit(f"observed overall HM (AA), years {','.join(map(str, years))}")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the .icechunk directory to write")
    ap.add_argument("--like", required=True,
                    help="a delivered prediction store to copy grid, chunking and encoding from")
    ap.add_argument("--years", default=",".join(map(str, YEARS)))
    ap.add_argument("--hm_dir", default=str(HM_DIR))
    # 1024, not the prediction exporter's 512: equal to the shard height, so every write
    # fills whole shards and none is read back to be rewritten.
    ap.add_argument("--row_chunk", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    years = [int(y) for y in a.years.split(",")]
    for y in years:
        if not source_path(y, a.hm_dir).exists():
            raise SystemExit(f"missing source {source_path(y, a.hm_dir)}")

    first = source_path(years[0], a.hm_dir)
    check_extent(first)
    with rasterio.open(first) as s:
        grid = {"height": s.height, "width": s.width, "transform": s.transform,
                "crs": s.crs, "bounds": s.bounds}
    for y in years[1:]:
        with rasterio.open(source_path(y, a.hm_dir)) as s:
            if (s.height, s.width, s.transform) != (grid["height"], grid["width"], grid["transform"]):
                raise SystemExit(f"{source_path(y, a.hm_dir)} is on a different grid from {first}")

    like = like_layout(Path(a.like))
    print(f"=== observed HM -> {a.out}")
    stats = write_store(Path(a.out), years, like, grid, a.row_chunk, Path(a.hm_dir), a.overwrite)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
