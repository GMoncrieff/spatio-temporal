#!/usr/bin/env python
"""The published deliverables: five COGs per year, plus one icechunk store per product.

Two formats, because they answer different questions.

  COG        single-band float32, cloud-optimised, BIGTIFF. What a GIS opens and what streams
             over HTTP. Five variables per year: lower, mean, upper, gt01, gt04.
  icechunk   the full quantile function as one labelled array
             (year, percentile, latitude, longitude), uint16. Sixty-four GeoTIFF bands whose
             meaning lives in a text tag is a working format, not a delivery one.

WHAT IS DERIVED AND WHAT IS PASSED THROUGH

  lower  Q(0.025)      passed through from the published triple
  mean   E[Q]          passed through -- the central band is the MEAN, not the median
  upper  Q(0.975)      passed through
  gt01   P(HM > 0.1)   derived: 1 - F(0.1), F inverted off the quantile function
  gt04   P(HM > 0.4)   derived: 1 - F(0.4)

The two exceedance rasters use the same interpolation convention as the scorer's ``pit`` --
one definition, imported, not re-implemented, because a second spelling of an inverse CDF is
how two readers of the same raster come to disagree.

PROJECTION AND EXTENT

The source grid is already EPSG:4326, 40000 x 17111 at 0.009 deg, bounds N 83.9970 to
S -70.0020. The requested 84.00 N / 70.00 S is that grid to within a third of a pixel, so
nothing is reprojected or resampled: a warp here would cost accuracy to satisfy a rounding.
``--assert_extent`` refuses a source whose bounds fall outside the requested box by more than
one pixel, so a future grid change cannot slip through silently.

HINDCAST YEARS

The model consumes a three-year window (t-10, t-5, t) and predicts t+5 ... t+20. HM exists
for 1990-2020 only, so the earliest usable window is (1990, 1995, 2000) and the earliest
hindcast targets are 2005, 2010, 2015, 2020. Years 2000-2015 would need base 1995, i.e.
HM_1985, which does not exist.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from score_distributional_model import pit, qf_levels, read_qf  # noqa: E402

VARIABLES = ("lower", "mean", "upper", "gt01", "gt04")
#: The two exceedance thresholds, in HM. Named once; the variable name is derived from them.
THRESHOLDS = {"gt01": 0.1, "gt04": 0.4}
#: Which published band each passed-through variable comes from.
PASSTHROUGH = {"lower": "lower", "mean": "central", "upper": "upper"}

REQ_NORTH, REQ_SOUTH = 84.0, -70.0

COG_CREATION = [
    "-co", "COMPRESS=DEFLATE",
    # 3, not 2: these are float32 rasters and PREDICTOR=2 is the integer predictor. The
    # floating-point predictor is what shrinks them.
    "-co", "PREDICTOR=3",
    "-co", "BLOCKSIZE=512",
    "-co", "OVERVIEW_RESAMPLING=AVERAGE",
    "-co", "BIGTIFF=YES",
    "-co", "NUM_THREADS=ALL_CPUS",
]

#: uint16 over [0, 1]: 1/65535 = 1.5e-5, half the int16 convention this project already
#: called lossless for HM and far below any quantity of interest.
U16_MAX = 65535


def src_paths(src_dir: Path, mode: str, base_year: int, year: int) -> dict:
    """Where the stitched hindcast / forward forecast writes each band."""
    if mode == "hindcast":
        stem = f"w{base_year}_prediction_{year}"
        return {q: src_dir / f"{stem}_{q}.tif" for q in ("lower", "central", "upper", "qf")}
    stem = f"prediction_{year}"
    return {q: src_dir / f"{stem}_{q}_blended.tif" for q in ("lower", "central", "upper", "qf")}


def check_extent(path: Path, tol_px: float = 1.0) -> dict:
    with rasterio.open(path) as s:
        b, t = s.bounds, s.transform
        px = abs(t.e)
        if str(s.crs).upper() != "EPSG:4326":
            raise SystemExit(f"{path}: CRS is {s.crs}, expected EPSG:4326")
        if (b.top - REQ_NORTH) > tol_px * px or (REQ_SOUTH - b.bottom) > tol_px * px:
            raise SystemExit(
                f"{path}: bounds N {b.top:.4f} S {b.bottom:.4f} fall outside the requested "
                f"{REQ_NORTH} / {REQ_SOUTH} box by more than {tol_px} pixel. Clip or rewarp "
                f"deliberately rather than letting the export decide.")
        return {"width": s.width, "height": s.height, "transform": t, "crs": s.crs,
                "bounds": b, "px": px}


# ------------------------------------------------------------------ exceedance rasters

def write_exceedance(qf_path: Path, out_paths: dict, row_chunk: int) -> None:
    """``P(HM > x)`` per pixel, streamed in row bands.

    Both thresholds come out of one pass over the quantile raster: it is the expensive read
    (64 float32 bands) and reading it twice to write two rasters doubles the only cost that
    matters here.
    """
    with rasterio.open(qf_path) as src:
        prof = src.profile.copy()
        H = src.height
    prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate",
                predictor=3, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
    writers = {k: rasterio.open(p, "w", **prof) for k, p in out_paths.items()}
    try:
        for r0 in range(0, H, row_chunk):
            nr = min(row_chunk, H - r0)
            u, q = read_qf(str(qf_path), r0, nr)
            flat = q.reshape(q.shape[0], -1)
            ok = np.isfinite(flat).all(axis=0)
            for name, thr in THRESHOLDS.items():
                out = np.full(flat.shape[1], np.nan, dtype=np.float32)
                if ok.any():
                    y = np.full(int(ok.sum()), float(thr))
                    out[ok] = (1.0 - pit(u, flat[:, ok], y)).astype(np.float32)
                writers[name].write(out.reshape(nr, -1), 1, window=((r0, r0 + nr), (0, q.shape[2])))
            del u, q, flat, ok
    finally:
        for w in writers.values():
            w.close()


def to_cog(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gdal_translate", "-of", "COG", *COG_CREATION, str(src), str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"gdal_translate failed for {src}:\n{r.stderr.strip()[-2000:]}")


def verify_cog(path: Path) -> str:
    """A COG is not a COG because the driver was asked for it. Read the layout back."""
    r = subprocess.run(["gdalinfo", "-json", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"gdalinfo failed on {path}")
    info = json.loads(r.stdout)
    bands = info.get("bands", [])
    if len(bands) != 1:
        raise SystemExit(f"{path}: {len(bands)} bands, expected 1")
    if bands[0]["type"] != "Float32":
        raise SystemExit(f"{path}: dtype {bands[0]['type']}, expected Float32")
    md = info.get("metadata", {}).get("IMAGE_STRUCTURE", {})
    if md.get("LAYOUT") != "COG":
        raise SystemExit(f"{path}: IMAGE_STRUCTURE LAYOUT is {md.get('LAYOUT')!r}, not COG")
    ov = len(bands[0].get("overviews", []))
    return f"COG, 1 band Float32, {ov} overview levels, {Path(path).stat().st_size/1e6:.1f} MB"


# ------------------------------------------------------------------ icechunk

def chunk_plan(n_lat, n_lon, n_pct, target_mb=10.0, shard_factor=4):
    """Chunks of ~``target_mb`` with the percentile axis WHOLE, then shards over them.

    Percentile is never chunked: a consumer asking "what is the distribution here" wants all
    64 levels of one pixel, and splitting that axis turns one read into several. The axis is
    64 x 2 bytes = 128 B per pixel, so a 10 MB chunk still covers ~81k pixels of map.

    Sharding is what keeps the file count sane. Chunks alone would put ~33,500 files on disk
    for a four-year global store; a 4x4 shard cuts that to ~2,176 without changing the read
    granularity, because a shard is one file holding many independently-addressable chunks.
    """
    per_px = n_pct * 2                      # uint16
    px_per_chunk = int(target_mb * 1024 * 1024 / per_px)
    side = int(np.sqrt(px_per_chunk))
    c_lat = max(64, min(n_lat, (side // 32) * 32))
    c_lon = max(64, min(n_lon, int(px_per_chunk / c_lat // 32) * 32))
    s_lat = min(n_lat, c_lat * shard_factor)
    s_lon = min(n_lon, c_lon * shard_factor)
    # A shard must be a whole number of chunks on every axis.
    s_lat = max(c_lat, (s_lat // c_lat) * c_lat)
    s_lon = max(c_lon, (s_lon // c_lon) * c_lon)
    return (c_lat, c_lon), (s_lat, s_lon)


def write_icechunk(out_path: Path, years, src_dir: Path, mode: str, base_year: int,
                   row_chunk: int, target_mb: float, shard_factor: int,
                   grid: dict, overwrite: bool):
    import icechunk
    import zarr

    if out_path.exists():
        if not overwrite:
            raise SystemExit(f"{out_path} exists; pass --overwrite")
        import shutil
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_qf = src_paths(src_dir, mode, base_year, years[0])["qf"]
    u = qf_levels(str(first_qf))[0]
    n_pct = len(u)
    H, W = grid["height"], grid["width"]
    (c_lat, c_lon), (s_lat, s_lon) = chunk_plan(H, W, n_pct, target_mb, shard_factor)
    chunk_mb = n_pct * c_lat * c_lon * 2 / 1024 / 1024
    shard_mb = chunk_mb * (s_lat // c_lat) * (s_lon // c_lon)
    n_shards = len(years) * -(-H // s_lat) * -(-W // s_lon)
    print(f"  chunks (year=1, percentile={n_pct}, lat={c_lat}, lon={c_lon}) = {chunk_mb:.2f} MB")
    print(f"  shards (year=1, percentile={n_pct}, lat={s_lat}, lon={s_lon}) = {shard_mb:.1f} MB"
          f"  -> {n_shards:,} shard files")

    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(out_path)))
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="w")

    t = grid["transform"]
    lat = (t.f + (np.arange(H) + 0.5) * t.e).astype("float64")
    lon = (t.c + (np.arange(W) + 0.5) * t.a).astype("float64")

    root.create_array("year", shape=(len(years),), dtype="int32",
                      chunks=(len(years),)).__setitem__(slice(None), np.asarray(years, "int32"))
    # The u-grid in PERCENT, which is what "percentile" means to a consumer.
    root.create_array("percentile", shape=(n_pct,), dtype="float64",
                      chunks=(n_pct,)).__setitem__(slice(None), (u * 100.0).astype("float64"))
    root.create_array("latitude", shape=(H,), dtype="float64",
                      chunks=(min(H, 65536),)).__setitem__(slice(None), lat)
    root.create_array("longitude", shape=(W,), dtype="float64",
                      chunks=(min(W, 65536),)).__setitem__(slice(None), lon)

    hm = root.create_array(
        "hm", shape=(len(years), n_pct, H, W), dtype="uint16",
        chunks=(1, n_pct, c_lat, c_lon), shards=(1, n_pct, s_lat, s_lon),
        fill_value=U16_MAX,
        compressors=[zarr.codecs.BloscCodec(cname="zstd", clevel=5,
                                            shuffle=zarr.codecs.BloscShuffle.shuffle)])

    hm.attrs.update({
        "long_name": "Human Modification, predicted quantile function",
        "scale_factor": 1.0 / U16_MAX, "add_offset": 0.0,
        "_FillValue": U16_MAX,
        "valid_range": [0, U16_MAX - 1],
        "comment": ("uint16 over [0, 1]: multiply by scale_factor to get HM. "
                    f"{U16_MAX} is the fill value, so the representable maximum is "
                    f"{(U16_MAX - 1) / U16_MAX:.6f}."),
        "grid_mapping": "crs",
    })
    for name, attrs in [
        ("year", {"long_name": "target year", "base_year": base_year}),
        ("percentile", {"long_name": "quantile level", "units": "percent",
                        "comment": "the published u-grid; 2.5, 50 and 97.5 are exact levels"}),
        ("latitude", {"units": "degrees_north", "standard_name": "latitude"}),
        ("longitude", {"units": "degrees_east", "standard_name": "longitude"}),
    ]:
        root[name].attrs.update(attrs)

    crs = root.create_array("crs", shape=(), dtype="int32", chunks=())
    crs[()] = 0
    crs.attrs.update({"grid_mapping_name": "latitude_longitude", "epsg_code": "EPSG:4326",
                      "spatial_ref": str(grid["crs"].to_wkt()),
                      "GeoTransform": " ".join(str(v) for v in t.to_gdal())})
    root.attrs.update({
        "title": f"HM {mode} quantile functions",
        "mode": mode, "base_year": base_year, "years": list(map(int, years)),
        "Conventions": "CF-1.10",
        "north_bound": float(grid["bounds"].top), "south_bound": float(grid["bounds"].bottom),
    })

    for yi, year in enumerate(years):
        qp = src_paths(src_dir, mode, base_year, year)["qf"]
        print(f"    {year}: {qp.name}", flush=True)
        for r0 in range(0, H, row_chunk):
            nr = min(row_chunk, H - r0)
            _, q = read_qf(str(qp), r0, nr)
            # 65535 is the FILL VALUE, so no real measurement may be allowed to land on it.
            # `rint(clip(q,0,1) * 65535)` gives exactly 65535 for any HM >= 0.99999237, and
            # E1v's upper tail is exponential on -log(1-HM) support, so the far upper
            # quantiles saturate at HM = 1 over a large share of land. MEASURED on the first
            # export: 31.7% of land at the 99.99th percentile of the 2040 forecast, and 24.5%
            # of the 2020 hindcast, were real values written as "missing". The attributes
            # already declared valid_range [0, 65534] and a representable maximum of
            # 65534/65535 -- the contract was right and the code did not implement it.
            code = np.rint(np.clip(q, 0.0, 1.0) * U16_MAX)
            np.minimum(code, U16_MAX - 1, out=code)
            enc = np.where(np.isfinite(q), code, float(U16_MAX))
            hm[yi, :, r0:r0 + nr, :] = enc.astype("uint16")
            del q, code, enc

    session.commit(f"{mode} quantile functions, base {base_year}, years "
                   f"{','.join(map(str, years))}")
    return {"chunk_mb": chunk_mb, "shard_mb": shard_mb, "n_shards": n_shards,
            "shape": [len(years), n_pct, H, W]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--mode", choices=["hindcast", "forecast"], required=True)
    ap.add_argument("--base_year", type=int, required=True)
    ap.add_argument("--years", required=True,
                    help="comma-separated target years; hindcast 2005,2010,2015,2020, "
                         "forecast 2025,2030,2035,2040")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--row_chunk", type=int, default=512)
    ap.add_argument("--chunk_mb", type=float, default=10.0)
    ap.add_argument("--shard_factor", type=int, default=4)
    ap.add_argument("--skip_cogs", action="store_true")
    ap.add_argument("--skip_icechunk", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    years = [int(y) for y in a.years.split(",")]
    src_dir, out = Path(a.src_dir), Path(a.out_dir)
    cog_dir = out / "cogs"
    cog_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for y in years for p in src_paths(src_dir, a.mode, a.base_year, y).values()
               if not p.exists()]
    if missing:
        raise SystemExit("missing source rasters:\n  " + "\n  ".join(missing))

    grid = check_extent(src_paths(src_dir, a.mode, a.base_year, years[0])["qf"])
    grid = {**grid, "height": grid["height"], "width": grid["width"]}
    print(f"grid  EPSG:4326  {grid['width']} x {grid['height']} @ {grid['px']}deg  "
          f"N {grid['bounds'].top:.4f} S {grid['bounds'].bottom:.4f}")

    if not a.skip_cogs:
        print("\n=== COGs ===")
        tmp = out / "_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        for year in years:
            sp = src_paths(src_dir, a.mode, a.base_year, year)
            ex = {k: tmp / f"hm_{year}_{k}.raw.tif" for k in THRESHOLDS}
            print(f"  {year}: deriving {', '.join(THRESHOLDS)} ...", flush=True)
            write_exceedance(sp["qf"], ex, a.row_chunk)
            for var in VARIABLES:
                dst = cog_dir / f"hm_{year}_{var}.tif"
                if dst.exists() and not a.overwrite:
                    raise SystemExit(f"{dst} exists; pass --overwrite")
                src = ex[var] if var in THRESHOLDS else sp[PASSTHROUGH[var]]
                to_cog(src, dst)
                print(f"    hm_{year}_{var}.tif  {verify_cog(dst)}")
            for p in ex.values():
                p.unlink(missing_ok=True)
        tmp.rmdir()

    if not a.skip_icechunk:
        print(f"\n=== icechunk ({a.mode}) ===")
        store = out / f"{a.mode}_qf.icechunk"
        info = write_icechunk(store, years, src_dir, a.mode, a.base_year, a.row_chunk,
                              a.chunk_mb, a.shard_factor, grid, a.overwrite)
        sz = sum(f.stat().st_size for f in store.rglob("*") if f.is_file())
        print(f"  wrote {store}  shape {info['shape']}  {sz/1e9:.2f} GB on disk")

    print("\ndone")


if __name__ == "__main__":
    main()
