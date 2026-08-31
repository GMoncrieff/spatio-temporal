#!/usr/bin/env python3
"""Pack the model's quantile-function rasters into one icechunk repository.

The ConvLSTM's forecast *is* ``Q_h(u | x)``: a full per-pixel quantile function, written by
the prediction path as one 64-band int16 GeoTIFF per (base year, target year). That is the
right working format -- every downstream stage reads it by window -- and the wrong delivery
format: sixty-four bands whose meaning lives in a text tag, spread over ten files, is not
something a consumer can open.

This writes them as a single array with named dimensions instead:

    quantile_forecast  (time, quantile, latitude, longitude)   int16, scale 3.0518509e-05

``xarray.open_zarr`` on the session store gives back a labelled DataArray with real
coordinates: ``time`` is the target year, ``quantile`` the u level each band stands for,
``latitude``/``longitude`` the pixel centres. ``base_year`` and ``horizon`` ride along the
time axis as auxiliary coordinates, because a hindcast entry is a (base, target) pair -- the
same target year is produced by several forecast origins and they are different forecasts.

Values are kept as stored int16 rather than dequantized: HM is bounded on [0, 1] and the
quantum is 3.05e-05, far below any quantity of interest, and float32 would triple the store.
``sentinel = -32768`` marks ocean and unpredicted pixels; ``value = int16 * scale``.

Chunks are ``(1, n_levels, 512, 512)`` -- one time step, ALL quantile levels, a 512 px tile.
The access pattern this format exists for is "give me this pixel's distribution", and that is
then a single chunk read rather than sixty-four.

    python scripts/package_qf_icechunk.py --src_dir <stitched> --mode hindcast --out <repo>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARRAY_NAME = "quantile_forecast"
SENTINEL = -32768
SPATIAL_CHUNK = 512
HINDCAST_RE = re.compile(r"^w(?P<base>\d{4})_prediction_(?P<year>\d{4})_qf\.tif$")
FORECAST_RE = re.compile(r"^prediction_(?P<year>\d{4})_qf(?:_blended)?\.tif$")


def discover(src_dir: Path, mode: str, base_year: int | None, keep_bases=None):
    """[(base_year, target_year, path)] sorted by (base, target).

    ``keep_bases`` restricts the store to particular forecast origins. The hindcast produces
    ten (base, target) pairs and every one of them is used for scoring and for fitting the
    residual spectrum, but only w2000 reaches +20 yr, so that is the window the long-term
    product is cut from -- the others exist to measure with, not to ship.
    """
    rx = HINDCAST_RE if mode == "hindcast" else FORECAST_RE
    rows = []
    for p in sorted(src_dir.iterdir()):
        m = rx.match(p.name)
        if not m:
            continue
        b = int(m.group("base")) if "base" in m.groupdict() and m.groupdict().get("base") \
            else base_year
        if b is None:
            raise SystemExit(f"{p.name} carries no base year; pass --base_year")
        rows.append((b, int(m.group("year")), p))
    if not rows:
        raise SystemExit(f"no quantile-function rasters matching {rx.pattern} under {src_dir}")
    if keep_bases:
        keep = set(int(x) for x in keep_bases)
        dropped = sorted({b for b, _, _ in rows} - keep)
        rows = [r for r in rows if r[0] in keep]
        if not rows:
            raise SystemExit(f"--base_years {sorted(keep)} matched nothing under {src_dir}")
        if dropped:
            print(f"  base years kept {sorted(keep)}, dropped {dropped}")
    return sorted(rows, key=lambda r: (r[0], r[1]))


def read_grid(path: Path):
    with rasterio.open(path) as s:
        tags = s.tags()
        if "u_levels" not in tags:
            raise SystemExit(f"{path} carries no u_levels tag; not a quantile-function raster")
        u = np.array([float(v) for v in tags["u_levels"].split(",")], dtype=np.float64)
        if np.diff(u).min() <= 0:
            raise SystemExit(f"{path}: u levels are not strictly increasing")
        if s.count != u.size:
            raise SystemExit(f"{path}: {s.count} bands against {u.size} u levels")
        return dict(u=u, H=s.height, W=s.width, transform=s.transform, crs=s.crs,
                    scale=float(tags.get("scale_factor", 1.0 / 32767.0)),
                    nodata=s.nodata, dtype=s.dtypes[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out", required=True, help="icechunk repository path (must not exist)")
    ap.add_argument("--mode", choices=["hindcast", "forecast"], required=True)
    ap.add_argument("--base_year", type=int, default=None,
                    help="forecast mode: the base year of the forward window (e.g. 2020). "
                         "Forecast rasters carry no base year in their filename.")
    ap.add_argument("--base_years", default=None,
                    help="comma-separated forecast origins to keep (e.g. '2000'). The "
                         "hindcast has ten (base, target) pairs and all ten are scored and "
                         "fitted against, but only w2000 reaches +20 yr, so the shipped "
                         "store is cut from that window. Default: keep everything found.")
    ap.add_argument("--block_rows", type=int, default=SPATIAL_CHUNK,
                    help="rows per read/write pass; one block is n_levels x rows x W int16")
    ap.add_argument("--verify_windows", type=int, default=24)
    ap.add_argument("--verify_seed", type=int, default=42)
    args = ap.parse_args(argv)

    import icechunk
    import zarr

    src_dir, out = Path(args.src_dir), Path(args.out)
    keep_bases = ([int(x) for x in args.base_years.split(",")]
                  if args.base_years else None)
    rows = discover(src_dir, args.mode, args.base_year, keep_bases)
    g = read_grid(rows[0][2])
    Q, H, W = int(g["u"].size), int(g["H"]), int(g["W"])
    T = len(rows)
    for b, y, p in rows[1:]:
        gg = read_grid(p)
        if (gg["H"], gg["W"]) != (H, W) or not np.array_equal(gg["u"], g["u"]):
            raise SystemExit(f"{p} does not share the grid or u levels of {rows[0][2]}")
        if gg["transform"] != g["transform"]:
            raise SystemExit(f"{p}: transform differs from {rows[0][2]}")

    print(f"=== packaging {T} quantile-function rasters ===")
    for b, y, p in rows:
        print(f"  w{b} -> {y}  (h={y - b:2d})  {p.name}")
    print(f"  grid {H} x {W}, {Q} levels, scale {g['scale']:.8g}, CRS {g['crs']}")
    print(f"  array {ARRAY_NAME}(time={T}, quantile={Q}, latitude={H}, longitude={W}) int16")

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is not empty — remove it first")
    out.mkdir(parents=True, exist_ok=True)
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(out)))
    session = repo.writable_session("main")
    root = zarr.create_group(session.store)

    tr = g["transform"]
    lat = tr.f + (np.arange(H) + 0.5) * tr.e
    lon = tr.c + (np.arange(W) + 0.5) * tr.a

    def coord(name, values, dims, dtype, **attrs):
        a = root.create_array(name, shape=(len(values),), chunks=(len(values),),
                              dtype=dtype, dimension_names=dims)
        a[:] = np.asarray(values, dtype=dtype)
        a.attrs.update(attrs)
        return a

    coord("time", [y for _, y, _ in rows], ("time",), "i4",
          long_name="target year of the forecast", units="year")
    coord("base_year", [b for b, _, _ in rows], ("time",), "i4",
          long_name="last observed year the forecast was issued from", units="year")
    coord("horizon", [y - b for b, y, _ in rows], ("time",), "i4",
          long_name="lead time", units="year")
    coord("quantile", g["u"], ("quantile",), "f8",
          long_name="cumulative probability level u of Q(u)")
    coord("latitude", lat, ("latitude",), "f8",
          long_name="latitude of pixel centre", units="degrees_north")
    coord("longitude", lon, ("longitude",), "f8",
          long_name="longitude of pixel centre", units="degrees_east")

    arr = root.create_array(
        ARRAY_NAME, shape=(T, Q, H, W),
        chunks=(1, Q, SPATIAL_CHUNK, SPATIAL_CHUNK),
        dtype="i2", fill_value=SENTINEL,
        dimension_names=("time", "quantile", "latitude", "longitude"),
    )
    arr.attrs.update({
        "long_name": "predicted quantile function of Human Modification",
        "description": "Q(u) for each pixel, horizon and probability level. "
                       "value = int16 * scale; int16 == sentinel means no prediction.",
        "scale": g["scale"], "offset": 0.0, "sentinel": SENTINEL,
        "units": "1", "valid_range": [0.0, 1.0],
        "coordinates": "base_year horizon",
        "grid_mapping": "spatial_ref",
    })
    root.attrs.update({
        "title": f"Distributional ConvLSTM {args.mode} quantile functions",
        "model": "e1 (spline head, neighbourhood-HM covariate, seed 46)",
        "mode": args.mode,
        "crs_wkt": g["crs"].to_wkt(), "epsg": g["crs"].to_epsg(),
        "transform": [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f],
        "sources": {f"w{b}_{y}": str(p) for b, y, p in rows},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
    })

    t0 = time.time()
    for t, (b, y, p) in enumerate(rows):
        with rasterio.open(p) as s:
            for r0 in range(0, H, args.block_rows):
                rr = min(args.block_rows, H - r0)
                blk = s.read(window=Window(0, r0, W, rr))
                if s.nodata is not None and int(s.nodata) != SENTINEL:
                    blk = np.where(blk == s.nodata, SENTINEL, blk).astype(np.int16)
                arr[t, :, r0:r0 + rr, :] = blk
                del blk
        print(f"  ✓ [{t + 1}/{T}] w{b} {y}  ({(time.time() - t0) / 60:.1f} min elapsed)")

    snapshot = session.commit(f"{args.mode} quantile functions: {T} steps, {Q} levels")
    print(f"  committed snapshot {snapshot}")

    # Prove the store rather than assume it: a write that transposed an axis or dropped a
    # level still opens fine and reads back plausible numbers.
    print("--- verifying against the source rasters ---")
    ro = repo.readonly_session("main")
    zr = zarr.open_group(ro.store, mode="r")
    za = zr[ARRAY_NAME]
    if za.shape != (T, Q, H, W):
        raise SystemExit(f"stored shape {za.shape} != {(T, Q, H, W)}")
    rng = np.random.default_rng(args.verify_seed)
    checked = 0
    for _ in range(args.verify_windows):
        t = int(rng.integers(0, T))
        r0 = int(rng.integers(0, max(1, H - 256)))
        c0 = int(rng.integers(0, max(1, W - 256)))
        with rasterio.open(rows[t][2]) as s:
            want = s.read(window=Window(c0, r0, 256, 256))
        got = np.asarray(za[t, :, r0:r0 + 256, c0:c0 + 256])
        if not np.array_equal(got, want):
            n = int((got != want).sum())
            raise SystemExit(f"store differs from {rows[t][2].name} at t={t} "
                             f"row {r0} col {c0}: {n} of {want.size} values")
        checked += int((want != SENTINEL).sum())
    if not np.array_equal(np.asarray(zr["quantile"][:]), g["u"]):
        raise SystemExit("stored u levels differ from the rasters'")
    if za.attrs["scale"] != g["scale"]:
        raise SystemExit("stored scale differs from the rasters'")
    print(f"  ✓ {args.verify_windows} windows, {checked:,} predicted values identical")
    print(f"  ✓ u grid and scale match")

    # And prove it opens as a labelled dataset, which is the whole point of the format.
    try:
        import xarray as xr
        ds = xr.open_zarr(ro.store, consolidated=False)
        da = ds[ARRAY_NAME]
        assert da.dims == ("time", "quantile", "latitude", "longitude"), da.dims
        print(f"  ✓ xarray dims {da.dims}, "
              f"time {list(ds.time.values)[:4]}{'...' if T > 4 else ''}, "
              f"{ds.sizes['quantile']} levels")
    except ImportError:
        print("  · xarray not installed; skipped the labelled-open check")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n✓ {out}  ({size / 1e9:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
