"""The observed-HM icechunk export: encoding, the nodata trap, and a store xarray can open.

Every guard here is paired with the control that would slip past it without the guard.
"""
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from export_observed import encode_u16, read_observed, write_store, source_path  # noqa: E402
from export_products import U16_MAX  # noqa: E402

SENTINEL = np.float32(3.4e38)


def _prediction_exporter_expression(q):
    """The three lines write_icechunk evaluates, copied verbatim as the reference."""
    code = np.rint(np.clip(q, 0.0, 1.0) * U16_MAX)
    np.minimum(code, U16_MAX - 1, out=code)
    return np.where(np.isfinite(q), code, float(U16_MAX)).astype("uint16")


def test_encoding_matches_prediction_exporter_bitwise_in_float32():
    # Dense sweep plus the edges that matter: 0, 1, the fill-collision threshold, NaN.
    x = np.concatenate([
        np.linspace(-0.1, 1.1, 2_000_003, dtype=np.float32),
        np.float32([0.0, 1.0, 0.99999237, 0.9999924, (U16_MAX - 0.5) / U16_MAX, np.nan]),
    ])
    assert np.array_equal(encode_u16(x), _prediction_exporter_expression(x))


def test_float64_input_is_refused():
    # Computing the code in float64 disagrees with the float32 encoder on ~6e-5 of values
    # (measured by verify_products.py), so the same HM would get different codes in the
    # observed and predicted stores.
    with pytest.raises(TypeError):
        encode_u16(np.array([0.5], dtype=np.float64))


def test_one_is_clamped_off_the_fill_value_and_nan_is_the_fill_value():
    out = encode_u16(np.float32([0.0, 1.0, np.nan]))
    assert out.tolist() == [0, U16_MAX - 1, U16_MAX]
    # Control: without the clamp, HM = 1.0 lands ON the fill value and reads as missing.
    assert int(np.rint(np.float32(1.0) * U16_MAX)) == U16_MAX


def _write_hm(path, arr, nodata=SENTINEL):
    h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(-180.0, 84.0, 0.009, 0.009),
                       nodata=float(nodata)) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_sentinel_nodata_becomes_missing_not_fully_modified(tmp_path):
    arr = np.full((4, 6), SENTINEL, np.float32)
    arr[1:3, 1:5] = np.float32([[0.0, 0.25, 0.5, 1.0], [0.1, 0.2, 0.3, 0.4]])
    p = tmp_path / "HM_2000_AA_1000.tiff"
    _write_hm(p, arr)
    with rasterio.open(p) as s:
        grid = {"transform": s.transform, "width": s.width}
    x = read_observed(p, grid, 0, 4)
    enc = encode_u16(x)
    ocean = arr == SENTINEL
    assert (enc[ocean] == U16_MAX).all()
    assert enc[1, 4] == U16_MAX - 1          # a real 1.0 stays data
    # Control: the prediction exporter's isfinite() test on the RAW array treats 3.4e38 as
    # finite, clips it to 1.0 and writes every ocean pixel as fully modified.
    naive = _prediction_exporter_expression(arr)
    assert (naive[ocean] == U16_MAX - 1).all()


def test_end_to_end_store_opens_in_xarray_with_dims_and_decodes(tmp_path):
    xr = pytest.importorskip("xarray")
    icechunk = pytest.importorskip("icechunk")
    zarr = pytest.importorskip("zarr")

    H, W, years = 40, 64, [1990, 1995]
    rng = np.random.default_rng(0)
    grid_t = from_origin(-180.0, 84.0, 0.009, 0.009)
    truth = {}
    for y in years:
        a = rng.uniform(0, 1, (H, W)).astype(np.float32)
        a[:5] = SENTINEL
        a[10, 10] = 1.0
        truth[y] = a
        _write_hm(source_path(y, tmp_path), a)
    with rasterio.open(source_path(years[0], tmp_path)) as s:
        grid = {"height": s.height, "width": s.width, "transform": s.transform,
                "crs": s.crs, "bounds": s.bounds}
    lat = (grid_t.f + (np.arange(H) + 0.5) * grid_t.e).astype("float64")
    lon = (grid_t.c + (np.arange(W) + 0.5) * grid_t.a).astype("float64")
    like = {
        "chunk_ll": (8, 16), "shard_ll": (16, 32), "fill": U16_MAX,
        "hm_attrs": {"long_name": "x", "scale_factor": 1.0 / U16_MAX, "add_offset": 0.0,
                     "_FillValue": U16_MAX, "valid_range": [0, U16_MAX - 1],
                     "grid_mapping": "crs"},
        "compressors": [zarr.codecs.BloscCodec(cname="zstd", clevel=5,
                                               shuffle=zarr.codecs.BloscShuffle.shuffle)],
        "latitude": lat, "longitude": lon,
        "coord_attrs": {"latitude": {"units": "degrees_north"},
                        "longitude": {"units": "degrees_east"}},
        "crs_attrs": {"grid_mapping_name": "latitude_longitude"},
        "shape_ll": (H, W),
    }
    out = tmp_path / "obs.icechunk"
    # row_chunk 12 is deliberately NOT a multiple of the shard height, so partial-shard
    # rewrites are exercised too.
    stats = write_store(out, years, like, grid, 12, tmp_path, overwrite=False)
    assert [s["valid_px"] for s in stats] == [(H - 5) * W] * 2

    sess = icechunk.Repository.open(icechunk.local_filesystem_storage(str(out))).readonly_session("main")
    ds = xr.open_zarr(sess.store, consolidated=False)
    assert ds["hm"].dims == ("year", "latitude", "longitude")
    assert ds["year"].values.tolist() == years
    got = ds["hm"].values                     # CF-decoded: float, NaN where fill
    for yi, y in enumerate(years):
        a = truth[y]
        ocean = a == SENTINEL
        assert np.isnan(got[yi][ocean]).all()
        assert np.isfinite(got[yi][~ocean]).all()
        # within one code everywhere; exactly the clamped maximum at the real 1.0
        assert np.nanmax(np.abs(got[yi][~ocean] - a[~ocean])) <= 1.0 / U16_MAX + 1e-9
        assert got[yi][10, 10] == pytest.approx((U16_MAX - 1) / U16_MAX, abs=1e-9)
