"""The delivered quantile-function store must carry the same numbers as its rasters.

The working format is one 64-band int16 GeoTIFF per (base year, target year), whose band
meaning lives in a text tag. The delivery format is a single icechunk array with named
dimensions. A write that transposed an axis, dropped a level, or mismatched the u grid to the
bands still opens fine and reads back plausible numbers -- so the round trip is asserted, not
assumed, and so is the labelled open, which is the only reason the format exists.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.models.quantile_spline import output_u_grid  # noqa: E402
import package_qf_icechunk as pk  # noqa: E402

H, W, NLEV = 60, 80, 64
TR = Affine(0.01, 0, -180.0, 0, -0.01, 84.0)
HINDCAST = [(2000, 2005), (2000, 2010), (2000, 2015), (2000, 2020),
            (2005, 2010), (2005, 2015), (2005, 2020),
            (2010, 2015), (2010, 2020), (2015, 2020)]


def _write_qf(path, seed, u, height=H, width=W, transform=TR):
    rng = np.random.default_rng(seed)
    base = rng.integers(-500, 500, size=(height, width)).astype(np.int32)
    step = rng.integers(1, 30, size=(height, width)).astype(np.int32)
    q = (base[None] + step[None] * np.arange(len(u))[:, None, None]).astype(np.int16)
    q[:, 3:9, 4:12] = pk.SENTINEL
    prof = dict(driver="GTiff", height=height, width=width, count=len(u), dtype="int16",
                nodata=pk.SENTINEL, crs="EPSG:4326", transform=transform)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(q)
        dst.update_tags(u_levels=",".join(repr(float(v)) for v in u),
                        scale_factor="3.0518509e-05", head_family="spline")
    return q


@pytest.fixture(scope="module")
def hindcast_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("stitched")
    u = output_u_grid(NLEV)
    truth = {}
    for i, (b, y) in enumerate(HINDCAST):
        truth[(b, y)] = _write_qf(d / f"w{b}_prediction_{y}_qf.tif", 100 + i, u)
        # decoys the discovery must ignore
        (d / f"w{b}_prediction_{y}_central.tif").write_bytes(b"")
    return d, u, truth


def _open(out):
    import icechunk
    import xarray as xr
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(str(out)))
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


def test_round_trip_and_labels(hindcast_dir, tmp_path):
    d, u, truth = hindcast_dir
    out = tmp_path / "qf.icechunk"
    assert pk.main(["--src_dir", str(d), "--mode", "hindcast", "--out", str(out)]) == 0
    ds = _open(out)
    da = ds["quantile_forecast"]
    assert da.dims == ("time", "quantile", "latitude", "longitude")
    assert da.shape == (len(HINDCAST), NLEV, H, W)
    assert list(ds["time"].values) == [y for _, y in HINDCAST]
    assert list(ds["base_year"].values) == [b for b, _ in HINDCAST]
    assert list(ds["horizon"].values) == [y - b for b, y in HINDCAST]
    assert np.array_equal(ds["quantile"].values, u)
    # pixel-centre coordinates, not corners
    assert ds["latitude"].values[0] == pytest.approx(84.0 - 0.005)
    assert ds["longitude"].values[0] == pytest.approx(-180.0 + 0.005)
    for t, (b, y) in enumerate(HINDCAST):
        assert np.array_equal(np.asarray(da.isel(time=t).values), truth[(b, y)]), f"w{b}->{y}"
    assert da.attrs["sentinel"] == pk.SENTINEL
    assert da.attrs["scale"] == pytest.approx(3.0518509e-05)


def test_forecast_mode_needs_and_uses_a_base_year(tmp_path):
    d = tmp_path / "preds"
    d.mkdir()
    u = output_u_grid(NLEV)
    for i, y in enumerate((2025, 2030, 2035, 2040)):
        _write_qf(d / f"prediction_{y}_qf_blended.tif", 200 + i, u)
    with pytest.raises(SystemExit, match="base year"):
        pk.discover(d, "forecast", None)
    out = tmp_path / "fc.icechunk"
    assert pk.main(["--src_dir", str(d), "--mode", "forecast", "--base_year", "2020",
                    "--out", str(out)]) == 0
    ds = _open(out)
    assert list(ds["time"].values) == [2025, 2030, 2035, 2040]
    assert list(ds["base_year"].values) == [2020] * 4
    assert list(ds["horizon"].values) == [5, 10, 15, 20]


def test_refuses_a_mismatched_grid(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    u = output_u_grid(NLEV)
    _write_qf(d / "w2000_prediction_2005_qf.tif", 1, u)
    _write_qf(d / "w2000_prediction_2010_qf.tif", 2, u, height=H + 1)
    with pytest.raises(SystemExit, match="does not share the grid"):
        pk.main(["--src_dir", str(d), "--mode", "hindcast", "--out", str(tmp_path / "x")])


def test_refuses_mismatched_u_levels(tmp_path):
    d = tmp_path / "badu"
    d.mkdir()
    _write_qf(d / "w2000_prediction_2005_qf.tif", 1, output_u_grid(NLEV))
    _write_qf(d / "w2000_prediction_2010_qf.tif", 2, output_u_grid(NLEV) ** 1.01)
    with pytest.raises(SystemExit, match="does not share the grid or u levels"):
        pk.main(["--src_dir", str(d), "--mode", "hindcast", "--out", str(tmp_path / "y")])


def test_refuses_to_overwrite_a_populated_store(hindcast_dir, tmp_path):
    d, _, _ = hindcast_dir
    out = tmp_path / "again.icechunk"
    pk.main(["--src_dir", str(d), "--mode", "hindcast", "--out", str(out)])
    with pytest.raises(SystemExit, match="already exists and is not empty"):
        pk.main(["--src_dir", str(d), "--mode", "hindcast", "--out", str(out)])


def test_refuses_a_directory_with_no_quantile_rasters(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "w2000_prediction_2005_central.tif").write_bytes(b"")
    with pytest.raises(SystemExit, match="no quantile-function rasters"):
        pk.discover(d, "hindcast", None)


def test_base_years_filter_ships_only_the_w2000_window(hindcast_dir, tmp_path):
    """Only w2000 reaches +20 yr, so the shipped store is cut from that window alone.

    All ten (base, target) pairs are still scored and fitted against; this asserts the
    delivery cut drops the other six and keeps the four target years intact.
    """
    d, u, truth = hindcast_dir
    out = tmp_path / "w2000.icechunk"
    assert pk.main(["--src_dir", str(d), "--mode", "hindcast", "--base_years", "2000",
                    "--out", str(out)]) == 0
    ds = _open(out)
    da = ds["quantile_forecast"]
    assert da.sizes["time"] == 4
    assert list(ds["time"].values) == [2005, 2010, 2015, 2020]
    assert list(ds["base_year"].values) == [2000] * 4
    assert list(ds["horizon"].values) == [5, 10, 15, 20]
    for t, y in enumerate((2005, 2010, 2015, 2020)):
        assert np.array_equal(np.asarray(da.isel(time=t).values), truth[(2000, y)])


def test_base_years_matching_nothing_is_an_error(hindcast_dir, tmp_path):
    d, _, _ = hindcast_dir
    with pytest.raises(SystemExit, match="matched nothing"):
        pk.discover(d, "hindcast", None, [1995])
