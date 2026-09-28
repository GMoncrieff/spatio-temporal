"""Slope and aspect from OUR elevation raster (scripts/prepare_terrain.py).

The grid is geographic (0.009 deg, EPSG:4326), so a pixel is 1.00 km wide at the equator and
0.50 km at 60 deg: a gradient taken in "pixels" would halve every high-latitude slope. The
metre spacing is per row, and the reference distances in these tests come from pyproj's
geodesic -- a second implementation sharing no code with the script's own spacing (rule 3).

Aspect is the downslope direction, clockwise from north, written as sin and cos because it is
circular. Where the ground is flat it is undefined, and both are 0. The -32768 fill (every row
south of 56 S) is nodata, and so is every pixel whose 3x3 window touches it. Longitude wraps:
the grid is exactly 360 deg wide, so column 0's west neighbour is the last column.
"""
import os
import sys

import numpy as np
import pytest
import rasterio
from pyproj import Geod
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import prepare_terrain as pt  # noqa: E402

RES = 0.009
GEOD = Geod(ellps="WGS84")


def _geod_dx(lat):
    return GEOD.inv(0.0, lat, RES, lat)[2]


def _geod_dy(lat):
    return GEOD.inv(0.0, lat - RES / 2, 0.0, lat + RES / 2)[2]


def _lats(top, n):
    return top - (np.arange(n) + 0.5) * RES


def test_spacing_matches_the_geodesic():
    for lat in (0.0, 30.0, 60.0, 75.0, -45.0):
        dx, dy = pt.pixel_spacing_m(np.array([lat]), RES, RES)
        assert dx[0] == pytest.approx(_geod_dx(lat), rel=2e-4)
        assert dy[0] == pytest.approx(_geod_dy(lat), rel=2e-4)


def _plane(lat0, n, grad_east=0.0, grad_north=0.0):
    """Elevation (m) of a plane with the given metric gradients, on n x n pixels."""
    lats = _lats(lat0 + n * RES / 2, n)
    x = np.array([[c * _geod_dx(lat) for c in range(n)] for lat in lats])
    y = np.cumsum([0.0] + [_geod_dy(la) for la in lats[1:]])[::-1][:, None] * np.ones((1, n))
    return grad_east * x + grad_north * y, lats


@pytest.mark.parametrize("lat0", [0.0, 60.0])
def test_a_plane_rising_east_faces_west(lat0):
    z, lats = _plane(lat0, 9, grad_east=0.05)
    slope, s, c = pt.terrain(z, lats, RES, RES, wrap=False)
    core = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(slope[core], np.degrees(np.arctan(0.05)), rtol=2e-3)
    np.testing.assert_allclose(s[core], -1.0, atol=2e-3)
    np.testing.assert_allclose(c[core], 0.0, atol=2e-3)


def test_a_plane_rising_north_faces_south():
    z, lats = _plane(30.0, 9, grad_north=0.1)
    slope, s, c = pt.terrain(z, lats, RES, RES, wrap=False)
    core = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(slope[core], np.degrees(np.arctan(0.1)), rtol=2e-3)
    np.testing.assert_allclose(s[core], 0.0, atol=2e-3)
    np.testing.assert_allclose(c[core], -1.0, atol=2e-3)


def test_the_same_pixel_rise_is_twice_as_steep_at_60_degrees():
    """Rule 12 in miniature: a slope in pixels would call these two equal."""
    z = np.tile(np.arange(9, dtype=float) * 10.0, (9, 1))
    s0 = pt.terrain(z, _lats(RES * 4.5, 9), RES, RES, wrap=False)[0][4, 4]
    s60 = pt.terrain(z, _lats(60 + RES * 4.5, 9), RES, RES, wrap=False)[0][4, 4]
    ratio = np.tan(np.radians(s60)) / np.tan(np.radians(s0))
    assert ratio == pytest.approx(_geod_dx(0.0) / _geod_dx(60.0), rel=2e-3)
    assert 1.95 < ratio < 2.05


def test_flat_ground_has_zero_slope_and_no_direction():
    slope, s, c = pt.terrain(np.full((5, 5), 250.0), _lats(10, 5), RES, RES, wrap=False)
    assert (slope == 0).all() and (s == 0).all() and (c == 0).all()


def test_the_fill_and_its_neighbours_are_nodata():
    z = np.full((7, 7), 100.0)
    z[3, 3] = -32768.0
    slope, s, c = pt.terrain(z, _lats(10, 7), RES, RES, wrap=False, fill=-32768.0)
    bad = np.zeros((7, 7), bool)
    bad[2:5, 2:5] = True
    for a in (slope, s, c):
        assert np.isnan(a[bad]).all() and np.isfinite(a[~bad]).all()


def test_longitude_wraps():
    """Rolling the raster east-west and rolling the answer back must change nothing."""
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 500, (6, 12))
    lats = _lats(20, 6)
    ref = pt.terrain(z, lats, RES, RES, wrap=True)
    rolled = pt.terrain(np.roll(z, 5, axis=1), lats, RES, RES, wrap=True)
    for a, b in zip(ref, rolled):
        np.testing.assert_array_equal(np.roll(a, 5, axis=1), b)


def _dem(tmp_path, h=23, w=40):
    """A full 360-deg-wide DEM so wrapping applies, with fill rows at the bottom."""
    rng = np.random.default_rng(2)
    z = rng.uniform(0, 3000, (h, w)).astype(np.float32)
    z[-3:, :] = -32768.0
    res = 360.0 / w
    p = tmp_path / "hm_static_ele_1000.tiff"
    with rasterio.open(p, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(-180, 80, res, res)) as d:
        d.write(z, 1)
    return p


def test_row_blocking_is_an_identity(tmp_path):
    p = _dem(tmp_path)
    a = pt.write_terrain(str(p), str(tmp_path / "a"), block_rows=4)
    b = pt.write_terrain(str(p), str(tmp_path / "b"), block_rows=1000)
    for k in a:
        with rasterio.open(a[k]) as x, rasterio.open(b[k]) as y:
            np.testing.assert_array_equal(x.read(1), y.read(1))


def test_written_rasters_declare_nan_nodata_and_keep_the_grid(tmp_path):
    p = _dem(tmp_path)
    out = pt.write_terrain(str(p), str(tmp_path / "o"), block_rows=5)
    assert set(out) == {"slope", "aspsin", "aspcos"}
    with rasterio.open(p) as src:
        grid = (src.transform, src.shape, src.crs)
    for path in out.values():
        with rasterio.open(path) as d:
            assert (d.transform, d.shape, d.crs) == grid
            assert np.isnan(d.nodata) and d.dtypes[0] == "float32"
            a = d.read(1)
            assert np.isnan(a[-4:]).all() and np.isfinite(a[:-4]).all()
