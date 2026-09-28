"""Normalisation statistics over LAND pixels only, exactly, for every channel.

E2a's sidecar was estimated from random 128 px windows over the whole grid, so every
channel's moments were pulled by the ocean -- climate values and zeros that the model never
trains on -- and elevation's by an undeclared -32768 fill besides. E2c's sidecar is built by
``scripts/compute_norm_stats.py``: an exact streaming pass over each raster, keeping a pixel
only where HM is valid (the same predicate the fold mask uses, ``src.land.hm_land``) and the
channel itself is not nodata or an undeclared fill.

Each test below would fail on the obvious wrong implementation: the whole-grid reference and
the fill-included reference are computed beside the land one and asserted to differ, so the
fixture is proven to discriminate (rule 5).
"""
import json
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import compute_norm_stats as cns  # noqa: E402
import torchgeo_dataloader as tdl  # noqa: E402

H, W = 40, 30
HM_NODATA = 3.3999999521443642e38
YEARS = [2000, 2005]
VARS = ["AG", "gdp"]


def _write(path, arr, nodata=None):
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
                crs="EPSG:4326", transform=from_origin(0, 40, 1, 1), nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
    return str(path)


@pytest.fixture
def grid(tmp_path):
    rng = np.random.default_rng(0)
    land = np.zeros((H, W), bool)
    land[5:30, 3:25] = True
    land[36, 10] = True                      # one land px in the fill rows
    hm, comps = {}, {}
    for k, y in enumerate(YEARS):
        a = rng.uniform(0, 1, (H, W)) * 0.5 + 0.1 * k
        hm[y] = _write(tmp_path / f"HM_{y}.tif", np.where(land, a, HM_NODATA), HM_NODATA)
        ag = np.where(land, rng.uniform(0, 0.3, (H, W)), np.nan)
        ag[10, 10] = np.nan                  # a NaN on land is excluded, not counted as 0
        gdp = np.where(land, rng.lognormal(12, 2, (H, W)), 0.0)   # ocean filled with 0
        comps[y] = [_write(tmp_path / f"HM_{y}_AG.tif", ag),
                    _write(tmp_path / f"HM_{y}_gdp.tif", gdp, -32768.0)]
    ele = np.where(land, rng.uniform(0, 3000, (H, W)), 0.0)
    ele[35:, :] = -32768.0                   # undeclared fill, as the real raster does
    tas = np.where(land, rng.normal(15, 8, (H, W)), 25.0)
    statics = [_write(tmp_path / "hm_static_ele_1000.tiff", ele),
               _write(tmp_path / "hm_static_tas_1000.tiff", tas)]
    return dict(land=land, hm=hm, comps=comps, statics=statics, ele=ele, tas=tas)


def _build(g, block_rows=7):
    return cns.build_norm_stats(
        hm_files=[g["hm"][y] for y in YEARS], component_files=g["comps"], years=YEARS,
        hm_vars=VARS, static_files=g["statics"], land_ref=g["hm"][YEARS[-1]],
        static_nodata={"hm_static_ele_1000.tiff": -32768.0}, block_rows=block_rows,
        workers=1)


def _read(p):
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float64)


def test_static_moments_are_exact_over_land_without_the_fill(grid):
    st = _build(grid)
    land = grid["land"]
    ele_raw, tas_raw = _read(grid["statics"][0]), _read(grid["statics"][1])
    ele = ele_raw[land & (ele_raw != -32768.0)]
    assert st["static_means"][0] == pytest.approx(ele.mean(), rel=1e-12)
    assert st["static_stds"][0] == pytest.approx(ele.std() + 1e-8, rel=1e-12)
    tas = tas_raw[land]
    assert st["static_means"][1] == pytest.approx(tas.mean(), rel=1e-12)
    # The fixture discriminates: the fill-included and whole-grid moments are far away.
    assert abs(ele_raw[land].mean() - ele.mean()) > 10     # one filled px of 551 moves it 62 m
    assert abs(tas_raw.mean() - tas.mean()) > 1


def test_hm_and_component_moments_are_land_only_and_skip_land_nan(grid):
    st = _build(grid)
    hm = np.concatenate([_read(grid["hm"][y])[grid["land"]] for y in YEARS])
    assert st["hm_mean"] == pytest.approx(hm.mean(), rel=1e-12)
    assert st["hm_std"] == pytest.approx(hm.std() + 1e-8, rel=1e-12)
    ag = np.concatenate([_read(grid["comps"][y][0])[grid["land"]] for y in YEARS])
    ag = ag[np.isfinite(ag)]
    assert st["comp_means"]["AG"] == pytest.approx(ag.mean(), rel=1e-12)
    gdp = np.concatenate([_read(grid["comps"][y][1])[grid["land"]] for y in YEARS])
    assert st["comp_means"]["gdp"] == pytest.approx(gdp.mean(), rel=1e-12)
    whole = np.concatenate([_read(grid["comps"][y][1]).ravel() for y in YEARS])
    assert abs(whole.mean() - gdp.mean()) > 0.1 * gdp.mean()


def test_row_blocking_is_an_identity(grid):
    a, b = _build(grid, block_rows=3), _build(grid, block_rows=1000)
    for k in ("hm_mean", "hm_std"):
        assert a[k] == pytest.approx(b[k], rel=1e-13)
    np.testing.assert_allclose(a["static_means"], b["static_means"], rtol=1e-13)
    np.testing.assert_allclose(a["static_stds"], b["static_stds"], rtol=1e-13)


def test_sidecar_declares_its_domain_and_fill_and_loads_into_the_dataset(grid, tmp_path):
    st = _build(grid)
    assert st["stats_domain"] == "land"
    assert st["static_nodata"] == {"hm_static_ele_1000.tiff": -32768.0}
    assert st["static_files"] == ["hm_static_ele_1000.tiff", "hm_static_tas_1000.tiff"]
    p = tmp_path / "ns.json"
    p.write_text(json.dumps(st))
    ds = object.__new__(tdl.HumanFootprintChipDataset)
    ds._static_files = grid["statics"]
    ds.include_components = True
    ds._load_norm_stats(json.loads(p.read_text()))
    assert ds.static_nodata == {"hm_static_ele_1000.tiff": -32768.0}
    assert ds.static_means[0] == st["static_means"][0]


def test_provenance_counts_the_land_it_used(grid):
    st = _build(grid)
    prov = st["provenance"]["static"]["hm_static_ele_1000.tiff"]
    n_land = int(grid["land"].sum())
    assert prov["n_land"] == n_land
    assert prov["n_excluded"] == 1          # the one land pixel under the fill
    assert prov["n"] == n_land - 1


def test_grid_check_tolerates_float_noise_and_refuses_an_offset(tmp_path):
    """The real gdp, population and static rasters carry a pixel height 1.73e-18 off HM's
    0.009 deg -- the same grid in float noise. A shifted grid is a different grid."""
    def write(name, t, arr, nodata=None):
        path = tmp_path / name
        with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1,
                           dtype="float32", crs="EPSG:4326", transform=t, nodata=nodata) as d:
            d.write(arr.astype(np.float32), 1)
        return str(path)
    ref_t = rasterio.Affine(0.009, 0, -180.0, 0, -0.009, 83.997)
    noisy_t = rasterio.Affine(0.009, 0, -180.0, 0, -0.009 - 1.73e-18, 83.997)
    assert noisy_t != ref_t, "the fixture must reproduce the real mismatch"
    hm = write("hm.tif", ref_t, np.full((H, W), 0.3), HM_NODATA)
    vals = np.arange(H * W, dtype=np.float32).reshape(H, W)
    assert cns._moments((write("noisy.tif", noisy_t, vals), hm, None, 7))["n"] == H * W
    shifted_t = rasterio.Affine(0.009, 0, -180.0 + 0.009, 0, -0.009, 83.997)
    with pytest.raises(ValueError, match="not on the grid"):
        cns._moments((write("shifted.tif", shifted_t, vals), hm, None, 7))
