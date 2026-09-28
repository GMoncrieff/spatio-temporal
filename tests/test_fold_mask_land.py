"""The fold mask must give fold ids to land, and to every chip that holds any.

``create_validity_mask`` read HM raw and kept ``~isnan(x) & (x >= 0)``. The HM rasters'
nodata is the FINITE 3.4e38, which passes, so the ocean was "valid" and 99.33% of the grid
got a fold id. Scores were unaffected -- the scorer masks land itself -- but every fold
predicted open ocean.

The fix reads HM through ``src.land.hm_land``. Chip selection for the folds is then a
separate choice: the builder's >= 20%-valid rule, applied to a now-correct mask, would
drop 1.78 M coastal and island land px out of every fold and so out of the hindcast
product. E2c's mask therefore keeps any chip with at least one land pixel
(``--fold_min_valid_px 1``: 14,631 chips, 35.0% of the grid, no land px lost).
"""
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import create_validity_mask as cvm  # noqa: E402

C = cvm.CHIP_SIZE
NODATA = 3.3999999521443642e38


@pytest.fixture
def hm(tmp_path):
    """4 x 4 chips. Chip (0,0) is all land, chip (1,1) is 30% land, chip (2,2) holds ONE
    land pixel, everything else is ocean at the finite nodata."""
    a = np.full((4 * C, 4 * C), NODATA, dtype=np.float32)
    a[:C, :C] = 0.2
    a[C:C + int(0.3 * C), C:2 * C] = 0.05
    a[2 * C + 5, 2 * C + 7] = 0.0            # HM exactly 0 is land
    p = tmp_path / "HM_2020_AA_1000.tiff"
    with rasterio.open(p, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(0, 10, 0.01, 0.01),
                       nodata=NODATA) as dst:
        dst.write(a, 1)
    return p, a


def test_validity_excludes_the_finite_nodata(hm):
    p, a = hm
    valid = cvm.compute_validity(str(p))[0]
    assert int(valid.sum()) == C * C + int(0.3 * C) * C + 1


def test_the_old_predicate_counted_the_ocean(hm):
    """The control: the fixture's ocean passes the predicate the builder used to apply."""
    _, a = hm
    assert int((~np.isnan(a) & (a >= 0)).sum()) == a.size


def _folds(hm, tmp_path, **kw):
    p, _ = hm
    valid, profile, transform, crs = cvm.compute_validity(str(p))
    out = tmp_path / "fold.tif"
    return cvm.create_kfold_splits(valid, profile, transform, crs, k=2, block_chips=1,
                                   out_path=str(out),
                                   manifest_path=str(tmp_path / "fold_manifest.csv"), **kw)


def test_any_land_chips_get_a_fold_and_ocean_chips_do_not(hm, tmp_path):
    _, a = hm
    fm = _folds(hm, tmp_path, min_valid_px=1)
    land = a < 1e30
    assert (fm[land] > 0).all(), "a land pixel was left without a fold"
    chip_has_land = land.reshape(4, C, 4, C).any(axis=(1, 3))
    chip_fold = fm.reshape(4, C, 4, C).max(axis=(1, 3))
    assert ((chip_fold > 0) == chip_has_land).all(), "an ocean chip got a fold id"


def test_the_default_threshold_is_unchanged(hm, tmp_path):
    """Without the flag the builder keeps its >= 20% rule: the 30% chip in, the 1 px out."""
    fm = _folds(hm, tmp_path)
    assert fm[:C, :C].min() > 0 and fm[C:2 * C, C:2 * C].min() > 0
    assert fm[2 * C:3 * C, 2 * C:3 * C].max() == 0
