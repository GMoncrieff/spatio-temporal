"""The fold stitcher, which is what turns k per-fold rasters into one deliverable raster.

Two modes with very different meanings, so both are pinned here: ``holdout`` must take each
pixel from the fold that held it out (any other choice makes the hindcast in-sample and the
scorecard meaningless), and ``mean`` must average every fold that has a finite value there.
The mean mode's NaN handling is the subtle one -- a fold whose prediction is NaN at a pixel
must be skipped, not counted as a zero, or the average is silently dragged toward zero.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.stitch import HORIZONS, QUANTILES, stitch_fold_predictions

H, W = 40, 60
K = 5


def _write(path, arr, transform=None, width=None, height=None):
    transform = transform if transform is not None else from_origin(0, 0, 1, 1)
    profile = dict(driver="GTiff", height=height or arr.shape[0], width=width or arr.shape[1],
                   count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                   nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
    return str(path)


@pytest.fixture
def folds(tmp_path):
    """Fold f predicts the constant f everywhere; the mask hands out vertical stripes."""
    mask = np.zeros((H, W), dtype="uint8")
    for f in range(1, K + 1):
        mask[:, (f - 1) * (W // K):f * (W // K)] = f
    mask_path = _write(tmp_path / "fold_mask.tif", mask.astype("float32"))
    paths = {f: _write(tmp_path / f"fold{f}.tif", np.full((H, W), float(f))) for f in range(1, K + 1)}
    return paths, mask_path, mask


def test_channel_order_constants():
    assert HORIZONS == (5, 10, 15, 20)
    assert QUANTILES == ("lower", "central", "upper")


def test_holdout_takes_each_pixel_from_its_own_fold(folds, tmp_path):
    paths, mask_path, mask = folds
    out = tmp_path / "holdout.tif"
    info = stitch_fold_predictions(paths, mask_path, str(out), block_rows=7, mode="holdout")

    with rasterio.open(out) as src:
        got = src.read(1)
        assert np.isnan(src.nodata)
    # Every pixel carries its own fold id; mask==0 stays NaN.
    covered = mask > 0
    assert np.array_equal(got[covered], mask[covered].astype("float32"))
    assert np.isnan(got[~covered]).all()
    assert info["n_valid_px"] == int(covered.sum())
    assert info["mode"] == "holdout"


def test_mean_averages_every_fold_at_every_pixel(folds, tmp_path):
    paths, mask_path, _ = folds
    out = tmp_path / "mean.tif"
    info = stitch_fold_predictions(paths, mask_path, str(out), block_rows=7, mode="mean")

    with rasterio.open(out) as src:
        got = src.read(1)
    # mean(1..5) == 3, everywhere -- including where the mask is 0, because mean mode
    # ignores the mask entirely. That is why it is in-sample at every pixel.
    assert np.allclose(got, 3.0)
    assert info["n_valid_px"] == H * W


def test_mean_skips_nan_contributors_rather_than_counting_them(tmp_path):
    mask_path = _write(tmp_path / "m.tif", np.ones((H, W), dtype="float32"))
    a = np.full((H, W), 2.0)
    b = np.full((H, W), 4.0)
    b[0, 0] = np.nan          # one fold has no prediction at this pixel
    paths = {1: _write(tmp_path / "f1.tif", a), 2: _write(tmp_path / "f2.tif", b)}

    out = tmp_path / "mean_nan.tif"
    stitch_fold_predictions(paths, mask_path, str(out), block_rows=13, mode="mean")
    with rasterio.open(out) as src:
        got = src.read(1)

    assert got[0, 0] == pytest.approx(2.0)   # not 1.0, which is what treating NaN as 0 gives
    assert np.allclose(got[1:], 3.0)


def test_block_size_does_not_change_the_result(folds, tmp_path):
    paths, mask_path, _ = folds
    reads = []
    for block_rows in (1, 7, H, H * 3):
        out = tmp_path / f"b{block_rows}.tif"
        stitch_fold_predictions(paths, mask_path, str(out), block_rows=block_rows, mode="holdout")
        with rasterio.open(out) as src:
            reads.append(src.read(1))
    for r in reads[1:]:
        assert np.array_equal(np.nan_to_num(reads[0], nan=-1), np.nan_to_num(r, nan=-1))


def test_geometry_mismatch_raises(folds, tmp_path):
    paths, mask_path, _ = folds
    paths = dict(paths)
    paths[3] = _write(tmp_path / "wrong_shape.tif", np.full((H, W + 4), 3.0))
    with pytest.raises(ValueError, match="geometry differs"):
        stitch_fold_predictions(paths, mask_path, str(tmp_path / "x.tif"), mode="holdout")


def test_unknown_mode_raises(folds, tmp_path):
    paths, mask_path, _ = folds
    with pytest.raises(ValueError, match="unknown stitch mode"):
        stitch_fold_predictions(paths, mask_path, str(tmp_path / "x.tif"), mode="median")
