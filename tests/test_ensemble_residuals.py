"""Phase 0 tests — rank-Gaussian transform, k-fold splits, grid-mode split filtering."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.residuals import (  # noqa: E402
    RankGaussianTransform,
    hash_uniform,
    stitch_fold_predictions,
    window_index_grid,
)


def synthetic_hm(n=200_000, p0=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.beta(0.7, 6.0, size=n)
    x[rng.random(n) < p0] = 0.0
    return np.clip(x, 0.0, 1.0)


def test_roundtrip_off_atom():
    x = synthetic_hm()
    tr = RankGaussianTransform.fit(x)
    pos = x[x > 0][:5000]
    idx = np.arange(pos.size)
    back = tr.inverse(tr.forward(pos, index=idx))
    # Interpolated empirical CDF: exact to knot resolution.
    assert np.nanmax(np.abs(back - pos)) < 5e-3


def test_zeros_roundtrip_to_exactly_zero():
    x = synthetic_hm()
    tr = RankGaussianTransform.fit(x)
    zeros = np.zeros(1000)
    z = tr.forward(zeros, index=np.arange(1000))
    back = tr.inverse(z)
    assert np.all(back == 0.0)


def test_zero_atom_quantile_matches_p0():
    x = synthetic_hm(p0=0.08)
    tr = RankGaussianTransform.fit(x)
    assert abs(tr.p0 - 0.08) < 0.01
    idx = np.arange(x.size)
    z = tr.forward(x, index=idx)
    from scipy.stats import norm

    frac_below = float((z < norm.ppf(tr.p0)).mean())
    assert abs(frac_below - tr.p0) < 0.01


def test_zero_atom_is_spread_not_a_spike():
    """A constant zero-atom image would break the zero-lag variogram; it must spread."""
    x = np.zeros(20_000)
    x[:1000] = np.linspace(0.01, 0.9, 1000)
    tr = RankGaussianTransform.fit(x)
    z = tr.forward(np.zeros(5000), index=np.arange(5000))
    assert np.unique(z).size > 1000
    assert z.std() > 0.05


def test_hash_uniform_is_deterministic_and_uniform():
    idx = np.arange(100_000)
    a = hash_uniform(idx)
    b = hash_uniform(idx)
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() < 1.0
    assert abs(a.mean() - 0.5) < 0.01


def test_same_index_gives_identical_z_for_equal_inputs():
    """obs and central both zero at a pixel must give an exactly zero residual."""
    tr = RankGaussianTransform.fit(synthetic_hm())
    idx = window_index_grid(10, 20, 4, 5, 1000)
    z1 = tr.forward(np.zeros((4, 5)), index=idx)
    z2 = tr.forward(np.zeros((4, 5)), index=idx)
    assert np.allclose(z1 - z2, 0.0)


def test_serialization_roundtrip(tmp_path):
    tr = RankGaussianTransform.fit(synthetic_hm())
    p = tmp_path / "t.json"
    tr.to_json(p)
    tr2 = RankGaussianTransform.from_json(p)
    x = np.linspace(0, 0.9, 100)
    idx = np.arange(100)
    assert np.allclose(tr.forward(x, index=idx), tr2.forward(x, index=idx), equal_nan=True)


# ---------------------------------------------------------------------------------------
def _write(path, arr, dtype="float32", nodata=None):
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
        "dtype": dtype, "crs": "EPSG:4326", "transform": from_origin(-180, 84, 0.009, 0.009),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
    return str(path)


def test_stitch_picks_the_fold_that_held_the_pixel_out(tmp_path):
    fold_mask = np.zeros((8, 8), dtype=np.uint8)
    fold_mask[:4] = 1
    fold_mask[4:] = 2
    mask_path = _write(tmp_path / "folds.tif", fold_mask, dtype="uint8")
    p1 = _write(tmp_path / "f1.tif", np.full((8, 8), 1.0))
    p2 = _write(tmp_path / "f2.tif", np.full((8, 8), 2.0))
    out = tmp_path / "stitched.tif"
    info = stitch_fold_predictions({1: p1, 2: p2}, mask_path, str(out))
    with rasterio.open(out) as s:
        got = s.read(1)
    assert np.all(got[:4] == 1.0)
    assert np.all(got[4:] == 2.0)
    assert info["n_valid_px"] == 64


def test_kfold_splits_partition_every_chip_exactly_once(tmp_path, monkeypatch):
    import importlib

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    cvm = importlib.import_module("create_validity_mask")

    valid = np.ones((cvm.CHIP_SIZE * 4, cvm.CHIP_SIZE * 5), dtype=np.uint8)
    chips = cvm.enumerate_valid_chips(valid)
    assert len(chips) == 20

    profile = {
        "driver": "GTiff", "height": valid.shape[0], "width": valid.shape[1], "count": 1,
        "dtype": "uint8", "crs": "EPSG:4326", "transform": from_origin(-180, 84, 0.009, 0.009),
    }
    monkeypatch.setattr(cvm, "FOLD_MASK_FILE", str(tmp_path / "fold.tif"))
    monkeypatch.setattr(cvm, "FOLD_MANIFEST_FILE", str(tmp_path / "fold.csv"))
    fold_mask = cvm.create_kfold_splits(valid, profile, profile["transform"], "EPSG:4326", k=5)

    counts = np.bincount(fold_mask.ravel(), minlength=6)
    assert counts[0] == 0, "every valid chip must land in exactly one fold"
    assert all(counts[f] == 4 * cvm.CHIP_SIZE ** 2 for f in range(1, 6))


def test_grid_mode_respects_the_split_mask(tmp_path):
    """Regression test for the confirmed bug: grid mode used to ignore the split mask."""
    import importlib

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    dl = importlib.import_module("torchgeo_dataloader")

    chip = 8
    split = np.zeros((32, 32), dtype=np.uint8)
    split[:8, :8] = 2          # only the top-left corner is "val"
    split[8:, :] = 1
    ds = dl.HumanFootprintChipDataset.__new__(dl.HumanFootprintChipDataset)
    ds.split_value = 2
    ds.exclude_split_values = None
    ds.mode = "grid"
    positions = ds._filter_split_positions(split, chip, chip, block=8)
    assert positions, "grid mode must return the chips overlapping the requested split"
    for i, j in positions:
        assert (split[i:i + chip, j:j + chip] == 2).any()
    assert all(i < 8 and j < 8 for i, j in positions)


def test_exclude_split_values_drops_any_touching_chip():
    import importlib

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    dl = importlib.import_module("torchgeo_dataloader")

    chip = 8
    split = np.ones((32, 32), dtype=np.uint8)
    split[12:20, 12:20] = 3          # held-out fold in the middle
    ds = dl.HumanFootprintChipDataset.__new__(dl.HumanFootprintChipDataset)
    ds.split_value = None
    ds.exclude_split_values = [3]
    ds.mode = "random"
    positions = ds._filter_split_positions(split, chip, chip, block=8)
    for i, j in positions:
        assert not (split[i:i + chip, j:j + chip] == 3).any()
    assert len(positions) == 16 - 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
