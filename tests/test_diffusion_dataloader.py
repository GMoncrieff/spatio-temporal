"""
Tests for the delta_20yr target mode of HumanFootprintChipDataset.

These tests require the full HM raster set (1990–2020 × all components + statics).
They are skipped automatically on hosts where the data is not available.
"""
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch
from torchgeo_dataloader import (
    get_dataloader,
    hm_files,
    component_files,
    static_files,
)

ALL_REQUIRED = list(hm_files) + list(static_files)
for year_files in component_files.values():
    ALL_REQUIRED.extend(year_files)
DATA_AVAILABLE = all(os.path.exists(f) for f in ALL_REQUIRED)
SMALL_REGION = PROJECT_ROOT / "config" / "region_to_predict_small.geojson"

pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="HM raster set incomplete on this host; run on training machine",
)


def _make_loader(**kwargs):
    defaults = dict(
        batch_size=2,
        chip_size=64,
        chips_per_epoch=4,
        target_mode="delta_20yr",
        stat_samples=128,
    )
    defaults.update(kwargs)
    return get_dataloader(**defaults)


def test_delta_mode_shapes():
    loader = _make_loader()
    batch = next(iter(loader))
    ds = loader.dataset
    assert batch["input_dynamic"].shape == (2, 3, ds.C_dyn, 64, 64)
    assert batch["input_static"].shape == (2, ds.C_static, 64, 64)
    assert batch["lonlat"].shape == (2, 64, 64, 2)
    assert batch["target_dhm"].shape == (2, 1, 64, 64)
    assert batch["hm_t_normalized"].shape == (2, 1, 64, 64)
    assert batch["valid_mask"].shape == (2, 64, 64)
    assert batch["valid_mask"].dtype == torch.bool


def test_delta_target_finite_in_valid_mask():
    loader = _make_loader()
    batch = next(iter(loader))
    target = batch["target_dhm"][:, 0]
    mask = batch["valid_mask"]
    assert torch.isfinite(target[mask]).all(), "non-finite target in valid_mask"


def test_dhm_normalization_roughly_zero_mean():
    loader = _make_loader(batch_size=8, chips_per_epoch=8, stat_samples=512)
    batch = next(iter(loader))
    target = batch["target_dhm"][:, 0]
    mask = batch["valid_mask"]
    if not mask.any():
        pytest.skip("no valid pixels in this batch")
    mean = target[mask].float().mean().item()
    assert abs(mean) < 1.0, f"normalized Δhm batch mean {mean} is far from zero"


@pytest.mark.skipif(not SMALL_REGION.exists(), reason="small-region GeoJSON missing")
def test_region_restriction_filters_chips():
    no_restriction = _make_loader().dataset
    with_restriction = _make_loader(restrict_to_region=str(SMALL_REGION)).dataset
    # Without a split mask or region, valid_split_positions stays None.
    assert no_restriction.valid_split_positions is None
    # With a region, it becomes a bounded non-empty list.
    assert with_restriction.valid_split_positions is not None
    assert len(with_restriction.valid_split_positions) > 0
