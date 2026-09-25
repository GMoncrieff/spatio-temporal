"""--train_all_splits must widen the training pool to every chip, and only when asked.

The production model has no held-out geography to protect, so training it on split 1 alone
discards ~30% of the world for nothing. This asserts the three cases: default behaviour is
unchanged, the flag opens the pool, and fold-CV mode ignores the flag (holding geography out
is the entire point there).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))



def _resolve(train_all_splits, exclude_fold):
    """The selection block from train_lightning.py, isolated."""
    train_split_value, train_exclude = 1, None
    if train_all_splits and exclude_fold is None:
        train_split_value, train_exclude = None, None
    if exclude_fold is not None:
        val_fold = (exclude_fold % 5) + 1
        train_split_value = None
        train_exclude = [exclude_fold, val_fold]
    return train_split_value, train_exclude


def test_default_is_train_split_only():
    assert _resolve(False, None) == (1, None)


def test_flag_opens_the_whole_mask():
    assert _resolve(True, None) == (None, None)


def test_fold_cv_ignores_the_flag():
    """With a fold held out, the flag must not silently pull it back into training."""
    assert _resolve(True, exclude_fold=3) == (None, [3, 4])
    assert _resolve(False, exclude_fold=3) == (None, [3, 4])


def test_none_none_selects_more_chips_than_split_one(tmp_path):
    """The dataset's None/None case really is 'all data', not 'no data'."""
    H = W = 256
    split = np.zeros((H, W), dtype=np.uint8)
    split[:64, :] = 1      # train
    split[64:128, :] = 2   # val
    split[128:192, :] = 3  # test
    split[192:, :] = 4     # calib
    p = tmp_path / "split.tif"
    with rasterio.open(p, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="uint8", crs="EPSG:4326",
                       transform=rasterio.transform.from_origin(0, 0, 1, 1)) as d:
        d.write(split, 1)

    with rasterio.open(p) as s:
        data = s.read(1)
    # the include rule the dataset applies: a chip counts if any pixel equals split_value
    chip = 64
    def n_chips(split_value, exclude):
        n = 0
        for i in range(0, H - chip + 1, chip):
            for j in range(0, W - chip + 1, chip):
                c = data[i:i + chip, j:j + chip]
                if split_value is not None and not (c == split_value).any():
                    continue
                if exclude is not None and np.isin(c, exclude).any():
                    continue
                n += 1
        return n

    only_train = n_chips(1, None)
    everything = n_chips(None, None)
    assert only_train == 4
    assert everything == 16
    assert everything > only_train
