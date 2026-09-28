"""Elevation's -32768 fill must not reach the model as an elevation.

``hm_static_ele_1000.tiff`` declares no nodata and fills 9.09% of the grid -- every row
south of 56 S, 558 of those pixels on land -- with -32768. E2a's statistics were sampled
with it included (mean -4200 m, std 11184 m; over land the truth is 675 m and 845 m), and
the input read passed it straight through, so those pixels entered the trunk at -2.55 sigma.
Fixing the statistics alone would make that WORSE: under land-only stats the same fill is
-39.6 sigma.

So the fix is a contract carried by the normalisation sidecar that must travel with the
checkpoints anyway: ``static_nodata`` names each static raster's undeclared fill value, and
the reader drops it before the existing NaN -> 0 (sea level, for elevation). E2a's sidecar
has no such key, so E2a reads exactly as it was trained. Both readers -- the training
dataset and the large-area prediction loop -- go through ONE function: the NaN-fill set
``{0, 4, 5, 6}`` used to be spelled out in each (rule 2).
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torchgeo_dataloader as tdl  # noqa: E402

ELE = "hm_static_ele_1000.tiff"
TAS = "hm_static_tas_1000.tiff"
DECLARED = {ELE: -32768.0}


def _ele():
    return np.array([[-32768.0, 100.0], [np.nan, 2000.0]], dtype=np.float32)


def test_declared_fill_becomes_sea_level_before_normalising():
    out = tdl.prepare_static(_ele(), 0, ELE, 675.0, 845.0, DECLARED)
    want = (np.array([[0.0, 100.0], [0.0, 2000.0]], dtype=np.float32) - 675.0) / 845.0
    np.testing.assert_allclose(out, want, rtol=0, atol=1e-6)


def test_without_a_declaration_e2a_reads_byte_identically():
    """E2a's sidecar declares nothing, and its checkpoints saw the fill."""
    mean, std = -4199.65966796875, 11184.3583984375
    out = tdl.prepare_static(_ele(), 0, ELE, mean, std, {})
    want = (np.nan_to_num(_ele(), nan=0.0) - mean) / std
    assert np.array_equal(out, want)


def test_nan_fill_stays_on_the_same_channels():
    """tas is not in the fill set: NaN there stays NaN, as before."""
    arr = np.array([[np.nan, 20.0]], dtype=np.float32)
    out = tdl.prepare_static(arr, 1, TAS, 13.0, 13.0, DECLARED)
    assert np.isnan(out[0, 0]) and out[0, 1] == np.float32((20.0 - 13.0) / 13.0)


def test_the_sidecar_carries_the_declaration_through_the_dataset():
    ds = object.__new__(tdl.HumanFootprintChipDataset)
    ds._static_files = [os.path.join("x", os.path.basename(f)) for f in tdl.static_files]
    ds.include_components = False
    stats = dict(hm_mean=0.1, hm_std=0.2, static_means=[0.0] * 7, static_stds=[1.0] * 7,
                 static_files=[os.path.basename(f) for f in tdl.static_files],
                 static_nodata=DECLARED)
    ds._load_norm_stats(stats)
    assert ds.static_nodata == DECLARED
    assert ds.norm_stats_dict()["static_nodata"] == DECLARED


def test_an_undeclared_sidecar_loads_as_no_fill():
    ds = object.__new__(tdl.HumanFootprintChipDataset)
    ds._static_files = list(tdl.static_files)
    ds.include_components = False
    ds._load_norm_stats(dict(hm_mean=0.1, hm_std=0.2, static_means=[0.0] * 7,
                             static_stds=[1.0] * 7))
    assert ds.static_nodata == {}


def test_both_static_readers_go_through_prepare_static():
    """Rule 27: a unit test of the function is not a test of the path. Both call sites must
    use it, and the fill set must not survive as a second spelling anywhere."""
    for rel in ("scripts/torchgeo_dataloader.py", "scripts/train_lightning.py"):
        src = open(os.path.join(ROOT, rel)).read()
        assert "prepare_static(" in src, f"{rel} does not call prepare_static"
        assert not re.search(r"nan_to_zero_static\s*=", src), f"{rel} still spells the fill set"
