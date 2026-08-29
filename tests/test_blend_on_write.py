"""Blending one raster at a time must give byte-identical values to blending them all first.

The prediction path used to build a dict of every horizon x quantile level before writing any
of them. There are `len(active_horizons) * (3 + n_qf_levels)` of those -- 268 for a
four-horizon window at 64 levels -- and on Africa each is a 63.1 Mpx float32 array, 0.252 GB.
The dict is therefore 67.6 GB, and because `np.full` touches every page it is all resident,
while the accumulators (`np.zeros`) stay sparse over ocean. Measured: ~98 GB peak for one
fold, and two folds in parallel were OOM-killed by the kernel with no traceback. On southern
Africa the same structure is 2 GB and nothing shows.

The fix evaluates the same expression at write time instead. This pins the property that makes
that safe: identical output, including where the mask is empty and where the accumulated
weight is zero.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _blend_all_then_write(keys, accum, wsum, shape):
    """The old shape: build every blended raster up front, then consume."""
    m = wsum > 0
    out = {}
    for key in keys:
        o = np.full(shape, np.nan, dtype=np.float32)
        o[m] = (accum[key][m] / wsum[m]).astype(np.float32)
        o[m] = np.clip(o[m], 0.0, 1.0)
        out[key] = o
    return [out[k] for k in keys]


def _blend_on_write(keys, accum, wsum, shape):
    """The new shape: blend each raster at the moment it is written."""
    m = wsum > 0

    def blend(key):
        o = np.full(shape, np.nan, dtype=np.float32)
        o[m] = (accum[key][m] / wsum[m]).astype(np.float32)
        o[m] = np.clip(o[m], 0.0, 1.0)
        return o

    return [blend(k) for k in keys]


def _case(rng, shape, weight_fill):
    keys = [f"h{h}_{q}" for h in range(2) for q in ("lower", "central", "upper", "qf000")]
    accum = {k: rng.normal(size=shape).astype(np.float32) * 3.0 for k in keys}
    wsum = weight_fill(rng, shape)
    return keys, accum, wsum


def test_identical_on_a_typical_field():
    rng = np.random.default_rng(0)
    shape = (37, 53)
    keys, accum, wsum = _case(
        rng, shape, lambda r, s: r.random(s).astype(np.float32))
    a = _blend_all_then_write(keys, accum, wsum, shape)
    b = _blend_on_write(keys, accum, wsum, shape)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True)


def test_identical_where_some_pixels_have_no_weight():
    """Unwritten pixels stay NaN; that is how nodata reaches the raster."""
    rng = np.random.default_rng(1)
    shape = (41, 29)

    def holes(r, s):
        w = r.random(s).astype(np.float32)
        w[r.random(s) < 0.4] = 0.0          # tiles never covered these
        return w

    keys, accum, wsum = _case(rng, shape, holes)
    a = _blend_all_then_write(keys, accum, wsum, shape)
    b = _blend_on_write(keys, accum, wsum, shape)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True)
        assert np.isnan(y[wsum == 0]).all(), "zero-weight pixels must remain NaN"


def test_identical_when_nothing_was_covered():
    """An empty mask must not divide by zero in either form."""
    rng = np.random.default_rng(2)
    shape = (13, 17)
    keys, accum, wsum = _case(rng, shape, lambda r, s: np.zeros(s, dtype=np.float32))
    a = _blend_all_then_write(keys, accum, wsum, shape)
    b = _blend_on_write(keys, accum, wsum, shape)
    for x, y in zip(a, b):
        assert np.isnan(x).all() and np.isnan(y).all()


def test_clamp_is_applied_in_both():
    """Values outside [0, 1] are clipped; the clip must not move across the refactor."""
    shape = (8, 8)
    keys = ["h0_central"]
    accum = {"h0_central": np.full(shape, 5.0, dtype=np.float32)}
    wsum = np.ones(shape, dtype=np.float32)
    a = _blend_all_then_write(keys, accum, wsum, shape)[0]
    b = _blend_on_write(keys, accum, wsum, shape)[0]
    assert np.array_equal(a, b)
    assert (b == 1.0).all(), "5.0 must clamp to 1.0"


def test_the_shipped_source_no_longer_builds_the_dict():
    """Guard the actual fix, not just an equivalent written in this file."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "train_lightning.py")).read()
    assert "out_horizons" not in src, "the pre-built blend dict is back"
    assert "def _blend(key):" in src, "blend-on-write helper missing"
