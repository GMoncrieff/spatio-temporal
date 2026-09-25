"""Neighbourhood-HM context rasters.

The model has never had any information about the *level* of development around a pixel — only
where past change happened, and its own HM value. These are the channels that close that gap.

Everything here is about one thing: the computation is streamed by row block with a halo, and a
halo is the single easiest place in this codebase to be quietly wrong. A block-boundary error
would produce a raster that looks entirely reasonable, trains a model, and biases every pixel
near a block edge. This project has been bitten by exactly that shape three times, so streaming
equivalence is asserted at several block sizes rather than argued for.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_hm_context import HM_SCALE, neighbourhood_stats, quantize  # noqa: E402


def reference(a, r, stat):
    """Brute force, whole array at once, no blocks."""
    from scipy.ndimage import maximum_filter, uniform_filter

    size = 2 * int(r) + 1
    if stat == "mean":
        return uniform_filter(a.astype(np.float64), size=size, mode="nearest")
    return maximum_filter(a.astype(np.float64), size=size, mode="nearest")


@pytest.mark.parametrize("stat", ["mean", "max"])
@pytest.mark.parametrize("r", [1, 3, 7])
def test_matches_a_brute_force_filter(stat, r):
    rng = np.random.default_rng(0)
    a = rng.random((64, 48)).astype(np.float32)
    got = neighbourhood_stats(a, [r], [stat], block_rows=1_000_000)[(stat, r)]
    np.testing.assert_allclose(got, reference(a, r, stat), atol=1e-5)


@pytest.mark.parametrize("stat", ["mean", "max"])
@pytest.mark.parametrize("block_rows", [4, 7, 16, 64])
def test_block_streaming_equals_one_pass(stat, block_rows):
    """The halo test. A wrong halo biases every pixel near a block edge and nothing else."""
    rng = np.random.default_rng(1)
    a = rng.random((64, 32)).astype(np.float32)
    r = 5
    one = neighbourhood_stats(a, [r], [stat], block_rows=1_000_000)[(stat, r)]
    many = neighbourhood_stats(a, [r], [stat], block_rows=block_rows)[(stat, r)]
    np.testing.assert_allclose(many, one, atol=1e-6)


def test_a_block_smaller_than_the_radius_still_works():
    """block_rows < r is the case where a naive halo silently truncates."""
    rng = np.random.default_rng(2)
    a = rng.random((40, 20)).astype(np.float32)
    one = neighbourhood_stats(a, [8], ["mean"], block_rows=1_000_000)[("mean", 8)]
    many = neighbourhood_stats(a, [8], ["mean"], block_rows=3)[("mean", 8)]
    np.testing.assert_allclose(many, one, atol=1e-6)


def test_nodata_does_not_leak_into_the_neighbourhood():
    """NaN is the validity mask everywhere in this codebase, and NaN * 0 is still NaN."""
    a = np.ones((32, 32), dtype=np.float32)
    a[10:14, 10:14] = np.nan
    out = neighbourhood_stats(a, [3], ["mean", "max"], block_rows=8)
    assert np.isfinite(out[("mean", 3)]).all(), "NaN spread into the mean"
    assert np.isfinite(out[("max", 3)]).all(), "NaN spread into the max"
    # Away from the hole the answer is still exactly 1.
    assert out[("mean", 3)][0, 0] == pytest.approx(1.0)
    assert out[("max", 3)][0, 0] == pytest.approx(1.0)


def test_max_is_at_least_mean_and_both_stay_in_range():
    rng = np.random.default_rng(3)
    a = rng.random((48, 48)).astype(np.float32)
    out = neighbourhood_stats(a, [3, 7], ["mean", "max"], block_rows=11)
    for r in (3, 7):
        assert np.all(out[("max", r)] >= out[("mean", r)] - 1e-6)
        assert out[("mean", r)].min() >= 0.0 and out[("mean", r)].max() <= 1.0
        assert out[("max", r)].min() >= 0.0 and out[("max", r)].max() <= 1.0


def test_a_larger_radius_is_smoother_and_its_max_is_larger():
    rng = np.random.default_rng(4)
    a = rng.random((64, 64)).astype(np.float32)
    out = neighbourhood_stats(a, [3, 30], ["mean", "max"], block_rows=17)
    assert out[("mean", 30)].std() < out[("mean", 3)].std()
    assert np.all(out[("max", 30)] >= out[("max", 3)] - 1e-6)


def test_int16_round_trip_is_lossless_to_the_storage_quantum():
    rng = np.random.default_rng(5)
    a = rng.random((16, 16)).astype(np.float32)
    q = quantize(a)
    assert q.dtype == np.int16
    np.testing.assert_allclose(q.astype(np.float32) * HM_SCALE, a, atol=HM_SCALE)


def test_nodata_quantizes_to_the_sentinel():
    a = np.array([[0.5, np.nan]], dtype=np.float32)
    q = quantize(a)
    assert q[0, 1] == -32768
    assert q[0, 0] != -32768
