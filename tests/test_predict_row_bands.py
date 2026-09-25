"""The row-banding plan for large-area prediction.

accum_horizons is len(active_horizons) * (3 + n_qf_levels) full-window float32 arrays -- 268
for a four-horizon window at 64 quantile levels, each 2.55 GiB on the 17111 x 40000 global
grid. Measured resident (touched 4 KiB pages, not the virtual size) is 185 GiB for one global
fold on a 125 GB box that runs two folds at once, so the global run bands the rows.

The invariant that makes a banded run equal an unbanded one is entirely in the plan: every
output row is kept by exactly one band, and every band accumulates every tile that touches
one of its kept rows. If either fails, the seam rows blend on incomplete weights -- which the
prediction code's own measurement puts at 3.1e-3, the size of the signal, while writing
perfectly plausible numbers.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_lightning import plan_row_bands  # noqa: E402

TILE = 128
GRIDS = [
    (0, 895, 64),        # southern Africa, the A/B extent
    (0, 17111, 64),      # the global grid
    (0, 1024, 64),       # exact multiple of the band size
    (137, 2003, 64),     # region not starting at row 0
    (0, 300, 128),       # stride == tile, no overlap
    (0, 700, 256),       # stride > tile: the unbanded path leaves gaps too
]
CHUNKS = [256, 512, 1024, 4096]


def _unbanded_starts(r0, r1, stride):
    return list(range(r0, r1, stride))


@pytest.mark.parametrize("r0,r1,stride", GRIDS)
@pytest.mark.parametrize("chunk", CHUNKS)
def test_every_row_kept_exactly_once(r0, r1, stride, chunk):
    bands = plan_row_bands(r0, r1, stride, TILE, chunk)
    covered = []
    for a, b, _, _, _ in bands:
        covered.extend(range(a, b))
    assert covered == list(range(r0, r1)), "kept rows must tile [r0, r1) with no gap or overlap"


@pytest.mark.parametrize("r0,r1,stride", GRIDS)
@pytest.mark.parametrize("chunk", CHUNKS)
def test_band_accumulates_every_tile_touching_a_kept_row(r0, r1, stride, chunk):
    all_starts = _unbanded_starts(r0, r1, stride)
    for a, b, acc0, acc1, starts in plan_row_bands(r0, r1, stride, TILE, chunk):
        needed = {i for i in all_starts if i < b and i + TILE > a}
        assert set(starts) == needed, f"band [{a},{b}) is missing tiles {needed - set(starts)}"
        assert acc0 <= min(starts), "accumulator starts after the first tile it must hold"
        assert acc1 >= b, "accumulator ends before the last kept row"
        assert acc1 <= r1 and acc0 >= r0, "accumulator escapes the region"


@pytest.mark.parametrize("r0,r1,stride", GRIDS)
def test_chunk_zero_is_the_unbanded_path(r0, r1, stride):
    bands = plan_row_bands(r0, r1, stride, TILE, 0)
    assert len(bands) == 1
    a, b, acc0, acc1, starts = bands[0]
    assert (a, b) == (r0, r1)
    assert (acc0, acc1) == (r0, r1)
    assert starts == _unbanded_starts(r0, r1, stride)


@pytest.mark.parametrize("r0,r1,stride", GRIDS)
@pytest.mark.parametrize("chunk", CHUNKS)
def test_accumulator_height_is_bounded_by_the_chunk_plus_two_tiles(r0, r1, stride, chunk):
    # This is the whole point: the accumulator must not scale with the region.
    for a, b, acc0, acc1, _ in plan_row_bands(r0, r1, stride, TILE, chunk):
        assert acc1 - acc0 <= chunk + 2 * max(TILE, stride)


def test_band_size_must_land_on_the_block_grid():
    # A band boundary off the output raster's 256-row block grid makes the windowed write a
    # read-modify-write of a compressed block, which is how a silently corrupt seam happens.
    with pytest.raises(ValueError, match="multiple of 256"):
        plan_row_bands(0, 1000, 64, TILE, 300)


@pytest.mark.parametrize("chunk,n_bands,first_acc,worst_acc,two_fold_gib",
                         [(512, 34, 576, 640, 51.1), (1024, 17, 1088, 1152, 92.0)])
def test_global_grid_sizes_the_production_run(chunk, n_bands, first_acc, worst_acc,
                                             two_fold_gib):
    # The global grid is 17111 x 40000 and the w2000 window has 268 accumulators. These are
    # the numbers the production run is budgeted against, dense worst case -- np.zeros stays
    # sparse over ocean, so the resident figure is lower.
    bands = plan_row_bands(0, 17111, 64, TILE, chunk)
    assert len(bands) == n_bands
    assert bands[0][:4] == (0, chunk, 0, first_acc)
    assert bands[-1][1] == 17111
    # The first band starts at row 0 so it has no halo above it; every later band carries
    # one stride of halo on each side, and that is the band to budget against.
    worst = max(b[3] - b[2] for b in bands)
    assert worst == worst_acc
    gib = 268 * worst * 40000 * 4 / 2**30
    assert 2 * gib == pytest.approx(two_fold_gib, abs=1.0), f"{2 * gib:.1f} GiB for two folds"
    assert 2 * gib < 100.0, "two concurrent folds must leave headroom on a 125 GB box"
