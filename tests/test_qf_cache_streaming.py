"""The ensemble's quantile-function cache must not read the raster whole.

``generate_ensemble.load_quantile_functions`` compacts a 64-band quantile-function raster to
its valid pixels. It used to do that with ``src.read()`` -- 64 x 684.4 Mpx x 2 B = 87.6 GB on
the global grid, plus a 23.6 GB gather, against 125 GB of DRAM. On Africa's 63.1 Mpx grid the
same call is 8.1 GB, which is why it survived every regional run.

Two things are asserted: the compacted result is bit-identical to the whole-read it replaces,
and the read is actually per band -- an edit that restored ``src.read()`` would still pass the
first test while reinstating the defect.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.models.quantile_spline import output_u_grid  # noqa: E402
import generate_ensemble as ge  # noqa: E402

H, W, NLEV = 40, 50, 64


@pytest.fixture(scope="module")
def qf_raster(tmp_path_factory):
    d = tmp_path_factory.mktemp("qf")
    u = output_u_grid(NLEV)
    rng = np.random.default_rng(3)
    base = rng.integers(-1000, 1000, size=(H, W)).astype(np.int32)
    step = rng.integers(1, 40, size=(H, W)).astype(np.int32)
    q = (base[None] + step[None] * np.arange(NLEV)[:, None, None]).astype(np.int16)
    q[:, 5:9, 6:11] = -32768                       # a nodata block
    p = d / "w2000_prediction_2005_qf.tif"
    prof = dict(driver="GTiff", height=H, width=W, count=NLEV, dtype="int16",
                nodata=-32768, crs="EPSG:4326",
                transform=Affine(0.01, 0, 10.0, 0, -0.01, 5.0))
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(q)
        dst.update_tags(u_levels=",".join(repr(float(v)) for v in u),
                        scale_factor="3.0518509e-05", head_family="spline")
    return str(p), q, u


@pytest.mark.parametrize("frac", [1.0, 0.31, 0.02])
def test_matches_the_whole_read_it_replaces(qf_raster, tmp_path, frac):
    path, q, u = qf_raster
    rng = np.random.default_rng(7)
    flat = np.flatnonzero(q[0].ravel() != -32768)
    idx = np.sort(rng.choice(flat, size=max(1, int(len(flat) * frac)), replace=False))
    idx = idx.astype(np.int64)

    out, u_ref = ge.load_quantile_functions({2005: path}, [2005], tmp_path, idx)
    got = np.load(out[2005], mmap_mode="r")

    want = q.reshape(NLEV, -1)[:, idx]      # exactly the expression that was replaced
    assert got.shape == want.shape
    assert got.dtype == want.dtype == np.int16
    assert np.array_equal(np.asarray(got), want)
    assert np.array_equal(u_ref, u)


def test_read_is_per_band_not_whole_raster(qf_raster, tmp_path, monkeypatch):
    path, q, _ = qf_raster
    idx = np.arange(0, H * W, 3, dtype=np.int64)
    calls = []
    real = rasterio.DatasetReader.read

    def spy(self, *a, **kw):
        if self.count == NLEV:
            calls.append((a[0] if a else kw.get("indexes")))
        return real(self, *a, **kw)

    monkeypatch.setattr(rasterio.DatasetReader, "read", spy)
    ge.load_quantile_functions({2005: path}, [2005], tmp_path, idx)
    assert calls, "the quantile raster was never read"
    assert all(c is not None for c in calls), \
        "a whole-raster src.read() is back: that is 87.6 GB on the global grid"
    assert sorted(calls) == list(range(1, NLEV + 1)), \
        f"expected one read per band, got {sorted(calls)[:5]}..."


def test_rejects_a_non_increasing_u_grid(tmp_path):
    # A degenerate u grid makes a segment slope 0/0 downstream and CRPS NaN for every pixel --
    # a metric failure that reads exactly like a model failure. Fail where the cause is.
    p = tmp_path / "bad_qf.tif"
    prof = dict(driver="GTiff", height=4, width=4, count=3, dtype="int16", nodata=-32768,
                crs="EPSG:4326", transform=Affine(0.01, 0, 0, 0, -0.01, 0))
    with rasterio.open(p, "w", **prof) as dst:
        dst.write(np.ones((3, 4, 4), dtype=np.int16))
        dst.update_tags(u_levels="0.1,0.5,0.5")
    with pytest.raises(SystemExit, match="not strictly increasing"):
        ge.load_quantile_functions({2005: str(p)}, [2005], tmp_path,
                                   np.arange(4, dtype=np.int64))
