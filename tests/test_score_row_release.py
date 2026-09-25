"""The scorer's row loop must not hold two window-years at once.

MEASURED on the global grid, 2026-09-18: one window-year of 184,573,321 px costs ~85 GiB
resident once ``load_row`` has compacted it. ``u, cell, pp, cons = load_row(...)`` rebinds
those names only when the call RETURNS, so row 2's arrays are built while row 1's are still
referenced, and the run peaked past 170 GiB against 125 GB of DRAM and died -- two hours in,
having written nothing. ``--row_chunk`` does not help: it bounds the transient band reads,
not the compacted per-pixel arrays, which are the bulk of the footprint.

On Africa the same window-year is ~29 GiB and two fit comfortably, which is why every
regional run passed. Fourth working set in this project that only global scale exercises.

The check is a weak reference to the previous row's ``observed`` array -- the largest thing
``cell`` holds, and a plain dict cannot be weak-referenced. If it is still alive when the
next ``load_row`` is entered, the loop is holding both rows -- which is exactly the state
that OOMed, and exactly what this test fails on when the ``del`` is removed.
"""
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.models.quantile_spline import output_u_grid  # noqa: E402
import score_distributional_model as sd  # noqa: E402

H, W = 120, 90


def _profile(count, dtype, nodata):
    return dict(driver="GTiff", height=H, width=W, count=count, dtype=dtype, nodata=nodata,
                crs="EPSG:4326", transform=Affine(0.01, 0, 10.0, 0, -0.01, 5.0))


def _write(path, arr, dtype="float32", nodata=np.nan):
    a = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", **_profile(a.shape[0], dtype, nodata)) as dst:
        dst.write(a.astype(dtype))
    return path


@pytest.fixture(scope="module")
def two_rows(tmp_path_factory):
    """A stitched dir with TWO target years, so the loop actually iterates twice."""
    d = tmp_path_factory.mktemp("stitched2")
    rng = np.random.default_rng(3)
    u = output_u_grid(64)
    hm0 = (rng.random((H, W)) * 0.4).astype(np.float32)
    _write(d / "HM_2000_AA_1000.tiff", hm0)
    ctx = np.stack([rng.random((H, W)).astype(np.float32),
                    (rng.random((H, W)) * 150.0).astype(np.float32)])
    _write(d / "change_context_w2000_1000.tif", ctx)

    for year in (2005, 2010):
        obs = np.clip(hm0 + rng.normal(0, 0.02, (H, W)), 0, 1).astype(np.float32)
        central = np.clip(hm0 + rng.normal(0, 0.01, (H, W)), 0, 1).astype(np.float32)
        spread = (0.01 + 0.05 * rng.random((H, W))).astype(np.float32)
        qf = np.clip(central[None] + spread[None] * (u[:, None, None] - 0.5) * 4.0,
                     0, 1).astype(np.float32)
        _write(d / f"HM_{year}_AA_1000.tiff", obs)
        _write(d / f"w2000_prediction_{year}_central.tif", central)
        _write(d / f"w2000_prediction_{year}_lower.tif",
               qf[int(np.argmin(np.abs(u - 0.025)))])
        _write(d / f"w2000_prediction_{year}_upper.tif",
               qf[int(np.argmin(np.abs(u - 0.975)))])
        qp = d / f"w2000_prediction_{year}_qf.tif"
        with rasterio.open(qp, "w", **_profile(len(u), "float32", np.nan)) as dst:
            dst.write(qf)
            dst.update_tags(u_levels=",".join(repr(float(v)) for v in u),
                            scale_factor="1.0", head_family="pwl", head_params="17")
    mask = d / "fold_mask.tif"
    with rasterio.open(mask, "w", **_profile(1, "int16", -32768)) as dst:
        dst.write(np.ones((H, W), dtype=np.int16), 1)
    return d, mask


def test_the_row_loop_releases_the_previous_window_year(two_rows, tmp_path, monkeypatch):
    d, mask = two_rows
    live = {}
    calls = []
    real = sd.load_row

    def spy(row, fold_sel, row_chunk=0):
        # Entering load_row for row N: row N-1's cell must already be unreachable.
        prev = live.get("ref")
        if prev is not None:
            assert prev() is None, (
                "the previous window-year's per-pixel arrays are still referenced when the "
                "next one is being built -- at 184.6 M px that is ~85 GiB held twice, which "
                "is the state that OOMed the global run")
        out = real(row, fold_sel, row_chunk)
        live["ref"] = weakref.ref(out[1]["observed"])
        calls.append(row["target_year"])
        return out

    monkeypatch.setattr(sd, "load_row", spy)
    old = sd.HM_DIR
    sd.HM_DIR = d
    try:
        rc = sd.main(["--stitched_dir", str(d), "--label", "rel", "--out_dir", str(tmp_path),
                      "--folds", "1", "--fold_mask", str(mask),
                      "--context_pattern", str(d / "change_context_w{year}_1000.tif"),
                      "--plots", "false", "--min_count", "50"])
    finally:
        sd.HM_DIR = old
    assert rc == 0
    assert calls == [2005, 2010], f"the loop must iterate twice, saw {calls}"
