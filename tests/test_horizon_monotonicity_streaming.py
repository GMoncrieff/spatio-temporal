"""The streamed horizon-monotonicity pass must equal the whole-raster one, bit for bit.

The whole-raster spelling held 3*H full-grid float64 arrays plus their np.stack copies plus
four accumulators, which OOM-killed a global run at 17111 x 40000. Streaming by row block is
equivalent because the cumulative maximum runs along the horizon axis and never couples two
pixels -- but "equivalent because I reasoned it through" is exactly the claim this project
has been burned by, so it is asserted here against a reference implementation.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from apply_recalibration import enforce_horizon_monotonicity  # noqa: E402

PROF = dict(driver="GTiff", dtype="float32", count=1, nodata=np.nan,
            crs="EPSG:4326", transform=rasterio.transform.from_origin(-180, 84, 0.009, 0.009))


def _reference(out_dir, suffix="_recal"):
    """The pre-streaming implementation, verbatim in behaviour."""
    out_dir = Path(out_dir)
    groups = {}
    for cen in sorted(out_dir.glob(f"*_prediction_*_central{suffix}.tif")):
        stem = cen.name.replace(f"_central{suffix}.tif", "")
        base = int(stem.split("_")[0][1:])
        year = int(stem.split("_")[-1])
        groups.setdefault(base, []).append((year - base, stem))
    total_lifted = n_px = 0
    for base, items in sorted(groups.items()):
        items.sort()
        if len(items) < 2:
            continue
        cen, lo, up, prof = [], [], [], None
        for _, stem in items:
            with rasterio.open(out_dir / f"{stem}_central{suffix}.tif") as sc:
                cen.append(sc.read(1).astype(np.float64)); prof = sc.profile.copy()
            with rasterio.open(out_dir / f"{stem}_lower{suffix}.tif") as sl:
                lo.append(sl.read(1).astype(np.float64))
            with rasterio.open(out_dir / f"{stem}_upper{suffix}.tif") as su:
                up.append(su.read(1).astype(np.float64))
        C, L, U = np.stack(cen), np.stack(lo), np.stack(up)
        valid = (np.isfinite(C).all(0) & np.isfinite(L).all(0)
                 & np.isfinite(U).all(0) & (C >= 0).all(0))
        hu = np.maximum(U - C, 0.0)
        hl = np.maximum(C - L, 0.0)
        hu2 = np.maximum.accumulate(hu, axis=0)
        hl2 = np.maximum.accumulate(hl, axis=0)
        lifted = ((hu2 > hu + 1e-12) | (hl2 > hl + 1e-12))
        total_lifted += int(lifted[:, valid].any(axis=0).sum())
        n_px += int(valid.sum())
        for i, (_, stem) in enumerate(items):
            new_up = np.where(valid, np.clip(C[i] + hu2[i], 0.0, 1.0), U[i])
            new_lo = np.where(valid, np.clip(C[i] - hl2[i], 0.0, 1.0), L[i])
            for arr, name in ((new_lo, "lower"), (new_up, "upper")):
                with rasterio.open(out_dir / f"{stem}_{name}{suffix}.tif", "w", **prof) as dst:
                    dst.write(arr.astype(prof["dtype"]), 1)
    return total_lifted


def _write_case(d, H, W, seed):
    """Four horizons with widths that genuinely dip, plus NaN and negative-central pixels."""
    rng = np.random.default_rng(seed)
    prof = dict(PROF, height=H, width=W)
    for h, year in ((5, 2005), (10, 2010), (15, 2015), (20, 2020)):
        cen = rng.uniform(0, 1, (H, W)).astype(np.float32)
        # deliberately non-monotone half-widths so the cummax has work to do
        hw = (rng.uniform(0.001, 0.05, (H, W)) * (1.0 + 0.5 * np.sin(h))).astype(np.float32)
        hwl = (rng.uniform(0.001, 0.05, (H, W)) * (1.0 + 0.5 * np.cos(h))).astype(np.float32)
        cen[0, 0] = np.nan                      # NaN central
        cen[1, 1] = -1.0                        # negative central -> invalid
        lo = (cen - hwl).astype(np.float32)
        up = (cen + hw).astype(np.float32)
        lo[2, 2] = np.nan                       # NaN bound
        stem = f"w2000_prediction_{year}"
        for arr, name in ((cen, "central"), (lo, "lower"), (up, "upper")):
            with rasterio.open(d / f"{stem}_{name}_recal.tif", "w", **prof) as dst:
                dst.write(arr, 1)


@pytest.mark.parametrize("block_rows", [1, 7, 64, 4096])
def test_streamed_equals_whole_raster(tmp_path, block_rows):
    a, b = tmp_path / "ref", tmp_path / "stream"
    a.mkdir(); b.mkdir()
    _write_case(a, 61, 43, seed=7)
    _write_case(b, 61, 43, seed=7)

    n_ref = _reference(a)
    n_new = enforce_horizon_monotonicity(b, block_rows=block_rows)

    assert n_ref == n_new, f"lifted count differs: {n_ref} vs {n_new}"
    assert n_ref > 0, "the fixture did not exercise the lifting path"
    for year in (2005, 2010, 2015, 2020):
        for name in ("lower", "upper", "central"):
            f = f"w2000_prediction_{year}_{name}_recal.tif"
            with rasterio.open(a / f) as x, rasterio.open(b / f) as y:
                xa, ya = x.read(1), y.read(1)
            assert np.array_equal(np.isnan(xa), np.isnan(ya)), f"{f}: NaN mask differs"
            m = ~np.isnan(xa)
            assert np.array_equal(xa[m], ya[m]), f"{f}: values differ"


def test_production_naming_is_processed(tmp_path):
    """Production rasters have no w{base}_ prefix and must still be made monotone.

    The glob used to require the prefix, so this pass silently skipped the entire forward
    product -- it would have shipped without the per-pixel cumulative max that makes T4.2
    structural. Never caught because no production model had ever been built.
    """
    d = tmp_path / "prod"; d.mkdir()
    H = W = 24
    prof = dict(PROF, height=H, width=W)
    rng = np.random.default_rng(3)
    for year, shrink in ((2025, 1.0), (2030, 0.4), (2035, 0.2), (2040, 0.1)):
        cen = rng.uniform(0.2, 0.8, (H, W)).astype(np.float32)
        hw = (0.05 * shrink * np.ones((H, W))).astype(np.float32)   # deliberately SHRINKING
        for arr, name in ((cen, "central"), ((cen - hw), "lower"), ((cen + hw), "upper")):
            with rasterio.open(d / f"prediction_{year}_{name}_recal.tif", "w", **prof) as dst:
                dst.write(arr.astype(np.float32), 1)

    lifted = enforce_horizon_monotonicity(d)
    assert lifted > 0, "production rasters were not processed at all"

    # every pixel's half-width must be non-decreasing across the four target years
    prev = None
    for year in (2025, 2030, 2035, 2040):
        with rasterio.open(d / f"prediction_{year}_central_recal.tif") as c, \
             rasterio.open(d / f"prediction_{year}_upper_recal.tif") as u, \
             rasterio.open(d / f"prediction_{year}_lower_recal.tif") as l:
            hw_u = u.read(1) - c.read(1)
            hw_l = c.read(1) - l.read(1)
        if prev is not None:
            assert (hw_u >= prev[0] - 1e-6).all(), f"upper half-width shrank at {year}"
            assert (hw_l >= prev[1] - 1e-6).all(), f"lower half-width shrank at {year}"
        prev = (hw_u, hw_l)


def test_single_horizon_group_is_untouched(tmp_path):
    """A base year with one horizon is skipped, and its rasters must not be rewritten."""
    d = tmp_path / "one"; d.mkdir()
    prof = dict(PROF, height=8, width=8)
    for name in ("central", "lower", "upper"):
        with rasterio.open(d / f"w2015_prediction_2020_{name}_recal.tif", "w", **prof) as dst:
            dst.write(np.full((8, 8), 0.5, dtype=np.float32), 1)
    before = {p.name: p.read_bytes() for p in d.glob("*.tif")}
    enforce_horizon_monotonicity(d)
    after = {p.name: p.read_bytes() for p in d.glob("*.tif")}
    assert before == after
