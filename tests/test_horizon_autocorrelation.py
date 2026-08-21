"""The AR(1) coupling estimator must survive a mostly-invalid grid.

`horizon_autocorrelation` converted its `n_sample_px` budget into a *block count*
(`n_sample_px // block**2` = seven 512 px blocks at the default) and drew those blocks
uniformly over the whole raster. On a regional extent that covered most of the region and the
estimate looked stable; on the global grid, which is 73% invalid, the blocks landed mostly in
nodata and the estimate was noise -- rho(h=15) ranged 0.319 to 0.827 over three seeds at two
sample sizes and did not converge when the sample was doubled.

These tests pin the fix: a known correlation is recovered on a sparsely-valid grid, and the
default estimator is deterministic.
"""

import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ensemble.residuals import horizon_autocorrelation  # noqa: E402

H, W = 1024, 2048
VALID_FRACTION = 0.25


def _write(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", nodata=np.nan, crs="EPSG:4326",
                       transform=from_origin(-180, 84, 0.009, 0.009)) as dst:
        dst.write(arr.astype(np.float32), 1)


@pytest.fixture
def manifest(tmp_path):
    """Two horizons of standardized residuals with a known correlation.

    Validity is a contiguous band, not scattered pixels, so that a uniformly-placed block can
    miss it entirely -- which is the geometry that broke the old sampler.
    """
    def build(rho, seed):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((H, W))
        b = rho * a + np.sqrt(1.0 - rho ** 2) * rng.standard_normal((H, W))
        invalid = np.ones((H, W), dtype=bool)
        band = int(H * VALID_FRACTION)
        r0 = H // 2 - band // 2
        invalid[r0:r0 + band] = False
        a[invalid] = np.nan
        b[invalid] = np.nan
        return a, b

    import pandas as pd
    rows, truth = [], {10: 0.55, 15: 0.30, 20: 0.80}
    prev = None
    for i, h in enumerate((5, 10, 15, 20)):
        if h == 5:
            rng = np.random.default_rng(0)
            cur = rng.standard_normal((H, W))
            invalid = np.ones((H, W), dtype=bool)
            band = int(H * VALID_FRACTION)
            r0 = H // 2 - band // 2
            invalid[r0:r0 + band] = False
            cur[invalid] = np.nan
        else:
            rho = truth[h]
            rng = np.random.default_rng(h)
            noise = rng.standard_normal((H, W))
            cur = rho * np.nan_to_num(prev) + np.sqrt(1 - rho ** 2) * noise
            cur[~np.isfinite(prev)] = np.nan
        p = tmp_path / f"z_h{h}.tif"
        _write(p, cur)
        rows.append({"window": "w-test", "horizon": h, "path_res_z": str(p)})
        prev = cur
    mpath = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(mpath, index=False)
    return str(mpath), truth


def test_recovers_known_rho_on_sparsely_valid_grid(manifest):
    mpath, truth = manifest
    rho = horizon_autocorrelation(mpath)
    for h, want in truth.items():
        assert rho[h] == pytest.approx(want, abs=0.05), f"h={h}: got {rho[h]}, want {want}"


def test_default_estimator_is_deterministic(manifest):
    """No seed dependence at all -- the stripe sampler does not draw randomly."""
    mpath, _ = manifest
    a = horizon_autocorrelation(mpath, random_seed=42)
    b = horizon_autocorrelation(mpath, random_seed=1234)
    for h in a:
        assert a[h] == b[h]


def test_stripe_offsets_agree(manifest):
    """Independent stripe phases are independent samples; they must agree closely."""
    mpath, _ = manifest
    vals = [horizon_autocorrelation(mpath, offset=o) for o in (0, 37, 111)]
    for h in vals[0]:
        got = [v[h] for v in vals]
        assert max(got) - min(got) < 0.05, f"h={h}: offsets disagree {got}"


def test_report_returns_pair_counts(manifest):
    mpath, _ = manifest
    rho, diag = horizon_autocorrelation(mpath, report=True)
    for h in rho:
        assert diag[h]["pairs"] > 100_000, f"h={h}: only {diag[h]['pairs']} pairs sampled"
