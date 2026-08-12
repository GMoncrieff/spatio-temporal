"""Phase 1b/1d tests — coverage collapse with scale, and the planted class-conditional failure."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.validate import (  # noqa: E402
    DHAT_LABELS,
    compute_block_coverage,
    compute_class_conditional_coverage,
    rollup_coverage,
    rollup_zonal,
    wilson_interval,
)

H, W = 256, 256


def _write(path, arr):
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
        "dtype": "float32", "crs": "EPSG:4326",
        "transform": from_origin(-180, 84, 0.009, 0.009), "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
    return str(path)


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(95, 100)
    assert lo < 0.95 < hi
    lo14, hi14 = wilson_interval(13, 14)
    lo846, hi846 = wilson_interval(804, 846)
    # The reason ecoregion is the scored unit and biome only reported.
    assert (hi14 - lo14) > (hi846 - lo846) * 3


def test_block_coverage_collapses_when_errors_are_spatially_correlated():
    """Pixel-scale coverage ~0.95 but a coherent regional bias kills the aggregate."""
    rng = np.random.default_rng(0)
    central = np.full((H, W), 0.5, dtype=np.float32)
    sigma = 0.1
    lower = central - 1.96 * sigma
    upper = central + 1.96 * sigma
    # Error = a single smooth field shared by every pixel in a 64px block.
    coarse = rng.standard_normal((H // 64, W // 64)) * sigma
    field = np.kron(coarse, np.ones((64, 64)))
    observed = central + field

    with_tmp = Path(__file__).parent / "_tmp_val"
    with_tmp.mkdir(exist_ok=True)
    lo_p = _write(with_tmp / "lo.tif", lower)
    hi_p = _write(with_tmp / "hi.tif", upper)
    ob_p = _write(with_tmp / "ob.tif", observed)
    df = compute_block_coverage(lo_p, hi_p, ob_p, block_sizes=(1, 64))
    px = float(df[df.scale_px == 1]["coverage"].iloc[0])
    blk = float(df[df.scale_px == 64]["coverage"].iloc[0])
    assert px > 0.9
    assert blk <= px, "aggregating correlated errors must not improve coverage"
    for p in with_tmp.glob("*.tif"):
        p.unlink()
    with_tmp.rmdir()


def _planted_manifest(tmp_path, bad_bin_idx=4, bad_cov=0.5, seed=0):
    """95% coverage everywhere except one Delta-hat bin, which sits at ~50%."""
    rng = np.random.default_rng(seed)
    dhat = np.zeros((H, W), dtype=np.float32)
    # Assign rows to Delta-hat bins by construction.
    bin_centres = [-0.02, -0.005, 0.005, 0.03, 0.10, 0.20]
    rows_per = H // len(bin_centres)
    for i, c in enumerate(bin_centres):
        dhat[i * rows_per:(i + 1) * rows_per] = c
    w = np.full((H, W), 0.1, dtype=np.float32)
    res = rng.normal(0, 1, size=(H, W)).astype(np.float32)
    # Scale residuals so each bin has the intended coverage under +/- w.
    from scipy.stats import norm

    z_ok = norm.ppf(1 - (1 - 0.95) / 2)
    z_bad = norm.ppf(1 - (1 - bad_cov) / 2)
    scale = np.full((H, W), 0.1 / z_ok, dtype=np.float32)
    scale[bad_bin_idx * rows_per:(bad_bin_idx + 1) * rows_per] = 0.1 / z_bad
    res = res * scale

    central = np.full((H, W), 0.3, dtype=np.float32)
    paths = {
        "path_res_native": _write(tmp_path / "res.tif", res),
        "path_dhat": _write(tmp_path / "dhat.tif", dhat),
        "path_hm_t0": _write(tmp_path / "hm0.tif", central - dhat),
        "path_w_up": _write(tmp_path / "wup.tif", w),
        "path_w_lo": _write(tmp_path / "wlo.tif", w),
    }
    return pd.DataFrame([{"horizon": 20, "base_year": 2000, "target_year": 2020, **paths}])


def test_class_audit_finds_what_the_pooled_number_hides(tmp_path):
    manifest = _planted_manifest(tmp_path)
    audit = compute_class_conditional_coverage(manifest)
    by_dhat = rollup_coverage(audit, by=["horizon", "dhat_bin", "dhat_bin_idx"])
    pooled = rollup_coverage(audit, by=["horizon"])

    planted_label = DHAT_LABELS[4]
    bad = by_dhat[by_dhat["dhat_bin"] == planted_label]["coverage"].iloc[0]
    good = by_dhat[by_dhat["dhat_bin"] != planted_label]["coverage"]

    assert abs(bad - 0.50) < 0.05, "the audit must see the planted 50% cell"
    assert (abs(good - 0.95) < 0.05).all()
    # The pooled number is dragged only part-way down — exactly the blindness being fixed.
    assert float(pooled["coverage"].iloc[0]) > bad + 0.2


def test_audit_reports_both_tails_separately(tmp_path):
    """Asymmetric miscoverage is the expected finding; one number would hide the side."""
    rng = np.random.default_rng(1)
    res = np.abs(rng.normal(0, 0.2, size=(H, W))).astype(np.float32)  # all above central
    w = np.full((H, W), 0.1, dtype=np.float32)
    manifest = pd.DataFrame([{
        "horizon": 20, "base_year": 2000, "target_year": 2020,
        "path_res_native": _write(tmp_path / "r.tif", res),
        "path_dhat": _write(tmp_path / "d.tif", np.zeros((H, W), np.float32)),
        "path_hm_t0": _write(tmp_path / "h.tif", np.full((H, W), 0.2, np.float32)),
        "path_w_up": _write(tmp_path / "u.tif", w),
        "path_w_lo": _write(tmp_path / "l.tif", w),
    }])
    audit = compute_class_conditional_coverage(manifest)
    assert audit["n_above_upper"].sum() > 0
    assert audit["n_below_lower"].sum() == 0


def test_thin_cells_are_reported_not_crashed(tmp_path):
    manifest = _planted_manifest(tmp_path)
    audit = compute_class_conditional_coverage(manifest)
    assert (audit["n_eff"] > 0).all()
    assert np.isfinite(audit["coverage"]).all()
    thin = audit[audit["n_eff"] < 100]
    assert np.isfinite(thin["wilson_lo"]).all() if len(thin) else True


def test_rollup_zonal_uses_pixel_weighted_means():
    df = pd.DataFrame({
        "zone_id": [1, 2], "n_px": [100, 900],
        "mean_lo": [0.0, 0.0], "mean_hi": [1.0, 1.0], "mean_ob": [0.9, 0.1],
    })
    df["BIOME_NAME"] = "b"
    out = rollup_zonal(df, "BIOME_NAME", thresholds=())
    assert abs(float(out["mean_ob"].iloc[0]) - (0.9 * 100 + 0.1 * 900) / 1000) < 1e-9
    assert bool(out["covered_mean"].iloc[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
