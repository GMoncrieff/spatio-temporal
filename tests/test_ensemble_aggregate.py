"""Phase 4 tests — aggregation math, rank histograms, scores, and the scorecard logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.aggregate import (  # noqa: E402
    aggregate_region_statistic,
    coverage_from_members,
    dequantize_block,
    energy_score,
    open_ensemble,
    rank_histogram,
    rank_histogram_test,
    sample_pairs,
    summarize_ensemble,
    variogram_score,
)
from src.ensemble.copula import DEFAULT_SCALE, INT16_SENTINEL, quantize  # noqa: E402

M, H, W = 40, 32, 32


def _make_store(tmp_path, values):
    """values: (M, n_h, H, W) float array -> a quantized zarr store."""
    import zarr

    path = tmp_path / "ens.zarr"
    z = zarr.open(str(path), mode="w", shape=values.shape, chunks=(10, 1, 16, 16),
                  dtype="i2", fill_value=INT16_SENTINEL)
    z[:] = quantize(values)
    z.attrs.update({"scale": DEFAULT_SCALE, "offset": 0.0, "sentinel": INT16_SENTINEL})
    return str(path)


def test_summarize_ensemble_percentiles():
    s = summarize_ensemble(np.linspace(0, 1, 101))
    assert abs(s["median"] - 0.5) < 1e-9
    assert abs(s["p2_5"] - 0.025) < 0.01
    assert abs(s["p97_5"] - 0.975) < 0.01
    assert s["n"] == 101


def test_region_statistic_is_computed_per_member(tmp_path):
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.1, 0.9, (M, 1, H, W)).astype(np.float32)
    store_path = _make_store(tmp_path, truth)
    store, attrs = open_ensemble(store_path)
    mask = np.zeros((H, W), dtype=bool)
    mask[:8, :8] = True
    got = aggregate_region_statistic(store, mask, 0, attrs=attrs)
    expected = truth[:, 0, :8, :8].reshape(M, -1).mean(axis=1)
    assert np.allclose(got, expected, atol=DEFAULT_SCALE * 2)


def test_area_above_threshold_statistic(tmp_path):
    vals = np.zeros((M, 1, H, W), dtype=np.float32)
    vals[:, 0, :16, :] = 0.5     # exactly half the pixels above 0.1
    store_path = _make_store(tmp_path, vals)
    store, attrs = open_ensemble(store_path)
    mask = np.ones((H, W), dtype=bool)
    got = aggregate_region_statistic(store, mask, 0, threshold=0.1, attrs=attrs)
    assert np.allclose(got, 0.5)


def test_coverage_from_members_is_a_two_sided_check():
    members = np.random.default_rng(0).normal(0, 1, (200, 500))
    obs = np.random.default_rng(1).normal(0, 1, 500)
    r = coverage_from_members(members, obs)
    assert 0.90 < r["coverage"] < 0.99
    assert abs(r["frac_below"] - r["frac_above"]) < 0.05

    biased = coverage_from_members(members, obs + 5)
    assert biased["coverage"] == 0.0
    assert biased["frac_above"] == 1.0


def test_rank_histogram_is_flat_for_a_calibrated_ensemble():
    rng = np.random.default_rng(0)
    members = rng.normal(0, 1, (50, 4000))
    obs = rng.normal(0, 1, 4000)
    hist = rank_histogram(members, obs)
    assert hist.size == 51
    assert hist.sum() == 4000
    assert rank_histogram_test(hist)["p_value"] > 0.01


def test_rank_histogram_is_u_shaped_when_underdispersed():
    rng = np.random.default_rng(0)
    members = rng.normal(0, 0.3, (50, 4000))    # too narrow
    obs = rng.normal(0, 1, 4000)
    hist = rank_histogram(members, obs)
    assert rank_histogram_test(hist)["p_value"] < 0.01
    edges = hist[0] + hist[-1]
    middle = hist[20:31].sum()
    assert edges > middle


def test_energy_score_prefers_the_truthful_ensemble():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 30)
    good = rng.normal(0, 1, (60, 30)) * 0.2 + y
    bad = rng.normal(5, 1, (60, 30))
    assert energy_score(good, y) < energy_score(bad, y)


def test_variogram_score_rewards_correct_spatial_structure():
    """Same marginals, different correlation: the score must separate them."""
    rng = np.random.default_rng(0)
    n = 60
    base = rng.normal(0, 1, n)
    truth = base + 0.1 * rng.normal(0, 1, n)
    # Correlated ensemble: members share the smooth structure of the truth.
    corr = np.stack([base + 0.3 * rng.normal(0, 1, n) for _ in range(50)])
    # Independent ensemble: identical marginal spread, no shared structure.
    indep = np.stack([rng.normal(base.mean(), base.std(), n) for _ in range(50)])
    pairs = sample_pairs(n, 400, rng=rng)
    assert variogram_score(corr, truth, pairs) < variogram_score(indep, truth, pairs)


def test_scorecard_flags_exactly_the_broken_target():
    """A synthetic run failing one target must produce exactly one failed row."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from validate_ensemble import Scorecard

    card = Scorecard()
    card.add("T2.1", "block coverage 10km", 0.951, "0.95 +/- 0.05", True)
    card.add("T2.2", "ecoregion mean", 0.949, "0.95 +/- 0.05", True)
    card.add("T2.3", "area>0.1", 0.55, "0.95 +/- 0.05", False, knob="T2_under")
    card.add("T2.6", "biome (reported)", 0.93, "reported with CI", None)
    df = card.df()
    scored = df[df["pass"].notna()]
    assert len(scored) == 3
    assert int((~scored["pass"]).sum()) == 1
    failed = scored[~scored["pass"]].iloc[0]
    assert "long-range weight" in failed["diagnosis"]


def test_dequantize_maps_sentinel_to_nan():
    q = np.array([[INT16_SENTINEL, 1000]], dtype=np.int16)
    out = dequantize_block(q, {"scale": DEFAULT_SCALE, "offset": 0.0})
    assert np.isnan(out[0, 0])
    assert abs(out[0, 1] - 1000 * DEFAULT_SCALE) < 1e-9




def test_multi_scale_block_stats_match_per_scale_computation(tmp_path):
    """Nested scales aggregated from the finest must equal computing each independently."""
    from src.ensemble.aggregate import (
        block_member_stats,
        block_member_stats_multi,
        block_observed,
        block_observed_multi,
    )
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(0)
    vals = rng.uniform(0.05, 0.95, (8, 1, 32, 32)).astype(np.float32)
    path = _make_store(tmp_path, vals)
    store, attrs = open_ensemble(path)

    multi = block_member_stats_multi(store, 0, [4, 8], attrs=attrs)
    for B in (4, 8):
        ref_mean, ref_valid, _ = block_member_stats(store, 0, B, attrs=attrs)
        got_mean, got_valid, _ = multi[B]
        assert np.allclose(got_mean, ref_mean, atol=1e-6, equal_nan=True)
        assert np.array_equal(got_valid, ref_valid)

    obs = rng.uniform(0, 1, (32, 32)).astype("float32")
    op = tmp_path / "obs.tif"
    with rasterio.open(op, "w", driver="GTiff", height=32, width=32, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(-180, 84, 0.009, 0.009), nodata=np.nan) as d:
        d.write(obs, 1)
    prof = {"height": 32, "width": 32, "transform": from_origin(-180, 84, 0.009, 0.009)}
    om = block_observed_multi(str(op), prof, [4, 8])
    for B in (4, 8):
        ref, _ = block_observed(str(op), prof, B)
        assert np.allclose(om[B][0], ref, atol=1e-6, equal_nan=True)




def test_interval_score_penalises_both_width_and_miscoverage():
    """The joint metric: a huge interval must not beat a well-sized one."""
    from src.ensemble.aggregate import interval_score

    y = np.linspace(0, 1, 200)
    tight_right = interval_score(y - 0.05, y + 0.05, y)
    huge = interval_score(y - 5.0, y + 5.0, y)
    tight_wrong = interval_score(y + 0.5, y + 0.6, y)
    assert tight_right["interval_score"] < huge["interval_score"], "width must cost"
    assert tight_right["interval_score"] < tight_wrong["interval_score"], "misses must cost"
    assert huge["penalty_term"] == 0.0
    assert tight_wrong["penalty_term"] > 0.0


def test_crps_prefers_the_sharper_calibrated_ensemble():
    from src.ensemble.aggregate import crps_from_members

    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    sharp = np.stack([y + rng.normal(0, 0.2, 500) for _ in range(60)])
    diffuse = np.stack([y + rng.normal(0, 3.0, 500) for _ in range(60)])
    assert crps_from_members(sharp, y) < crps_from_members(diffuse, y)


def test_spread_skill_ratio_detects_under_and_over_dispersion():
    from src.ensemble.aggregate import spread_skill_ratio

    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, 2000)
    # A correctly dispersed ensemble is centred on a *forecast* that misses the truth by
    # the same sigma as the member spread — not on the truth itself.
    forecast = truth + rng.normal(0, 1, 2000)
    good = np.stack([forecast + rng.normal(0, 1, 2000) for _ in range(80)])
    under = np.stack([forecast + rng.normal(0, 0.2, 2000) for _ in range(80)])
    assert abs(spread_skill_ratio(good, truth)["ratio"] - 1.0) < 0.25
    assert spread_skill_ratio(under, truth)["ratio"] < 0.75


def test_member_diversity_flags_duplicate_members():
    from src.ensemble.aggregate import member_diversity

    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (32, 32))
    identical = np.stack([base] * 5)
    diverse = np.stack([base + rng.normal(0, 1, (32, 32)) for _ in range(5)])
    assert member_diversity(identical)["mean_pairwise_corr"] > 0.999
    assert member_diversity(diverse)["mean_pairwise_corr"] < 0.9


def test_change_distribution_sees_an_implausible_negative_tail(tmp_path):
    """T6's job: catch members that manufacture HM collapses the world does not produce."""
    import rasterio
    from rasterio.transform import from_origin

    from src.ensemble.aggregate import change_distribution

    n, size = 12, 64
    hm0 = np.full((size, size), 0.5, dtype=np.float32)
    tr = from_origin(-180, 84, 0.009, 0.009)
    base_p = tmp_path / "hm0.tif"
    with rasterio.open(base_p, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:4326", transform=tr, nodata=np.nan) as d:
        d.write(hm0, 1)

    rng = np.random.default_rng(0)
    # Members drift symmetrically; reality only ever increases.
    members = (hm0[None, None] + rng.normal(0, 0.12, (n, 1, size, size))).astype(np.float32)
    store_path = _make_store(tmp_path, np.clip(members, 0, 1))
    store, attrs = open_ensemble(store_path)

    obs = np.clip(hm0 + np.abs(rng.normal(0, 0.02, (size, size))), 0, 1).astype(np.float32)
    obs_p = tmp_path / "obs.tif"
    with rasterio.open(obs_p, "w", driver="GTiff", height=size, width=size, count=1,
                       dtype="float32", crs="EPSG:4326", transform=tr, nodata=np.nan) as d:
        d.write(obs, 1)

    prof = {"height": size, "width": size, "transform": tr}
    out = change_distribution(store, 0, str(base_p), prof, attrs=attrs, observed_path=str(obs_p))
    th = out["thresholds"]
    mem = dict(zip(th, out["member_frac"]))
    ob = dict(zip(th, out["observed_frac"]))
    assert mem[-0.15] > 0.05, "the planted symmetric ensemble must show a fat negative tail"
    assert ob[-0.15] == 0.0, "the observation never decreases"
    assert out["member_q01"] < out["observed_q01"]




def test_observed_aggregation_respects_the_ensemble_mask(tmp_path):
    """Aggregates are only comparable when both sides average the same pixels.

    Without a shared mask the observed block mean includes pixels the ensemble never
    covers (coastlines, prediction gaps), and the two aggregates differ for reasons that
    have nothing to do with calibration — which reads as a coverage collapse.
    """
    import rasterio
    from rasterio.transform import from_origin

    from src.ensemble.aggregate import block_observed

    size, B = 32, 16
    tr = from_origin(-180, 84, 0.009, 0.009)
    obs = np.ones((size, size), dtype="float32")
    obs[:, :16] = 9.0                      # a region the ensemble does not cover
    mask = np.full((size, size), np.nan, dtype="float32")
    mask[:, 16:] = 0.5                     # ensemble covers only the right half

    def w(name, arr):
        p = tmp_path / name
        with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                           dtype="float32", crs="EPSG:4326", transform=tr, nodata=np.nan) as d:
            d.write(arr, 1)
        return str(p)

    op, mp = w("obs.tif", obs), w("mask.tif", mask)
    prof = {"height": size, "width": size, "transform": tr}

    unmasked, _ = block_observed(op, prof, B, min_valid_frac=0.0)
    masked, valid = block_observed(op, prof, B, min_valid_frac=0.0, mask_path=mp)
    assert np.allclose(unmasked[:, 0], 9.0), "left blocks are all uncovered pixels"
    assert not valid[:, 0].any(), "uncovered blocks must drop out entirely"
    assert np.allclose(masked[:, 1], 1.0), "covered blocks average only covered pixels"




def test_clustered_sampling_yields_nearby_pairs():
    """Structure scores need close pairs; a uniform global sample provides almost none."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from validate_ensemble import _clustered_sample

    from src.ensemble.aggregate import sample_pairs

    H, W = 4000, 8000
    valid = np.ones((H, W), dtype=bool)
    rng = np.random.default_rng(0)

    uniform = rng.choice(H * W, 1500, replace=False)
    clustered = _clustered_sample(valid, 1500, rng, patch=512, n_patches=12)

    def n_close(idx):
        rr, cc = np.unravel_index(idx, (H, W))
        coords = np.stack([rr, cc], axis=1).astype(float)
        i, _ = sample_pairs(idx.size, 20000, rng=np.random.default_rng(1),
                            coords=coords, max_dist=500)
        return i.size

    # The advantage grows with grid size: on this 4000x8000 test grid clustering buys
    # ~4.5x, while on the real 17111x40000 grid uniform sampling yields 26 usable pairs
    # out of 20000 requested and clustering yields thousands.
    assert n_close(clustered) > 3 * max(n_close(uniform), 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
