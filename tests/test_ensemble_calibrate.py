"""Phase 1.5 tests — conformal factors, shrinkage, guards, and the identity case."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.calibrate import (  # noqa: E402
    S_CEIL,
    S_FLOOR,
    ScaleFactorTable,
    ScoreStore,
    _conformal_quantile,
    _enforce_horizon_monotonicity,
    _weighted_median,
    as_identity,
    collapse_to_global,
    decide_recalibration,
    fit_scale_factors,
)


def _store_with(inflation_by_class, n_per_class=4000, n_chips=400, seed=0, horizons=(5, 20)):
    """Scores whose true spread is `inflation` times the model's stated half width."""
    rng = np.random.default_rng(seed)
    store = ScoreStore(cap=20000, random_seed=seed)
    for h in horizons:
        for (d_idx, infl) in inflation_by_class.items():
            key = (h, d_idx, 0, 0)
            for fold in (1, 2, 3, 4, 5):
                # E = |residual| / w with residual ~ N(0, infl * w / 1.96)
                r = rng.normal(0, infl / 1.959964, size=n_per_class)
                store.add(key, fold, up=r[r > 0], lo=-r[r <= 0],
                          chips=np.arange(n_chips) + 10_000 * fold + 1000 * d_idx,
                          n_total=n_per_class)
    return store


def test_recovers_a_planted_per_class_inflation():
    store = _store_with({0: 1.0, 1: 2.0})
    f = fit_scale_factors(store, alpha=0.05, n0=200.0, smooth=False)
    a = f[(f.dhat_bin_idx == 0)]["s_up_raw"].mean()
    b = f[(f.dhat_bin_idx == 1)]["s_up_raw"].mean()
    assert abs(a - 1.0) < 0.12
    assert abs(b - 2.0) < 0.25


def test_conformal_quantile_gives_at_least_nominal_coverage():
    rng = np.random.default_rng(0)
    hits = []
    for trial in range(200):
        cal = np.abs(rng.normal(size=60))
        s = _conformal_quantile(cal, n_eff=60, level=0.95)
        test = np.abs(rng.normal(size=500))
        hits.append(float((test <= s).mean()))
    assert np.mean(hits) >= 0.95 - 0.01


def test_shrinkage_limits_move_to_global_and_to_raw():
    thin = _store_with({0: 1.0, 1: 5.0}, n_per_class=200, n_chips=3)
    f = fit_scale_factors(thin, n0=200.0, smooth=False)
    lam = f["lambda"].to_numpy()   # 3 chips x 5 folds = 15 -> 15/(15+200)
    assert (lam < 0.1).all(), "a handful of chips must be shrunk hard toward the global factor"
    wild = f[f.dhat_bin_idx == 1]
    # The shrunk factor must sit far closer to the global target than to its own noisy raw.
    to_global = abs(wild["s_up_shrunk"] - wild["s_up_global"])
    to_raw = abs(wild["s_up_shrunk"] - wild["s_up_raw"])
    assert (to_global < 0.25 * to_raw).all()

    fat = _store_with({0: 1.0}, n_per_class=5000, n_chips=100_000)
    f2 = fit_scale_factors(fat, n0=200.0, smooth=False)
    assert (f2["lambda"] > 0.99).all()


def test_bounds_and_horizon_monotonicity_are_enforced():
    df = pd.DataFrame({
        "horizon": [5, 10, 15, 20], "dhat_bin_idx": [1] * 4, "hm_bin_idx": [0] * 4,
        "biome": [0] * 4, "n_eff": [500] * 4,
        "s_up": [2.0, 1.0, 3.0, 0.5], "s_lo": [1.0, 1.0, 1.0, 1.0],
    })
    out = _enforce_horizon_monotonicity(df).sort_values("horizon")
    assert list(out["s_up"]) == [2.0, 2.0, 3.0, 3.0]

    store = _store_with({0: 100.0})   # absurd inflation
    f = fit_scale_factors(store, smooth=False)
    assert (f["s_up"] <= S_CEIL + 1e-9).all()
    assert (f["s_up"] >= S_FLOOR - 1e-9).all()


def test_thin_cells_follow_their_primary_stratum_not_the_global_factor():
    """A rare change class fragments across HM bin x biome; it must still get its own factor.

    This is the failure seen on the real global audit: the >0.15 change bin holds plenty of
    chips in total but only a handful per (HM bin, biome) cell, so flat shrinkage pulled
    every one of them back to the per-horizon factor and the miscoverage survived.
    """
    rng = np.random.default_rng(0)
    store = ScoreStore(cap=20000, random_seed=0)
    # dhat bin 0: common, well calibrated. dhat bin 4: rare, 3x too narrow, and split
    # across 20 (hm_bin, biome) cells that are individually thin.
    for fold in (1, 2, 3, 4, 5):
        for hm in range(2):
            for biome in range(10):
                r = rng.normal(0, 1.0 / 1.959964, size=3000)
                store.add((20, 0, hm, biome), fold, up=r[r > 0], lo=-r[r <= 0],
                          chips=np.arange(300) + 100000 * fold + 137 * (hm * 10 + biome),
                          n_total=3000)
                r = rng.normal(0, 3.0 / 1.959964, size=60)
                store.add((20, 4, hm, biome), fold, up=r[r > 0], lo=-r[r <= 0],
                          chips=np.arange(6) + 900000 * fold + 977 * (hm * 10 + biome),
                          n_total=60)

    f = fit_scale_factors(store, n0=200.0, smooth=False)
    rare = f[f.dhat_bin_idx == 4]
    common = f[f.dhat_bin_idx == 0]
    assert (common["s_up"] - 1.0).abs().mean() < 0.15
    # Pooled over its cells the rare class has ample data, so its factor must approach 3,
    # not sit near the ~1 the global factor would impose.
    assert rare["s_up_group"].mean() > 2.0
    assert rare["s_up"].mean() > 2.0


def test_identity_case_is_a_true_no_op():
    store = _store_with({0: 1.0, 1: 1.0})
    f = fit_scale_factors(store)
    decision = decide_recalibration(f)
    assert decision["decision"] == "keep", decision
    ident = as_identity(f)
    assert (ident["s_up"] == 1.0).all() and (ident["s_lo"] == 1.0).all()


def test_decision_is_global_when_the_bias_is_uniform():
    store = _store_with({0: 1.6, 1: 1.6, 2: 1.6})
    f = fit_scale_factors(store)
    decision = decide_recalibration(f)
    assert decision["decision"] == "global", decision
    g = collapse_to_global(f)
    assert g["s_up"].nunique() <= len(f["horizon"].unique())


def test_decision_is_stratified_when_classes_disagree():
    store = _store_with({0: 1.0, 1: 2.5, 2: 4.0})
    decision = decide_recalibration(fit_scale_factors(store))
    assert decision["decision"] == "stratified", decision


def test_scale_factor_table_lookup_and_fallback():
    df = pd.DataFrame({
        "horizon": [20, 20], "dhat_bin_idx": [0, 1], "hm_bin_idx": [0, 0], "biome": [1, 1],
        "n_eff": [100, 100], "s_up": [1.2, 2.0], "s_lo": [0.9, 1.1],
    })
    t = ScaleFactorTable(df)
    up, lo = t.lookup(20, np.array([[0, 1]]), np.array([[0, 0]]), np.array([[1, 1]]))
    assert np.allclose(up, [[1.2, 2.0]])
    assert np.allclose(lo, [[0.9, 1.1]])
    # Unknown biome falls back to the (horizon, dhat) average, never to a crash.
    up2, _ = t.lookup(20, np.array([0]), np.array([0]), np.array([99]))
    assert np.isfinite(up2).all()


def test_weighted_median_ignores_a_single_wild_class():
    v = np.array([1.0, 1.05, 0.95, 500.0])
    w = np.array([1000.0, 1000.0, 1000.0, 1.0])
    assert abs(_weighted_median(v, w) - 1.0) < 0.1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
