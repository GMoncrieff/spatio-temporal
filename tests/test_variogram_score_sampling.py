"""T3.2's sampling fix, on a synthetic field with the geometry that broke it.

The measured problem: after the central-field change, most of the map has near-zero
ensemble spread. A pair whose two endpoints are both degenerate contributes the *same*
quantity to a correlated ensemble and to an independent one — for both, the members
collapse onto the central forecast — so it cancels in the T3.2 ratio while still diluting
it. Uniform sampling draws mostly such pairs and the comparison becomes uninformative
(measured: 1.0% improvement against a 30% target).

These tests plant that geometry explicitly: a small "live" region with real spread and
real spatial correlation, inside a large degenerate background. The correlated ensemble is
genuinely better there, so a sampler that finds the live region must report a much larger
improvement than one that does not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.aggregate import (  # noqa: E402
    informative_pair_fraction,
    sample_pairs,
    variogram_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

H = W = 256
LIVE = (slice(16, 64), slice(16, 64))   # 48x48 live patch in a 256x256 map
M = 20


def _smooth(field, k=12, passes=3):
    """Cheap box blur, to give the correlated ensemble genuine spatial structure."""
    out = field.astype(np.float64)
    for _ in range(passes):
        pad = np.pad(out, ((k, k), (k, k)), mode="reflect")
        acc = np.zeros_like(out)
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                acc += pad[k + di:k + di + out.shape[0], k + dj:k + dj + out.shape[1]]
        out = acc / ((2 * k + 1) ** 2)
    return out


def build_case(seed=0):
    """Return (spread, valid, truth, correlated members, independent members).

    Two properties make this reproduce the real failure rather than a caricature:

    * **The degenerate background is not empty.** Its members collapse onto a non-trivial
      central forecast, and the truth differs from that forecast, so a dead-dead pair
      contributes the *same positive amount* to both the correlated ensemble's score and
      the null's. That common term is what pulls the ratio toward 1. A background of exact
      zeros would contribute nothing to either and would not dilute at all — which is a
      much easier problem than the one being fixed.
    * **The null is the correlated member's own values, spatially shuffled**, which is the
      real contract: identical marginals, zero spatial structure. Drawing the null from a
      fresh distribution would confound the structure difference with a scale difference.
    """
    rng = np.random.default_rng(seed)
    live = np.zeros((H, W), dtype=bool)
    live[LIVE] = True
    spread = np.where(live, 1.0, 1e-6)

    # The central forecast exists everywhere; the ensemble simply has no spread around it
    # outside the live region.
    central = _smooth(rng.standard_normal((H, W))) * 30.0
    truth = central + np.where(live, _smooth(rng.standard_normal((H, W))) * 30.0,
                               rng.standard_normal((H, W)) * 2.0)

    live_idx = np.flatnonzero(live.ravel())
    corr, indep = [], []
    for _ in range(M):
        pert = np.zeros(H * W)
        pert[live_idx] = (_smooth(rng.standard_normal((H, W))) * 30.0).ravel()[live_idx]
        corr.append((central.ravel() + pert).reshape(H, W))
        shuf = np.zeros(H * W)
        shuf[live_idx] = rng.permutation(pert[live_idx])
        indep.append((central.ravel() + shuf).reshape(H, W))
    valid = np.ones((H, W), dtype=bool)
    return spread, valid, truth, np.stack(corr), np.stack(indep)


def _improvement(idx, truth, corr, indep, rng):
    rr, cc = np.unravel_index(idx, (H, W))
    coords = np.stack([rr, cc], axis=1).astype(float)
    pairs = sample_pairs(idx.size, 200000, rng=rng, coords=coords, max_dist=24)
    y = truth.ravel()[idx]
    X = corr.reshape(M, -1)[:, idx]
    Xn = indep.reshape(M, -1)[:, idx]
    vs = variogram_score(X, y, pairs)
    vs_null = variogram_score(Xn, y, pairs)
    return 1.0 - vs / max(vs_null, 1e-12), pairs


def test_spread_weighting_finds_the_live_region_that_uniform_sampling_misses():
    from validate_ensemble import _clustered_sample

    spread, valid, truth, corr, indep = build_case()
    rng = np.random.default_rng(1)
    idx_u = _clustered_sample(valid, 1500, rng, patch=64, n_patches=12)
    idx_w = _clustered_sample(valid, 1500, np.random.default_rng(1), patch=64,
                              n_patches=12, spread=spread)

    live_u = float(np.mean(spread.ravel()[idx_u] > 1e-3))
    live_w = float(np.mean(spread.ravel()[idx_w] > 1e-3))
    assert live_w > 0.9, f"weighted sample only {live_w:.2%} live"
    assert live_w > live_u + 0.5, f"weighting barely helped: {live_u:.2%} -> {live_w:.2%}"


def test_spread_weighting_recovers_the_improvement_uniform_sampling_dilutes():
    """How much of the *achievable* improvement each sampler recovers.

    The absolute size of the improvement is a property of this fixture, not a contract, so
    it is measured rather than asserted: an oracle sample drawn entirely from the live
    region defines what is there to be found, and the two samplers are judged against it.
    """
    from validate_ensemble import _clustered_sample

    spread, valid, truth, corr, indep = build_case()
    live_only = np.zeros((H, W), dtype=bool)
    live_only[LIVE] = True

    idx_o = _clustered_sample(live_only, 1500, np.random.default_rng(1), patch=48,
                              n_patches=12)
    idx_u = _clustered_sample(valid, 1500, np.random.default_rng(1), patch=64, n_patches=12)
    idx_w = _clustered_sample(valid, 1500, np.random.default_rng(1), patch=64,
                              n_patches=12, spread=spread)

    imp_o, _ = _improvement(idx_o, truth, corr, indep, np.random.default_rng(2))
    imp_u, _ = _improvement(idx_u, truth, corr, indep, np.random.default_rng(2))
    imp_w, _ = _improvement(idx_w, truth, corr, indep, np.random.default_rng(2))

    assert imp_o > 0.05, f"fixture has no improvement to find ({imp_o:.3f})"
    assert imp_w >= 0.75 * imp_o, (
        f"spread-weighted recovered only {imp_w / imp_o:.0%} of the achievable "
        f"improvement ({imp_w:.3f} of {imp_o:.3f})")
    assert imp_u <= 0.5 * imp_o, (
        f"uniform sampling was supposed to dilute the signal but recovered "
        f"{imp_u / imp_o:.0%} ({imp_u:.3f} of {imp_o:.3f})")


def test_informative_pair_fraction_sees_the_dilution():
    spread, valid, truth, corr, indep = build_case()
    rng = np.random.default_rng(3)
    idx = np.arange(H * W)
    coords = np.stack(np.unravel_index(idx, (H, W)), axis=1).astype(float)
    pairs = sample_pairs(idx.size, 200000, rng=rng, coords=coords, max_dist=24)
    frac = informative_pair_fraction(spread.ravel(), pairs)
    # The live patch is (48/256)^2 ≈ 3.5% of the map, so almost no random pair has both
    # endpoints in it.
    assert frac < 0.10, frac

    all_live = informative_pair_fraction(np.ones(H * W), pairs)
    assert all_live == pytest.approx(1.0)


def test_variogram_score_is_a_mean_not_a_sum():
    """A sum makes the score depend on how many pairs survived the distance filter."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.standard_normal(n)
    X = rng.standard_normal((8, n))
    p_small = sample_pairs(n, 200, rng=np.random.default_rng(1))
    p_big = sample_pairs(n, 4000, rng=np.random.default_rng(2))
    s_small = variogram_score(X, y, p_small)
    s_big = variogram_score(X, y, p_big)
    assert s_small == pytest.approx(s_big, rel=0.25), (s_small, s_big)


def test_weights_do_not_change_the_scale_of_the_score():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.standard_normal(n)
    X = rng.standard_normal((8, n))
    pairs = sample_pairs(n, 2000, rng=np.random.default_rng(1))
    plain = variogram_score(X, y, pairs)
    uniform_w = variogram_score(X, y, pairs, weights=np.full(len(pairs[0]), 3.7))
    assert plain == pytest.approx(uniform_w)


def test_empty_pair_set_is_nan_not_a_crash():
    y = np.zeros(4)
    X = np.zeros((3, 4))
    empty = (np.array([], dtype=int), np.array([], dtype=int))
    assert np.isnan(variogram_score(X, y, empty))
    assert np.isnan(informative_pair_fraction(np.ones(4), empty))


def test_degenerate_spread_falls_back_to_uniform_sampling():
    """No spread anywhere must not crash or return an empty sample."""
    from validate_ensemble import _clustered_sample

    valid = np.ones((H, W), dtype=bool)
    idx = _clustered_sample(valid, 500, np.random.default_rng(0), patch=64,
                            spread=np.zeros((H, W)))
    assert idx.size > 0
    assert np.all(idx < H * W)
