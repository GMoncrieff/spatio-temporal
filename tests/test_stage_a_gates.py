"""Stage A: the scoring instruments, tested in the direction they were failing.

Three gates on the published cards passed *because* of the defect this round exists to
fix — T8.2 and T8.3 reward a silent far field, T6.4 was satisfied at 254x the observed
quantity — and T2.5 was scored on 401 rank bins holding 158 observations, which is valid
but close to powerless against the miscalibration those histograms actually carry. Each
test below plants the geometry that made the old instrument wrong and asserts that the new
one calls it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.ensemble.aggregate import pool_rank_bins, rank_histogram_test  # noqa: E402
from src.ensemble.validate import near_far_contrast, remote_band_error  # noqa: E402


# ---------------------------------------------------------------------------- T8.2
def test_remote_band_error_passes_a_faithful_far_field():
    err, _ = remote_band_error(1.7e-4, 1.7e-4, 1.8e-7, 1.8e-7)
    assert err == pytest.approx(0.0, abs=1e-12)


def test_remote_band_error_catches_a_silent_far_field():
    """The global reading: 2.3% of the observed increase rate, 28x the decrease rate.

    The old gate (member <= 0.002) passed this at 3.8e-06 — emphatically, and for the
    wrong reason.
    """
    err, notes = remote_band_error(3.84e-6, 1.699e-4, 4.98e-6, 1.785e-7)
    assert err > 0.301
    assert "x0.0226" in notes[0]


def test_remote_band_error_is_two_sided():
    """Too loud is as wrong as too quiet, and by the same amount."""
    quiet, _ = remote_band_error(1e-4 / 8, 1e-4, 1e-6, 1e-6)
    loud, _ = remote_band_error(1e-4 * 8, 1e-4, 1e-6, 1e-6)
    assert quiet == pytest.approx(loud)
    assert quiet > 0.301


def test_remote_band_error_falls_back_to_the_ceiling_when_observed_is_zero():
    """Southern Africa's branch: no observed remote change, so no ratio exists."""
    ok, notes = remote_band_error(0.0005, 0.0, 0.0, 0.0)
    assert ok == pytest.approx(0.0)
    assert "ceiling" in notes[0]
    invented, _ = remote_band_error(0.03, 0.0, 0.0, 0.0)
    assert invented > 0.301


# ---------------------------------------------------------------------------- T8.3
def test_near_far_contrast_catches_an_over_concentrated_ensemble():
    """Global: member contrast 90,596 against an observed 1,527, and the old gate passed."""
    member, observed, ratio = near_far_contrast(0.3478, 3.8395e-6, 0.2594, 1.6993e-4)
    assert member == pytest.approx(90595.6, rel=1e-3)
    assert observed == pytest.approx(1526.8, rel=1e-3)
    assert ratio > 2.0


def test_near_far_contrast_accepts_a_matching_contrast():
    _, _, ratio = near_far_contrast(0.26, 1.7e-4, 0.26, 1.7e-4)
    assert 0.5 <= ratio <= 2.0


def test_near_far_contrast_is_nan_without_an_observed_contrast():
    _, _, ratio = near_far_contrast(0.2, 1e-6, 0.2, 0.0)
    assert np.isnan(ratio)


# ---------------------------------------------------------------------------- T2.5
def _domed_hist(n, K, rng, concentration=0.25):
    """n draws whose ranks cluster in the middle — an over-dispersed ensemble."""
    ranks = np.clip((rng.normal(0.5, concentration, n) * K).astype(int), 0, K - 1)
    return np.bincount(ranks, minlength=K)


def test_pool_rank_bins_conserves_counts_and_lifts_the_expected_occupancy():
    rng = np.random.default_rng(0)
    hist = np.bincount(rng.integers(0, 401, 158), minlength=401)
    pooled, expected = pool_rank_bins(hist, min_expected=16.0)
    assert pooled.sum() == hist.sum()
    assert expected.sum() == pytest.approx(hist.sum())
    assert expected.min() >= 16.0
    assert pooled.size < hist.size


def test_pooling_holds_size_and_buys_power():
    """Why the pooling is there — and it is not the reason it first looked like.

    401 bins over 158 units is 0.39 expected per cell, which reads as a textbook violation
    of the chi-square conditions. It is not one: for *equiprobable* cells the
    approximation survives, and the unpooled test rejects uniform ranks at very close to
    its nominal size. What it cannot do is see a broad smooth dome spread thinly over 401
    cells. Both halves are asserted, because only the second justifies the change.
    """
    from scipy.stats import chisquare

    rng = np.random.default_rng(7)
    n, K, reps = 158, 401, 400

    unpooled_size = pooled_size = 0
    for _ in range(reps):
        hist = np.bincount(rng.integers(0, K, n), minlength=K)
        unpooled_size += chisquare(hist, np.full(K, n / K))[1] < 0.01
        pooled_size += rank_histogram_test(hist, min_expected=16.0)["p_value"] < 0.01
    # Both hold their nominal 1%; the unpooled test was never invalid.
    assert unpooled_size / reps < 0.04
    assert pooled_size / reps < 0.04

    unpooled_power = pooled_power = 0
    for _ in range(reps):
        hist = _domed_hist(n, K, rng)
        unpooled_power += chisquare(hist, np.full(K, n / K))[1] < 0.01
        pooled_power += rank_histogram_test(hist, min_expected=16.0)["p_value"] < 0.01
    # Measured at ~0.59 unpooled against ~0.90 pooled on the same draws.
    assert pooled_power > unpooled_power + 0.15 * reps


def test_pooled_chi_square_still_catches_real_over_dispersion():
    """Africa and global rank histograms are domed: centre bins ~2x uniform, tails ~0.1x.

    Pooling must sharpen the finding, never soften it — the deviation is what the round is
    for, and an instrument change that made it disappear would be the wrong change.
    """
    rng = np.random.default_rng(3)
    for n in (158, 804):
        test = rank_histogram_test(_domed_hist(n, 401, rng), min_expected=16.0)
        assert test["p_value"] < 0.01
        assert test["min_expected"] >= 16.0
        assert test["n"] == n


def test_rank_histogram_test_reports_what_it_scored_on():
    rng = np.random.default_rng(1)
    test = rank_histogram_test(np.bincount(rng.integers(0, 401, 804), minlength=401))
    assert test["n_bins"] == 50
    assert test["n"] == 804


# ---------------------------------------------------------------------------- T3.2
def test_structure_budget_falls_with_separation_and_is_read_at_the_scored_lag():
    """Africa's own fit: 57% of the variance is still correlated at 25 px, 27% at 78 px.

    T3.2 caps pairs at half the member practical range but scores at their *mean*
    separation, and the budget is steep enough between the two that reading it at the cap
    would overstate the difficulty by about a factor of two.
    """
    from validate_ensemble import _structure_budget

    fit = {"nugget": 0.01172, "var_short": 0.02436, "range_short_px": 11.3216,
           "var_long": 0.05877, "range_long_px": 85.36, "sill": 0.09485}
    b25, b78 = _structure_budget(fit, 25.0), _structure_budget(fit, 78.0)
    assert b25 == pytest.approx(0.571, abs=0.01)
    assert b78 == pytest.approx(0.269, abs=0.01)
    assert b25 > b78 > _structure_budget(fit, 200.0)


def test_structure_budget_is_nan_without_a_fit():
    from validate_ensemble import _structure_budget

    assert np.isnan(_structure_budget(None, 25.0))
    assert np.isnan(_structure_budget({"sill": 0.0}, 25.0))


def test_structure_budget_is_regional_which_is_why_a_fixed_threshold_misleads():
    """Southern Africa's 14% budget at 25 px was read as "the target is unreachable".

    Africa's fit gives 57% at the same lag. The two extents were being scored against the
    same 0.30 threshold with a 4x difference in what the data made available.
    """
    from validate_ensemble import _structure_budget

    southern = {"nugget": 0.0014, "var_short": 0.0030, "range_short_px": 6.25,
                "var_long": 0.0056, "range_long_px": 20.0, "sill": 0.0101}
    africa = {"nugget": 0.01172, "var_short": 0.02436, "range_short_px": 11.3216,
              "var_long": 0.05877, "range_long_px": 85.36, "sill": 0.09485}
    assert _structure_budget(southern, 25.0) < 0.5 * _structure_budget(africa, 25.0)


# ---------------------------------------------------------------------------- T4.2
def test_population_spread_matches_the_two_piece_normal_analytically():
    """With no shape and no clip binding, the quadrature must reproduce the closed form.

    For x = cen + sigma(z) * z with a two-piece normal, E[x] and Var[x] have exact
    expressions; a symmetric case (sl == sr) reduces to the plain normal, which is the
    cleanest thing to pin the integrator against.
    """
    from validate_ensemble import population_spread

    cen = np.full((4, 4), 0.5)
    sd = 0.01
    got = population_spread(cen, np.full_like(cen, sd), np.full_like(cen, sd))
    assert np.allclose(got, sd, rtol=1e-6)


def test_population_spread_is_asymmetric_when_the_half_widths_are():
    from validate_ensemble import population_spread

    cen = np.full((3, 3), 0.5)
    sym = population_spread(cen, np.full_like(cen, 0.01), np.full_like(cen, 0.01))
    asym = population_spread(cen, np.full_like(cen, 0.01), np.full_like(cen, 0.03))
    assert (asym > sym).all()


def test_population_spread_sees_the_clip():
    """Members are clipped to [0,1], so a marginal pressed against a bound is narrower.

    This is the same clip that makes the physical floor exact (4-diag), and T4.2 has to
    measure the distribution the sampler actually produces, not the one before clipping.
    """
    from validate_ensemble import population_spread

    wide = np.full((3, 3), 0.4)
    interior = population_spread(np.full((3, 3), 0.5), wide, wide)
    at_bound = population_spread(np.full((3, 3), 0.02), wide, wide)
    assert at_bound.max() < interior.min()


def test_population_spread_does_not_depend_on_rho():
    """The property the whole T4.2 rewrite rests on, asserted rather than assumed.

    `population_spread` never sees rho — it integrates the marginal — so this is a
    structural guarantee, and the test exists to keep it structural. Measured on two real
    M=50 Africa ensembles differing only in the coupling (0.9 against the measured
    {10: 0.374, 15: 0.347, 20: 0.769}), the gate returned 0.7304648026222341 for both to
    every digit, while the old sample statistic read 0.6194 and 0.4540.
    """
    import inspect

    from validate_ensemble import population_spread

    sig = inspect.signature(population_spread)
    assert "rho" not in sig.parameters
    src = inspect.getsource(population_spread)
    assert "rho" not in src.split('"""')[2]  # not in the body, only in the docstring
