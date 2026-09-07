"""The two conv-spline gates, checked on forecasts whose answer is known by construction.

These metrics decide which experiments get adopted, so the risk is not that they crash --
it is that they read plausibly while measuring the wrong thing. Two failure modes get their
own tests: a metric that moves when only the *export grid* moved (E0 would then be recorded
as a model fix), and a PIT-structure statistic that calls sampling noise "structure".
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qf_diagnostics import (  # noqa: E402
    F_MAX_DENSITY, NEEDLE_EPS, REF_U, fence_gate, fence_stats, implied_density,
    interp_to, needle_mass, pit_structure, zero_leak,
)


def clean_forecast(u, n_px=500, scale=0.05, centre=0.3):
    """A smooth logistic-ish quantile function: no collapsed segments anywhere."""
    z = np.log(np.clip(u, 1e-6, 1 - 1e-6) / (1 - np.clip(u, 1e-6, 1 - 1e-6)))
    return (centre + scale * z)[:, None] * np.ones((1, n_px))


def fenced_forecast(u, n_px=500, n_needles=5, mass=0.30):
    """The pathology, built deliberately: ``mass`` of the probability inside needle-wide gaps.

    Levels are collapsed toward each other in the middle of the grid, exactly as a positive-
    derivative constraint does when the target wants a flat segment.
    """
    q = clean_forecast(u, n_px)
    mid = u.size // 2
    lo, hi = mid - n_needles, mid + n_needles
    anchor = q[mid]
    step = NEEDLE_EPS * 0.2
    for k, i in enumerate(range(lo, hi + 1)):
        q[i] = anchor + (k - n_needles) * step
    q = np.maximum.accumulate(q, axis=0)
    return q


# ------------------------------------------------------------------ needles

def test_a_clean_forecast_has_no_needles():
    u = REF_U
    q = clean_forecast(u)
    assert float(needle_mass(u, q).max()) < 1e-9
    assert float(np.nanmax(implied_density(u, q))) < F_MAX_DENSITY


def test_the_fence_is_detected_and_its_mass_is_about_right():
    u = REF_U
    q = fenced_forecast(u, n_needles=8)
    nm = needle_mass(u, q)
    # 16 collapsed segments out of 256, so the mass they carry is what the grid assigns them.
    dp = np.diff(u)
    expected = dp[u.size // 2 - 8:u.size // 2 + 8].sum()
    assert float(nm.mean()) == pytest.approx(expected, rel=0.25), f"{nm.mean()} vs ~{expected}"
    assert float(np.nanmax(implied_density(u, q))) > F_MAX_DENSITY


def test_max_density_flags_a_spike_sharper_than_the_noise_supports():
    u = REF_U
    q = fenced_forecast(u)
    s = fence_stats(u, q, "x")
    assert s["max_density_p99_x"] > F_MAX_DENSITY
    assert s["over_f_max_frac_x"] > 0.9
    clean = fence_stats(u, clean_forecast(u), "x")
    assert clean["over_f_max_frac_x"] == 0.0, "control: a smooth forecast must not trip it"


def test_gap_frac_zero_stays_below_gap_frac_needle():
    """The signature that separates a monotonicity artefact from a genuine atom."""
    s = fence_stats(REF_U, fenced_forecast(REF_U), "x")
    assert s["gap_frac_zero_x"] < s["gap_frac_needle_x"]


# ------------------------------------------------------------------ the export-grid trap

def test_reference_reading_survives_a_change_of_export_grid():
    """The point of the two readings. Re-spacing the levels must NOT move the ``_ref`` gate.

    Both grids describe the same underlying distribution, so a metric that is a property of
    the forecast has to agree. Without this, E0 -- which changes only the export -- would be
    recorded as having fixed the model.
    """
    from scipy.stats import norm
    coarse = norm.cdf(np.linspace(norm.ppf(1e-4), norm.ppf(1 - 1e-4), 64))
    skewed = np.unique(np.concatenate([
        norm.cdf(np.linspace(norm.ppf(1e-4), norm.ppf(0.1), 20)),
        norm.cdf(np.linspace(norm.ppf(0.1), norm.ppf(0.9), 24)),
        norm.cdf(np.linspace(norm.ppf(0.9), norm.ppf(1 - 1e-4), 20))]))

    a = fence_gate(coarse, clean_forecast(coarse))
    b = fence_gate(skewed, clean_forecast(skewed))
    assert a["max_density_p99_ref"] == pytest.approx(b["max_density_p99_ref"], rel=0.10), (
        f"the reference reading moved with the export grid: "
        f"{a['max_density_p99_ref']:.1f} vs {b['max_density_p99_ref']:.1f}")


def test_interp_to_preserves_density_when_subdividing():
    """Why the reference reading works at all: subdividing a segment leaves dp/dq alone."""
    from scipy.stats import norm
    coarse = norm.cdf(np.linspace(norm.ppf(1e-3), norm.ppf(1 - 1e-3), 32))
    q = clean_forecast(coarse, n_px=10)
    dense = np.sort(np.unique(np.concatenate([coarse, (coarse[:-1] + coarse[1:]) / 2])))
    d_coarse = np.nanmax(implied_density(coarse, q))
    d_dense = np.nanmax(implied_density(dense, interp_to(coarse, q, dense)))
    assert d_dense == pytest.approx(d_coarse, rel=1e-6)


# ------------------------------------------------------------------ PIT structure

def test_uniform_pit_reads_as_noise_at_every_bin_count():
    rng = np.random.default_rng(0)
    s = pit_structure(rng.random(200_000))
    for b in (20, 40, 60):
        assert 0.5 < s[f"pit_rms_se_{b}"] < 1.6, f"{b} bins: {s[f'pit_rms_se_{b}']:.2f}"
    assert 0.7 < s["pit_growth_vs_noise"] < 1.4, s["pit_growth_vs_noise"]


def test_raw_rms_grows_with_bins_under_pure_noise_which_is_why_it_is_not_the_statistic():
    """The control for the whole design: raw RMS is not comparable across bin counts."""
    rng = np.random.default_rng(1)
    s = pit_structure(rng.random(200_000))
    assert s["pit_rms_60"] > 1.4 * s["pit_rms_20"], "raw RMS did not grow -- test is wrong"
    assert abs(s["pit_rms_se_60"] - s["pit_rms_se_20"]) < 0.5, "SE units did not stabilise it"


def test_real_structure_is_distinguished_from_noise():
    """Three spikes of resolvable width at the incumbent's measured peak locations.

    sigma = 0.02 is chosen to match the growth the incumbent actually shows (~0.75); at
    sigma 0.004 growth reads 0.97 because a feature narrower than the finest bin occupies one
    bin at every resolution and therefore grows exactly like noise. That is a real limit of
    the growth statistic, which is why `rms_se` is the primary test and growth the secondary.
    """
    rng = np.random.default_rng(2)
    base = rng.random(200_000)
    spikes = np.concatenate([rng.normal(loc, 0.02, 20_000) for loc in (0.25, 0.51, 0.73)])
    s = pit_structure(np.clip(np.concatenate([base, spikes]), 0, 1))
    assert s["pit_rms_se_60"] > 3.0, "real structure must read far above the noise level"
    assert s["pit_growth_vs_noise"] < 0.85, s["pit_growth_vs_noise"]


def test_a_delta_narrow_spike_is_caught_by_rms_even_though_growth_misses_it():
    """The documented blind spot, pinned so it cannot be forgotten or silently widen."""
    rng = np.random.default_rng(6)
    base = rng.random(200_000)
    spikes = np.concatenate([rng.normal(loc, 0.004, 20_000) for loc in (0.25, 0.51, 0.73)])
    s = pit_structure(np.clip(np.concatenate([base, spikes]), 0, 1))
    assert s["pit_rms_se_60"] > 10.0, "rms_se must still see it"
    assert s["pit_growth_vs_noise"] > 0.9, (
        "if this ever drops, the blind spot closed and the docstring is now wrong")


def test_pit_mean_bias_is_visible_at_every_resolution():
    rng = np.random.default_rng(3)
    biased = np.clip(rng.beta(2.2, 1.8, 100_000), 0, 1)
    s = pit_structure(biased)
    assert s["pit_rms_se_20"] > 2.0


# ------------------------------------------------------------------ the zero-crossing leak

def test_zero_leak_is_one_for_a_forecast_that_matches_the_truth():
    rng = np.random.default_rng(4)
    n = 20_000
    hm0 = np.full(n, 0.2)
    u = REF_U
    q = clean_forecast(u, n_px=n, scale=0.01, centre=0.2)
    # Draw the truth from the forecast itself: predicted and observed mass must then agree.
    y = np.array([np.interp(v, u, q[:, i]) for i, v in enumerate(rng.random(n))])
    out = zero_leak(u, q, y, hm0)
    assert out["zero_leak_neg_ratio"] == pytest.approx(1.0, abs=0.15), out
    assert out["zero_leak_pos_ratio"] == pytest.approx(1.0, abs=0.30), out


def test_zero_leak_catches_a_forecast_shifted_across_zero():
    rng = np.random.default_rng(5)
    n = 20_000
    hm0 = np.full(n, 0.2)
    u = REF_U
    q = clean_forecast(u, n_px=n, scale=0.01, centre=0.2 - 0.01)   # shifted low
    # The truth is mostly small increases with an 18% noise-driven decrease tail, which is
    # what the observable actually looks like. A truth that never decreases would make the
    # observed denominator exactly zero and the ratio NaN -- no finding, just a broken test.
    y = hm0 + np.where(rng.random(n) < 0.18, -np.abs(rng.normal(0, 0.001, n)),
                       np.abs(rng.normal(0, 0.002, n)))
    out = zero_leak(u, q, y, hm0)
    assert out["zero_leak_neg_obs"] > 0, "the control has no observed mass to compare against"
    assert out["zero_leak_neg_ratio"] > 1.5, out
