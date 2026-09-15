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
    F_MAX_DENSITY, NEEDLE_EPS, REF_U, fence_gate, fence_per_pixel, fence_reduce,
    fence_stats, implied_density, interp_to, needle_mass,
    pit_structure, zero_leak,
)
from src.models.quantile_spline import output_u_grid  # noqa: E402


def _grid(n_px=1000, n_levels=64, seed=0):
    """A random monotone quantile function -- irregular on purpose, so a blocked reduction
    that mishandled a column would not be hidden by every column being the same."""
    rng = np.random.default_rng(seed)
    u = output_u_grid(n_levels)
    q = np.sort(rng.random((n_levels, n_px)) * rng.uniform(0.01, 0.9, n_px), axis=0)
    return u, q.astype(np.float32)


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


# --------------------------------------------------- blocking the gate must change nothing

@pytest.mark.parametrize("block", [1, 7, 100, 999, 1000, 1001, 100_000])
def test_fence_per_pixel_is_exactly_block_invariant(block):
    """The block loop bounds ``[n_seg, n_px]``; it must not move a single bit.

    Measured motivation: the reference reading is 255 segments, so one Africa fold at once
    builds a 13.6 GB float64 density array -- the largest term in the scorer's measured
    59.1 GB peak. Every quantity is pixel-independent, so blocking is an identity rather than
    an approximation, and this is where that claim is checked.

    The three counts and the per-pixel maximum are asserted **bit-identical** -- they are
    order-independent reductions and nothing may move them. ``needle_mass`` is a sum, and
    NumPy's pairwise reduction blocks differently for a 1-pixel-wide array than for a wide
    one, so it is allowed one ULP and no more. Measured: it agrees exactly from block 7
    upward and differs by 1.1e-16 relative at block 1, which is the only slack here.
    """
    u, q = _grid(n_px=1000, seed=5)
    whole = fence_per_pixel(u, q, block=0)
    part = fence_per_pixel(u, q, block=block)
    assert set(whole) == set(part)
    for k in ("n_finite", "n_needle", "n_zero", "px_max_density"):
        assert np.array_equal(whole[k], part[k], equal_nan=True), k
    assert whole["needle_mass"] == pytest.approx(part["needle_mass"], rel=1e-15)


@pytest.mark.parametrize("block", [1, 333, 100_000])
def test_the_reference_reading_is_block_invariant_too(block):
    """``u_interp`` moves the interpolation inside the loop, which is the whole point."""
    u, q = _grid(n_px=1000, seed=6)
    whole = fence_per_pixel(u, q, u_interp=REF_U, block=0)
    part = fence_per_pixel(u, q, u_interp=REF_U, block=block)
    for k in ("n_finite", "n_needle", "n_zero", "px_max_density"):
        assert np.array_equal(whole[k], part[k], equal_nan=True), k
    assert whole["needle_mass"] == pytest.approx(part["needle_mass"], rel=1e-15)
    # and it must still equal the way the caller used to spell it: interpolate the whole
    # raster onto REF_U first, then read the fence off that grid.
    direct = fence_per_pixel(REF_U, interp_to(u, q, REF_U), block=0)
    for k in whole:
        assert np.array_equal(whole[k], direct[k], equal_nan=True), f"{k} vs the old spelling"


def test_blocking_never_materialises_the_whole_reference_grid(monkeypatch):
    """Prove the loop is real: the biggest array interp_to returns must be bounded by the
    block, or the identity above is passing on a code path that still allocates in full."""
    import src.qf_diagnostics as qd
    seen = []
    real = qd.interp_to

    def spy(u_src, q, u_dst):
        out = real(u_src, q, u_dst)
        seen.append(out.shape[1])
        return out

    monkeypatch.setattr(qd, "interp_to", spy)
    u, q = _grid(n_px=5000, seed=7)
    qd.fence_per_pixel(u, q, u_interp=qd.REF_U, block=1000)
    assert seen, "interp_to was never called"
    assert max(seen) <= 1000, f"a block of {max(seen)} pixels was interpolated whole"


def test_a_degenerate_pixel_counts_against_f_max_instead_of_vanishing():
    """A zero-width segment is infinite density: the strongest needle there is.

    ``fence_reduce`` used to filter it out before every density statistic, so the pixels that
    had fenced WORST were the ones the gate could not see. Measured consequence on b1_s42:
    34% of adjacent quantile levels exported to the same int16 code, and the density gate was
    reported over the remainder. This is the control for the fix.
    """
    u, q = _grid(n_px=200, seed=11)
    q = q.copy()
    q[30, :50] = q[29, :50]          # 50 pixels get one exactly-flat segment
    px = fence_per_pixel(u, q)
    assert (~np.isfinite(px["px_max_density"])).sum() == 50

    out = fence_reduce(px, "export")
    assert out["px_degenerate_frac_export"] == pytest.approx(50 / 200)
    # every degenerate pixel is over any ceiling, so the fraction cannot be below their share
    assert out["over_f_max_frac_export"] >= 50 / 200
    # and the surviving percentiles stay finite so they still rank the rest
    assert np.isfinite(out["max_density_p99_export"])


def test_no_degenerate_pixels_leaves_the_old_numbers_alone():
    """The fix must not move a reading that had nothing degenerate in it."""
    u, q = _grid(n_px=200, seed=12)
    px = fence_per_pixel(u, q)
    assert np.isfinite(px["px_max_density"]).all(), "fixture has a degenerate pixel"
    out = fence_reduce(px, "export")
    assert out["px_degenerate_frac_export"] == 0.0
    manual = float((px["px_max_density"] > F_MAX_DENSITY).mean())
    assert out["over_f_max_frac_export"] == pytest.approx(manual)


# ------------------------------------------- the support boundary is not the picket fence

def _clamped_forecast(u, n_px=200, lo=0.0):
    """A forecast whose lower tail is flat ON the clamp, as HM near zero produces."""
    q = clean_forecast(u, n_px, scale=0.02, centre=0.03)
    return np.maximum(q, lo)


def test_a_clamp_atom_is_reported_as_a_clamp_and_not_as_a_needle():
    """HM cannot go below 0, so mass at the floor is correct behaviour, not a fence.

    Measured on b1_s42: 100% of pixels whose two lowest quantile levels were identical had
    Q = 0.0 exactly there, and the clamp contributed 0.3% of the needle mass against the
    core's 73.3%. Counting it as degeneracy put a ~0.6 constant on every arm's gate.
    """
    u = REF_U
    q = _clamped_forecast(u)
    assert (q[0] == 0.0).all(), "the fixture is not actually clamped"
    px = fence_per_pixel(u, q, clamp=(0.0, 1.0))
    out = fence_reduce(px, "export")

    assert out["px_clamp_frac_export"] == 1.0, "the clamp was not detected"
    assert out["gap_frac_clamp_export"] > 0
    assert out["px_degenerate_frac_export"] == 0.0, "a clamp atom counted as degenerate"
    assert np.isfinite(out["max_density_p99_export"]), "the clamp poisoned the density"
    assert out["over_f_max_frac_export"] == 0.0, "the clamp counted against f_max"


def test_a_real_collapse_away_from_the_boundary_still_counts():
    """Prove the exclusion is narrow: only segments ON the boundary are spared."""
    u = REF_U
    q = _clamped_forecast(u)
    mid = u.size // 2
    q[mid + 1] = q[mid]                      # a genuine interior collapse, far from 0 and 1
    out = fence_reduce(fence_per_pixel(u, q, clamp=(0.0, 1.0)), "export")
    assert out["px_degenerate_frac_export"] == 1.0, "an interior collapse was excused"
    assert out["over_f_max_frac_export"] == 1.0
    assert out["px_clamp_frac_export"] == 1.0, "the clamp should still be reported too"


def test_the_clamp_exclusion_does_nothing_to_an_unclamped_forecast():
    """A forecast that never touches the boundary must read exactly as before."""
    u, q = _grid(n_px=300, seed=21)
    q = (q * 0.5 + 0.25).astype(np.float32)          # strictly inside (0, 1)
    out = fence_reduce(fence_per_pixel(u, q, clamp=(0.0, 1.0)), "export")
    assert out["px_clamp_frac_export"] == 0.0
    assert out["gap_frac_clamp_export"] == 0.0
    wide = fence_reduce(fence_per_pixel(u, q, clamp=(-1.0, 2.0)), "export")
    for k in out:
        assert out[k] == pytest.approx(wide[k], nan_ok=True), f"{k} moved with the clamp"
