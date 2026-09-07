"""The distributional scorer's arithmetic.

Every configuration in this phase is ranked on numbers this file produces, so a bias here
reorders the slate while looking entirely plausible. The CRPS closed form in particular was
wrong on its first writing -- the crossing point is clipped to a segment end far more often
than it lands inside one, and the second piece's error at its own origin is then not zero --
and the only thing that caught it was a second implementation to compare against.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.models.quantile_spline import output_u_grid  # noqa: E402
from score_distributional_model import (  # noqa: E402
    crps_piecewise_linear, pit, quantile_at,
)


def grid_and_q(n_px=400, n_levels=64, seed=0):
    rng = np.random.default_rng(seed)
    u = output_u_grid(n_levels)
    q = np.sort(rng.random((n_levels, n_px)) * rng.uniform(0.01, 0.9, n_px), axis=0)
    return u, q.astype(np.float32)


def crps_reference(u, q, y, n_grid=400_001):
    """Dense numerical integral of the pinball loss. Shares no code with the closed form."""
    uf = np.linspace(0.0, 1.0, n_grid)
    out = np.empty(y.size)
    for i in range(y.size):
        e = y[i] - np.interp(uf, u, q[:, i])
        out[i] = 2.0 * np.trapezoid(e * (uf - (e < 0)), uf)
    return out


# ------------------------------------------------------------------ the ranking metric

@pytest.mark.parametrize("where", ["inside", "outside", "on a knot"])
def test_crps_closed_form_matches_a_dense_integral(where):
    u, q = grid_and_q()
    if where == "inside":
        rng = np.random.default_rng(1)
        y = q[0] + (q[-1] - q[0]) * rng.random(q.shape[1])
    elif where == "outside":
        # Beyond the represented range on both sides: the flat tails past u[0] and u[-1] are
        # 2e-4 of the probability and are exactly the part the far-field question is about.
        y = np.where(np.arange(q.shape[1]) % 2 == 0, q[0] - 0.05, q[-1] + 0.05)
    else:
        y = q[31].copy()
    got = crps_piecewise_linear(u, q, y)
    want = crps_reference(u, q, y)
    assert np.max(np.abs(got - want)) < 1e-6


def test_crps_of_a_degenerate_forecast_is_the_absolute_error():
    u = output_u_grid(64)
    y = np.linspace(0.05, 0.95, 50)
    q = np.tile(np.full(y.size, 0.4, dtype=np.float32), (u.size, 1))
    # 1e-7, not 1e-9: the raster is float32, where 0.4 is 0.40000000596.
    np.testing.assert_allclose(crps_piecewise_linear(u, q, y), np.abs(y - 0.4), atol=1e-7)


def test_crps_is_minimised_by_the_distribution_the_data_came_from():
    """Propriety, measured. A sharper or a wider forecast must both score worse."""
    from scipy.stats import norm

    u = output_u_grid(64)
    n = 40000
    rng = np.random.default_rng(5)
    y = 0.4 + 0.05 * rng.standard_normal(n)
    scores = {}
    for f in (0.4, 0.7, 1.0, 1.4, 2.5):
        q = np.tile((0.4 + 0.05 * f * norm.ppf(u))[:, None], (1, n)).astype(np.float32)
        scores[f] = float(crps_piecewise_linear(u, q, y).mean())
    assert min(scores, key=scores.get) == 1.0, scores


def test_crps_is_chunk_invariant():
    u, q = grid_and_q(n_px=997)
    y = q[20].copy()
    np.testing.assert_allclose(crps_piecewise_linear(u, q, y, chunk=100),
                               crps_piecewise_linear(u, q, y, chunk=10_000), rtol=0, atol=1e-12)


# ------------------------------------------------------------------ PIT and exceedance

def test_pit_inverts_the_quantile_function():
    u, q = grid_and_q()
    for level in (0.001, 0.025, 0.25, 0.5, 0.975, 0.999):
        back = pit(u, q, quantile_at(u, q, level))
        assert np.max(np.abs(back - level)) < 1e-9


def test_pit_saturates_outside_the_represented_range():
    u, q = grid_and_q()
    assert np.all(pit(u, q, q[0] - 0.1) == 0.0)
    assert np.all(pit(u, q, q[-1] + 0.1) == 1.0)


def test_pit_is_uniform_when_the_forecast_is_right():
    from scipy.stats import norm

    u = output_u_grid(64)
    n = 20000
    rng = np.random.default_rng(9)
    y = 0.4 + 0.05 * rng.standard_normal(n)
    q = np.tile((0.4 + 0.05 * norm.ppf(u))[:, None], (1, n)).astype(np.float32)
    p = pit(u, q, y)
    ks = np.max(np.abs(np.sort(p) - (np.arange(n) + 0.5) / n))
    assert ks < 0.02, f"PIT KS {ks:.4f} on a correctly specified forecast"


def test_exceedance_probability_is_one_minus_the_cdf():
    """The closed form that replaces predict_change_rates.py, checked against sampling."""
    from scipy.stats import norm

    u = output_u_grid(64)
    n = 30000
    rng = np.random.default_rng(11)
    hm0 = 0.30
    col = hm0 + 0.05 * norm.ppf(u)
    q = np.tile(col[:, None], (1, n)).astype(np.float32)
    # Interpolate, do not snap to the level above: snapping biases every draw upward and
    # would make the closed form look wrong when it is the sampler that is.
    draws = np.interp(rng.random(n), u, col)
    for thr in (0.01, 0.05):
        closed = float((1.0 - pit(u, q, np.full(n, hm0 + thr))).mean())
        sampled = float(((draws - hm0) > thr).mean())
        assert abs(closed - sampled) < 0.01, f"thr={thr}: {closed:.4f} vs {sampled:.4f}"


# ------------------------------------------------------------------ the grid itself

def test_output_grid_is_strictly_increasing_and_carries_the_gates():
    """A duplicated level makes the segment slope 0/0 and CRPS NaN for every pixel.

    That is a metric failure that reads exactly like a model failure, so it is pinned here
    rather than left to be noticed in a scorecard.
    """
    for n in (16, 32, 64, 128):
        u = output_u_grid(n)
        assert u.size == n
        assert np.all(np.diff(u) > 0)
        for gate in (0.025, 0.5, 0.975):
            assert np.isclose(u, gate).any(), f"n={n} lost the {gate} gate"


# ------------------------------------------------------------ conv-spline: the two gates

def test_gate_stats_are_produced_and_named_for_both_readings():
    """The scorecard must actually carry the gates, under both the export and ref readings.

    A gate computed in a side script is a gate that stops being run. This checks it is wired
    into the scorer's own record, and that the two readings are separately named -- reporting
    only one is how an export-grid re-spacing (E0) gets recorded as a model fix.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from score_distributional_model import gate_stats

    u, q = grid_and_q(n_px=2000)
    rng = np.random.default_rng(7)
    y = np.array([np.interp(v, u, q[:, i]) for i, v in enumerate(rng.random(q.shape[1]))])
    cell = {"qf": q, "observed": y, "hm_t0": np.full(y.size, 0.2)}
    out = gate_stats(u, cell, np.ones(y.size, dtype=bool))

    for stat in ("needle_mass_median", "needle_mass_p90", "max_density_p99", "over_f_max_frac"):
        for reading in ("export", "ref"):
            assert f"{stat}_{reading}" in out, f"missing {stat}_{reading}"
    for k in ("pit_rms_se_20", "pit_rms_se_60", "pit_growth_vs_noise",
              "zero_leak_neg_ratio", "zero_leak_pos_ratio"):
        assert k in out, f"missing {k}"


def test_gate_stats_declines_a_stratum_too_small_to_measure():
    """A PIT histogram over 40 pixels is noise. Return nothing rather than a number."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from score_distributional_model import gate_stats

    u, q = grid_and_q(n_px=2000)
    y = q[32]
    cell = {"qf": q, "observed": y, "hm_t0": np.full(y.size, 0.2)}
    sel = np.zeros(y.size, dtype=bool)
    sel[:40] = True
    assert gate_stats(u, cell, sel) == {}
