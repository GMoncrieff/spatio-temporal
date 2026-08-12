"""Phase 3 tests — the marginal reproduces all three published quantiles exactly."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.copula import (  # noqa: E402
    DEFAULT_SCALE,
    INT16_SENTINEL,
    copula_sample_member,
    dequantize,
    fit_marginal_two_piece_normal,
    marginal_from_z,
    marginal_ppf,
    quantization_error_bound,
    quantize,
)


def _triples(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    central = rng.uniform(0.02, 0.9, n).astype(np.float32)
    w_lo = rng.uniform(0.001, 0.02, n).astype(np.float32)
    w_up = rng.uniform(0.001, 0.05, n).astype(np.float32)   # deliberately asymmetric
    return np.maximum(central - w_lo, 0), central, np.minimum(central + w_up, 1)


def test_median_is_exactly_the_central_forecast():
    lower, central, upper = _triples()
    p = fit_marginal_two_piece_normal(lower, central, upper)
    med = marginal_ppf(np.full(central.shape, 0.5), p)
    assert np.allclose(med, central, atol=1e-6)


def test_tails_reproduce_the_published_quantiles():
    lower, central, upper = _triples()
    p = fit_marginal_two_piece_normal(lower, central, upper)
    assert np.allclose(marginal_ppf(np.full(central.shape, 0.025), p), lower, atol=1e-5)
    assert np.allclose(marginal_ppf(np.full(central.shape, 0.975), p), upper, atol=1e-5)


def test_asymmetry_is_preserved():
    p = fit_marginal_two_piece_normal(np.array([0.4]), np.array([0.5]), np.array([0.9]))
    assert p["scale_right"][0] > 3 * p["scale_left"][0]


def test_ppf_is_monotone_in_u():
    lower, central, upper = _triples(n=5)
    p = fit_marginal_two_piece_normal(lower, central, upper)
    us = np.linspace(0.001, 0.999, 200)
    vals = np.stack([marginal_ppf(np.full(5, u), p) for u in us])
    assert np.all(np.diff(vals, axis=0) >= -1e-7)


def test_zero_field_maps_every_pixel_to_central():
    lower, central, upper = _triples()
    p = fit_marginal_two_piece_normal(lower, central, upper)
    member = copula_sample_member(np.zeros_like(central), p)
    assert np.allclose(member, central, atol=1e-6)


def test_member_median_over_many_draws_matches_central():
    lower, central, upper = _triples(n=200)
    p = fit_marginal_two_piece_normal(lower, central, upper)
    rng = np.random.default_rng(0)
    members = np.stack([copula_sample_member(rng.standard_normal(central.shape), p)
                        for _ in range(999)])
    med = np.median(members, axis=0)
    assert np.nanmax(np.abs(med - central)) < 0.01


def test_ensemble_tails_match_the_marginal_tails():
    lower, central, upper = _triples(n=50)
    p = fit_marginal_two_piece_normal(lower, central, upper)
    rng = np.random.default_rng(1)
    members = np.stack([copula_sample_member(rng.standard_normal(central.shape), p)
                        for _ in range(4000)])
    assert np.nanmax(np.abs(np.percentile(members, 2.5, axis=0) - lower)) < 0.01
    assert np.nanmax(np.abs(np.percentile(members, 97.5, axis=0) - upper)) < 0.01


def test_values_stay_within_bounds():
    p = fit_marginal_two_piece_normal(np.array([0.0]), np.array([0.001]), np.array([1.0]))
    v = marginal_from_z(np.array([-40.0, 40.0]), p)
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_int16_roundtrip_is_within_half_a_step():
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 1, 10_000).astype(np.float32)
    q = quantize(v)
    back = dequantize(q)
    assert np.nanmax(np.abs(back - v)) <= quantization_error_bound(DEFAULT_SCALE) + 1e-9


def test_sentinel_survives_the_roundtrip():
    v = np.array([np.nan, 0.5], dtype=np.float32)
    q = quantize(v)
    assert q[0] == INT16_SENTINEL
    back = dequantize(q)
    assert np.isnan(back[0]) and abs(back[1] - 0.5) < DEFAULT_SCALE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
