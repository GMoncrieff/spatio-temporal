"""Phase 1c tests — variogram fitting recovers planted structure."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.fields import empirical_variogram_from_field, generate_correlated_field  # noqa: E402
from src.ensemble.variogram import (  # noqa: E402
    extrapolate_h20_check,
    fit_nugget_multirange_model,
    fits_to_field_params,
    nugget_two_range_model,
)


def test_recovers_planted_nugget_and_ranges():
    field = generate_correlated_field(
        512, 512, ranges_px=[4.0, 60.0], weights=[0.5, 0.4], nugget=0.1,
        rng=np.random.default_rng(0), wrap_lon=False,
    )
    centres, gamma, counts = empirical_variogram_from_field(field, max_lag_px=200, seed=1)
    fit = fit_nugget_multirange_model(centres, gamma, counts, max_lag_px=200)

    assert fit["converged"]
    assert fit["r2"] > 0.95
    assert abs(fit["sill"] - 1.0) < 0.15                      # T3.1 tolerance
    assert abs(fit["range_long_px"] - 60.0) / 60.0 < 0.25
    assert abs(fit["nugget_fraction"] - 0.1) < 0.10


def test_model_is_monotone_and_hits_the_sill():
    h = np.linspace(0.1, 500, 200)
    g = nugget_two_range_model(h, 0.1, 0.4, 5.0, 0.5, 80.0)
    assert np.all(np.diff(g) >= -1e-12)
    assert abs(g[-1] - 1.0) < 1e-6


def test_pure_nugget_field_fits_as_nearly_all_nugget():
    rng = np.random.default_rng(3)
    field = rng.standard_normal((256, 256))
    centres, gamma, counts = empirical_variogram_from_field(field, max_lag_px=60, seed=2)
    fit = fit_nugget_multirange_model(centres, gamma, counts, max_lag_px=60)
    assert fit["nugget_fraction"] > 0.8


def test_fits_to_field_params_normalizes():
    row = {"nugget": 0.2, "var_short": 0.3, "var_long": 0.5,
           "range_short_px": 4.0, "range_long_px": 50.0}
    p = fits_to_field_params(row)
    assert abs(sum(p["weights"]) + p["nugget"] - 1.0) < 1e-9
    assert p["ranges_px"] == [4.0, 50.0]


def test_h20_extrapolation_is_reported_as_weak_evidence():
    by_h = {
        5: {"sill": 1.0, "practical_range_px": 50.0, "nugget": 0.1},
        10: {"sill": 1.4, "practical_range_px": 70.0, "nugget": 0.12},
        15: {"sill": 1.7, "practical_range_px": 85.0, "nugget": 0.13},
    }
    out = extrapolate_h20_check(by_h, {"sill": 1.95, "practical_range_px": 97.0, "nugget": 0.14})
    assert out["status"] == "ok"
    assert 0.8 < out["ratio_sill"] < 1.2
    assert "weak evidence" in out["evidence"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
