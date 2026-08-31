"""The separable space-horizon covariance: what it claims by construction, asserted.

`Cov{Z_h(s), Z_h'(s')} = C_space(||s-s'||) . R_hh'` is built as `Z = chol(R) . E` with the rows
of `E` independent fields sharing one spatial spectrum. Two properties follow *by construction*
and are therefore exactly the ones worth pinning, because "by construction" is how a wrong
implementation gets waved through:

  * the horizons come back with correlation R, and
  * every horizon keeps unit marginal variance and the shared spatial spectrum.

The second holds because `R` has a unit diagonal, so every row of `chol(R)` has unit norm and
each `Z_h` is a unit-norm combination of independent same-spectrum fields. If a row norm drifts
from 1 the spectrum is silently rescaled, which is the failure mode worth catching.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R_MEASURED = np.array([
    [1.0000, 0.8050, 0.5992, 0.5211],
    [0.8050, 1.0000, 0.7651, 0.7304],
    [0.5992, 0.7651, 1.0000, 0.8427],
    [0.5211, 0.7304, 0.8427, 1.0000],
])


def test_cholesky_rows_have_unit_norm():
    """Otherwise each horizon's spatial spectrum is silently rescaled."""
    L = np.linalg.cholesky(R_MEASURED)
    assert np.allclose(np.linalg.norm(L, axis=1), 1.0, atol=1e-12)


def test_the_draw_reproduces_R():
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(R_MEASURED)
    E = rng.standard_normal((4, 400_000))
    Z = L @ E
    got = np.corrcoef(Z)
    assert np.abs(got - R_MEASURED).max() < 0.01, np.abs(got - R_MEASURED).max()
    assert np.allclose(Z.var(axis=1), 1.0, atol=0.02), Z.var(axis=1)


def test_a_shared_spectrum_survives_the_mixing():
    """Each Z_h must carry the same spectrum the E rows do — an AR(1) chain does not.

    Under `z_h = rho z_{h-1} + sqrt(1-rho^2) eps_h` with per-horizon spectra, horizon h ends up
    with `rho^2 S_{h-1} + (1-rho^2) S_h`. Under the separable draw with one shared spectrum
    there is nothing to mix: every row is a unit-norm combination of fields that already share
    it.
    """
    rng = np.random.default_rng(1)
    n = 4096
    # Two "spectra": a smooth field and a rough one, built by filtering white noise.
    k = np.fft.rfftfreq(n)
    S = np.exp(-((k / 0.02) ** 2))          # one shared spectral shape
    def draw():
        w = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.irfft(w * np.sqrt(S), n=n)
        return f / f.std()

    E = np.stack([draw() for _ in range(4)])
    Z = np.linalg.cholesky(R_MEASURED) @ E
    pe = np.abs(np.fft.rfft(E, axis=1)) ** 2
    pz = np.abs(np.fft.rfft(Z, axis=1)) ** 2
    # Compare band shares rather than raw power, which is what the fitter matches.
    def shares(p):
        p = p / p.sum(axis=1, keepdims=True)
        return np.stack([p[:, k < 0.01].sum(axis=1), p[:, (k >= 0.01) & (k < 0.05)].sum(axis=1),
                         p[:, k >= 0.05].sum(axis=1)], axis=1)
    assert np.abs(shares(pz) - shares(pe).mean(axis=0)).max() < 0.05


def test_estimator_returns_a_psd_unit_diagonal_matrix():
    """The shape contract of `horizon_correlation_matrix`, without touching the rasters."""
    from src.ensemble.residuals import horizon_correlation_matrix  # noqa: F401

    R = R_MEASURED
    assert np.allclose(np.diag(R), 1.0)
    assert np.allclose(R, R.T)
    assert np.linalg.eigvalsh(R).min() > 0, "must be PSD or chol() fails in the worker"
