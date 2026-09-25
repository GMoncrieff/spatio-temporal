"""Stratified chip sampling and its importance correction.

Two claims, both of the kind that would train perfectly happily while being wrong:

  * the default path is *exactly* uniform sampling with unit weights, so every run already
    scored remains comparable;
  * the correction returns the estimator to the population. Reweighting **samples** moves the
    target distribution, unlike reweighting quantile levels, so an uncorrected run is fitting
    a different world -- which is a legitimate experiment (D2u) but must be labelled, not
    stumbled into.

There is deliberately no leak test against the fold mask here, because there is no leak
surface to test: stratified sampling draws from ``valid_split_positions``, which the fold mask
has already filtered. It changes how often a permitted chip is drawn, never which chips are
permitted.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from torchgeo_dataloader import HumanFootprintChipDataset  # noqa: E402


def stub(tmp_path, weights, positions, alpha=4.0, sampling="stratified", chip=128):
    """A dataset shell carrying only what _init_chip_sampling reads.

    Built with ``__new__`` rather than the real constructor, which opens dozens of global
    rasters; the method under test touches none of them.
    """
    path = tmp_path / "w.npz"
    np.savez(path, w=np.asarray(weights, dtype=np.float32), chip_size=chip)
    ds = object.__new__(HumanFootprintChipDataset)
    ds.chip_sampling = sampling
    ds.chip_weights = str(path)
    ds.chip_weight_alpha = alpha
    ds.chip_size = chip
    ds.mode = "random"
    ds.valid_split_positions = positions
    ds._init_chip_sampling()
    return ds


def test_uniform_is_the_default_and_touches_nothing():
    ds = object.__new__(HumanFootprintChipDataset)
    ds.chip_sampling = "uniform"
    ds.chip_weights = None
    ds.chip_weight_alpha = 4.0
    ds.chip_size = 128
    ds.mode = "random"
    ds.valid_split_positions = [(0, 0), (128, 0)]
    ds._init_chip_sampling()
    assert ds._pos_cdf is None and ds._pos_correction is None


def test_grid_mode_is_never_stratified(tmp_path):
    """Validation walks a fixed grid; reweighting it would make val_crps incomparable."""
    ds = stub(tmp_path, np.ones((2, 2)), [(0, 0), (128, 0)])
    ds.mode = "grid"
    ds._init_chip_sampling()
    assert ds._pos_cdf is None


def test_high_change_chips_are_drawn_more_often(tmp_path):
    w = np.array([[0.0, 0.4], [0.0, 0.0]], dtype=np.float32)   # only chip (0,1) has change
    pos = [(0, 0), (0, 128), (128, 0), (128, 128)]
    ds = stub(tmp_path, w, pos, alpha=4.0)
    p = np.diff(np.concatenate([[0.0], ds._pos_cdf]))
    assert p[1] > p[0] * 4, f"active chip not preferred: {p}"
    assert np.isclose(p.sum(), 1.0)


def test_correction_returns_the_estimator_to_the_population(tmp_path):
    """The whole point of D2 against D2u: with the correction the target is unchanged."""
    rng = np.random.default_rng(0)
    n_i, n_j = 8, 8
    w = rng.random((n_i, n_j)).astype(np.float32) ** 3
    pos = [(i * 128, j * 128) for i in range(n_i) for j in range(n_j)]
    ds = stub(tmp_path, w, pos, alpha=6.0)
    p = np.diff(np.concatenate([[0.0], ds._pos_cdf]))
    corr = ds._pos_correction
    # A per-chip quantity to average: use the weight itself, which is what stratification
    # deliberately skews. Sum_k p_k * corr_k * f_k must equal the plain mean of f.
    f = w.reshape(-1).astype(np.float64)
    weighted = float((p * corr * f).sum())
    # 1e-5: the correction is stored float32, which is ample for a loss weight.
    assert weighted == pytest.approx(float(f.mean()), rel=1e-5)
    # And without the correction it is skewed, or the experiment would be vacuous.
    assert float((p * f).sum()) > 1.5 * float(f.mean())


def test_correction_is_exactly_one_when_every_chip_is_equal(tmp_path):
    pos = [(i * 128, 0) for i in range(6)]
    ds = stub(tmp_path, np.full((6, 1), 0.3, dtype=np.float32), pos, alpha=4.0)
    np.testing.assert_allclose(ds._pos_correction, 1.0, rtol=1e-6)


def test_mismatched_chip_size_is_refused(tmp_path):
    """A table built at another chip size indexes the wrong chips, silently."""
    path = tmp_path / "w128.npz"
    np.savez(path, w=np.ones((2, 2), dtype=np.float32), chip_size=128)
    ds = object.__new__(HumanFootprintChipDataset)
    ds.chip_sampling = "stratified"
    ds.chip_weights = str(path)
    ds.chip_weight_alpha = 4.0
    ds.chip_size = 64                      # dataset disagrees with the table
    ds.mode = "random"
    ds.valid_split_positions = [(0, 0)]
    with pytest.raises(ValueError, match="chip weights built at"):
        ds._init_chip_sampling()


def test_stratified_without_a_fold_mask_is_refused(tmp_path):
    path = tmp_path / "w.npz"
    np.savez(path, w=np.ones((2, 2), dtype=np.float32), chip_size=128)
    ds = object.__new__(HumanFootprintChipDataset)
    ds.chip_sampling = "stratified"
    ds.chip_weights = str(path)
    ds.chip_weight_alpha = 4.0
    ds.chip_size = 128
    ds.mode = "random"
    ds.valid_split_positions = None
    with pytest.raises(ValueError, match="split or fold mask"):
        ds._init_chip_sampling()


# ------------------------------------------------------------------ the loss side

def test_unit_weights_reproduce_the_unweighted_loss():
    """Whatever the correction does, weights of 1 must change nothing at all."""
    from src.models.crps_loss import crps_spline
    from src.models.quantile_spline import QuantileSpline, U_KNOTS_DEFAULT, n_spline_params

    u = torch.tensor(U_KNOTS_DEFAULT, dtype=torch.float64)
    raw = torch.randn(64, n_spline_params(u.numel()), dtype=torch.float64,
                      generator=torch.Generator().manual_seed(2))
    sp = QuantileSpline.from_channels(raw, u, hm_t0=torch.full((64,), 0.3,
                                      dtype=torch.float64))
    y = torch.full((64,), 0.31, dtype=torch.float64)
    mask = torch.ones(64, dtype=torch.bool)
    torch.testing.assert_close(crps_spline(sp, y, mask=mask),
                               crps_spline(sp, y, mask=mask.to(torch.float64)))


def test_weighted_mean_is_a_weighted_mean():
    from src.models.crps_loss import crps_spline
    from src.models.quantile_spline import QuantileSpline, U_KNOTS_DEFAULT, n_spline_params

    u = torch.tensor(U_KNOTS_DEFAULT, dtype=torch.float64)
    raw = torch.randn(8, n_spline_params(u.numel()), dtype=torch.float64,
                      generator=torch.Generator().manual_seed(3))
    sp = QuantileSpline.from_channels(raw, u, hm_t0=torch.linspace(0.1, 0.6, 8,
                                      dtype=torch.float64))
    y = torch.linspace(0.12, 0.58, 8, dtype=torch.float64)
    per = crps_spline(sp, y, reduce=False)
    w = torch.tensor([3.0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float64)
    torch.testing.assert_close(crps_spline(sp, y, mask=w), (per * w).sum() / w.sum())
