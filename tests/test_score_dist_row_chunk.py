"""Banding the distributional scorer must not move a single number.

``score_distributional_model.load_row`` used to read the whole 64-band quantile-function
raster at once. That is 16 GB on Africa's 63.1 Mpx grid and **175 GB** on the global
17111 x 40000 one, with the nodata replacement doubling it -- the same shape of defect as the
prediction accumulators: a working set the screening region never exercised.

Everything the scorer computes per pixel is pixel-independent and every stratum is a mean
over a boolean mask, so reading in horizontal bands is an identity. This asserts that it is,
against band sizes that divide the raster evenly, unevenly, and not at all.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from src.models.quantile_spline import output_u_grid  # noqa: E402
import score_distributional_model as sd  # noqa: E402

H, W = 300, 200
INT16_SCALE = 1.0 / 32767.0


def _profile(count, dtype, nodata):
    return dict(driver="GTiff", height=H, width=W, count=count, dtype=dtype, nodata=nodata,
                crs="EPSG:4326", transform=Affine(0.01, 0, 10.0, 0, -0.01, 5.0))


@pytest.fixture(scope="module")
def rasters(tmp_path_factory):
    d = tmp_path_factory.mktemp("stitched")
    rng = np.random.default_rng(11)
    u = output_u_grid(64)

    hm0 = rng.random((H, W)).astype(np.float32) * 0.4
    obs = np.clip(hm0 + rng.normal(0, 0.02, (H, W)), 0, 1).astype(np.float32)
    central = np.clip(hm0 + rng.normal(0, 0.01, (H, W)), 0, 1).astype(np.float32)
    # A quantile function: monotone in u about the central value, widening with lead time.
    z = np.asarray([-2.5, 2.5])
    spread = (0.01 + 0.05 * rng.random((H, W))).astype(np.float32)
    qf = np.clip(central[None] + spread[None] * (u[:, None, None] - 0.5) * 4.0, 0, 1)
    i_lo = int(np.argmin(np.abs(u - 0.025)))
    i_hi = int(np.argmin(np.abs(u - 0.975)))

    # Ocean: a nodata block that must be dropped identically in every band.
    hole = (slice(40, 90), slice(30, 70))
    obs[hole] = np.nan
    hm0[hole] = np.nan

    def write(name, arr, dtype="float32", nodata=np.nan):
        p = d / name
        a = np.atleast_3d(arr.T).T if arr.ndim == 2 else arr
        with rasterio.open(p, "w", **_profile(a.shape[0], dtype, nodata)) as dst:
            dst.write(a.astype(dtype))
        return p

    qi = np.where(np.isfinite(qf), np.round(qf / INT16_SCALE), -32768)
    qp = _profile(len(u), "int16", -32768)
    qpath = d / "w2000_prediction_2005_qf.tif"
    with rasterio.open(qpath, "w", **qp) as dst:
        dst.write(np.clip(qi, -32768, 32767).astype(np.int16))
        dst.update_tags(u_levels=",".join(repr(float(v)) for v in u),
                        scale_factor="3.0518509e-05", head_family="spline")

    write("w2000_prediction_2005_central.tif", central)
    write("w2000_prediction_2005_lower.tif", qf[i_lo].astype(np.float32))
    write("w2000_prediction_2005_upper.tif", qf[i_hi].astype(np.float32))
    obs_p = write("HM_2005_AA_1000.tiff", obs)
    base_p = write("HM_2000_AA_1000.tiff", hm0)
    # context band 2 is distance to past change, which drives the distance strata
    dist = rng.random((H, W)).astype(np.float32) * 150.0
    ctx = np.stack([rng.random((H, W)).astype(np.float32), dist])
    ctx_p = write("change_context_w2000_1000.tif", ctx)

    return dict(path_central=str(d / "w2000_prediction_2005_central.tif"),
                path_lower=str(d / "w2000_prediction_2005_lower.tif"),
                path_upper=str(d / "w2000_prediction_2005_upper.tif"),
                path_qf=str(qpath), path_observed=str(obs_p),
                path_baseline=str(base_p), path_context=str(ctx_p))


@pytest.mark.parametrize("chunk", [64, 100, 128, 256, 299, 301])
@pytest.mark.parametrize("with_fold_sel", [False, True])
def test_row_chunk_is_an_identity(rasters, chunk, with_fold_sel):
    sel = None
    if with_fold_sel:
        sel = np.zeros((H, W), dtype=bool)
        sel[::2] = True          # a mask that straddles every band boundary
    u0, cell0, pp0, cons0 = sd.load_row(rasters, sel, 0)
    u1, cell1, pp1, cons1 = sd.load_row(rasters, sel, chunk)

    assert np.array_equal(u0, u1)
    assert cell0["observed"].size > 1000, "the fixture must actually score something"
    for k in cell0:
        assert np.array_equal(cell0[k], cell1[k], equal_nan=True), f"cell[{k}] moved"
    assert set(pp0) == set(pp1)
    for k in pp0:
        assert np.array_equal(pp0[k], pp1[k], equal_nan=True), f"per-pixel [{k}] moved"
    for k in cons0:
        assert cons0[k] == pytest.approx(cons1[k], rel=0, abs=0), f"consistency[{k}] moved"


def test_banding_never_reads_the_whole_quantile_raster(rasters, monkeypatch):
    # The point of the change is the peak, so assert the read is windowed -- a future edit
    # that quietly restores src.read() would still pass the identity test above.
    seen = []
    real = rasterio.DatasetReader.read

    def spy(self, *a, **kw):
        if self.count == 64:
            seen.append(kw.get("window"))
        return real(self, *a, **kw)

    monkeypatch.setattr(rasterio.DatasetReader, "read", spy)
    sd.load_row(rasters, None, 64)
    assert seen, "the quantile raster was never read"
    assert all(w is not None for w in seen), "a band read the whole quantile-function raster"
    assert all(w.height <= 64 for w in seen), f"window heights {[w.height for w in seen]}"
