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


# ------------------------------------------------- the gates, and the scorecard's figures

@pytest.fixture(scope="module")
def fold_mask(rasters, tmp_path_factory):
    p = tmp_path_factory.mktemp("mask") / "fold_mask.tif"
    with rasterio.open(p, "w", **_profile(1, "int16", -32768)) as dst:
        dst.write(np.ones((H, W), dtype=np.int16), 1)
    return p


@pytest.fixture(scope="module")
def scored(rasters, fold_mask, tmp_path_factory, monkeypatch_module=None):
    """One full ``main()`` pass over the fixture, so the streamed path is exercised."""
    d = Path(rasters["path_central"]).parent
    out = tmp_path_factory.mktemp("score")
    old = sd.HM_DIR
    sd.HM_DIR = d
    try:
        rc = sd.main(["--stitched_dir", str(d), "--label", "fx", "--out_dir", str(out),
                      "--folds", "1", "--fold_mask", str(fold_mask),
                      "--context_pattern", str(d / "change_context_w{year}_1000.tif"),
                      "--plot_horizon", "5"])
    finally:
        sd.HM_DIR = old
    assert rc == 0
    return out


def test_the_scorer_end_to_end_records_both_gates(scored):
    """The gates must survive the *streamed* path, not merely a hand-built cell.

    They did not. ``load_row`` drops ``cell["qf"]`` before concatenating its bands and
    ``gate_stats`` read it back out, so every real scoring run raised ``KeyError: 'qf'`` on
    the pooled stratum -- while the unit test above passed, because it built the cell itself
    with the quantile function still in it. The two gates are the whole reason this phase
    exists; this is the control that fires when they are not measured.
    """
    import pandas as pd
    df = pd.read_csv(scored / "dist_fx.csv")
    pooled = df[df["stratum"] == "pooled"]
    assert len(pooled)
    for reading in ("export", "ref"):
        for stat in ("needle_mass_median", "needle_mass_p90", "max_density_p99",
                     "over_f_max_frac"):
            col = f"{stat}_{reading}"
            assert col in df.columns, f"the scorecard lost {col}"
            assert pooled[col].notna().all(), f"{col} is null on the pooled row"
    for col in ("pit_rms_se_20", "pit_rms_se_60", "zero_leak_neg_ratio"):
        assert col in df.columns and pooled[col].notna().all(), col

    import json
    summary = json.loads((scored / "summary_fx.json").read_text())
    assert "needle_mass_median_export_5" in summary
    assert "needle_mass_median_ref_5" in summary


def test_the_scorecard_carries_both_figures(scored):
    """Numbers with nothing to look at is what the phase doc's section 3 forbids."""
    dens = scored / "densities_fx.png"
    pit = scored / "pit_fx.png"
    card = scored / "scorecard_fx.html"
    for p in (dens, pit, card):
        assert p.exists(), f"{p.name} was not written"
    assert dens.stat().st_size > 20_000 and pit.stat().st_size > 5_000
    text = card.read_text()
    assert text.count("data:image/png;base64,") == 2, "a figure is not embedded"
    assert "needle p50 ref" in text and "maxdens p99 ref" in text, "the gates are not tabled"


def test_the_nine_pixels_are_a_property_of_the_mask_not_of_the_run(rasters, fold_mask):
    """Two experiments on the same folds must draw the same nine pixels, or the panels
    cannot be compared. Seeded over the fold mask for exactly that reason."""
    with rasterio.open(rasters["path_central"]) as src:
        sel = np.ones((src.height, src.width), dtype=bool)
    a = sd.sample_pixels(rasters, sel, n=9, seed=0)
    b = sd.sample_pixels(rasters, sel, n=9, seed=0)
    c = sd.sample_pixels(rasters, sel, n=9, seed=1)
    assert a is not None and a[1].shape[1] == 9
    assert a[4] == b[4], "the same seed drew different pixels"
    assert a[4] != c[4], "the seed does not select"
    assert np.isfinite(a[1]).all()
    assert np.all(np.diff(a[1], axis=0) >= -1e-9), "a sampled quantile function is not monotone"


# ------------------------------------------------- the quantile raster's own storage scale

def _write_qf(path, u, q, dtype):
    """Write a quantile raster the way train_lightning does, in either storage."""
    f32 = dtype == "float32"
    prof = _profile(len(u), "float32" if f32 else "int16",
                    np.nan if f32 else -32768)
    prof.update(tiled=True, blockxsize=256, blockysize=256, BIGTIFF="YES")
    with rasterio.open(path, "w", **prof) as dst:
        if f32:
            dst.write(q.astype(np.float32))
        else:
            qi = np.where(np.isfinite(q), np.round(q / INT16_SCALE), -32768)
            dst.write(np.clip(qi, -32768, 32767).astype(np.int16))
        dst.update_tags(u_levels=",".join(repr(float(v)) for v in u),
                        scale_factor="1.0" if f32 else "3.0518509e-05",
                        head_family="spline")
    return path


def test_the_reader_takes_the_scale_from_the_raster_not_from_a_constant(tmp_path):
    """Both storages must read back as the same forecast, in HM.

    The scale used to be a constant in the reader and a literal in the writer's tag -- the
    same quantity in two places (rule 2). With a float32 export that disagreement is not
    cosmetic: applying the int16 scale to a float raster divides every quantile by 32767 and
    reads as a catastrophically narrow forecast, which is exactly the defect this phase was
    diagnosing when the export changed. It must not be reachable by accident.
    """
    u = np.asarray(sd.output_u_grid(64)) if hasattr(sd, "output_u_grid") else None
    from src.models.quantile_spline import output_u_grid
    u = output_u_grid(64)
    rng = np.random.default_rng(3)
    q = np.sort(rng.random((64, H, W)).astype(np.float32) * 0.4 + 0.1, axis=0)

    p16 = _write_qf(tmp_path / "qf_i16.tif", u, q, "int16")
    p32 = _write_qf(tmp_path / "qf_f32.tif", u, q, "float32")

    u16, q16 = sd.read_qf(str(p16))
    u32, q32 = sd.read_qf(str(p32))
    assert np.array_equal(u16, u32)
    # int16 is lossy at its own quantum and float32 is not; they must still agree to it.
    assert np.nanmax(np.abs(q16 - q32)) <= INT16_SCALE
    # float32 must come back as what was written, exactly.
    assert np.allclose(q32, q, rtol=0, atol=0, equal_nan=True)


def test_a_float_raster_tagged_with_the_int16_scale_is_refused(tmp_path):
    """Prove the guard fires on the control: the wrong tag is the silent-disaster case."""
    from src.models.quantile_spline import output_u_grid
    u = output_u_grid(64)
    q = np.sort(np.random.default_rng(4).random((64, H, W)).astype(np.float32), axis=0)
    p = _write_qf(tmp_path / "bad.tif", u, q, "float32")
    with rasterio.open(p, "r+") as dst:
        dst.update_tags(scale_factor="3.0518509e-05")
    with pytest.raises(SystemExit, match="must be tagged 1.0"):
        sd.qf_levels(str(p))


def test_float32_storage_removes_the_zero_gap_floor(tmp_path):
    """The reason for the change, asserted rather than argued.

    A forecast whose adjacent levels differ by less than one int16 quantum exports to the
    SAME code, so the gap reads as exactly zero and the implied density as infinite. b1_s42
    put 34% of its gaps there and max_density_p99 pinned at dp_max/quantum = 1531.8 at every
    horizon. float32 must leave those gaps positive and the density finite.
    """
    from src.models.quantile_spline import output_u_grid
    from src.qf_diagnostics import F_MAX_DENSITY, fence_per_pixel
    u = output_u_grid(64)
    # A core an order of magnitude narrower than the int16 quantum, which is b1's regime.
    step = INT16_SCALE / 10.0
    q = (0.3 + step * np.arange(64)[:, None, None]).astype(np.float32) * np.ones((1, H, W),
                                                                                np.float32)
    p16 = _write_qf(tmp_path / "n16.tif", u, q, "int16")
    p32 = _write_qf(tmp_path / "n32.tif", u, q, "float32")

    _, a = sd.read_qf(str(p16))
    _, b = sd.read_qf(str(p32))
    fa = fence_per_pixel(u, a.reshape(64, -1))
    fb = fence_per_pixel(u, b.reshape(64, -1))
    assert fa["n_zero"].sum() > 0, "the int16 control did not collapse; the fixture is wrong"
    assert fb["n_zero"].sum() == 0, "float32 still collapses adjacent levels"
    # int16 cannot express the gap at all, so the density is infinite and the pixel used to
    # be DROPPED from the density gate -- the worst pixels leaving no trace. float32 resolves
    # it into a finite number the gate can actually rank.
    assert not np.isfinite(fa["px_max_density"]).any(), "the int16 control is not degenerate"
    assert np.isfinite(fb["px_max_density"]).all(), "float32 still collapses a segment"
    assert fb["px_max_density"].max() > F_MAX_DENSITY
