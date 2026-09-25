"""The global E1v runner must run E1v, and every check on it must fail closed.

Three things this pins, each of which would produce a plausible multi-hour run of the wrong
thing and say nothing:

``run_global_dist_hindcast.sh`` and ``run_global_dist_forecast.sh`` source
``scripts/dist_base_args.sh``, whose BASE_ARGS hardcodes ``--head_family spline`` (the
29-parameter rational-quadratic head), ``--seed 46`` and e1's context flags. Pointed at this
phase they would train e1 and nothing in their verifiers would notice: ``verify_loss_weights``
there never reads ``--mu_mse_weight`` and ``verify_context_channels`` greps for a string
instead of reading the trunk's channel count. So the first test is simply which file the
global runner sources.

``BASE_ARGS`` names ``--predict_row_chunk 2048``, sized for Africa. On the 17111 x 40000 grid
that is 81.8 GiB of prediction accumulators for ONE fold against 125 GB of DRAM, with two
folds running at once. The runner appends 512 after it and argparse keeps the last
occurrence -- so the test reads the EFFECTIVE argument string, not the source.

``--isqf_tails`` has already once been accepted, allocated and never read. 17 params/horizon
with the tails and 15 without is the only fingerprint that separates E1v from a tailless arm,
so ``verify_head_fingerprint`` is exercised against a control for each way it can be wrong,
including the one that actually happened: a banner that does not mention the tails on a run
that has them.

No torch, no data: these drive bash and small rasters directly.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RUNNER = os.path.join(ROOT, "scripts", "run_global_model.sh")
BASE = os.path.join(ROOT, "scripts", "conv_spline_base.sh")

E1V_FLAGS = ("--head_family pwl --isqf_tails True --isqf_space neglog "
             "--free_scale True --mu_mse_weight 0.0")

BANNER = ("Spline head:       family {fam}, knots default14 (n=15, bins=14), "
          "slopes learned{extra}, {n} params/horizon\n")


def _args_dump(model="E1v", **env):
    e = dict(os.environ, ALLOW_GLOBAL="1", MODEL=model, **env)
    r = subprocess.run(["bash", RUNNER, "args"], capture_output=True, text=True, env=e, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    out = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


# --------------------------------------------------------------------- the configuration

def test_runner_sources_the_conv_spline_baseline_not_the_e1_one():
    """Only the `source` lines. The comments name dist_base_args.sh on purpose: the reason
    this runner exists is that the two e1-era global runners source it."""
    sourced = [l.split()[1] for l in open(RUNNER).read().splitlines()
               if l.strip().startswith("source ")]
    assert any("conv_spline_base.sh" in s for s in sourced)
    assert not any("dist_base_args.sh" in s for s in sourced)


def test_effective_args_carry_every_e1v_flag():
    args = _args_dump()["TRAIN_ARGS"]
    for flag in ("--head_family pwl", "--isqf_tails True", "--isqf_space neglog",
                 "--free_scale True", "--mu_mse_weight 0.0"):
        assert flag in args, f"{flag} missing from the effective arguments"


def test_head_family_pwl_is_the_last_word_over_base_args_spline():
    """BASE_ARGS names --head_family spline first; argparse keeps the last occurrence."""
    args = _args_dump()["TRAIN_ARGS"]
    assert args.rindex("--head_family pwl") > args.rindex("--head_family spline")


def test_mu_mse_zero_is_the_last_word_over_base_args_one():
    args = _args_dump()["TRAIN_ARGS"]
    assert args.rindex("--mu_mse_weight 0.0") > args.rindex("--mu_mse_weight 1.0")


def test_row_chunk_512_is_the_last_word_over_africas_2048():
    """2048 rows is 81.8 GiB of accumulators per fold on the global grid; 512 is 20.4."""
    args = _args_dump()["TRAIN_ARGS"]
    assert args.rindex("--predict_row_chunk 512") > args.rindex("--predict_row_chunk 2048")


def test_the_phase_context_covariate_is_named_and_expected():
    d = _args_dump()
    assert d["EXPECT_CTX_CHANNELS"] == "12"
    assert "--hm_context_stats mean,max" in d["TRAIN_ARGS"]
    assert "--context_radii 3,30,100" in d["TRAIN_ARGS"]


def test_global_region_and_the_512px_block_fold_mask():
    d = _args_dump()
    assert d["REGION"].endswith("region_to_predict_large.geojson")
    assert "fold_mask_b4_1000" in d["FOLD_MASK"]
    assert d["FOLDS"] == "1,2,3,4,5", "a global hindcast mosaic needs every fold"


def test_the_scorer_is_banded_even_though_the_base_exports_zero():
    """conv_spline_base.sh exports SCORE_ROW_CHUNK=0 -- Africa is affordable unbanded -- and
    "0" is neither unset nor empty, so ${SCORE_ROW_CHUNK:-512} KEEPS it. The global scorer
    then reads a whole window-year in one call: `Unable to allocate 163. GiB for an array
    with shape (64, 17111, 40000)`. The control is the value the base actually exports."""
    d = _args_dump(SCORE_ROW_CHUNK="0")
    assert int(d["SCORE_ROW_CHUNK"]) > 0, "the global score would read 163 GiB in one call"


def test_the_global_score_row_chunk_is_overridable_by_its_own_name():
    d = _args_dump(GLOBAL_SCORE_ROW_CHUNK="256")
    assert d["SCORE_ROW_CHUNK"] == "256"


def test_every_row_chunk_in_the_dump_is_bounded():
    d = _args_dump()
    for k in ("ROW_CHUNK", "SCORE_ROW_CHUNK", "ICE_ROW_CHUNK"):
        assert 0 < int(d[k]) <= 2048, f"{k}={d[k]} is not a global-safe band"


def test_the_runner_refuses_the_globe_without_allow_global():
    e = dict(os.environ)
    e.pop("ALLOW_GLOBAL", None)
    r = subprocess.run(["bash", RUNNER, "args"], capture_output=True, text=True, env=e, cwd=ROOT)
    assert r.returncode != 0
    assert "ALLOW_GLOBAL" in r.stderr


# --------------------------------------------------------------------- the smoke gate

@pytest.mark.parametrize("stage", ["hindcast", "stitch", "score", "forecast",
                                   "export_hindcast", "export_forecast"])
def test_every_long_stage_refuses_without_a_smoke_receipt(tmp_path, stage):
    """PY=/bin/false so a failure of this test cannot start a global training run."""
    e = dict(os.environ, ALLOW_GLOBAL="1", LOG_DIR=str(tmp_path), PY="/bin/false")
    r = subprocess.run(["bash", RUNNER, stage], capture_output=True, text=True, env=e, cwd=ROOT)
    assert r.returncode != 0
    assert "no smoke receipt" in r.stderr


def test_a_receipt_for_different_code_is_refused(tmp_path):
    (tmp_path / "smoke_ok_E1v.stamp").write_text(
        "code_hash=deadbeefdeadbeef\nflags=" + _args_dump()["MODEL_FLAGS"] + "\n")
    e = dict(os.environ, ALLOW_GLOBAL="1", LOG_DIR=str(tmp_path), PY="/bin/false")
    r = subprocess.run(["bash", RUNNER, "hindcast"], capture_output=True, text=True,
                       env=e, cwd=ROOT)
    assert r.returncode != 0
    assert "different code" in r.stderr


def test_a_current_receipt_is_accepted(tmp_path):
    """The control for the two above: the gate must also be passable."""
    d = _args_dump(LOG_DIR=str(tmp_path))
    (tmp_path / "smoke_ok_E1v.stamp").write_text(
        f"code_hash={d['code_hash']}\nflags={d['MODEL_FLAGS']}\n")
    script = (f'cd {ROOT}\n'
              f'export ALLOW_GLOBAL=1 LOG_DIR={tmp_path}\n'
              f'source scripts/run_global_model.sh args >/dev/null\n'
              f'require_smoke probe\n')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "matches the code on disk" in r.stdout


# --------------------------------------------------------------------- head fingerprint

def _head(tmp_path, name, fam="pwl", extra=", tails neglog, free scale", n=17):
    p = tmp_path / name
    p.write_text(BANNER.format(fam=fam, extra=extra, n=n))
    return p


def _verify_head(log, fam="pwl", n=17, flags=E1V_FLAGS):
    script = (f'source "{BASE}"\nverify_head_fingerprint "{log}" probe {fam} {n} "{flags}"\n')
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def test_head_fingerprint_accepts_e1v(tmp_path):
    assert _verify_head(_head(tmp_path, "ok.log"))


def test_head_fingerprint_rejects_the_e1_spline_head(tmp_path):
    assert not _verify_head(_head(tmp_path, "e1.log", fam="spline", extra="", n=29))


def test_head_fingerprint_rejects_the_isqf_sibling(tmp_path):
    """E1d is the same 17 parameters on the other family — the count alone cannot tell them
    apart, which is why the family is checked beside it."""
    assert not _verify_head(_head(tmp_path, "e1d.log", fam="isqf"))


def test_head_fingerprint_rejects_a_tailless_arm(tmp_path):
    assert not _verify_head(_head(tmp_path, "e2a.log", extra=", free scale", n=15))


def test_head_fingerprint_rejects_a_banner_that_hides_the_tails(tmp_path):
    """The failure that actually happened: E1v ran WITH the tails and its banner said only
    'free scale, 17 params/horizon', because the line was gated on head_family == 'isqf'.
    The count is right and the run is right; the fingerprint is not, and a fingerprint
    nobody can read is rule 28."""
    assert not _verify_head(_head(tmp_path, "hidden.log", extra=", free scale", n=17))


def test_head_fingerprint_rejects_the_wrong_tail_space(tmp_path):
    """neglog beat logit on both tails, twice. They are different models."""
    assert not _verify_head(_head(tmp_path, "logit.log", extra=", tails logit, free scale"))


def test_head_fingerprint_fails_closed_on_a_missing_log(tmp_path):
    assert not _verify_head(tmp_path / "does_not_exist.log")


def test_head_fingerprint_fails_closed_on_a_log_with_no_banner(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("")
    assert not _verify_head(p)


# --------------------------------------------------------------------- row banding

def _verify_rows(log, want):
    script = f'source "{BASE}"\nverify_row_banding "{log}" probe {want}\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def _rows_log(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(f"  Row banding: 34 bands of {rows} rows; 268 accumulators x 0.076 GiB "
                 f"= 20.4 GiB worst case\n")
    return p


def test_row_banding_accepts_the_global_value(tmp_path):
    assert _verify_rows(_rows_log(tmp_path, "ok.log", 512), 512)


def test_row_banding_rejects_africas_2048_on_the_global_grid(tmp_path):
    """Banding happened, and the run would still have wanted 81.8 GiB per fold. A check that
    only asks whether banding happened passes this."""
    assert not _verify_rows(_rows_log(tmp_path, "africa.log", 2048), 512)


def test_row_banding_fails_closed_on_an_unbanded_run(tmp_path):
    p = tmp_path / "none.log"
    p.write_text("no banding line here\n")
    assert not _verify_rows(p, 512)


def test_row_banding_fails_closed_on_a_missing_log(tmp_path):
    assert not _verify_rows(tmp_path / "nope.log", 512)


# --------------------------------------------------------------------- the raster's own tag

def _qf_raster(path, fam, params=None):
    # The gate levels must be present or the reader refuses the raster before it ever
    # reaches the tag -- which is itself a check, and not the one under test here.
    u = np.array([0.025, 0.5, 0.975])
    n_levels = len(u)
    prof = dict(driver="GTiff", height=8, width=8, count=n_levels, dtype="float32",
                crs="EPSG:4326", transform=Affine(0.009, 0, 0, 0, -0.009, 0),
                nodata=np.nan, BIGTIFF="YES")
    with rasterio.open(path, "w", **prof) as d:
        for i in range(n_levels):
            d.write(np.full((8, 8), 0.1 * (i + 1), dtype="float32"), i + 1)
            d.set_band_description(i + 1, f"u={u[i]:.6f}")
        tags = {"u_levels": ",".join(repr(float(v)) for v in u), "scale_factor": "1.0",
                "head_family": fam}
        if params is not None:
            tags["head_params"] = str(params)
        d.update_tags(**tags)
    return path


def _check_tag(pred_dir, fam, params=0):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "check_qf_raster.py"),
           "--pred_dir", str(pred_dir), "--expect_tag_family", fam, "--min_px", "16"]
    if params:
        cmd += ["--expect_tag_params", str(params)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).returncode == 0


def test_raster_tag_check_accepts_the_head_that_wrote_it(tmp_path):
    _qf_raster(tmp_path / "a_qf_blended.tif", "pwl", 17)
    assert _check_tag(tmp_path, "pwl", 17)


def test_raster_tag_check_rejects_the_hardcoded_spline_literal(tmp_path):
    """The control is real: every quantile raster this project has ever written, E1v's own
    promoted output included, carries head_family="spline" because ``update_tags`` was given
    the literal. Harmless to the scorer, which does not read it; wrong metadata on a
    delivered product."""
    _qf_raster(tmp_path / "a_qf_blended.tif", "spline", 29)
    assert not _check_tag(tmp_path, "pwl", 17)


def test_raster_tag_check_rejects_the_tailless_parameter_count(tmp_path):
    _qf_raster(tmp_path / "a_qf_blended.tif", "pwl", 15)
    assert not _check_tag(tmp_path, "pwl", 17)


def _float_raster(path, nodata, fill_nan_frac=0.5, value=0.25):
    a = np.full((16, 16), value, dtype="float32")
    a[: int(16 * fill_nan_frac)] = np.nan
    prof = dict(driver="GTiff", height=16, width=16, count=1, dtype="float32",
                crs="EPSG:4326", transform=Affine(0.009, 0, 0, 0, -0.009, 0), nodata=nodata)
    with rasterio.open(path, "w", **prof) as d:
        d.write(a, 1)
    return path


def _count(path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    m = importlib.import_module("check_prediction_complete")
    importlib.reload(m)
    return m.count_valid(path)[0]


def test_a_sentinel_nodata_does_not_make_nan_count_as_data(tmp_path):
    """The forward product's triple declared nodata=3.4e38 and was FILLED with NaN. Counting
    `a != nodata` alone then reports every NaN as valid -- measured, 370.8% of the land."""
    p = _float_raster(tmp_path / "sentinel.tif", 3.4e38)
    assert _count(p) == 128, "half the pixels are NaN and must not count"


def test_nan_nodata_still_counts_only_finite_pixels(tmp_path):
    """The control in the other direction: the stitched rasters declare NaN correctly."""
    p = _float_raster(tmp_path / "nan.tif", float("nan"))
    assert _count(p) == 128


def test_a_sentinel_that_is_actually_written_is_excluded(tmp_path):
    p = _float_raster(tmp_path / "both.tif", 0.25, fill_nan_frac=0.25)
    assert _count(p) == 0, "every finite pixel equals the declared nodata"


def test_the_prediction_writer_declares_the_nodata_it_writes(tmp_path):
    """The triple inherited the HM reference's 3.4e38 while filling with NaN; only the
    forward product ships it, because the stitcher sets NaN itself."""
    src = open(os.path.join(ROOT, "scripts", "train_lightning.py")).read()
    i = src.index("out_profile.update({")
    block = src[i:i + 2000]
    assert "'nodata': np.nan," in block, "the triple's profile must declare NaN"


def test_the_writer_no_longer_passes_a_literal_family(tmp_path):
    """Read the source, because there is no cheap way to run the writer: the tag must come
    off the constructed module, the same place the banner reads its parameter count."""
    src = open(os.path.join(ROOT, "scripts", "train_lightning.py")).read()
    assert 'head_family="spline")' not in src
    assert "_tag_fam = getattr(_tag_mod, 'head_family', None)" in src


# --------------------------------------------------------------------- the model table

E2A_FLAGS = "--head_family pwl --free_scale True --mu_mse_weight 0.0"


def test_e2a_resolves_to_its_own_flags_family_and_param_count():
    """15, not the 16 E2a was SCORED at on Africa: --free_scale did not actually remove the
    scale channel until 2026-09-16, so the Africa arm carried a dead one. A global run on
    current code is a different head and its count must say so."""
    d = _args_dump("E2a")
    assert d["MODEL_FLAGS"] == E2A_FLAGS
    assert d["MODEL_FAMILY"] == "pwl"
    assert d["MODEL_PARAMS"] == "15"
    assert "--isqf_tails" not in d["TRAIN_ARGS"], "E2a is the arm WITHOUT the learned tails"


def test_e1v_and_e2a_differ_only_by_the_tails():
    """The whole point of running E2a globally: on current code the two are the same family
    at free scale and differ by the tail flags alone."""
    a = _args_dump("E1v")["MODEL_FLAGS"]
    b = _args_dump("E2a")["MODEL_FLAGS"]
    # A set difference cannot express this: "True" is a token of --free_scale True in BOTH.
    assert a.replace("--isqf_tails True --isqf_space neglog ", "") == b
    assert "--free_scale True" in a and "--free_scale True" in b
    assert "--head_family pwl" in a and "--head_family pwl" in b


def test_each_model_writes_to_its_own_roots_and_receipt():
    """An E1v receipt must not authorise an E2a run, and neither may overwrite the other's
    rasters or products."""
    a, b = _args_dump("E1v"), _args_dump("E2a")
    for k in ("HIND_ROOT", "FC_ROOT", "PROD_ROOT", "STAMP"):
        assert a[k] != b[k], f"{k} collides between models"
    assert "E2a" in b["STAMP"] and "E1v" in a["STAMP"]


def test_an_unknown_model_is_refused():
    e = dict(os.environ, ALLOW_GLOBAL="1", MODEL="E9z")
    r = subprocess.run(["bash", RUNNER, "args"], capture_output=True, text=True, env=e,
                       cwd=ROOT)
    assert r.returncode != 0
    assert "unknown MODEL" in r.stderr


def test_a_receipt_for_the_other_models_configuration_is_refused(tmp_path):
    """The code hash can match while the configuration does not -- one runner, two models."""
    d = _args_dump("E2a", LOG_DIR=str(tmp_path))
    (tmp_path / "smoke_ok_E2a.stamp").write_text(
        f"code_hash={d['code_hash']}\nflags={_args_dump('E1v')['MODEL_FLAGS']}\n")
    e = dict(os.environ, ALLOW_GLOBAL="1", LOG_DIR=str(tmp_path), PY="/bin/false", MODEL="E2a")
    r = subprocess.run(["bash", RUNNER, "hindcast"], capture_output=True, text=True,
                       env=e, cwd=ROOT)
    assert r.returncode != 0
    assert "different configuration" in r.stderr
