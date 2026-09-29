"""Slope and aspect enter the model as three static channels, exactly like elevation.

``--terrain_covariates True`` appends ``hm_static_terrain_{slope,aspsin,aspcos}_1000.tiff``
to the static list. One function (``torchgeo_dataloader.static_file_list``) defines that list
for the training dataset, the large-area prediction loop and the statistics builder, so the
three cannot disagree about which channels exist (rule 2). Default off: E1v and E2a keep their
seven channels and read exactly as they trained.

The run must be able to prove the channels arrived. ``Static channels:`` is printed off the
dataset AND the constructed module, and ``verify_static_fingerprint`` derives the expected
count from the flags (7, or 10 with the flag) -- a flag accepted and never read would
otherwise train a 7-channel model and report a null (conventions: "a new experiment needs a
test that it changed something").
"""
import inspect
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torchgeo_dataloader as tdl  # noqa: E402
from prepare_terrain import TERRAIN_NAMES  # noqa: E402

BASE = os.path.join(ROOT, "scripts", "conv_spline_base.sh")
RUNNER = os.path.join(ROOT, "scripts", "run_global_model.sh")
TERRAIN = [TERRAIN_NAMES[k] for k in ("slope", "aspsin", "aspcos")]


def test_default_static_list_is_unchanged():
    assert tdl.static_file_list() == list(tdl.static_files)
    assert len(tdl.static_file_list(False)) == 7


def test_terrain_appends_the_three_channels_the_script_writes():
    files = tdl.static_file_list(True)
    assert files[:7] == list(tdl.static_files)
    assert [os.path.basename(f) for f in files[7:]] == TERRAIN


@pytest.mark.parametrize("name", TERRAIN)
def test_terrain_nodata_reads_as_zero_like_elevation(name):
    arr = np.array([[np.nan, 2.0]], dtype=np.float32)
    out = tdl.prepare_static(arr, name, 1.0, 2.0, {})
    np.testing.assert_allclose(out, [[-0.5, 0.5]])


def test_get_dataloader_takes_the_flag():
    assert "terrain_covariates" in inspect.signature(tdl.get_dataloader).parameters


def test_a_seven_channel_sidecar_is_refused_by_a_ten_channel_dataset():
    """The existing guard is what stops E2a's sidecar training E2c: prove it fires."""
    ds = object.__new__(tdl.HumanFootprintChipDataset)
    ds._static_files = tdl.static_file_list(True)
    ds.include_components = False
    stats = dict(hm_mean=0.1, hm_std=0.2, static_means=[0.0] * 7, static_stds=[1.0] * 7,
                 static_files=[os.path.basename(f) for f in tdl.static_files])
    with pytest.raises(ValueError, match="different static-channel set"):
        ds._load_norm_stats(stats)


def test_every_static_reader_uses_the_one_list():
    tl = open(os.path.join(ROOT, "scripts", "train_lightning.py")).read()
    assert "--terrain_covariates" in tl
    assert tl.count("terrain_covariates=args.terrain_covariates") >= 3, "all three loaders"
    # The prediction loop reads the list the training dataset read, not a re-derivation.
    assert "static_list_paths = list(PREDICT_STATS['static_files'])" in tl, "the prediction loop"
    assert "'static_files': list(_ds_train._static_files)" in tl
    assert not re.search(r"static_list_paths = list\(static_files", tl)
    ns = open(os.path.join(ROOT, "scripts", "compute_norm_stats.py")).read()
    assert "static_file_list(args.terrain_covariates," in ns


def test_static_banner_reads_the_module():
    from train_lightning import _static_banner
    from src.models.spatiotemporal_predictor import SpatioTemporalPredictor
    m = SpatioTemporalPredictor(hidden_dim=8, num_layers=1, num_static_channels=10,
                                num_dynamic_channels=1, use_location_encoder=False)
    line = _static_banner(tdl.static_file_list(True), m)
    assert line.startswith("Static channels:   10 ")
    assert "hm_static_terrain_aspcos_1000.tiff" in line and "module 10" in line


# --------------------------------------------------------------------- the log check

E2C = "--head_family pwl --free_scale True --mu_mse_weight 0.0 --kernel_size 5 --terrain_covariates True"
E2A = "--head_family pwl --free_scale True --mu_mse_weight 0.0"


def _banner(n, terrain, module=None):
    names = [os.path.basename(f) for f in tdl.static_file_list(terrain)][:n]
    return f"Static channels:   {n} ({', '.join(names)}); module {module or n}\n"


def _verify(log, flags):
    script = f'source "{BASE}"\nverify_static_fingerprint "{log}" probe "{flags}"\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def _log(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_static_fingerprint_accepts_e2c(tmp_path):
    assert _verify(_log(tmp_path, "ok.log", _banner(10, True)), E2C)


def test_static_fingerprint_accepts_e2a(tmp_path):
    assert _verify(_log(tmp_path, "e2a.log", _banner(7, False)), E2A)


def test_static_fingerprint_rejects_a_flag_that_never_arrived(tmp_path):
    assert not _verify(_log(tmp_path, "seven.log", _banner(7, False)), E2C)


def test_static_fingerprint_rejects_terrain_nobody_asked_for(tmp_path):
    assert not _verify(_log(tmp_path, "ten.log", _banner(10, True)), E2A)


def test_static_fingerprint_rejects_a_module_that_disagrees(tmp_path):
    assert not _verify(_log(tmp_path, "mod.log", _banner(10, True, module=7)), E2C)


def test_static_fingerprint_fails_closed(tmp_path):
    assert not _verify(tmp_path / "nope.log", E2C)
    assert not _verify(_log(tmp_path, "empty.log", "LOSS WEIGHTS\n"), E2C)


# --------------------------------------------------------------------- the runner

def _args(model):
    e = dict(os.environ, ALLOW_GLOBAL="1", MODEL=model)
    r = subprocess.run(["bash", RUNNER, "args"], capture_output=True, text=True, env=e, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)


def test_e2c_asks_for_terrain_and_e2a_does_not():
    assert "--terrain_covariates True" in _args("E2c")["MODEL_FLAGS"]
    assert "--terrain_covariates" not in _args("E2a")["TRAIN_ARGS"]


def test_fold_log_check_reads_the_static_channels():
    src = open(RUNNER).read()
    body = src[src.index("verify_fold_log() {"):]
    body = body[:body.index("\n}\n")]
    assert "verify_static_fingerprint" in body


def test_the_receipt_pins_the_terrain_rasters_for_e2c():
    src = open(RUNNER).read()
    assert "MODEL_EXTRA_INPUTS" in src[src.index("inputs_hash()"):src.index("inputs_hash()") + 400]
    assert all(n in src for n in TERRAIN)
