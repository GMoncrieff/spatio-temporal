"""E2d reads no protected-area data: the two WDPA masks leave the static channels.

``--protected_area_covariates False`` drops ``hm_static_iucn_{nostrict,strict}_1000.tiff``
from ``torchgeo_dataloader.static_file_list``, the one list the training dataset, the
large-area prediction loop (through ``PREDICT_STATS['static_files']``) and the statistics
builder read. Default on: E1v, E2a and E2c keep the channels they trained on.

The fill rule moved from positions to basenames with this change, because removing channels
5 and 6 shifts everything after them. The old positional set is pinned below as the control
that the new one reproduces it on every existing model's list.

What the run must prove is the ABSENCE of the channels, so ``verify_static_fingerprint``
checks the count derived from the flags (7 - 2 + 3 = 8 for E2d) AND that no iucn raster
appears in the banner -- a count alone passes an eight-channel list that still holds one.
"""
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torchgeo_dataloader as tdl  # noqa: E402

BASE = os.path.join(ROOT, "scripts", "conv_spline_base.sh")
RUNNER = os.path.join(ROOT, "scripts", "run_global_model.sh")
PA = ["hm_static_iucn_nostrict_1000.tiff", "hm_static_iucn_strict_1000.tiff"]
E2D_STATIC = ["hm_static_ele_1000.tiff", "hm_static_tas_1000.tiff", "hm_static_tasmin_1000.tiff",
              "hm_static_pr_1000.tiff", "hm_static_dpi_dsi_1000.tiff",
              "hm_static_terrain_slope_1000.tiff", "hm_static_terrain_aspsin_1000.tiff",
              "hm_static_terrain_aspcos_1000.tiff"]
# The positional fill set every model before E2d was trained and predicted with.
OLD_POSITIONAL_FILL = {0, 4, 5, 6, 7, 8, 9}


def _names(files):
    return [os.path.basename(f) for f in files]


def test_the_protected_area_files_are_the_two_iucn_masks():
    assert _names(tdl.PROTECTED_AREA_FILES) == PA


def test_e2d_list_is_e2c_without_the_masks():
    assert _names(tdl.static_file_list(True, False)) == E2D_STATIC
    assert _names(tdl.static_file_list(True, False)) == [
        n for n in _names(tdl.static_file_list(True)) if n not in PA]


def test_defaults_keep_the_masks():
    assert _names(tdl.static_file_list()) == _names(tdl.static_files)
    assert set(PA) <= set(_names(tdl.static_file_list(True)))
    assert len(tdl.static_file_list(True)) == 10


@pytest.mark.parametrize("terrain", [False, True])
def test_fill_set_by_name_reproduces_the_old_positions(terrain):
    """E1v/E2a (seven) and E2c (ten) must zero-fill exactly the channels they trained with."""
    names = _names(tdl.static_file_list(terrain))
    by_name = {i for i, n in enumerate(names) if n in tdl.NAN_TO_ZERO_STATIC}
    assert by_name == {i for i in OLD_POSITIONAL_FILL if i < len(names)}


def test_e2d_fills_elevation_dsi_and_terrain_only():
    filled = [n for n in E2D_STATIC if n in tdl.NAN_TO_ZERO_STATIC]
    assert filled == [E2D_STATIC[0], E2D_STATIC[4], *E2D_STATIC[5:]]
    arr = np.array([[np.nan, 2.0]], dtype=np.float32)
    for n in E2D_STATIC:
        out = tdl.prepare_static(arr, n, 0.0, 1.0, {})
        assert np.isnan(out[0, 0]) == (n not in filled), n


def test_get_dataloader_takes_the_flag():
    p = inspect.signature(tdl.get_dataloader).parameters["protected_area_covariates"]
    assert p.default is True


def test_an_e2c_sidecar_is_refused_by_an_e2d_dataset():
    """Ten channels of moments indexed by position would standardise slope with iucn's."""
    ds = object.__new__(tdl.HumanFootprintChipDataset)
    ds._static_files = tdl.static_file_list(True, False)
    ds.include_components = False
    ten = _names(tdl.static_file_list(True))
    stats = dict(hm_mean=0.1, hm_std=0.2, static_means=[0.0] * 10, static_stds=[1.0] * 10,
                 static_files=ten)
    with pytest.raises(ValueError, match="different static-channel set"):
        ds._load_norm_stats(stats)


def test_trainer_wires_the_flag_through_every_loader():
    tl = open(os.path.join(ROOT, "scripts", "train_lightning.py")).read()
    assert '"--protected_area_covariates"' in tl
    assert tl.count("protected_area_covariates=args.protected_area_covariates") == 3


def test_stats_builder_takes_the_flag():
    ns = open(os.path.join(ROOT, "scripts", "compute_norm_stats.py")).read()
    assert "not args.drop_protected_areas" in ns


# --------------------------------------------------------------------- the log check

E2C = "--head_family pwl --free_scale True --mu_mse_weight 0.0 --kernel_size 5 --terrain_covariates True"
E2D = E2C + " --protected_area_covariates False"


def _banner(names, module=None):
    return f"Static channels:   {len(names)} ({', '.join(names)}); module {module or len(names)}\n"


def _verify(log, flags):
    script = f'source "{BASE}"\nverify_static_fingerprint "{log}" probe "{flags}"\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def _log(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_static_fingerprint_accepts_e2d(tmp_path):
    assert _verify(_log(tmp_path, "ok.log", _banner(E2D_STATIC)), E2D)


def test_static_fingerprint_still_accepts_e2c(tmp_path):
    assert _verify(_log(tmp_path, "e2c.log", _banner(_names(tdl.static_file_list(True)))), E2C)


def test_static_fingerprint_rejects_masks_that_were_never_dropped(tmp_path):
    assert not _verify(_log(tmp_path, "ten.log", _banner(_names(tdl.static_file_list(True)))), E2D)


def test_static_fingerprint_rejects_a_mask_hidden_in_the_right_count(tmp_path):
    """Eight channels, terrain in order, and still a WDPA mask: the count alone passes this."""
    sneaky = E2D_STATIC[:4] + [PA[1]] + E2D_STATIC[5:]
    assert len(sneaky) == 8
    assert not _verify(_log(tmp_path, "sneaky.log", _banner(sneaky)), E2D)


def test_static_fingerprint_rejects_e2d_banner_under_e2c_flags(tmp_path):
    assert not _verify(_log(tmp_path, "eight.log", _banner(E2D_STATIC)), E2C)


def test_static_fingerprint_fails_closed_for_e2d(tmp_path):
    assert not _verify(tmp_path / "nope.log", E2D)
    assert not _verify(_log(tmp_path, "empty.log", "LOSS WEIGHTS\n"), E2D)


# --------------------------------------------------------------------- the runner

def _args(model):
    e = dict(os.environ, ALLOW_GLOBAL="1", MODEL=model)
    r = subprocess.run(["bash", RUNNER, "args"], capture_output=True, text=True, env=e, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)


def test_e2d_is_e2c_plus_the_one_flag():
    e2c, e2d = _args("E2c"), _args("E2d")
    assert e2d["MODEL_FLAGS"] == e2c["MODEL_FLAGS"] + " --protected_area_covariates False"
    assert "--protected_area_covariates" not in e2c["TRAIN_ARGS"]
    assert e2d["FOLD_MASK"] == e2c["FOLD_MASK"]
    assert e2d["NORM_STATS"].endswith("norm_stats_E2d.json")


def test_e2d_has_its_own_receipt_and_roots():
    e2d = _args("E2d")
    assert e2d["STAMP"].endswith("smoke_ok_E2d.stamp")
    assert e2d["HIND_ROOT"].endswith("g_E2d_hind") and e2d["PROD_ROOT"].endswith("products/E2d")
