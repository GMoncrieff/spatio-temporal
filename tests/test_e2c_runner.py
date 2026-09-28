"""E2c on the global runner: its own trunk, its own inputs, and checks that fail closed.

E2c is E2a with three changes, and each one reaches the run by a different road:

* the trunk (``--kernel_size 5``) through MODEL_FLAGS -- checked off the module's banner by
  ``verify_trunk_fingerprint``;
* the land-only normalisation sidecar through ``--norm_stats_json``. If the named sidecar
  does not exist, train_lightning SAMPLES fresh stats the old way -- ocean and -32768 fill
  included -- and writes them to that very path, so a missing file becomes a contaminated
  one silently. ``require_norm_stats`` refuses first, and the log is read back;
* the land fold mask through ``FOLD_MASK``.

The two inputs are data, not code, so the smoke receipt's code hash cannot see them change.
The receipt therefore pins their bytes too (``inputs_hash``).

E2a must keep reading exactly what it trained on: the old mask and ``norm_stats.json``.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "scripts", "run_global_model.sh")
BASE = os.path.join(ROOT, "scripts", "conv_spline_base.sh")


def _args_dump(model, **env):
    e = dict(os.environ, ALLOW_GLOBAL="1", MODEL=model, **env)
    r = subprocess.run(["bash", RUNNER, "args"], capture_output=True, text=True, env=e, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)


def _last(flag, s):
    vals = re.findall(rf"{flag} (\S+)", s)
    return vals[-1] if vals else None


def test_e2c_effective_args():
    d = _args_dump("E2c")
    assert _last("--kernel_size", d["TRAIN_ARGS"]) == "5"
    assert _last("--head_family", d["TRAIN_ARGS"]) == "pwl"
    assert _last("--free_scale", d["TRAIN_ARGS"]) == "True"
    assert _last("--mu_mse_weight", d["TRAIN_ARGS"]) == "0.0"
    assert "--isqf_tails" not in d["TRAIN_ARGS"]
    assert (d["MODEL_FAMILY"], d["MODEL_PARAMS"]) == ("pwl", "15")
    assert d["FOLD_MASK"] == "data/raw/hm_global/fold_mask_b4_land_1000.tif"
    assert d["NORM_STATS"] == "data/conv_spline/norm_stats_E2c.json"
    assert d["MODEL_NORM_DOMAIN"] == "land"
    assert d["HIND_ROOT"].endswith("/g_E2c_hind") and d["PROD_ROOT"].endswith("/products/E2c")


def test_e2a_keeps_the_inputs_it_trained_on():
    d = _args_dump("E2a")
    assert "--kernel_size" not in d["TRAIN_ARGS"]
    assert d["FOLD_MASK"] == "data/raw/hm_global/fold_mask_b4_1000.tif"
    assert d["NORM_STATS"] == "data/conv_spline/norm_stats.json"
    assert d["MODEL_NORM_DOMAIN"] == "sampled"


def _body(fn):
    src = open(RUNNER).read()
    start = src.index(f"{fn}() {{")
    return src[start:src.index("\n}\n", start)]


def test_the_forecast_command_spells_no_kernel_and_no_sidecar_literal():
    body = _body("run_forecast_train")
    assert "--kernel_size" not in body
    assert '--norm_stats_json "$NORM_STATS"' in body
    assert "norm_stats.json" not in body


def test_every_fold_training_call_names_the_sidecar():
    src = open(RUNNER).read()
    calls = [c for c in src.split("scripts/run_hindcast_folds.py")[1:] if "--stage train" in c[:200]]
    assert len(calls) == 2, "expected the smoke and the hindcast fold calls"
    for c in calls:
        head = c[:c.index("--extra_train_args")]
        assert '--norm_stats_json "$NORM_STATS"' in head


def test_fold_log_check_reads_the_trunk_and_the_sidecar():
    body = _body("verify_fold_log")
    assert "verify_trunk_fingerprint" in body and "verify_norm_stats_log" in body


# --------------------------------------------------------------------- require_norm_stats

def _require(tmp_path, model, sidecar):
    script = (f'cd {ROOT}\nexport ALLOW_GLOBAL=1 MODEL={model} NORM_STATS={sidecar} '
              f'LOG_DIR={tmp_path}\nsource scripts/run_global_model.sh args >/dev/null\n'
              f'require_norm_stats probe\n')
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def test_require_norm_stats_refuses_a_missing_sidecar(tmp_path):
    assert not _require(tmp_path, "E2c", tmp_path / "absent.json")


def test_require_norm_stats_refuses_a_sampled_sidecar_for_e2c(tmp_path):
    p = tmp_path / "sampled.json"
    p.write_text('{"hm_mean": 0.1}')
    assert not _require(tmp_path, "E2c", p)


def test_require_norm_stats_accepts_a_land_sidecar_for_e2c(tmp_path):
    p = tmp_path / "land.json"
    p.write_text('{"hm_mean": 0.1, "stats_domain": "land"}')
    assert _require(tmp_path, "E2c", p)


def test_require_norm_stats_accepts_e2as_own_sidecar(tmp_path):
    assert _require(tmp_path, "E2a", os.path.join(ROOT, "data/conv_spline/norm_stats.json"))


# --------------------------------------------------------------------- the log check

def _verify_ns(log, path):
    script = f'source "{BASE}"\nverify_norm_stats_log "{log}" probe "{path}"\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


LOADED = ("Loaded normalization stats from {p}\n"
          "Using cached normalization stats (hm_mean=0.085, hm_std=0.153)\n")


def test_norm_stats_log_accepts_the_named_sidecar(tmp_path):
    p = tmp_path / "ok.log"
    p.write_text(LOADED.format(p="data/conv_spline/norm_stats_E2c.json"))
    assert _verify_ns(p, "data/conv_spline/norm_stats_E2c.json")


def test_norm_stats_log_rejects_e2as_sidecar_on_an_e2c_run(tmp_path):
    p = tmp_path / "wrong.log"
    p.write_text(LOADED.format(p="data/conv_spline/norm_stats.json"))
    assert not _verify_ns(p, "data/conv_spline/norm_stats_E2c.json")


def test_norm_stats_log_rejects_freshly_sampled_stats(tmp_path):
    """No 'Loaded' line: the sidecar was absent and the dataset sampled its own."""
    p = tmp_path / "sampled.log"
    p.write_text("Computing per-variable normalization statistics for static layers...\n")
    assert not _verify_ns(p, "data/conv_spline/norm_stats_E2c.json")


def test_norm_stats_log_fails_closed_on_a_missing_log(tmp_path):
    assert not _verify_ns(tmp_path / "nope.log", "data/conv_spline/norm_stats_E2c.json")


# --------------------------------------------------------------------- the receipt

def _stamp(tmp_path, d, inputs_hash):
    (tmp_path / "smoke_ok_E2c.stamp").write_text(
        f"code_hash={d['code_hash']}\nflags={d['MODEL_FLAGS']}\ninputs_hash={inputs_hash}\n")
    script = (f'cd {ROOT}\nexport ALLOW_GLOBAL=1 MODEL=E2c LOG_DIR={tmp_path}\n'
              f'source scripts/run_global_model.sh args >/dev/null\nrequire_smoke probe\n')
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_receipt_refuses_different_inputs(tmp_path):
    d = _args_dump("E2c", LOG_DIR=str(tmp_path))
    r = _stamp(tmp_path, d, "0" * 16)
    assert r.returncode != 0 and "different inputs" in r.stderr


def test_receipt_accepts_the_inputs_it_was_smoked_on(tmp_path):
    d = _args_dump("E2c", LOG_DIR=str(tmp_path))
    r = _stamp(tmp_path, d, d["inputs_hash"])
    assert r.returncode == 0, r.stderr
