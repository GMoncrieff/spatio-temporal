"""The conv-spline runners must write where the scorers read, and the verifiers must fail closed.

Two defects this pins, both of which made a 50-minute run worthless without saying so.

``run_central_experiment.sh`` hardcoded ``ROOT=data/ensemble/exp/<name>`` and ignored the
``EXP_ROOT`` its conv-spline callers export, so the runner wrote to one tree and
``score_distributional_model.py`` read another. Same for ``--log_dir`` against the log the
verifiers grep.

``verify_loss_weights`` then *passed* on a log that did not exist. Every expected weight in
this phase is 0.0 and awk coerces an empty string to 0, so three comparisons all read 0==0 and
the check reported success having read nothing -- failing open, in the one direction that is
silent. It is the check that exists to catch `run_hindcast_folds.py` injecting the frozen
product's weights, so an inherited weight set would have read as a finding.

No torch, no data: these drive bash directly.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BANNER = (
    "============================================================\n"
    "LOSS WEIGHTS\n"
    "============================================================\n"
    "MSE weight:        1.0 (fixed)\n"
    "SSIM weight:       {ssim}\n"
    "Laplacian weight:  {lap}\n"
    "Histogram weight:  {hist} (warmup: 5 epochs)\n"
)


def _verify(log_path, extra=""):
    """Run verify_loss_weights out of conv_spline_base.sh; True iff it accepts the log."""
    script = (
        f'source "{ROOT}/scripts/conv_spline_base.sh"\n'
        f'verify_loss_weights "{log_path}" probe "{extra}"\n'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def _write(tmp_path, name, **kw):
    p = tmp_path / name
    p.write_text(BANNER.format(**kw))
    return p


def test_verify_loss_weights_accepts_the_phase_baseline(tmp_path):
    assert _verify(_write(tmp_path, "good.log", ssim="0.0", lap="0.0", hist="0.0"))


def test_verify_loss_weights_rejects_the_inherited_product_weights(tmp_path):
    """--ssim 0.2 --laplacian 0.3 --histogram 1.0 is what run_hindcast_folds.py injects."""
    assert not _verify(_write(tmp_path, "bad.log", ssim="0.2", lap="0.3", hist="1.0"))


def test_verify_loss_weights_fails_closed_on_a_missing_log(tmp_path):
    """The regression: this returned success, because "" + 0 == 0 == the expected weight."""
    assert not _verify(tmp_path / "does_not_exist.log")


def test_verify_loss_weights_fails_closed_without_a_banner(tmp_path):
    p = tmp_path / "nobanner.log"
    p.write_text("epoch 0\nval_crps 0.001\n")
    assert not _verify(p)


def _runner_args(tmp_path, env_extra):
    """Run run_central_experiment.sh with a stub interpreter; return the args it would pass."""
    stub = tmp_path / "fakepy"
    stub.write_text('#!/usr/bin/env bash\necho "$@" >> "$STUB_LOG"\n')
    stub.chmod(0o755)
    out = tmp_path / "args.txt"
    env = {
        **os.environ, "PY": str(stub), "STUB_LOG": str(out),
        "BASE_ARGS": "--head_family spline", **env_extra,
    }
    for k in ("EXP_ROOT", "LOG_DIR"):
        if k not in env_extra:
            env.pop(k, None)
    subprocess.run(
        ["./scripts/run_central_experiment.sh", "probe_run", "0", "1", ""],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    return out.read_text().split()


def _flag(args, name):
    return args[args.index(name) + 1]


def test_runner_honours_exp_root_and_log_dir(tmp_path):
    """What the runner writes must be what run_conv_spline_{baseline,slate}.sh then reads."""
    args = _runner_args(tmp_path, {
        "EXP_ROOT": str(tmp_path / "exp"), "LOG_DIR": str(tmp_path / "logs"),
    })
    assert _flag(args, "--output_root") == str(tmp_path / "exp" / "probe_run")
    assert _flag(args, "--log_dir") == str(tmp_path / "logs")


@pytest.mark.skipif(
    os.path.exists(os.path.join(ROOT, "data/ensemble/exp/probe_run")),
    reason="would clobber a real legacy run directory",
)
def test_runner_defaults_to_the_legacy_tree(tmp_path):
    """run_dist_*.sh, run_model_slate.sh and promote_model_experiment.sh set neither var."""
    args = _runner_args(tmp_path, {})
    assert _flag(args, "--output_root") == "data/ensemble/exp/probe_run"
    assert _flag(args, "--log_dir") == "data/ensemble/logs"
    for d in ("data/ensemble/exp/probe_run", "data/ensemble/exp", "data/ensemble"):
        try:
            os.rmdir(os.path.join(ROOT, d))
        except OSError:
            break
