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
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BANNER = (
    "============================================================\n"
    "LOSS WEIGHTS\n"
    "============================================================\n"
    "MSE weight:        {mse}\n"
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


def _write(tmp_path, name, mse="1.0", **kw):
    p = tmp_path / name
    p.write_text(BANNER.format(mse=mse, **kw))
    return p


def test_verify_loss_weights_accepts_the_phase_baseline(tmp_path):
    assert _verify(_write(tmp_path, "good.log", ssim="0.0", lap="0.0", hist="0.0"))


def test_verify_loss_weights_rejects_the_inherited_product_weights(tmp_path):
    """--ssim 0.2 --laplacian 0.3 --histogram 1.0 is what run_hindcast_folds.py injects."""
    assert not _verify(_write(tmp_path, "bad.log", ssim="0.2", lap="0.3", hist="1.0"))


def test_verify_loss_weights_accepts_the_auxiliary_mse_at_its_named_value(tmp_path):
    """BASE_ARGS names --mu_mse_weight 1.0; a log reading 1.0 is the baseline."""
    assert _verify(_write(tmp_path, "mse1.log", mse="1.0", ssim="0.0", lap="0.0", hist="0.0"))


def test_verify_loss_weights_accepts_zero_mse_when_the_arm_asks_for_it(tmp_path):
    """E0a, E1b, E1c and E2a carry --mu_mse_weight 0.0: CRPS alone, no term on E[Q]."""
    log = _write(tmp_path, "mse0.log", mse="0.0", ssim="0.0", lap="0.0", hist="0.0")
    assert _verify(log, extra="--mu_mse_weight 0.0")


def test_verify_loss_weights_rejects_an_inherited_mse_on_a_free_scale_arm(tmp_path):
    """The regression: the arm asked for 0.0 and the run trained at the 1.0 default.

    Before this was checked the banner printed the literal "1.0 (fixed)" whatever the flag
    said, so the log could not distinguish the two runs at all -- an arm defined by not
    having an MSE term would have read as one that did, or vice versa, with nothing to see.
    """
    log = _write(tmp_path, "inherited.log", mse="1.0", ssim="0.0", lap="0.0", hist="0.0")
    assert not _verify(log, extra="--mu_mse_weight 0.0")


def test_verify_loss_weights_fails_closed_without_the_mse_line(tmp_path):
    """An arm expecting 0.0 is exactly where an unparsed weight coerces to 0 and passes."""
    p = tmp_path / "no_mse_line.log"
    p.write_text("============\nLOSS WEIGHTS\n============\n"
                 "SSIM weight:       0.0\nLaplacian weight:  0.0\n"
                 "Histogram weight:  0.0 (warmup: 5 epochs)\n")
    assert not _verify(p, extra="--mu_mse_weight 0.0")


def test_banner_reports_the_flag_rather_than_a_literal():
    """The fingerprint has to move with the run, or every check above passes vacuously."""
    src = open(os.path.join(ROOT, "scripts", "train_lightning.py")).read()
    line = next(l for l in src.splitlines() if "MSE weight:" in l and "print" in l)
    assert "mu_mse_weight" in line, f"MSE banner is hardcoded: {line.strip()}"


def test_verify_loss_weights_fails_closed_on_a_missing_log(tmp_path):
    """The regression: this returned success, because "" + 0 == 0 == the expected weight."""
    assert not _verify(tmp_path / "does_not_exist.log")


def test_verify_loss_weights_fails_closed_without_a_banner(tmp_path):
    p = tmp_path / "nobanner.log"
    p.write_text("epoch 0\nval_crps 0.001\n")
    assert not _verify(p)


CTX_LOG = "Context channels:  {n_args} (radii 1,2,4)\nContext into trunk: {n_trunk} channels; heads: {heads}\n"


def _verify_ctx(log_path):
    """Run verify_context_wiring out of conv_spline_base.sh; True iff it accepts the log."""
    script = (
        f'source "{ROOT}/scripts/conv_spline_base.sh"\n'
        f'verify_context_wiring "{log_path}" probe\n'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def _ctx_log(tmp_path, name, n_args=12, n_trunk=12, heads="none"):
    p = tmp_path / name
    p.write_text(CTX_LOG.format(n_args=n_args, n_trunk=n_trunk, heads=heads))
    return p


def test_verify_context_wiring_accepts_the_hardwired_arrangement(tmp_path):
    assert _verify_ctx(_ctx_log(tmp_path, "ok.log"))


def test_verify_context_wiring_rejects_a_trunk_that_got_nothing(tmp_path):
    """There is no flag to misread any more, so this is the failure that remains: the
    covariate is configured and the module was built without it."""
    assert not _verify_ctx(_ctx_log(tmp_path, "zero.log", n_args=12, n_trunk=0))


def test_verify_context_wiring_rejects_a_width_disagreement(tmp_path):
    """--context_radii / --hm_context_stats imply 12; the trunk was built for 8."""
    assert not _verify_ctx(_ctx_log(tmp_path, "mismatch.log", n_args=12, n_trunk=8))


def test_verify_context_wiring_rejects_the_argparse_default_covariate(tmp_path):
    """The two counts above agree whenever the flags and the module agree -- including when
    both are argparse defaults nobody chose.

    That is how b1 came to be scripted on eight trunk channels while the phase doc called it
    "e1 plus exactly one change": e1's covariate is --context_radii 3,30,100 with
    --hm_context_stats mean,max (twelve), and BASE_ARGS named neither, so the default
    --context_radii 1,3,10,30,100 with no HM summaries rode along and the wiring check read
    8 == 8 and passed. Rule 16, and rule 5's "ON is not the same as ONLY". This is the only
    line that can tell the phase's covariate from the default one, so it gets its control.
    """
    assert not _verify_ctx(_ctx_log(tmp_path, "default_ctx.log", n_args=8, n_trunk=8))
    assert not _verify_ctx(_ctx_log(tmp_path, "mean_only.log", n_args=9, n_trunk=9))


def test_base_args_names_the_phase_covariate_and_the_count_matches_it():
    """The named count and the named flags must not drift apart -- context_channel_count is
    the one definition, so it is what decides, not a number typed twice."""
    import sys
    sys.path.insert(0, ROOT)
    from src.models.change_weights import context_channel_count

    out = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/scripts/conv_spline_base.sh"\n'
                       f'printf "%s\\n%s" "$BASE_ARGS" "$EXPECT_CTX_CHANNELS"'],
        capture_output=True, text=True, check=True).stdout
    args, expect = out.rsplit("\n", 1)
    assert "--context_radii 3,30,100" in args
    assert "--hm_context_stats mean,max" in args
    assert "--hm_context_radii 3,30,100" in args
    assert int(expect) == context_channel_count((3, 30, 100), ("mean", "max"), (3, 30, 100))


def test_verify_context_wiring_rejects_a_head_consumer(tmp_path):
    """e1's arrangement, reintroduced. The heads must not receive it a second time."""
    assert not _verify_ctx(
        _ctx_log(tmp_path, "heads.log", heads="central, quantile"))


def test_verify_context_wiring_fails_closed_on_a_missing_log(tmp_path):
    assert not _verify_ctx(tmp_path / "does_not_exist.log")


def test_verify_context_wiring_fails_closed_without_the_fingerprint(tmp_path):
    p = tmp_path / "nofingerprint.log"
    p.write_text("epoch 0\nval_crps 0.001\n")
    assert not _verify_ctx(p)


def test_base_args_no_longer_names_a_context_flag():
    """The wiring is part of the model, so there is nothing for --extra_train_args to undo."""
    out = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/scripts/conv_spline_base.sh"\nprintf "%s" "$BASE_ARGS"'],
        capture_output=True, text=True, check=True).stdout
    for flag in ("--trunk_context", "--central_context", "--quantile_context"):
        assert flag not in out, f"{flag} is back in BASE_ARGS; it no longer exists"
    assert "--mu_mse_weight 1.0" in out


def test_the_context_flags_are_gone_from_argparse():
    """Removed, not defaulted: a flag that is accepted and ignored is the dormant path this
    branch exists to avoid, and a script still passing one should fail loudly."""
    src = open(os.path.join(ROOT, "scripts", "train_lightning.py")).read()
    for flag in ('"--trunk_context"', '"--central_context"', '"--quantile_context"'):
        assert flag not in src, f"{flag} is still an argparse argument"


def test_no_script_passes_a_removed_context_flag():
    """Anything still passing one would die with 'unrecognized arguments' mid-run.

    Comment lines are excluded on purpose: conv_spline_base.sh names the three removed flags
    in prose, to say that they are gone and why. Naming them is the point of that comment.
    """
    import glob
    for path in glob.glob(os.path.join(ROOT, "scripts", "*.sh")):
        code = "\n".join(l for l in open(path).read().splitlines()
                         if not l.lstrip().startswith("#"))
        for flag in ("--trunk_context", "--central_context", "--quantile_context"):
            assert flag not in code, f"{os.path.basename(path)} still passes {flag}"


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


# ------------------------------------------------------ weight averaging, and its control

WA_LOG = ("[weight averaging] wrote the mean of the last {n} epochs into 214 tensors\n"
          "Prediction will use the end-of-training checkpoint: /x/final_fold1_1.ckpt\n")


def _verify_wa(log_path, want="20"):
    script = (f'source "{ROOT}/scripts/conv_spline_base.sh"\n'
              f'verify_weight_averaging "{log_path}" probe {want}\n')
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True).returncode == 0


def _wa_log(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_verify_weight_averaging_accepts_the_phase_baseline(tmp_path):
    assert _verify_wa(_wa_log(tmp_path, "ok.log", WA_LOG.format(n=20)))


def test_verify_weight_averaging_rejects_a_run_that_selected_one_epoch(tmp_path):
    """The failure this exists to catch: averaging silently off, so the run carries the
    argmin's gate band (25-115% of the mean, measured) and reads as a null."""
    body = "Checkpoint selection monitors: val_crps\nepoch=67-step=884.ckpt\n"
    assert not _verify_wa(_wa_log(tmp_path, "argmin.log", body))


def test_verify_weight_averaging_rejects_a_different_window(tmp_path):
    assert not _verify_wa(_wa_log(tmp_path, "w5.log", WA_LOG.format(n=5)))


def test_verify_weight_averaging_rejects_averaging_that_was_not_predicted(tmp_path):
    """Averaging that ran but never reached prediction is the silent half of this."""
    body = "[weight averaging] wrote the mean of the last 20 epochs into 214 tensors\n"
    assert not _verify_wa(_wa_log(tmp_path, "notpred.log", body))


def test_verify_weight_averaging_fails_closed_on_a_missing_log(tmp_path):
    assert not _verify_wa(tmp_path / "nope.log")


def test_base_args_names_the_weight_averaging_window():
    out = subprocess.run(
        ["bash", "-c", f'source "{ROOT}/scripts/conv_spline_base.sh"\nprintf "%s" "$BASE_ARGS"'],
        capture_output=True, text=True, check=True).stdout
    assert "--weight_avg_last 20" in out


def test_both_runners_verify_weight_averaging():
    """A verifier that no runner calls is a verifier that checks nothing."""
    for f in ("run_conv_spline_baseline.sh", "run_conv_spline_slate.sh"):
        code = open(os.path.join(ROOT, "scripts", f)).read()
        assert "verify_weight_averaging" in code, f"{f} does not verify weight averaging"


def test_the_scale_arms_runner_keeps_the_doc_s_order_and_the_mse_subtraction():
    """E0a must come first and alone, and every --free_scale arm must zero the auxiliary MSE.

    Order is the design: each arm below E0a conflates the horizon-cumulative constraint with
    the anchor/scale factorisation, so E0a is the only one that changes one thing. And an MSE
    term pinning E[Q] is not a bystander to an experiment about where the width comes from --
    a null would not separate "the factorisation was earning its keep" from "the MSE supplied
    what it used to".
    """
    code = open(os.path.join(ROOT, "scripts", "run_conv_spline_scale_arms.sh")).read()
    arms = re.findall(r'^\s+"([A-Za-z0-9_]+)\|([^|]*)\|', code, re.M)
    names = [a for a, _ in arms]
    assert names[0].startswith("E0a"), f"E0a is not first: {names}"
    assert names[-1].startswith("E1c"), f"E1c is not last: {names}"
    assert len(names) == 6, names
    for name, flags in arms:
        if "--free_scale" in flags or name.startswith("E0a"):
            assert "--mu_mse_weight 0.0" in flags, f"{name} keeps the auxiliary MSE"
    # E0a must use the narrow flag, never --free_scale, or it cannot answer its own question.
    e0a = dict(arms)["E0a_nocumw"]
    assert "--spline_cumulative_width False" in e0a
    assert "--free_scale" not in e0a, "E0a respelled as --free_scale answers a different question"
    # both --isqf_space values get an arm
    spaces = {s for _, f in arms for s in ("logit", "neglog") if f"--isqf_space {s}" in f}
    assert spaces == {"logit", "neglog"}, spaces


def test_the_scale_arms_runner_verifies_everything_the_baseline_does():
    code = open(os.path.join(ROOT, "scripts", "run_conv_spline_scale_arms.sh")).read()
    for v in ("verify_loss_weights", "verify_context_wiring", "verify_weight_averaging"):
        assert v in code, f"the scale-arm runner does not call {v}"
    assert "guard_region" in code
