#!/usr/bin/env bash
# The conv-spline phase's baseline, defined once. Every runner sources this; every experiment
# is a stated delta from it.
#
# **Every loss weight must be named here explicitly.** `run_hindcast_folds.py` injects the
# frozen product's own weights (--ssim_weight 0.2 --laplacian_weight 0.3 --histogram_weight 1.0)
# into every fold command, and --extra_train_args is appended last, so argparse takes the final
# occurrence. Anything BASE_ARGS does not name is silently inherited from a product this phase
# is not evaluating. That is not hypothetical -- it cost a 35-minute run in the previous phase,
# caught only by reading the run's own LOSS WEIGHTS banner.
#
# The neighbourhood context -- distance to past change and the neighbourhood HM summaries --
# goes into the TRUNK, beside elevation and climate, and into no head. That is not a setting:
# --trunk_context, --central_context and --quantile_context no longer exist, and the wiring is
# part of the model. e1 fed the heads and never the trunk; b1 and everything after it feed the
# trunk and never the heads.
#
# Nothing to name in BASE_ARGS, therefore, and nothing that --extra_train_args could override
# back. What is still worth verifying is that the covariate arrived at all: the run's own log
# reports the channel count the trunk was built with, read off the module rather than off a
# flag, and verify_context_wiring below refuses a run where that is zero or disagrees with the
# channel count --context_radii / --hm_context_stats imply.

export REGION="${REGION:-config/region_africa.geojson}"
export FOLD_MASK="${FOLD_MASK:-data/raw/hm_global/fold_mask_b4_1000.tif}"
export VAL_STRIDE="${VAL_STRIDE:-1024}"
export MAX_EPOCHS="${MAX_EPOCHS:-150}"
export FOLDS="${FOLDS:-1,2}"
export GPUS="${GPUS:-0,1}"
export EXP_ROOT="${EXP_ROOT:-data/conv_spline/exp}"
# Named here, not spelled out in each runner: the verifiers read the log that
# run_central_experiment.sh writes, and two spellings of one path is how the check
# ended up pointed at a file that never existed.
export LOG_DIR="${LOG_DIR:-data/conv_spline/logs}"
export SCORE_DIR="${SCORE_DIR:-data/conv_spline/scores}"
export PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"

EXPECT_SSIM="0.0"
EXPECT_LAP="0.0"
EXPECT_HIST="0.0"
# The auxiliary MSE on E[Q]. Named here because it was the one loss weight the phase left to
# an argparse default, which is rule 16's exact shape: every arm carried a second objective
# nobody chose. The free-scale arms (E0a, E1b, E1c, E2a) set it to 0.0 -- an MSE term pinning
# the first moment is not a neutral bystander to an experiment about where the width comes
# from -- and they pass `--mu_mse_weight 0.0` in their own flags, which BASE_ARGS being first
# lets them override.
EXPECT_MU_MSE="1.0"

# val_crps, not val_total_loss: an experiment that sets --mu_mse_weight 0 would otherwise be
# selecting epochs on a different quantity from every other run in the slate.
export BASE_ARGS="--head_family spline --central_residual True --checkpoint_monitor val_crps \
--mu_mse_weight ${EXPECT_MU_MSE} \
--ssim_weight ${EXPECT_SSIM} --laplacian_weight ${EXPECT_LAP} --histogram_weight ${EXPECT_HIST}"

# Read the effective weights back out of the fold log and refuse to continue if they are not
# what BASE_ARGS asked for. Verify the fingerprint before trusting a retrain: a run on
# inherited defaults reads exactly like a real finding.
verify_loss_weights() {
  local log="$1" name="$2" expect_extra="${3:-}"
  local got_ssim got_lap got_hist got_mse
  # Fail CLOSED on a log that is missing or carries no banner. Every expected weight in this
  # phase is 0.0 and awk coerces an empty string to 0, so a missing log made all three
  # comparisons read 0==0 and the check PASSED having read nothing -- the exact failure this
  # function exists to prevent, in the one direction that is silent. Rule: prove a check
  # fires on a control, or it checks nothing.
  if [ ! -r "$log" ]; then
    echo "FATAL: ${name}: no readable fold log at ${log}; loss weights unverified." >&2
    echo "       The run cannot be trusted -- an inherited weight set reads as a finding." >&2
    return 1
  fi
  if ! grep -q "LOSS WEIGHTS" "$log"; then
    echo "FATAL: ${name}: ${log} has no LOSS WEIGHTS banner; loss weights unverified." >&2
    return 1
  fi
  got_mse=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/MSE weight:/ {print $3; exit}')
  got_ssim=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/SSIM weight:/ {print $3; exit}')
  got_lap=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/Laplacian weight:/ {print $3; exit}')
  got_hist=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/Histogram weight:/ {print $3; exit}')
  local want_ssim="$EXPECT_SSIM" want_lap="$EXPECT_LAP" want_mse="$EXPECT_MU_MSE"
  case "$expect_extra" in
    *--ssim_weight*)      want_ssim=$(sed -E 's/.*--ssim_weight ([0-9.]+).*/\1/' <<<"$expect_extra") ;;
  esac
  case "$expect_extra" in
    *--laplacian_weight*) want_lap=$(sed -E 's/.*--laplacian_weight ([0-9.]+).*/\1/' <<<"$expect_extra") ;;
  esac
  case "$expect_extra" in
    *--mu_mse_weight*)    want_mse=$(sed -E 's/.*--mu_mse_weight ([0-9.]+).*/\1/' <<<"$expect_extra") ;;
  esac
  if [ -z "$got_ssim" ] || [ -z "$got_lap" ] || [ -z "$got_hist" ] || [ -z "$got_mse" ]; then
    echo "FATAL: ${name}: could not parse all four weights out of ${log}" >&2
    echo "  got  mse='${got_mse}' ssim='${got_ssim}' lap='${got_lap}' hist='${got_hist}'" >&2
    return 1
  fi
  if ! awk -v a="$got_mse" -v b="$want_mse" 'BEGIN{exit !(a+0==b+0)}' \
     || ! awk -v a="$got_ssim" -v b="$want_ssim" 'BEGIN{exit !(a+0==b+0)}' \
     || ! awk -v a="$got_lap" -v b="$want_lap" 'BEGIN{exit !(a+0==b+0)}' \
     || ! awk -v a="$got_hist" 'BEGIN{exit !(a+0==0)}'; then
    echo "FATAL: ${name} trained on the wrong loss weights." >&2
    echo "  got  mse=${got_mse} ssim=${got_ssim} lap=${got_lap} hist=${got_hist}" >&2
    echo "  want mse=${want_mse} ssim=${want_ssim} lap=${want_lap} hist=0" >&2
    return 1
  fi
  echo "  ✓ ${name}: loss weights verified (mse=${got_mse} ssim=${got_ssim} lap=${got_lap} hist=${got_hist})"
}

# The trunk must actually receive the context, and under --central_residual the ConvLSTM's
# gradient from the *central* loss is zero by construction -- so a trunk that never got the
# covariate trains, scores, and looks exactly like a trunk that got it and ignored it. There
# is no flag to inspect any more, so the fingerprint is the module's own channel count, and
# this cross-checks it against the count --context_radii / --hm_context_stats imply. Prove a
# check fires on a control before trusting it (tests/test_conv_spline_paths.py).
verify_context_wiring() {
  local log="$1" name="$2" n_args n_trunk
  if [ ! -r "$log" ]; then
    echo "FATAL: ${name}: no readable fold log at ${log}; context wiring unverified." >&2
    return 1
  fi
  n_args=$(grep -m1 "^Context channels:" "$log" | awk '{print $3}')
  n_trunk=$(grep -m1 "^Context into trunk:" "$log" | awk '{print $4}')
  if [ -z "$n_trunk" ] || [ -z "$n_args" ]; then
    echo "FATAL: ${name}: ${log} carries no context fingerprint." >&2
    echo "  Context channels='${n_args:-<nothing read>}' into trunk='${n_trunk:-<nothing read>}'" >&2
    return 1
  fi
  if ! [ "$n_trunk" -gt 0 ] 2>/dev/null; then
    echo "FATAL: ${name}: the trunk was built with ${n_trunk} context channels." >&2
    echo "       The covariate never reached the model; the run is not this configuration." >&2
    return 1
  fi
  if [ "$n_trunk" != "$n_args" ]; then
    echo "FATAL: ${name}: trunk has ${n_trunk} context channels, the flags imply ${n_args}." >&2
    echo "       Check --context_radii / --hm_context_stats against the checkpoint, if any." >&2
    return 1
  fi
  if ! grep -q "^Context into trunk:.*heads: none" "$log"; then
    echo "FATAL: ${name}: the log does not say the heads are excluded." >&2
    return 1
  fi
  echo "  ✓ ${name}: context wiring verified (${n_trunk} channels -> trunk, heads none)"
}

# Africa, not southern Africa, and not the globe. Southern Africa's far-field band holds ZERO
# pixels and its [0,0.01) HM stratum is 6% of the region against 40% of Africa, so a
# stratified finding measured there is provisional at best and has already cost this project
# a whole phase. The globe is for promotion only, on instruction, never for iteration.
guard_region() {
  case "${REGION}" in
    *africa.geojson) ;;
    *large*|*global*)
      echo "FATAL: REGION looks global. This phase iterates on Africa; promote on " >&2
      echo "       instruction only, with ALLOW_GLOBAL=1." >&2
      [ "${ALLOW_GLOBAL:-0}" = "1" ] || return 1 ;;
    *) echo "  ! REGION=${REGION} is neither Africa nor global -- continuing, but check it" >&2 ;;
  esac
}
