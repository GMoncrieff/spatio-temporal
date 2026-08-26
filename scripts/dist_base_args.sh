# Shared configuration for the distributional phase. Sourced by run_dist_floor.sh and
# run_dist_slate.sh so the floor and the variants cannot describe different baselines.
#
# **Every loss weight must be named here explicitly.** `run_hindcast_folds.py` injects the
# incumbent's own weights (--ssim_weight 0.2 --laplacian_weight 0.3 --histogram_weight 1.0)
# into every fold command, and --extra_train_args is appended last, so argparse takes the
# final occurrence. Anything BASE_ARGS does not name is therefore silently inherited from the
# frozen triple-head product. That is not hypothetical: the first floor run was launched with
# SSIM 0.2 and Laplacian 0.3 active while the slate defined D0 as the probabilistic loss
# alone, and every D3/D4 comparison against it would have been meaningless while looking fine.
#
# D0 is the probabilistic loss alone. The spatial terms are not assumed to be necessary; the
# ladder is probabilistic -> +SSIM (D3) -> +Laplacian (D4) -> both (D5, only if each earns it).

export FOLD_MASK="${FOLD_MASK:-data/raw/hm_global/fold_mask_b4_1000.tif}"
export VAL_STRIDE="${VAL_STRIDE:-1024}"
export MAX_EPOCHS="${MAX_EPOCHS:-150}"

EXPECT_SSIM="0.0"
EXPECT_LAP="0.0"
EXPECT_HIST="0.0"

# val_crps, not val_total_loss: D6 sets --mu_mse_weight 0, so a monitor carrying the auxiliary
# MSE would select epochs on a different quantity for that run than for every other one.
export BASE_ARGS="--head_family spline --central_residual True --central_context True \
--quantile_context True --checkpoint_monitor val_crps \
--ssim_weight ${EXPECT_SSIM} --laplacian_weight ${EXPECT_LAP} --histogram_weight ${EXPECT_HIST}"

# Read the effective weights back out of the fold log and refuse to continue if they are not
# what BASE_ARGS asked for. Verify the fingerprint before trusting a retrain: a run on
# inherited defaults reads exactly like a real finding.
verify_loss_weights() {
  local log="$1" name="$2" expect_extra="${3:-}"
  local got_ssim got_lap got_hist
  got_ssim=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/SSIM weight:/ {print $3; exit}')
  got_lap=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/Laplacian weight:/ {print $3; exit}')
  got_hist=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/Histogram weight:/ {print $3; exit}')
  local want_ssim="$EXPECT_SSIM" want_lap="$EXPECT_LAP"
  case "$expect_extra" in
    *--ssim_weight*)      want_ssim=$(sed -E 's/.*--ssim_weight ([0-9.]+).*/\1/' <<<"$expect_extra") ;;
  esac
  case "$expect_extra" in
    *--laplacian_weight*) want_lap=$(sed -E 's/.*--laplacian_weight ([0-9.]+).*/\1/' <<<"$expect_extra") ;;
  esac
  if ! awk -v a="$got_ssim" -v b="$want_ssim" 'BEGIN{exit !(a+0==b+0)}' \
     || ! awk -v a="$got_lap" -v b="$want_lap" 'BEGIN{exit !(a+0==b+0)}' \
     || ! awk -v a="$got_hist" 'BEGIN{exit !(a+0==0)}'; then
    echo "FATAL: ${name} trained on the wrong loss weights." >&2
    echo "  got  ssim=${got_ssim} lap=${got_lap} hist=${got_hist}" >&2
    echo "  want ssim=${want_ssim} lap=${want_lap} hist=0" >&2
    echo "  (run_hindcast_folds.py injects the incumbent's weights; name yours in BASE_ARGS)" >&2
    return 1
  fi
  echo "  ✓ ${name} loss weights verified: ssim=${got_ssim} lap=${got_lap} hist=${got_hist}"
}

# Prove the gate's own flags took effect. The gate exists to test two levers, and a run where
# neither engaged would report "no effect" indistinguishably from a lever that does nothing —
# which is how this project lost a whole k=5 run once. Both leave a signature in the log.
verify_gate_flags() {
  local log="$1" name="$2" flags="$3" ok=1
  case "$flags" in
    *--weight_avg_last*)
      if grep -q "\[weight averaging\] wrote the mean of the last" "$log"; then
        echo "  ✓ ${name} weight averaging: $(grep -o 'mean of the last [0-9]* epochs' "$log" | tail -1)"
      else
        echo "FATAL: ${name} asked for --weight_avg_last and the log has no averaging line." >&2
        ok=0
      fi
      if grep -q "Prediction will use the end-of-training checkpoint" "$log"; then
        echo "  ✓ ${name} predicts from the averaged end-of-training checkpoint"
      else
        echo "FATAL: ${name} averaged but prediction did not repoint at the end checkpoint." >&2
        ok=0
      fi ;;
  esac
  case "$flags" in
    *--lr_schedule\ cosine*)
      if grep -q "\[lr schedule\] cosine" "$log"; then
        echo "  ✓ ${name} $(grep -o '\[lr schedule\].*' "$log" | tail -1)"
      else
        echo "FATAL: ${name} asked for cosine and the log shows no schedule." >&2
        ok=0
      fi ;;
  esac
  case "$flags" in
    *--checkpoint_select\ final*)
      if grep -q "Prediction will use the end-of-training checkpoint" "$log"; then
        echo "  ✓ ${name} predicts from the final epoch, not a selected one"
      else
        echo "FATAL: ${name} asked for --checkpoint_select final and prediction did not." >&2
        ok=0
      fi ;;
  esac
  [ "$ok" = 1 ]
}

# Prove the covariate reached the model. E1 is the round's headline experiment and its whole
# content is twelve context channels instead of eight; a run that silently fell back to eight
# would read as "the covariate does nothing".
verify_context_channels() {
  local log="$1" name="$2" flags="$3"
  case "$flags" in
    *--hm_context_stats*)
      # -i: the banner prints "Context channels:", capitalised. This grep was written
      # lowercase and never run against real output, so it failed on every correct run and
      # killed the slate under set -e -- a guard against a missing covariate that fired when
      # the covariate was present. Confirm the thing a check greps for is actually emitted.
      local line
      line=$(grep -oi "context channels:.*" "$log" | tail -1)
      # Two ways this must fail, and the original caught neither. It grepped lowercase while
      # the banner capitalises, so it failed on every CORRECT run and killed the slate under
      # set -e. Fixing only the case then made it pass a run whose banner says "no hm context"
      # -- the exact silent fallback it exists to catch. Assert the covariate is present, not
      # merely that some channel line was printed.
      if [ -z "$line" ]; then
        echo "FATAL: ${name} asked for --hm_context_stats and the log shows no channel line." >&2
        return 1
      fi
      case "$line" in
        *"no hm context"*)
          echo "FATAL: ${name} asked for --hm_context_stats but the model built WITHOUT it:" >&2
          echo "       ${line}" >&2
          return 1 ;;
      esac
      echo "  ✓ ${name} ${line}" ;;
  esac
  return 0
}
