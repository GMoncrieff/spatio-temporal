#!/usr/bin/env bash
# Smoke EVERY arm before committing a multi-hour slate to it.
#
# Why this exists: of the ten defects found setting this phase up, EIGHT were invisible to the
# test suite and only appeared when the code met a real raster -- including a prediction writer
# that could not decode pwl or isqf at all (E1/E2/E4 and all four scale arms would each have
# trained for an hour and written nothing), and two errors in section 4.1's own specification.
# The code was internally consistent and carefully unit-tested throughout.
#
# One epoch, eight chips, one fold, one window, and only a handful of 128 px prediction blocks
# -- so this is a CODE PATH check, not a measurement. Nothing here is a result. What it proves
# per arm: the loss runs, the head decodes, the quantile raster is written, and what comes back
# off disk is a monotone finite quantile function with the channel count the arm implies.
#
#   ./scripts/smoke_conv_spline_arms.sh                 # every arm, ~5 min each
#   ONLY="E1 E2a" ./scripts/smoke_conv_spline_arms.sh   # a subset
set -euo pipefail

source "$(dirname "$0")/conv_spline_base.sh"
guard_region

SMOKE_ROOT="${SMOKE_ROOT:-/mnt/hdd1/spatio-temporal/data/conv_spline/smoke}"
SMOKE_LOGS="${SMOKE_LOGS:-data/conv_spline/logs/smoke}"
BLOCKS="${BLOCKS:-40}"
mkdir -p "$SMOKE_ROOT" "$SMOKE_LOGS"

# name | extra flags | expected params/horizon | expected head family
# The family is checked too: the banner said "Spline head" for every family until
# 2026-09-15, so an arm could run the wrong head and log as though it had not.
ARMS=(
  "b1|--head_family spline|29|spline"
  "E0|--u_grid_spacing skew|29|spline"
  "E1|--head_family isqf|16|isqf"
  "E2|--head_family pwl|16|pwl"
  "E3|--crps_z_weight 1.0 --crps_z_scale 0.001|29|spline"
  "E4|--head_family pwl --spline_gap_floor True|16|pwl"
  "E5|--spline_knots skew14|29|spline"
  "E0a|--spline_cumulative_width False --mu_mse_weight 0.0|29|spline"
  "E1a_logit|--head_family isqf --isqf_tails True --isqf_space logit|18|isqf"
  "E1a_neglog|--head_family isqf --isqf_tails True --isqf_space neglog|18|isqf"
  "E1b|--head_family isqf --free_scale True --mu_mse_weight 0.0|15|isqf"
  "E2a|--head_family pwl --free_scale True --mu_mse_weight 0.0|15|pwl"
  "E1c|--head_family isqf --isqf_tails True --isqf_space logit --free_scale True --mu_mse_weight 0.0|17|isqf"
  "E1d|--head_family isqf --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0|17|isqf"
  # The six knot/floor arms. Every one carries --free_scale, which as of 2026-09-16 actually
  # REMOVES the scale channel rather than leaving it emitted and unread, so the expected
  # counts below are one lower than the same arms carried before that change.
  "E2i|--head_family pwl --free_scale True --mu_mse_weight 0.0 --spline_knots skew14|15|pwl"
  "E2ii|--head_family pwl --free_scale True --mu_mse_weight 0.0 --spline_knots skew11|12|pwl"
  "E1i|--head_family isqf --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0 --spline_knots skew14|17|isqf"
  "E1ii|--head_family isqf --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0 --spline_knots skew11|14|isqf"
  "E2iv|--head_family pwl --free_scale True --mu_mse_weight 0.0 --spline_knots dense24|25|pwl"
  "E1iv|--head_family isqf --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0 --spline_knots dense24|27|isqf"
  # E1v: E1d's tails and transform on E2a's anchor. Same 17 params as E1d, different family.
  "E1v|--head_family pwl --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0|17|pwl"
)

PASS=(); FAIL=()
for ENTRY in "${ARMS[@]}"; do
  IFS='|' read -r TAG FLAGS WANT_P WANT_FAM <<<"$ENTRY"
  if [ -n "${ONLY:-}" ] && ! grep -qw "$TAG" <<<"$ONLY"; then continue; fi
  echo; echo "=============================================================="
  echo "=== SMOKE ${TAG}   ${FLAGS}"
  echo "=============================================================="
  ROOT="${SMOKE_ROOT}/${TAG}"
  rm -rf "$ROOT"
  if $PY -u scripts/run_hindcast_folds.py \
        --stage train --folds 1 --gpus "${GPUS%%,*}" \
        --region "$REGION" --windows 2000 --fold_mask "$FOLD_MASK" \
        --output_root "$ROOT" --log_dir "$SMOKE_LOGS" --tag "_smoke_${TAG}" \
        --max_epochs 1 --train_chips 8 --val_stride 4096 --num_workers 2 --disable_wandb \
        --extra_train_args "${BASE_ARGS} ${FLAGS} --predict_subsample_blocks ${BLOCKS} --seed 42" \
     && $PY scripts/check_qf_raster.py --pred_dir "${ROOT}/preds" --expect_params "$WANT_P" \
            --log "${SMOKE_LOGS}/hindcast_fold1_smoke_${TAG}.log" --flags "$FLAGS" \
            --expect_family "$WANT_FAM"; then
    PASS+=("$TAG")
  else
    FAIL+=("$TAG")
    echo "!!! SMOKE FAILED: ${TAG}" >&2
  fi
done

echo; echo "=============================================================="
echo "passed (${#PASS[@]}): ${PASS[*]:-none}"
echo "FAILED (${#FAIL[@]}): ${FAIL[*]:-none}"
echo "=============================================================="
[ ${#FAIL[@]} -eq 0 ] || exit 1
