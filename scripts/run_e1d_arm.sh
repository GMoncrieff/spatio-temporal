#!/usr/bin/env bash
# E1d -- E1c with the other transform. One arm, run on its own.
#
# E1c is Park et al. as published: learned tails on a logit-transformed support, free scale.
# E1d changes exactly one flag, --isqf_space logit -> neglog, and nothing else.
#
# Why it exists. The E1a pair measured the two transforms with the scale still injected, and
# they did not tie: far_tail_excess at h=5 read 10.75 for logit against 3.50 for neglog, and
# neglog cut the LOWER tail too (pit_lt_0001 0.00256 against logit's 0.01418). Section 4.1
# predicted the opposite -- it argued the miss is two-sided and near-symmetric, which logit can
# address and neglog, unbounded above only, structurally cannot. On seed 42 that reasoning did
# not hold. E1c inherits logit from the paper rather than from that measurement, so the
# free-scale cell has only ever been run on the transform that lost.
#
# This is one seed against a metric with no b1 floor (far_tail_excess and pit_lt_0001 were
# added to the scorer after the baseline ran), so E1d settles whether the transform ordering
# survives the free-scale arrangement -- not whether either is right.
#
# Separate file rather than a row in run_conv_spline_scale_arms.sh because that script was
# running when this arm was asked for, and bash reads a script incrementally by byte offset.
#
#   ./scripts/run_e1d_arm.sh
set -euo pipefail

source "$(dirname "$0")/conv_spline_base.sh"
guard_region

SEED="${SEED:-42}"
FIRST_FOLD="${FOLDS%%,*}"
NAME="E1d_paper_neglog_s${SEED}"
FLAGS="--head_family isqf --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0"

echo "=============================================================="
echo "=== ${NAME}"
echo "=== Park et al. as published, on the transform the E1a pair favoured"
echo "=== delta from E1c: --isqf_space neglog (was logit)"
echo "=============================================================="

EXP_ROOT="$EXP_ROOT" REGION="$REGION" FOLD_MASK="$FOLD_MASK" \
  ./scripts/run_central_experiment.sh "$NAME" "$GPUS" "$FOLDS" "${FLAGS} --seed ${SEED}"

LOG="${LOG_DIR}/hindcast_fold${FIRST_FOLD}_${NAME}.log"
verify_loss_weights "$LOG" "$NAME" "$FLAGS"
verify_context_wiring "$LOG" "$NAME"
verify_weight_averaging "$LOG" "$NAME"

$PY -u scripts/score_distributional_model.py \
    --stitched_dir "${EXP_ROOT}/${NAME}/stitched" \
    --label "$NAME" --folds "$FOLDS" --out_dir "$SCORE_DIR" \
    --fold_mask "$FOLD_MASK" --row_chunk "$SCORE_ROW_CHUNK"
