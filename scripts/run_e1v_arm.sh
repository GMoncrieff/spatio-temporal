#!/usr/bin/env bash
# E1v -- E1d's tails and transform on E2a's anchor. One arm, run on its own.
#
# WHY. Once --free_scale removed the anchor/scale factorisation, pwl and isqf stopped being
# two parameterisations and became one ladder differing by a constant shift. Measured on
# identical channels: both give width95 = 0.02618 exactly, but pwl pins the free channel to
# Q(0.5) while isqf pins it to Q(0.0), so with the zero-init persistence skip E2a starts
# centred on persistence and 100% of E1d's pixels start with their median ABOVE it, by about
# half the ladder width. Section 4.1 called that out -- "adding persistence to the median is a
# defensible prior, and adding it to the bottom of a ladder is not" -- and kept it anyway, to
# stay faithful to Park et al.
#
# E1d vs E2a therefore confounds three things: the tails, the transform, and the anchor.
# E1v holds the first two fixed and moves only the third, so
#
#   E1v vs E1d  =  the anchor alone      (Q(0.5) against Q(0.0))
#   E1v vs E2a  =  the tails alone       (the same A/B E1 vs E1a is on the other family)
#
# WHAT HAD TO CHANGE. --isqf_tails refused head_family=pwl, correctly: the tail machinery
# lived on ISQFQuantile, so the two channels would have been allocated and never read -- the
# inert-flag failure this project keeps meeting. Nothing in it is ISQF-specific (it reads
# q_knots, u_knots and the last two channels), so it moved to _PWLBase and both families now
# allocate AND read them. Pinned by tests/test_free_scale_channel.py, including that the
# interior is bit-identical to E2a so the tails are appended rather than carved.
#
# Separate file rather than a row in run_conv_spline_knot_arms.sh because that script was
# running when this arm was asked for, and bash reads a script incrementally by byte offset.
#
#   ./scripts/run_e1v_arm.sh
set -euo pipefail

source "$(dirname "$0")/conv_spline_base.sh"
guard_region

SEED="${SEED:-42}"
FIRST_FOLD="${FOLDS%%,*}"
NAME="E1v_pwl_neglog_s${SEED}"
FLAGS="--head_family pwl --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0"

echo "=============================================================="
echo "=== ${NAME}"
echo "=== E1d's tails and transform on E2a's anchor: 17 params, family pwl"
echo "=== delta from E1d: --head_family pwl (was isqf)"
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
