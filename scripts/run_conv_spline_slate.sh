#!/usr/bin/env bash
# The conv-spline experiment slate. Every entry is one stated delta from b1.
#
# Refuses to run without b1's floor on disk, because a variant ranked against a floor that
# does not exist is the failure the previous phase spent itself on.
#
#   ./scripts/run_conv_spline_slate.sh                 # E0-E5
#   ONLY="E2 E4" ./scripts/run_conv_spline_slate.sh    # a subset
#   SEEDS="42 43" ./scripts/run_conv_spline_slate.sh   # replicate the survivors
#
# E6-E8 are deliberately absent. They are chosen from what E0-E5 measure; writing them now
# would be guessing, and a slate whose last third is guesswork is how twenty-two experiments
# got screened against a blind spot.
set -euo pipefail

source "$(dirname "$0")/conv_spline_base.sh"
guard_region

SEEDS="${SEEDS:-42}"
FIRST_FOLD="${FOLDS%%,*}"
FLOOR_PREFIX="${FLOOR_PREFIX:-b1}"

if ! ls "${SCORE_DIR}"/summary_${FLOOR_PREFIX}_s*.json >/dev/null 2>&1; then
  echo "FATAL: no ${FLOOR_PREFIX} floor in ${SCORE_DIR}." >&2
  echo "       Run ./scripts/run_conv_spline_baseline.sh first. Ranking a variant against" >&2
  echo "       a floor that does not exist is how the last phase manufactured a win." >&2
  exit 1
fi

# name | extra train args | one line on what it tests
#
# E0 is export-only and needs no retraining -- it re-evaluates b1's own checkpoints on a
# different u-grid. It is in the slate so the scorecard row exists beside the others, and
# because its *_ref gate should barely move: that is the control separating "the render
# stopped fencing" from "the distribution stopped fencing".
SLATE=(
  "E0_ugrid|--u_grid_spacing skew|export-only: 8 of the 64 published levels move out of the flat middle"
  "E1_isqf|--head_family isqf|Park 2022, bounded: free first knot + increments, no exponential tails"
  "E2_pwl|--head_family pwl|linear pieces, CRPS in closed form, 16 params/horizon not 29"
  "E3_zcrps|--crps_z_weight 1.0 --crps_z_scale 0.001|a second CRPS term in asinh space, so the core costs something"
  "E4_gapfloor|--head_family pwl --spline_gap_floor True|no segment may imply a density above 578"
  "E5_skew|--spline_knots skew14|the same 14 bins, re-placed for the measured right skew"
)

for ENTRY in "${SLATE[@]}"; do
  IFS='|' read -r TAG FLAGS WHY <<<"$ENTRY"
  if [ -n "${ONLY:-}" ] && ! grep -qw "${TAG%%_*}" <<<"$ONLY"; then continue; fi
  for SEED in $SEEDS; do
    NAME="${TAG}_s${SEED}"
    echo "=============================================================="
    echo "=== ${NAME}"
    echo "=== ${WHY}"
    echo "=== delta from b1: ${FLAGS}"
    echo "=============================================================="
    EXP_ROOT="$EXP_ROOT" REGION="$REGION" FOLD_MASK="$FOLD_MASK" \
      ./scripts/run_central_experiment.sh "$NAME" "$GPUS" "$FOLDS" "${FLAGS} --seed ${SEED}"

    LOG="${LOG_DIR}/hindcast_fold${FIRST_FOLD}_${NAME}.log"
    verify_loss_weights "$LOG" "$NAME" "$FLAGS"
    verify_trunk_context "$LOG" "$NAME"

    $PY -u scripts/score_distributional_model.py \
        --stitched_dir "${EXP_ROOT}/${NAME}/stitched" \
        --label "$NAME" --folds "$FOLDS" --out_dir "$SCORE_DIR" \
        --fold_mask "$FOLD_MASK"
  done
done

echo
$PY scripts/compare_conv_spline_runs.py --score_dir "$SCORE_DIR" --floor_prefix "$FLOOR_PREFIX"
