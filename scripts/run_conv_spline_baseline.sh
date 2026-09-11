#!/usr/bin/env bash
# Step 1 of the conv-spline phase: establish b1 -- e1 plus trunk context -- and its own floor.
#
# Three seeds, not one. The previous phase judged ten variants against a floor measured from
# three replicates that all landed in one mode, read the band as 0.86, and adopted nothing
# when a variant's replicates spread by 4.3. A floor is a property of *this* configuration
# and it is measured before anything is ranked against it.
#
# Nothing here is comparable to a dist-convlstm number. The trunk change alone makes b1 a
# different model, and the scorecard has two gates e1 never had.
#
#   ./scripts/run_conv_spline_baseline.sh              # b1, three seeds, Africa
#   SEEDS="42" ./scripts/run_conv_spline_baseline.sh   # one seed, for a quick check
set -euo pipefail

source "$(dirname "$0")/conv_spline_base.sh"
guard_region

SEEDS="${SEEDS:-42 43 44}"
PREFIX="${PREFIX:-b1}"
FIRST_FOLD="${FOLDS%%,*}"

for SEED in $SEEDS; do
  NAME="${PREFIX}_s${SEED}"
  echo "=============================================================="
  echo "=== ${NAME}  folds ${FOLDS}  seed ${SEED}  region $(basename "$REGION")"
  echo "=============================================================="
  EXP_ROOT="$EXP_ROOT" REGION="$REGION" FOLD_MASK="$FOLD_MASK" \
    ./scripts/run_central_experiment.sh "$NAME" "$GPUS" "$FOLDS" "--seed ${SEED}"

  LOG="${LOG_DIR}/hindcast_fold${FIRST_FOLD}_${NAME}.log"
  verify_loss_weights "$LOG" "$NAME"
  verify_context_wiring "$LOG" "$NAME"

  $PY -u scripts/score_distributional_model.py \
      --stitched_dir "${EXP_ROOT}/${NAME}/stitched" \
      --label "$NAME" --folds "$FOLDS" --out_dir "$SCORE_DIR" \
      --fold_mask "$FOLD_MASK"
done

echo
echo "=== b1 established. Its floor is the spread across ${SEEDS}, not any single run."
$PY scripts/compare_conv_spline_runs.py --score_dir "$SCORE_DIR" --floor_prefix "$PREFIX"
