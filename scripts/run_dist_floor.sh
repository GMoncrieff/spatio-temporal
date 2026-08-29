#!/usr/bin/env bash
# Three replicates of the distributional baseline, to measure this configuration's own
# run-to-run floor before anything is compared against it.
#
# The previous model phase judged 22 experiments against a floor borrowed from a *different*
# configuration and manufactured a win that replication removed. A floor is a property of the
# configuration, not of the harness, so it is measured here first and nothing is ranked until
# it exists -- and the baseline is defined in one place, scripts/dist_base_args.sh, so it
# cannot drift away from the variants that are deltas from it.
set -euo pipefail

source "$(dirname "$0")/dist_base_args.sh"

FOLDS="${FOLDS:-1,2}"
GPUS="${GPUS:-0,1}"
SEEDS="${SEEDS:-42 43 44}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
SCORE_DIR="${SCORE_DIR:-data/ensemble/exp/dist_scores}"
FIRST_FOLD="${FOLDS%%,*}"
# Round 1's floor lives under d0_*, and it was measured with the cdf gradient bug present, so
# it stays on disk as the before-side of that comparison. Override to re-measure the floor
# without clobbering it.
PREFIX="${PREFIX:-d0}"

for SEED in $SEEDS; do
  NAME="${PREFIX}_s${SEED}"
  echo "=============================================================="
  echo "=== ${NAME}  folds ${FOLDS}  seed ${SEED}"
  echo "=============================================================="
  ./scripts/run_central_experiment.sh "$NAME" "$GPUS" "$FOLDS" "--seed ${SEED}"
  verify_loss_weights "data/ensemble/logs/hindcast_fold${FIRST_FOLD}_${NAME}.log" "$NAME"
  $PY -u scripts/score_distributional_model.py \
      --stitched_dir "data/ensemble/exp/${NAME}/stitched" \
      --label "$NAME" --folds "$FOLDS" --out_dir "$SCORE_DIR" \
      --fold_mask "$FOLD_MASK"
done
echo "=== floor complete: ${SCORE_DIR}/summary_${PREFIX}_s*.json"
