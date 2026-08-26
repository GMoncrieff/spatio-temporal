#!/usr/bin/env bash
# Replicate a slate variant that cleared the bar.
#
# Written as its own script rather than a flag on run_dist_slate.sh because that one may still
# be running: bash reads a script incrementally by byte offset, so inserting lines mid-run makes
# it resume at a stale offset and execute a fragment.
#
#   ./scripts/run_dist_replicate.sh <variant> "<flags>" [seeds...]
set -euo pipefail

source "$(dirname "$0")/dist_base_args.sh"

NAME="${1:?usage: $0 <variant> \"<flags>\" [seeds...]}"
FLAGS="${2:?}"
shift 2
SEEDS="${*:-43 44}"

FOLDS="${FOLDS:-1,2}"
GPUS="${GPUS:-0,1}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
SCORE_DIR="${SCORE_DIR:-data/ensemble/exp/dist_scores}"
FIRST_FOLD="${FOLDS%%,*}"

for SEED in $SEEDS; do
  RUN="${NAME}_s${SEED}"
  if [ -f "${SCORE_DIR}/summary_${RUN}.json" ]; then
    echo "=== ${RUN} already scored, skipping"; continue
  fi
  echo "=============================================================="
  echo "=== ${RUN}: ${FLAGS} --seed ${SEED}"
  echo "=============================================================="
  ./scripts/run_central_experiment.sh "$RUN" "$GPUS" "$FOLDS" "${FLAGS} --seed ${SEED}"
  verify_loss_weights "data/ensemble/logs/hindcast_fold${FIRST_FOLD}_${RUN}.log" "$RUN" "$FLAGS"
  # Prove the lever engaged. A replicate whose flag silently did nothing reads exactly
  # like a lever that has no effect, which is how this project lost a k=5 run once.
  verify_gate_flags "data/ensemble/logs/hindcast_fold${FIRST_FOLD}_${RUN}.log" "$RUN" "$FLAGS"
  $PY -u scripts/score_distributional_model.py \
      --stitched_dir "data/ensemble/exp/${RUN}/stitched" \
      --label "$RUN" --folds "$FOLDS" --out_dir "$SCORE_DIR"
done
echo "=== replicates complete: ${SCORE_DIR}"
