#!/usr/bin/env bash
# Phase 1: the stability gate.
#
# Round 1 adopted nothing because `tail_reach20` varies by 4.3 across replicates of one
# configuration, which made six of ten variants undecidable. No screening protocol can resolve
# a knob whose effect is smaller than that, so this runs before any science.
#
# Judged ONLY on the spread across seeds, never on the level. A configuration is adopted if its
# tail_reach20 spread is materially smaller than the baseline's while its median crps_skill5,
# pit_ks5 and rmse5 are no worse.
#
# NOT tested: checkpoint selection. Round 1 measured the correlation between selected epoch and
# tail_reach20 at 0.150, with counterexamples both ways. That hypothesis is already rejected.
set -euo pipefail

source "$(dirname "$0")/dist_base_args.sh"

FOLDS="${FOLDS:-1,2}"
GPUS="${GPUS:-0,1}"
SEEDS="${SEEDS:-42 43 44}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
SCORE_DIR="${SCORE_DIR:-data/ensemble/exp/dist_scores}"
FIRST_FOLD="${FOLDS%%,*}"

# s0 is the control that makes s1 interpretable. Weight averaging rewrites the weights in
# on_train_end, and train_lightning.py then repoints prediction at the end-of-training
# checkpoint — so --checkpoint_monitor val_crps stops mattering under s1 and s3, and those
# configurations change TWO things at once: they average, AND they stop selecting an epoch.
# s0 stops selecting without averaging, so the two mechanisms can be told apart. Rule 10.
declare -A GATE=(
  [s0]="--checkpoint_select final"
  [s1]="--weight_avg_last 20"
  [s2]="--lr_schedule cosine"
  [s3]="--weight_avg_last 20 --lr_schedule cosine"
)
ORDER="${*:-s0 s1 s2 s3}"

for NAME in $ORDER; do
  FLAGS="${GATE[$NAME]:-}"
  if [ -z "$FLAGS" ]; then echo "unknown gate config $NAME" >&2; exit 2; fi
  for SEED in $SEEDS; do
    RUN="${NAME}_s${SEED}"
    if [ -f "${SCORE_DIR}/summary_${RUN}.json" ]; then
      echo "=== ${RUN} already scored, skipping"; continue
    fi
    echo "=============================================================="
    echo "=== ${RUN}: ${FLAGS} --seed ${SEED}"
    echo "=============================================================="
    ./scripts/run_central_experiment.sh "$RUN" "$GPUS" "$FOLDS" "${FLAGS} --seed ${SEED}"
    LOG="data/ensemble/logs/hindcast_fold${FIRST_FOLD}_${RUN}.log"
    verify_loss_weights "$LOG" "$RUN"
    verify_gate_flags "$LOG" "$RUN" "$FLAGS"
    $PY -u scripts/score_distributional_model.py \
        --stitched_dir "data/ensemble/exp/${RUN}/stitched" \
        --label "$RUN" --folds "$FOLDS" --out_dir "$SCORE_DIR"
  done
done
echo "=== gate complete: ${SCORE_DIR}/summary_s*.json"
