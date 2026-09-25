#!/usr/bin/env bash
# Phase 2 of docs/superpowers/specs/2026-08-25-distributional-round-2-design.md.
#
# Runs only after the Phase 1 gate has been read and its winner adopted. Set GATE_FLAGS to the
# winning stability configuration's flags and SEEDS to what its measured spread supports — the
# spec's rule is two seeds if the gate brought tail_reach20 spread below ~1.0, three otherwise.
#
#   GATE_FLAGS="--weight_avg_last 20" SEEDS="42 43" ./scripts/run_dist_phase2.sh
#
# Deliberately refuses to start without GATE_FLAGS. Running Phase 2 on the pre-gate recipe is
# the mistake the gate exists to prevent, and "I forgot to set it" looks exactly like "the gate
# did not help".
set -euo pipefail

source "$(dirname "$0")/dist_base_args.sh"

if [ -z "${GATE_FLAGS+x}" ]; then
  echo "FATAL: set GATE_FLAGS to the adopted stability configuration (\"\" if the gate found" >&2
  echo "       nothing, which is a deliberate choice and not a default)." >&2
  exit 3
fi
export BASE_ARGS="${BASE_ARGS} ${GATE_FLAGS}"

FOLDS="${FOLDS:-1,2}"
GPUS="${GPUS:-0,1}"
SEEDS="${SEEDS:-42 43 44}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
SCORE_DIR="${SCORE_DIR:-data/ensemble/exp/dist_scores}"
FIRST_FOLD="${FOLDS%%,*}"

HM_CTX="--context_radii 3,30,100 --hm_context_radii 3,30,100"
declare -A SLATE=(
  # the covariate: the model has never had any information about the LEVEL of development
  # around a pixel, only where past change happened
  [e1]="${HM_CTX} --hm_context_stats mean,max"
  [e1a]="${HM_CTX} --hm_context_stats mean"
  # the knot grid: three knots between u=0.10 and u=0.90 while 53% of pixels move by <0.001
  [e2]="--spline_knots body_dense"
  [e3]="--spline_knots deep_lower"
  # P(u<0.001) reads 7x nominal at h=20: the model is blindsided by declines
  [e4]="--crps_tail_weight_lo 1.0"
  # h=20 receives a quarter of h=5's gradient; cov50 diverges across horizons
  [e5]="--horizon_loss_weights 1,1.33,2,4"
  [e6]="--shape_head_hidden_layers 2"
  [e7]="--shape_head_width 64"
)
# Ordered by the size of the mechanism, because the floor decides what can be seen.
# e5 (both spatial terms) from round 1 is not repeated: SSIM was measured clearly worse.
ORDER="${*:-e1 e2 e1a e5 e4 e3 e6 e7}"

for NAME in $ORDER; do
  FLAGS="${SLATE[$NAME]:-}"
  if [ -z "$FLAGS" ]; then echo "unknown variant $NAME" >&2; exit 2; fi
  for SEED in $SEEDS; do
    # PREFIX keeps two regions' slates apart. Without it an Africa run named e1_s45 finds a
    # SOUTHERN AFRICA summary of the same name, skips itself, and the comparison silently mixes
    # regions -- or worse, overwrites the other region's result where the seeds overlap.
    RUN="${PREFIX:-}${NAME}_s${SEED}"
    if [ -f "${SCORE_DIR}/summary_${RUN}.json" ]; then
      echo "=== ${RUN} already scored, skipping"; continue
    fi
    echo "=============================================================="
    echo "=== ${RUN}: ${FLAGS} --seed ${SEED}   (gate: ${GATE_FLAGS:-none})"
    echo "=============================================================="
    ./scripts/run_central_experiment.sh "$RUN" "$GPUS" "$FOLDS" "${FLAGS} --seed ${SEED}"
    LOG="data/ensemble/logs/hindcast_fold${FIRST_FOLD}_${RUN}.log"
    verify_loss_weights "$LOG" "$RUN" "$FLAGS"
    verify_gate_flags "$LOG" "$RUN" "$GATE_FLAGS"
    verify_context_channels "$LOG" "$RUN" "$FLAGS"
    $PY -u scripts/score_distributional_model.py \
        --stitched_dir "data/ensemble/exp/${RUN}/stitched" \
        --label "$RUN" --folds "$FOLDS" --out_dir "$SCORE_DIR" \
      --fold_mask "$FOLD_MASK"
  done
done
echo "=== phase 2 complete: ${SCORE_DIR}"
