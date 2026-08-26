#!/usr/bin/env bash
# The distributional slate: one run per variant, screened on southern Africa.
#
# Judged against the floor scripts/run_dist_floor.sh measures, the way round 2 of the previous
# phase learned to: **the margin beyond the baseline's own range must exceed that range's
# width.** A single new draw falls outside the range of three base runs about half the time
# under the null, so "outside the range" is barely a test, and the looser bar produced two
# false positives that the stricter one removed. scripts/compare_dist_runs.py applies it.
#
#   ./scripts/run_dist_slate.sh [variant ...]      (default: the whole slate)
set -euo pipefail

source "$(dirname "$0")/dist_base_args.sh"

FOLDS="${FOLDS:-1,2}"
GPUS="${GPUS:-0,1}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
SCORE_DIR="${SCORE_DIR:-data/ensemble/exp/dist_scores}"
FIRST_FOLD="${FOLDS%%,*}"

declare -A SLATE=(
  [d2]="--chip_sampling stratified --chip_sampling_correct True"
  [d2u]="--chip_sampling stratified --chip_sampling_correct False"
  [d1a]="--crps_tail_weight 1.0"
  [d1b]="--crps_tail_weight 4.0"
  [d3]="--ssim_weight 0.2"
  [d4]="--laplacian_weight 0.3"
  [d5]="--ssim_weight 0.2 --laplacian_weight 0.3"
  [d6]="--mu_mse_weight 0.0"
  [d7]="--isolate_shape_grad True"
  [d8]="--spline_slopes fritsch"
  [n1]="--dist_loss nll"
)
# Ordered by the size of the mechanism, because the measured floor decides what can be seen:
# crps_skill at h=5 has a 18% band and pit_ks at h=5 a 5% one, but the h=20 distributional rows
# run 40-89% wide. A variant whose expected effect is a fraction of its metric's band will
# report nothing whichever way it went, so the ones with a real lever go first.
#
# D9 (fewer knots, testing whether the tail resolution is load-bearing) is **not run**: its
# expected effect is small against a band that wide, and a null would mean "under-powered",
# not "no effect". Recorded in docs/dist_model_phase.md rather than silently dropped.
#
# d5 is deliberately absent from the default order: it is run only if d3 and d4 each clear on
# their own. Naming it explicitly runs it anyway.
DEFAULT_ORDER="d2 d6 d1b n1 d3 d4 d2u d7 d1a d8"

for NAME in ${@:-$DEFAULT_ORDER}; do
  FLAGS="${SLATE[$NAME]:-}"
  if [ -z "$FLAGS" ]; then echo "unknown variant $NAME" >&2; exit 2; fi
  if [ -f "${SCORE_DIR}/summary_${NAME}.json" ]; then
    echo "=== ${NAME} already scored, skipping"
    continue
  fi
  echo "=============================================================="
  echo "=== ${NAME}: ${FLAGS}"
  echo "=============================================================="
  ./scripts/run_central_experiment.sh "$NAME" "$GPUS" "$FOLDS" "$FLAGS"
  verify_loss_weights "data/ensemble/logs/hindcast_fold${FIRST_FOLD}_${NAME}.log" "$NAME" "$FLAGS"
  $PY -u scripts/score_distributional_model.py \
      --stitched_dir "data/ensemble/exp/${NAME}/stitched" \
      --label "$NAME" --folds "$FOLDS" --out_dir "$SCORE_DIR"
done
echo "=== slate complete: ${SCORE_DIR}"
