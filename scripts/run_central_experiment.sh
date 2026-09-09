#!/usr/bin/env bash
# One central-field experiment: train the held-out folds, predict southern Africa, stitch.
#
# Prediction is regional, which is what makes this affordable — the global fold run took
# 121 min per fold and almost all of it was writing a 17111x40000 raster four times. The
# fold models themselves are still trained on global chips, so the held-out geography is
# genuinely out of sample; only the extent that gets scored has shrunk.
#
#   ./scripts/run_central_experiment.sh <name> <gpu> <folds> "<extra train args>"
#
# Example:
#   ./scripts/run_central_experiment.sh e1_residual 0 1 "--central_residual True"

set -euo pipefail

NAME="${1:?usage: $0 <name> <gpu> <folds> [extra_train_args]}"
GPU="${2:?}"
FOLDS="${3:?}"
EXTRA="${4:-}"

PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
TRAIN_CHIPS="${TRAIN_CHIPS:-100}"
# Grid validation walks the whole globe, so at the production stride of 512 it is 67 of the
# 80 steps in an epoch — five times the cost of the training it is there to monitor. 2048
# keeps enough chips to rank epochs while making the epoch about training again.
VAL_STRIDE="${VAL_STRIDE:-2048}"
# Each worker's rasterio handles cost ~7 GB resident, so two concurrent experiments at the
# default of 4 workers each put the box into swap.
NUM_WORKERS="${NUM_WORKERS:-3}"
BASE_ARGS="${BASE_ARGS:-}"
# Output root. Defaults to the legacy ensemble path so every existing caller
# (run_dist_*.sh, run_model_slate.sh, promote_model_experiment.sh) is unchanged; the
# conv-spline runners export EXP_ROOT=data/conv_spline/exp and score ${EXP_ROOT}/<name>/
# stitched, so a hardcoded ROOT here writes where nothing reads.
ROOT="${EXP_ROOT:-data/ensemble/exp}/${NAME}"
# Southern Africa by default (CLAUDE.md: all iteration is regional). REGION= overrides it —
# used for the Africa-wide run, whose point is that southern Africa is an unrepresentative
# sample of HM level: the [0,0.01) stratum is 6% of it and 40% of Africa, and that stratum
# carries the defect the class fits are chasing.
REGION="${REGION:-config/region_to_predict_small.geojson}"
# The k-fold mask the run trains, restricts prediction and stitches against. Defaults to the
# production 128 px checkerboard every existing checkpoint was trained on. Stage C of
# docs/background/improvement_plan.md points this at fold_mask_b4_1000.tif (512 px blocks), which puts
# 31.9% of held-out pixels beyond one residual correlation length instead of 0.0% — and
# which makes every earlier checkpoint and scorecard non-comparable, so it is set explicitly
# and never by default.
FOLD_MASK="${FOLD_MASK:-data/raw/hm_global/fold_mask_1000.tif}"

mkdir -p "$ROOT"
# "no extra flags" is NOT the production architecture — it is argparse defaults, and those
# have --central_residual False, under which the central head predicts absolute HM and must
# reconstruct the baseline through the trunk. That costs sd ~0.0075 HM of spurious change on
# pixels that did not move, which is larger than the signal. A k=5 run launched this way
# scored h=5 skill -0.50 against africa_k5's +0.13 and looked exactly like a fold-mask
# finding. Pass the reference set through BASE_ARGS; run_model_slate.sh and
# promote_model_experiment.sh hold it as PHASE_REF.
if [ -z "${BASE_ARGS}${EXTRA}" ]; then
  echo "WARNING: no BASE_ARGS and no extra flags — training with argparse DEFAULTS," >&2
  echo "         which is NOT the shipped configuration (--central_residual defaults False)." >&2
  echo "         The shipped set is:" >&2
  echo "           --central_residual True --central_context True \\" >&2
  echo "           --monotone_quantile_width True --quantile_context True" >&2
  echo "         Set BASE_ARGS to it, or export ALLOW_DEFAULT_ARCH=1 to proceed anyway." >&2
  [ -n "${ALLOW_DEFAULT_ARCH:-}" ] || exit 3
fi
echo "=== ${NAME} | GPU ${GPU} | folds ${FOLDS} | fold_mask $(basename "$FOLD_MASK") ==="
echo "    args: ${BASE_ARGS} ${EXTRA}"

$PY -u scripts/run_hindcast_folds.py \
    --stage train --folds "$FOLDS" --gpus "$GPU" \
    --region "$REGION" --windows all --fold_mask "$FOLD_MASK" \
    --output_root "$ROOT" \
    --log_dir "${LOG_DIR:-data/ensemble/logs}" \
    --tag "_${NAME}" \
    --max_epochs "$MAX_EPOCHS" --train_chips "$TRAIN_CHIPS" \
    --val_stride "$VAL_STRIDE" --num_workers "$NUM_WORKERS" \
    --wandb_group "central-${NAME}" \
    --extra_train_args "${BASE_ARGS} ${EXTRA}"

$PY -u scripts/run_hindcast_folds.py \
    --stage stitch --folds "$FOLDS" \
    --region "$REGION" --windows all --fold_mask "$FOLD_MASK" \
    --output_root "$ROOT" --keep_fold_rasters

echo "=== ${NAME} stitched -> ${ROOT}/stitched ==="
