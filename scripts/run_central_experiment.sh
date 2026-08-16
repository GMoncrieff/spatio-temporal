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
ROOT="data/ensemble/exp/${NAME}"
# Southern Africa by default (CLAUDE.md: all iteration is regional). REGION= overrides it —
# used for the Africa-wide run, whose point is that southern Africa is an unrepresentative
# sample of HM level: the [0,0.01) stratum is 6% of it and 40% of Africa, and that stratum
# carries the defect the class fits are chasing.
REGION="${REGION:-config/region_to_predict_small.geojson}"

mkdir -p "$ROOT"
echo "=== ${NAME} | GPU ${GPU} | folds ${FOLDS} | ${EXTRA:-<production architecture>} ==="

$PY -u scripts/run_hindcast_folds.py \
    --stage train --folds "$FOLDS" --gpus "$GPU" \
    --region "$REGION" --windows all \
    --output_root "$ROOT" \
    --log_dir data/ensemble/logs \
    --tag "_${NAME}" \
    --max_epochs "$MAX_EPOCHS" --train_chips "$TRAIN_CHIPS" \
    --val_stride "$VAL_STRIDE" --num_workers "$NUM_WORKERS" \
    --wandb_group "central-${NAME}" \
    --extra_train_args "${BASE_ARGS} ${EXTRA}"

$PY -u scripts/run_hindcast_folds.py \
    --stage stitch --folds "$FOLDS" \
    --region "$REGION" --windows all \
    --output_root "$ROOT" --keep_fold_rasters

echo "=== ${NAME} stitched -> ${ROOT}/stitched ==="
