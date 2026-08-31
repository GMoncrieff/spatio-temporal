#!/usr/bin/env bash
# Global production hindcast for the distributional model (e1), k=5, out-of-sample.
#
#   ./scripts/run_global_dist_hindcast.sh <stage> <folds>
#     stage  = train | stitch
#     folds  = comma-separated, e.g. "1,2"
#
# This is NOT run_central_experiment.sh with a different REGION. Three things differ and each
# one is load-bearing at global scale:
#
#   1. --output_root is the HDD path, passed directly. Root has ~120 GB and the per-fold
#      rasters plus the stitched hindcast come to ~320 GB.
#   2. --predict_row_chunk. accum_horizons is 268 full-window float32 arrays for the w2000
#      window at 64 quantile levels; on the 17111 x 40000 grid that is 185 GiB resident for
#      ONE fold (measured, not projected) against 125 GB of DRAM, and two folds run at once.
#      Africa's grid is 63.1 Mpx so the same accumulators are ~22 GB there, and the only
#      configuration ever run globally was the 12-accumulator triple head.
#   3. The loss-weight banner is read back out of every fold log before the run is trusted.
#      run_hindcast_folds.py injects --ssim_weight 0.2 --laplacian_weight 0.3
#      --histogram_weight 1.0 into every fold command, and --extra_train_args is appended
#      last, so anything BASE_ARGS does not NAME is silently inherited from the frozen
#      triple-head product.
set -uo pipefail
cd /home/glenn/spatio-temporal

STAGE="${1:?usage: $0 <train|stitch> <folds>}"
FOLDS="${2:?}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"

source scripts/dist_base_args.sh

NAME="${NAME:-g_e1_hind}"
HDD_ROOT="${HDD_ROOT:-/mnt/hdd1/spatio-temporal/data/ensemble/exp/${NAME}}"
LINK="data/ensemble/exp/${NAME}"
REGION="${REGION:-config/region_to_predict_large.geojson}"
FOLD_MASK="${FOLD_MASK:-data/raw/hm_global/fold_mask_b4_1000.tif}"
VAL_STRIDE="${VAL_STRIDE:-1024}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
TRAIN_CHIPS="${TRAIN_CHIPS:-100}"
NUM_WORKERS="${NUM_WORKERS:-3}"
GPUS="${GPUS:-0,1}"
ROW_CHUNK="${ROW_CHUNK:-512}"
# e1 = the promoted configuration: the neighbourhood-HM covariate on top of the spline head.
# BASE_ARGS (from dist_base_args.sh) names the head, the residual/context flags, the
# checkpoint monitor and all three loss weights; this names the rest.
E1_EXTRA="${E1_EXTRA:---context_radii 3,30,100 --hm_context_radii 3,30,100 \
--hm_context_stats mean,max --weight_avg_last 20 --seed 46}"
EXTRA="${BASE_ARGS} ${E1_EXTRA} --predict_row_chunk ${ROW_CHUNK}"

mkdir -p "$HDD_ROOT"
[ -e "$LINK" ] || ln -s "$HDD_ROOT" "$LINK"

echo "=============================================================================="
echo "GLOBAL PRODUCTION HINDCAST — ${NAME} | stage ${STAGE} | folds ${FOLDS}"
echo "=============================================================================="
echo "  region       ${REGION}"
echo "  fold mask    ${FOLD_MASK}"
echo "  output root  ${HDD_ROOT}"
echo "  row chunk    ${ROW_CHUNK}"
echo "  extra args   ${EXTRA}"
df -h / /mnt/hdd1 | sed 's/^/  /'
echo

if [ "$STAGE" = "train" ]; then
  $PY -u scripts/run_hindcast_folds.py \
      --stage train --folds "$FOLDS" --gpus "$GPUS" \
      --region "$REGION" --windows all --fold_mask "$FOLD_MASK" \
      --output_root "$HDD_ROOT" \
      --log_dir data/ensemble/logs \
      --tag "_${NAME}" \
      --max_epochs "$MAX_EPOCHS" --train_chips "$TRAIN_CHIPS" \
      --val_stride "$VAL_STRIDE" --num_workers "$NUM_WORKERS" \
      --wandb_group "global-${NAME}" \
      --extra_train_args "$EXTRA"
  rc=$?
  echo
  echo "--- loss-weight and lever fingerprints, read back from the fold logs ---"
  ok=1
  for f in ${FOLDS//,/ }; do
    log="data/ensemble/logs/hindcast_fold${f}_${NAME}.log"
    [ -f "$log" ] || { echo "FATAL: no log for fold ${f} at ${log}"; ok=0; continue; }
    verify_loss_weights    "$log" "fold${f}" "" || ok=0
    verify_gate_flags      "$log" "fold${f}" "$EXTRA" || ok=0
    verify_context_channels "$log" "fold${f}" "$EXTRA" || ok=0
    grep -q "Row banding: " "$log" \
      && echo "  ✓ fold${f} $(grep -m1 -o 'Row banding: .*' "$log")" \
      || { echo "FATAL: fold${f} did not band its prediction rows" >&2; ok=0; }
  done
  [ "$ok" = 1 ] || { echo "REFUSING to continue: a fold did not train the configuration asked for."; exit 4; }
  exit $rc
elif [ "$STAGE" = "stitch" ]; then
  # holdout, never mean: a mean-stitched raster is in-sample everywhere and must never be
  # scored. The display product is stitched separately.
  $PY -u scripts/run_hindcast_folds.py \
      --stage stitch --stitch_mode holdout --folds "$FOLDS" \
      --region "$REGION" --windows all --fold_mask "$FOLD_MASK" \
      --output_root "$HDD_ROOT" --keep_fold_rasters
  exit $?
else
  echo "unknown stage: $STAGE" >&2; exit 2
fi
