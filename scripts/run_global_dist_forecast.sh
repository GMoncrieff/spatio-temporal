#!/usr/bin/env bash
# Step 4 of the global production phase: retrain e1 on ALL data to 2020, then forecast
# 2025-2040 on the global grid.
#
#   ./scripts/run_global_dist_forecast.sh [gpu]
#
# This is not a fold model and not a mosaic. --train_all_splits trains on every valid chip
# in the split mask instead of the 70% train split; with no held-out geography to protect,
# restricting a forward model to split 1 throws away 30% of the world for nothing. The flag
# is ignored under --exclude_fold, so it cannot pull a held-out fold back into training.
# Validation still runs on split 2 and is in-sample by construction -- unavoidable for a
# production model, and the reason the configuration is validated by k-fold instead.
#
# Three fingerprints are read back before the artifacts are trusted, because a production
# model has no held-out score that would reveal a wrong one:
#   PRODUCTION MODE banner present and FOLD-CV MODE absent -- the flag actually engaged
#   LOSS WEIGHTS all zero and 12 context channels             -- rule 32, the silent inherit
#   Row banding                                                -- 268 accumulators over the
#     full grid with no fold restriction; unbanded this is ~136 GiB resident
set -uo pipefail
cd /home/glenn/spatio-temporal
GPU="${1:-0}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
# CKPT re-predicts from an existing forward checkpoint instead of retraining. The first
# global attempt trained fine and then died writing a classic TIFF at the 4 GiB ceiling;
# there is no reason to spend another 38 min on the same weights.
CKPT="${CKPT:-}"
source scripts/dist_base_args.sh

NAME="${NAME:-g_e1_fc}"
HDD_ROOT="${HDD_ROOT:-/mnt/hdd1/spatio-temporal/data/ensemble/exp/${NAME}}"
LINK="data/ensemble/exp/${NAME}"
REGION="${REGION:-config/region_to_predict_large.geojson}"
ROW_CHUNK="${ROW_CHUNK:-512}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
TRAIN_CHIPS="${TRAIN_CHIPS:-100}"
VAL_STRIDE="${VAL_STRIDE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-3}"
E1_EXTRA="${E1_EXTRA:---context_radii 3,30,100 --hm_context_radii 3,30,100 \
--hm_context_stats mean,max --weight_avg_last 20 --seed 46}"
LOG="data/ensemble/logs/${NAME}.log"

mkdir -p "$HDD_ROOT/preds"
[ -e "$LINK" ] || ln -s "$HDD_ROOT" "$LINK"

echo "=== global forward model + 2025-2040 forecast | ${NAME} | GPU ${GPU} ==="
echo "  region     ${REGION}"
echo "  out        ${HDD_ROOT}/preds"
echo "  row chunk  ${ROW_CHUNK}"
echo "  args       ${BASE_ARGS} ${E1_EXTRA}"

if [ -n "$CKPT" ]; then
  [ -f "$CKPT" ] || { echo "FATAL: CKPT=$CKPT does not exist" >&2; exit 2; }
  echo "  PREDICT-ONLY from ${CKPT}"
  MAX_EPOCHS=0
  E1_EXTRA="${E1_EXTRA/--weight_avg_last 20/}"   # already baked into the checkpoint
  CKPT_ARG=(--checkpoint "$CKPT")
else
  CKPT_ARG=()
fi

CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 $PY -u scripts/train_lightning.py \
    --train_all_splits True "${CKPT_ARG[@]}" \
    --max_epochs "$MAX_EPOCHS" --train_chips "$TRAIN_CHIPS" \
    --val_stride "$VAL_STRIDE" --num_workers "$NUM_WORKERS" \
    --batch_size 8 --devices 1 \
    --norm_stats_json data/ensemble/norm_stats.json \
    --run_full_set_evaluation False --run_large_area_prediction True \
    --predict_region "$REGION" \
    --predict_final_year 2040 \
    --predict_stride 64 --predict_batch_size 32 \
    --predict_row_chunk "$ROW_CHUNK" \
    --predict_output_dir "${HDD_ROOT}/preds" \
    --predict_output_prefix "" \
    --hidden_dim 64 --num_layers 4 --kernel_size 3 \
    --locenc_out_channels 8 --locenc_legendre_polys 10 \
    --histogram_lambda_w2 0.1 --histogram_warmup_epochs 0 \
    ${BASE_ARGS} ${E1_EXTRA} \
    --wandb_group "global-${NAME}" --wandb_run_name "${NAME}" \
    --wandb_tags "global,forecast,e1" > "$LOG" 2>&1
rc=$?
echo "  train+predict exited rc=${rc}"

echo
echo "--- fingerprints ---"
ok=1
if grep -q "PRODUCTION MODE: training on EVERY chip in the split mask" "$LOG"; then
  echo "  ✓ production mode engaged (no geography held out)"
else
  echo "FATAL: --train_all_splits did not engage; this model trained on split 1 only." >&2; ok=0
fi
if grep -q "FOLD-CV MODE" "$LOG"; then
  echo "FATAL: FOLD-CV MODE banner present -- a fold was held out of the forward model." >&2; ok=0
else
  echo "  ✓ no fold held out"
fi
verify_loss_weights     "$LOG" "forecast" "" || ok=0
verify_gate_flags       "$LOG" "forecast" "$E1_EXTRA" || ok=0
verify_context_channels "$LOG" "forecast" "$E1_EXTRA" || ok=0
grep -q "Row banding: " "$LOG" && echo "  ✓ $(grep -m1 -o 'Row banding: .*' "$LOG")" \
  || { echo "FATAL: the forecast did not band its prediction rows" >&2; ok=0; }
# Existence is not completeness. The first attempt left all sixteen rasters on disk,
# truncated at the 4 GiB classic-TIFF ceiling, with every unwritten row reading back as
# finite ZEROS -- so `[ -f ]` passed and the numbers looked plausible. Count the valid
# pixels instead: a complete raster holds ~184.6M of 684.4M, a truncated one reads far more
# because the unwritten remainder is not nodata. And assert the container is BigTIFF.
$PY - "$HDD_ROOT" <<'PYEOF' || ok=0
import sys, numpy as np, rasterio
from pathlib import Path
root = Path(sys.argv[1]) / "preds"
LAND, TOL = 184_600_000, 0.10
bad = 0
for y in (2025, 2030, 2035, 2040):
    for q in ("central", "lower", "upper", "qf"):
        p = root / f"prediction_{y}_{q}_blended.tif"
        if not p.exists():
            print(f"FATAL: missing {p}"); bad += 1; continue
        ver = int(np.frombuffer(open(p, "rb").read(4)[2:4],
                                dtype="<u2" if open(p, "rb").read(2) == b"II" else ">u2")[0])
        with rasterio.open(p) as s:
            n = 0
            for _, w in s.block_windows(1):
                a = s.read(1, window=w)
                n += int((a != s.nodata).sum() if s.dtypes[0] == "int16"
                         else np.isfinite(a).sum())
        frac = n / LAND
        flag = ""
        if not (1 - TOL) <= frac <= (1 + TOL):
            flag = "  <<< INCOMPLETE OR OVERFULL"; bad += 1
        if q == "qf" and ver != 43:
            flag += "  <<< NOT BIGTIFF"; bad += 1
        print(f"  {p.name:44s} v{ver} {n:>12,} px  {frac:5.1%} of land{flag}")
if bad:
    print(f"FATAL: {bad} forecast rasters are not complete"); sys.exit(1)
print("  OK: 16 forecast rasters, every one complete and BigTIFF where needed")
PYEOF
[ "$ok" = 1 ] || { echo "REFUSING: the forward model is not the configuration asked for."; exit 4; }
exit $rc
