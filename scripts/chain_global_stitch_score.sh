#!/usr/bin/env bash
# Steps 2b and 3: holdout-stitch the five folds, prove the mosaic, then score the
# distributional model alone -- no ensemble.
#
# Stitch mode is holdout and never mean. A mean-stitched raster averages all five folds at
# every pixel, so no pixel is out of sample any more; it is right for a forward product and
# wrong for anything scored, and the two are one flag apart.
set -uo pipefail
cd /home/glenn/spatio-temporal
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
NAME="${NAME:-g_e1_hind}"
ROOT="/mnt/hdd1/spatio-temporal/data/ensemble/exp/${NAME}"
FOLD_MASK=data/raw/hm_global/fold_mask_b4_1000.tif
CHAIN_LOG=data/ensemble/logs/g_e1_hind_chain.log

echo "[ss-chain] waiting for the hindcast waves to finish"
until grep -q "wave 3 exited rc=0" "$CHAIN_LOG" 2>/dev/null; do
  if grep -qE "ABORT|wave [23] exited rc=[1-9]" "$CHAIN_LOG" 2>/dev/null; then
    echo "[ss-chain] ABORT: the hindcast chain failed"; exit 3
  fi
  sleep 60
done
echo "[ss-chain] waves done at $(date -Is)"

for f in 1 2 3 4 5; do
  n=$(grep -c "Total time:" "data/ensemble/logs/hindcast_fold${f}_${NAME}.log" 2>/dev/null || echo 0)
  [ "$n" -ge 4 ] || { echo "[ss-chain] ABORT: fold ${f} has ${n}/4 windows"; exit 4; }
done
echo "[ss-chain] all five folds have four windows"

echo "[ss-chain] stitching (holdout) at $(date -Is)"
./scripts/run_global_dist_hindcast.sh stitch 1,2,3,4,5 \
  > data/ensemble/logs/g_e1_hind_stitch.log 2>&1
rc=$?
echo "[ss-chain] stitch exited rc=${rc} at $(date -Is)"
[ "$rc" -eq 0 ] || exit "$rc"

# Prove the mosaic. Each stitched raster must cover essentially every land pixel, because
# the five fold territories partition the grid -- a shortfall means a fold's rasters were
# missing and the stitcher skipped them with a warning nobody read.
$PY - <<'PYEOF' || exit 5
import sys, glob, os
import numpy as np, rasterio
root = "/mnt/hdd1/spatio-temporal/data/ensemble/exp/g_e1_hind/stitched"
paths = sorted(glob.glob(os.path.join(root, "w*_prediction_*_*.tif")))
if len(paths) != 40:
    raise SystemExit(f"FATAL: {len(paths)} stitched rasters, expected 40 (10 window-years x 4)")
bad = 0
for p in paths:
    with rasterio.open(p) as s:
        n = 0
        for _, w in s.block_windows(1):
            a = s.read(1, window=w)
            n += int((a != s.nodata).sum() if s.dtypes[0] == "int16"
                     else np.isfinite(a).sum())
    frac = n / 184_600_000
    flag = "" if frac > 0.95 else "   <<< SHORTFALL"
    if flag: bad += 1
    print(f"  {os.path.basename(p):46s} {n:>12,} px  {frac:5.1%} of land{flag}")
if bad:
    raise SystemExit(f"FATAL: {bad} stitched rasters cover under 95% of land")
print("  OK: 40 rasters, every one covering >95% of land")
PYEOF

echo "[ss-chain] scoring the distributional model at $(date -Is)"
$PY -u scripts/score_distributional_model.py \
    --stitched_dir "${ROOT}/stitched" --label g_e1_global \
    --out_dir "data/ensemble/exp/${NAME}_score" \
    --folds 1,2,3,4,5 --fold_mask "$FOLD_MASK" --row_chunk 512 \
    > data/ensemble/logs/g_e1_hind_score.log 2>&1
rc=$?
echo "[ss-chain] scorecard exited rc=${rc} at $(date -Is)"
[ "$rc" -eq 0 ] && tail -30 data/ensemble/logs/g_e1_hind_score.log
exit "$rc"
