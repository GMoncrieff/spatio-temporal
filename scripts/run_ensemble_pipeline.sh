#!/usr/bin/env bash
# End-to-end driver for the spatiotemporal residual ensemble (docs/ensemble_uncertainty_plan.md).
#
# Phase 0 (fold retraining + hindcast prediction) is the long pole and is run separately by
# scripts/run_hindcast_folds.py; everything downstream of it is here.
#
#   ./scripts/run_ensemble_pipeline.sh small     # southern-Africa validation pass
#   ./scripts/run_ensemble_pipeline.sh global    # full run, after Phase 0 finishes
#
# Two ordering constraints are not negotiable, both because the copula preserves whatever
# marginals it is handed: the class-conditional audit must complete before the
# recalibration decision, and that decision must be final before the ensemble is generated.

set -euo pipefail

SCALE="${1:-global}"
PY="${PY:-python}"
GPUS="${GPUS:-0,1}"

case "$SCALE" in
  small)  MEMBERS=20; BLOCKS="1,10,100"; PAIRS=80000;  MAXLAG=256  ;;
  global) MEMBERS=50; BLOCKS="10,100,1000"; PAIRS=300000; MAXLAG=1024 ;;
  *) echo "usage: $0 [small|global]" >&2; exit 2 ;;
esac

echo "=== Phase 1 + 1.5: diagnostics, variograms, class audit, recalibration decision ==="
$PY -u scripts/run_diagnostics.py \
    --block_sizes "$BLOCKS" --n_pairs "$PAIRS" --max_lag_px "$MAXLAG" \
    --wandb_group "ensemble-${SCALE}"

DECISION=$($PY -c "import json;print(json.load(open('data/ensemble/calibration/recalibration_decision.json'))['decision'])")
echo "    recalibration decision: ${DECISION}"

echo "=== Phase 1.5c: apply the rescale (central rasters are copied, never regenerated) ==="
$PY -u scripts/apply_recalibration.py --targets both

echo "=== Phase 3: hindcast ensemble on the recalibrated marginals ==="
$PY -u scripts/generate_ensemble.py \
    --central_dir data/ensemble/hindcast/recal --recal_dir data/ensemble/hindcast/recal \
    --central_pattern 'w2000_prediction_{year}_central_recal.tif' \
    --recal_pattern 'w2000_prediction_{year}_{q}_recal.tif' \
    --years 2005,2010,2015,2020 --base_year 2000 --members "$MEMBERS" \
    --out data/ensemble/hindcast_members.zarr --gpus "$GPUS" \
    --wandb_group "ensemble-${SCALE}"

echo "=== Phase 3: independent-pixel null (same marginals, no spatial structure) ==="
$PY -u scripts/generate_ensemble.py \
    --central_dir data/ensemble/hindcast/recal --recal_dir data/ensemble/hindcast/recal \
    --central_pattern 'w2000_prediction_{year}_central_recal.tif' \
    --recal_pattern 'w2000_prediction_{year}_{q}_recal.tif' \
    --years 2005,2010,2015,2020 --base_year 2000 --members "$MEMBERS" --independent \
    --out data/ensemble/hindcast_members_null.zarr --gpus "$GPUS" --disable_wandb

echo "=== Phase 4: scorecard ==="
$PY -u scripts/validate_ensemble.py \
    --ensemble data/ensemble/hindcast_members.zarr \
    --null_ensemble data/ensemble/hindcast_members_null.zarr \
    --block_sizes "$BLOCKS" --wandb_group "ensemble-${SCALE}"

if [ "$SCALE" = "global" ]; then
  echo "=== Phase 3: production 2025-2040 ensemble (recalibrated marginals) ==="
  $PY -u scripts/generate_ensemble.py --members "$MEMBERS" --gpus "$GPUS" \
      --years 2025,2030,2035,2040 --base_year 2020 \
      --out data/ensemble/members.zarr --wandb_group "ensemble-${SCALE}"
fi

echo
echo "Scorecard: data/ensemble/validation/scorecard.csv"
