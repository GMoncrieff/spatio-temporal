#!/usr/bin/env bash
# Phase 1 -> 4 for one experiment's stitched hindcast, on southern Africa.
#
# Everything downstream of the model is re-derived per experiment, not reused: the
# recalibration factors, the fitted field spectrum and the AR(1) coupling are all
# estimated from *this* model's residuals. Carrying any of them over from another
# configuration would score a new central field through an old model's error structure,
# which is the class of mistake that has produced the most convincing wrong numbers in
# this project.
#
#   ./scripts/run_region_loop.sh <experiment-name> [members]
#
# Expects data/ensemble/exp/<name>/stitched/w{base}_prediction_{year}_{q}.tif, which
# scripts/run_central_experiment.sh produces.

set -euo pipefail

NAME="${1:?usage: $0 <experiment-name> [members]}"
MEMBERS="${2:-20}"

PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
GPUS="${GPUS:-0,1}"
ROOT="data/ensemble/exp/${NAME}"
REGION_ROOT="data/ensemble/region/southern_africa"
BLOCKS="${BLOCKS:-1,10,100}"
PAIRS="${PAIRS:-80000}"
MAXLAG="${MAXLAG:-256}"
DIST_RASTER="${REGION_ROOT}/covariates/w2000_dist_past_change.tif"

test -d "${ROOT}/stitched" || { echo "no stitched rasters under ${ROOT}" >&2; exit 2; }

echo "=== ${NAME} · central-field diagnostics (pre-ensemble, cheapest signal) ==="
$PY -u scripts/diagnose_central_field.py --label "$NAME" \
    --stitched_dir "${ROOT}/stitched" --out_dir "${ROOT}/central_diag" \
    --wandb_group "central-${NAME}"

echo "=== ${NAME} · Phase 0 residuals from the fold-restricted rasters ==="
# The stitched rasters already carry only held-out fold pixels, so no further split
# masking is needed or wanted.
$PY -u scripts/build_region_residuals.py \
    --pred_dir "${ROOT}/stitched" --pred_suffix "" --keep_splits all \
    --out_dir "${ROOT}/residuals" \
    --covariate_dir "${REGION_ROOT}/covariates"

echo "=== ${NAME} · Phase 1 + 1.5: coverage, variograms, class audit, recal decision ==="
$PY -u scripts/run_diagnostics.py \
    --manifest "${ROOT}/residuals/manifest.csv" --out_dir "${ROOT}" \
    --fold_mask "${REGION_ROOT}/fold_mask.tif" \
    --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
    --block_sizes "$BLOCKS" --n_pairs "$PAIRS" --max_lag_px "$MAXLAG" \
    --wandb_group "central-${NAME}"

DECISION=$($PY -c "import json;print(json.load(open('${ROOT}/calibration/recalibration_decision.json'))['decision'])")
echo "    recalibration decision: ${DECISION}"

echo "=== ${NAME} · Phase 1.5c: apply the rescale (central rasters copied, never regenerated) ==="
$PY -u scripts/apply_recalibration.py --targets hindcast \
    --factors "${ROOT}/calibration/scale_factors.csv" \
    --hindcast_dir "${ROOT}/stitched" --hindcast_suffix "" \
    --hindcast_out "${ROOT}/recal" \
    --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
    --dist_raster "$DIST_RASTER"

echo "=== ${NAME} · Phase 2: fit the field spectrum to *this* model's residuals ==="
$PY -u scripts/fit_field_spectra.py \
    --manifest "${ROOT}/residuals/manifest.csv" \
    --out "${ROOT}/spectral_fits.json"

echo "=== ${NAME} · Phase 3: ensemble on the recalibrated marginals ==="
$PY -u scripts/generate_ensemble.py \
    --central_dir "${ROOT}/recal" --recal_dir "${ROOT}/recal" \
    --central_pattern 'w2000_prediction_{year}_central_recal.tif' \
    --recal_pattern 'w2000_prediction_{year}_{q}_recal.tif' \
    --years 2005,2010,2015,2020 --base_year 2000 --members "$MEMBERS" \
    --spectral_fits "${ROOT}/spectral_fits.json" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "${ROOT}/residuals/horizon_autocorrelation.json" \
    --wrap_lon False \
    --out "${ROOT}/members.zarr" --gpus "$GPUS" \
    --wandb_group "central-${NAME}"

echo "=== ${NAME} · Phase 3: independent-pixel null (same marginals, no spatial structure) ==="
$PY -u scripts/generate_ensemble.py \
    --central_dir "${ROOT}/recal" --recal_dir "${ROOT}/recal" \
    --central_pattern 'w2000_prediction_{year}_central_recal.tif' \
    --recal_pattern 'w2000_prediction_{year}_{q}_recal.tif' \
    --years 2005,2010,2015,2020 --base_year 2000 --members "$MEMBERS" --independent \
    --spectral_fits "${ROOT}/spectral_fits.json" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "${ROOT}/residuals/horizon_autocorrelation.json" \
    --wrap_lon False \
    --out "${ROOT}/null.zarr" --gpus "$GPUS" --disable_wandb

echo "=== ${NAME} · Phase 4: T1-T8 scorecard ==="
$PY -u scripts/validate_ensemble.py \
    --ensemble "${ROOT}/members.zarr" --null_ensemble "${ROOT}/null.zarr" \
    --recal_dir "${ROOT}/recal" \
    --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "${ROOT}/residuals/horizon_autocorrelation.json" \
    --recal_manifest "${ROOT}/recal/recal_manifest.json" \
    --dist_raster "$DIST_RASTER" \
    --block_sizes "$BLOCKS" --out_dir "${ROOT}/validation" \
    --wandb_group "central-${NAME}"

echo
echo "Scorecard: ${ROOT}/validation/scorecard.csv"
echo "Central diagnostics: ${ROOT}/central_diag/"
