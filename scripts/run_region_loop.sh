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
#
# Environment overrides:
#   SHAPE=none|measured|<u_bound>
#                          marginal shape, re-fitted from this run's own residuals.
#                          "none"     the two-piece normal.
#                          "measured" the configuration this phase settled on: pooled body,
#                                     upper tail bound 0.999 out to 100 px and 0.975 beyond,
#                                     lower bound held at 0.025. See docs/next_phase_marginals.md
#                                     section 5.8 for why each of those three is what it is.
#                          <number>   a single symmetric bound, for sweeping.
#   SUFFIX=<tag>           suffix the ensemble/validation outputs, so two marginal
#                          families can sit side by side under one experiment.

set -euo pipefail

NAME="${1:?usage: $0 <experiment-name> [members]}"
MEMBERS="${2:-20}"

PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
GPUS="${GPUS:-0,1}"
ROOT="data/ensemble/exp/${NAME}"
# The region whose fold mask, ecoregion raster and distance covariate the loop scores
# against. Southern Africa by default (CLAUDE.md: all iteration is regional); set
# REGION_ROOT=data/ensemble/region/africa to run the same loop on the Africa extent, whose
# assets are cropped from the global rasters on the identical window.
REGION_ROOT="${REGION_ROOT:-data/ensemble/region/southern_africa}"
# Base year of the prediction window. The hindcast loop scores w2000 -> 2005..2020; a
# forward product would set BASE_YEAR=2020 and YEARS=2025,2030,2035,2040.
BASE_YEAR="${BASE_YEAR:-2000}"
YEARS="${YEARS:-2005,2010,2015,2020}"
BLOCKS="${BLOCKS:-1,10,100}"
PAIRS="${PAIRS:-80000}"
MAXLAG="${MAXLAG:-256}"
DIST_RASTER="${REGION_ROOT}/covariates/w${BASE_YEAR}_dist_past_change.tif"
SHAPE="${SHAPE:-none}"
WIDTHS="${WIDTHS:-none}"
SUFFIX="${SUFFIX:-}"
MEMBERS_ZARR="${ROOT}/members${SUFFIX}.zarr"
NULL_ZARR="${ROOT}/null${SUFFIX}.zarr"
VALIDATION_DIR="${ROOT}/validation${SUFFIX}"
SHAPE_JSON="${ROOT}/marginal_shape${SUFFIX}.json"
WIDTH_JSON="${ROOT}/width_factors${SUFFIX}.json"
# Narrowing writes its own recal and residual dirs so the base pair stays intact for
# comparison; without it, everything downstream reads the base pair.
if [ "$WIDTHS" != "none" ]; then
  RECAL_DIR="${ROOT}/recal_w${SUFFIX}"; RESID_DIR="${ROOT}/residuals_w${SUFFIX}"
else
  RECAL_DIR="${ROOT}/recal"; RESID_DIR="${ROOT}/residuals"
fi
SHAPE_ARGS=()      # for generate_ensemble, which has no --dist_raster of its own
VAL_SHAPE_ARGS=()  # for validate_ensemble, which already passes --dist_raster

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

if [ "$WIDTHS" != "none" ]; then
  echo "=== ${NAME} · Phase 1.5d: per-class half-width factors ==="
  # The marginal shape normalizes to the published bounds (that is what pins T5.2), so it
  # re-injects whatever width error they carry and cannot fix it itself. This is the lever
  # that can. Residuals are rebuilt against the narrowed bounds because the shape is fitted
  # on them next — fitting to one width and generating from another breaks T5.2 silently.
  $PY -u scripts/fit_width_factors.py \
      --manifest "${ROOT}/residuals/manifest.csv" --out "$WIDTH_JSON"
  $PY -u scripts/apply_recalibration.py --targets hindcast \
      --factors "${ROOT}/calibration/scale_factors.csv" \
      --hindcast_dir "${ROOT}/stitched" --hindcast_suffix "" \
      --hindcast_out "$RECAL_DIR" \
      --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
      --dist_raster "$DIST_RASTER" --width_factors "$WIDTH_JSON"
  $PY -u scripts/build_region_residuals.py \
      --pred_dir "$RECAL_DIR" --pred_suffix _recal --keep_splits all \
      --out_dir "$RESID_DIR" --covariate_dir "${REGION_ROOT}/covariates"
fi

echo "=== ${NAME} · Phase 2: fit the field spectrum to *this* model's residuals ==="
$PY -u scripts/fit_field_spectra.py \
    --manifest "${ROOT}/residuals/manifest.csv" \
    --out "${ROOT}/spectral_fits.json"

if [ "$SHAPE" != "none" ]; then
  echo "=== ${NAME} · Phase 2b: fit the marginal shape to *this* model's residuals ==="
  # Re-derived per configuration for the same reason the recalibration and the spectrum
  # are: a shape fitted to another model's residuals describes another model's error.
  if [ "$SHAPE" = "measured" ]; then
    FIT_ARGS=(--by_band true --pooled_body true
              --u_bound 0.999,0.999,0.999,0.999,0.999,0.975 --u_bound_lo 0.025)
  else
    FIT_ARGS=(--u_bound "$SHAPE")
  fi
  $PY -u scripts/fit_marginal_shape.py \
      --manifest "${RESID_DIR}/manifest.csv" \
      --out "$SHAPE_JSON" "${FIT_ARGS[@]}"
  SHAPE_ARGS=(--marginal_shape "$SHAPE_JSON" --dist_raster "$DIST_RASTER")
  VAL_SHAPE_ARGS=(--marginal_shape "$SHAPE_JSON")
fi

echo "=== ${NAME} · Phase 3: ensemble on the recalibrated marginals ==="
$PY -u scripts/generate_ensemble.py \
    --central_dir "$RECAL_DIR" --recal_dir "$RECAL_DIR" \
    --central_pattern "w${BASE_YEAR}_prediction_{year}_central_recal.tif" \
    --recal_pattern "w${BASE_YEAR}_prediction_{year}_{q}_recal.tif" \
    --years "$YEARS" --base_year "$BASE_YEAR" --members "$MEMBERS" \
    --spectral_fits "${ROOT}/spectral_fits.json" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "${ROOT}/residuals/horizon_autocorrelation.json" \
    --wrap_lon False "${SHAPE_ARGS[@]+"${SHAPE_ARGS[@]}"}" \
    --out "$MEMBERS_ZARR" --gpus "$GPUS" \
    --wandb_group "central-${NAME}"

echo "=== ${NAME} · Phase 3: independent-pixel null (same marginals, no spatial structure) ==="
$PY -u scripts/generate_ensemble.py \
    --central_dir "$RECAL_DIR" --recal_dir "$RECAL_DIR" \
    --central_pattern "w${BASE_YEAR}_prediction_{year}_central_recal.tif" \
    --recal_pattern "w${BASE_YEAR}_prediction_{year}_{q}_recal.tif" \
    --years "$YEARS" --base_year "$BASE_YEAR" --members "$MEMBERS" --independent \
    --spectral_fits "${ROOT}/spectral_fits.json" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "${ROOT}/residuals/horizon_autocorrelation.json" \
    --wrap_lon False "${SHAPE_ARGS[@]+"${SHAPE_ARGS[@]}"}" \
    --out "$NULL_ZARR" --gpus "$GPUS" --disable_wandb

echo "=== ${NAME} · Phase 4: T1-T8 scorecard ==="
$PY -u scripts/validate_ensemble.py \
    --ensemble "$MEMBERS_ZARR" --null_ensemble "$NULL_ZARR" \
    --recal_dir "$RECAL_DIR" \
    --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "${ROOT}/residuals/horizon_autocorrelation.json" \
    --recal_manifest "${RECAL_DIR}/recal_manifest.json" \
    --dist_raster "$DIST_RASTER" "${VAL_SHAPE_ARGS[@]+"${VAL_SHAPE_ARGS[@]}"}" \
    --block_sizes "$BLOCKS" --out_dir "$VALIDATION_DIR" \
    --wandb_group "central-${NAME}"

echo
echo "Scorecard: ${VALIDATION_DIR}/scorecard.csv"
echo "Central diagnostics: ${ROOT}/central_diag/"
