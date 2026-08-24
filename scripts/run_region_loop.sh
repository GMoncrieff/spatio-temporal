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
#                                     lower bound held at 0.025. See docs/background/next_phase_marginals.md
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
# Longitude wrap. generate_ensemble.py auto-detects it from the grid (W >= 39000 and the
# transform starting at -180), which is exactly the global raster; every regional extent is a
# crop and does not wrap. False stays the default so regional runs are unchanged, but a global
# run must set WRAP=True or the field is generated with a discontinuity at the antimeridian and
# T3.5's seam gate scores a seam the sampler was told not to close.
WRAP="${WRAP:-False}"
# Peak validator memory follows this knob (validate_ensemble.py sizes every streaming stage
# from it; T5 spends it at a quarter of face value by design). The script's own default is
# 8 GB, which is right for southern Africa and leaves a 125 GB box idle on the global grid.
MEM_BUDGET_GB="${MEM_BUDGET_GB:-}"
# The icechunk stores are the largest artifacts in the project (~450 GB each at M=400 on the
# global grid), so they need to be addressable independently of ROOT, which lives on /.
MEMBERS_OUT="${MEMBERS_OUT:-}"
NULL_OUT="${NULL_OUT:-}"
# Ensembles are icechunk repositories (see generate_ensemble.py): the write is one
# transaction, so a run killed part-way leaves no store rather than a plausible-looking
# directory of sentinel.
MEMBERS_ZARR="${MEMBERS_OUT:-${ROOT}/members${SUFFIX}.icechunk}"
NULL_ZARR="${NULL_OUT:-${ROOT}/null${SUFFIX}.icechunk}"
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

# The AR(1) coupling between horizons, measured on this model's own standardized residuals.
# build_region_residuals.py does not write it -- only run_hindcast_folds.py --stage residuals
# did, which the regional loop never calls. Every regional run so far therefore fell back to
# rho = 0.9 inside generate_ensemble.py and left T4.1 reported-only rather than scored,
# including the Africa 111/133 card. Deriving it here makes the loop self-contained; the
# consequence is that T4.1 becomes a scored row, so the pass denominator moves.
$PY -u -c "
import json, sys
sys.path.insert(0, '.')
from src.ensemble.residuals import horizon_autocorrelation
rho = horizon_autocorrelation('${ROOT}/residuals/manifest.csv')
json.dump({str(k): v for k, v in rho.items()}, open('${ROOT}/residuals/horizon_autocorrelation.json', 'w'), indent=2)
print('  horizon autocorrelation rho =', rho)
"

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
    --wrap_lon "$WRAP" "${SHAPE_ARGS[@]+"${SHAPE_ARGS[@]}"}" \
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
    --wrap_lon "$WRAP" "${SHAPE_ARGS[@]+"${SHAPE_ARGS[@]}"}" \
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
    ${MEM_BUDGET_GB:+--mem_budget_gb "$MEM_BUDGET_GB" --mem_trace} \
    --wandb_group "central-${NAME}"

echo
echo "Scorecard: ${VALIDATION_DIR}/scorecard.csv"
echo "Central diagnostics: ${ROOT}/central_diag/"
