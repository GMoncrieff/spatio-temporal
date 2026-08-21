#!/usr/bin/env bash
# Every validator stage, once, on a four-member ensemble.
#
#   ./scripts/run_smoke_tier.sh <experiment-name>
#
# The point is coverage of the *code paths*, not of the model: M=4 cannot pass a coverage
# target and is not asked to. What it can do is execute every stage against the real grid,
# the real rasters and all four block scales in a few minutes, which is what the three
# defects that cost the last round needed to surface --
#
#   * a rho estimator that turned its sample budget into seven 512 px blocks and drew them
#     uniformly over a grid that is 73% invalid;
#   * an FFT in the field sampler that outgrew a 24 GB card at global width;
#   * a hard gate scored on a NaN, because the antimeridian holds no valid pixels.
#
# All three are shape-and-scale bugs. None needed 400 members to appear, and each of them
# was found instead by a run that had already spent hours.
#
# Nothing is fitted here. The recalibration, spectrum, marginal shape and AR(1) coupling
# are read from the experiment as they stand -- a smoke tier that re-derives them is
# testing a different chain from the one it is protecting.
#
# Environment overrides mirror run_region_loop.sh: REGION_ROOT, BASE_YEAR, YEARS, BLOCKS,
# WRAP, MEM_BUDGET_GB, MEMBERS, MEMBERS_OUT, NULL_OUT, SUFFIX, RECAL_SUBDIR.

set -euo pipefail

NAME="${1:?usage: $0 <experiment-name>}"

PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
GPUS="${GPUS:-0,1}"
ROOT="data/ensemble/exp/${NAME}"
REGION_ROOT="${REGION_ROOT:-data/ensemble/region/africa}"
BASE_YEAR="${BASE_YEAR:-2000}"
YEARS="${YEARS:-2005,2010,2015,2020}"
# All four scales, always: the 1000 km rows exist on no regional card, so the only place
# their code path runs before a global scorecard is here.
BLOCKS="${BLOCKS:-1,10,100,1000}"
MEMBERS="${MEMBERS:-4}"
WRAP="${WRAP:-False}"
MEM_BUDGET_GB="${MEM_BUDGET_GB:-20}"
SUFFIX="${SUFFIX:-_smoke}"
RECAL_SUBDIR="${RECAL_SUBDIR:-recal_w}"

RECAL_DIR="${ROOT}/${RECAL_SUBDIR}"
DIST_RASTER="${REGION_ROOT}/covariates/w${BASE_YEAR}_dist_past_change.tif"
MEMBERS_ZARR="${MEMBERS_OUT:-${ROOT}/members${SUFFIX}.icechunk}"
NULL_ZARR="${NULL_OUT:-${ROOT}/null${SUFFIX}.icechunk}"
VALIDATION_DIR="${ROOT}/validation${SUFFIX}"
SHAPE_JSON="${ROOT}/marginal_shape.json"
RHO_JSON="${ROOT}/residuals/horizon_autocorrelation.json"

test -d "$RECAL_DIR" || { echo "no recalibrated rasters under ${RECAL_DIR}" >&2; exit 2; }

SHAPE_ARGS=(); VAL_SHAPE_ARGS=()
if [ -f "$SHAPE_JSON" ]; then
  SHAPE_ARGS=(--marginal_shape "$SHAPE_JSON" --dist_raster "$DIST_RASTER")
  VAL_SHAPE_ARGS=(--marginal_shape "$SHAPE_JSON")
fi

echo "=== smoke · ${NAME} · M=${MEMBERS} members + null on ${REGION_ROOT} ==="
for OUT in "$MEMBERS_ZARR" "$NULL_ZARR"; do
  EXTRA=()
  [ "$OUT" = "$NULL_ZARR" ] && EXTRA=(--independent)
  $PY -u scripts/generate_ensemble.py \
      --central_dir "$RECAL_DIR" --recal_dir "$RECAL_DIR" \
      --central_pattern "w${BASE_YEAR}_prediction_{year}_central_recal.tif" \
      --recal_pattern "w${BASE_YEAR}_prediction_{year}_{q}_recal.tif" \
      --years "$YEARS" --base_year "$BASE_YEAR" --members "$MEMBERS" \
      --spectral_fits "${ROOT}/spectral_fits.json" \
      --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
      --rho_json "$RHO_JSON" \
      --wrap_lon "$WRAP" "${SHAPE_ARGS[@]+"${SHAPE_ARGS[@]}"}" \
      --out "$OUT" --gpus "$GPUS" --disable_wandb "${EXTRA[@]+"${EXTRA[@]}"}"
done

echo "=== smoke · ${NAME} · every stage, all four block scales ==="
$PY -u scripts/validate_ensemble.py \
    --ensemble "$MEMBERS_ZARR" --null_ensemble "$NULL_ZARR" \
    --recal_dir "$RECAL_DIR" \
    --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "$RHO_JSON" \
    --recal_manifest "${RECAL_DIR}/recal_manifest.json" \
    --dist_raster "$DIST_RASTER" "${VAL_SHAPE_ARGS[@]+"${VAL_SHAPE_ARGS[@]}"}" \
    --block_sizes "$BLOCKS" --out_dir "$VALIDATION_DIR" \
    --mem_budget_gb "$MEM_BUDGET_GB" --mem_trace \
    --wandb_group "smoke-${NAME}"

echo "=== smoke · ${NAME} · checking the card is complete and finite ==="
$PY -u scripts/check_smoke.py --scorecard "${VALIDATION_DIR}/scorecard.csv" \
    --block_sizes "$BLOCKS"
