#!/usr/bin/env bash
# Regenerate an existing configuration's ensemble at a higher member count and re-score it.
#
#   ./scripts/rescore_at_m800.sh <exp> <recal_subdir> <shape_json> <suffix> [members]
#
# Why this exists: several scorecard rows — T5.1, T5.2, T1.2, T1.3, T2.5 — are estimated from
# the member sample, and their Monte-Carlo tolerances tighten as 1/sqrt(M). The marginal phase
# found an M=100 comparison between two marginal families that *reversed* at M=400 for exactly
# this reason, so a configuration whose residual is spikier is penalised more at fixed M.
# docs/background/model_phase.md section 5.3 measured E5's shape slope at the median falling 24-43%
# against the baseline's, and predicted that its two lost rows recover at higher M.
#
# **Both configurations have to be rescored**, or the comparison swaps one confound for
# another. Nothing here retrains anything: the recalibrated rasters, the fitted spectrum, the
# width factors and the marginal shape are all reused exactly as generated.

set -euo pipefail

EXP="${1:?usage: $0 <exp> <recal_subdir> <shape_json> <suffix> [members]}"
RECAL_SUB="${2:?}"
SHAPE_JSON="${3:?}"
SUFFIX="${4:?}"
MEMBERS="${5:-800}"

PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
GPUS="${GPUS:-0,1}"
ROOT="data/ensemble/exp/${EXP}"
REGION_ROOT="data/ensemble/region/southern_africa"
RECAL_DIR="${ROOT}/${RECAL_SUB}"
DIST="${REGION_ROOT}/covariates/w2000_dist_past_change.tif"
MEM="${ROOT}/members${SUFFIX}.icechunk"
NUL="${ROOT}/null${SUFFIX}.icechunk"
VAL="${ROOT}/validation${SUFFIX}"

COMMON=(--central_dir "$RECAL_DIR" --recal_dir "$RECAL_DIR"
        --central_pattern 'w2000_prediction_{year}_central_recal.tif'
        --recal_pattern 'w2000_prediction_{year}_{q}_recal.tif'
        --years 2005,2010,2015,2020 --base_year 2000 --members "$MEMBERS"
        --spectral_fits "${ROOT}/spectral_fits.json"
        --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv"
        --rho_json "${ROOT}/residuals/horizon_autocorrelation.json"
        --wrap_lon False --marginal_shape "$SHAPE_JSON" --dist_raster "$DIST"
        --gpus "$GPUS")

echo "=== ${EXP}${SUFFIX}: generating M=${MEMBERS} from ${RECAL_DIR} ==="
[ -d "$MEM" ] || $PY -u scripts/generate_ensemble.py "${COMMON[@]}" --out "$MEM" --disable_wandb
[ -d "$NUL" ] || $PY -u scripts/generate_ensemble.py "${COMMON[@]}" --out "$NUL" --independent --disable_wandb

echo "=== ${EXP}${SUFFIX}: scoring T1-T8 at M=${MEMBERS} ==="
$PY -u scripts/validate_ensemble.py \
    --ensemble "$MEM" --null_ensemble "$NUL" --recal_dir "$RECAL_DIR" \
    --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "${ROOT}/residuals/horizon_autocorrelation.json" \
    --recal_manifest "${RECAL_DIR}/recal_manifest.json" \
    --dist_raster "$DIST" --marginal_shape "$SHAPE_JSON" \
    --block_sizes 1,10,100 --out_dir "$VAL" \
    --wandb_group "m${MEMBERS}-rescore"

echo "Scorecard: ${VAL}/scorecard.csv"
