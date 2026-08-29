#!/usr/bin/env bash
# The distributional lineage's Phase 0-4: residuals -> spectrum -> ensemble -> T1-T8 scorecard.
#
# This is run_region_loop.sh with the post-hoc chain removed, because that chain is what the
# branch exists to delete. Two stages are deliberately ABSENT and their absence is the point:
#
#   * no `apply_recalibration.py` -- there are no width factors to fit. The published bounds
#     are Q(0.025) and Q(0.975) read off the model's own quantile function.
#   * no `fit_marginal_shape.py` -- the marginal IS the quantile function. Reshaping it with
#     an empirical fit would put back the layer under test.
#
# What is KEPT, and why, is just as deliberate. The spatial spectrum and the AR(1) horizon
# coupling are properties of the *field*, not of the marginal: a per-pixel quantile function
# says nothing about how neighbouring pixels co-vary, and independent draws would be white
# noise. Both are re-derived from THIS model's own residuals, never carried over -- a spectrum
# fitted to another model's residuals describes another model's error.
#
#   ./scripts/run_dist_ensemble_loop.sh <exp_root> <members>
#
set -uo pipefail
cd /home/glenn/spatio-temporal

ROOT="${1:?usage: $0 <exp_root> <members>}"
MEMBERS="${2:-400}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"

REGION_NAME="${REGION_NAME:-africa}"
REGION_GEOJSON="${REGION_GEOJSON:-config/region_africa.geojson}"
REGION_ROOT="data/ensemble/region/${REGION_NAME}"
# Only w2000 reaches +20 yr, so the scored window is fixed. Every h=20 number is in-sample in
# time whatever the geography.
BASE_YEAR="${BASE_YEAR:-2000}"
YEARS="${YEARS:-2005,2010,2015,2020}"
BLOCKS="${BLOCKS:-1,10,100}"
PAIRS="${PAIRS:-80000}"
MAXLAG="${MAXLAG:-256}"
GPUS="${GPUS:-0,1}"
# Africa at M=400 is ~85 GB per ensemble and members+null is ~170 GB, which root does not
# have. The HDD path is passed directly, never through the symlink: the store's directory is
# cleared with shutil.rmtree and that refuses on a symbolic link.
STORE_DIR="${STORE_DIR:-/mnt/hdd1/spatio-temporal/data/ensemble/exp/$(basename "$ROOT")}"
MEMBERS_STORE="${STORE_DIR}/members_m${MEMBERS}.icechunk"
NULL_STORE="${STORE_DIR}/null_m${MEMBERS}.icechunk"
# A regional working set can hide a quadratic: validate_ensemble sizes its streaming stages
# against this rather than allocating (M, H, W) float64 and dying at B=1.
MEM_BUDGET_GB="${MEM_BUDGET_GB:-24}"

STITCHED="${ROOT}/stitched"
DIST_RASTER="${REGION_ROOT}/covariates/w${BASE_YEAR}_dist_past_change.tif"
mkdir -p "$STORE_DIR"

echo "=== region root (${REGION_NAME}) ==="
if [ ! -f "$DIST_RASTER" ]; then
  $PY -u scripts/make_region_root.py --region "$REGION_GEOJSON" --name "$REGION_NAME" --verify
else
  echo "  already built: ${REGION_ROOT}"
fi

echo "=== Phase 0: residuals from this model's own held-out rasters ==="
$PY -u scripts/build_region_residuals.py \
    --pred_dir "$STITCHED" --pred_suffix "" --keep_splits all \
    --out_dir "${ROOT}/residuals" \
    --covariate_dir "${REGION_ROOT}/covariates" || exit 1

echo "=== Phase 0b: AR(1) horizon coupling from those residuals ==="
$PY -u -c "
import json, sys
sys.path.insert(0, '.')
from src.ensemble.residuals import horizon_autocorrelation
rho = horizon_autocorrelation('${ROOT}/residuals/manifest.csv')
json.dump({str(k): v for k, v in rho.items()}, open('${ROOT}/residuals/horizon_autocorrelation.json', 'w'), indent=2)
print('  rho =', rho)
" || exit 1

echo "=== Phase 1: coverage, variograms, class audit ==="
$PY -u scripts/run_diagnostics.py \
    --manifest "${ROOT}/residuals/manifest.csv" --out_dir "${ROOT}" \
    --fold_mask "${REGION_ROOT}/fold_mask.tif" \
    --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
    --block_sizes "$BLOCKS" --n_pairs "$PAIRS" --max_lag_px "$MAXLAG" \
    --stages coverage,variogram,audit || exit 1

echo "=== Phase 2: field spectrum, fitted to THIS model's residuals ==="
$PY -u scripts/fit_field_spectra.py \
    --manifest "${ROOT}/residuals/manifest.csv" \
    --out "${ROOT}/spectral_fits.json" || exit 1

COMMON=(--central_dir "$STITCHED" --recal_dir "$STITCHED"
        --central_pattern "w${BASE_YEAR}_prediction_{year}_central.tif"
        --recal_pattern "w${BASE_YEAR}_prediction_{year}_{q}.tif"
        --qf_dir "$STITCHED" --qf_pattern "w${BASE_YEAR}_prediction_{year}_qf.tif"
        --years "$YEARS" --base_year "$BASE_YEAR"
        --spectral_fits "${ROOT}/spectral_fits.json"
        --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv"
        --rho_json "${ROOT}/residuals/horizon_autocorrelation.json"
        --wrap_lon False --gpus "$GPUS")

echo "=== Phase 2c: smoke the sampler at M=8 before spending hours on M=${MEMBERS} ==="
# Prove the chain before a long run. A mis-indexed slab or a wrong scale produces members
# outside their own pixel's quantile range, which this catches in ~1 min; the full
# marginal-recovery test needs M large and runs after Phase 3.
SMOKE="${STORE_DIR}/smoke_m8.icechunk"
rm -rf "$SMOKE"
$PY -u scripts/generate_ensemble.py "${COMMON[@]}" \
    --members 8 --out "$SMOKE" --disable_wandb > "${ROOT}/smoke_generate.log" 2>&1 || {
      echo "FATAL: M=8 smoke generation failed; see ${ROOT}/smoke_generate.log"; exit 1; }
$PY -u scripts/check_qf_ensemble.py --ensemble "$SMOKE" \
    --qf "${STITCHED}/w${BASE_YEAR}_prediction_2020_qf.tif" --horizon_index 3 \
    --mode bounds || { echo "FATAL: the sampler does not reproduce its own quantile range"; exit 1; }
rm -rf "$SMOKE"

echo "=== Phase 3: ensemble, marginal read from the model's quantile function ==="
$PY -u scripts/generate_ensemble.py "${COMMON[@]}" --members "$MEMBERS" \
    --out "$MEMBERS_STORE" --wandb_group "dist-$(basename "$ROOT")" || exit 1

echo "=== Phase 3b: independent-pixel null (same marginals, no spatial structure) ==="
$PY -u scripts/generate_ensemble.py "${COMMON[@]}" --members "$MEMBERS" --independent \
    --out "$NULL_STORE" --disable_wandb || exit 1

echo "=== Phase 3c: the members must reproduce the quantile function they came from ==="
$PY -u scripts/check_qf_ensemble.py --ensemble "$MEMBERS_STORE" \
    --qf "${STITCHED}/w${BASE_YEAR}_prediction_2020_qf.tif" --horizon_index 3 \
    --mode marginal | tee "${ROOT}/qf_marginal_check.txt" || exit 1

echo "=== Phase 4: T1-T8 scorecard ==="
$PY -u scripts/validate_ensemble.py \
    --ensemble "$MEMBERS_STORE" --null_ensemble "$NULL_STORE" \
    --recal_dir "$STITCHED" \
    --central_pattern "w{base}_prediction_{year}_central.tif" \
    --recal_pattern "w{base}_prediction_{year}_{q}.tif" \
    --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
    --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv" \
    --rho_json "${ROOT}/residuals/horizon_autocorrelation.json" \
    --dist_raster "$DIST_RASTER" \
    --block_sizes "$BLOCKS" --out_dir "${ROOT}/validation" \
    --mem_budget_gb "$MEM_BUDGET_GB" --mem_trace \
    --wandb_group "dist-$(basename "$ROOT")" || exit 1

echo
echo "Scorecard: ${ROOT}/validation/scorecard.csv"
