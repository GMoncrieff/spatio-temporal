#!/usr/bin/env bash
# The V4b ensemble on the global grid. A thin env wrapper over
# run_dist_ensemble_variant2.sh, which stays the Africa-shaped driver that produced the
# promoted evidence.
#
#   ./scripts/run_global_dist_ensemble.sh <exp_root> <members>
#
# Everything that differs from an Africa run is here and is a property of the grid, not a
# preference:
#
#   WRAP_LON=auto   the full-globe raster's antimeridian is a real join. generate_ensemble.py
#                   detects it from the grid (W >= 39000, origin at -180); the Africa driver
#                   passes an explicit False, which overrides the detection and would put a
#                   seam down the Pacific in every member.
#   FOLD_MASK       fold_mask_b4 (512 px blocks), what this lineage trains and stitches
#                   against -- not the region root's 128 px production checkerboard.
#   MEM_BUDGET_GB   validate_ensemble sizes its streaming stages against this. Africa at
#                   M=100 died allocating (M, H, W) float64 for the B=1 block pass; the
#                   global grid is 10.8x that.
#
# V4b, frozen (docs/dist_ensemble_phase.md): PIT-space spectrum, no appended broad component,
# Student-t copula at nu=7 with the chi2 factor drawn stratified.
set -uo pipefail
cd /home/glenn/spatio-temporal

ROOT="${1:?usage: $0 <exp_root> <members>}"
MEMBERS="${2:-400}"

export REGION_NAME="${REGION_NAME:-global}"
export REGION_GEOJSON="${REGION_GEOJSON:-config/region_to_predict_large.geojson}"
export FOLD_MASK="${FOLD_MASK:-data/raw/hm_global/fold_mask_b4_1000.tif}"
export WRAP_LON="${WRAP_LON:-auto}"
export MEM_BUDGET_GB="${MEM_BUDGET_GB:-48}"
export STORE_DIR="${STORE_DIR:-/mnt/hdd1/spatio-temporal/data/ensemble/exp/$(basename "$ROOT")}"
# V4b: PIT space, no hand-added broad component.
export SPECTRA_FLAGS="${SPECTRA_FLAGS:---fit_space pit --qf_dir ${SHARED_ROOT:-$ROOT}/stitched --long_weight 0}"
# V4b: Student-t copula, nu measured at 7, chi2 factor drawn stratified.
export GEN_FLAGS="${GEN_FLAGS:---copula t --copula_df 7 --copula_w_draw stratified}"

echo "=== global V4b ensemble | ${ROOT} | M=${MEMBERS} ==="
echo "  region      ${REGION_NAME} (${REGION_GEOJSON})"
echo "  fold mask   ${FOLD_MASK}"
echo "  wrap_lon    ${WRAP_LON}"
echo "  mem budget  ${MEM_BUDGET_GB} GB"
echo "  spectra     ${SPECTRA_FLAGS}"
echo "  gen         ${GEN_FLAGS}"
df -h / /mnt/hdd1 | sed 's/^/  /'
exec ./scripts/run_dist_ensemble_variant2.sh "$ROOT" "$MEMBERS"
