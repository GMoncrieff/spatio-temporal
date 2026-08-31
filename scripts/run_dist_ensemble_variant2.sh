#!/usr/bin/env bash
# The distributional lineage's Phase 0-4, parameterised so one hindcast can feed several
# ensemble/correlation variants.
#
# This is run_dist_ensemble_loop.sh with three changes, and each one is here because the
# original cost something:
#
#   1. Phase 4 now passes --qf_dir. The original put it in COMMON, which only
#      generate_ensemble.py consumes, so `./scripts/run_dist_ensemble_loop.sh` reproduced
#      scorecard_twopiece/ (77/46) and not scorecard_qf/ (83/40) -- T3 recovering normal
#      scores by inverting a two-piece normal the members were never drawn from, and T5.1
#      comparing the sample median to the central head rather than to Q(0.5). The frozen
#      qf card came from a separate hand-run.
#   2. SPECTRA_FLAGS / GEN_FLAGS pass through to the two stages a correlation variant
#      actually changes, so a variant is a flag rather than an edit.
#   3. SKIP_DIAGNOSTICS=1 reuses SHARED_ROOT's residuals and diagnostics. They are a pure
#      function of the stitched hindcast and the model, so every variant that leaves the
#      hindcast alone would rebuild them byte-identically for ~25 min and 2.8 GB.
#
# The two deliberate absences of the original are unchanged: no apply_recalibration.py
# (there are no width factors -- the bounds are Q(0.025) and Q(0.975) off the model's own
# quantile function) and no fit_marginal_shape.py (the marginal IS the quantile function).
# The spatial spectrum and the horizon coupling stay, because a per-pixel quantile function
# says nothing about how neighbouring pixels co-vary and independent draws would be white
# noise. Both are re-derived from THIS model's own residuals, never carried over.
#
#   ./scripts/run_dist_ensemble_variant.sh <exp_root> <members>
#
# env:
#   SHARED_ROOT       hindcast to read stitched/residuals/diagnostics from (default: the
#                     exp_root itself, i.e. the original single-experiment behaviour)
#   SKIP_DIAGNOSTICS  1 to reuse SHARED_ROOT's residuals + diagnostics instead of rebuilding
#   SPECTRA_FLAGS     extra args to scripts/fit_field_spectra.py    (e.g. "--long_weight 0")
#   SPECTRA_JSON      use this spectral_fits.json instead of fitting one (skips Phase 2)
#   GEN_FLAGS         extra args to scripts/generate_ensemble.py
#
set -uo pipefail
cd /home/glenn/spatio-temporal

ROOT="${1:?usage: $0 <exp_root> <members>}"
MEMBERS="${2:-400}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"

SHARED_ROOT="${SHARED_ROOT:-$ROOT}"
SKIP_DIAGNOSTICS="${SKIP_DIAGNOSTICS:-0}"
SPECTRA_FLAGS="${SPECTRA_FLAGS:-}"
GEN_FLAGS="${GEN_FLAGS:-}"

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
# wrap_lon is a physical property of the grid, not a preference: on the full-globe raster the
# antimeridian is a real join, and sampling it unwrapped puts a seam down the Pacific in every
# member. generate_ensemble.py auto-detects it (W >= 39000 and the transform origin at -180),
# but an explicit flag overrides the detection -- which is why this defaults to the Africa
# runs' False and takes "auto" to hand the decision back.
WRAP_LON="${WRAP_LON:-False}"
# The fold mask the diagnostics stratify on. The region root's fold_mask.tif is the 128 px
# production checkerboard; the distributional lineage trains and stitches against the 512 px
# fold_mask_b4, and a mask that does not match the rasters silently scores a subset.
FOLD_MASK="${FOLD_MASK:-${REGION_ROOT}/fold_mask.tif}"

DIST_RASTER="${REGION_ROOT}/covariates/w${BASE_YEAR}_dist_past_change.tif"
mkdir -p "$ROOT" "$STORE_DIR"

# The hindcast rasters are read-only inputs shared by every variant, so the variant root
# links to them rather than copying 12 GB. A symlink is safe here precisely because nothing
# rmtree's it; the ensemble store above is the thing that must stay a real path.
link_shared() {  # $1 = subdirectory name
  if [ ! -e "${ROOT}/$1" ]; then
    ln -s "$(realpath "${SHARED_ROOT}/$1")" "${ROOT}/$1" || exit 1
    echo "  linked ${ROOT}/$1 -> ${SHARED_ROOT}/$1"
  fi
}
[ "$SHARED_ROOT" != "$ROOT" ] && link_shared stitched
STITCHED="${ROOT}/stitched"

echo "=== region root (${REGION_NAME}) ==="
if [ ! -f "$DIST_RASTER" ]; then
  $PY -u scripts/make_region_root.py --region "$REGION_GEOJSON" --name "$REGION_NAME" --verify
else
  echo "  already built: ${REGION_ROOT}"
fi

if [ "$SKIP_DIAGNOSTICS" = "1" ]; then
  echo "=== Phase 0-1: reusing ${SHARED_ROOT}'s residuals and diagnostics ==="
  # Residuals are (stitched, model, region) and nothing else; the variants below change the
  # spectrum fitted TO them or the field drawn FROM it, never the residual itself.
  [ "$SHARED_ROOT" != "$ROOT" ] && { link_shared residuals; link_shared diagnostics; }
  for f in "${ROOT}/residuals/manifest.csv" \
           "${ROOT}/residuals/horizon_autocorrelation.json" \
           "${ROOT}/diagnostics/variogram_fits.csv"; do
    [ -f "$f" ] || { echo "FATAL: SKIP_DIAGNOSTICS=1 but $f is missing"; exit 1; }
  done
else
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
      --fold_mask "${FOLD_MASK}" \
      --ecoregion_raster "${REGION_ROOT}/ecoregion.tif" \
      --block_sizes "$BLOCKS" --n_pairs "$PAIRS" --max_lag_px "$MAXLAG" \
      --stages coverage,variogram,audit || exit 1
fi

if [ -n "${SPECTRA_JSON:-}" ]; then
  # A PIT-space fit reads an 8 GB quantile-function raster per window and costs the better
  # part of an hour on Africa, and `add_long_scale_component` is a pure post-hoc rescale of
  # the NNLS solution -- so two variants differing only in --long_weight share one fit.
  # scripts/derive_long_component.py produces the other from it exactly.
  echo "=== Phase 2: reusing the fitted spectrum ${SPECTRA_JSON} ==="
  [ -f "$SPECTRA_JSON" ] || { echo "FATAL: SPECTRA_JSON=${SPECTRA_JSON} is missing"; exit 1; }
  cp "$SPECTRA_JSON" "${ROOT}/spectral_fits.json" || exit 1
  $PY -c "
import json,sys
b=json.load(open('${ROOT}/spectral_fits.json'))
print('  fit_space=%s long_weight=%s' % (b.get('fit_space','?'), b.get('long_weight','?')))
for h in sorted(b['by_horizon'], key=int):
    f=b['by_horizon'][h]
    t=sum(f['weights'])+f['nugget']
    assert abs(t-1)<1e-9, 'h=%s weights+nugget=%r' % (h,t)
    print('  h=%-3s nugget %.4f  n_ranges %d  long %s' % (h, f['nugget'], len(f['ranges_px']),
          f.get('long_scale',{}).get('weight','-')))
" || exit 1
else
  echo "=== Phase 2: field spectrum, fitted to THIS model's residuals ==="
  echo "  SPECTRA_FLAGS: ${SPECTRA_FLAGS:-<none: argparse defaults>}"
  $PY -u scripts/fit_field_spectra.py \
      --manifest "${ROOT}/residuals/manifest.csv" \
      --out "${ROOT}/spectral_fits.json" $SPECTRA_FLAGS || exit 1
fi

QF=(--qf_dir "$STITCHED" --qf_pattern "w${BASE_YEAR}_prediction_{year}_qf.tif")
COMMON=(--central_dir "$STITCHED" --recal_dir "$STITCHED"
        --central_pattern "w${BASE_YEAR}_prediction_{year}_central.tif"
        --recal_pattern "w${BASE_YEAR}_prediction_{year}_{q}.tif"
        "${QF[@]}"
        --years "$YEARS" --base_year "$BASE_YEAR"
        --spectral_fits "${ROOT}/spectral_fits.json"
        --variogram_fits "${ROOT}/diagnostics/variogram_fits.csv"
        --rho_json "${ROOT}/residuals/horizon_autocorrelation.json"
        --gpus "$GPUS")
if [ "$WRAP_LON" != "auto" ]; then
  COMMON+=(--wrap_lon "$WRAP_LON")
fi

echo "=== Phase 2c: smoke the sampler at M=8 before spending hours on M=${MEMBERS} ==="
# Prove the chain before a long run. A mis-indexed slab or a wrong scale produces members
# outside their own pixel's quantile range, which this catches in ~1 min; the full
# marginal-recovery test needs M large and runs after Phase 3.
SMOKE="${STORE_DIR}/smoke_m8.icechunk"
rm -rf "$SMOKE"
$PY -u scripts/generate_ensemble.py "${COMMON[@]}" $GEN_FLAGS \
    --members 8 --out "$SMOKE" --disable_wandb > "${ROOT}/smoke_generate.log" 2>&1 || {
      echo "FATAL: M=8 smoke generation failed; see ${ROOT}/smoke_generate.log"; exit 1; }
$PY -u scripts/check_qf_ensemble.py --ensemble "$SMOKE" \
    --qf "${STITCHED}/w${BASE_YEAR}_prediction_2020_qf.tif" --horizon_index 3 \
    --mode bounds || { echo "FATAL: the sampler does not reproduce its own quantile range"; exit 1; }
rm -rf "$SMOKE"

echo "=== Phase 3: ensemble, marginal read from the model's quantile function ==="
$PY -u scripts/generate_ensemble.py "${COMMON[@]}" $GEN_FLAGS --members "$MEMBERS" \
    --out "$MEMBERS_STORE" --wandb_group "dist-$(basename "$ROOT")" || exit 1

echo "=== Phase 3b: independent-pixel null (same marginals, no spatial structure) ==="
$PY -u scripts/generate_ensemble.py "${COMMON[@]}" $GEN_FLAGS --members "$MEMBERS" --independent \
    --out "$NULL_STORE" --disable_wandb || exit 1

echo "=== Phase 3c: the members must reproduce the quantile function they came from ==="
$PY -u scripts/check_qf_ensemble.py --ensemble "$MEMBERS_STORE" \
    --qf "${STITCHED}/w${BASE_YEAR}_prediction_2020_qf.tif" --horizon_index 3 \
    --mode marginal | tee "${ROOT}/qf_marginal_check.txt" || exit 1

echo "=== Phase 4: T1-T8 scorecard ==="
# --qf_dir is the whole point of this line existing separately from the loop script: without
# it T3 recovers normal scores through a two-piece normal, T5.1 scores against the central
# head instead of Q(0.5), and T4.1 reads an attenuated between-horizon correlation.
$PY -u scripts/validate_ensemble.py \
    --ensemble "$MEMBERS_STORE" --null_ensemble "$NULL_STORE" \
    --recal_dir "$STITCHED" \
    --qf_dir "$STITCHED" --qf_pattern "w{base}_prediction_{year}_qf.tif" \
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
