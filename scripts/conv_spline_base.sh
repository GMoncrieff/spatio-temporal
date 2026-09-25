#!/usr/bin/env bash
# The conv-spline phase's baseline, defined once. Every runner sources this; every experiment
# is a stated delta from it.
#
# **Every loss weight must be named here explicitly.** `run_hindcast_folds.py` injects the
# frozen product's own weights (--ssim_weight 0.2 --laplacian_weight 0.3 --histogram_weight 1.0)
# into every fold command, and --extra_train_args is appended last, so argparse takes the final
# occurrence. Anything BASE_ARGS does not name is silently inherited from a product this phase
# is not evaluating. That is not hypothetical -- it cost a 35-minute run in the previous phase,
# caught only by reading the run's own LOSS WEIGHTS banner.
#
# The neighbourhood context -- distance to past change and the neighbourhood HM summaries --
# goes into the TRUNK, beside elevation and climate, and into no head. That is not a setting:
# --trunk_context, --central_context and --quantile_context no longer exist, and the wiring is
# part of the model. e1 fed the heads and never the trunk; b1 and everything after it feed the
# trunk and never the heads.
#
# Where it goes is not a setting. WHICH covariate it is still is, and that is named here.
# b1 is "e1 plus exactly one change", and e1 is --context_radii 3,30,100 with the
# neighbourhood-HM summaries at mean,max over 3,30,100 -- twelve trunk channels
# (docs/dist_model_phase2.md, and the shipped global run). Left to argparse this defaults to
# --context_radii 1,3,10,30,100 and NO hm stats: eight channels, e1's dropped fine radii
# back, and the HM summaries the phase doc says reach the trunk simply absent. That is rule
# 16 in its exact shape -- "no extra flags" is not the architecture, it is argparse defaults
# -- and verify_context_wiring could not see it, because it compared the module against the
# same defaults and read 8 == 8. So the count is named, and checked against the run's log.

export REGION="${REGION:-config/region_africa.geojson}"
export FOLD_MASK="${FOLD_MASK:-data/raw/hm_global/fold_mask_b4_1000.tif}"
export VAL_STRIDE="${VAL_STRIDE:-1024}"
export MAX_EPOCHS="${MAX_EPOCHS:-150}"
export FOLDS="${FOLDS:-1,2}"
export GPUS="${GPUS:-0,1}"
# The HDD path spelled out, not data/conv_spline/exp behind a symlink. NOTE the reason
# CLAUDE.md used to give for this -- "shutil.rmtree refuses on a symbolic link" -- is about
# code that no longer exists: that rmtree lived in migrate_zarr_to_icechunk.py, which commit
# 60c6a9e deleted with the ensemble layer, and `git grep rmtree` now finds nothing.
# The live reason is size. Measured on the 2026-09-14 smoke: one fold writes 8.0 GB of
# per-fold rasters over ten window-years (the qf raster deflates from 8.1 GB raw to ~0.85 GB),
# and --keep_fold_rasters retains a set per fold beside the stitched mosaic. Logs and scores
# stay on the SSD under data/conv_spline/: they are small and every verifier greps them.
export EXP_ROOT="${EXP_ROOT:-/mnt/hdd1/spatio-temporal/data/conv_spline/exp}"
# Named here, not spelled out in each runner: the verifiers read the log that
# run_central_experiment.sh writes, and two spellings of one path is how the check
# ended up pointed at a file that never existed.
export LOG_DIR="${LOG_DIR:-data/conv_spline/logs}"
export SCORE_DIR="${SCORE_DIR:-data/conv_spline/scores}"
export PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"

EXPECT_SSIM="0.0"
EXPECT_LAP="0.0"
EXPECT_HIST="0.0"
# The auxiliary MSE on E[Q]. Named here because it was the one loss weight the phase left to
# an argparse default, which is rule 16's exact shape: every arm carried a second objective
# nobody chose. The free-scale arms (E0a, E1b, E1c, E2a) set it to 0.0 -- an MSE term pinning
# the first moment is not a neutral bystander to an experiment about where the width comes
# from -- and they pass `--mu_mse_weight 0.0` in their own flags, which BASE_ARGS being first
# lets them override.
EXPECT_MU_MSE="1.0"

# e1's neighbourhood covariate: 3 occupancy radii + 3 fixed bands + mean,max over 3 HM
# radii = 12. EXPECT_CTX_CHANNELS must be recomputed with the flags if either changes --
# src.models.change_weights.context_channel_count is the one definition.
EXPECT_CTX="--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean,max"
EXPECT_CTX_CHANNELS="12"

# Africa's prediction accumulators are len(active_horizons) * (3 + 64) full-window float32
# arrays -- 268 on the four-horizon window. MEASURED on the 2026-09-14 smoke: one fold peaks
# at 40.7 GB resident, and b1 runs two folds concurrently on a 125 GB box, so unbanded is
# ~81 GB before GDAL's cache. (plan_row_bands' docstring estimated ~22 GB for Africa; that
# was low by 1.85x.) 2048 rows caps a fold near 17 GB and costs ~6% more prediction work,
# because each band recomputes the tiles in its halo. The banded write is an identity --
# tests/test_predict_row_bands.py.
EXPECT_ROW_CHUNK="--predict_row_chunk 2048"

# float32, not the ensemble's old int16 x 1/32767. MEASURED on b1_s42, which is why this is
# not a preference: the int16 quantum is 3.05e-05 in ABSOLUTE HM, b1's core is narrower than
# that, and 34% of adjacent quantile levels exported to the SAME code. max_density_p99 then
# read 1531.8 at every horizon -- exactly dp_max / one quantum, the statistic's ceiling
# rather than a density -- and the pixels whose gap was exactly zero were dropped from the
# gate entirely for having infinite density. Both gates were unreadable. Doubles the qf
# raster (~19 GB -> ~35 GB per experiment); the HDD has room and a pinned metric does not.
EXPECT_QF_DTYPE="--predict_qf_dtype float32"

# Average the last 20 epochs instead of letting ModelCheckpoint pick one. MEASURED on the
# three-seed argmin floor (data/conv_spline/scores/floor_argmin/README.md): val_crps plateaus
# after ~epoch 30 and then oscillates by +/- 0.001, so the argmin picked epochs 67, 124, 146,
# 67, 127, 111 across six fold-models -- a coin toss among ~100 candidates. Accuracy did not
# care (rmse20 reproducible to 0.1%); the GATES did, with bands of 25-115% of their own mean
# on three replicates of one configuration. Every arm in this phase is judged on those gates,
# so a floor that wide makes the slate unrankable.
#
# This flag's own help text records the same finding from a prior phase (140/85/60 across
# three seeds) and BOTH shipped global configs pass it -- run_global_dist_hindcast.sh:47 and
# run_global_dist_forecast.sh:40. BASE_ARGS did not, which is rule 16 for the third time in
# this phase. Implies --checkpoint_select final, so prediction runs on the averaged weights.
EXPECT_WA="--weight_avg_last 20"

# val_crps, not val_total_loss: an experiment that sets --mu_mse_weight 0 would otherwise be
# selecting epochs on a different quantity from every other run in the slate.
# --free_scale False is pinned because train_lightning.py's defaults are E2a's (free scale on)
# since 2026-09-25, and the rational-quadratic spline refuses --free_scale. BASE_ARGS is the b1
# baseline and must keep meaning b1; E1v/E2a turn it back on in their MODEL_FLAGS, which come later.
export BASE_ARGS="--head_family spline --free_scale False --central_residual True --checkpoint_monitor val_crps \
--mu_mse_weight ${EXPECT_MU_MSE} ${EXPECT_CTX} ${EXPECT_ROW_CHUNK} ${EXPECT_QF_DTYPE} ${EXPECT_WA} \
--ssim_weight ${EXPECT_SSIM} --laplacian_weight ${EXPECT_LAP} --histogram_weight ${EXPECT_HIST}"

# The scorer's own working set, measured the same way and worse: 59.1 GB peak on ONE fold.
# read_qf pulls the whole 64-band raster (16.1 GB on Africa) and the ref-grid gate built
# [256, n_px] and [255, n_px] temporaries on top of it -- the float64 density array alone was
# 13.6 GB. That is now blocked by pixel inside qf_diagnostics.fence_per_pixel, which bounds it
# regardless of how many pixels a stratum has: MEASURED 59.1 -> 32.1 GB on the same rasters,
# 31 min against 33, and all 88 summary metrics bit-identical. Two folds project to ~48 GB
# (the 16.1 GB raster read is fixed; the rest doubles), which fits with room, so no row
# banding by default.
#
# --row_chunk is still there and still a proven identity (tests/test_score_dist_row_chunk.py),
# but it is NOT free: at 512 it cost ~36% wall clock, which is 4 h across this phase's 14
# arms. Raise it only if a measurement says the peak bites -- and measure, do not project.
export SCORE_ROW_CHUNK="${SCORE_ROW_CHUNK:-0}"

# Read the effective weights back out of the fold log and refuse to continue if they are not
# what BASE_ARGS asked for. Verify the fingerprint before trusting a retrain: a run on
# inherited defaults reads exactly like a real finding.
verify_loss_weights() {
  local log="$1" name="$2" expect_extra="${3:-}"
  local got_ssim got_lap got_hist got_mse
  # Fail CLOSED on a log that is missing or carries no banner. Every expected weight in this
  # phase is 0.0 and awk coerces an empty string to 0, so a missing log made all three
  # comparisons read 0==0 and the check PASSED having read nothing -- the exact failure this
  # function exists to prevent, in the one direction that is silent. Rule: prove a check
  # fires on a control, or it checks nothing.
  if [ ! -r "$log" ]; then
    echo "FATAL: ${name}: no readable fold log at ${log}; loss weights unverified." >&2
    echo "       The run cannot be trusted -- an inherited weight set reads as a finding." >&2
    return 1
  fi
  if ! grep -q "LOSS WEIGHTS" "$log"; then
    echo "FATAL: ${name}: ${log} has no LOSS WEIGHTS banner; loss weights unverified." >&2
    return 1
  fi
  got_mse=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/MSE weight:/ {print $3; exit}')
  got_ssim=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/SSIM weight:/ {print $3; exit}')
  got_lap=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/Laplacian weight:/ {print $3; exit}')
  got_hist=$(grep -A5 "LOSS WEIGHTS" "$log" | awk '/Histogram weight:/ {print $3; exit}')
  local want_ssim="$EXPECT_SSIM" want_lap="$EXPECT_LAP" want_mse="$EXPECT_MU_MSE"
  case "$expect_extra" in
    *--ssim_weight*)      want_ssim=$(sed -E 's/.*--ssim_weight ([0-9.]+).*/\1/' <<<"$expect_extra") ;;
  esac
  case "$expect_extra" in
    *--laplacian_weight*) want_lap=$(sed -E 's/.*--laplacian_weight ([0-9.]+).*/\1/' <<<"$expect_extra") ;;
  esac
  case "$expect_extra" in
    *--mu_mse_weight*)    want_mse=$(sed -E 's/.*--mu_mse_weight ([0-9.]+).*/\1/' <<<"$expect_extra") ;;
  esac
  if [ -z "$got_ssim" ] || [ -z "$got_lap" ] || [ -z "$got_hist" ] || [ -z "$got_mse" ]; then
    echo "FATAL: ${name}: could not parse all four weights out of ${log}" >&2
    echo "  got  mse='${got_mse}' ssim='${got_ssim}' lap='${got_lap}' hist='${got_hist}'" >&2
    return 1
  fi
  if ! awk -v a="$got_mse" -v b="$want_mse" 'BEGIN{exit !(a+0==b+0)}' \
     || ! awk -v a="$got_ssim" -v b="$want_ssim" 'BEGIN{exit !(a+0==b+0)}' \
     || ! awk -v a="$got_lap" -v b="$want_lap" 'BEGIN{exit !(a+0==b+0)}' \
     || ! awk -v a="$got_hist" 'BEGIN{exit !(a+0==0)}'; then
    echo "FATAL: ${name} trained on the wrong loss weights." >&2
    echo "  got  mse=${got_mse} ssim=${got_ssim} lap=${got_lap} hist=${got_hist}" >&2
    echo "  want mse=${want_mse} ssim=${want_ssim} lap=${want_lap} hist=0" >&2
    return 1
  fi
  echo "  ✓ ${name}: loss weights verified (mse=${got_mse} ssim=${got_ssim} lap=${got_lap} hist=${got_hist})"
}

# The trunk must actually receive the context, and under --central_residual the ConvLSTM's
# gradient from the *central* loss is zero by construction -- so a trunk that never got the
# covariate trains, scores, and looks exactly like a trunk that got it and ignored it. There
# is no flag to inspect any more, so the fingerprint is the module's own channel count, and
# this cross-checks it against the count --context_radii / --hm_context_stats imply. Prove a
# check fires on a control before trusting it (tests/test_conv_spline_paths.py).
verify_context_wiring() {
  local log="$1" name="$2" n_args n_trunk
  if [ ! -r "$log" ]; then
    echo "FATAL: ${name}: no readable fold log at ${log}; context wiring unverified." >&2
    return 1
  fi
  n_args=$(grep -m1 "^Context channels:" "$log" | awk '{print $3}')
  n_trunk=$(grep -m1 "^Context into trunk:" "$log" | awk '{print $4}')
  if [ -z "$n_trunk" ] || [ -z "$n_args" ]; then
    echo "FATAL: ${name}: ${log} carries no context fingerprint." >&2
    echo "  Context channels='${n_args:-<nothing read>}' into trunk='${n_trunk:-<nothing read>}'" >&2
    return 1
  fi
  if ! [ "$n_trunk" -gt 0 ] 2>/dev/null; then
    echo "FATAL: ${name}: the trunk was built with ${n_trunk} context channels." >&2
    echo "       The covariate never reached the model; the run is not this configuration." >&2
    return 1
  fi
  if [ "$n_trunk" != "$n_args" ]; then
    echo "FATAL: ${name}: trunk has ${n_trunk} context channels, the flags imply ${n_args}." >&2
    echo "       Check --context_radii / --hm_context_stats against the checkpoint, if any." >&2
    return 1
  fi
  # ON is not the same as ONLY, and neither is the same as THE RIGHT ONE. The two counts
  # above agree whenever the flags and the module agree -- including when both are argparse
  # defaults nobody chose. This is the only line that can tell b1's covariate from the
  # default one.
  if [ -n "${EXPECT_CTX_CHANNELS:-}" ] && [ "$n_trunk" != "$EXPECT_CTX_CHANNELS" ]; then
    echo "FATAL: ${name}: trunk has ${n_trunk} context channels; this phase's baseline is" >&2
    echo "       ${EXPECT_CTX_CHANNELS} (${EXPECT_CTX})." >&2
    echo "       An arm on a different covariate is not comparable to the floor." >&2
    return 1
  fi
  if ! grep -q "^Context into trunk:.*heads: none" "$log"; then
    echo "FATAL: ${name}: the log does not say the heads are excluded." >&2
    return 1
  fi
  echo "  ✓ ${name}: context wiring verified (${n_trunk} channels -> trunk, heads none)"
}

# The averaged weights must actually be what got predicted. Two fingerprints, both printed by
# the run itself: the averaging writes the mean into the tensors, and prediction is repointed
# at the end-of-training checkpoint rather than the monitored one. A run that silently fell
# back to the argmin would carry the wide gate band and read as a null.
verify_weight_averaging() {
  local log="$1" name="$2" want="${3:-20}" got
  if [ ! -r "$log" ]; then
    echo "FATAL: ${name}: no readable fold log at ${log}; weight averaging unverified." >&2
    return 1
  fi
  got=$(grep -m1 "^\[weight averaging\] wrote the mean of the last" "$log" | awk '{print $9}')
  if [ -z "$got" ]; then
    echo "FATAL: ${name}: no weight-averaging line in ${log}." >&2
    echo "       The run selected a single epoch, and its gates carry the argmin band." >&2
    return 1
  fi
  if [ "$got" != "$want" ]; then
    echo "FATAL: ${name}: averaged ${got} epochs, expected ${want}." >&2
    return 1
  fi
  if ! grep -q "Prediction will use the end-of-training checkpoint" "$log"; then
    echo "FATAL: ${name}: averaging ran but prediction was not repointed at it." >&2
    return 1
  fi
  echo "  ✓ ${name}: weight averaging verified (mean of the last ${got} epochs, predicted)"
}

# Africa, not southern Africa, and not the globe. Southern Africa's far-field band holds ZERO
# pixels and its [0,0.01) HM stratum is 6% of the region against 40% of Africa, so a
# stratified finding measured there is provisional at best and has already cost this project
# a whole phase. The globe is for promotion only, on instruction, never for iteration.
guard_region() {
  case "${REGION}" in
    *africa.geojson) ;;
    *large*|*global*)
      # It printed FATAL and then continued whenever ALLOW_GLOBAL=1, so the one line a log
      # reader greps said the run had been refused when it had been authorised. Rule 28: a
      # banner that does not distinguish the two cases is not a fingerprint of either.
      if [ "${ALLOW_GLOBAL:-0}" = "1" ]; then
        echo "  ✓ GLOBAL RUN AUTHORISED: REGION=${REGION}, ALLOW_GLOBAL=1"
      else
        echo "FATAL: REGION looks global and ALLOW_GLOBAL is not 1." >&2
        echo "       Screening iterates on Africa; the globe runs on instruction." >&2
        return 1
      fi ;;
    *) echo "  ! REGION=${REGION} is neither Africa nor global -- continuing, but check it" >&2 ;;
  esac
}

# The head itself must be the one asked for. Three distinct flags shape it and every one of
# them has already failed silently once: --isqf_tails was gated on head_family=="isqf" so a
# pwl arm allocated the channels and never read them; --free_scale emitted a scale channel it
# had stopped reading, so four scored arms ran a parameter heavier than advertised; and the
# banner printed "Spline head" with the rational-quadratic parameter count for all three
# families. The parameter count is what distinguishes them all -- E1v is 17 (1 location + 14
# increments + 2 tail rates) and a tailless pwl free-scale arm is 15 -- so it is the
# fingerprint, and the family and the two modifiers are checked beside it because a count
# alone cannot say WHICH two parameters were added.
#
#   verify_head_fingerprint <log> <name> <family> <params> "<flags>"
verify_head_fingerprint() {
  local log="$1" name="$2" want_fam="$3" want_np="$4" flags="${5:-}"
  local line got_fam got_np
  if [ ! -r "$log" ]; then
    echo "FATAL: ${name}: no readable fold log at ${log}; head fingerprint unverified." >&2
    return 1
  fi
  line=$(grep -m1 "^Spline head:" "$log" || true)
  if [ -z "$line" ]; then
    echo "FATAL: ${name}: ${log} has no head banner; the head is unverified." >&2
    echo "       A triple-head run prints none, and that is the loudest way this fails." >&2
    return 1
  fi
  got_fam=$(sed -E 's/^Spline head:[[:space:]]+family ([A-Za-z]+),.*/\1/' <<<"$line")
  got_np=$(sed -E 's/.*[^0-9]([0-9]+) params\/horizon.*/\1/' <<<"$line")
  if [ "$got_fam" != "$want_fam" ]; then
    echo "FATAL: ${name}: head family is '${got_fam}', expected '${want_fam}'." >&2
    echo "  banner: ${line}" >&2
    return 1
  fi
  if [ "$got_np" != "$want_np" ]; then
    echo "FATAL: ${name}: head emitted ${got_np} params/horizon, expected ${want_np}." >&2
    echo "  banner: ${line}" >&2
    return 1
  fi
  # The two modifiers, read off the same line. Derived from the flags rather than passed
  # separately so a runner cannot ask for tails and forget to check for them.
  case "$flags" in
    *--isqf_tails\ True*)
      local want_space
      want_space=$(sed -E 's/.*--isqf_space ([A-Za-z]+).*/\1/' <<<"$flags")
      case "$flags" in *--isqf_space*) ;; *) want_space="logit" ;; esac
      if ! grep -q ", tails ${want_space}," <<<"$line"; then
        echo "FATAL: ${name}: asked for --isqf_tails on ${want_space}; the banner does not" >&2
        echo "       say so. The tails were allocated and never read once already." >&2
        echo "  banner: ${line}" >&2
        return 1
      fi ;;
    *)
      if grep -q ", tails " <<<"$line"; then
        echo "FATAL: ${name}: the banner reports learned tails nobody asked for." >&2
        return 1
      fi ;;
  esac
  case "$flags" in
    *--free_scale\ True*)
      if ! grep -q ", free scale," <<<"$line"; then
        echo "FATAL: ${name}: asked for --free_scale; the banner does not say so." >&2
        echo "  banner: ${line}" >&2
        return 1
      fi ;;
    *)
      if grep -q ", free scale," <<<"$line"; then
        echo "FATAL: ${name}: the banner reports a free scale nobody asked for." >&2
        return 1
      fi ;;
  esac
  echo "  ✓ ${name}: head verified (${got_fam}, ${got_np} params/horizon)"
}

# Prediction accumulators are len(active_horizons) * (3 + 64) full-window float32 arrays --
# 268 on a four-horizon window -- and each one spans the band. On the 17111 x 40000 global
# grid a 2048-row band is 0.305 GiB per accumulator, so BASE_ARGS' Africa-sized
# --predict_row_chunk 2048 is 81.8 GiB for ONE fold and two folds run at once, against
# 125 GB of DRAM. 512 rows is 20.4 GiB. The value is therefore not a detail: the check reads
# back the number the run actually banded on, not merely that it banded at all.
#
#   verify_row_banding <log> <name> <expected rows>
verify_row_banding() {
  local log="$1" name="$2" want="$3" line got
  if [ ! -r "$log" ]; then
    echo "FATAL: ${name}: no readable log at ${log}; row banding unverified." >&2
    return 1
  fi
  line=$(grep -m1 "Row banding: " "$log" || true)
  if [ -z "$line" ]; then
    echo "FATAL: ${name}: the run did not band its prediction rows." >&2
    echo "       Unbanded global accumulators are ~185 GiB resident for one fold." >&2
    return 1
  fi
  got=$(sed -E 's/.*bands of ([0-9]+) rows.*/\1/' <<<"$line")
  if [ "$got" != "$want" ]; then
    echo "FATAL: ${name}: banded on ${got} rows, expected ${want}." >&2
    echo "  ${line}" >&2
    return 1
  fi
  echo "  ✓ ${name}: $(sed 's/^ *//' <<<"$line")"
}
