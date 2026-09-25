#!/usr/bin/env bash
# The global production run for E1v — BOTH deliverable products, one script.
#
#   ALLOW_GLOBAL=1 ./scripts/run_global_e1v.sh <stage>
#
#     smoke            one epoch, one fold, one window, a handful of prediction blocks, on
#                      the REAL global grid, through every stage including the exporter.
#                      Writes the receipt every long stage refuses to run without.
#     hindcast         k=5 fold training + prediction, base year 2000 -> 2005/2010/2015/2020
#     stitch           holdout mosaic of the five folds
#     score            the scorecard, out of sample, on the stitched hindcast
#     forecast         --train_all_splits on ALL global data, base 2020 -> 2025..2040
#     export_hindcast  five COGs per year + one icechunk store
#     export_forecast  the same for the forward product
#     all              every one of the above, in that order, stopping at the first failure
#
# WHY THIS SCRIPT EXISTS AT ALL. run_global_dist_hindcast.sh and run_global_dist_forecast.sh
# source scripts/dist_base_args.sh, whose BASE_ARGS hardcodes --head_family spline (the
# 29-parameter rational-quadratic head), --seed 46 and e1's context flags. They carry NONE of
# E1v's flags. Pointed at this phase they would train e1, log like e1, score like e1, and
# nothing in them would say so — the accepted-logged-and-inert failure this project has met
# four times. This one sources conv_spline_base.sh, which is where E1v's baseline lives, and
# names the delta from it in one place.
#
# WHAT IS CHECKED BEFORE A RESULT IS BELIEVED, per fold, read back out of the run's own log:
#   loss weights        all four, --mu_mse_weight included (dist_base_args.sh never checked it)
#   context wiring      the TRUNK's channel count off the module, against EXPECT_CTX_CHANNELS
#                       (dist_base_args.sh greps for a "no hm context" string instead)
#   head fingerprint    family pwl AND 17 params/horizon AND ", tails neglog" AND
#                       ", free scale". 17 with the tails, 15 without: the count is the only
#                       thing that separates E1v from a tailless arm, and --isqf_tails has
#                       already once been allocated and never read.
#   row banding         the NUMBER of rows, not merely that banding happened. See ROW_CHUNK.
#   weight averaging    the mean of the last 20 epochs, and prediction repointed at it
#   completeness        valid pixels against the 184.6 M land total, two-sided, BigTIFF
#
# MEASURED ON THIS BOX, not projected (the projections in this project have been wrong in
# both directions). Global smoke, 2026-09-17: one fold, one horizon, 120 prediction blocks
# = 12.7 min; stitching one target year = 10.0 min; five COGs + one icechunk year = 18 min;
# peak python RSS 35.5 GiB, and it is the STITCH that peaks, not prediction (prediction held
# 11.5 GiB with 67 accumulators, so 268 of them is ~31 GiB per fold and two folds fit).
# Land is 184.5 M px of the 684.4 M grid.
#
# THE TWO PRODUCTS SHARE NOTHING, so they can run at once on one GPU each:
#   GPUS=0 ALLOW_GLOBAL=1 ./scripts/run_global_e1v.sh hindcast   # 5 folds, serial
#   GPUS=1 ALLOW_GLOBAL=1 ./scripts/run_global_e1v.sh forecast   # the forward model
# That trades the hindcast's two-GPU parallelism for overlapping the forward model, which is
# the single longest serial job: it predicts the WHOLE globe from one model, where each
# hindcast fold predicts about a third of it.
#
# ROW_CHUNK IS NOT A DETAIL. Prediction holds len(active_horizons) * (3 + 64) full-window
# float32 accumulators — 268 on a four-horizon window. On the 17111 x 40000 grid BASE_ARGS'
# Africa-sized --predict_row_chunk 2048 is 0.305 GiB each, 81.8 GiB for ONE fold, and two
# folds run at once against 125 GB of DRAM. 512 rows is 20.4 GiB. Appended last so it wins.
set -uo pipefail
cd /home/glenn/spatio-temporal

STAGE="${1:?usage: [MODEL=E1v|E2a] ALLOW_GLOBAL=1 $0 <smoke|hindcast|stitch|score|forecast|export_hindcast|export_forecast|all>}"

# Set before sourcing: conv_spline_base.sh takes each of these as ${VAR:-<africa default>}.
export REGION="${REGION:-config/region_to_predict_large.geojson}"
export FOLD_MASK="${FOLD_MASK:-data/raw/hm_global/fold_mask_b4_1000.tif}"
export FOLDS="${FOLDS:-1,2,3,4,5}"
export GPUS="${GPUS:-0,1}"
export EXP_ROOT="${EXP_ROOT:-/mnt/hdd1/spatio-temporal/data/conv_spline/global}"
export LOG_DIR="${LOG_DIR:-data/conv_spline/logs/global}"
export SCORE_DIR="${SCORE_DIR:-data/conv_spline/scores/global}"
export VAL_STRIDE="${VAL_STRIDE:-1024}"
export MAX_EPOCHS="${MAX_EPOCHS:-150}"

source scripts/conv_spline_base.sh
guard_region || exit 1

# ---------------------------------------------------------------- the configuration
# ONE model table, because the head's family, its parameter count and its flags must not be
# able to drift apart. verify_head_fingerprint is handed all three from here, so a model
# added below cannot be run without its fingerprint also being declared.
#
# The parameter count is NOT decoration: 17 with the learned tails and 15 without is the only
# thing that distinguishes E1v from E2a on the same family at free scale, and --isqf_tails
# has already once been accepted, allocated and never read.
#
#   MODEL=E2a ALLOW_GLOBAL=1 ./scripts/run_global_model.sh smoke
case "${MODEL:=E1v}" in
  E1v)
    # Promoted 2026-09-17 at the close of the conv-spline screening phase.
    # models/production/E1v/README.md and models/production/E1v_global/README.md.
    MODEL_FLAGS="--head_family pwl --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0"
    MODEL_FAMILY="pwl"; MODEL_PARAMS="17" ;;
  E2a)
    # The runner-up. Same family and same free scale as E1v, WITHOUT the learned tails, so
    # E1v vs E2a on current code is a clean A/B on the tails alone.
    #
    # NOTE it is 15 params/horizon here and was scored at 16 on Africa: --free_scale did not
    # actually REMOVE the scale channel until 2026-09-16, so the Africa arm carried a dead
    # one. This run is therefore the "E2a re-run on current code" the promotion page says is
    # needed to isolate the tails -- and its numbers are NOT directly comparable to the
    # Africa E2a scorecard.
    MODEL_FLAGS="--head_family pwl --free_scale True --mu_mse_weight 0.0"
    MODEL_FAMILY="pwl"; MODEL_PARAMS="15" ;;
  *)
    echo "FATAL: unknown MODEL='${MODEL}'. Known: E1v, E2a." >&2
    echo "       Add it to the table in $0 together with its family and parameter count;" >&2
    echo "       a model without a declared fingerprint cannot be verified." >&2
    exit 2 ;;
esac
MODEL_FLAGS="${MODEL_FLAGS_OVERRIDE:-$MODEL_FLAGS}"
SEED="${SEED:-42}"

ROW_CHUNK="${ROW_CHUNK:-512}"
WINDOWS="${WINDOWS:-2000}"          # base year 2000 -> targets 2005/2010/2015/2020
HIND_YEARS="2005,2010,2015,2020"
FC_YEARS="2025,2030,2035,2040"
TRAIN_CHIPS="${TRAIN_CHIPS:-100}"
NUM_WORKERS="${NUM_WORKERS:-3}"
# NOT ${SCORE_ROW_CHUNK:-512}. conv_spline_base.sh has already exported SCORE_ROW_CHUNK=0
# -- Africa's 7778-wide rasters are affordable unbanded and banding cost 36% wall clock
# there -- and "0" is neither unset nor empty, so `:-` keeps it. The global scorer then read
# a whole window-year in one call and died on `Unable to allocate 163. GiB for an array with
# shape (64, 17111, 40000)`. Caught by the smoke, which is the only reason it was not found
# three hours into the scoring stage. A global default needs its OWN name.
# 512 rows is 64 x 512 x 40000 x 4 = 5.2 GB, doubled by read_qf's .astype copy.
SCORE_ROW_CHUNK="${GLOBAL_SCORE_ROW_CHUNK:-512}"
ICE_ROW_CHUNK="${GLOBAL_ICE_ROW_CHUNK:-512}"

HIND_NAME="${HIND_NAME:-g_${MODEL}_hind}"
FC_NAME="${FC_NAME:-g_${MODEL}_fc}"
HIND_ROOT="${EXP_ROOT}/${HIND_NAME}"
FC_ROOT="${EXP_ROOT}/${FC_NAME}"
PROD_ROOT="${PROD_ROOT:-/mnt/hdd1/spatio-temporal/data/conv_spline/products/${MODEL}}"
SMOKE_ROOT="${SMOKE_ROOT:-${EXP_ROOT}/smoke_${MODEL}}"
MON_DIR="${MON_DIR:-data/conv_spline/logs/global/monitor}"

# --predict_row_chunk 512 is appended AFTER BASE_ARGS' 2048, and argparse keeps the last
# occurrence. So is MODEL_FLAGS' --mu_mse_weight 0.0 against BASE_ARGS' 1.0.
TRAIN_ARGS="${BASE_ARGS} ${MODEL_FLAGS} --predict_row_chunk ${ROW_CHUNK} --seed ${SEED}"

mkdir -p "$LOG_DIR" "$SCORE_DIR" "$MON_DIR" "$EXP_ROOT"

# The receipt. A smoke proves the CODE it ran, so the stamp records a hash of the code it
# ran: if train_lightning.py, the head, the stitcher, the scorer, the exporter or this script
# changes afterwards, the receipt is void and the long stage refuses. A stale green smoke is
# worse than none, because it is believed.
CODE_FILES="scripts/train_lightning.py scripts/run_hindcast_folds.py scripts/conv_spline_base.sh \
scripts/run_global_model.sh scripts/export_products.py scripts/score_distributional_model.py \
scripts/check_qf_raster.py scripts/check_prediction_complete.py scripts/verify_products.py \
src/stitch.py src/models/spatiotemporal_predictor.py src/models/quantile_pwl.py \
src/models/lightning_module.py"
# Per MODEL: a receipt earned by smoking E1v must never authorise an E2a run.
STAMP="${LOG_DIR}/smoke_ok_${MODEL}.stamp"

code_hash() { cat $CODE_FILES 2>/dev/null | sha256sum | cut -c1-16; }

require_smoke() {
  local want got
  want="$(code_hash)"
  if [ ! -r "$STAMP" ]; then
    echo "REFUSING ${1}: no smoke receipt at ${STAMP}." >&2
    echo "  Run:  ALLOW_GLOBAL=1 $0 smoke" >&2
    echo "  Eight of the twelve defects this project met on first contact with real data" >&2
    echo "  were invisible to a green test suite. The smoke is the gate, not the suite." >&2
    return 1
  fi
  local got_flags
  got_flags=$(awk '/^flags=/{sub(/^flags=/,""); print}' "$STAMP")
  if [ "$got_flags" != "$MODEL_FLAGS" ]; then
    echo "REFUSING ${1}: the smoke receipt is for a different configuration." >&2
    echo "  receipt ${got_flags}" >&2
    echo "  now     ${MODEL_FLAGS}" >&2
    return 1
  fi
  got=$(awk '/^code_hash=/{sub(/^code_hash=/,""); print}' "$STAMP")
  if [ "$got" != "$want" ]; then
    echo "REFUSING ${1}: the smoke receipt is for different code." >&2
    echo "  receipt ${got}   now ${want}" >&2
    echo "  Re-smoke, or the long run is gated on a build that no longer exists." >&2
    return 1
  fi
  echo "  ✓ smoke receipt ${got} matches the code on disk, for ${MODEL}"
}

# monitor_resources.sh APPENDS, so one CSV accumulates every invocation of a stage. The
# first version of stop_monitor took the max over the whole file and reported 99.0 GiB for an
# export that actually peaked at 24.6 -- it was quoting a scorer that had shared the box with
# an earlier, aborted run of the same stage. A peak that silently belongs to a different run
# is rule 25's shape exactly. So the start time is recorded and the summary reads only rows
# at or after it.
start_monitor() {
  local tag="$1"
  MON_START="$(date -Is)"
  MON_TAG="$tag"
  ./scripts/monitor_resources.sh "${MON_DIR}/${tag}.csv" 10 >/dev/null 2>&1 &
  MON_PID=$!
  # Any exit path -- including an external SIGTERM -- must take the monitor with it. An
  # aborted stage once left its monitor sampling for ten hours and poisoning the next run's
  # summary.
  trap 'kill "${MON_PID:-0}" 2>/dev/null' EXIT INT TERM
  echo "  · resource monitor pid ${MON_PID} -> ${MON_DIR}/${tag}.csv (from ${MON_START})"
}
stop_monitor() {
  [ -n "${MON_PID:-}" ] || return 0
  kill "$MON_PID" 2>/dev/null
  wait "$MON_PID" 2>/dev/null
  trap - EXIT INT TERM
  local csv="${MON_DIR}/${1}.csv"
  [ -r "$csv" ] || return 0
  # The number the projections were wrong about twice. Report it, do not project it -- and
  # report THIS run's, not the file's.
  awk -F, -v t0="${MON_START}" '
      NR>1 && $1 >= t0 {n++; if ($13+0>m) m=$13+0; if ($5+0>d) d=$5+0; if ($7+0>s) s=$7+0}
      END {if (n) printf "  · peak over %d samples: largest python RSS %.1f GiB, DRAM used %.0f GiB, swap %.0f GiB\n", n, m, d, s;
           else printf "  · peak: no samples recorded for this run\n"}' "$csv"
  MON_PID=""
}

banner() {
  echo
  echo "=============================================================================="
  echo "$*"
  echo "=============================================================================="
  df -h / /mnt/hdd1 | sed 's/^/  /'
}

# Every per-fold fingerprint, in one place so no stage can check a different set from another.
verify_fold_log() {
  local log="$1" name="$2" want_wa="${3:-20}" ok=1
  verify_loss_weights     "$log" "$name" "$MODEL_FLAGS"                 || ok=0
  verify_context_wiring   "$log" "$name"                              || ok=0
  verify_head_fingerprint "$log" "$name" "$MODEL_FAMILY" "$MODEL_PARAMS" "$MODEL_FLAGS" || ok=0
  verify_row_banding      "$log" "$name" "$ROW_CHUNK"                 || ok=0
  if [ "$want_wa" != "0" ]; then
    verify_weight_averaging "$log" "$name" "$want_wa"                 || ok=0
  fi
  [ "$ok" = 1 ]
}

# ---------------------------------------------------------------- stages

do_smoke() {
  local root="${SMOKE_ROOT}" logs="${LOG_DIR}/smoke"
  rm -rf "$root"
  mkdir -p "$root" "$logs"
  banner "SMOKE ${MODEL} — the real global grid, one epoch, one fold, ${SMOKE_BLOCKS:-400} prediction blocks"
  echo "  Nothing here is a measurement. What it proves: the ${MODEL} head trains, decodes"
  echo "  and WRITES on the 17111 x 40000 grid; the raster reads back monotone through the"
  echo "  scorer's own reader with the right head tag; the scorer, the stitcher, the"
  echo "  exporter, the COG layout and the icechunk shard plan all survive global width; and"
  echo "  --train_all_splits — a branch that by construction has never run on this"
  echo "  configuration — engages."
  echo
  start_monitor smoke

  smoke_fail() {
    stop_monitor smoke
    rm -f "$STAMP"
    echo
    echo "!!! SMOKE FAILED at ${1} — the long run is gated on this and will not start." >&2
    return 1
  }

  # One target year, not four. The accumulator working set is per BAND and identical either
  # way; what four years buys is four full-grid raster writes, and an all-NaN 64-band global
  # raster alone takes 12 min to deflate. A gate that costs more than it saves does not
  # get run.
  local smoke_extra="${TRAIN_ARGS} --predict_subsample_blocks ${SMOKE_BLOCKS:-400} \
--predict_max_target_year 2005"

  echo "--- 1/6 hindcast fold path (train + predict) ---"
  $PY -u scripts/run_hindcast_folds.py \
      --stage train --folds 1 --gpus "${GPUS%%,*}" \
      --region "$REGION" --windows "$WINDOWS" --fold_mask "$FOLD_MASK" \
      --output_root "${root}/hind" --log_dir "$logs" --tag "_smoke_hind" \
      --max_epochs 1 --train_chips 8 --val_stride 8192 --num_workers 2 --disable_wandb \
      --extra_train_args "$smoke_extra" || { smoke_fail "1/6 train+predict"; return 1; }
  local hlog="${logs}/hindcast_fold1_smoke_hind.log"
  verify_fold_log "$hlog" "smoke-hind" 1 || { smoke_fail "1/6 fingerprints"; return 1; }
  $PY scripts/check_qf_raster.py --pred_dir "${root}/hind/preds" \
      --expect_params "$MODEL_PARAMS" --expect_family "$MODEL_FAMILY" \
      --expect_tag_family "$MODEL_FAMILY" --expect_tag_params "$MODEL_PARAMS" \
      --log "$hlog" --flags "$MODEL_FLAGS" || { smoke_fail "1/6 qf raster"; return 1; }

  echo
  echo "--- 2/6 stitch ---"
  $PY -u scripts/run_hindcast_folds.py \
      --stage stitch --stitch_mode holdout --folds 1 \
      --region "$REGION" --windows "$WINDOWS" --fold_mask "$FOLD_MASK" \
      --output_root "${root}/hind" --keep_fold_rasters \
      || { smoke_fail "2/6 stitch"; return 1; }

  echo
  echo "--- 3/6 score ---"
  # Scoring is a multi-hour stage and was the only one no smoke had ever touched. That is
  # exactly where the KeyError lived that made every real scoring run drop both gates:
  # gate_stats had a passing unit test that built its own input, and the real caller does
  # `del cell["qf"]` before it. It is also where the 163 GiB read was found.
  $PY -u scripts/score_distributional_model.py \
      --stitched_dir "${root}/hind/stitched" --label smoke_global --folds 1 \
      --out_dir "${root}/scores" --fold_mask "$FOLD_MASK" \
      --row_chunk "$SCORE_ROW_CHUNK" --min_count 200 \
      || { smoke_fail "3/6 score"; return 1; }

  echo
  echo "--- 4/6 export (COGs + icechunk) on the global grid ---"
  $PY -u scripts/export_products.py \
      --src_dir "${root}/hind/stitched" --mode hindcast --base_year 2000 --years 2005 \
      --out_dir "${root}/products" --row_chunk "$ICE_ROW_CHUNK" --overwrite \
      || { smoke_fail "4/6 export"; return 1; }
  $PY -u scripts/verify_products.py \
      --products "${root}/products" --src_dir "${root}/hind/stitched" \
      --mode hindcast --base_year 2000 --years 2005 --n_px 2000 \
      || { smoke_fail "4/6 verify products"; return 1; }

  echo
  echo "--- 5/6 forward path: --train_all_splits, base 2020 ---"
  local fclog="${logs}/forecast_smoke.log"
  rm -rf "${root}/fc"; mkdir -p "${root}/fc/preds"
  run_forecast_train "${root}/fc/preds" "$fclog" 1 \
      "--predict_subsample_blocks ${SMOKE_BLOCKS:-400} --predict_max_target_year 2025" \
      || { smoke_fail "5/6 forward train+predict"; return 1; }
  verify_forecast_log "$fclog" "smoke-fc" 1 || { smoke_fail "5/6 forward fingerprints"; return 1; }
  $PY scripts/check_qf_raster.py --pred_dir "${root}/fc/preds" \
      --expect_params "$MODEL_PARAMS" --expect_family "$MODEL_FAMILY" \
      --expect_tag_family "$MODEL_FAMILY" --expect_tag_params "$MODEL_PARAMS" \
      --log "$fclog" --flags "$MODEL_FLAGS" || { smoke_fail "5/6 forward qf raster"; return 1; }

  echo
  echo "--- 6/6 receipt ---"
  stop_monitor smoke
  { echo "code_hash=$(code_hash)"
    echo "model=${MODEL}"
    echo "when=$(date -Is)"
    echo "git=$(git rev-parse --short HEAD 2>/dev/null)"
    echo "flags=${MODEL_FLAGS}"
    echo "row_chunk=${ROW_CHUNK}"
    echo "score_row_chunk=${SCORE_ROW_CHUNK}"
    echo "region=${REGION}"; } > "$STAMP"
  echo "  ✓ SMOKE PASSED — receipt written to ${STAMP}"
  return 0
}

do_hindcast() {
  require_smoke hindcast || return 1
  banner "GLOBAL HINDCAST — ${HIND_NAME} | folds ${FOLDS} | window ${WINDOWS} -> ${HIND_YEARS}"
  echo "  args: ${TRAIN_ARGS}"
  mkdir -p "$HIND_ROOT"
  start_monitor hindcast
  $PY -u scripts/run_hindcast_folds.py \
      --stage train --folds "$FOLDS" --gpus "$GPUS" \
      --region "$REGION" --windows "$WINDOWS" --fold_mask "$FOLD_MASK" \
      --output_root "$HIND_ROOT" --log_dir "$LOG_DIR" --tag "_${HIND_NAME}" \
      --max_epochs "$MAX_EPOCHS" --train_chips "$TRAIN_CHIPS" \
      --val_stride "$VAL_STRIDE" --num_workers "$NUM_WORKERS" \
      --wandb_group "global-${HIND_NAME}" \
      --extra_train_args "$TRAIN_ARGS"
  local rc=$? ok=1
  stop_monitor hindcast
  echo
  echo "--- fingerprints, read back from each fold's own log ---"
  for f in ${FOLDS//,/ }; do
    verify_fold_log "${LOG_DIR}/hindcast_fold${f}_${HIND_NAME}.log" "fold${f}" 20 || ok=0
  done
  # A RANGE, not a number with a tolerance. Prediction is restricted to the tiles that
  # OVERLAP the fold and keeps every pixel of each -- so a fold raster covers about 1.9x its
  # own territory, MEASURED on fold_mask_b4_1000: own land 35.8-38.4 M px, tile-covered
  # 67.4-71.6 M. An "expected a fifth of the land" check would have refused all five folds
  # at the end of a nine-hour run. What the bound is really for is the ceiling: a raster
  # truncated at the classic-TIFF 4 GiB limit reads its unwritten rows back as finite zeros
  # and reports close to the whole 684.4 M grid.
  echo
  echo "--- per-fold raster completeness (a fold covers its territory dilated by one tile) ---"
  $PY scripts/check_prediction_complete.py --dir "${HIND_ROOT}/preds" \
      --patterns "fold*_w${WINDOWS}_prediction_*_qf_blended.tif" \
      --min_px 40000000 --max_px 120000000 --require_bigtiff "_qf_" || ok=0
  [ "$ok" = 1 ] || { echo "REFUSING to continue: a fold did not train or write the configuration asked for." >&2; return 4; }
  return $rc
}

do_stitch() {
  require_smoke stitch || return 1
  banner "STITCH (holdout) — ${HIND_NAME}"
  echo "  holdout, never mean: a mean mosaic is in-sample everywhere and must never be scored."
  start_monitor stitch
  $PY -u scripts/run_hindcast_folds.py \
      --stage stitch --stitch_mode holdout --folds "$FOLDS" \
      --region "$REGION" --windows "$WINDOWS" --fold_mask "$FOLD_MASK" \
      --output_root "$HIND_ROOT" --keep_fold_rasters
  local rc=$?
  stop_monitor stitch
  [ $rc -eq 0 ] || return $rc
  echo
  echo "--- stitched completeness: five folds together must cover the land exactly once ---"
  $PY scripts/check_prediction_complete.py --dir "${HIND_ROOT}/stitched" \
      --patterns "w${WINDOWS}_prediction_*_central.tif,w${WINDOWS}_prediction_*_qf.tif" \
      --tol 0.10 --expect_files 8 --require_bigtiff "_qf" || return 4
}

do_score() {
  require_smoke score || return 1
  banner "SCORE — ${HIND_NAME}, out of sample, folds ${FOLDS}"
  start_monitor score
  $PY -u scripts/score_distributional_model.py \
      --stitched_dir "${HIND_ROOT}/stitched" \
      --label "$HIND_NAME" --folds "$FOLDS" --out_dir "$SCORE_DIR" \
      --fold_mask "$FOLD_MASK" --row_chunk "$SCORE_ROW_CHUNK"
  local rc=$?
  stop_monitor score
  return $rc
}

# The forward model's command, written ONCE. The smoke and the production run differ only in
# the arguments passed here: a second spelling of this invocation is how the smoke comes to
# test something the real run does not do.
run_forecast_train() {
  local pred_dir="$1" log="$2" epochs="$3" extra="${4:-}"
  mkdir -p "$pred_dir" "$(dirname "$log")"
  echo "  log -> ${log}"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" PYTHONUNBUFFERED=1 $PY -u scripts/train_lightning.py \
      --train_all_splits True \
      --max_epochs "$epochs" --train_chips "$TRAIN_CHIPS" \
      --val_stride "$VAL_STRIDE" --num_workers "$NUM_WORKERS" \
      --batch_size 8 --devices 1 \
      --norm_stats_json data/conv_spline/norm_stats.json \
      --run_full_set_evaluation False --run_large_area_prediction True \
      --predict_region "$REGION" \
      --predict_final_year 2040 \
      --predict_stride 64 --predict_batch_size 32 \
      --predict_output_dir "$pred_dir" \
      --predict_output_prefix "" \
      --hidden_dim 64 --num_layers 4 --kernel_size 3 \
      --locenc_out_channels 8 --locenc_legendre_polys 10 \
      --histogram_lambda_w2 0.1 --histogram_warmup_epochs 0 \
      ${TRAIN_ARGS} ${extra} \
      --wandb_group "global-${FC_NAME}" --wandb_run_name "${FC_NAME}" \
      --wandb_tags "global,forecast,${MODEL}" > "$log" 2>&1
  local rc=$?
  echo "  train+predict exited rc=${rc}"
  return $rc
}

# A forward model has no held-out score, so the only evidence it is the right model is its
# own log. --train_all_splits is also the one flag whose failure is silent and expensive:
# without it the production model trains on the 70% split and discards 30% of the world.
verify_forecast_log() {
  local log="$1" name="$2" want_wa="${3:-20}" ok=1
  if grep -q "PRODUCTION MODE: training on EVERY chip in the split mask" "$log"; then
    echo "  ✓ ${name}: production mode engaged (no geography held out)"
  else
    echo "FATAL: ${name}: --train_all_splits did not engage; this model trained on split 1." >&2
    ok=0
  fi
  if grep -q "FOLD-CV MODE" "$log"; then
    echo "FATAL: ${name}: FOLD-CV MODE banner present — a fold was held out." >&2; ok=0
  else
    echo "  ✓ ${name}: no fold held out"
  fi
  verify_fold_log "$log" "$name" "$want_wa" || ok=0
  [ "$ok" = 1 ]
}

do_forecast() {
  require_smoke forecast || return 1
  banner "GLOBAL FORWARD MODEL + FORECAST — ${FC_NAME} | base 2020 -> ${FC_YEARS}"
  echo "  args: ${TRAIN_ARGS}"
  mkdir -p "${FC_ROOT}/preds"
  start_monitor forecast
  local log="${LOG_DIR}/${FC_NAME}.log"
  run_forecast_train "${FC_ROOT}/preds" "$log" "$MAX_EPOCHS"
  local rc=$? ok=1
  stop_monitor forecast
  echo
  echo "--- fingerprints ---"
  verify_forecast_log "$log" "$FC_NAME" 20 || ok=0
  echo
  echo "--- completeness: 16 rasters, every one over all the land ---"
  $PY scripts/check_prediction_complete.py --dir "${FC_ROOT}/preds" \
      --patterns "prediction_*_central_blended.tif,prediction_*_lower_blended.tif,prediction_*_upper_blended.tif,prediction_*_qf_blended.tif" \
      --tol 0.10 --expect_files 16 --require_bigtiff "_qf_" || ok=0
  [ "$ok" = 1 ] || { echo "REFUSING: the forward model is not the configuration asked for." >&2; return 4; }
  return $rc
}

do_export() {
  local mode="$1" src="$2" base="$3" years="$4"
  require_smoke "export_${mode}" || return 1
  banner "EXPORT ${mode} — five COGs per year + one icechunk store"
  start_monitor "export_${mode}"
  $PY -u scripts/export_products.py \
      --src_dir "$src" --mode "$mode" --base_year "$base" --years "$years" \
      --out_dir "${PROD_ROOT}/${mode}" --row_chunk "$ICE_ROW_CHUNK" --overwrite
  local rc=$?
  [ $rc -eq 0 ] && { $PY -u scripts/verify_products.py \
      --products "${PROD_ROOT}/${mode}" --src_dir "$src" --mode "$mode" \
      --base_year "$base" --years "$years" --n_px 20000; rc=$?; }
  stop_monitor "export_${mode}"
  return $rc
}

case "$STAGE" in
  # Print the arguments that would actually take effect and stop. BASE_ARGS names
  # --predict_row_chunk 2048 and --mu_mse_weight 1.0, and this script appends 512 and 0.0
  # after them; "the flags I passed" is not "the flags that took effect", so there is a
  # stage that prints the latter and a test that reads it.
  args)
    echo "STAGE=args"
    echo "REGION=${REGION}"
    echo "FOLD_MASK=${FOLD_MASK}"
    echo "FOLDS=${FOLDS}"
    echo "WINDOWS=${WINDOWS}"
    echo "ROW_CHUNK=${ROW_CHUNK}"
    echo "SCORE_ROW_CHUNK=${SCORE_ROW_CHUNK}"
    echo "ICE_ROW_CHUNK=${ICE_ROW_CHUNK}"
    echo "EXPECT_CTX_CHANNELS=${EXPECT_CTX_CHANNELS}"
    echo "MODEL=${MODEL}"
    echo "MODEL_FLAGS=${MODEL_FLAGS}"
    echo "MODEL_FAMILY=${MODEL_FAMILY}"
    echo "MODEL_PARAMS=${MODEL_PARAMS}"
    echo "TRAIN_ARGS=${TRAIN_ARGS}"
    echo "HIND_ROOT=${HIND_ROOT}"
    echo "FC_ROOT=${FC_ROOT}"
    echo "PROD_ROOT=${PROD_ROOT}"
    echo "STAMP=${STAMP}"
    echo "code_hash=$(code_hash)"
    ;;
  smoke)            do_smoke ;;
  hindcast)         do_hindcast ;;
  stitch)           do_stitch ;;
  score)            do_score ;;
  forecast)         do_forecast ;;
  export_hindcast)  do_export hindcast "${HIND_ROOT}/stitched" 2000 "$HIND_YEARS" ;;
  export_forecast)  do_export forecast "${FC_ROOT}/preds"      2020 "$FC_YEARS" ;;
  all)
    do_smoke     || exit 1
    do_hindcast  || exit 1
    do_stitch    || exit 1
    do_score     || exit 1
    do_forecast  || exit 1
    do_export hindcast "${HIND_ROOT}/stitched" 2000 "$HIND_YEARS" || exit 1
    do_export forecast "${FC_ROOT}/preds"      2020 "$FC_YEARS"   || exit 1
    banner "ALL STAGES COMPLETE — products under ${PROD_ROOT}"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
