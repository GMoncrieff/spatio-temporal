#!/usr/bin/env bash
# Section 4.1's scale arms. Separate from run_conv_spline_slate.sh because the ORDER is the
# design: every arm below E0a conflates the horizon-cumulative constraint with the
# anchor/scale factorisation, and E0a is the only one that changes one thing.
#
#   ./scripts/run_conv_spline_scale_arms.sh              # all of them, in order
#   ONLY="E0a" ./scripts/run_conv_spline_scale_arms.sh   # one
#
# Refuses without b1's floor, for the reason the E0-E5 slate does.
set -euo pipefail

source "$(dirname "$0")/conv_spline_base.sh"
guard_region

SEEDS="${SEEDS:-42}"
FIRST_FOLD="${FOLDS%%,*}"
FLOOR_PREFIX="${FLOOR_PREFIX:-b1}"

if ! ls "${SCORE_DIR}"/summary_${FLOOR_PREFIX}_s*.json >/dev/null 2>&1; then
  echo "FATAL: no ${FLOOR_PREFIX} floor in ${SCORE_DIR}. Run the baseline first." >&2
  exit 1
fi

# ORDER IS LOAD-BEARING, and it is the doc's:
#
#   E0a first and ALONE -- the only arm that changes one thing. --spline_cumulative_width
#     False removes the horizon accumulation and nothing else, on the incumbent head. Every
#     --free_scale arm below removes the accumulation only as a CONSEQUENCE of removing the
#     factorisation, so none of them can separate the two questions. If E0a's bands widen
#     with lead time unprompted, the constraint was free insurance and everything downstream
#     inherits that finding instead of re-deriving it against a different spline class.
#   E1a next -- two arms, because --isqf_space is an experiment rather than a setting: the
#     measured far-tail miss is two-sided and near-symmetric (pit_lt_0001 / pit_gt_0999 =
#     1.017 / 1.254 / 0.971 across b1's three seeds), which logit can address and neglog,
#     unbounded above only, structurally cannot.
#   E1b and E2a PAIRED -- the same ladder up to which knot the free location attaches to, so
#     run together or the control is wasted. E1b keeps Park et al.'s q0 at Q(0.0); E2a's
#     anchor is Q(0.5). What the pair measures is where persistence enters.
#   E1c LAST, and unconditionally -- the fourth cell of a 2x2 over {tails, free scale}.
#     Gating it on movement in E1a or E1b would drop it exactly when an interaction is the
#     only remaining explanation.
#
# All four free-scale arms carry --mu_mse_weight 0.0: an MSE term pinning E[Q] is not a
# bystander to an experiment about where the width comes from, and the subtraction has to be
# complete or a null cannot distinguish "the factorisation was earning its keep" from "the
# MSE term supplied what it used to".
ARMS=(
  "E0a_nocumw|--spline_cumulative_width False --mu_mse_weight 0.0|the constraint alone, on the incumbent: no new code, one thing changed"
  "E1a_tails_logit|--head_family isqf --isqf_tails True --isqf_space logit|learned two-sided tail rates, symmetric transform"
  "E1a_tails_neglog|--head_family isqf --isqf_tails True --isqf_space neglog|the same, unbounded above only"
  "E1b_freescale|--head_family isqf --free_scale True --mu_mse_weight 0.0|Park et al.'s scale arrangement; q0 at Q(0.0)"
  "E2a_pwl_freescale|--head_family pwl --free_scale True --mu_mse_weight 0.0|the same ladder, anchor at Q(0.5)"
  "E1c_paper|--head_family isqf --isqf_tails True --isqf_space logit --free_scale True --mu_mse_weight 0.0|Park et al. as published"
)

for ENTRY in "${ARMS[@]}"; do
  IFS='|' read -r TAG FLAGS WHY <<<"$ENTRY"
  if [ -n "${ONLY:-}" ] && ! grep -qw "${TAG%%_*}" <<<"$ONLY"; then continue; fi
  for SEED in $SEEDS; do
    NAME="${TAG}_s${SEED}"
    echo "=============================================================="
    echo "=== ${NAME}"
    echo "=== ${WHY}"
    echo "=== delta from b1: ${FLAGS}"
    echo "=============================================================="
    EXP_ROOT="$EXP_ROOT" REGION="$REGION" FOLD_MASK="$FOLD_MASK" \
      ./scripts/run_central_experiment.sh "$NAME" "$GPUS" "$FOLDS" "${FLAGS} --seed ${SEED}"

    LOG="${LOG_DIR}/hindcast_fold${FIRST_FOLD}_${NAME}.log"
    verify_loss_weights "$LOG" "$NAME" "$FLAGS"
    verify_context_wiring "$LOG" "$NAME"
    verify_weight_averaging "$LOG" "$NAME"

    $PY -u scripts/score_distributional_model.py \
        --stitched_dir "${EXP_ROOT}/${NAME}/stitched" \
        --label "$NAME" --folds "$FOLDS" --out_dir "$SCORE_DIR" \
        --fold_mask "$FOLD_MASK" --row_chunk "$SCORE_ROW_CHUNK"
  done
done

echo
$PY scripts/compare_conv_spline_runs.py --score_dir "$SCORE_DIR" --floor_prefix "$FLOOR_PREFIX"
