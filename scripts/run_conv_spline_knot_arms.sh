#!/usr/bin/env bash
# The knot / floor slate: six arms on the two free-scale heads that survived section 4.1.
#
#   ./scripts/run_conv_spline_knot_arms.sh              # all six, in order
#   ONLY="E2iv" ./scripts/run_conv_spline_knot_arms.sh  # one
#
# Refuses without b1's floor, for the reason the other two slates do.
#
# WHAT CHANGED UNDER ALL SIX. --free_scale now genuinely removes the scale channel. Until
# 2026-09-16 it only stopped the DECODER reading channel 1 while the width head still emitted
# it, so E1b/E2a ran at 16 params and E1c/E1d at 18 where section 4.1 advertised 15 and 17,
# each carrying one dead channel per horizon that took no gradient from the quantile path.
# Every arm below is therefore NOT directly comparable to E1b/E2a/E1c/E1d as scored: the head
# is one parameter smaller and the width head is gone entirely. That is the point -- simplicity
# is a scoring criterion in this project, and these are now the cheapest heads on the slate
# for real rather than on paper.
#
# THE SUBSTRATES. E2a and E1d, the two best-measured free-scale arms:
#   E2a  pwl,  free scale                    crps_skill5 0.2167 (best), degen5 0.0000
#   E1d  isqf, free scale, neglog tails      skill20 0.2771,  far_tail_excess5 2.65
#
# WHY THESE SIX.
#   i    skew14 -- E5's re-placed grid, which was the only arm at or above b1 on every
#        central column while also zeroing degen5. Never tried on a free-scale head.
#   ii   skew11 -- the strong form: body collapsed to its minimum, 11 bins not 14, capacity
#        moved above u = 0.75. Asks whether the body needed resolution at all, which
#        CLAUDE.md still lists as open. Also the cheapest heads here, 12 and 14 params.
#   iv   dense24 -- the other side of ii. A strict SUPERSET of default14, so no knot moves
#        and the only variable is capacity: 24 bins against 14, at 25 and 27 params the most
#        expensive heads run in this phase. Placement (i) and capacity (ii, iv) are kept
#        apart deliberately; a grid that did both at once would make a null unattributable.
#
# NOT ON THIS SLATE, BUT LANDED WITH IT. Working out how --spline_gap_floor should compose
# with --free_scale showed that BOTH spellings of the density floor had hm_std the wrong way
# up, dividing where the algebra multiplies. Measured, E4's floor capped the implied density
# at ~7020 against its own stated 578. E4's scorecard had already said so and nobody had read
# it that way -- over_f_max_frac_ref_5 = 0.7464 on the one arm built to make a density above
# f_max structurally impossible (rule 12). E4's result therefore stands as measured but not as
# described. The fix is in src/models/quantile_pwl.py:density_floor_width and is independent
# of any arm here; no arm on this slate uses --spline_gap_floor.
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

E2A="--head_family pwl --free_scale True --mu_mse_weight 0.0"
E1D="--head_family isqf --isqf_tails True --isqf_space neglog --free_scale True --mu_mse_weight 0.0"

# Paired so the two families move together at each step: a finding that shows up on one head
# and not the other is a finding about the head, and that is only visible if both are run.
ARMS=(
  "E2i_pwl_skew14|${E2A} --spline_knots skew14|E2a on E5's re-placed grid, 15 params"
  "E1i_isqf_skew14|${E1D} --spline_knots skew14|E1d on the same grid, 17 params"
  "E2ii_pwl_skew11|${E2A} --spline_knots skew11|the strong form: 11 bins, 12 params"
  "E1ii_isqf_skew11|${E1D} --spline_knots skew11|the strong form on isqf, 14 params"
  "E2iv_pwl_dense24|${E2A} --spline_knots dense24|24 bins, a strict superset of default14: were there simply too few, 25 params"
  "E1iv_isqf_dense24|${E1D} --spline_knots dense24|the same on isqf, 27 params -- the most expensive head on any slate here"
)

for ENTRY in "${ARMS[@]}"; do
  IFS='|' read -r TAG FLAGS WHY <<<"$ENTRY"
  if [ -n "${ONLY:-}" ] && ! grep -qw "${TAG%%_*}" <<<"$ONLY"; then continue; fi
  for SEED in $SEEDS; do
    NAME="${TAG}_s${SEED}"
    echo "=============================================================="
    echo "=== ${NAME}"
    echo "=== ${WHY}"
    echo "=== flags: ${FLAGS}"
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
