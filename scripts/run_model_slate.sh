#!/usr/bin/env bash
# Run a queue of model-phase screening experiments back to back, scoring each one.
#
#   ./scripts/run_model_slate.sh <slate-file> [folds]
#
# Slate file lines are  name | extra train flags | max_epochs  (the last two optional).
# Blank lines and lines starting with # are skipped. Every experiment is trained on the
# same folds and scored on the same pixels, so the deltas between them are attributable.
#
# The phase reference adds two things to the configuration e5_all_k5 shipped:
#
#   --histogram_weight 0        the histogram term carries no gradient at all (its counts
#                               are boolean comparisons into a plain zeros buffer), so it
#                               has never trained anything — but it *is* in val_total_loss
#                               at weight 1.0 and swings ~60x between epochs, which makes
#                               checkpoint selection partly a lottery.
#   --checkpoint_monitor val_central_loss
#                               val_total_loss includes pinball, so a quantile-only change
#                               selects a different epoch and therefore a different central
#                               field. Monitoring the central objective alone means a
#                               quantile-only experiment must leave the central field
#                               bit-identical — which is a free correctness check on every
#                               such run rather than a confound.
#
# One experiment failing does not stop the queue; failures are listed at the end.

set -uo pipefail

SLATE="${1:?usage: $0 <slate-file> [folds]}"
FOLDS="${2:-1,2}"

PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
GPUS="${GPUS:-0,1}"
SCORE_DIR="${SCORE_DIR:-data/ensemble/exp/scores}"
PHASE_REF="${PHASE_REF:---central_residual True --central_context True --monotone_quantile_width True --quantile_context True --histogram_weight 0 --checkpoint_monitor val_central_loss}"

mkdir -p "$SCORE_DIR"
FAILED=()
t_slate=$(date +%s)

while IFS='|' read -r NAME EXTRA EPOCHS; do
    NAME="$(echo "${NAME:-}" | xargs)"
    [ -z "$NAME" ] && continue
    case "$NAME" in \#*) continue ;; esac
    EXTRA="$(echo "${EXTRA:-}" | xargs)"
    EPOCHS="$(echo "${EPOCHS:-}" | xargs)"

    if [ -f "${SCORE_DIR}/summary_${NAME}.json" ]; then
        echo "=== ${NAME}: already scored, skipping ==="
        continue
    fi

    t0=$(date +%s)
    echo
    echo "############################################################"
    echo "### ${NAME}  |  folds ${FOLDS}  |  epochs ${EPOCHS:-150}"
    echo "### flags: ${EXTRA:-<phase reference>}"
    echo "############################################################"

    if MAX_EPOCHS="${EPOCHS:-150}" BASE_ARGS="$PHASE_REF" \
       ./scripts/run_central_experiment.sh "$NAME" "$GPUS" "$FOLDS" "$EXTRA"; then
        $PY -u scripts/score_model_experiment.py \
            --label "$NAME" --folds "$FOLDS" --out_dir "$SCORE_DIR" \
            --stitched_dir "data/ensemble/exp/${NAME}/stitched" \
            || FAILED+=("${NAME} (scoring)")
    else
        echo "!!! ${NAME} FAILED — continuing with the rest of the slate"
        FAILED+=("${NAME} (training)")
    fi
    echo "### ${NAME} took $(( ($(date +%s) - t0) / 60 )) min"
done < "$SLATE"

echo
echo "=== slate done in $(( ($(date +%s) - t_slate) / 60 )) min ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'FAILED: %s\n' "${FAILED[@]}"
else
    echo "all experiments completed"
fi
