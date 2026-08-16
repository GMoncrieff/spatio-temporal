#!/usr/bin/env bash
# Promote a screened configuration: train all five folds, then run the whole downstream
# chain that decides whether it actually improved anything.
#
#   ./scripts/promote_model_experiment.sh <name> "<extra train flags>" [max_epochs] [members]
#
# A model change that improves a validation loss and not the scorecard or the per-member
# metric has improved nothing, so screening on two folds is never the verdict. This runs:
#
#   1. k=5 training + regional prediction + stitch          (~60 min)
#   2. the stratified central/width score on all five folds (~1 min)
#   3. run_region_loop.sh with SHAPE=measured WIDTHS=measured, which re-derives the
#      recalibration, the spectrum, the AR(1) coupling, the width factors and the marginal
#      shape from *this model's own* residuals                (~45 min at M=400)
#   4. the per-member metric, which is the honest test of the marginals
#
# Baselines to beat, from data/ensemble/exp/e5_all_k5: scorecard 101/127 (validation_hm)
# and 15/20 cells with the observation inside the member 5-95% range (member_distance_hm).

set -euo pipefail

NAME="${1:?usage: $0 <name> \"<extra train flags>\" [max_epochs] [members]}"
EXTRA="${2:-}"
EPOCHS="${3:-150}"
MEMBERS="${4:-400}"

PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
GPUS="${GPUS:-0,1}"
ROOT="data/ensemble/exp/${NAME}"
REGION_ROOT="data/ensemble/region/southern_africa"
SCORE_DIR="${SCORE_DIR:-data/ensemble/exp/scores}"
PHASE_REF="${PHASE_REF:---central_residual True --central_context True --monotone_quantile_width True --quantile_context True --histogram_weight 0 --checkpoint_monitor val_central_loss}"

echo "############ promoting ${NAME} to k=5 (${EPOCHS} epochs, M=${MEMBERS}) ############"

if [ ! -d "${ROOT}/stitched" ] || [ ! -f "${ROOT}/stitched/w2000_prediction_2020_central.tif" ]; then
    MAX_EPOCHS="$EPOCHS" BASE_ARGS="$PHASE_REF" \
        ./scripts/run_central_experiment.sh "$NAME" "$GPUS" 1,2,3,4,5 "$EXTRA"
else
    echo "=== ${NAME}: stitched rasters already present, skipping training ==="
fi

$PY -u scripts/score_model_experiment.py \
    --label "${NAME}_k5" --out_dir "$SCORE_DIR" --stitched_dir "${ROOT}/stitched"

SHAPE=measured WIDTHS=measured ./scripts/run_region_loop.sh "$NAME" "$MEMBERS"

$PY -u scripts/member_distance_relationship.py \
    --ensembles "${NAME}=${ROOT}/members.zarr" \
    --recal_dir "${ROOT}/recal_w" \
    --dist_raster "${REGION_ROOT}/covariates/w2000_dist_past_change.tif" \
    --members "$MEMBERS" --out_dir "${ROOT}/member_distance" \
    --wandb_group "model-${NAME}" \
    --wandb_run_name "member-wise realism · ${NAME}"

echo
echo "=== ${NAME} downstream ==="
$PY - "$NAME" <<'PYEOF'
import sys, pandas as pd
from pathlib import Path
name = sys.argv[1]
root = Path(f"data/ensemble/exp/{name}")
sc = pd.read_csv(root / "validation" / "scorecard.csv")
sc["fam"] = sc["id"].astype(str).str.split(".").str[0]
print(f"scorecard: {int(sc['pass'].sum())}/{int(sc['pass'].notna().sum())}   (baseline 101/127)")
print(sc.groupby("fam")["pass"].agg(["sum", "count"]).T.to_string())
st = root / "member_distance" / "member_distance_stats.csv"
if st.exists():
    m = pd.read_csv(st)
    inside = int(m["obs_inside_90pct_hi"].sum())
    print(f"per-member: {inside}/{len(m)} cells inside   (baseline 15/20)")
    print(f"mean |rank - M/2|: {(m['obs_rank_hi'] - m['M'] / 2).abs().mean():.1f}   (baseline 115.6)")
PYEOF
