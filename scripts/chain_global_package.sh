#!/usr/bin/env bash
# Last step of the ConvLSTM phase: publish the deliverables once the hindcast has been
# stitched and scored and the forward model has finished predicting.
#
# Waits on both branches because they are independent -- fold 5 and the forward model run
# concurrently on separate GPUs -- and packages nothing until each has reported success.
# Nothing here deletes anything: preds/ is 200 GB and regenerable only by re-running twenty
# hours of prediction, so that decision is not a side effect of publishing.
set -uo pipefail
cd /home/glenn/spatio-temporal
SS=data/ensemble/logs/g_e1_ss_chain.log
FC=data/ensemble/logs/g_e1_fc_chain.log

echo "[pkg] waiting for the stitch+score chain and the forward model"
while true; do
  ss_ok=$(grep -c "scorecard exited rc=0" "$SS" 2>/dev/null || echo 0)
  fc_ok=$(grep -c "forecast exited rc=0" "$FC" 2>/dev/null || echo 0)
  if grep -qE "ABORT|exited rc=[1-9]" "$SS" 2>/dev/null; then
    echo "[pkg] ABORT: the stitch/score chain failed"; tail -20 "$SS"; exit 3; fi
  if grep -qE "ABORT|exited rc=[1-9]" "$FC" 2>/dev/null; then
    echo "[pkg] ABORT: the forward model failed"; tail -20 "$FC"; exit 4; fi
  [ "$ss_ok" -ge 1 ] && [ "$fc_ok" -ge 1 ] && break
  sleep 60
done
echo "[pkg] both branches done at $(date -Is); publishing"
./scripts/package_global_products.sh all
rc=$?
echo "[pkg] packaging exited rc=${rc} at $(date -Is)"
exit "$rc"
