#!/usr/bin/env bash
# Last step of the ConvLSTM phase: publish the deliverables once the hindcast has been
# stitched and scored and the forward model has finished predicting.
#
# Waits on both branches because they are independent -- fold 5 and the forward model run
# concurrently on separate GPUs -- and packages nothing until each reports success. Nothing
# here deletes anything: preds/ is 200 GB and regenerable only by re-running twenty hours of
# prediction, so that is not a side effect of publishing.
set -uo pipefail
cd /home/glenn/spatio-temporal
SS=data/ensemble/logs/g_e1_ss_chain.log
FC=data/ensemble/logs/g_e1_fc_chain.log

# grep -c prints 0 AND exits 1 when a file exists with no match, so `|| echo 0` appends a
# second line and every numeric test then errors out while the loop spins forever. Count
# with a form that always yields exactly one integer.
count () { grep -c -- "$2" "$1" 2>/dev/null | head -1 || true; }

echo "[pkg] waiting for the stitch+score chain and the forward model"
while true; do
  ss_ok=$(count "$SS" "scorecard exited rc=0"); ss_ok=${ss_ok:-0}
  fc_ok=$(count "$FC" "forecast exited rc=0");  fc_ok=${fc_ok:-0}
  if grep -qE "ABORT|exited rc=[1-9]" "$SS" 2>/dev/null; then
    echo "[pkg] ABORT: the stitch/score chain failed"; tail -20 "$SS"; exit 3; fi
  if grep -qE "ABORT|exited rc=[1-9]" "$FC" 2>/dev/null; then
    echo "[pkg] ABORT: the forward model failed"; tail -20 "$FC"; exit 4; fi
  if [ "$ss_ok" -ge 1 ] 2>/dev/null && [ "$fc_ok" -ge 1 ] 2>/dev/null; then break; fi
  sleep 60
done
echo "[pkg] both branches done at $(date -Is); publishing"
./scripts/package_global_products.sh all
rc=$?
echo "[pkg] packaging exited rc=${rc} at $(date -Is)"
exit "$rc"
