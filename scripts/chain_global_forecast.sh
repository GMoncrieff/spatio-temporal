#!/usr/bin/env bash
# Wave 3 is a single fold, so one GPU sits idle for ~4 h. The forward model depends on
# nothing in the hindcast, so it trains and predicts there instead.
#
# Memory: fold 5 predicting is 25.6 GiB and the forward model another 25.6 GiB once it
# reaches prediction. Two concurrent folds were measured at 55 GiB peak with 68 GiB spare,
# so this is the same envelope, not a new one.
set -uo pipefail
cd /home/glenn/spatio-temporal
CHAIN_LOG=data/ensemble/logs/g_e1_hind_chain.log
echo "[fc-chain] waiting for wave 3 to start"
until grep -q "launching wave 3" "$CHAIN_LOG" 2>/dev/null; do
  if grep -qE "ABORT|exited rc=[1-9]" "$CHAIN_LOG" 2>/dev/null; then
    echo "[fc-chain] ABORT: the hindcast chain failed before wave 3"; exit 3
  fi
  sleep 60
done
echo "[fc-chain] wave 3 up at $(date -Is); starting the forward model on GPU 1"
./scripts/run_global_dist_forecast.sh 1
rc=$?
echo "[fc-chain] forecast exited rc=${rc} at $(date -Is)"
exit "$rc"
