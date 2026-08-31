#!/usr/bin/env bash
# Run the remaining hindcast waves back to back so the GPUs never idle at a handover.
#
# Waves, not one five-fold call, because run_hindcast_folds.py verifies nothing until every
# fold it was given has finished: a wave boundary is where the loss-weight and lever
# fingerprints get read back, and a wave that trained the wrong configuration must stop the
# next one rather than be discovered five folds later.
set -uo pipefail
cd /home/glenn/spatio-temporal
WAIT_PID="${WAIT_PID:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[chain] waiting for pid ${WAIT_PID} (wave 1) to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[chain] wave 1 process gone at $(date -Is)"
fi
for f in data/ensemble/logs/hindcast_fold1_g_e1_hind.log data/ensemble/logs/hindcast_fold2_g_e1_hind.log; do
  n=$(grep -c "Total time:" "$f")
  [ "$n" -ge 4 ] || { echo "[chain] ABORT: $f has only ${n}/4 windows"; exit 3; }
done
grep -q "REFUSING to continue" data/ensemble/logs/g_e1_hind_wave1.log && {
  echo "[chain] ABORT: wave 1 failed its own fingerprint check"; exit 4; }
echo "[chain] wave 1 verified, launching wave 2 (folds 3,4) at $(date -Is)"
./scripts/run_global_dist_hindcast.sh train 3,4 > data/ensemble/logs/g_e1_hind_wave2.log 2>&1
rc=$?
echo "[chain] wave 2 exited rc=${rc} at $(date -Is)"
[ "$rc" -eq 0 ] || exit "$rc"
echo "[chain] launching wave 3 (fold 5) at $(date -Is)"
./scripts/run_global_dist_hindcast.sh train 5 > data/ensemble/logs/g_e1_hind_wave3.log 2>&1
rc=$?
echo "[chain] wave 3 exited rc=${rc} at $(date -Is)"
exit "$rc"
