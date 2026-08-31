#!/usr/bin/env bash
# Publish the ConvLSTM deliverables, then shard and upload them. One chain this time, so
# there is a single place the sequence is described.
#
# Re-armed after the forward model's first attempt died writing a classic TIFF at the 4 GiB
# ceiling. It now re-predicts from the checkpoint it had already trained; the completeness
# check in run_global_dist_forecast.sh counts valid pixels rather than testing for a file,
# because the truncated rasters existed and read back as finite zeros.
set -uo pipefail
cd /home/glenn/spatio-temporal
SS=data/ensemble/logs/g_e1_ss_chain.log
FC=data/ensemble/logs/g_e1_fc_chain2.log
P=/mnt/hdd1/spatio-temporal/data/ensemble/products
Q="$P/quantiles"
count () { grep -c -- "$2" "$1" 2>/dev/null | head -1 || true; }

echo "[pub] waiting for the scorecard and the forward model"
while true; do
  ss_ok=$(count "$SS" "scorecard exited rc=0"); ss_ok=${ss_ok:-0}
  fc_ok=$(count "$FC" "OK: 16 forecast rasters"); fc_ok=${fc_ok:-0}
  grep -qE "ABORT|scorecard exited rc=[1-9]" "$SS" 2>/dev/null && {
    echo "[pub] ABORT: stitch/score failed"; exit 3; }
  grep -q "REFUSING" "$FC" 2>/dev/null && {
    echo "[pub] ABORT: the forward model failed its completeness check"; exit 4; }
  { [ "$ss_ok" -ge 1 ] && [ "$fc_ok" -ge 1 ]; } 2>/dev/null && break
  sleep 60
done
echo "[pub] both branches done at $(date -Is)"

echo "[pub] packaging"
./scripts/package_global_products.sh all || { echo "[pub] ABORT: packaging failed"; exit 5; }

for r in hindcast forecast; do
  [ -d "$Q/${r}_qf.icechunk" ] || { echo "[pub] ABORT: $Q/${r}_qf.icechunk missing"; exit 6; }
done
need=$(( $(du -sb "$Q/hindcast_qf.icechunk" | cut -f1) + $(du -sb "$Q/forecast_qf.icechunk" | cut -f1) ))
free=$(df -B1 --output=avail /mnt/hdd1 | tail -1)
echo "[pub] shards need $(numfmt --to=iec $need), HDD has $(numfmt --to=iec $free) free"
[ "$free" -ge $(( need + 50000000000 )) ] || { echo "[pub] ABORT: not enough room"; exit 7; }

echo "[pub] sharding at $(date -Is)"
./scripts/archive_icechunk_shards.sh "$Q/hindcast_qf.icechunk" "$Q/hindcast_tar" 2G || exit 8
./scripts/archive_icechunk_shards.sh "$Q/forecast_qf.icechunk" "$Q/forecast_tar" 2G || exit 9

echo
echo "[pub] uploading at $(date -Is)"
RC=0
run () { echo; echo "[pub] \$ $*"; "$@" || { echo "[pub] rclone exited $?"; RC=1; }; }

run rclone copy "/mnt/hdd1/spatio-temporal/data/ensemble/products/cogs/hindcast/" "box:HM_forecasting/data/ensemble/global/products/cogs/hindcast/" --fast-list --transfers 8 --checkers 16 --progress --stats 30s --retries 10 --low-level-retries 20

run rclone copy "/mnt/hdd1/spatio-temporal/data/ensemble/products/cogs/forecast/" "box:HM_forecasting/data/ensemble/global/products/cogs/forecast/" --fast-list --transfers 8 --checkers 16 --progress --stats 30s --retries 10 --low-level-retries 20

run rclone copy "/mnt/hdd1/spatio-temporal/data/ensemble/products/quantiles/hindcast_tar/" "box:HM_forecasting/data/ensemble/global/products/quantiles/hindcast_tar/" --fast-list --transfers 8 --checkers 16 --progress --stats 30s --retries 10 --low-level-retries 20

run rclone copy "/mnt/hdd1/spatio-temporal/data/ensemble/products/quantiles/forecast_tar/" "box:HM_forecasting/data/ensemble/global/products/quantiles/forecast_tar/" --fast-list --transfers 8 --checkers 16 --progress --stats 30s --retries 10 --low-level-retries 20

echo
echo "[pub] verifying the remote against local"
for d in cogs/hindcast cogs/forecast quantiles/hindcast_tar quantiles/forecast_tar; do
  echo "[pub] check ${d}"
  rclone check "${P}/${d}/" "box:HM_forecasting/data/ensemble/global/products/${d}/" \
    --one-way --fast-list 2>&1 | tail -4 || RC=1
done
echo
echo "[pub] CONVLSTM PHASE COMPLETE at $(date -Is) rc=${RC}"
exit "$RC"
