#!/usr/bin/env bash
# After the ConvLSTM phase: shard the two quantile stores and push the products to Box.
#
# Runs unattended once packaging reports success, then stops. Nothing here touches the
# ensemble phase.
#
# The four rclone invocations are the operator's, verbatim, except that the third and fourth
# as supplied were missing the closing quote before `box:` -- bash would have joined source
# and destination into one argument and swallowed every flag into an unterminated string.
set -uo pipefail
cd /home/glenn/spatio-temporal
PKG=data/ensemble/logs/g_e1_pkg_chain.log
P=/mnt/hdd1/spatio-temporal/data/ensemble/products
Q="$P/quantiles"

count () { grep -c -- "$2" "$1" 2>/dev/null | head -1 || true; }

echo "[up] waiting for packaging to finish"
while true; do
  ok=$(count "$PKG" "packaging exited rc=0"); ok=${ok:-0}
  if grep -qE "ABORT|packaging exited rc=[1-9]" "$PKG" 2>/dev/null; then
    echo "[up] ABORT: packaging failed"; tail -20 "$PKG"; exit 3; fi
  [ "$ok" -ge 1 ] 2>/dev/null && break
  sleep 60
done
echo "[up] packaging done at $(date -Is)"

for r in hindcast forecast; do
  [ -d "$Q/${r}_qf.icechunk" ] || { echo "[up] ABORT: $Q/${r}_qf.icechunk missing"; exit 4; }
done

# The shards are a second copy of both stores. Refuse rather than fill the disk.
need=$(( $(du -sb "$Q/hindcast_qf.icechunk" | cut -f1) + $(du -sb "$Q/forecast_qf.icechunk" | cut -f1) ))
free=$(df -B1 --output=avail /mnt/hdd1 | tail -1)
echo "[up] shards need $(numfmt --to=iec $need), HDD has $(numfmt --to=iec $free) free"
if [ "$free" -lt $(( need + 50000000000 )) ]; then
  echo "[up] ABORT: not enough room for the shards plus 50 GB of headroom"; exit 5; fi

echo "[up] sharding at $(date -Is)"
./scripts/archive_icechunk_shards.sh "$Q/hindcast_qf.icechunk" "$Q/hindcast_tar" 2G || exit 6
./scripts/archive_icechunk_shards.sh "$Q/forecast_qf.icechunk" "$Q/forecast_tar" 2G || exit 7

echo
echo "[up] uploading at $(date -Is)"
RC=0
run () { echo; echo "[up] \$ $*"; "$@" || { echo "[up] rclone exited $?"; RC=1; }; }

run rclone copy "/mnt/hdd1/spatio-temporal/data/ensemble/products/cogs/hindcast/" "box:HM_forecasting/data/ensemble/global/products/cogs/hindcast/" --fast-list --transfers 8 --checkers 16 --progress --stats 30s --retries 10 --low-level-retries 20

run rclone copy "/mnt/hdd1/spatio-temporal/data/ensemble/products/cogs/forecast/" "box:HM_forecasting/data/ensemble/global/products/cogs/forecast/" --fast-list --transfers 8 --checkers 16 --progress --stats 30s --retries 10 --low-level-retries 20

run rclone copy "/mnt/hdd1/spatio-temporal/data/ensemble/products/quantiles/hindcast_tar/" "box:HM_forecasting/data/ensemble/global/products/quantiles/hindcast_tar/" --fast-list --transfers 8 --checkers 16 --progress --stats 30s --retries 10 --low-level-retries 20

run rclone copy "/mnt/hdd1/spatio-temporal/data/ensemble/products/quantiles/forecast_tar/" "box:HM_forecasting/data/ensemble/global/products/quantiles/forecast_tar/" --fast-list --transfers 8 --checkers 16 --progress --stats 30s --retries 10 --low-level-retries 20

# Not part of the operator's four commands: prove what landed matches what was sent. A copy
# that silently dropped a shard leaves an archive that cannot be restored, and nothing would
# say so until someone tried.
echo
echo "[up] verifying the remote against local"
for d in cogs/hindcast cogs/forecast quantiles/hindcast_tar quantiles/forecast_tar; do
  echo "[up] check ${d}"
  rclone check "${P}/${d}/" "box:HM_forecasting/data/ensemble/global/products/${d}/" \
    --one-way --fast-list 2>&1 | tail -4 || RC=1
done

echo
echo "[up] finished at $(date -Is) rc=${RC}"
exit "$RC"
