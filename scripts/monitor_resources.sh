#!/usr/bin/env bash
# Sample the four resources the global production run can die on and append them to a CSV.
#
#   ./scripts/monitor_resources.sh <out.csv> [interval_s]
#
# Sample faster than the thing you are watching for. The Africa OOM went from steady to
# SIGKILLed in under 60 s and a 120 s poll saw nothing, so the default here is 10 s. SIGKILL
# leaves no traceback: this CSV is the only evidence of what the working set was doing.
#
# Columns: iso time, seconds since start, root GiB free, HDD GiB free, DRAM GiB used,
# DRAM GiB available, swap GiB used, per-GPU MiB used, per-GPU utilisation, 1-min load,
# and the RSS of the largest python process.
set -uo pipefail
OUT="${1:?usage: $0 <out.csv> [interval_s]}"
INT="${2:-10}"
if [ ! -f "$OUT" ]; then
  echo "iso,t_s,root_free_gib,hdd_free_gib,dram_used_gib,dram_avail_gib,swap_used_gib,gpu0_mib,gpu1_mib,gpu0_util,gpu1_util,load1,max_py_rss_gib,n_py" > "$OUT"
fi
T0=$(date +%s)
while true; do
  now=$(date +%s)
  read -r rootf hddf <<<"$(df -BG --output=avail / /mnt/hdd1 2>/dev/null | tail -n +2 | tr -d 'G' | tr '\n' ' ')"
  read -r used avail <<<"$(free -g | awk '/^Mem:/ {print $3, $7}')"
  swp=$(free -g | awk '/^Swap:/ {print $3}')
  gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//')
  gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//')
  ld=$(awk '{print $1}' /proc/loadavg)
  # ps rss is in KiB; only real python workers, not this script's own pipeline
  read -r rss npy <<<"$(ps -eo rss,comm --no-headers 2>/dev/null | awk '$2 ~ /^python/ {n++; if ($1>m) m=$1} END {printf "%.2f %d", m/1048576, n+0}')"
  echo "$(date -Is),$((now-T0)),${rootf:-0},${hddf:-0},${used:-0},${avail:-0},${swp:-0},${gm:-,},${gu:-,},${ld},${rss:-0},${npy:-0}" >> "$OUT"
  sleep "$INT"
done
