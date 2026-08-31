#!/usr/bin/env bash
# The long-term ConvLSTM deliverables, in one place.
#
#   ./scripts/package_global_products.sh [hindcast|forecast|all]
#
# Two formats, because they answer different questions:
#
#   COGs      the triple (lower, central, upper) per year -- what a GIS opens, streams over
#             HTTP and draws. Twelve per product.
#   icechunk  the full quantile function as one labelled array
#             (time, quantile, latitude, longitude) -- the model's actual forecast. Sixty-four
#             GeoTIFF bands whose meaning lives in a text tag is a working format, not a
#             delivery one.
#
# Only the w2000 window ships from the hindcast. All ten (base, target) pairs are stitched,
# scored and fitted against, but only w2000 reaches +20 yr, so it is the one window whose
# store spans the full 5/10/15/20 horizon set. The other six are measurement inputs.
set -uo pipefail
cd /home/glenn/spatio-temporal
WHAT="${1:-all}"
PY="${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}"
HIND=/mnt/hdd1/spatio-temporal/data/ensemble/exp/g_e1_hind
FC=/mnt/hdd1/spatio-temporal/data/ensemble/exp/g_e1_fc
OUT="${OUT:-/mnt/hdd1/spatio-temporal/data/ensemble/products}"
mkdir -p "$OUT/cogs/hindcast" "$OUT/cogs/forecast" "$OUT/quantiles"

do_hindcast () {
  echo "=== hindcast triples -> COGs (w2000, 4 years) ==="
  $PY -u scripts/make_cogs.py \
      --src_dir "${HIND}/stitched" --out_dir "${OUT}/cogs/hindcast" \
      --prefix hm_hindcast_e1_w2000 --base_year 2000 --years 2005,2010,2015,2020 \
      --src_pattern "w{base}_prediction_{year}_{q}.tif" --overwrite \
      --verify_windows 48 --min_verified_px 2000000 || return 1
  echo "=== hindcast quantile functions -> icechunk ==="
  rm -rf "${OUT}/quantiles/hindcast_qf.icechunk"
  $PY -u scripts/package_qf_icechunk.py \
      --src_dir "${HIND}/stitched" --mode hindcast --base_years 2000 \
      --out "${OUT}/quantiles/hindcast_qf.icechunk" || return 1
}

do_forecast () {
  echo "=== forecast triples -> COGs (2025-2040) ==="
  $PY -u scripts/make_cogs.py \
      --src_dir "${FC}/preds" --out_dir "${OUT}/cogs/forecast" \
      --prefix hm_forecast_e1_w2020 --base_year 2020 --years 2025,2030,2035,2040 \
      --src_pattern "prediction_{year}_{q}_blended.tif" --overwrite \
      --verify_windows 48 --min_verified_px 2000000 || return 1
  echo "=== forecast quantile functions -> icechunk ==="
  rm -rf "${OUT}/quantiles/forecast_qf.icechunk"
  $PY -u scripts/package_qf_icechunk.py \
      --src_dir "${FC}/preds" --mode forecast --base_year 2020 \
      --out "${OUT}/quantiles/forecast_qf.icechunk" || return 1
}

rc=0
case "$WHAT" in
  hindcast) do_hindcast || rc=1 ;;
  forecast) do_forecast || rc=1 ;;
  all)      do_hindcast && do_forecast || rc=1 ;;
  *) echo "usage: $0 [hindcast|forecast|all]" >&2; exit 2 ;;
esac

echo
echo "=== products under ${OUT} ==="
for d in hindcast forecast; do
  echo "  ${OUT}/cogs/${d}:"
  ls -la "${OUT}/cogs/${d}" 2>/dev/null | tail -n +4 | awk '{printf "    %-44s %8.2f GB\n", $9, $5/1e9}'
done
du -sh "${OUT}/cogs"/* 2>/dev/null | sed 's/^/  /'
du -sh "${OUT}/quantiles"/*.icechunk 2>/dev/null | sed 's/^/  /'
df -h /mnt/hdd1 | tail -1 | sed 's/^/  /'
exit $rc
