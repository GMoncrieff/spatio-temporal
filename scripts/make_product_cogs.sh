#!/usr/bin/env bash
# Publish the three raw (uncalibrated) ConvLSTM products as COGs.
#
# Sources are the frozen g1_foldb4 rasters. Set C's header disagrees with what it stores --
# it declares nodata 3.4e38 while filling with NaN -- so it is republished with --dst_nodata
# nan. Sets A and B already carry nodata=NaN from the stitcher and need no override.
set -euo pipefail

PY=${PY:-/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python}
SRC=${SRC:-/mnt/hdd1/spatio-temporal/data/ensemble/exp/g1_foldb4}
G=${G:-/mnt/hdd1/spatio-temporal/data/conv_update/global}

echo "=== A: out-of-fold hindcast (out-of-sample; the set to score against) ==="
$PY -u scripts/make_cogs.py --src_dir "$SRC/stitched" --out_dir "$G/cogs_hindcast" \
    --prefix hm_hindcast --base_year 2000 --years 2005,2010,2015,2020 \
    --src_pattern "w{base}_prediction_{year}_{q}.tif"

echo "=== B: fold-mean hindcast (in-sample at every pixel; display only) ==="
$PY -u scripts/make_cogs.py --src_dir "$SRC/mean_run/stitched_mean" --out_dir "$G/cogs_hindcast_mean" \
    --prefix hm_hindcast_mean --base_year 2000 --years 2005,2010,2015,2020 \
    --src_pattern "w{base}_prediction_{year}_{q}.tif"

echo "=== C: forecast 2025-2040 (production model, no fold, no mosaic) ==="
$PY -u scripts/make_cogs.py --src_dir "$SRC/forecast_preds" --out_dir "$G/cogs_forecast" \
    --prefix hm_forecast --base_year 2020 --years 2025,2030,2035,2040 \
    --src_pattern "prediction_{year}_{q}_blended.tif" --dst_nodata nan

echo "=== all 36 COGs written and verified ==="
