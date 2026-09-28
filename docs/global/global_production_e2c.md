# The global production run — E2c, both products

E2c is the production model. This document records its configuration, the inputs built for it,
how each setting is verified, its out-of-sample scores, the delivered products and what the
run cost. The model itself is specified in `docs/methodology.pdf`.

    MODEL=E2c ALLOW_GLOBAL=1 ./scripts/run_global_model.sh <stage>

Run 2026-09-26 on `main`. Detailed scorecard, base 2000 → 2020:
https://claude.ai/artifact/27KxeCDVnCCQ9ULnY9pNv7
(`docs/global/scores/scorecard_detailed_g_E2c_hind.html`).

## Configuration

    --head_family pwl --free_scale True --mu_mse_weight 0.0 --kernel_size 5 --terrain_covariates True

with the normalisation sidecar `data/conv_spline/norm_stats_E2c.json` and, for the k-fold
hindcast, the fold mask `data/raw/hm_global/fold_mask_b4_land_1000.tif`. Every other setting
is an argparse default of `scripts/train_lightning.py`.

- **Trunk:** ConvLSTM, 4 layers × 64 hidden, 5×5 kernels; receptive radius 12 px
  (`r = (k-1)/2 × (Σd + (T-1)·max d)` with T = 3 input steps).
- **Inputs:** 41 channels per step — 11 dynamic, 10 static (7 layers plus slope, sin(aspect),
  cos(aspect)), 12 neighbourhood context, 8 location.
- **Head:** piecewise-linear quantile function on 15 knots, free scale, no tail parameters,
  15 params/horizon.
- **Training:** closed-form CRPS only; 150 epochs; prediction uses the mean of the last 20
  epochs' weights. 3,365,444 parameters are used.

The code's defaults are not this configuration, and leaving out any of the flags, the sidecar
or the fold mask does not fail — it silently builds a different model. Run E2c through the
runner, which sets all of them and checks each one.

## Inputs built for this run

### Slope and aspect

`scripts/prepare_terrain.py`, 2.0 min for the globe:

    hm_static_terrain_slope_1000.tiff    slope, degrees
    hm_static_terrain_aspsin_1000.tiff   sin(aspect)   aspect = downslope bearing,
    hm_static_terrain_aspcos_1000.tiff   cos(aspect)   clockwise from north

Horn (1981) 3×3 gradient on `hm_static_ele_1000.tiff`, using each row's true metre spacing on
the WGS84 ellipsoid — a pixel is 1.00 km wide at the equator and ~0.12 km at 83 N. Longitude
wraps at ±180. Aspect is 0/0 on flat ground, which includes the sea (the DEM fills it with 0 m).
The DEM's −32768 fill (every row south of 56 S) and every pixel whose 3×3 window touches it are
NaN, declared as nodata: exactly 62,280,000 px, the fill rows plus one halo row.

Checked on the written rasters: against slope from plain central differences on geodesic
spacing — an independent implementation — they correlate at 0.93–0.99 on four test windows;
`sin² + cos²` is 1 on every sloped pixel; the maximum slope is 76.4°. Over land the slope has
mean 2.0° and standard deviation 3.4°.

The `hm_static_ele_slope/asp_*` rasters also present in `data/raw/hm_global/` are not derived
from this DEM (correlation 0.51–0.89 with the slope above) and are not used.

### Normalisation statistics

`scripts/compute_norm_stats.py --terrain_covariates`, 3.6 min: exact streaming moments over HM
land pixels for HM, the ten dynamic covariates and the ten static channels. HM:
μ = 0.087103, σ = 0.149811 over 1,292,270,721 px (seven years). Elevation: μ = 675 m,
σ = 845 m, with the 558 filled land pixels excluded. The sidecar also declares
`static_nodata: {"hm_static_ele_1000.tiff": -32768}`, and the reader
(`torchgeo_dataloader.prepare_static`, shared by training and prediction) turns that fill into
NaN and then 0 m, so it is read as sea level.

### Fold mask

`scripts/create_validity_mask.py --folds_only --k 5 --fold_block_chips 4 --fold_min_valid_px 1`:
a fold id for every 128 px chip holding any land, grouped into 512 px super-blocks and assigned
greedily with seed 42. 14,631 chips, 35.0% of the grid; 2,922–2,933 chips and 36.5–38.1 M land
pixels per fold. Every land pixel has a fold except 36,262 in the partial chip strip at the
grid edge. Each training pool holds 8,774–8,786 positions, and each validation fold 42–50 tiles
at stride 1024 px.

## How each setting is verified, out of the run's own log

The runner's model table names E2c's flags, family, parameter count, fold mask and sidecar
together. Eight fingerprints are read back from every fold's log and the forward model's log,
each printed off the constructed module rather than the flags:

- loss weights (all four);
- context wiring: 12 channels into the trunk, none into the heads;
- `Spline head: family pwl, ..., free scale, 15 params/horizon`;
- `Trunk: ConvLSTM 4 layers x 64, kernel 5, ..., receptive radius 12 px`;
- `Static channels: 10 (...); module 10`, with the three terrain rasters in order;
- `Loaded normalization stats from data/conv_spline/norm_stats_E2c.json` — the sidecar
  actually read (a missing sidecar would otherwise be re-sampled over the whole grid and written
  to that path, so `require_norm_stats` refuses to start without it);
- row banding (512 rows);
- weight averaging (mean of the last 20 epochs, predicted).

The smoke receipt pins the code, the flags and the inputs (the sidecar and fold mask by bytes,
the terrain rasters by name, size and mtime). All five folds and the forward model passed all
eight checks, and the forward model also the production-mode banner. The first ConvLSTM layer's
weights are `(256, 105, 5, 5)`: 41 inputs + 64 hidden, 5×5.

## Results, out of sample on all 184,573,321 land pixels

Base 2000 → 2005/2010/2015/2020; every pixel predicted by the fold model that never trained on
it; skill against persistence.

| h | CRPS | CRPS skill | RMSE | RMSE skill | cov50 | cov80 | cov95 | PIT mean | PIT KS |
|---|---|---|---|---|---|---|---|---|---|
| +5 | 0.002977 | 0.1551 | 0.013322 | 0.0406 | 0.421 | 0.773 | 0.950 | 0.519 | 0.066 |
| +10 | 0.005318 | 0.2608 | 0.020326 | 0.1875 | 0.427 | 0.762 | **0.917** | 0.543 | 0.087 |
| +15 | 0.007256 | 0.2988 | 0.026441 | 0.2293 | 0.449 | 0.760 | 0.953 | 0.541 | 0.087 |
| +20 | 0.008949 | 0.2987 | 0.031190 | 0.2406 | 0.482 | 0.772 | 0.955 | 0.540 | 0.085 |

`far_tail_excess` is 7.64 / 7.73 / 8.09 / 8.52 (1 = calibrated).

### By distance to past change

CRPS skill at +20 yr runs 0.419 / 0.276 / 0.190 / 0.114 / 0.097 / 0.263 over the bands
0–1 / 1–3 / 3–10 / 10–30 / 30–100 / >100 px. Beyond 100 px (33,565,179 px), by horizon:

| >100 px | +5 | +10 | +15 | +20 |
|---|---|---|---|---|
| cov95 | 0.981 | **0.861** | 0.941 | 0.935 |
| CRPS skill | −0.147 | **−1.445** | 0.347 | 0.263 |
| RMSE skill | −0.014 | −0.166 | 0.058 | 0.008 |
| P(u > 0.999) | 0.0004 | 0.0006 | 0.0012 | 0.0021 |

At 30–100 px the +10 yr coverage is 0.897.

### Known limitations, which travel with the product

- **Beyond 100 px from past change at +10 yr, the 95% interval covers 86.1% of observations and
  the forecast loses to persistence on CRPS (skill −1.445).** Pooled, this is the +10 yr coverage
  of 0.917. Far-field coverage does not change monotonically with lead time, because nothing
  constrains how the interval width grows from one horizon to the next. Do not describe E2c's
  far-field intervals as calibrated.
- The far-field point forecast does not beat persistence (RMSE skill −0.014 at +5 yr,
  +0.008 at +20 yr).
- The PIT mean is 0.52–0.54, with recurring spikes near 0.22, 0.52, 0.77 and 0.95; the
  subsampled PIT over 48,911,382 px reads 0.5534.
- The central intervals are too narrow: cov50 0.42–0.48, cov80 0.76–0.77.
- The lower far tail is over-populated: `P(u<0.001)` 0.0116–0.0125 against 0.001.

## The products

Built and verified at `/mnt/hdd1/spatio-temporal/data/conv_spline/products/E2c/`.

| | COGs | icechunk |
|---|---|---|
| hindcast (base 2000 → 2005/2010/2015/2020) | 20 files, 10 GB | 68 GB, `[4, 64, 17111, 40000]` uint16 |
| forecast (base 2020 → 2025/2030/2035/2040) | 20 files, 11 GB | 70 GB, same shape |

158 GB total. Chunks `(1, 64, 256, 320)`, shards `(1, 64, 1024, 1280)`, 2,650 files per store
including manifests. Per-year COGs are 205–746 MB, `gt04` smallest and `upper` largest.

`verify_products.py` is green on both: passthrough bit-exact (`max |d| = 0`) on every year, the
exceedance round trip within 1.95e-07 on every year, and on both stores `code mismatches 0` and
`fill/finite collisions 0`. The printed `round trip max |err| 1.53e-05` is one full uint16 step:
a source value of exactly 1.0 is clamped to code 65534 by design, because 65535 is the fill.

The icechunk arrays declare no `dimension_names`, so `xarray.open_zarr` cannot open them; read
them with `zarr` directly.

## Reproducing a prediction from the checkpoints

    python scripts/train_lightning.py \
      --checkpoint models/production/E2c_global/final_foldNone_3573819.ckpt \
      --kernel_size 5 --terrain_covariates True --norm_stats_json data/conv_spline/norm_stats_E2c.json \
      --max_epochs 0 --run_large_area_prediction True --run_full_set_evaluation False \
      --predict_region config/region_to_predict_small.geojson --predict_final_year 2040

Checked 2026-09-27: 0 warm-started convs, `kernel 5`, `Static channels: 10 ... module 10`,
51 s. Against the delivered forecast over the same region the interior matches to a median of
5e-07 HM (p99 7.7e-04, max 0.025), not bit-identically: a region's tiles start at its own
corner, here 4 and 9 px off the global 64 px tile grid, so the overlap blending differs.

## Measured costs — two RTX A5000, 125 GB RAM

| stage | wall clock | peak python RSS |
|---|---|---|
| terrain rasters | 2.0 min | — |
| normalisation statistics (84 rasters, exact) | 3.6 min | — |
| smoke | 63 min | 34.8 GiB |
| hindcast, 5 folds (2 GPUs) | **292.0 min**; folds 96.7 / 97.5 / 96.5 / 96.2 / 98.8 | 35.9 GiB |
| forward model train + predict (1 GPU, overlapping fold 5 and the stitch) | 3 h 57 | 35.9 GiB |
| stitch, 16 rasters | 160.4 min | 35.9 GiB |
| score, 4 window-years (alone) | 5 h 01 | **102.0 GiB** |
| detailed scorecard | 3 min | — |
| export hindcast + forecast, in parallel | 2 h 52 | 25.8 GiB each |

The chain ran unattended from the end of the smoke to the end of the exports in 15 h 31, stopping
at the first failure had there been one (`data/conv_spline/logs/global/chain_E2c.log`). The
scorer needs the machine to itself. Disk: `g_E2c_hind` 306 GB, `g_E2c_fc` 129 GB, smoke 3.1 GB,
products 158 GB.

## Files

- Checkpoints: `models/production/E2c_global/` — `final_fold{1..5}_*.ckpt` and the forward
  model `final_foldNone_3573819.ckpt`, with `norm_stats_E2c.json` (must travel with them) and
  the fold manifest. Provenance: `docs/global/checkpoints/E2c_global_README.md`. Not in git.
- Inputs: `data/conv_spline/norm_stats_E2c.json`, `data/raw/hm_global/fold_mask_b4_land_1000.tif`,
  `data/raw/hm_global/hm_static_terrain_{slope,aspsin,aspcos}_1000.tiff`.
- Scores: `docs/global/scores/*_g_E2c_hind.*` (copied from `data/conv_spline/scores/global/`).
- Logs: `data/conv_spline/logs/global/*E2c*`.
