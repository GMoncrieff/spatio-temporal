# Fitting and running the model

Everything needed to reproduce this branch's global products from raw data: environment, input
requirements, and every script in order.

This is the model-only path. The `ensemble` branch adds a recalibration layer and a 400-member
ensemble on top of the same ConvLSTM; **none of that is on this branch**, so the products here
are the model's own central estimate and 2.5/97.5 quantile heads, uncalibrated. Where a step
below has a calibration counterpart on that branch, it is called out.

The model itself is documented in [`model_architecture.md`](model_architecture.md), and what
changed relative to `main` — including why `main` checkpoints will not load — in
[`model_update.md`](model_update.md).

---

## 0. Environment and hardware

```bash
# Conda env — NOT base
PY=/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python
```

| requirement | value |
|---|---|
| GPUs | 2 × 24 GB (RTX A5000). Prediction fits comfortably on one. |
| RAM | ~15 GB peak. Nothing on this branch holds the whole grid in float64. |
| disk (fast) | ~50 GB for code, checkpoints, small artifacts |
| disk (bulk) | ~90 GB for a full regeneration; 27 GB for the COGs alone |
| GDAL CLI | `gdal_translate` and `gdalinfo` on PATH, for COG creation |
| experiment tracking | Weights & Biases (pass `--disable_wandb` to skip) |

### 0.1 Input data

All rasters are on one grid: **17,111 × 40,000 px, EPSG:4326, 0.009° (~1 km)**, origin
(−180, 83.997). Any mismatch in transform or shape fails loudly at the first read.

Under `data/raw/hm_global/`:

| input | count | note |
|---|---|---|
| `HM_{year}_AA_1000.tiff` | 7 | the HM index itself, 1990–2020 in 5-yr steps. **This is the target.** |
| `HM_{year}_{AG,BU,EX,FR,HI,NS,PO,TI,gdp,population}_1000.tiff` | 10 × 7 | component layers — the 11 dynamic input channels are these plus AA |
| `hm_static_*.tiff` | 11 | elevation, slope, aspect (sin/cos), TPI, dpi/dsi, precipitation, temperature (mean/min), IUCN strict and non-strict protection |
| `ecoregion_id_1000.tif` + `ecoregion_lookup.csv` | 1 + 1 | 804 ecoregions with biome/realm lookup |

Years must be 1990, 1995, 2000, 2005, 2010, 2015, 2020. The hindcast uses input windows ending
2000/2005/2010/2015; the forward product uses 2010/2015/2020.

### 0.2 Derived inputs — build these once

**Change-context rasters** (two bands per window: signed past change, and distance to nearest
past change > 0.01). These *must* be computed on the full raster — deriving them inside a 128 px
training chip makes the 100 px radius saturate into "is there any change in this chip".

```bash
$PY scripts/prepare_change_context.py --out_dir data/raw/hm_global
# → change_context_w{2000,2005,2010,2015,2020}_1000.tif   (~2.2 GB each)
```

They are not optional: `--central_context` and `--quantile_context` read band 1 and band 2 of
these, and they are what makes the central head's input 72 channels rather than 64.

**Split mask and fold mask.** The split mask is the 70/10/10/10 train/val/test/calib partition
used by the production model. The fold mask carries five contiguous 512 px territories for
cross-validation — `--fold_block_chips 4` is what makes them 4 × 128 = 512 px, which must exceed
the ~300 px residual correlation length. At `--fold_block_chips 1` the folds are a 128 px
checkerboard whose tile sits *inside* that correlation length, so every held-out tile is ringed
by trained-on tiles and held-out skill reads optimistically.

```bash
$PY scripts/create_validity_mask.py                       # → split_mask_1000.tif
$PY scripts/create_validity_mask.py --folds_only --k 5 \
      --fold_block_chips 4 \
      --fold_mask_out data/raw/hm_global/fold_mask_b4_1000.tif
```

**Normalisation statistics.** `data/norm_stats.json` is written by the first training run and
reused by every later one as a sidecar, so all folds and the production model share one
normalisation. It is **required at prediction time too**: `hm_mean`/`hm_std` are plain
attributes, absent from the `.ckpt`, and without them inference builds the context at a
different scale than training did.

**Region root** (only needed for §4's diagnostics, which read the distance covariates):

```bash
$PY scripts/make_global_region_root.py --out_root data/ensemble/region/global --verify
```

### 0.3 Disk layout

Bulk artifacts must not land on the root filesystem:

```bash
G=/mnt/hdd1/spatio-temporal/data/conv_update/global
mkdir -p $G && ln -sfn $G data/global
```

| artifact | size |
|---|---|
| per-fold prediction rasters | 22 GB (holdout, all windows) / 24 GB (unrestricted, w2000 only) |
| stitched hindcast | 16 GB (all windows) / 6.4 GB (w2000 only) |
| forecast rasters | 6.1 GB |
| COGs (3 sets of 12) | 27 GB |

---

## The shipped products

The COGs on this branch were built from the frozen `g1_foldb4` checkpoints — the same six
models the `ensemble` branch trained on global chips — rather than from a retrain. Training is
not deterministic on this box, so a retrain gives a different model with no measured benefit,
and every number carried over from that branch would stop being comparable. What this branch's
code *is* proven to do is reproduce those models' rasters exactly; see §6.

`data/conv_update/PROVENANCE.json` records the checkpoints, the flag string, the source rasters
and the per-set sample status.

```bash
CKPTS="1=./spatio-temporal-convlstm/qnr2xa1i/checkpoints/epoch=108-step=1417.ckpt,\
2=./spatio-temporal-convlstm/rg7mzmmh/checkpoints/epoch=114-step=1495.ckpt,\
3=./spatio-temporal-convlstm/9ywng4jo/checkpoints/epoch=126-step=1651.ckpt,\
4=./spatio-temporal-convlstm/lw5lxhip/checkpoints/epoch=122-step=1599.ckpt,\
5=./spatio-temporal-convlstm/s064xfza/checkpoints/epoch=124-step=1625.ckpt"

ARCH="--central_residual True --central_context True \
      --monotone_quantile_width True --quantile_context True"
```

> **`ARCH` is not optional, and "no extra flags" is not the production architecture — it is
> argparse defaults.** `--central_residual` defaults to False, under which the central head
> predicts absolute HM and reconstructs the baseline through the trunk. A k=5 run launched on
> defaults scored h=5 skill −0.50 against +0.13 with these flags, and read exactly like a
> fold-mask finding. The flags are required even under `--max_epochs 0`, because the model is
> built from the CLI args before the state dict is loaded into it; without them every fold dies
> with `size mismatch for model.central_heads.0.0.weight: [64, 72, 3, 3] vs [64, 64, 3, 3]`.

Verify the checkpoints before spending anything on them:

```bash
$PY scripts/check_checkpoint_fingerprint.py \
    --manifest data/conv_update/PROVENANCE.json \
    --control artifacts/model-khrpthgy:v0/model.ckpt
```

The control must be **REJECTED** on `head_in 64`. Without it the check has not been shown to
reject anything and its verdict on the real checkpoints means nothing.

---

## 1. Hindcast prediction

Five fold models each predict only the territory they never trained on.

### 1a. From existing checkpoints — predict only

```bash
$PY -u scripts/run_hindcast_folds.py --stage train --max_epochs 0 \
  --folds 1,2,3,4,5 --gpus 0,1 \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --region config/region_to_predict_large.geojson --windows 2000 \
  --output_root $G --norm_stats_json data/norm_stats.json \
  --val_stride 1024 --num_workers 3 --keep_fold_rasters \
  --fold_checkpoints "$CKPTS" --extra_train_args "$ARCH" \
  --wandb_group global-conv-update
```

`--windows 2000` is what the deliverable needs: inputs 1990/1995/2000, targets 2005/2010/2015/
2020. Use `--windows all` only if you also want the shorter w2005/w2010/w2015 pairs, which cost
~2.5× and are needed for residual work this branch does not do.

Each fold log must read `✓ Checkpoint loaded with 0 warm-started quantile-head convs`.

> **Do not run heavy CPU work alongside prediction.** A single concurrent analysis job cut
> throughput from 170 to 135 tiles/s.

### 1b. Training from scratch

Drop `--max_epochs 0` and `--fold_checkpoints`, set `--max_epochs 150`. ~40 min per fold on top.
Everything else is identical.

---

## 2. Stitch

```bash
$PY -u scripts/run_hindcast_folds.py --stage stitch --stitch_mode holdout \
  --folds 1,2,3,4,5 --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --region config/region_to_predict_large.geojson --windows 2000 \
  --output_root $G --keep_fold_rasters --disable_wandb
```

Produces `$G/stitched/w2000_prediction_{year}_{q}.tif`. **Every raster must report
184,573,321 valid px** = `count(fold_mask_b4 ∈ {1..5} ∧ valid HM)`.

`holdout` is a hard mosaic — adjacent fold territories come from different models, and the join
shows in the **upper bound only**: 2.01× the within-fold step, against 1.08× for the central
field and the lower bound. That is because the folds agree on central and lower (mean pairwise
0.0053 and 0.0040) and not on upper (0.0383). A single fold's own raster is seamless at every
period from 64 to 1024 px, so neither the model nor the overlap blending is involved.

---

## 3. The seamless display product

A fold-mean raster removes the seam by averaging all five folds at every pixel. **This requires
re-predicting with the restriction mask disabled**: the holdout run computes each fold only over
its own territory plus a tile halo, so the five folds share no pixels and cannot be averaged.

```bash
$PY -u scripts/run_hindcast_folds.py --stage train --max_epochs 0 \
  --folds 1,2,3,4,5 --gpus 0,1 \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --region config/region_to_predict_large.geojson --windows 2000 \
  --output_root $G/mean_run --norm_stats_json data/norm_stats.json \
  --val_stride 1024 --num_workers 3 --keep_fold_rasters \
  --fold_checkpoints "$CKPTS" \
  --extra_train_args "$ARCH --predict_restrict_mask ''"

$PY -u scripts/run_hindcast_folds.py --stage stitch --stitch_mode mean \
  --folds 1,2,3,4,5 --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --region config/region_to_predict_large.geojson --windows 2000 \
  --output_root $G/mean_run --keep_fold_rasters --disable_wandb
```

`--predict_restrict_mask ''` works because `--extra_train_args` is appended last and argparse
takes the final occurrence; an empty value is falsy. Confirm the fold logs contain **no**
`Restriction mask:` line. The mean stitch must report **184,608,551** px — *more* than the
holdout, because it fills the union of fold coverage rather than the mask.

> **The mean product is in-sample at every pixel**: four of five folds trained on any given
> location. Use it for display only. Never score it. Its interval is also narrower than any
> single fold's, because averaging discards the between-fold spread rather than adding it.

---

## 4. Central-field diagnostics

```bash
$PY -u scripts/diagnose_central_field.py --label conv_update_g1 \
  --stitched_dir $G/stitched --out_dir $G/central_diag \
  --region_root data/ensemble/region/global --disable_wandb
```

The cheapest signal available: skill against **persistence**, not zero, stratified by distance
band and HM level. Score against persistence because the median 20-year HM change is 0.0001 — a
pooled RMSE looked unremarkable while the central forecast was losing to "nothing will change"
by 2.2× in MSE at h=5. Expect skill positive at all four horizons and rising with lead time. If
it is negative, stop: something is wrong upstream and nothing downstream fixes it.

---

## 5. The forward product

**No fold is excluded, and every chip is used.** Without `--train_all_splits True` the run
trains on the 70% train split alone and discards a third of the world for no benefit — the
forward model has no held-out geography to protect.

A fold model cannot forecast forward: each deliberately never saw a fifth of the world. The
forward product therefore comes from a sixth model in the identical configuration, reading HM at
2010/2015/2020 and predicting 2025/2030/2035/2040. It has no mosaic and no seam.

```bash
HP="--hidden_dim 64 --num_layers 4 --kernel_size 3 --locenc_out_channels 8 \
--locenc_legendre_polys 10 --ssim_weight 0.2 --laplacian_weight 0.3 \
--histogram_weight 1.0 --histogram_lambda_w2 0.1 --histogram_warmup_epochs 0"

$PY -u scripts/train_lightning.py \
    --max_epochs 150 --train_chips 100 --val_chips 40 --val_stride 1024 \
    --batch_size 8 --num_workers 3 --devices 1 --seed 42 --train_all_splits True \
    --norm_stats_json data/norm_stats.json \
    --run_full_set_evaluation False --run_large_area_prediction True \
    --predict_region config/region_to_predict_large.geojson \
    --predict_stride 64 --predict_batch_size 32 \
    --predict_output_dir $G/forecast_preds \
    --predict_input_years 2010,2015,2020 --predict_output_prefix "" \
    $HP $ARCH --wandb_group central-conv-update
```

**Verify:** the log must print `PRODUCTION MODE: training on EVERY chip in the split mask`, must
**not** print `FOLD-CV MODE`, and `Pre-computing valid positions` should appear only for splits 2
and 3 — the training pool needs no restriction and therefore no precomputation.

`--predict_input_years 2010,2015,2020` is used because the forward window is not in
`run_hindcast_folds.py`'s window list, so this calls `train_lightning.py` directly.

To predict only, from the existing production checkpoint, add
`--max_epochs 0 --checkpoint ./spatio-temporal-convlstm/6zkppztt/checkpoints/epoch=12-step=169.ckpt`.

---

## 6. Publish the COGs

```bash
./scripts/make_product_cogs.sh
```

Three sets of twelve, each verified for COG layout, overviews, and transform/CRS/nodata/value
identity against source:

| set | source | out | sample status |
|---|---|---|---|
| A | `$G/stitched` | `$G/cogs_hindcast/hm_hindcast_{year}_{q}.tif` | out-of-sample; the only set that may be scored |
| B | `$G/mean_run/stitched_mean` | `$G/cogs_hindcast_mean/hm_hindcast_mean_{year}_{q}.tif` | in-sample; display only |
| C | `$G/forecast_preds` | `$G/cogs_forecast/hm_forecast_{year}_{q}.tif` | production model |

Set C is republished with `--dst_nodata nan`. The prediction writer used to copy the source
raster's profile and override only `count`/`dtype`/`compress`, so it inherited `nodata=3.4e38`
while filling invalid pixels with NaN — a header disagreeing with its own data, which renders
every ocean pixel as valid in anything that honours it. The writer now sets `nodata=np.nan`, so
rasters produced from here on need no override; the frozen `forecast_preds/` predate the fix.

---

## 7. Proving a change to the model code

The products ship from frozen checkpoints, so the code has to be shown to reproduce them. Eight
gates, in order; each one gates the next.

| gate | command | pass condition |
|---|---|---|
| **S0** unit tests | `pytest tests/ -q` | 7 legacy failures (stale: they assert a 4-channel output from a model that emits 12), everything else passes |
| **S1** architecture | `scripts/check_checkpoint_fingerprint.py --manifest … --control …` | 6 checkpoints at `head_in 72`; control **rejected** at 64 |
| **S2** fold-CV training | `train_lightning.py --exclude_fold 1 --max_epochs 2 …` | `FOLD-CV MODE`, `ConvLSTM grad norm: 0.000000`, non-zero `val_total_loss` |
| **S3** production training | same with `--train_all_splits True` | `PRODUCTION MODE:` banner, no `FOLD-CV MODE`, precompute only for splits 2/3 |
| **S4** checkpoint load | S2 plus `--checkpoint <fold1>` | `✓ Checkpoint loaded with 0 warm-started quantile-head convs` |
| **S5** fold prediction | predict fold 1 over `config/region_smoke_aligned.geojson` | **bit-identical** to the frozen `preds/fold1_w2000_*` on fold-1 interior pixels |
| **S6** forecast prediction | predict the same window from the production checkpoint | max abs diff < 1e-5 vs `forecast_preds/`; new raster `nodata=nan` |
| **S7** stitcher | re-stitch one global raster in both modes | bit-identical to `stitched/` and `stitched_mean/`; counts 184,573,321 / 184,608,551 |

`ConvLSTM grad norm: 0.000000` is the `--central_residual` fingerprint: the trunk gets no
gradient from the central loss. On argparse defaults it prints ~0.04 and the initial central
loss is 20× higher. It is the one line that separates "the production architecture" from "a run
that started and looked fine".

> **`config/region_smoke_aligned.geojson` is phase-aligned on purpose, and the comparison is
> worthless without that.** Prediction tiles are enumerated as `range(r0, r1, stride)` from the
> *region bbox origin*, so a region whose origin is not congruent to the global run's modulo
> `--predict_stride` blends every pixel from a different set of tile offsets. The unaligned
> southern-Africa box gives max |diff| up to 1.5e-2 against the frozen rasters with mean |diff|
> at 1e-6 — which reads like a real defect and is entirely tile phase. The aligned window
> (origin 12352, 21504; both multiples of 64, matching the global bbox origin at 0,0) gives
> exact equality on pixels inset one tile from the edge.

`--predict_restrict_mask` is exact for kept pixels *within one tile grid*: every tile overlapping
a kept pixel is still processed, so those pixels get the same blended value an unrestricted run
would give. It does not make two different tile grids agree.
