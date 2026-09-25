# Fitting and running the model

The runbook for the global product, with **measured** cost at every stage. Every number in the
cost tables came off the run of 2026-08-30/31 on this box; nothing is projected.

Method in `docs/global_methodology.md`. What the numbers mean in
`docs/dist_global_scorecard.md`.

---

## 0. Environment and hardware

```bash
# Conda env — NOT base
PY=/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python
```

Two RTX A5000 (24 GB each), 125 GB DRAM. Root SSD 877 GB; large artifacts live on
`/mnt/hdd1/spatio-temporal/data` behind symlinks in `data/ensemble/`. **Root is usually the
binding constraint — pass HDD paths directly for anything large.**

Experiments track to W&B.

### 0.1 Inputs

```
data/raw/hm_global/
  HM_{1990..2020}_{AA,AG,BU,EX,FR,HI,NS,PO,TI,gdp,population}_1000.tiff
  change_context_w{2000,2005,2010,2015,2020}_1000.tif   (~2.2 GB each)
  hm_context_w{2000,2005,2010,2015,2020}_1000.tif       (~750 MB each)
  fold_mask_b4_1000.tif        512 px blocks, k=5      <- this lineage
  fold_mask_1000.tif           128 px checkerboard     <- the frozen product's
  split_mask_1000.tif          70/10/10/10             <- the forward model's
  ecoregion_id_1000.tif
```

Build the neighbourhood-HM covariate once if absent:

```bash
$PY scripts/prepare_hm_context.py          # 2:45 per base year, 0.75 GB each
```

**Any long-range covariate must be precomputed on the full raster.** Radii ≥ 30 px saturate
against a 128 px training chip and the covariate silently becomes a constant.

### 0.2 Disk budget, measured

| stage | on disk |
|---|---|
| hindcast `preds/` (5 folds × 10 window-years) | **207 GB** — deletable once the stitch is verified |
| hindcast `stitched/` (40 rasters) | **142 GB** |
| forecast `preds/` (16 rasters) | **59 GB** |
| products (24 COGs + 2 quantile stores + 51 tar shards) | **216 GB** |
| **total for the ConvLSTM phase** | **~625 GB** |

---

## 1. Hindcast — train and predict, k = 5

```bash
./scripts/run_global_dist_hindcast.sh train 1,2      # wave 1
./scripts/run_global_dist_hindcast.sh train 3,4      # wave 2
./scripts/run_global_dist_hindcast.sh train 5        # wave 3
```

or chain them so no GPU idles at a handover:

```bash
WAIT_PID=<wave1 pid> nohup ./scripts/chain_global_waves.sh &
```

The driver sources `scripts/dist_base_args.sh`, adds the e1 flags and `--predict_row_chunk 512`,
writes to the HDD, and **refuses to hand over unless every fold log reads back five
fingerprints**. Waves rather than one five-fold call because `run_hindcast_folds.py` verifies
nothing until every fold it was given has finished: a wave boundary is where a wrong
configuration must stop the next wave rather than surface five folds later.

### 1a. Verify before continuing — non-negotiable

The driver does this automatically and exits non-zero on failure. Read it anyway:

```
✓ foldN loss weights verified: ssim=0.0 lap=0.0 hist=0.0
✓ foldN weight averaging: mean of the last 20 epochs
✓ foldN predicts from the averaged end-of-training checkpoint
✓ foldN Context channels:  12 (radii 3,30,100, hm mean,max @ 3,30,100)
✓ foldN Row banding: 34 bands of 512 rows; 268 accumulators x 0.095 GiB = 25.6 GiB worst case
```

`run_hindcast_folds.py` injects the frozen product's `--ssim_weight 0.2 --laplacian_weight 0.3
--histogram_weight 1.0` into every fold command, and `--extra_train_args` is appended last —
**anything `BASE_ARGS` does not name is silently inherited.** A run trained that way looks
entirely normal.

### 1b. Measured cost

Prediction cost is linear in the number of accumulators, which is
`n_horizons × (3 + n_qf_levels)`:

**`T = 15.1 min + 0.243 × n_accumulators`**, fitted on fold 1 and accurate to ~1% on the other
eighteen fold-windows.

| window | horizons | accumulators | measured, all five folds |
|---|---|---|---|
| w2000 | 4 | 268 | 80:45 · 83:14 · 81:45 · 81:21 · 81:33 |
| w2005 | 3 | 201 | 63:04 · 66:33 · 65:17 · 65:00 · 62:58 |
| w2010 | 2 | 134 | 48:11 · 49:38 · 47:52 · 47:39 · 49:26 |
| w2015 | 1 | 67 | 31:47 · 32:07 · 31:32 · 31:13 · 32:51 |

| | measured |
|---|---|
| train, 150 epochs, 2 folds in parallel | **38 min** |
| predict all four windows, one fold | **~223 min** |
| one fold, train + predict | **262–270 min** (3% spread across five folds) |
| **k = 5 on two GPUs, three waves** | **13.4 h** |

### 1c. Memory

The prediction accumulators are the thing that kills a global run. 268 full-window float32
arrays at 2.55 GiB each; the pages one fold's tiles touch come to **136 GiB**. `--predict_row_chunk
512` brings that to **25.6 GiB per fold**; two concurrent folds peaked at **68 GiB used with 54
GiB free**, swap untouched.

Never run this unbanded on the global grid. It is a SIGKILL with no traceback.

---

## 2. Stitch

```bash
./scripts/run_global_dist_hindcast.sh stitch 1,2,3,4,5
```

**`--stitch_mode holdout`** — the default, and the only mode anything scored may use. `mean`
averages all five folds at every pixel: seamless, in-sample everywhere, correct only for a
forward product that held nothing out.

**Measured: 4 h 35 min** for 40 rasters (10 window-years × lower/central/upper/qf), 142 GB.

Verify coverage before scoring. Each stitched raster must hold essentially every land pixel,
because the five fold territories partition the grid; `stitch_fold_predictions` skips a missing
fold with a printed warning and returns success.

```
w2000_prediction_2005_central.tif   184,573,321 px  100.0% of land
... all 40 identical ...
OK: 40 rasters, every one covering >95% of land
```

`scripts/chain_global_stitch_score.sh` does the stitch, the check and the scorecard in sequence
and aborts on any shortfall.

---

## 3. Score the model — no ensemble

```bash
$PY scripts/score_distributional_model.py \
    --stitched_dir <hdd>/g_e1_hind/stitched --label g_e1_global \
    --out_dir data/ensemble/exp/g_e1_hind_score \
    --folds 1,2,3,4,5 --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
    --row_chunk 512
```

**`--row_chunk 512` is required globally.** Without it `read_qf` loads all 64 bands of a
window-year at once — 175 GB on this grid, 16 GB on Africa. Banding is an exact identity: verified
bit-identical across 199 rows × 28 columns of output at three band sizes.

**`--fold_mask` must be the mask prediction used.** The script's default points at southern
Africa; a mismatched mask silently scores a subset. It self-checks and refuses below 50% coverage
— look for `fold mask covers 100.0% of the finite pixels`.

**Measured: 6 h 22 min**, ~38 min per window-year row. Peak **92 GiB RSS, 30 GiB free**, swap 0 —
the peak is per-row and every row has identical pixel counts, so it plateaus rather than climbs.
**Run this with nothing else on the box.**

Outputs to `--out_dir`: `dist_*.csv`, `central_*.csv`, `consistency_*.csv`, `summary_*.json`.

---

## 4. The forward model and forecast

```bash
./scripts/run_global_dist_forecast.sh 0
# or, re-predicting from an existing forward checkpoint:
CKPT=models/checkpoints/final_foldNone_XXXXXXX.ckpt ./scripts/run_global_dist_forecast.sh 1
```

Trains with `--train_all_splits True` and no `--exclude_fold`, then predicts the 2020-base window
forward to 2025–2040 over the whole grid with no fold restriction.

Fingerprints, all six enforced by the driver:

```
✓ production mode engaged (no geography held out)     <- PRODUCTION MODE banner present
✓ no fold held out                                    <- FOLD-CV MODE absent
✓ forecast loss weights verified: ssim=0.0 lap=0.0 hist=0.0
✓ forecast Context channels:  12 (...)
✓ Row banding: 34 bands of 512 rows; 268 accumulators
✓ OK: 16 forecast rasters, every one complete and BigTIFF where needed
```

**The last check counts pixels; it does not test for a file.** The first global attempt left all
sixteen rasters on disk, truncated at the 4 GiB classic-TIFF ceiling, with every unwritten row
reading back as **finite zeros** — an existence check passed them and the numbers looked
plausible. A complete raster holds 184,608,551 px; a truncated one reads *more*, because the
unwritten remainder is not nodata.

| | measured |
|---|---|
| train, 150 epochs, one GPU | 38 min |
| predict 2025–2040, whole grid | **181 min** |
| output | 16 rasters, 59 GB; quantile rasters 13.8–14.2 GB each |

The forward model can share the box with the last hindcast fold — wave 3 uses one GPU, so
`scripts/chain_global_forecast.sh` starts the forward model on the other and takes ~4 h off the
critical path.

---

## 5. Publish

```bash
./scripts/package_global_products.sh all
```

**COGs** — 12 hindcast (w2000) and 12 forecast, via `gdal_translate -of COG`, then *proved*:
`LAYOUT=COG`, overviews present, transform/CRS/nodata preserved, and sampled pixels identical to
source.

```bash
--verify_windows 48 --min_verified_px 2000000
```

is not optional globally. The value comparison skips nodata and the grid is 73% ocean, so the
default 12-window sample can verify **nothing** and still report success. Measured: 11,505,616
valid px compared per raster.

**Quantile stores** — `scripts/package_qf_icechunk.py`, one array
`quantile_forecast(time, quantile, latitude, longitude)` int16, chunks `(1, 64, 512, 512)`.
`--base_years 2000` cuts the hindcast store to the only window reaching +20 yr.

| | measured |
|---|---|
| 24 COGs | ~40 min, **16 GB** (8.0 GB each set) |
| hindcast quantile store | **19.7 min**, 49 GB, 4 time steps |
| forecast quantile store | **22.9 min**, 52 GB, 4 time steps |

Each store is verified against its source rasters — 24 random windows compared exactly, u grid
and scale checked, and opened through xarray to confirm the dimension names survived.

---

## 6. Archive and upload

```bash
./scripts/archive_icechunk_shards.sh <repo>.icechunk <out_dir> 2G
```

Tars an icechunk repo into fixed-size shards for cloud transfer. An icechunk repo is thousands of
small files; uploading them individually is dominated by per-object latency. Shards are plain
uncompressed tar — the zarr chunks inside are already compressed.

The script **proves the shards reconstruct**: the tar stream is hashed as it is written, the
shards are hashed back, the rebuilt stream is listed with `tar -t`, and the file count is compared
to the source. A `MANIFEST.txt` carries the per-shard SHA-256s, the whole-archive hash and the
restore command.

```
hindcast_qf.icechunk   5,100 files, 49G  ->  25 shards
forecast_qf.icechunk   5,128 files, 52G  ->  26 shards
restore:  cat <name>.tar.part* | tar -xf -
```

Round-trip verified end to end: shards → tar → extract → a working icechunk repo whose data,
coordinates and attributes are byte-identical.

Then `rclone copy` each of the four product directories to Box, followed by `rclone check
--one-way` — a copy that silently dropped one shard leaves an unrestorable archive and nothing
else reports it.

| | measured |
|---|---|
| sharding both stores | **43 min** |
| upload, 8 transfers | ~21 MiB/s; 117 GB ≈ 95 min |

---

## Measured cost summary — the whole ConvLSTM phase

| stage | wall clock |
|---|---|
| train + predict, k=5, two GPUs | 13.4 h |
| stitch, 40 rasters | 4.6 h |
| model scorecard | 6.4 h |
| forward model train + predict | 3.6 h (overlapped with wave 3) |
| package 24 COGs + 2 quantile stores | 1.4 h |
| shard both stores | 0.7 h |
| upload 117 GB | ~1.6 h |
| **total** | **~28 h**, of which ~4 h overlapped |

---

## Operational notes

- **Never edit a shell script while it is running.** Bash reads by byte offset and will resume at
  a stale one. Copy to a new name and launch that. Likewise `pkill -f <pattern>` matches the shell
  running it — use `ps -eo pid,cmd | grep "patt[e]rn"` and kill by pid.
- **Sample memory faster than the thing you are watching for.** An OOM here went from steady to
  SIGKILLed in under 60 s; a 120 s poll saw nothing. `scripts/monitor_resources.sh <csv> 10`
  samples root, HDD, DRAM, swap, both GPUs and the largest python RSS every 10 s. When SIGKILL
  leaves no traceback, that CSV is the only evidence of what happened.
- **A watchdog must also detect the process vanishing.** "The job died" and "the job is thinking"
  look identical in a log that has simply stopped.
- **Prediction is bit-deterministic at a fixed batch size** but not across batch sizes; changing
  `--predict_batch_size` moves outputs by ~1e-5. Keep it fixed across folds of one product.
