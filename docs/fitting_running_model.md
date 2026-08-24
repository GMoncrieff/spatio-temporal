# Fitting and running the model

Everything needed to reproduce the global product from raw data: environment, input
requirements, and every script in order, with measured wall-clock and peak memory for each.

The methodology behind these steps is in `docs/global_ensemble_methodology.md`; the results are
in `docs/global_scorecard.md`. This document is the runbook.

**Total cost, end to end: roughly 50 hours**, of which one 28-hour scorecard run and one
2.5-hour prediction leg dominate. Everything else is minutes to a couple of hours.

---

## 0. Environment and hardware

```bash
# Conda env — NOT base
PY=/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python
```

| requirement | value |
|---|---|
| GPUs | 2 × 24 GB (RTX A5000). Field generation peaks at 20.5 GB on one card. |
| RAM | 125 GB. The binding step is the width-factor fit at **98 GB peak**. |
| disk (fast) | ~50 GB for code, checkpoints, small artifacts |
| disk (bulk) | **~1.1 TB** for one full run — see §0.3 |
| GDAL CLI | `gdal_translate` and `gdalinfo` on PATH, for COG creation |
| experiment tracking | Weights & Biases (pass `--disable_wandb` to skip) |

### 0.1 Input data

All rasters are on one grid: **17,111 × 40,000 px, EPSG:4326, 0.009° (~1 km)**, origin
(−180, 83.997). Any mismatch in transform or shape will fail loudly at the first read.

Under `data/raw/hm_global/`:

| input | count | note |
|---|---|---|
| `HM_{year}_AA_1000.tiff` | 7 | the HM index itself, 1990–2020 in 5-yr steps. **This is the target.** |
| `HM_{year}_{AG,BU,EX,FR,HI,NS,PO,TI,gdp,population}_1000.tiff` | 10 × 7 | component layers — the 11 dynamic input channels are these plus AA |
| `hm_static_*.tiff` | 11 | elevation, slope, aspect (sin/cos), TPI, dpi/dsi, precipitation, temperature (mean/min), IUCN strict and non-strict protection |
| `ecoregion_id_1000.tif` + `ecoregion_lookup.csv` | 1 + 1 | 804 ecoregions with biome/realm lookup, for T2 |

Years must be 1990, 1995, 2000, 2005, 2010, 2015, 2020. The hindcast uses input windows ending
2000/2005/2010/2015; the forward product uses 2010/2015/2020.

### 0.2 Derived inputs — build these once

**Change-context rasters** (two bands per window: signed past change, and distance to nearest
past change > 0.01). These *must* be computed on the full raster — deriving them inside a
128 px training chip makes the 100 px radius saturate into "is there any change in this chip".

```bash
$PY scripts/prepare_change_context.py --out_dir data/raw/hm_global
# → change_context_w{2000,2005,2010,2015,2020}_1000.tif   (~2.2 GB each)
```

**Split mask and fold mask.** The split mask is the 70/10/10/10 train/val/test/calib partition
used by the production model. The fold mask carries five contiguous 512 px territories for
cross-validation — `--fold_block_chips 4` is what makes them 4 × 128 = 512 px, which must exceed
the ~300 px residual correlation length.

```bash
$PY scripts/create_validity_mask.py                       # → split_mask_1000.tif
$PY scripts/create_validity_mask.py --folds_only --k 5 \
      --fold_block_chips 4 \
      --fold_mask_out data/raw/hm_global/fold_mask_b4_1000.tif
```

**Region root** — the assets the evaluation reads, gathered under one path:

```bash
$PY scripts/make_global_region_root.py --out_root data/ensemble/region/global --verify
ln -sfn /path/to/data/raw/hm_global/fold_mask_b4_1000.tif \
        data/ensemble/region/global/fold_mask_b4.tif
```

This yields `data/ensemble/region/global/` containing `ecoregion.tif`, `fold_mask_b4.tif` and
`covariates/w{year}_dist_past_change.tif` for the five base years.

**Normalisation statistics.** `data/ensemble/norm_stats.json` is written by the first training
run and reused by every later one as a sidecar, so all folds and the production model share one
normalisation.

### 0.3 Disk layout

Bulk artifacts must not land on the root filesystem. Point the experiment root at large storage
and symlink it in:

```bash
G=/mnt/hdd1/spatio-temporal/data/ensemble/exp/g1_foldb4
mkdir -p $G && ln -sfn $G data/ensemble/exp/g1_foldb4
```

| artifact | size |
|---|---|
| per-fold prediction rasters | 22 GB (holdout) / 24 GB (unrestricted, w2000 only) |
| stitched hindcast | 16 GB |
| residuals (× 2 — before and after calibration) | 34 + 35 GB |
| calibrated rasters | 17 GB |
| **hindcast ensemble, M=400** | **398 GB** |
| **independent-pixel null, M=400** | **410 GB** |
| **forecast ensemble, M=400** | **421 GB** |
| COGs (3 sets of 12) | 27 GB |

The null can be deleted after the scorecard; that is what makes ~1.1 TB sufficient rather than
1.5 TB.

> **Pass icechunk `--out` paths directly, never through a symlink.** The store's directory is
> cleared with `shutil.rmtree`, which refuses a symbolic link.

---

## 1. Hindcast prediction

Five fold models each predict only the territory they never trained on.

### 1a. If the fold checkpoints already exist — predict only

```bash
CKPTS="1=path/to/fold1.ckpt,2=...,3=...,4=...,5=..."

$PY -u scripts/run_hindcast_folds.py --stage train --max_epochs 0 \
  --folds 1,2,3,4,5 --gpus 0,1 \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --region config/region_to_predict_large.geojson --windows all \
  --output_root $G --norm_stats_json data/ensemble/norm_stats.json \
  --val_stride 1024 --num_workers 3 --keep_fold_rasters \
  --fold_checkpoints "$CKPTS" \
  --extra_train_args "--central_residual True --central_context True \
                      --monotone_quantile_width True --quantile_context True" \
  --wandb_group global-g1_foldb4 --tag _g1
```

**Cost: 152 min** for five folds across two GPUs.

> **The architecture flags are required even though `--max_epochs 0` trains nothing.**
> `train_lightning.py` builds the model from the CLI args and loads the state dict into it.
> Without them every fold dies in under a minute with
> `size mismatch for model.central_heads.0.0.weight: [64, 72, 3, 3] vs [64, 64, 3, 3]`.
> They also gate whether `change_context` enters the batch at all.

> **Do not run heavy CPU work alongside prediction.** A single concurrent analysis job cut
> throughput from 170 to 135 tiles/s.

### 1b. If training from scratch

Drop `--max_epochs 0` and `--fold_checkpoints`, and set `--max_epochs 150`. Add ~40 min per
fold. Everything else is identical.

### 1c. Verify before continuing

```bash
grep -c "warm-started" data/ensemble/logs/hindcast_fold*_g1.log   # expect the load message
```

Each fold log must read `✓ Checkpoint loaded with 0 warm-started quantile-head convs`. Also
confirm the checkpoints are the right architecture before you start — read the state dict and
assert `central_head_in == hidden_dim + 8` (72), `central_context_channels == 8`,
`central_residual`, `monotone_quantile_width`. **Include a known-wrong checkpoint as a control
and confirm it is rejected**, or the check is not discriminating.

---

## 2. Stitch

```bash
$PY -u scripts/run_hindcast_folds.py --stage stitch --stitch_mode holdout \
  --folds 1,2,3,4,5 --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --region config/region_to_predict_large.geojson --windows all \
  --output_root $G --keep_fold_rasters --disable_wandb
```

**Cost: ~40 min.** Produces 30 rasters in `$G/stitched/` (10 window × horizon pairs × 3
quantiles).

**Verify.** Compute the expected valid-pixel count *before* the run —
`count(fold_mask_b4 ∈ {1..5} ∧ valid HM)` — and check every stitched raster reports it. On this
grid that is **184,573,321**, and all 30 rasters must agree. Then confirm provenance: sample
pixels and check each equals the value from the fold that held it out.

---

## 3. Central-field diagnostics

```bash
$PY -u scripts/diagnose_central_field.py --label g1_foldb4 \
  --stitched_dir $G/stitched --out_dir $G/central_diag \
  --region_root data/ensemble/region/global --wandb_group central-g1_foldb4
```

**Cost: ~20 min.** The cheapest signal available — skill against persistence, stratified by
distance band and HM level. Expect skill positive at all four horizons and rising with lead
time. If it is negative, stop: something is wrong upstream and nothing downstream will fix it.

---

## 4. Residuals

```bash
$PY -u scripts/build_region_residuals.py \
  --pred_dir $G/stitched --pred_suffix "" --keep_splits all \
  --out_dir $G/residuals \
  --covariate_dir data/ensemble/region/global/covariates
```

**Cost: ~60 min, 34 GB.** Every downstream fit reads this manifest. All 10 rows must report the
full valid-pixel count with 0 training pixels dropped.

---

## 5. AR(1) horizon coupling

```bash
$PY -u -c "
import json, sys; sys.path.insert(0,'.')
from src.ensemble.residuals import horizon_autocorrelation
rho = horizon_autocorrelation('$G/residuals/manifest.csv')
json.dump({str(k): v for k, v in rho.items()},
          open('$G/residuals/horizon_autocorrelation.json','w'), indent=2)
print(rho)"
```

**Cost: ~5 min.** The estimator reads evenly spaced full-width row stripes — deterministic, no
seed.

**Verify by re-running with `offset=128, 256, 384`.** Independent stripe phases should agree to
a few hundredths. If they disagree by more than ~0.05 the estimate is a sampling artifact.

> **If this file is missing, `generate_ensemble.py` silently falls back to ρ = 0.9** and T4.1
> scores against a value nothing measured. Always confirm the JSON exists before generating.

---

## 6. Coverage, variograms and the calibration audit

```bash
$PY -u scripts/run_diagnostics.py \
  --manifest $G/residuals/manifest.csv --out_dir $G \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --ecoregion_raster data/ensemble/region/global/ecoregion.tif \
  --block_sizes 1,10,100,1000 --n_pairs 300000 --max_lag_px 1024 \
  --wandb_group central-g1_foldb4
```

**Cost: ~2 h.** Writes `$G/diagnostics/variogram_fits.csv`, which the field generator and T3's
structure budget both read, plus the coverage audit under `$G/calibration/`.

---

## 7. Calibration factors

### 7a. Choose which distance bands to correct

Fit **once** on three folds with all six bands, then derive every subset by masking. Each
distance band is the root of its own shrinkage hierarchy and the horizon-monotonicity pass is
keyed on `(band, dhat, HM)`, so a restricted JSON reproduces a restricted fit exactly.

```bash
/usr/bin/time -v $PY -u scripts/fit_width_factors.py \
  --manifest $G/residuals/manifest.csv \
  --class_axis band --bands 0,1,2,3,4,5 --monotone_horizons True \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif --fit_folds 1,2,3 \
  --out $G/bs_b012345.json                        # 38 min, 76 GB peak

# control: prove the masking shortcut is exact
$PY -u scripts/fit_width_factors.py ... --bands 0,1,2 --out $G/bs_b012_control.json
```

Derive variants (set excluded bands to `[1.0, 1.0]`), then score them on the two held-out folds
and audit the tail:

```bash
$PY -u scripts/score_width_variants.py \
  --manifest $G/residuals/manifest.csv \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif --score_folds 4,5 \
  --variants identity=identity b012=... b012345=... \
  --out_csv $G/band_sweep_global.csv            # 2h23, 116 GB peak

$PY -u scripts/band_tail_audit.py --manifest $G/residuals/manifest.csv \
  --horizon 20 --variants b012=... b012345=...  # ~10 min
```

> **Decide on both readings, never the interval score alone.** A pooled average is structurally
> blind to a defect living in a thousandth of the pixels. Check the held-out interval score
> *and* the far-band p99.9 half-width ratio, *and* class-conditional coverage. On global data
> the pooled score prefers doing nothing while a remote class sits at 0.67 coverage.

**Settled answer for this product: correct all six bands.**

### 7b. Production fit, all folds

```bash
/usr/bin/time -v $PY -u scripts/fit_width_factors.py \
  --manifest $G/residuals/manifest.csv \
  --class_axis band --bands 0,1,2,3,4,5 --monotone_horizons True \
  --out $G/unified_factors.json
```

**Cost: 34 min, 98 GB peak.** This is the binding memory step of the whole pipeline — **run it
with nothing else on the machine.**

---

## 8. Apply the calibration

**One call per input window, each with its own distance raster.** Only 49–61% of valid pixels
keep the same distance band between windows, so a single raster mis-keys about half of them in
three of the four windows.

```bash
for Y in 2000 2005 2010 2015; do
  mkdir -p $G/stitched_w$Y
  for f in $G/stitched/w${Y}_prediction_*.tif; do ln -sf "$f" $G/stitched_w$Y/$(basename $f); done
  $PY -u scripts/apply_recalibration.py --targets hindcast --factors none \
    --width_factors $G/unified_factors.json --monotone_horizons False \
    --hindcast_dir $G/stitched_w$Y --hindcast_suffix "" --hindcast_out $G/recal_u \
    --dist_raster data/ensemble/region/global/covariates/w${Y}_dist_past_change.tif
done

# then the per-pixel horizon monotonicity pass, once over every group
$PY -u -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from apply_recalibration import enforce_horizon_monotonicity
enforce_horizon_monotonicity('$G/recal_u', '_recal')"
```

**Cost: 68 min total, 14 GB peak.**

`--factors none` puts the layer in unified mode: the conformal table becomes the identity and
the width factors carry everything. Confirm the log reads
`conformal table: IDENTITY (unified mode — width factors carry everything)` and
`width factors keyed on 'band'`.

Deferring monotonicity to a single final pass avoids re-running it once per window.

---

## 9. Rebuild residuals against the calibrated bounds

```bash
$PY -u scripts/build_region_residuals.py \
  --pred_dir $G/recal_u --pred_suffix _recal --keep_splits all \
  --out_dir $G/residuals_u \
  --covariate_dir data/ensemble/region/global/covariates
```

**Cost: ~60 min, 35 GB.** Required: the marginal shape normalises to whatever half-widths it is
fitted on. Fitting it to the old widths while generating from the new ones breaks the tail gates
silently.

---

## 10. Field spectrum

```bash
$PY -u scripts/fit_field_spectra.py \
  --manifest $G/residuals/manifest.csv --out $G/spectral_fits.json
```

**Cost: ~15 min.** Note this reads the **pre-calibration** residuals — the spatial structure of
the error is a property of the model, not of the interval width. Defaults carry the fixed range
basis, Matérn `nu = 0.5`, and the `--long_range 1500 --long_weight 0.40` reservoir.

---

## 11. Marginal shape

Sweep the tail bound first — closed-form, no generation required:

```bash
$PY -u scripts/predict_change_rates.py \
  --manifest $G/residuals_u/manifest.csv \
  --sweep_u_bound 0.975,0.99,0.999,1.0 \
  --out_csv $G/u_bound_sweep_global.csv           # ~45 min
```

> **Never sweep a marginal by generating ensembles.** The member *mean* of P(Δ > threshold) is
> available in closed form, and this script shares no code with the GPU sampler — so agreement
> is evidence and disagreement localises a bug.

Then fit at the chosen bound:

```bash
$PY -u scripts/fit_marginal_shape.py \
  --manifest $G/residuals_u/manifest.csv --out $G/marginal_shape.json \
  --by_band true --pooled_body true --u_bound 0.999 --u_bound_lo 0.025
```

**Cost: 12 min, 72 GB peak.** Expect 0.999 and 1.0 to be identical — that is the signal the
lever is exhausted.

---

## 12. Smoke test — do this before the long run

Four members, every stage, all four block scales. It costs about 90 minutes and has caught
broken chains that would otherwise have surfaced hours into a 28-hour job.

```bash
REGION_ROOT=data/ensemble/region/global \
WRAP=True MEM_BUDGET_GB=60 BLOCKS=1,10,100,1000 \
RECAL_SUBDIR=recal_u SUFFIX=_smoke MEMBERS=4 \
MEMBERS_OUT=$G/members_smoke.icechunk NULL_OUT=$G/null_smoke.icechunk \
./scripts/run_smoke_tier.sh g1_foldb4
```

It ends by running `check_smoke.py`, which must print
`✓ every stage ran, every scored value is finite, every block scale reported`.

**Also read four things by hand:**

| check | expected |
|---|---|
| ρ in the generator log | the measured values, **not** 0.9 |
| `wrap_lon` | `True` on the global grid |
| T3.5 seam row | scored *or* marked not-evaluable — never a NaN failure |
| `--mem_trace` peak | ~69 GB against a 60 GB budget; the budget is a target, not a bound |

Delete the smoke stores afterwards.

---

## 13. Generate the ensemble and score it

```bash
COMMON=(--central_dir $G/recal_u --recal_dir $G/recal_u
        --central_pattern "w2000_prediction_{year}_central_recal.tif"
        --recal_pattern "w2000_prediction_{year}_{q}_recal.tif"
        --years 2005,2010,2015,2020 --base_year 2000 --members 400
        --spectral_fits $G/spectral_fits.json
        --variogram_fits $G/diagnostics/variogram_fits.csv
        --rho_json $G/residuals/horizon_autocorrelation.json
        --wrap_lon True --marginal_shape $G/marginal_shape.json
        --dist_raster data/ensemble/region/global/covariates/w2000_dist_past_change.tif
        --gpus 0,1)

$PY -u scripts/generate_ensemble.py "${COMMON[@]}" \
    --out $G/members_m400.icechunk --wandb_group central-g1_foldb4      # 50 min, 398 GB

$PY -u scripts/generate_ensemble.py "${COMMON[@]}" --independent \
    --out $G/null_m400.icechunk --disable_wandb                          # 79 min, 410 GB

$PY -u scripts/validate_ensemble.py \
    --ensemble $G/members_m400.icechunk --null_ensemble $G/null_m400.icechunk \
    --recal_dir $G/recal_u \
    --ecoregion_raster data/ensemble/region/global/ecoregion.tif \
    --variogram_fits $G/diagnostics/variogram_fits.csv \
    --rho_json $G/residuals/horizon_autocorrelation.json \
    --recal_manifest $G/recal_u/recal_manifest.json \
    --dist_raster data/ensemble/region/global/covariates/w2000_dist_past_change.tif \
    --marginal_shape $G/marginal_shape.json \
    --block_sizes 1,10,100,1000 --out_dir $G/validation_m400 \
    --mem_budget_gb 60 --mem_trace --wandb_group central-g1_foldb4       # 28 h
```

The null exists only to give T3.2 and T3.3 a baseline; **delete it once the card is written** to
free 410 GB.

Cross-check the card against the closed-form second implementation:

```bash
$PY -u scripts/predict_change_rates.py --manifest $G/residuals_u/manifest.csv \
    --marginal_shape $G/marginal_shape.json
$PY -u scripts/member_distance_relationship.py \
    --ensembles g1=$G/members_m400.icechunk --members 400
```

---

## 14. Hindcast deliverables

```bash
$PY -u scripts/make_cogs.py --src_dir $G/recal_u --out_dir $G/cogs_hindcast \
    --prefix hm_hindcast --base_year 2000 --years 2005,2010,2015,2020
```

**Cost: ~35 min, 8.7 GB.** Twelve COGs, each verified for COG layout, overviews, and
transform/CRS/nodata/value identity against source. The M=400 store is itself the ensemble
deliverable.

---

## 15. The forward product

### 15a. Train the production model

**No fold is excluded, and every chip is used.** Without `--train_all_splits True` the run
trains on the 70% train split alone and discards a third of the world for no benefit — the
forward model has no held-out geography to protect.

```bash
ARCH="--hidden_dim 64 --num_layers 4 --kernel_size 3 --locenc_out_channels 8 \
--locenc_legendre_polys 10 --ssim_weight 0.2 --laplacian_weight 0.3 \
--histogram_weight 1.0 --histogram_lambda_w2 0.1 --histogram_warmup_epochs 0 \
--central_residual True --central_context True --monotone_quantile_width True \
--quantile_context True"

$PY -u scripts/train_lightning.py \
    --max_epochs 150 --train_chips 100 --val_chips 40 --val_stride 1024 \
    --batch_size 8 --num_workers 3 --devices 1 --seed 42 --train_all_splits True \
    --norm_stats_json data/ensemble/norm_stats.json \
    --run_full_set_evaluation False --run_large_area_prediction True \
    --predict_region config/region_to_predict_large.geojson \
    --predict_stride 64 --predict_batch_size 32 \
    --predict_output_dir $G/forecast_preds \
    --predict_input_years 2010,2015,2020 --predict_output_prefix "" \
    $ARCH --wandb_group central-g1_foldb4 --wandb_run_name production-g1_foldb4
```

**Cost: ~2.5 h** (training plus one global prediction pass).

**Verify:** the log must print `PRODUCTION MODE: training on EVERY chip in the split mask`, and
must **not** print `FOLD-CV MODE`. `Pre-computing valid positions` should appear only for splits
2 and 3 — the training pool needs no restriction and therefore no precomputation.

The configuration must match the hindcast folds exactly, or the residual statistics fitted in
§7–§11 describe a different model.

`--predict_input_years 2010,2015,2020` is used because the forward window is not in
`run_hindcast_folds.py`'s window list, so this calls `train_lightning.py` directly.

### 15b. Calibrate, generate, publish

```bash
$PY -u scripts/apply_recalibration.py --targets production --factors none \
  --width_factors $G/unified_factors.json --monotone_horizons True \
  --production_dir $G/forecast_preds --production_out $G/forecast_recal \
  --production_years 2025,2030,2035,2040 --production_base_year 2020 \
  --dist_raster data/ensemble/region/global/covariates/w2020_dist_past_change.tif

$PY -u scripts/generate_ensemble.py \
  --central_dir $G/forecast_recal --recal_dir $G/forecast_recal \
  --central_pattern "prediction_{year}_central_recal.tif" \
  --recal_pattern "prediction_{year}_{q}_recal.tif" \
  --years 2025,2030,2035,2040 --base_year 2020 --members 400 \
  --spectral_fits $G/spectral_fits.json \
  --variogram_fits $G/diagnostics/variogram_fits.csv \
  --rho_json $G/residuals/horizon_autocorrelation.json \
  --wrap_lon True --marginal_shape $G/marginal_shape.json \
  --dist_raster data/ensemble/region/global/covariates/w2020_dist_past_change.tif \
  --out $G/forecast_members_m400.icechunk --gpus 0,1

$PY -u scripts/make_cogs.py --src_dir $G/forecast_recal --out_dir $G/cogs_forecast \
  --prefix hm_forecast --base_year 2020 --years 2025,2030,2035,2040 \
  --src_pattern "prediction_{year}_{q}_recal.tif"
```

**Cost: 64 min generation (421 GB), ~35 min COGs.** Note the production rasters carry no
`w{base}_` prefix, hence the explicit `--src_pattern`.

The forward ensemble reuses the hindcast-derived spectrum, marginal and ρ. This assumes the
error structure is stationary in time, which cannot be validated — there is no future data —
and weakens the further out the forecast runs.

---

## 16. Optional — the seamless display product

The scored hindcast is a mosaic: adjacent fold territories come from different models, and
wherever they disagree the join is visible, mostly in the upper bound. A fold-mean raster
removes it.

**This requires re-predicting with the restriction mask disabled**, because the holdout run only
computes each fold over its own territory plus a tile halo — the five folds have *no* pixels in
common, so they cannot be averaged.

```bash
$PY -u scripts/run_hindcast_folds.py --stage train --max_epochs 0 \
  --folds 1,2,3,4,5 --gpus 0,1 \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --region config/region_to_predict_large.geojson --windows 2000 \
  --output_root $G/mean_run --norm_stats_json data/ensemble/norm_stats.json \
  --val_stride 1024 --num_workers 3 --keep_fold_rasters \
  --fold_checkpoints "$CKPTS" \
  --extra_train_args "--central_residual True --central_context True \
     --monotone_quantile_width True --quantile_context True --predict_restrict_mask ''"

$PY -u scripts/run_hindcast_folds.py --stage stitch --stitch_mode mean \
  --folds 1,2,3,4,5 --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
  --region config/region_to_predict_large.geojson --windows 2000 \
  --output_root $G/mean_run --keep_fold_rasters --disable_wandb

$PY -u scripts/apply_recalibration.py --targets hindcast --factors none \
  --width_factors $G/unified_factors.json --monotone_horizons True \
  --hindcast_dir $G/mean_run/stitched_mean --hindcast_suffix "" \
  --hindcast_out $G/recal_u_mean \
  --dist_raster data/ensemble/region/global/covariates/w2000_dist_past_change.tif

$PY -u scripts/make_cogs.py --src_dir $G/recal_u_mean --out_dir $G/cogs_hindcast_mean \
  --prefix hm_hindcast_mean --base_year 2000 --years 2005,2010,2015,2020
```

**Cost: ~5 h for the w2000 window** (2.3 h prediction + stitch + calibration + COGs).

`--predict_restrict_mask ''` works because `--extra_train_args` is appended last and argparse
takes the final occurrence; an empty value is falsy, so no restriction is applied. Confirm the
fold logs contain **no** `Restriction mask:` line.

**Verify** that all five folds now cover 100% of valid pixels and that the all-five intersection
is the full valid set. The mean stitch should report **more** pixels than the holdout
(184,608,551 vs 184,573,321) — it fills the union of fold coverage.

> **The mean product is in-sample at every pixel** — four of five folds trained on any given
> location. Use it for display only. Never score it, and never compare it to the scorecard.

---

## 17. Figures

### 17a. Hindcast panels — 3 × 5, one per site × year

```bash
$PY -u scripts/plot_forecast_panel.py \
  --ensemble $G/members_m400.icechunk --recal_dir $G/recal_u \
  --base_year 2000 --years 2005,2010,2015,2020 \
  --n_windows 3 --size 768 --n_random 3 \
  --label "g1_foldb4 global hindcast, M=400" \
  --out_dir $G/panels_global --disable_wandb
```

**Cost: ~13 min. Produces 12 figures** — three sites × four target years, named
`forecast_panel_{busiest,mid-change,quietest}_{year}.png`.

Layout, 3 rows × 5 columns:

| row | content |
|---|---|
| 1 | observed Δ, central forecast Δ, lower 2.5% Δ, upper 97.5% Δ |
| 2 | three random members, the highest-change member, the lowest-change member |
| 3 | each of those members' change histogram, with the observation overlaid |

Row 2 brackets the ensemble with its two extremes rather than sampling its front, because three
arbitrary indices say nothing about spread. Row 3 exists because the eye cannot judge from a map
whether a field carries twice as much moderate change as reality — that is visible in a
histogram and invisible in a map.

`--n_windows 3` selects three sites spanning the observed-change spectrum, busiest to quietest,
requiring each candidate window to be at least 60% land.

### 17b. Forecast panels — 4 × 5, one per site

The forward product has no observation, so the hindcast layout cannot be used: row 1 opens with
observed Δ and row 3 overlays the observed histogram. A separate script drops those and uses the
rows for the four horizons instead.

```bash
$PY -u scripts/plot_forecast_horizons.py \
  --ensemble $G/forecast_members_m400.icechunk --recal_dir $G/forecast_recal \
  --base_year 2020 --years 2025,2030,2035,2040 --n_sites 3 --size 768 \
  --label "g1_foldb4 global forecast, M=400" \
  --out_dir $G/panels_forecast --disable_wandb
```

**Cost: ~7 min. Produces 3 figures**, `forecast_horizons_{busiest,mid-change,quietest}.png`.

Layout, 4 rows (2025 / 2030 / 2035 / 2040) × 5 columns:

| column | content |
|---|---|
| 1 | central Δ |
| 2 | lower 2.5% Δ |
| 3 | upper 97.5% Δ |
| 4 | interval width |
| 5 | **one member, the same one at every horizon** |

Three choices worth knowing when reading these:

- **Column 5 follows a single member across all four leads**, chosen as the highest-change
  member at 2040. Without truth there is nothing to compare a spread of members against, so the
  question shifts from "how wide is the ensemble" to "is a member a coherent story". The
  horizons within a member are coupled by the measured ρ, so a member developing fast by 2040
  must already be developing by 2025 — that is what the column shows.
- **Column 4 must be non-decreasing top to bottom** in every panel. That is the per-pixel
  horizon monotonicity of §8, visible as a property rather than asserted.
- **Each column carries its own colour scale, shared down the rows.** Rows must share or growth
  with lead time is normalised away — that is the comparison the figure exists to make. Columns
  must not: the central Δ and the upper bound differ by a factor of three or more, and a single
  scale drives the bound and member columns to solid colour at the long leads.

Sites are ranked on *predicted* Δ at the longest horizon, since there is no observed field.

---

## 18. Operational notes

**Long runs.** Launch with `setsid nohup ... &` and redirect output to a file. A plain `nohup`
inherits the process group and dies if the launching shell is killed.

**Never edit a shell script while it is running.** Bash reads scripts incrementally by byte
offset, so inserting lines mid-run makes it resume at a stale offset and execute a fragment.
Copy to a new name and launch that.

**`pgrep -f` / `pkill -f` match the shell running them.** Use `ps -eo pid,cmd | grep "patt[e]rn"`
and kill by pid.

**Resuming.** Every stage writes to disk, so a failure late in the chain does not require
re-running from the top. Re-issue only the failed command and what follows it.

**Regenerating an ensemble.** Every member's seed is recorded in the manifest beside the store,
so any ensemble can be deleted to reclaim space and rebuilt exactly.

### Measured cost summary

| stage | wall clock | peak RAM |
|---|---|---|
| 1. hindcast prediction (predict-only) | 152 min | — |
| 2. stitch | 40 min | — |
| 3. central diagnostics | 20 min | — |
| 4. residuals | 60 min | — |
| 5. AR(1) ρ | 5 min | — |
| 6. diagnostics | 2 h | — |
| 7a. band sweep (fit + score + audit) | 3 h | 116 GB |
| 7b. production width fit | 34 min | **98 GB** |
| 8. apply calibration | 68 min | 14 GB |
| 9. rebuild residuals | 60 min | — |
| 10. spectrum | 15 min | 45 GB |
| 11. u_bound sweep + marginal | 57 min | 72 GB |
| 12. smoke tier | 90 min | 69 GB |
| 13. generate ×2 + scorecard | **31 h** | 69 GB |
| 14. hindcast COGs | 35 min | — |
| 15. forward model end to end | 4 h | — |
| 16. mean display product (optional) | 5 h | — |
| 17. figures | 20 min | — |
