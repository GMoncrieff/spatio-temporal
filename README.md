# Spatio-Temporal Human Modification Forecasting

## Overview

This project implements a **distributional ConvLSTM** that forecasts the Human Modification (HM) index as a **full probability distribution per pixel**. At each of **four forecast horizons** (5, 10, 15, 20 years) the model emits a monotone quantile function `Q_h(u | x)` — the whole predictive distribution of HM at that pixel, from which any quantile, interval, mean or exceedance probability can be read. 

The production model, **E2c**, is trained with the **CRPS** (continuous ranked probability score), computed in closed form on a piecewise-linear quantile function with **15 parameters per horizon**. Its ConvLSTM trunk has a 12 px receptive field, its inputs are standardised with statistics computed over land only, and its static covariates include slope and aspect.

### Key Features

- **Full predictive distribution per pixel**: a 64-level quantile function at every horizon
- **Multi-horizon forecasting**: 5yr, 10yr, 15yr, 20yr ahead predictions
- **Closed-form CRPS training**: a piecewise-linear quantile head
- **Residual parameterisation**: the model predicts change on top of the current HM level, starting from persistence
- **Rich covariates**: HM history plus 10 dynamic covariates (HM stressors, GDP, population), 10 static variables (elevation, slope, aspect, climate, protected areas) and 12 channels of precomputed neighbourhood context
- **Location encoding**: spherical-harmonic positional embeddings for spatial awareness
- **Out-of-sample validation**: k=5 spatial-block cross-validation, scored against persistence on every land pixel of the globe
- **Delivered products**: Cloud-Optimised GeoTIFFs and icechunk quantile-function stores for a hindcast (2005–2020) and a forecast (2025–2040)

## Documentation
- **[Technical methodology](docs/methodology.pdf)** - The model in full: inputs, architecture, loss, training, prediction and evaluation
- **[Global production run](docs/global/global_production_e2c.md)** - Configuration, verification, scorecard, products and costs

**Topics covered:**
- **Input Data**: Dynamic variables, static covariates, neighbourhood context, location encoding
- **Model Architecture**: ConvLSTM trunk, the piecewise-linear quantile head
- **Loss Function**: Closed-form CRPS on the quantile function
- **Training**: Production configuration, weight averaging, CRPS-based checkpoint selection
- **Accuracy Assessment**: CRPS and RMSE skill against persistence, interval coverage, PIT calibration, distance-stratified scores
- **Prediction**: Tile-based processing, the quantile-function raster, COG and icechunk products


## Project Structure

```
spatio_temporal/
├── config/                         # Configuration files
│   ├── region_africa.geojson       # Development region (Africa, 63.1 Mpx)
│   ├── region_to_predict_large.geojson  # Global prediction region
│   ├── region_to_predict.geojson
│   ├── region_to_predict_small.geojson  # Small region for quick checks
│   ├── config.yaml                 # Fallback prediction region when --predict_region is omitted
│   └── sweep_*.yaml                # W&B hyperparameter sweep configs
│
├── data/
│   ├── raw/
│   │   └── hm_global/              # Global training data
│   │       ├── HM_YEAR_VARIABLE_1000.tiff    # Dynamic variables (1990-2020)
│   │       ├── hm_static_VARIABLE_1000.tiff  # Static covariates
│   │       ├── hm_static_terrain_{slope,aspsin,aspcos}_1000.tiff  # Slope and aspect of the elevation raster
│   │       ├── change_context_wYEAR_1000.tif # Precomputed distance-to-past-change
│   │       ├── hm_context_wYEAR_1000.tif     # Precomputed neighbourhood HM
│   │       └── fold_mask_b4_land_1000.tif    # k=5 spatial folds over land (512 px blocks)
│   └── conv_spline/
│       ├── norm_stats_E2c.json     # Normalisation statistics, exact over land
│       └── logs/                   # Run logs (every verifier reads these)
│
├── models/
│   ├── checkpoints/                # Lightning's default checkpoint directory (shared)
│   └── production/
│       └── E2c_global/             # The six production checkpoints (5 folds + forward model) + norm stats
│
├── src/
│   ├── models/                     # Model architecture
│   │   ├── spatiotemporal_predictor.py  # ConvLSTM trunk + quantile head decoder
│   │   ├── quantile_pwl.py         # Piecewise-linear quantile function + closed-form CRPS
│   │   ├── quantile_spline.py      # Knot grid and shape-head initialisation
│   │   ├── crps_loss.py            # CRPS loss
│   │   ├── change_weights.py       # Neighbourhood context channels
│   │   ├── lightning_module.py     # PyTorch Lightning wrapper
│   │   └── convlstm.py             # ConvLSTM implementation
│   ├── locationencoder/            # Spatial position encoding
│   ├── land.py                     # What counts as land (HM valid and not its 3.4e38 nodata)
│   ├── strata.py                   # Distance bands and other scoring strata
│   ├── stitch.py                   # Fold mosaic for the out-of-sample hindcast
│   ├── qf_diagnostics.py           # Quantile-function diagnostics
│   └── qf_plots.py                 # Scorecard figures
│
├── scripts/                        # Main scripts
│   ├── train_lightning.py          # Training + prediction pipeline
│   ├── torchgeo_dataloader.py      # Data loading with per-variable normalization
│   ├── conv_spline_base.sh         # Shared run arguments and log verifiers for the global runner
│   ├── run_global_model.sh         # Global runner: smoke, hindcast, stitch, score, forecast, export
│   ├── run_hindcast_folds.py       # k-fold training/prediction orchestrator
│   ├── score_distributional_model.py  # The scorecard
│   ├── build_detailed_scorecard.py # Detailed HTML scorecard for a scored hindcast
│   ├── export_products.py          # COGs + icechunk quantile-function stores
│   ├── export_observed.py          # Observed HM as an icechunk store aligned with the products
│   ├── verify_products.py          # Verifies delivered products against their sources
│   ├── prepare_change_context.py   # Builds change_context_w*.tif
│   ├── prepare_hm_context.py       # Builds hm_context_w*.tif
│   ├── prepare_terrain.py          # Builds slope and aspect from the elevation raster
│   ├── compute_norm_stats.py       # Exact land-only normalisation statistics
│   └── create_validity_mask.py     # Land mask and fold masks
│
├── tests/                          # Unit tests (pytest)
│
└── docs/                           # Documentation
    ├── methodology.pdf             # 📘 Technical methodology (source in methodology_src/)
    └── global/
        ├── global_production_e2c.md  # The production run: configuration, scorecard, products, costs
        ├── checkpoints/            # Provenance of the production checkpoints
        └── scores/                 # Global scorecards
```

## Setup

The development environment is a conda env named `spatio-temporal-dl`: Python 3.12, PyTorch 2.6, PyTorch Lightning 2.5, torchgeo 0.7, zarr 3, icechunk 2, xarray, and the GDAL command-line tools (`gdal_translate` builds the COGs).

```bash
# Create environment with Python 3.12
conda create -n spatio-temporal-dl python=3.12
conda activate spatio-temporal-dl

# Install PyTorch (choose based on your hardware)
pip install light-the-torch

# Install PyTorch with optimal hardware support (CPU/CUDA/MPS)
ltt install torch torchvision

# Install core dependencies via conda
conda install -c conda-forge \
    pytorch-lightning \
    torchmetrics \
    rasterio \
    gdal \
    shapely \
    pyproj \
    scipy \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn \
    xarray

# Install additional packages via pip
pip install torchgeo einops wandb "zarr>=3" icechunk
```

**Notes**:
- Python 3.12 is recommended for best compatibility
- `light-the-torch` (ltt)  automatically detects your hardware and installs the appropriate PyTorch version
- There is no `requirements.txt` or `environment.yml`: the commands above are the dependency list, and `setup.py` declares no install requirements

### 2. Data Structure

Data should be organized in `data/raw/hm_global/`:

**Dynamic variables** (time-varying, 1990-2020 at 5-year intervals):
```
HM_1990_AA_1000.tiff    # Target variable (Human Modification total)
HM_1990_AG_1000.tiff    # Agriculture
HM_1990_BU_1000.tiff    # Built-up areas
HM_1990_gdp_1000.tiff   # GDP
HM_1990_population_1000.tiff
... (repeat for 1995, 2000, 2005, 2010, 2015, 2020 and all HM stressors:
     AG, BU, EX, FR, HI, NS, PO, TI, gdp, population)
```

**Static variables** (time-invariant):
```
hm_static_ele_1000.tiff             # Elevation
hm_static_tas_1000.tiff             # Mean temperature
hm_static_tasmin_1000.tiff          # Minimum temperature
hm_static_pr_1000.tiff              # Precipitation
hm_static_dpi_dsi_1000.tiff
hm_static_iucn_strict_1000.tiff     # Protected areas (strict)
hm_static_iucn_nostrict_1000.tiff   # Protected areas (other)
```

The elevation raster declares no nodata and fills every row south of 56 S with `-32768`. That value is declared as a fill in the normalisation sidecar (`static_nodata`) and is read as sea level, not as an elevation.

**Derived inputs for the production model** (build once, in this order):
```bash
python scripts/prepare_terrain.py        # -> hm_static_terrain_{slope,aspsin,aspcos}_1000.tiff
                                         #    slope (deg) and sin/cos of aspect, Horn 3x3 on
                                         #    WGS84 metric spacing, 2 min for the globe
python scripts/compute_norm_stats.py --out data/conv_spline/norm_stats_E2c.json --terrain_covariates
                                         # exact moments over land for all 10 static channels,
                                         # HM and the covariates; ~4 min
python scripts/create_validity_mask.py --folds_only --k 5 --fold_block_chips 4 \
    --fold_min_valid_px 1 --fold_mask_out data/raw/hm_global/fold_mask_b4_land_1000.tif
                                         # a fold id for every 128 px chip holding any land
```

**Precomputed neighbourhood context** (one per base year; must be built on the full raster, never inside a training chip, or radii of 30–100 px saturate against the chip edge):
```bash
python scripts/prepare_change_context.py   # -> change_context_w{2000..2020}_1000.tif
python scripts/prepare_hm_context.py       # -> hm_context_w{2000..2020}_1000.tif
```

The HM rasters declare nodata as `3.4e38`, not NaN. Any code that reads them must map that sentinel to missing; an `isfinite` test alone treats the ocean as fully modified land.

### 3. W&B Setup (Optional but Recommended)

```bash
wandb login
# Or set WANDB_API_KEY environment variable
```

## Usage

### Training from Scratch

**The production model, E2c, is the argparse defaults of `train_lightning.py` plus two flags and its own normalisation sidecar**, and a hindcast also needs its fold mask:

```bash
--kernel_size 5 --terrain_covariates True --norm_stats_json data/conv_spline/norm_stats_E2c.json
--fold_mask data/raw/hm_global/fold_mask_b4_land_1000.tif      # k-fold hindcast only
```

Leaving any of them out does not fail: it silently trains a different model (a 3×3 trunk, seven static channels, or statistics sampled over the whole grid). The global runner (`MODEL=E2c`, below) sets all four and checks each one in the run's log.

#### Basic Training Run

```bash
python scripts/train_lightning.py --max_epochs 50 \
  --kernel_size 5 --terrain_covariates True --norm_stats_json data/conv_spline/norm_stats_E2c.json
```

#### Full Training with All Options

Every other option already defaults to its production value (see [Key Training Arguments](#key-training-arguments)). Add `--train_all_splits True` for the forward model, which holds no geography out:

```bash
python scripts/train_lightning.py \
  --kernel_size 5 --terrain_covariates True --norm_stats_json data/conv_spline/norm_stats_E2c.json \
  --run_large_area_prediction True \
  --predict_region config/region_africa.geojson
```

The run log prints what it actually built, read off the constructed model. These lines are the check that the configuration took effect:

```
Trunk:             ConvLSTM 4 layers x 64, kernel 5, dilation 1,1,1,1, receptive radius 12 px over 3 timesteps
Static channels:   10 (hm_static_ele_1000.tiff, ..., hm_static_terrain_aspcos_1000.tiff); module 10
Spline head:       family pwl, knots default14 (n=15, bins=14), slopes learned, free scale, 15 params/horizon
Loaded normalization stats from data/conv_spline/norm_stats_E2c.json
```

#### Quick Development Run (Smoke Test)

```bash
python scripts/train_lightning.py \
  --fast_dev_run \
  --batch_size 2 \
  --hidden_dim 16 \
  --run_large_area_prediction False \
  --disable_wandb
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_epochs` | 150 | Number of training epochs |
| `--train_chips` | 100 | Chips sampled per training epoch |
| `--val_chips` | 40 | Chips sampled per validation epoch |
| `--val_stride` | 1024 | Grid stride for validation sampling |
| `--batch_size` | 8 | Batch size for training/validation |
| `--hidden_dim` | 64 | ConvLSTM hidden dimension |
| `--num_layers` | 4 | Number of ConvLSTM layers |
| `--kernel_size` | 3 | ConvLSTM kernel. **E2c: 5**, a 12 px receptive radius (3 gives 6 px) |
| `--terrain_covariates` | False | Add slope, sin(aspect) and cos(aspect) as static channels. **E2c: True** |
| `--num_workers` | 3 | Data loader workers (0=single-threaded) |
| `--head_family` | `pwl` | Output head: `pwl` is the piecewise-linear quantile function |
| `--spline_knots` | `default14` | Quantile-level knot grid (15 knots, 14 bins) |
| `--free_scale` | True | Learn the quantile increments directly, with no anchor/scale factorisation |
| `--isqf_tails` / `--isqf_space` | False / `logit` | Learned exponential tails on `-log(1 - HM)` support (off) |
| `--central_residual` | True | Predict change on top of HM at the base year, starting from persistence |
| `--mu_mse_weight` | 0.0 | MSE on the mean; 0 means the model trains on CRPS alone |
| `--ssim_weight` / `--laplacian_weight` / `--histogram_weight` | 0.0 / 0.0 / 0.0 | Additional central-field losses, all off |
| `--checkpoint_monitor` | `val_crps` | Quantity used to select epochs |
| `--weight_avg_last` | 20 | Predict with the mean of the last N epochs' weights |
| `--context_radii` | `3,30,100` | Radii (px) for past-change occupancy context |
| `--hm_context_radii` / `--hm_context_stats` | `3,30,100` / `mean,max` | Neighbourhood-HM context channels |
| `--use_location_encoder` | true | Use spherical-harmonic position encoding (8 channels, 10 Legendre polynomials) |
| `--norm_stats_json` | `data/conv_spline/norm_stats.json` | Normalisation statistics. **E2c: `data/conv_spline/norm_stats_E2c.json`**. Loaded if present; if the file is missing, statistics are *sampled over the whole grid* and written there, which is not what E2c trained on |
| `--fold_mask` | None | k-fold mask for the hindcast. **E2c: `data/raw/hm_global/fold_mask_b4_land_1000.tif`** |
| `--train_all_splits` | False | Train on all data, holding no geography out (True for the forecast model) |
| `--seed` | 42 | Random seed |

### Making Predictions with Existing Checkpoint

**An E2c checkpoint needs E2c's flags.** `--checkpoint` restores the checkpoint's hyperparameters and then overrides the head and context settings from the command line, and the static channel list comes from the flags, so pass `--kernel_size 5 --terrain_covariates True --norm_stats_json data/conv_spline/norm_stats_E2c.json`. The load is non-strict, so a mismatch can show only as a `randomly initialised: N tensors` line in the log. A correct load prints `Checkpoint loaded with 0 warm-started convs` and no randomly initialised tensors.

**Normalisation statistics are not stored in the checkpoint.** They are `data/conv_spline/norm_stats_E2c.json`, with a copy in `models/production/E2c_global/`. The sidecar also declares elevation's `-32768` fill, which the reader drops, so it is part of the model, not a cache. Neither the checkpoints nor the sidecars are in git, so copy them together when moving to another machine.

#### Option 1: Load from W&B Artifact

```bash
python scripts/train_lightning.py \
  --checkpoint "model-xxxxxx:v0" \
  --max_epochs 0 \
  --run_large_area_prediction True \
  --predict_region config/region_to_predict.geojson
```

**How to find your W&B artifact name:**
1. Go to your W&B project: https://wandb.ai/glennwithtwons/spatio-temporal-convlstm
2. Click on a run
3. Go to "Artifacts" tab
4. Copy the artifact name (e.g., `model-xxxxxx:v0`)

#### Option 2: Load from Local Checkpoint File

The production forward model (trained on all data, base year 2020) is `models/production/E2c_global/final_foldNone_3573819.ckpt`:

```bash
python scripts/train_lightning.py \
  --checkpoint models/production/E2c_global/final_foldNone_3573819.ckpt \
  --kernel_size 5 --terrain_covariates True --norm_stats_json data/conv_spline/norm_stats_E2c.json \
  --max_epochs 0 \
  --run_large_area_prediction True \
  --run_full_set_evaluation False \
  --predict_region config/region_to_predict_small.geojson \
  --predict_final_year 2040
```

This recipe was checked on 2026-09-27: it loads with 0 warm-started convs and reproduces the delivered forecast over that region to a median difference of 5e-07 HM (99th percentile 8e-04). It is not bit-identical, because prediction tiles start at the region's own corner, 4 and 9 px off the global run's 64 px tile grid, so the overlap blending differs; the 12 px receptive field makes tile edges matter.

#### Prediction Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--predict_region` | None | Path to GeoJSON file defining prediction area |
| `--predict_final_year` | 2040 | Last target year: 2040 predicts from base 2020, 2020 from base 2000 |
| `--predict_stride` | 64 | Stride between tiles for overlap blending |
| `--predict_batch_size` | 32 | Tiles processed in parallel on GPU |
| `--predict_qf_dtype` | `float32` | Storage for the quantile raster (`int16` is coarser and quantises the forecast core) |
| `--predict_row_chunk` | 0 | Accumulate in row bands to bound memory; 512 on the global grid |
| `--predict_output_dir` | `data/predictions` | Where the GeoTIFFs are written |

#### Prediction Output

For each horizon (5yr, 10yr, 15yr, 20yr), four GeoTIFF files are created in `data/predictions/` (or `--predict_output_dir`):

```
data/predictions/
├── prediction_2025_qf_blended.tif       # The quantile function: 64 bands, one per level
├── prediction_2025_lower_blended.tif    # Q(0.025)
├── prediction_2025_central_blended.tif  # E[Q], the MEAN of the distribution
├── prediction_2025_upper_blended.tif    # Q(0.975)
├── prediction_2030_qf_blended.tif
... (and so on for 2030, 2035, 2040)
```

The `qf` raster is the product. Its 64 bands are quantile levels from 0.000248 to 0.9999, with 0.025, 0.5 and 0.975 as exact levels. The level grid, `head_family` and `head_params` are stored as raster tags. `lower`/`upper` are two of those levels, and `central` is the mean of the whole distribution, not its median.

**Reading the distribution** with the scorer's own reader, which applies the stored scale and maps nodata to NaN:
```python
import numpy as np
from score_distributional_model import read_qf   # with scripts/ on sys.path

u, Q = read_qf("data/predictions/prediction_2040_qf_blended.tif", row_off=0, n_rows=512)
# u: the 64 quantile levels; Q: [64, rows, width] float32, NaN outside land
lower = Q[np.argmin(np.abs(u - 0.025))]   # 0.025 and 0.975 are exact levels on the grid
upper = Q[np.argmin(np.abs(u - 0.975))]
width95 = upper - lower                   # width of the 95% predictive interval
```
Exceedance probabilities such as P(HM > 0.4) invert the quantile function; `scripts/export_products.py` derives them with the scorer's `pit`.

### Experiment Tracking (W&B)

All training runs are logged to: **https://wandb.ai/glennwithtwons/spatio-temporal-convlstm**

**Logged metrics include:**
- CRPS: `train_crps_total`, `val_crps` (the checkpoint monitor)
- Point accuracy: `train_mae_total`, `val_mae_total`
- Coverage calibration: `val_coverage_total` (target: ~95%)
- Visualizations: Multi-horizon predictions with uncertainty bounds

**Disable W&B:**
```bash
python scripts/train_lightning.py --disable_wandb
```

## Global Prediction

The global products are built by one runner, which selects the model from a table naming its flags, head family, parameter count, fold mask and normalisation sidecar together. **Always pass `MODEL=E2c`**: the runner holds other configurations and does not default to E2c.

```bash
MODEL=E2c ALLOW_GLOBAL=1 ./scripts/run_global_model.sh <stage>
# stages: args | smoke | hindcast | stitch | score | forecast
#         | export_hindcast | export_forecast | all
```

| Stage | What it does | Measured cost |
|-------|--------------|---------------|
| `smoke` | One epoch, one fold, a few hundred blocks through **every** stage on the real global grid; writes the receipt that pins the code, the flags and the input files | ~1 h |
| `hindcast` | k=5 folds, base 2000 → 2005/2010/2015/2020, both GPUs | ~97 min/fold, 4 h 52 total |
| `stitch` | Holdout mosaic: each pixel from the fold that never saw it | ~2 h 40 |
| `score` | The scorecard over every land pixel. **Run it alone**: it peaks at ~102 GiB | ~5 h |
| `forecast` | The `--train_all_splits` forward model, base 2020 → 2025/2030/2035/2040 | ~4 h |
| `export_*` | Five COGs per year + one icechunk store, then verification | ~2 h 50 (both in parallel) |

Measured on the 2026-09-26 E2c run. The forward model can run on one GPU while fold 5 and the stitch run, and the two exports run together; the scorer needs the machine to itself.

### Products

Written under `/mnt/hdd1/spatio-temporal/data/conv_spline/products/<MODEL>/`:

```
<MODEL>/hindcast/cogs/hm_{2005..2020}_{lower,mean,upper,gt01,gt04}.tif
<MODEL>/hindcast/hindcast_qf.icechunk     # (year, percentile, latitude, longitude), uint16
<MODEL>/forecast/cogs/hm_{2025..2040}_{lower,mean,upper,gt01,gt04}.tif
<MODEL>/forecast/forecast_qf.icechunk
observed/observed_hm.icechunk             # observed HM 1990-2020, (year, latitude, longitude)
```

- **COGs**: single-band Float32, EPSG:4326 at 0.009°, DEFLATE with `PREDICTOR=3`. `lower` = Q(0.025), `mean` = E[Q], `upper` = Q(0.975), `gt01` = P(HM > 0.1), `gt04` = P(HM > 0.4).
- **icechunk**: all 64 quantile levels per pixel, uint16 over [0, 1] (`scale_factor` 1/65535, fill 65535, data clamped to 65534). Chunks of 256 × 320 px hold the percentile axis whole, sharded 4 × 4.
- **Observed**: overall HM for 1990–2020 on the same grid, encoding and chunk footprint as the prediction stores, so observed and predicted tiles line up.
- The prediction stores do not yet declare `dimension_names`, so `xarray.open_zarr` cannot open them; read them with `zarr` directly. The observed store does declare them.

### Global scorecard (E2c, out of sample, 184,573,321 land pixels)

| Horizon | CRPS skill | RMSE skill | 50% coverage | 95% coverage |
|---------|------------|------------|--------------|--------------|
| +5 yr | 0.155 | 0.041 | 0.421 | 0.950 |
| +10 yr | 0.261 | 0.188 | 0.427 | 0.917 |
| +15 yr | 0.299 | 0.229 | 0.449 | 0.953 |
| +20 yr | 0.299 | 0.241 | 0.482 | 0.955 |

Skill is measured against **persistence** (HM unchanged from the base year).

**Known limitations**
- **The far-field intervals are too narrow at +10 yr.** Beyond 100 px from past change at +10 yr, the 95% interval covers 86.1% of observations, and the forecast loses to persistence there on CRPS (skill −1.45). Pooled over all land this shows as the +10 yr coverage of 0.917. Far-field coverage does not change monotonically with lead time (0.981, 0.861, 0.941 and 0.935 at +5 to +20 yr), because nothing constrains how the interval width grows from one horizon to the next. The distribution is bounded by its outermost knots (see [Model Architecture](#model-architecture)), so it cannot reach far into the tail.
- **The far-field point forecast does not beat persistence**: RMSE skill beyond 100 px is −0.014 at +5 yr and +0.008 at +20 yr.
- **The distribution is centred slightly low.** The PIT mean is 0.52–0.54 against 0.50, with recurring spikes near 0.22, 0.52, 0.77 and 0.95.
- **The central intervals are too narrow**: 50% coverage is 0.42–0.48 and 80% coverage 0.76–0.77.
- **The lower far tail is over-populated**: P(u < 0.001) is 0.0116–0.0125 against 0.001.

The full scorecard, stratified by distance to past change, is `docs/global/scores/scorecard_detailed_g_E2c_hind.html`; see `docs/global/global_production_e2c.md`.

## Data: Human Modification (HM)

We forecast the Human Modification (HM) index, a spatially explicit measure of anthropogenic modification across landscapes.

- **Source paper** (Nature Scientific Data):
  - "Theobald, D. M., Oakleaf, J. R., Moncrieff, G., Voigt, M., Kiesecker, J., & Kennedy, C. M. (2025). Global extent and change in human modification of terrestrial ecosystems from 1990 to 2022. Scientific Data, 12(1), 606."
  - paper available at https://www.nature.com/articles/s41597-025-04892-2
  - data available at https://zenodo.org/records/16907328

**Key characteristics:**
- **Temporal cadence**: 5-year intervals (1990, 1995, ..., 2020)
- **Model inputs**: 3 most recent HM timesteps + 10 dynamic covariates + 10 static variables (7 layers + slope and aspect) + 12 neighbourhood-context channels + location encoding
- **Target variable**: AA (total Human Modification)
- **Data range**: [0, 1]
- **Coverage**: Near-global extent, 17111 × 40000 grid at 0.009° (~1 km), 184.6 M land pixels

The model reads a three-step window `(t-10, t-5, t)`, so with HM available for 1990–2020 the earliest window is (1990, 1995, 2000) and the out-of-sample hindcast targets are 2005–2020.

## Model Architecture

### Quantile-Function Head

A ConvLSTM trunk (4 layers, 64 hidden, 5×5 kernels) encodes the input window, and the neighbourhood context enters the trunk directly. The trunk's receptive radius is 12 px: every convolution on the path from an input pixel to the last layer's last state adds 2 px, over 4 layers and 2 recurrent steps. A decoder then emits the parameters of one quantile function per horizon:

```
ConvLSTM trunk (41 input channels per step: 11 dynamic, 10 static, 12 context, 8 location)
           ↓
   Shared representation
           ↓
  4 horizons × 15 parameters
           ↓
 ┌────────────────────────────────────────────────┐
 │ 1 location    — the median, Q(0.5), built on   │
 │                 the base-year HM               │
 │ 14 increments — positive steps between the     │
 │                 15 knots of a piecewise-linear │
 │                 quantile function Q(u), from   │
 │                 u = 0 to u = 1                 │
 └────────────────────────────────────────────────┘
           ↓
  Q_h(u | x), monotone by construction, clamped to [0, 1]
```

The increments are positive, so the quantile function can never cross itself. With `--central_residual` the ladder sits on top of the base-year HM, so an untrained model starts from persistence. That is **15 parameters per horizon**. The knots span u = 0 to u = 1 and there are no tail parameters, so the predicted distribution is bounded by its outermost knots, Q(0) and Q(1).

### Loss Function

```
All horizons receive:
  CRPS(Q_h, y), in closed form on the piecewise-linear quantile function

Selected on:  val_crps
Predicted with:  the mean of the last 20 epochs' weights
```

The CRPS integrates the pinball loss over every quantile level at once, so a single proper scoring rule trains the location, spread and tails of the distribution together. For a piecewise-linear `Q(u)` that integral is exact, so no quadrature is involved. The code's other loss terms (MSE on the mean, SSIM, Laplacian pyramid, histogram) are all weighted 0.

## Contributing

Pull requests and issues are welcome. Please open an issue to discuss significant changes beforehand.

## Citation

If you use this code, please cite the HM dataset:

```bibtex
@article{theobald2025global,
  title={Global extent and change in human modification of terrestrial ecosystems from 1990 to 2022},
  author={Theobald, David M and Oakleaf, James R and Moncrieff, Glenn and Voigt, Maria and Kiesecker, Joe and Kennedy, Christina M},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={606},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
