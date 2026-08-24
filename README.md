# Spatio-Temporal Human Modification Forecasting

## Overview

This project implements a **ConvLSTM-based spatio-temporal forecasting** pipeline with **quantile regression** for uncertainty quantification of the Human Modification (HM) index. The model predicts future HM values at **four forecast horizons** (5, 10, 15, 20 years) with **three predictions per horizon**: lower quantile (2.5%), central estimate, and upper quantile (97.5%).

The central head predicts the *change* from t0 rather than the level, the interval is built
structurally around it, and held-out geography comes from five-fold cross-validation on
contiguous 512 px territories. Measured globally on 184.5M out-of-sample pixels, skill against
persistence is **+0.043 / +0.174 / +0.212 / +0.222** at +5/+10/+15/+20 years, with pooled
interval coverage of 94.9-95.6% against a 95% target and no calibration layer.

If you are coming from `main`, read **[docs/model_update.md](docs/model_update.md)** — the model
and the training recipe both changed, and `main` checkpoints will not load.

### Key Features

- **Multi-horizon forecasting**: 5yr, 10yr, 15yr, 20yr ahead predictions
- **Residual central head**: predicts the *change* from t0 through a zero-initialised skip, so training starts at exact persistence
- **Structurally monotone intervals**: quantile heads emit cumulative non-negative half-widths around a detached centre, so `lower <= central <= upper` always holds and spread cannot narrow with lead time
- **Long-range change context**: 8 channels derived from precomputed distance-to-past-change, reaching well beyond the trunk's ~10 px receptive radius
- **Spatial k-fold cross-validation**: five contiguous 512 px fold territories, larger than the ~300 px residual correlation length, so held-out skill is honest
- **Rich covariates**: 11 dynamic variables (HM components + GDP + population) and 7 static variables (elevation, climate, protected areas)
- **Shared normalization sidecar**: one `norm_stats.json` across every fold and the production model
- **Cloud-optimized outputs**: verified COGs with overviews
- **W&B integration**: Comprehensive experiment tracking and visualization

## Documentation

- **[What changed from `main`](docs/model_update.md)** - the model and fitting differences, why each was made, and what it means for anything trained on `main`
- **[Model architecture and training](docs/model_architecture.md)** - the shipped ConvLSTM: heads, context channels, k-fold recipe, what the flags mean, and which flags are settled dead ends
- **[Fitting and running the model](docs/fitting_running_model.md)** - the runbook: environment, derived inputs, every script in order, the COG step, and the eight gates that prove a code change
- **[Technical Documentation](docs/simple_model_architecture_and_training.md)** - superseded; the pre-k-fold model, kept for history
- **[Pinball loss gradient isolation](docs/pinball_loss_gradient_isolation.md)** - superseded for the shipped configuration, where the isolation is structural; still describes the path taken when `--monotone_quantile_width` is off

## Project Structure

```
spatio_temporal/
├── config/
│   ├── config.yaml                      # only inference.region_geojson is read
│   ├── sweep_config.yaml                # W&B hyperparameter sweep config
│   ├── region_to_predict_large.geojson  # the global extent -- what products use
│   ├── region_to_predict.geojson        # southern Africa, for fast iteration
│   └── region_smoke_aligned.geojson     # tile-phase-aligned window for gate S5/S6
│
├── data/
│   ├── norm_stats.json                  # normalisation sidecar, shared by every model
│   ├── conv_update/PROVENANCE.json      # checkpoints, flags and sources for the products
│   ├── global/ -> (bulk storage)        # stitched rasters, forecast rasters, the COGs
│   └── raw/hm_global/                   # global training data
│       ├── HM_YEAR_VARIABLE_1000.tiff   # dynamic variables (1990-2020)
│       ├── hm_static_VARIABLE_1000.tiff # static covariates
│       ├── change_context_wYEAR_1000.tif# 2 bands: past change, distance to past change
│       ├── split_mask_1000.tif          # 70/10/10/10 train/val/test/calib
│       └── fold_mask_b4_1000.tif        # five contiguous 512 px fold territories
│
├── src/
│   ├── models/
│   │   ├── spatiotemporal_predictor.py  # ConvLSTM trunk + central and quantile heads
│   │   ├── lightning_module.py          # PyTorch Lightning wrapper, manual optimization
│   │   ├── convlstm.py                  # ConvLSTM cell (optional per-layer dilation)
│   │   ├── change_weights.py            # the 8 change-context channels, distance bands
│   │   ├── pinball_loss.py              # quantile regression loss
│   │   ├── losses.py                    # Laplacian pyramid loss
│   │   └── histogram_loss.py
│   ├── prediction/
│   │   └── stitch.py                    # fold stitching: holdout and mean modes
│   ├── locationencoder/                 # spherical-harmonic position encoding
│   ├── evaluation/ preprocessing/ utils/
│
├── scripts/
│   ├── train_lightning.py               # training + prediction pipeline
│   ├── run_hindcast_folds.py            # k-fold orchestration: train/predict, then stitch
│   ├── torchgeo_dataloader.py           # data loading, split/fold filtering, normalisation
│   ├── create_validity_mask.py          # split mask and k-fold mask construction
│   ├── prepare_change_context.py        # the change-context rasters (build once)
│   ├── make_cogs.py                     # COG conversion with a verify pass
│   ├── make_product_cogs.sh             # the three product sets
│   ├── check_checkpoint_fingerprint.py  # prove a checkpoint is the shipped architecture
│   ├── diagnose_central_field.py        # skill vs persistence, stratified
│   └── compare_central_runs.py          # compare two runs, flags byte-identical results
│
├── tests/                               # pytest, flat, synthetic tensors
│
├── docs/
│   ├── model_architecture.md            # the shipped model
│   ├── fitting_running_model.md         # the runbook and the eight gates
│   ├── simple_model_architecture_and_training.md  # superseded
│   └── pinball_loss_gradient_isolation.md         # superseded for the shipped config
│
└── setup.py
```

## Setup

```bash
# Create environment with Python 3.12
conda create -n hmforecast python=3.12
conda activate hmforecast

# Install PyTorch (choose based on your hardware)
pip install light-the-torch

# Install PyTorch with optimal hardware support (CPU/CUDA/MPS)
ltt install torch torchvision

# Install core dependencies via conda
conda install -c conda-forge \
    pytorch-lightning \
    torchmetrics \
    rasterio \
    shapely \
    pyproj \
    scipy \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn

# Install additional packages via pip
pip install torchgeo einops wandb
```

**Notes**:
- Python 3.12 is recommended for best compatibility
- `light-the-torch` (ltt)  automatically detects your hardware and installs the appropriate PyTorch version

### 2. Data Structure

Data should be organized in `data/raw/hm_global/`:

**Dynamic variables** (time-varying, 1990-2020 at 5-year intervals):
```
HM_1990_AA_1000.tiff    # Target variable (Human Modification total)
HM_1990_AG_1000.tiff    # Agriculture
HM_1990_BU_1000.tiff    # Built-up areas
HM_1990_gdp_1000.tiff   # GDP
HM_1990_population_1000.tiff
... (repeat for 1995, 2000, 2005, 2010, 2015, 2020 and all HM stressors)
```

**Static variables** (time-invariant):
```
hm_static_ele_1000.tiff          # Elevation
hm_static_tas_1000.tiff          # Mean temperature
hm_static_pr_1000.tiff           # Precipitation
hm_static_iucn_strict_1000.tiff  # Protected areas
...
```

**Derived inputs** — build these once, before training. They are not optional: the change
context supplies the 8 channels that take the heads from 64 to 72 input channels, and the fold
mask is what makes held-out geography honest.

```bash
# Change context: 2 bands per window (signed past change, distance to past change).
# Must be computed on the FULL raster -- inside a 128 px chip the 100 px radius
# saturates into "is there any change anywhere in this chip".
python scripts/prepare_change_context.py --out_dir data/raw/hm_global
# -> change_context_w{2000,2005,2010,2015,2020}_1000.tif  (~2.2 GB each)

# Split mask (70/10/10/10) and the k-fold mask.
python scripts/create_validity_mask.py
# -> validity_mask_1000.tif, split_mask_1000.tif

python scripts/create_validity_mask.py --folds_only --k 5 --fold_block_chips 4 \
  --fold_mask_out data/raw/hm_global/fold_mask_b4_1000.tif
```

`--fold_block_chips 4` gives 4 x 128 = 512 px fold territories. That has to exceed the ~300 px
residual correlation length; at `1` the folds are a 128 px checkerboard and every held-out tile
sits inside the correlation length of tiles its own model trained on.

All rasters share one grid: **17,111 x 40,000 px, EPSG:4326, 0.009 deg (~1 km)**, origin
(-180, 83.997). Any mismatch fails loudly at the first read.

### 3. W&B Setup (Optional but Recommended)

```bash
wandb login
# Or set WANDB_API_KEY environment variable
```

## Usage

### Training from Scratch

#### Basic Training Run

```bash
python scripts/train_lightning.py \
  --max_epochs 50 \
  --batch_size 8 \
  --hidden_dim 64 \
  --num_layers 4 \
  --central_residual True --central_context True \
  --monotone_quantile_width True --quantile_context True
```

Even the minimal invocation carries the four architecture flags. Without them this is a
different model — see "Full Training with All Options" below.

#### Full Training with All Options

**The four architecture flags are not optional, and argparse defaults are not the shipped
configuration** — `--central_residual` defaults to `False`, under which the central head
predicts absolute HM and reconstructs the baseline through the trunk. A k=5 run launched on
defaults scored h=5 skill −0.50 against +0.13 with the flags set.

```bash
ARCH="--central_residual True --central_context True \
      --monotone_quantile_width True --quantile_context True"
HP="--hidden_dim 64 --num_layers 4 --kernel_size 3 --locenc_out_channels 8 \
--locenc_legendre_polys 10 --ssim_weight 0.2 --laplacian_weight 0.3 \
--histogram_weight 1.0 --histogram_lambda_w2 0.1 --histogram_warmup_epochs 0"

# One cross-validation fold: fold 1 held out of training entirely, fold 2 for validation.
python scripts/train_lightning.py \
  --max_epochs 150 --train_chips 100 --val_chips 40 --val_stride 1024 \
  --batch_size 8 --num_workers 3 --devices 1 --seed 42 \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif --exclude_fold 1 --n_folds 5 \
  --norm_stats_json data/norm_stats.json \
  $HP $ARCH
```

Verify the fingerprint in the log before trusting the run:

```
FOLD-CV MODE: holding out fold 1 (out-of-sample), validating on fold 2
  ConvLSTM grad norm: 0.000000 (from central loss only)
```

`ConvLSTM grad norm: 0.000000` is what `--central_residual` looks like — the trunk gets no
gradient from the central loss. On defaults it reads ~0.04 with an initial central loss 20x
higher.

#### The production model

The forward model has no held-out geography to protect, so `--train_all_splits True` uses every
valid chip instead of the 70% train split. It is ignored under `--exclude_fold`, so it can never
pull a held-out fold back into training.

```bash
python scripts/train_lightning.py \
  --max_epochs 150 --train_chips 100 --val_chips 40 --val_stride 1024 \
  --batch_size 8 --num_workers 3 --devices 1 --seed 42 --train_all_splits True \
  --norm_stats_json data/norm_stats.json $HP $ARCH
```

The log must print `PRODUCTION MODE: training on EVERY chip in the split mask` and must **not**
print `FOLD-CV MODE`.

#### Quick Development Run (Smoke Test)

```bash
python scripts/train_lightning.py \
  --max_epochs 2 --train_chips 8 --val_chips 8 --batch_size 2 --num_workers 2 \
  --val_stride 1024 --disable_wandb \
  --run_full_set_evaluation False --run_large_area_prediction False \
  --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif --exclude_fold 1 \
  --norm_stats_json data/norm_stats.json $HP $ARCH
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--central_residual` | false | **Set true.** Central head predicts change from t0 via a zero-init skip |
| `--central_context` | false | **Set true.** 8 change-context channels into the central heads |
| `--quantile_context` | false | **Set true.** Same 8 channels into the quantile heads |
| `--monotone_quantile_width` | false | **Set true.** Cumulative half-widths around a detached centre |
| `--fold_mask` | None | Fold-id raster; with `--exclude_fold` it replaces the split mask |
| `--exclude_fold` | None | Fold held out of both training and validation |
| `--n_folds` | 5 | Number of folds |
| `--val_fold` | None | Validation fold; defaults to `(exclude_fold % n_folds) + 1` |
| `--train_all_splits` | false | Train on every valid chip. Ignored under `--exclude_fold` |
| `--norm_stats_json` | None | Normalisation sidecar, shared by every fold and the production model |
| `--val_stride` | `--stride` | Grid stride for val/test. At 1024, fold 2 of the b4 mask gives 131 chips; production split 2 gives 58 |
| `--max_epochs` | 100 | Number of training epochs (0 = predict only) |
| `--train_chips` | 200 | Chips sampled per training epoch |
| `--batch_size` | 8 | Batch size for training/validation |
| `--hidden_dim` | 64 | ConvLSTM hidden dimension |
| `--num_layers` | 2 | ConvLSTM layers (**production uses 4**) |
| `--checkpoint_monitor` | `val_total_loss` | Includes pinball, so a quantile-only change also moves the selected central field. Use `val_central_loss` for central-only A/Bs |
| `--seed` | 42 | Not sufficient for determinism — cuDNN algorithm selection varies run to run |

Many further flags exist from a 22-experiment sweep that adopted none of them. They all default
to off; see `docs/model_architecture.md` §9 before turning one on.

### Making Predictions with Existing Checkpoint

The architecture flags are **still required** under `--max_epochs 0`: the model is built from
the CLI args before the state dict is loaded into it. Without them the load fails with
`size mismatch for model.central_heads.0.0.weight: [64, 72, 3, 3] vs [64, 64, 3, 3]`.

Check a checkpoint first:

```bash
python scripts/check_checkpoint_fingerprint.py \
  --manifest data/conv_update/PROVENANCE.json \
  --control artifacts/model-khrpthgy:v0/model.ckpt
```

The control must be **REJECTED**; without it the check has not been shown to reject anything.

#### Forecast 2025-2040 from a production checkpoint

```bash
python scripts/train_lightning.py --max_epochs 0 --devices 1 \
  --checkpoint spatio-temporal-convlstm/6zkppztt/checkpoints/epoch=12-step=169.ckpt \
  --norm_stats_json data/norm_stats.json \
  --run_full_set_evaluation False --run_large_area_prediction True \
  --predict_region config/region_to_predict_large.geojson \
  --predict_stride 64 --predict_batch_size 32 \
  --predict_output_dir data/global/forecast_preds \
  --predict_input_years 2010,2015,2020 --predict_output_prefix "" \
  $HP $ARCH
```

#### Fold hindcast, all five folds across both GPUs

```bash
python scripts/run_hindcast_folds.py --stage train --max_epochs 0 \
  --folds 1,2,3,4,5 --gpus 0,1 --windows 2000 \
  --region config/region_to_predict_large.geojson \
  --output_root data/global --norm_stats_json data/norm_stats.json \
  --fold_checkpoints "$CKPTS" --extra_train_args "$ARCH"

python scripts/run_hindcast_folds.py --stage stitch --stitch_mode holdout \
  --folds 1,2,3,4,5 --windows 2000 --output_root data/global --disable_wandb
```

#### Prediction Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--predict_region` | None | GeoJSON defining the prediction area |
| `--predict_stride` | 64 | Stride between tiles for overlap blending |
| `--predict_batch_size` | 16 | Tiles processed in parallel on GPU |
| `--predict_input_years` | None | e.g. `2010,2015,2020`; targets are +5/10/15/20 |
| `--predict_output_dir` | `data/predictions` | Where rasters are written |
| `--predict_output_prefix` | None | Prepended to every filename, e.g. `fold1_w2000_` |
| `--predict_max_target_year` | None | Skip horizons past the last observed year |
| `--predict_restrict_mask` / `--predict_restrict_values` | None | Predict only the given fold's territory (~5x cheaper, exact for kept pixels) |
| `--predict_all_windows` | false | All four hindcast windows on one checkpoint load |

#### Prediction Output

For each horizon, three single-band float32 GeoTIFFs with `nodata=NaN`:

```
data/global/forecast_preds/
├── prediction_2025_lower_blended.tif    # Lower 2.5% quantile
├── prediction_2025_central_blended.tif  # Central estimate
├── prediction_2025_upper_blended.tif    # Upper 97.5% quantile
... (and so on for 2030, 2035, 2040)
```

**Uncertainty mapping:**
```python
uncertainty = upper - lower  # Width of the 95% interval
```

### Published products

Three sets of twelve cloud-optimized GeoTIFFs under `data/global/`, all **raw model output with
no recalibration** — this branch has no calibration layer.

```bash
./scripts/make_product_cogs.sh
```

| set | directory | years | inputs | sample status |
|---|---|---|---|---|
| out-of-fold hindcast | `cogs_hindcast/hm_hindcast_{year}_{q}.tif` | 2005, 2010, 2015, 2020 | 1990/1995/2000 | out-of-sample everywhere; **the only set that may be scored** |
| fold-mean hindcast | `cogs_hindcast_mean/hm_hindcast_mean_{year}_{q}.tif` | 2005, 2010, 2015, 2020 | 1990/1995/2000 | **in-sample** — four of five folds trained on any pixel. Seamless; display only |
| forecast | `cogs_forecast/hm_forecast_{year}_{q}.tif` | 2025, 2030, 2035, 2040 | 2010/2015/2020 | production model; no fold, no mosaic, no seam |

`{q}` is `central`, `lower` or `upper`. The fold-mean set is seamless because it averages all
five folds at every pixel, which also makes its interval narrower than any single fold's — it
discards the between-fold spread rather than adding it. Never score it.

Provenance for all three is in `data/conv_update/PROVENANCE.json`.

### Experiment Tracking (W&B)

All training runs are logged to: **https://wandb.ai/glennwithtwons/spatio-temporal-convlstm**

**Logged metrics include:**
- Per-horizon losses: `train/val_mae_5yr`, `train/val_ssim_loss_10yr`, etc.
- Quantile losses: `train/val_pinball_lower_total`, `train/val_pinball_upper_total`
- Coverage calibration: `val_coverage_total` (target: ~95%)
- Visualizations: Multi-horizon predictions with uncertainty bounds

**Disable W&B:**
```bash
python scripts/train_lightning.py --disable_wandb
```

## Data: Human Modification (HM)

We forecast the Human Modification (HM) index, a spatially explicit measure of anthropogenic modification across landscapes.

- **Source paper** (Nature Scientific Data):
  - "Theobald, D. M., Oakleaf, J. R., Moncrieff, G., Voigt, M., Kiesecker, J., & Kennedy, C. M. (2025). Global extent and change in human modification of terrestrial ecosystems from 1990 to 2022. Scientific Data, 12(1), 606."
  - paper available at https://www.nature.com/articles/s41597-025-04892-2
  - data available at https://zenodo.org/records/16907328

**Key characteristics:**
- **Temporal cadence**: 5-year intervals (1990, 1995, ..., 2020)
- **Model inputs**: 3 most recent HM timesteps + 11 dynamic covariates + 7 static variables
- **Target variable**: AA (total Human Modification)
- **Data range**: [0, 1]
- **Coverage**: Near-global extent

## Model Architecture

See **[docs/model_architecture.md](docs/model_architecture.md)** for the full account. In brief:

### Trunk

4-layer ConvLSTM, `hidden_dim 64`, 3x3 kernels, fed 11 dynamic + 7 static channels plus 8
optional location-encoder channels. Its receptive radius is only ~10 px, which is why the
long-range change context is supplied as an explicit covariate rather than learned.

### Heads

12 heads, 3 per horizon, all reading the trunk's last hidden state concatenated with the 8
change-context channels (in-channels 72, not 64):

```
ConvLSTM -> last hidden state (+ 8 change-context channels)
           |
    +------+------+
    v      v      v
 Lower  Central  Upper
  head    head    head
    |      |       |
   2.5%  best    97.5%
         est.
```

**Central head** (`hidden_dim -> hidden_dim -> 1`): predicts the *change* from t0, added to HM
at t0 through a zero-initialised skip. Training therefore starts at exact persistence, and
"nothing happens" — the right answer for 53-73% of pixels — costs nothing to say.

**Quantile heads** (`hidden_dim -> hidden_dim/2 -> 1`): emit non-negative half-width
*increments* through a softplus, accumulated across horizons around the **detached** central
prediction. Two properties follow structurally rather than by clipping: `lower <= central <=
upper` always holds, and spread cannot narrow with lead time.

### Loss

```
Central objective (trunk + central heads):
  MSE + 0.2*SSIM + 0.3*Laplacian + 1.0*Histogram

Quantile objective (quantile heads only):
  Pinball(q=0.025) + Pinball(q=0.975)
```

The two are gradient-isolated: the central forecast enters the quantile heads detached, so the
pinball objective shapes the interval without moving the central field. The fingerprint is the
startup line `ConvLSTM grad norm: 0.000000`.

### Skill

Measured globally on held-out geography, against **persistence** — the median 20-year HM change
is 0.0001, so pooled RMSE can look fine while the model loses to predicting no change at all.

| horizon | skill vs persistence |
|---|---|
| +5 yr | +0.110 |
| +10 yr | +0.183 |
| +15 yr | +0.226 |
| +20 yr | +0.222 |

### Known defect

`P(dHM > 0.05)` beyond 100 px from past change reads about 0.034 of observed. This is a property
of the quantile heads — far-band half-widths are ~0.004, so +0.05 is ~12 half-widths out.

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
