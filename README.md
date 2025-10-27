# Spatio-Temporal Human Modification Forecasting

## Overview

This project implements a **ConvLSTM-based spatio-temporal forecasting** pipeline with **quantile regression** for uncertainty quantification of the Human Modification (HM) index. The model predicts future HM values at **four forecast horizons** (5, 10, 15, 20 years) with **three predictions per horizon**: lower quantile (2.5%), central estimate, and upper quantile (97.5%).

### Key Features

- **Multi-horizon forecasting**: 5yr, 10yr, 15yr, 20yr ahead predictions
- **Uncertainty quantification**: Probabilistic predictions with calibrated confidence intervals
- **Independent prediction heads**: Separate neural networks for central estimate vs. uncertainty bounds
- **Rich covariates**: 11 dynamic variables (HM components + GDP + population) and 7 static variables (elevation, climate, protected areas)
- **Per-variable normalization**: Handles vastly different scales (GDP in billions, HM in 0-1 range)
- **Location encoding**: Learnable positional embeddings for spatial awareness
- **W&B integration**: Comprehensive experiment tracking and visualization

## Documentation
- **[Technical Documentation](docs/simple_model_architecture_and_training.md)** - Detailed guide with implementation code

**Topics covered:**
- **Input Data**: Dynamic variables, static covariates, location encoding
- **Data Transformations**: Per-variable normalization, NaN handling, data leakage prevention
- **Model Architecture**: ConvLSTM, independent prediction heads, design decisions
- **Loss Functions**: MSE, SSIM, Laplacian, Histogram (with warmup), Pinball loss
- **Training**: Hyperparameters, optimization, early stopping, W&B tracking
- **Accuracy Assessment**: Evaluation metrics, validation protocols, visualization
- **Prediction**: Tile-based processing, output formats, GIS integration

## Project Structure

```
spatio_temporal/
├── config/                         # Configuration files
│   ├── config.yaml                 # Main training config
│   ├── sweep_config.yaml           # W&B hyperparameter sweep config
│   ├── region_to_predict.geojson   # Region boundaries for prediction
│   └── region_to_predict_small.geojson
│
├── data/
│   └── raw/
│       └── hm_global/              # Global training data
│           ├── HM_YEAR_VARIABLE_1000.tiff  # Dynamic variables (1990-2020)
│           └── hm_static_VARIABLE_1000.tiff # Static covariates
│
├── src/
│   ├── models/                     # Model architecture
│   │   ├── spatiotemporal_predictor.py  # Main ConvLSTM + independent heads
│   │   ├── lightning_module.py     # PyTorch Lightning wrapper
│   │   ├── convlstm.py            # ConvLSTM implementation
│   │   ├── pinball_loss.py        # Quantile regression loss
│   │   ├── laplacian_pyramid_loss.py
│   │   └── histogram_loss.py
│   ├── locationencoder/           # Spatial position encoding
│   ├── evaluation/                # Evaluation utilities
│   ├── preprocessing/             # Data preprocessing
│   └── utils/                     # Helper functions
│
├── scripts/                       # Main scripts
│   ├── train_lightning.py        # Training + prediction pipeline
│   ├── torchgeo_dataloader.py   # Data loading with per-variable normalization
│   └── create_validity_mask.py  # NaN/no-data handling
│
├── tests/                        # Unit tests
│
├── docs/                         # Documentation
│   └── model_architecture_and_training.md  # 📘 Complete technical guide
│
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
└── setup.py                      # Package setup
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
  --num_layers 2
```

#### Full Training with All Options

```bash
python scripts/train_lightning.py \
  --max_epochs 100 \
  --train_chips 500 \
  --val_chips 100 \
  --batch_size 8 \
  --hidden_dim 64 \
  --num_layers 2 \
  --num_workers 4 \
  --ssim_weight 2.0 \
  --laplacian_weight 1.0 \
  --histogram_weight 0.67 \
  --histogram_warmup_epochs 20 \
  --use_location_encoder true \
  --locenc_out_channels 8 \
  --predict_after_training true \
  --predict_region config/region_to_predict.geojson
```

#### Quick Development Run (Smoke Test)

```bash
python scripts/train_lightning.py \
  --fast_dev_run \
  --batch_size 2 \
  --hidden_dim 16
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_epochs` | 100 | Number of training epochs |
| `--train_chips` | 200 | Chips sampled per training epoch |
| `--val_chips` | 40 | Chips sampled per validation epoch |
| `--batch_size` | 8 | Batch size for training/validation |
| `--hidden_dim` | 64 | ConvLSTM hidden dimension |
| `--num_layers` | 2 | Number of ConvLSTM layers |
| `--num_workers` | 0 | Data loader workers (0=single-threaded) |
| `--ssim_weight` | 2.0 | Weight for SSIM loss |
| `--laplacian_weight` | 1.0 | Weight for Laplacian pyramid loss |
| `--histogram_weight` | 0.67 | Weight for histogram loss |
| `--use_location_encoder` | true | Use learnable spatial position encoding |
| `--predict_after_training` | true | Run prediction after training completes |

### Making Predictions with Existing Checkpoint

#### Option 1: Load from W&B Artifact

```bash
python scripts/train_lightning.py \
  --checkpoint "model-xxxxxx:v0" \
  --max_epochs 0 \
  --predict_after_training true \
  --predict_region config/region_to_predict.geojson \
  --predict_stride 64 \
  --predict_batch_size 16
```

**How to find your W&B artifact name:**
1. Go to your W&B project: https://wandb.ai/glennwithtwons/spatio-temporal-convlstm
2. Click on a run
3. Go to "Artifacts" tab
4. Copy the artifact name (e.g., `model-xxxxxx:v0`)

#### Option 2: Load from Local Checkpoint File

```bash
python scripts/train_lightning.py \
  --checkpoint "checkpoints/best_model.ckpt" \
  --max_epochs 0 \
  --predict_after_training true \
  --predict_region config/region_to_predict.geojson
```

#### Prediction Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--predict_region` | None | Path to GeoJSON file defining prediction area |
| `--predict_stride` | 64 | Stride between tiles for overlap blending |
| `--predict_batch_size` | 16 | Tiles processed in parallel on GPU |

#### Prediction Output

For each horizon (5yr, 10yr, 15yr, 20yr), three GeoTIFF files are created:

```
data/predictions/REGION_NAME/
├── prediction_2025_lower_blended.tif    # Lower 2.5% quantile
├── prediction_2025_central_blended.tif  # Central estimate
├── prediction_2025_upper_blended.tif    # Upper 97.5% quantile
├── prediction_2030_lower_blended.tif
├── prediction_2030_central_blended.tif
├── prediction_2030_upper_blended.tif
... (and so on for 2035, 2040)
```

**Uncertainty mapping:**
```python
uncertainty = upper - lower  # Width of 95% confidence interval
```

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

### Independent Prediction Heads

The model uses **12 separate neural network heads** (3 per horizon):

```
ConvLSTM → Shared representation
           ↓
    ┌──────┼──────┐
    ↓      ↓      ↓
 Lower  Central Upper
  Head    Head   Head
    ↓      ↓      ↓
  2.5%   Best   97.5%
         Est.
```

**Central heads**: Full-size (hidden_dim → hidden_dim → 1), optimized for accuracy + spatial patterns  
**Quantile heads**: Smaller (hidden_dim → hidden_dim/2 → 1), optimized only for uncertainty bounds

### Loss Function

```
Central prediction receives:
  MSE + 2.0×SSIM + 1.0×Laplacian + 0.67×Histogram

Lower quantile receives:
  Pinball loss (q=0.025)

Upper quantile receives:
  Pinball loss (q=0.975)
```

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
