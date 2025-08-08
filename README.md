# Spatio-Temporal Deep Learning Project

## Overview
This project implements a ConvLSTM-based spatio-temporal forecasting pipeline for the Human Modification (HM) index. The model consumes recent HM chips plus static covariates (elevation) and predicts future HM images over multiple autoregressive horizons (e.g., h=1..4). Experiments, metrics, and visualizations are tracked with Weights & Biases (W&B).

## Project Structure
```
├── data/                   # Data directory
│   ├── raw/               # Raw, immutable data
│   ├── processed/         # Cleaned and processed data
│   └── external/          # External datasets
├── src/                   # Source code
│   ├── models/            # Model definitions
│   ├── utils/             # Utility functions
│   ├── preprocessing/     # Data preprocessing scripts
│   ├── training/          # Training scripts
│   └── evaluation/        # Model evaluation scripts
├── notebooks/             # Jupyter notebooks for exploration
├── experiments/           # Experiment tracking and results
├── models/                # Trained models
│   ├── saved_models/      # Final trained models
│   └── checkpoints/       # Training checkpoints
├── logs/                  # Training and experiment logs
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── requirements.txt       # Python dependencies
└── setup.py              # Package setup
```

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure your environment variables
3. Run initial data preprocessing
4. Start training your models

## Usage

### Training
- Default training script:
  ```bash
  python scripts/train_lightning.py
  ```
- Key options (configured in `scripts/train_lightning.py`):
  - `forecast_horizon`: number of AR steps to evaluate and visualize (e.g., 4).
  - Validation dataloader’s `future_horizons` should match or exceed `forecast_horizon` when ground truth is available.

### Experiment tracking (W&B)
We log training/validation metrics and visualizations to W&B:

- Project: https://wandb.ai/glennwithtwons/spatio-temporal-convlstm

Highlights:
- Per-epoch aggregated multi-horizon validation metrics (h=1..K).
- A single end-of-fit panel from the best checkpoint showing targets, predictions, and differences per horizon. Missing ground-truth is masked (distinct “bad” color) and each row is annotated with the count of valid pixels.

See the W&B project page for complete run details and artifacts.

## Data: Human Modification (HM)

We forecast the Human Modification (HM) index, a spatially explicit measure of anthropogenic modification across landscapes. HM values are scaled to 0–10,000 for regression stability and interpretability in physical units during logging.

- Source paper (Nature Scientific Data):
  - “Global human modification time series (1990–2020) at 5-year intervals”
  - https://www.nature.com/articles/s41597-025-04892-2

Key characteristics used here:
- Temporal cadence: 5-year intervals (1990 → 1995 → … → 2020).
- Model inputs: 3 most recent HM timesteps per chip plus a static elevation raster.
- Targets: immediate next HM image (h=1) and optional future images for h≥2 evaluation when available.
- Normalization: z-score normalization applied internally; all W&B images and “original” MAE metrics are logged after inverse-transform to physical HM units (0–10k).

Data directories (example):
- `data/raw/hm/` — HM GeoTIFFs (1990–2020, 5-year intervals), pre-aligned/cropped.
- `data/raw/static/` — Static elevation GeoTIFF (aligned to HM grid).

## Contributing
Pull requests and issues are welcome. Please open an issue to discuss significant changes beforehand.
