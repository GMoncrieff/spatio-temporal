# Spatio-Temporal Human Modification Forecasting

## Overview

This project forecasts the Human Modification (HM) index 20 years ahead using a
**conditional 2D diffusion U-Net**. Quantile bounds and uncertainty come from
ensemble sampling at inference rather than dedicated quantile heads — every
prediction is a draw from the conditional posterior P(Δhm | covariates), so
quantiles emerge as empirical statistics over N samples per chip.

The earlier ConvLSTM + hybrid-loss approach is preserved verbatim under
`baselines/convlstm/` for the data-paper comparison.

### Key features

- **Single 20-year horizon, generative**: predicts Δhm = HM(t+20) − HM(t)
  rather than absolute HM, with HM(t) supplied as an explicit conditioning
  channel.
- **Distributional fidelity over pointwise accuracy**: ensemble samples capture
  *where* and *how much* change occurs, instead of collapsing the conditional
  to a smeared mean.
- **Reuses the existing data layer**: per-variable normalization, validity
  masks, regional GeoJSON masking, and W&B integration carry over unchanged.
- **MPS / CUDA / CPU**: end-to-end pipeline runs on Apple Silicon, NVIDIA, or
  CPU with no code changes.

## Documentation

- **[Diffusion v1 results](docs/diffusion_v1_results.md)** — end-to-end
  evaluation on the small dev region (model details, metrics, sample grid).
- **[Baseline (ConvLSTM) technical guide](docs/simple_model_architecture_and_training.md)**
  — original multi-horizon, quantile-head ConvLSTM design.

## Project structure

```
spatio_temporal/
├── config/
│   ├── region_to_predict.geojson
│   ├── region_to_predict_small.geojson      # dev region used by default
│   └── ...
│
├── data/raw/hm_global/                       # rasters (not tracked)
│
├── src/
│   ├── locationencoder/                      # shared LocationEncoder package
│   └── models/
│       ├── diffusion_unet.py                 # ConditionalDiffusionUNet (UNet2DModel wrapper)
│       └── diffusion_lightning.py            # DiffusionLightningModule
│
├── baselines/
│   └── convlstm/                             # FROZEN ConvLSTM baseline
│       ├── models/                           # convlstm.py, spatiotemporal_predictor.py,
│       │                                     #   lightning_module.py, hybrid losses
│       ├── scripts/train_lightning.py
│       └── tests/                            # 21 ConvLSTM tests
│
├── scripts/
│   ├── torchgeo_dataloader.py                # adapted: target_mode='delta_20yr', restrict_to_region
│   ├── train_diffusion.py                    # diffusion training entrypoint
│   ├── predict_region_diffusion.py           # ensemble region prediction
│   ├── evaluate_diffusion.py                 # tile-level MAE + histogram-intersection eval
│   └── ...                                   # shared visualization utilities
│
├── tests/
│   ├── test_diffusion_dataloader.py          # delta_20yr dataloader shape tests
│   ├── test_diffusion_unet.py                # forward shape / param count
│   ├── test_diffusion_training.py            # loss decreases on synthetic batch
│   └── test_torchgeo_dataloader.py
│
├── docs/
│   ├── diffusion_v1_results.md
│   └── simple_model_architecture_and_training.md
│
├── outputs/
│   └── diffusion_v1/                         # report figures + metrics.json
│
├── environment.yml                            # mamba env spec
└── setup.py
```

## Setup

```bash
git checkout diffusion
mamba env create -f environment.yml -n spatio-diffusion
mamba activate spatio-diffusion
pip install -e .
```

The env pins:
- Python 3.11
- PyTorch ≥ 2.3 (with MPS on Apple Silicon, CUDA on Linux+nvidia)
- Lightning ≥ 2.2, diffusers ≥ 0.27, transformers, torchgeo, rasterio, etc.

For CUDA hosts add `pytorch-cuda=12.1` (or the version matching your driver)
to `environment.yml` before creating the env.

### W&B (optional)

```bash
wandb login
```

Disable per-run with `--disable_wandb`.

### Data layout

Rasters live under `data/raw/hm_global/`. Some files were originally staged
under `data/raw/hm_global/smal/`; the dataloader transparently resolves either
location via `_resolve()` in `scripts/torchgeo_dataloader.py`. All rasters are
on the same global grid (1 km, EPSG:4326).

## Usage

### Diffusion training

```bash
python scripts/train_diffusion.py \
  --max_epochs 50 \
  --base_channels 128 \
  --batch_size 4 \
  --train_chips 256 \
  --val_chips 32 \
  --restrict_to_region config/region_to_predict_small.geojson \
  --wandb_run_name v2-mac-50epoch-base128
```

Common knobs:

| Argument | Default | Description |
|---|---|---|
| `--max_epochs` | 5 | Training epochs |
| `--chip_size` | 64 | Spatial chip size |
| `--batch_size` | 16 | Mini-batch size |
| `--base_channels` | 128 | U-Net base channels (spec target) |
| `--channel_mults` | `[1, 2, 2, 4]` | Per-stage channel multipliers |
| `--num_train_timesteps` | 1000 | Diffusion training timesteps |
| `--num_inference_steps` | 30 | DDIM sampling steps at inference |
| `--ensemble_n` | 16 | Samples per chip aggregated in prediction |
| `--restrict_to_region` | `region_to_predict_small.geojson` | Region mask for chips |
| `--precision` | `32-true` | Use `bf16-mixed` on Ampere+ CUDA |

### Region prediction

```bash
python scripts/predict_region_diffusion.py \
  --checkpoint <path-to.ckpt> \
  --predict_region config/region_to_predict_small.geojson \
  --predict_stride 32 \
  --ensemble_n 16 \
  --num_inference_steps 30
```

Outputs four GeoTIFFs to `data/predictions_diffusion/`:

```
prediction_dhm_2020_median.tif   # ensemble median
prediction_dhm_2020_q025.tif     # 2.5th percentile (lower band)
prediction_dhm_2020_q975.tif     # 97.5th percentile (upper band)
prediction_dhm_2020_std.tif      # ensemble std (uncertainty)
```

### Tile-level evaluation

```bash
python scripts/evaluate_diffusion.py \
  --checkpoint <path-to.ckpt> \
  --wandb_run glennwithtwons/spatio-temporal-diffusion/<runid>
```

Generates `outputs/diffusion_v1/{map_comparison.png, tile_metrics.png,
samples_vs_observed.png, metrics.json}` and writes
`docs/diffusion_v1_results.md`.

## Architecture (diffusion v1)

- **Backbone**: `diffusers.UNet2DModel`, 4 resolution stages, `base_channels=128`,
  channel multipliers `[1, 2, 2, 4]`, self-attention at the lowest two stages,
  ~74 M parameters.
- **Conditioning** (49 channels concatenated to the noisy target every step):
  3 timesteps × 11 dynamic vars (33) + 7 static vars + 8 LocationEncoder
  channels + 1 explicit HM(t) reference channel.
- **Objective**: v-prediction with cosine schedule
  (`DDPMScheduler(prediction_type="v_prediction", beta_schedule="squaredcos_cap_v2")`).
- **Sampler**: DDIM at inference, 30 steps, 16-sample ensemble per chip.
- **Aggregation**: ensemble median (central), 2.5 % / 97.5 % quantiles
  (bounds), std (uncertainty).

## Baseline (ConvLSTM)

The original ConvLSTM with 12 independent prediction heads (4 horizons × 3
quantiles) lives under `baselines/convlstm/`. To run it:

```bash
python baselines/convlstm/scripts/train_lightning.py --max_epochs 50
```

Baseline tests:

```bash
pytest baselines/convlstm/tests
```

The baseline is frozen — no further development is expected on it; it exists
for the data-paper comparison.

## Data: Human Modification (HM)

We forecast the Human Modification (HM) index, a spatially explicit measure of
anthropogenic modification across landscapes.

- **Source paper** (Nature Scientific Data): Theobald, D. M., Oakleaf, J. R.,
  Moncrieff, G., Voigt, M., Kiesecker, J., & Kennedy, C. M. (2025). Global
  extent and change in human modification of terrestrial ecosystems from 1990
  to 2022. *Scientific Data*, 12(1), 606.
  Paper: <https://www.nature.com/articles/s41597-025-04892-2>
  Data: <https://zenodo.org/records/16907328>

**Available timesteps**: 1990, 1995, 2000, 2005, 2010, 2015, 2020.
For the 20yr-horizon diffusion, the only valid input window in the available
data is `(1990, 1995, 2000) → 2020` — the dataloader forces this when
`target_mode="delta_20yr"`.

## Citation

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
