# Spatio-Temporal Human Modification Forecasting

## Overview

This project forecasts the Human Modification (HM) index 20 years ahead using a
**conditional 2D diffusion U-Net with a CorrDiff residual decomposition**.
Quantile bounds and uncertainty come from ensemble sampling at inference —
every prediction is a draw from the conditional posterior P(Δhm | covariates),
so quantiles emerge as empirical statistics over N samples per chip.

A small deterministic *mean head* predicts the conditional-mean Δhm μ; the
diffusion U-Net learns only the residual r = target − μ. This frees the
diffusion's stochastic capacity from re-modelling the bulk and lets it express
rare-event tails properly.

The earlier ConvLSTM + hybrid-loss approach is preserved verbatim under
`baselines/convlstm/` for the data-paper comparison.

### Key features

- **Single 20-year horizon, generative**: predicts Δhm = HM(t+20) − HM(t)
  rather than absolute HM, with HM(t) supplied as an explicit conditioning
  channel.
- **CorrDiff residual decomposition**: a small joint-trained mean head
  predicts μ deterministically; the diffusion residual handles the rare-event
  tail.
- **Tile-aware loss stack**: per-tile histogram intersection, tile-mean MSE,
  multi-scale binary pattern matching, and Wasserstein marginal regularisation
  align training with the user-target metrics rather than per-pixel matching.
- **FIDE-style magnitude conditioning**: scalar `M = max(|Δhm|)` per chip is
  an explicit conditioning channel; at inference the user can request a
  target magnitude.
- **MPS / CUDA / CPU**: end-to-end pipeline runs on Apple Silicon, NVIDIA, or
  CPU with no code changes. Predict uses `torch.mps.empty_cache()` between
  batches to keep MPS inference fast.

## Documentation

- **[Diffusion experiment summary](docs/diffusion_experiment_summary.md)** —
  full iteration history (v8 → v37+), trade-off space, lessons.
- **[Diversity investigation notes](docs/diversity_investigation_notes.md)** —
  current focus: making samples diverse beyond uniform per-pixel grain.
- **[Diffusion results (older)](docs/diffusion_v1_results.md)** — first-cycle
  per-metric report (v8 → v26), kept for reference.
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
│       └── diffusion_lightning.py            # DiffusionLightningModule + CorrDiffMeanHead
│
├── baselines/
│   └── convlstm/                             # FROZEN ConvLSTM baseline
│       ├── models/                           # convlstm.py, spatiotemporal_predictor.py,
│       │                                     #   lightning_module.py, hybrid losses
│       ├── scripts/train_lightning.py
│       └── tests/                            # 21 ConvLSTM tests
│
├── scripts/
│   ├── torchgeo_dataloader.py                # target_mode='delta_20yr', restrict_to_region,
│   │                                         #   target_max_dhm scalar, weighted sampling
│   ├── train_diffusion.py                    # diffusion training entrypoint
│   ├── predict_region_diffusion.py           # ensemble region prediction (MPS-cache-safe)
│   ├── evaluate_diffusion.py                 # tile-level + tail-sensitive eval
│   └── ...                                   # shared visualization utilities
│
├── tests/
│   ├── test_diffusion_dataloader.py
│   ├── test_diffusion_unet.py
│   ├── test_diffusion_training.py
│   └── test_torchgeo_dataloader.py
│
├── docs/
│   └── diffusion_v1_results.md
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

### Quickstart (v26 production recipe — recommended)

The current best configuration on the small dev region, targeting the
user-stated criteria of (a) pixel coverage on high-change bins and (b)
tile-level histogram intersection.

**Train (100 epochs ≈ 110 min on Apple M-series MPS):**

```bash
python scripts/train_diffusion.py \
  --max_epochs 100 \
  --use_ema \
  --weighted_sampling --weight_alpha 1 \
  --pattern_loss_weight 0.5 --pattern_scales 8 16 32 \
  --tile_mean_loss_weight 15.0 --tile_mean_scales 8 16 32 64 \
  --wasserstein_loss_weight 0.5 \
  --hist_loss_weight 1.0 --hist_temperature 0.05 --hist_scales 16 32 \
  --tv_loss_weight 1.0 --tv_loss_target_floor 0.05 \
  --min_snr_gamma 5 \
  --use_magnitude_cond --m_dropout_prob 0.3 --m_norm_scale 0.5 \
  --use_mean_head --mean_head_hidden 128 --mean_loss_weight 0.1 \
  --mean_head_pixel_weight_alpha 1.0 --mean_head_pixel_weight_eps 0.01 \
  --restrict_to_region config/region_to_predict_small.geojson \
  --wandb_run_name v26-100ep-tail-focused
```

**Predict (~80 min at n=64 on MPS):**

```bash
python scripts/predict_region_diffusion.py \
  --checkpoint <path-to.ckpt> \
  --predict_region config/region_to_predict_small.geojson \
  --predict_stride 64 \
  --ensemble_n 64 \
  --m_target 0.7
```

**Evaluate:**

```bash
python scripts/evaluate_diffusion.py \
  --checkpoint <path-to.ckpt> \
  --wandb_run glennwithtwons/spatio-temporal-diffusion/<runid>
```

This writes `docs/diffusion_v1_results.md` with the iteration history,
per-bin coverage, tail diagnostics (Q-Q max-of-field, R95p, exceedance POD),
and a sample grid.

### Common training knobs

| Argument | Default | Description |
|---|---|---|
| `--max_epochs` | 5 | Training epochs |
| `--chip_size` | 64 | Spatial chip size |
| `--batch_size` | 16 | Mini-batch size |
| `--base_channels` | 128 | U-Net base channels |
| `--channel_mults` | `[1, 2, 2, 4]` | Per-stage channel multipliers |
| `--use_ema` | off | Track EMA weights for sampling |
| `--weighted_sampling` | off | Oversample chips with high `max|Δhm|` |
| `--num_train_timesteps` | 1000 | Diffusion training timesteps |
| `--num_inference_steps` | 30 | DDIM sampling steps |
| `--ensemble_n` | 16 | Samples per chip (use 32–64 in production) |
| `--restrict_to_region` | small region | Region mask for chips |
| `--precision` | `32-true` | Use `bf16-mixed` on Ampere+ CUDA |

### CorrDiff / loss-stack knobs

| Argument | Default | Description |
|---|---|---|
| `--use_mean_head` | off | Enable CorrDiff residual decomposition |
| `--mean_head_hidden` | 64 | Mean head hidden channels |
| `--mean_loss_weight` | 1.0 | Mean head MSE multiplier |
| `--mean_head_pixel_weight_alpha` | 0.0 | Pixel weight = `|target|^α + ε`; α=1 lifts tail predictions |
| `--mean_head_pixel_weight_eps` | 0.01 | Floor for the pixel weight |
| `--tile_mean_loss_weight` | 0.0 | Multi-scale tile-mean MSE on x0_pred |
| `--tile_mean_scales` | `[8, 16]` | Avg-pool scales (use `8 16 32 64`) |
| `--wasserstein_loss_weight` | 0.0 | 1D marginal Wasserstein on pixel histogram |
| `--hist_loss_weight` | 0.0 | Per-tile soft-histogram intersection loss |
| `--hist_temperature` | 0.01 | Sigmoid temp for soft binning |
| `--hist_scales` | `[16, 32]` | Avg-pool scales for the histogram loss |
| `--pattern_loss_weight` | 0.0 | Multi-scale binary-pattern matching loss |
| `--tv_loss_weight` | 0.0 | TV loss on x0_pred (smooths samples) |
| `--tv_loss_target_floor` | 0.05 | Masks TV out where `|target| > floor` (preserves tail) |
| `--min_snr_gamma` | 0.0 | Hang-2023 min-SNR-γ weighting (5 is standard) |
| `--use_magnitude_cond` | off | FIDE-style block-maxima conditioning channel |
| `--m_dropout_prob` | 0.0 | Drop M to null with this probability during training |
| `--m_norm_scale` | 0.5 | Divisor for raw `|Δhm|` max in the conditioning channel |

### Region prediction outputs

Four GeoTIFFs in `data/predictions_diffusion/`:

```
prediction_dhm_2020_median.tif   # ensemble median
prediction_dhm_2020_q025.tif     # 2.5th percentile (lower band)
prediction_dhm_2020_q975.tif     # 97.5th percentile (upper band)
prediction_dhm_2020_std.tif      # ensemble std (uncertainty)
```

## Architecture

### Diffusion model

- **Backbone**: `diffusers.UNet2DModel`, 4 resolution stages, `base_channels=128`,
  channel multipliers `[1, 2, 2, 4]`, self-attention at the lowest two stages,
  ~74 M parameters.
- **Conditioning** (56 channels concatenated to the noisy target every step):
  3 timesteps × 13 dynamic vars (HM + 12 component/distance) + 7 static vars +
  8 LocationEncoder channels + 1 HM(t) reference + 1 magnitude scalar M.
- **Objective**: v-prediction with cosine schedule
  (`DDPMScheduler(prediction_type="v_prediction", beta_schedule="squaredcos_cap_v2")`).
- **Sampler**: DDIM at inference, 30 steps. Production ensemble = 64 samples.

### CorrDiff mean head

A small 4-layer conv stack (`CorrDiffMeanHead`) takes the same conditioning
the U-Net sees and outputs a deterministic μ. Trained jointly with an MSE
loss on `(μ, target)`; the diffusion model learns the residual
`r = target − μ.detach()` instead of the full target. At inference,
`sample()` adds μ back so callers always see the full predicted Δhm.

The mean head's MSE can be **pixel-weighted by `|target|^α`** so the rare
high-magnitude pixels aren't averaged out by the dominant near-zero bulk —
this is the key knob that unlocks high-bucket coverage. Combined with the
`mean_loss_weight` multiplier, this trades bulk-vs-tail along a continuum
that the three production recipes (v20 / v24 / v26) cover.

### Loss stack rationale

The user's success criterion is (a) **pixel coverage** (q025–q975 envelope
brackets observed pixels, including high-change ones) and (b) **tile-level
amount of change** (per-tile aggregate magnitude / histogram match). Pixel-
precise magnitude matching is *not* a goal. The loss stack reflects this:

- Tile-mean MSE + per-tile histogram intersection + multi-scale binary pattern
  loss → tile-level criteria.
- Marginal Wasserstein → distribution-shape match across the batch.
- TV(masked) → smooths individual samples in the smooth (zero-near) region
  without touching the tail.
- Mean-head pixel weighting → unlock high-magnitude predictions on hotspot
  tiles.

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
