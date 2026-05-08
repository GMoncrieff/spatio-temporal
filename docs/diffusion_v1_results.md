# Diffusion v1 — Sanity Evaluation on the Small Region

## Setup

- Branch: `diffusion`
- Region: `config/region_to_predict_small.geojson`
- Aggregate-tile size: **16×16** pixels (10 km blocks).
- Histogram bins (rarity-weighted, matches baseline `histogram_loss.py`):
  `[-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0]`.
- W&B run: [glennwithtwons/spatio-temporal-diffusion/ebcdx6ax](https://wandb.ai/glennwithtwons/spatio-temporal-diffusion/ebcdx6ax)

## Model & checkpoint

- Checkpoint: `/Users/glen.moncrieff/python/spatio_temporal/spatio-temporal-diffusion/ebcdx6ax/checkpoints/dhm-diffusion-epoch33-valloss0.1684.ckpt`
- Stopping epoch: 33 (global step 2176)
- Architecture: `ConditionalDiffusionUNet` (`diffusers.UNet2DModel`)
  - sample_size = 64
  - base_channels = 128
  - channel_mults = [1, 2, 2, 4]
  - attention_head_dim = 64, attention_at_low_two = True
  - layers_per_block = 2
  - cond_channels = 55 (3 timesteps × 11 dyn + 7 static + locenc + 1 hm_t)
  - parameter count = **74.1 M**
- Diffusion: `DDPMScheduler(prediction_type="v_prediction", beta_schedule="squaredcos_cap_v2")`
  - num_train_timesteps = 1000
  - inference sampler: DDIM at 12 steps
- LocationEncoder: `('sphericalharmonics', 'siren')`, out_channels=8
- Optimizer: `AdamW(lr=0.0001, weight_decay=0.01)`

## Results

| Metric | Value |
|---|---|
| Tiles with valid coverage | 2,003 of 7,150 |
| Tile-mean MAE (median) | **0.0019** |
| Tile-mean MAE (mean) | 0.0055 |
| Tile-mean MAE (95th %ile) | 0.0224 |
| Histogram intersection (median) | **0.862** |
| Histogram intersection (mean) | 0.789 |
| Pearson r (predicted vs. observed tile-mean Δhm) | 0.517 |
| Coverage rate (q025 ≤ obs ≤ q975) | 0.799 (target ≈ 0.95) |

## Figures

- Map comparison: ![](../outputs/diffusion_v1/map_comparison.png)
- Tile-level summaries: ![](../outputs/diffusion_v1/tile_metrics.png)


## Random samples vs. observed change

5 randomly-drawn validation chips (rows). Column 1 = observed Δhm
(2020 − 2000); columns 2–6 = independent draws from the trained diffusion
model's conditional posterior. Sample variability captures the model's
uncertainty about *where* and *how much* change occurs.

![](../outputs/diffusion_v1/samples_vs_observed.png)

## Caveats / Notes

- The aggregate tile is the smallest meaningful spatial unit at which we can
  *fairly* compare a generative ensemble to a deterministic observation — pixel
  agreement is not the goal of v1 (see migration plan).
- Prediction stride and ensemble size may have been reduced for iteration
  speed on Apple Silicon; widen them for the data-paper run.
- Coverage rate is computed on the (q025, q975) band — if it drifts far below
  ~0.95, the model is over-confident; if much higher, the bands are too wide.
