# Diffusion v1 — Sanity Evaluation on the Small Region

## Setup

- Branch: `diffusion`
- Region: `config/region_to_predict_small.geojson`
- Aggregate-tile size: **16×16** pixels (10 km blocks).
- Histogram bins (rarity-weighted, matches baseline `histogram_loss.py`):
  `[-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0]`.
- W&B run: [glennwithtwons/spatio-temporal-diffusion/48cotpid](https://wandb.ai/glennwithtwons/spatio-temporal-diffusion/48cotpid)

## Model & checkpoint

- Checkpoint: `spatio-temporal-diffusion/48cotpid/checkpoints/dhm-diffusion-epoch12-valloss0.0828.ckpt`
- Stopping epoch: 12 (global step 208)
- Architecture: `ConditionalDiffusionUNet` (`diffusers.UNet2DModel`)
  - sample_size = 64
  - base_channels = 128
  - channel_mults = [1, 2, 2, 4]
  - attention_head_dim = 64, attention_at_low_two = True
  - layers_per_block = 2
  - cond_channels = 56 (3 timesteps × 11 dyn + 7 static + locenc + 1 hm_t)
  - parameter count = **74.1 M**
- Diffusion: `DDPMScheduler(prediction_type="v_prediction", beta_schedule="squaredcos_cap_v2")`
  - num_train_timesteps = 1000
  - inference sampler: DDIM at 12 steps
- LocationEncoder: `('sphericalharmonics', 'siren')`, out_channels=8
- Optimizer: `AdamW(lr=0.0001, weight_decay=0.01)`

## Results

| Metric | Value |
|---|---|
| Tiles with valid coverage | 4,394 of 7,150 |
| Tile-mean MAE (median) | **0.0050** |
| Tile-mean MAE (mean) | 0.0085 |
| Tile-mean MAE (95th %ile) | 0.0297 |
| Histogram intersection (median) | **0.438** |
| Histogram intersection (mean) | 0.485 |
| Pearson r (predicted vs. observed tile-mean Δhm) | 0.378 |
| Coverage rate (q025 ≤ obs ≤ q975), all bins | 0.949 (target ≈ 0.95) |

### Coverage stratified by Δhm change bin

How often the observed Δhm falls inside the predicted (q025, q975) band, broken
out by magnitude bin. The overall coverage number is dominated by the no-change
bin, where most pixels live; the harder-to-cover bins are the rare large-change
ones where the model has to express genuine uncertainty.

| Δhm bin | n pixels | Coverage |
|---|---:|---:|
| `[ -1.000, -0.005]` | 56,357 | 0.814 |
| `[ -0.005, +0.005]` | 742,163 | 1.000 |
| `[ +0.005, +0.020]` | 116,330 | 0.998 |
| `[ +0.020, +0.100]` | 97,708 | 0.699 |
| `[ +0.100, +0.200]` | 10,296 | 0.000 |
| `[ +0.200, +0.400]` | 1,614 | 0.000 |
| `[ +0.400, +0.600]` | 120 | 0.000 |
| `[ +0.600, +1.000]` | 7 | 0.000 |

## Tail diagnostics

These metrics directly answer "is the model under-predicting magnitude
*somewhere in the field*?" — the user's stated success criterion. Methods follow
WassDiff (IEEE TGRS 2025), ExtremeCast (AAAI 2024), and the Aich et al. (GMD
2026) bias-correction work.

### R95p (mass above 95th percentile of obs)

- Threshold: 0.0667
- Observed tail mass: 1245.5662
- Predicted tail mass: 0.0000
- **Ratio (pred / obs):** **0.000** (<<1 = under-predicting tail; ~1 = calibrated; >1 = over)

### Tail exceedance (deterministic + q975 ensemble support)

| Threshold | n(obs>t) | n(pred>t) | POD | CSI | FAR | q975 soft-POD |
|---|---|---|---|---|---|---|
| +0.050 |     42,889 |        571 | 0.002 | 0.002 | 0.816 | 0.782 |
| +0.100 |     12,037 |          0 | 0.000 | 0.000 | nan | 0.000 |
| +0.200 |      1,741 |          0 | 0.000 | 0.000 | nan | 0.000 |
| +0.400 |        127 |          0 | 0.000 | 0.000 | nan | 0.000 |

![Q-Q max-of-field](../outputs/diffusion_v1/qq_max_of_field.png)

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
