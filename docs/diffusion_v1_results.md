# Diffusion Δhm Forecasting — Results on the Small Region

## Setup

- Branch: `diffusion`
- Region: `config/region_to_predict_small.geojson`
- Aggregate-tile size: **16×16** pixels (10 km blocks).
- Histogram bins (rarity-weighted, matches baseline `histogram_loss.py`):
  `[-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0]`.
- W&B run: [glennwithtwons/spatio-temporal-diffusion/424151ye](https://wandb.ai/glennwithtwons/spatio-temporal-diffusion/424151ye)


## Iteration history (best-of-run on user-target metrics)

User targets (re-stated): improve **per-bin pixel coverage** (especially
bins > 0.05) and **tile-level histogram intersection** (`xinter_median`).
Pixel-precise magnitude matching is *not* a goal.

| Run | bin > 0.1 cov | xinter | Pearson r | MAE med | cov rate | pred_max | Notes |
|---|---|---|---|---|---|---|---|
| v8 baseline | 0.000 | 0.860 | 0.561 | 0.0034 | 0.942 | 0.056 | EMA + weighted + min-SNR (stride 32 eval) |
| v10 / v11 | 0.000 | 0.55 / 0.85 | 0.55 | varies | varies | 0.022 / 0.011 | signed-log target — regression |
| v12 | 0.000 | 0.438 | 0.378 | 0.0050 | 0.949 | 0.056 | 2A+2B+2C — flat on tail |
| **v13** | **0.069** | 0.400 | 0.665 | 0.0074 | 0.954 | **0.208** | **CorrDiff residual — tail wall breaks** |
| v14 | 0.072 | 0.430 | 0.677 | 0.0067 | 0.953 | 0.221 | + tile-aware loss redesign |
| v15 | 0.085 | 0.521 | 0.677 | 0.0042 | 0.959 | 0.214 | + 2× ensemble at inference (n=32) |
| v17 | 0.086 | 0.399 | 0.699 | 0.0097 | 0.958 | 0.206 | + per-tile hist_loss=1.0 — regression |
| **v18** | **0.087** | **0.529** | 0.681 | 0.0043 | **0.959** | 0.215 | **gentler hist_loss=0.3 — best on user targets** |

v18 is the operational best for the user's stated criteria; v13 was the
architectural breakthrough that unlocked the rest. Bins ≥ 0.2 remain near
0% coverage across all runs — those events (1614 + 120 + 7 pixels in the
small region) need either focal-cropping training data or a wider U-Net.

### Production-ready recipe (v18)

Training:
```
--use_ema --weighted_sampling --weight_alpha 1
--pattern_loss_weight 0.3 --pattern_scales 8 16 32
--tile_mean_loss_weight 0.5 --tile_mean_scales 8 16 32
--wasserstein_loss_weight 0.2
--hist_loss_weight 0.3 --hist_temperature 0.05 --hist_scales 16 32
--min_snr_gamma 5
--use_magnitude_cond --m_dropout_prob 0.3 --m_norm_scale 0.5
--use_mean_head --mean_head_hidden 128 --mean_loss_weight 1.0
```

Inference:
```
--ensemble_n 32 --m_target 0.7 --predict_stride 64
```

## Model & checkpoint

- Checkpoint: `spatio-temporal-diffusion/424151ye/checkpoints/dhm-diffusion-epoch12-valloss0.4287.ckpt`
- Stopping epoch: 12 (global step 208)
- Architecture: `ConditionalDiffusionUNet` (`diffusers.UNet2DModel`)
  - sample_size = 64
  - base_channels = 128
  - channel_mults = [1, 2, 2, 4]
  - attention_head_dim = 64, attention_at_low_two = True
  - layers_per_block = 2
  - cond_channels = 56 (3 timesteps × 11 dyn + 7 static + locenc + 1 hm_t)
  - parameter count = **74.5 M**
- Diffusion: `DDPMScheduler(prediction_type="v_prediction", beta_schedule="squaredcos_cap_v2")`
  - num_train_timesteps = 1000
  - inference sampler: DDIM at 12 steps
- LocationEncoder: `('sphericalharmonics', 'siren')`, out_channels=8
- Optimizer: `AdamW(lr=0.0001, weight_decay=0.01)`

## Results

| Metric | Value |
|---|---|
| Tiles with valid coverage | 4,394 of 7,150 |
| Tile-mean MAE (median) | **0.0072** |
| Tile-mean MAE (mean) | 0.0090 |
| Tile-mean MAE (95th %ile) | 0.0236 |
| Histogram intersection (median) | **0.480** |
| Histogram intersection (mean) | 0.510 |
| Pearson r (predicted vs. observed tile-mean Δhm) | 0.693 |
| Coverage rate (q025 ≤ obs ≤ q975), all bins | 0.956 (target ≈ 0.95) |

### Coverage stratified by Δhm change bin

How often the observed Δhm falls inside the predicted (q025, q975) band, broken
out by magnitude bin. The overall coverage number is dominated by the no-change
bin, where most pixels live; the harder-to-cover bins are the rare large-change
ones where the model has to express genuine uncertainty.

| Δhm bin | n pixels | Coverage |
|---|---:|---:|
| `[ -1.000, -0.005]` | 56,357 | 0.802 |
| `[ -0.005, +0.005]` | 742,163 | 0.998 |
| `[ +0.005, +0.020]` | 116,330 | 0.984 |
| `[ +0.020, +0.100]` | 97,708 | 0.802 |
| `[ +0.100, +0.200]` | 10,296 | 0.087 |
| `[ +0.200, +0.400]` | 1,614 | 0.002 |
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
- Predicted tail mass: 111.8560
- **Ratio (pred / obs):** **0.090** (<<1 = under-predicting tail; ~1 = calibrated; >1 = over)

### Tail exceedance (deterministic + q975 ensemble support)

| Threshold | n(obs>t) | n(pred>t) | POD | CSI | FAR | q975 soft-POD |
|---|---|---|---|---|---|---|
| +0.050 |     42,889 |     13,760 | 0.152 | 0.130 | 0.528 | 0.829 |
| +0.100 |     12,037 |        956 | 0.031 | 0.030 | 0.607 | 0.195 |
| +0.200 |      1,741 |          5 | 0.003 | 0.003 | 0.000 | 0.011 |
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
  *fairly* compare a generative ensemble to a deterministic observation —
  per-pixel magnitude matching is explicitly *not* a goal of this work.
- Coverage rate is computed on the (q025, q975) band — if it drifts far below
  ~0.95, the model is over-confident; if much higher, the bands are too wide.
- Bins ≥ 0.2 still show near-zero coverage across all iterations: those
  events are rare (~1700 pixels region-wide in [0.2, 0.4]) and concentrate
  on a handful of hotspot tiles. Closing this gap likely needs focal-cropped
  training data (chip-centred on hotspots) or a wider U-Net backbone.
- Apple Silicon is fast enough for development at `--ensemble_n 32`; for
  the data-paper run on CUDA, bump to `--ensemble_n 64+` and consider a
  larger network with `base_channels=192`.
