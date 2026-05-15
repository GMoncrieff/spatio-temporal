# Diffusion Δhm Forecasting — Results on the Small Region

## Setup

- Branch: `diffusion`
- Region: `config/region_to_predict_small.geojson`
- Aggregate-tile size: **16×16** pixels (10 km blocks).
- Histogram bins (rarity-weighted, matches baseline `histogram_loss.py`):
  `[-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0]`.
- W&B run: [glennwithtwons/spatio-temporal-diffusion/8i4lbs9l](https://wandb.ai/glennwithtwons/spatio-temporal-diffusion/8i4lbs9l)


## Iteration history (best-of-run on user-target metrics)

User targets (re-stated): improve **per-bin pixel coverage** (especially
bins > 0.05) and **tile-level histogram intersection** (`xinter_median`).
Pixel-precise magnitude matching is *not* a goal.

| Run | bin > 0.1 cov | bin > 0.2 cov | xinter | Pearson r | MAE med | cov rate | pred_max | Notes |
|---|---|---|---|---|---|---|---|---|
| v8 baseline | 0.000 | 0.000 | 0.860 | 0.561 | 0.0034 | 0.942 | 0.056 | EMA + weighted + min-SNR (stride 32 eval) |
| v10/v11 | 0.000 | 0.000 | 0.55/0.85 | 0.55 | varies | varies | 0.022/0.011 | signed-log target — regression |
| v12 | 0.000 | 0.000 | 0.438 | 0.378 | 0.0050 | 0.949 | 0.056 | 2A+2B+2C — flat on tail |
| **v13** | **0.069** | 0.000 | 0.400 | 0.665 | 0.0074 | 0.954 | **0.208** | **CorrDiff residual — tail wall breaks** |
| v14 | 0.072 | 0.000 | 0.430 | 0.677 | 0.0067 | 0.953 | 0.221 | + tile-aware loss redesign |
| v15 | 0.085 | 0.002 | 0.521 | 0.677 | 0.0042 | 0.959 | 0.214 | + 2× ensemble at inference (n=32) |
| v17 | 0.086 | 0.001 | 0.399 | 0.699 | 0.0097 | 0.958 | 0.206 | + per-tile hist_loss=1.0 — regression |
| v18 | 0.087 | 0.002 | 0.529 | 0.681 | 0.0043 | 0.959 | 0.215 | gentler hist_loss=0.3 |
| v19 | 0.087 | 0.002 | 0.481 | 0.693 | 0.0072 | 0.956 | 0.222 | + TV(masked) — tail preserved, neg-bias worsened |
| **v20** | 0.083 | 0.007 | **0.604** | **0.738** | **0.0034** | 0.958 | 0.356 | **+ strong chip-mean anchor — best balanced** |
| v21 | 0.394 | 0.039 | 0.133 | 0.606 | 0.0299 | 0.483 | 0.505 | + α=1.0 pixel-weighted mean head — tail breaks, bulk breaks |
| v22 | 0.135 | 0.005 | 0.471 | 0.687 | 0.0051 | 0.952 | 0.256 | α=0.3 — mediocre middle |
| v23 | **0.441** | 0.037 | 0.175 | 0.622 | 0.0279 | 0.485 | 0.413 | α=1.0 + 3× bulk anchors — best bin>0.1 cov |
| v24 | 0.247 | 0.042 | 0.343 | 0.534 | 0.0129 | 0.718 | 0.389 | mw=0.3 — best balance bin>0.2 |
| **v25** | 0.239 | **0.048** | 0.246 | 0.412 | 0.0209 | 0.543 | 0.411 | **mw=0.1 — best upper-tail; first q975→0.4** |

v13 was the architectural breakthrough (CorrDiff residual). v20 is
the best **balanced** result. v24 is the best **bin>0.2 / balance**.
v25 has the **best upper-tail** including first non-zero q975 reach
to 0.4. Choice depends on which trade-off you want.

### Production recipes

**v20 — bulk-balanced** (best xinter, MAE, coverage rate, Pearson):
```
Training:
  --use_ema --weighted_sampling --weight_alpha 1
  --pattern_loss_weight 0.3 --pattern_scales 8 16 32
  --tile_mean_loss_weight 5.0 --tile_mean_scales 8 16 32 64
  --wasserstein_loss_weight 0.2
  --hist_loss_weight 0.3 --hist_temperature 0.05 --hist_scales 16 32
  --tv_loss_weight 1.0 --tv_loss_target_floor 0.05
  --min_snr_gamma 5
  --use_magnitude_cond --m_dropout_prob 0.3 --m_norm_scale 0.5
  --use_mean_head --mean_head_hidden 128 --mean_loss_weight 1.0
Inference:
  --ensemble_n 32 --m_target 0.7 --predict_stride 64
```

**v25 — tail-focused** (best bin>0.2 cov, POD@0.2, q975 soft-POD@0.4):
```
Training: v20 +
  --tile_mean_loss_weight 15.0 (vs 5.0)
  --wasserstein_loss_weight 0.5 (vs 0.2)
  --hist_loss_weight 1.0 (vs 0.3)
  --pattern_loss_weight 0.5 (vs 0.3)
  --mean_loss_weight 0.1 (vs 1.0)
  --mean_head_pixel_weight_alpha 1.0 --mean_head_pixel_weight_eps 0.01
Inference: same as v20
```

**v24 — middle ground** (decent bulk + decent tail): v25 settings but
`--mean_loss_weight 0.3`.

The trade-off: stronger mean head pixel weighting (α=1.0) lifts hotspot
predictions but pulls the bulk's median pred from +0.001 to ~+0.02.
Pixel-precise matching is *not* a goal of this work; mean preservation
of the bulk distribution is. v20 maximises the latter; v25 maximises
upper-tail coverage. v24 splits the difference.

Performance note: predict_region_diffusion.py now calls
`torch.mps.empty_cache()` between batches and explicitly deletes
intermediate GPU tensors. Without that, MPS memory accumulates across
batches and the predict run slows from ~40 min to 2-3 h.

## Model & checkpoint

- Checkpoint: `spatio-temporal-diffusion/8i4lbs9l/checkpoints/dhm-diffusion-epoch94-valloss0.7645.ckpt`
- Stopping epoch: 94 (global step 1520)
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
| Tile-mean MAE (median) | **0.0130** |
| Tile-mean MAE (mean) | 0.0152 |
| Tile-mean MAE (95th %ile) | 0.0352 |
| Histogram intersection (median) | **0.289** |
| Histogram intersection (mean) | 0.325 |
| Pearson r (predicted vs. observed tile-mean Δhm) | 0.678 |
| Coverage rate (q025 ≤ obs ≤ q975), all bins | 0.829 (target ≈ 0.95) |

### Coverage stratified by Δhm change bin

How often the observed Δhm falls inside the predicted (q025, q975) band, broken
out by magnitude bin. The overall coverage number is dominated by the no-change
bin, where most pixels live; the harder-to-cover bins are the rare large-change
ones where the model has to express genuine uncertainty.

| Δhm bin | n pixels | Coverage |
|---|---:|---:|
| `[ -1.000, -0.005]` | 56,357 | 0.675 |
| `[ -0.005, +0.005]` | 742,163 | 0.905 |
| `[ +0.005, +0.020]` | 116,330 | 0.692 |
| `[ +0.020, +0.100]` | 97,708 | 0.577 |
| `[ +0.100, +0.200]` | 10,296 | 0.247 |
| `[ +0.200, +0.400]` | 1,614 | 0.061 |
| `[ +0.400, +0.600]` | 120 | 0.050 |
| `[ +0.600, +1.000]` | 7 | 0.000 |

## Tail diagnostics

These metrics directly answer "is the model under-predicting magnitude
*somewhere in the field*?" — the user's stated success criterion. Methods follow
WassDiff (IEEE TGRS 2025), ExtremeCast (AAAI 2024), and the Aich et al. (GMD
2026) bias-correction work.

### R95p (mass above 95th percentile of obs)

- Threshold: 0.0667
- Observed tail mass: 1245.5662
- Predicted tail mass: 941.0035
- **Ratio (pred / obs):** **0.755** (<<1 = under-predicting tail; ~1 = calibrated; >1 = over)

### Tail exceedance (deterministic + q975 ensemble support)

| Threshold | n(obs>t) | n(pred>t) | POD | CSI | FAR | q975 soft-POD |
|---|---|---|---|---|---|---|
| +0.050 |     42,889 |     60,826 | 0.364 | 0.177 | 0.743 | 0.799 |
| +0.100 |     12,037 |      9,955 | 0.160 | 0.096 | 0.806 | 0.500 |
| +0.200 |      1,741 |        299 | 0.071 | 0.065 | 0.585 | 0.183 |
| +0.400 |        127 |         15 | 0.087 | 0.084 | 0.267 | 0.110 |

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
