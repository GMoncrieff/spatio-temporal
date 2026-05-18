# Diffusion Δhm Forecasting — Results on the Small Region

## Setup

- Branch: `diffusion`
- Region: `config/region_to_predict_small.geojson`
- Aggregate-tile size: **16×16** pixels (10 km blocks).
- Histogram bins (rarity-weighted, matches baseline `histogram_loss.py`):
  `[-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0]`.

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
| v25 | 0.239 | 0.048 | 0.246 | 0.412 | 0.0209 | 0.543 | 0.411 | mw=0.1 — first q975→0.4 |
| v26 | 0.247 | 0.061 | 0.289 | 0.678 | 0.0130 | 0.829 | 0.491 | v25 + 100 ep + n=64 — first bin>0.4 cov (0.050) |
| v36c | 0.426 | 0.125 | 0.544 | 0.709 | 0.0065 | 0.770 | 0.529 | v26 ckpt + inference levers — balanced |
| v36d | 0.528 | 0.203 | 0.534 | 0.686 | 0.0074 | 0.741 | 0.577 | aggressive levers — tail-focused |
| v36e | 0.464 | 0.158 | 0.543 | 0.704 | 0.0066 | 0.756 | 0.552 | sweet-spot levers — bulk-balanced |
| v36f | 0.529 | 0.194 | 0.546 | 0.707 | 0.0069 | 0.774 | 0.542 | + μ-masked residual (scale mode) |
| v36h | 0.524 | 0.196 | 0.547 | 0.699 | 0.0070 | 0.712 | 0.567 | + μ-mask GATE mode — cleanest backgrounds (hot/flat 1.46) |
| **v36i** | **0.607** | **0.251** | **0.546** | **0.702** | **0.0067** | **0.720** | **0.557** | **+ low-freq spatial cond perturb — hotspots shift across samples (hot/flat 1.59)** |

v13 was the architectural breakthrough (CorrDiff residual). v26 was
the previous best trained model. The v36 family changed the game by
showing that **inference-time levers on the v26 checkpoint** dominate
every trained variant: per-bin coverage at high-change bins roughly
doubles, MAE halves, Pearson r jumps 0.44 → 0.70, and Q-Q max pred
climbs from 0.40 to 0.55 (obs 0.66). Three training-side diversity
losses (v37/v37b/v37c) saturated their objectives but left inference
per-pixel std unchanged at 0.025; the U-Net + DDIM combo can't be made
more diverse through training-time loss formulations, only through
inference levers.

### Production recipes

All three production recipes use the **same v26 checkpoint** with
different inference-time lever combinations. No retraining.

**v36i — low-freq spatial perturbation + gate mode, recommended**
(hotspots shift across samples — true spatial diversity, hot/flat =
1.59, bin > 0.1 cov 0.607):
```
python scripts/predict_region_diffusion.py   --checkpoint <v26-ckpt>   --predict_region config/region_to_predict_small.geojson   --output_dir data/predictions_diffusion/v36i   --ensemble_n 16 --predict_batch_size 2 --num_inference_steps 30   --m_sample_diverse --m_sample_min 0.2 --m_sample_max 0.8   --residual_scale_pos 5.0 --residual_scale_neg 0.30   --residual_mask_threshold 0.07 --residual_mask_softness 0.01   --residual_mask_mode gate   --cond_perturb_std 0.05 --cond_perturb_lowfreq_size 6
```

The `--cond_perturb_lowfreq_size` flag turns the conditioning
perturbation into a coherent low-frequency spatial field: per sample,
generate a 6×6 random Gaussian, bilinear-upsample to the chip size,
scale by 0.05, add to the conditioning. Coherent enough to *shift μ's
hotspot predictions in space* across samples (the missing piece that
per-pixel grain in earlier recipes couldn't deliver). Result: bin>0.1
coverage jumps 0.524 → 0.607 because different ensemble members now
place hotspots at slightly different locations and brackett more
high-change pixels collectively.

**v36h — μ-mask GATE mode** (cleanest backgrounds, less spatial diversity,
hot/flat = 1.46):
```
python scripts/predict_region_diffusion.py   --checkpoint <v26-ckpt>   --predict_region config/region_to_predict_small.geojson   --output_dir data/predictions_diffusion/v36h   --ensemble_n 16 --predict_batch_size 2 --num_inference_steps 30   --m_sample_diverse --m_sample_min 0.05 --m_sample_max 1.2   --residual_scale_pos 4.0 --residual_scale_neg 0.35   --residual_mask_threshold 0.04 --residual_mask_softness 0.015   --residual_mask_mode gate   --cond_perturb_std 0.0
```

GATE mode multiplies the residual by `weight * scale` at every pixel
(rather than `1 + (scale-1) * weight`). At flat pixels where |μ| <
threshold, weight ≈ 0 so the residual goes to ≈ 0 and `sample = μ`
exactly — truly smooth, near-zero backgrounds. Hotspots get the full
scaling. Trade-off: q025-q975 envelope is too narrow at flat pixels
to bracket the small non-zero obs values there → coverage_rate drops
from 0.774 (v36f) to 0.712. Use this when visual cleanness matters
more than calibration; use v36f when calibration matters more.

**v36f — μ-masked scale mode** (mid-ground, calibrated):
```
python scripts/predict_region_diffusion.py   --checkpoint <v26-ckpt>   --predict_region config/region_to_predict_small.geojson   --output_dir data/predictions_diffusion/v36f   --ensemble_n 16 --predict_batch_size 2 --num_inference_steps 30   --m_sample_diverse --m_sample_min 0.05 --m_sample_max 1.2   --residual_scale_pos 4.0 --residual_scale_neg 0.35   --residual_mask_threshold 0.03 --residual_mask_softness 0.02
```

Same structure as v36h but in "scale" mode: at flat pixels the
residual is multiplied by ≈ 1 (kept as the trained baseline residual),
so sample ≈ μ + tiny residual. Backgrounds have a faint 0.024 RMS
grain; in return coverage_rate stays at 0.774 (best of all recipes).

**v36e — uniform sweet spot** (background grain ok):
```
python scripts/predict_region_diffusion.py   --checkpoint <v26-ckpt>   --predict_region config/region_to_predict_small.geojson   --output_dir data/predictions_diffusion/v36e   --ensemble_n 16 --predict_batch_size 2 --num_inference_steps 30   --m_sample_diverse --m_sample_min 0.05 --m_sample_max 1.2   --residual_scale_pos 3.5 --residual_scale_neg 0.35   --cond_perturb_std 0.08
```

**v36c — bulk-balanced** (best MAE, Pearson, R95p calibration):
```
... --m_sample_max 1.0 --residual_scale_pos 3.0     --residual_scale_neg 0.40 --cond_perturb_std 0.08
```

**v36d — tail-focused** (highest bin>0.2 / bin>0.4 coverage):
```
... --m_sample_max 1.5 --residual_scale_pos 4.0     --residual_scale_neg 0.30 --cond_perturb_std 0.10
```

The four inference levers:
- `--m_sample_diverse`: each ensemble member draws its own m_target ~
  Uniform[min, max]. Different m → different μ AND different residual.
- `--residual_scale_pos/neg`: asymmetric scaling of the CorrDiff
  residual before adding μ back. >1 on positive side lifts Q-Q max;
  <1 on negative side matches obs negative-pixel fraction.
- `--cond_perturb_std`: small Gaussian noise on the full conditioning
  per sample → structural diversity at hotspots (hot/flat std ratio
  jumps from ~1.0 to ~1.36).

The trade-off across the v36 family: more aggressive levers (v36d)
push the upper tail further but slightly worsen MAE and inflate R95p
above 1.0. v36c is the safest; v36d for risk maps; v36e splits.

Performance note: predict_region_diffusion.py now calls
`torch.mps.empty_cache()` between batches and explicitly deletes
intermediate GPU tensors. Without that, MPS memory accumulates across
batches and the predict run slows from ~40 min to 2-3 h.


## Results

| Metric | Value |
|---|---|
| Tiles with valid coverage | 4,397 of 7,150 |
| Tile-mean MAE (median) | **0.0075** |
| Tile-mean MAE (mean) | 0.0118 |
| Tile-mean MAE (95th %ile) | 0.0403 |
| Histogram intersection (median) | **0.531** |
| Histogram intersection (mean) | 0.504 |
| Pearson r (predicted vs. observed tile-mean Δhm) | 0.697 |
| Coverage rate (q025 ≤ obs ≤ q975), all bins | 0.759 (target ≈ 0.95) |

### Coverage stratified by Δhm change bin

How often the observed Δhm falls inside the predicted (q025, q975) band, broken
out by magnitude bin. The overall coverage number is dominated by the no-change
bin, where most pixels live; the harder-to-cover bins are the rare large-change
ones where the model has to express genuine uncertainty.

| Δhm bin | n pixels | Coverage |
|---|---:|---:|
| `[ -1.000, -0.005]` | 59,954 | 0.465 |
| `[ -0.005, +0.005]` | 789,483 | 0.842 |
| `[ +0.005, +0.020]` | 123,519 | 0.561 |
| `[ +0.020, +0.100]` | 103,808 | 0.578 |
| `[ +0.100, +0.200]` | 10,953 | 0.470 |
| `[ +0.200, +0.400]` | 1,746 | 0.171 |
| `[ +0.400, +0.600]` | 127 | 0.071 |
| `[ +0.600, +1.000]` | 7 | 0.143 |

## Tail diagnostics

These metrics directly answer "is the model under-predicting magnitude
*somewhere in the field*?" — the user's stated success criterion. Methods follow
WassDiff (IEEE TGRS 2025), ExtremeCast (AAAI 2024), and the Aich et al. (GMD
2026) bias-correction work.

### R95p (mass above 95th percentile of obs)

- Threshold: 0.0668
- Observed tail mass: 1327.2455
- Predicted tail mass: 1804.9911
- **Ratio (pred / obs):** **1.360** (<<1 = under-predicting tail; ~1 = calibrated; >1 = over)

### Tail exceedance (deterministic + q975 ensemble support)

| Threshold | n(obs>t) | n(pred>t) | POD | CSI | FAR | q975 soft-POD |
|---|---|---|---|---|---|---|
| +0.050 |     45,659 |    100,314 | 0.476 | 0.175 | 0.783 | 0.913 |
| +0.100 |     12,833 |     19,802 | 0.208 | 0.089 | 0.865 | 0.715 |
| +0.200 |      1,880 |        811 | 0.106 | 0.080 | 0.755 | 0.336 |
| +0.400 |        134 |         19 | 0.097 | 0.093 | 0.316 | 0.134 |

![Q-Q max-of-field](../outputs/diffusion_v36r/qq_max_of_field.png)
![Q-Q mean-of-field](../outputs/diffusion_v36r/qq_mean_of_field.png)

## Figures

- Map comparison: ![](../outputs/diffusion_v36r/map_comparison.png)
- Tile-level summaries: ![](../outputs/diffusion_v36r/tile_metrics.png)

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
