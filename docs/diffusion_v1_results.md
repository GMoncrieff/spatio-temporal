# Diffusion v1 — Sanity Evaluation on the Small Region

## Setup

- Branch: `diffusion`
- Region: `config/region_to_predict_small.geojson`
- Aggregate-tile size: **16×16** pixels (10 km blocks).
- Histogram bins (rarity-weighted, matches baseline `histogram_loss.py`):
  `[-1.0, -0.005, 0.005, 0.02, 0.1, 0.2, 0.4, 0.6, 1.0]`.

## Results

| Metric | Value |
|---|---|
| Tiles with valid coverage | 2,003 of 7,150 |
| Tile-mean MAE (median) | **0.0034** |
| Tile-mean MAE (mean) | 0.0066 |
| Tile-mean MAE (95th %ile) | 0.0220 |
| Histogram intersection (median) | **0.859** |
| Histogram intersection (mean) | 0.793 |
| Pearson r (predicted vs. observed tile-mean Δhm) | 0.561 |
| Coverage rate (q025 ≤ obs ≤ q975), all bins | 0.942 (target ≈ 0.95) |

### Coverage stratified by Δhm change bin

How often the observed Δhm falls inside the predicted (q025, q975) band, broken
out by magnitude bin. The overall coverage number is dominated by the no-change
bin, where most pixels live; the harder-to-cover bins are the rare large-change
ones where the model has to express genuine uncertainty.

| Δhm bin | n pixels | Coverage |
|---|---:|---:|
| `[ -1.000, -0.005]` | 26,031 | 0.706 |
| `[ -0.005, +0.005]` | 345,050 | 0.993 |
| `[ +0.005, +0.020]` | 51,561 | 0.979 |
| `[ +0.020, +0.100]` | 40,825 | 0.726 |
| `[ +0.100, +0.200]` | 4,020 | 0.000 |
| `[ +0.200, +0.400]` | 525 | 0.000 |
| `[ +0.400, +0.600]` | 19 | 0.000 |
| `[ +0.600, +1.000]` | 0 | — |

## Tail diagnostics

These metrics directly answer "is the model under-predicting magnitude
*somewhere in the field*?" — the user's stated success criterion. Methods follow
WassDiff (IEEE TGRS 2025), ExtremeCast (AAAI 2024), and the Aich et al. (GMD
2026) bias-correction work.

### R95p (mass above 95th percentile of obs)

- Threshold: 0.0623
- Observed tail mass: 501.7283
- Predicted tail mass: 0.0000
- **Ratio (pred / obs):** **0.000** (<<1 = under-predicting tail; ~1 = calibrated; >1 = over)

### Tail exceedance (deterministic + q975 ensemble support)

| Threshold | n(obs>t) | n(pred>t) | POD | CSI | FAR | q975 soft-POD |
|---|---|---|---|---|---|---|
| +0.050 |     17,401 |      1,042 | 0.012 | 0.012 | 0.792 | 0.926 |
| +0.100 |      4,564 |          0 | 0.000 | 0.000 | nan | 0.000 |
| +0.200 |        544 |          0 | 0.000 | 0.000 | nan | 0.000 |
| +0.400 |         19 |          0 | 0.000 | 0.000 | nan | 0.000 |

![Q-Q max-of-field](../outputs/diffusion_v1/qq_max_of_field.png)

## Figures

- Map comparison: ![](../outputs/diffusion_v1/map_comparison.png)
- Tile-level summaries: ![](../outputs/diffusion_v1/tile_metrics.png)

## Caveats / Notes

- The aggregate tile is the smallest meaningful spatial unit at which we can
  *fairly* compare a generative ensemble to a deterministic observation — pixel
  agreement is not the goal of v1 (see migration plan).
- Prediction stride and ensemble size may have been reduced for iteration
  speed on Apple Silicon; widen them for the data-paper run.
- Coverage rate is computed on the (q025, q975) band — if it drifts far below
  ~0.95, the model is over-confident; if much higher, the bands are too wide.
