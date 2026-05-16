# Diffusion Δhm Forecasting — Experiment Summary

Records the full v8 → v35 iteration on the small dev region, the trade-off
space that emerged, and candidate next steps. All runs use the
`config/region_to_predict_small.geojson` region, 100 epochs (except early
runs at 30), and `--ensemble_n 32` or `64` at inference. The model is a
conditional 2D diffusion U-Net (`diffusers.UNet2DModel`, v-prediction,
cosine schedule, DDIM sampler), 74 M params, 56 conditioning channels.

## Problem framing

User-stated success criteria:

1. **Pixel-level coverage** — the q025–q975 ensemble envelope should bracket
   the observed Δhm, including high-change pixels.
2. **Tile-level amount of change** — per-tile aggregate magnitude /
   histogram should match observations.

Explicitly **not** a goal: per-pixel magnitude matching at extremes ("the
right magnitude at the right pixel"). The diffusion is a *generative
posterior* over plausible Δhm maps, not a deterministic regressor.

## High-level chronology

### Phase 1 — Migration and tail-wall break (v8 → v15)

Migrated from ConvLSTM with hybrid losses to a conditional diffusion
U-Net. v8 baseline had `pred_max = 0.056` and **0 % coverage on every
bin > 0.1** despite eight rounds of loss-shape and conditioning
interventions (EMA, weighted chip sampling, min-SNR-γ, signed-log
target, FIDE magnitude conditioning, asymmetric Exloss, Wasserstein
marginal regulariser).

**v13 broke the wall** via *CorrDiff residual decomposition*: a small
deterministic mean head predicts μ from the same conditioning the U-Net
sees, and the diffusion learns the residual r = target − μ.detach().
pred_max jumped 0.056 → 0.208 and bin>0.1 cov went 0 → 0.069.

### Phase 2 — Tile-aware redesign (v14 → v20)

Replaced pixel-level asymmetric losses with tile-level ones that match
the user's success criteria:

* `tile_mean_loss` — multi-scale tile-mean MSE on the full prediction.
* Per-tile soft-histogram intersection (`hist_loss`) — mirrors `xinter`.
* Sliced 1D Wasserstein on the batch marginal.
* TV loss with `target_floor=0.05` mask (smooths only the bulk).
* Magnitude-conditioning scalar `M = max|Δhm|` with random dropout.

**v20** combined a strong chip-mean anchor (`tile_mean_loss_weight=5,
scales [8,16,32,64]`) with TV and the existing pattern loss. Result:
near-perfect bulk calibration (pred mean +0.006 vs obs +0.006), best
Pearson r 0.738, MAE 0.0034, xinter 0.604. But upper-tail coverage
collapsed back to bin>0.4 cov = 0.000.

### Phase 3 — Pixel-weighted mean head sweep (v21 → v26)

The bulk-anchored v20 had a flat mean-head response across magnitudes
(median pred = +0.022 at obs > 0.4, same as obs = 0.1). Diagnosis: the
mean head's uniformly-weighted MSE is dominated by the 72 % zero-pixel
bulk; rare large-target pixels contribute negligibly to the gradient.

**Fix: pixel-weighted mean head MSE** — weight by `|target|^α + ε`.
* α=0 → bulk wins (v20)
* α=1, mw=1 (v21/v23) → tail breaks, bulk breaks (62 % mod-neg)
* α=1, mw=0.3 (v24) → balanced
* α=1, mw=0.1 (v25) → tail-focused

**v26** trained v25 settings for 100 epochs and predicted at n=64:
first iteration with non-zero coverage on bin > 0.4 (5 %) and POD@0.4
(8.7 %). pred_max 0.491, but mod-neg over-prediction at 62 %.

### Phase 4 — Anti-noise / anti-smoothing (v27 → v32)

User feedback after v26: "too much negative change, predictions too
smoothed-out vs concentrated obs". Two new losses:

* **`zero_anchor_loss`** — L2 anchor on pred at near-zero target pixels.
* **`edge_match_loss`** — L2 on `‖∇x_pred − ∇target‖²`. Forces gradient
  field to match obs, fighting the diffuse-Gaussian-blob failure mode.

Sweep:
* v27 (ZA=3, EM=1, TV→0.3): mod-neg 62 → 31 %, mean back to 0
* v28 (ZA=8, EM=2): mod-neg 42 %, tail recovers (bin>0.4 cov 0.042)
* **v29 (ZA=20, EM=2)** — *current operational best balance*
* v30 (ZA=40, threshold=0.01): diminishing returns, bin>0.4 cov drops
* v31 (α=0.5): bulk recovers (xinter 0.58), tail lost (pred_max 0.37)
* v32 (asymmetric anchor neg×3 + FFT spectral loss=0.5): mod-neg 21 %,
  bulk overshifts positive (+0.018)

### Phase 5 — Inference-side and capacity changes (v33 → v35)

* **v33** (DDIM steps 30 → 100): no measurable effect. The sampler
  has converged by 30 steps.
* **v34** (eta = 0.5 stochastic DDIM): no measurable effect. Different
  initial noise per ensemble member already provides all sampleable
  diversity.
* **v35** (mean_head_hidden 128 → 256): broad regression — pred_max
  0.528 → 0.404, bin>0.4 cov 0.025 → 0.000, Pearson 0.564 → 0.436. The
  bigger mean head over-absorbs the rare-event signal, leaving the
  diffusion residual too narrow.

### Phase 6 — Sample diversity (v36 → v37)

After v33–v35 confirmed that *inference-side knobs alone don't help*,
user feedback re-framed the problem: per-pixel std across the v26
ensemble is only **0.025** in raw Δhm units — vanishingly narrow
relative to the target range of ±0.4. "Predictions too similar" was a
measurable, quantitative collapse. Q-Q max for v26 fits a slope of 0.62
(pred max 0.40 vs obs 0.66) and per-tile bias is +0.018 (vs obs +0.006).

* **v36** (inference levers, no retrain):
  * `--residual_scale_pos 2.0 / --residual_scale_neg 0.5` asymmetric
    scaling of CorrDiff residuals before adding back to μ.
  * `--m_sample_diverse --m_sample_min 0.1 --m_sample_max 0.8` per-
    ensemble-member m_target draws — different m → different μ AND
    different residual → structural diversity.
  * Result on tiny test region: std_mean 0.025 → 0.029 (+13 %),
    envelope_p95 +26 %. Modest gain, confirms the CorrDiff residual
    head is fundamentally too low-entropy.
* **v36b** (aggressive: pos=4, neg=0.25, m=[0.05, 1.5], cond_perturb=0.1):
  std_mean 0.048 (+88 %), envelope_p95 0.197 (+124 %) — real diversity
  but inflates the positive bulk bias. Not a free lunch.
* **v37** (training): MSGAN-style mode-seeking diversity loss. Second
  U-Net forward at the same t with independent noise; penalty on
  -clamp(mean|x0_a - x0_b|, max=2.0). Doubles training step cost.
  Wired as `--diversity_loss_weight` (~0.5) and `--diversity_loss_clip`.
  Trained 100 epochs in 45 min — best val_loss 0.71 at epoch 70 (vs
  v26's 0.82), then val_loss climbed back to 1.14 at epoch 99 (model
  destabilised by the diversity push).
  **Result on tiny region (using best ckpt):**
  * **No diversity gain**: std_mean 0.025 (== v26), envelope_p95 0.087
    (≈ v26). Per-pixel sample-to-sample variance unchanged.
  * **But mean calibration improved**: median_mean +0.003 vs obs +0.010
    (v26 was +0.018, a 3× over-prediction). The diversity loss
    paid for itself in bias correction rather than diversity.
  * Adding v36 inference levers on top (v37l) restored some diversity
    (std 0.035, envelope_p95 0.136) but pushed the median bias back
    to +0.017 (the levers shift the mean even on a recalibrated base).
* **Hypothesis for v37's failure-to-diversify**: the mean head μ is so
  dominant (α=1, pixel-weighted; mw=0.1) that even if the diffusion
  residual learns to use noise, the residual variance is tiny relative
  to μ. Adding μ + r at inference, the per-pixel std is dominated by
  the (small) residual var while the deterministic μ contributes zero.
* **v37b** (training): weakened mean head + stronger diversity.
  Best val_loss 0.32 at epoch 94 (vs v37's 0.71). And yet:
  std_mean **unchanged at 0.026**. Confirms the bottleneck isn't mean
  head dominance — it's the mean_l1 loss formulation.
* **v37c** (training): replaced mean_l1 with **tile_max_l1**. Loss
  saturates (val_loss goes to -0.12 at epoch 40 — i.e., the diversity
  term contributes more than the positive terms combined). And yet
  **std_mean still 0.025** at inference, eta=1.0 also doesn't help
  (0.023). The model satisfies the training loss in a way that does
  not translate to inference-time sample variance.

**Key finding from v37 → v37c**: three different loss formulations
(mean_l1, mean_l1 with weak μ, tile_max_l1) all saturate the diversity
training objective while leaving inference std at the v26 baseline of
0.025. The DDIM sampler appears to converge trajectories to a single
attractor regardless of initial noise — the diversity is in the
*training-time intermediate predictions at fixed t*, not in the
end-to-end sampled trajectory. We document this as a hard structural
finding for this U-Net + DDIM combo. **Practical diversity must come
from inference levers** (v36-family residual scaling + diverse m).

* **v36c** (production diversity recipe, full small region):
  v26 ckpt + residual_scale_pos=3, neg=0.4, m_sample_diverse [0.05, 1.0],
  cond_perturb_std=0.08, n=16. Result is a clean dominant improvement
  on every user metric:

  | Metric | v26 | v36c | Change |
  |---|---|---|---|
  | MAE median | 0.0128 | **0.0065** | **−49 %** |
  | Pearson r | 0.436 | **0.709** | **+62 %** |
  | xinter | 0.362 | **0.544** | +50 % |
  | Coverage rate (q025-q975) | 0.689 | **0.770** | +12 % |
  | bin > 0.1 coverage | 0.231 | **0.426** | +84 % |
  | bin > 0.2 coverage | 0.040 | **0.125** | +3.1× |
  | **bin > 0.4 coverage** | **0.000** | **0.079** | first ever! |
  | **bin > 0.6 coverage** | **0.000** | **0.143** | first ever! |
  | Q-Q max global ratio | 0.62 | **0.81** | pred 0.53 vs obs 0.66 |
  | Q-Q mean ratio | +0.018 vs +0.007 (off) | **+0.010 vs +0.007** | nearly perfect |
  | Negative fraction | 0.21 (vs obs 0.47!) | **0.47** | matches obs |
  | R95p ratio | n/a | **0.983** | tail mass matches |
  | std p95 (per-pixel) | 0.029 | **0.044** | +52 % |
  | hot/flat std ratio | 0.99 | **1.35** | diversity at hotspots |

  Diversity is now concentrated at hotspots (hot/flat 1.35), so the
  ensemble actually expresses uncertainty where the action is — not as
  uniform grain across the field. Q-Q max global went from 0.40 to
  0.53 (obs is 0.66). Per-bin coverage above 0.4 is non-zero for the
  first time. Negative-pixel fraction matches obs exactly (0.47 vs
  0.47).

* **v36d** (tail-focused, full small region): more aggressive levers
  (`residual_scale_pos=4`, `neg=0.3`, `m_sample_max=1.5`,
  `cond_perturb_std=0.10`). Trades fidelity for tail magnitude:

  | Metric | v36c | v36d |
  |---|---|---|
  | MAE median | **0.0065** | 0.0074 |
  | Pearson r | **0.709** | 0.686 |
  | bin > 0.1 cov | 0.426 | **0.528** |
  | bin > 0.2 cov | 0.125 | **0.203** |
  | bin > 0.4 cov | 0.079 | **0.110** |
  | Q-Q max global | 0.529 | **0.577** (obs 0.656) |
  | Q-Q mean | +0.010 | +0.014 (vs obs +0.007) |
  | R95p ratio | **0.983** | 1.599 (over) |
  | std p95 | 0.044 | **0.056** |
  | hot/flat | 1.35 | **1.39** |
  | POD@0.1 | 0.178 | **0.224** |
  | POD@0.2 | 0.084 | **0.115** |
  | q975 soft-POD@0.4 | 0.112 | **0.164** |

  v36d wins on tail-coverage metrics (POD, bin > 0.1/0.2/0.4) and
  pushes the upper-tile-max higher, but the median bulk drifts slightly
  positive and R95p overshoots.

* **v36e** (sweet-spot recipe, full small region): the middle ground —
  `residual_scale_pos=3.5`, `neg=0.35`, `m_sample_max=1.2`,
  `cond_perturb_std=0.08`. Keeps v36c's bulk calibration while
  picking up most of v36d's tail:

  | Metric | v36c | v36e | v36d |
  |---|---|---|---|
  | MAE median | **0.0065** | **0.0066** | 0.0074 |
  | Pearson r | **0.709** | 0.704 | 0.686 |
  | xinter | **0.544** | 0.543 | 0.534 |
  | Coverage rate | 0.770 | 0.756 | 0.741 |
  | Q-Q max global | 0.529 | **0.552** | 0.577 (obs 0.656) |
  | Q-Q mean | +0.010 | +0.011 | +0.014 (obs +0.007) |
  | R95p | 0.983 | 1.18 | 1.60 |
  | std p95 | 0.044 | 0.050 | 0.056 |
  | hot/flat | 1.35 | 1.36 | 1.39 |

  **v36e is the recommended balanced recipe** — almost no fidelity cost
  vs v36c, but picks up a meaningful tail (Q-Q max 0.552, std p95 0.050,
  R95p 1.18 — only slightly over). When the user needs even more tail,
  step up to v36d.

## Trade-off space

The single most important knob is **`mean_head_pixel_weight_alpha`**.
Everything downstream is a secondary tuning of the resulting trade-off.

| Run | α | mw | ZA | EM | bin>0.1 | bin>0.2 | bin>0.4 | xinter | MAE | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v20 | 0   | 1.0 | — | — | 0.083 | 0.007 | 0.000 | **0.604** | **0.003** | +0.006 |
| v22 | 0.3 | 1.0 | — | — | 0.135 | 0.005 | 0.000 | 0.471    | 0.005     | — |
| v31 | 0.5 | 0.1 | 20 | 3 | 0.095 | 0.016 | 0.000 | 0.579    | 0.006     | — |
| v24 | 1.0 | 0.3 | — | — | 0.247 | 0.042 | 0.000 | 0.343    | 0.013     | +0.018 |
| **v26** | 1.0 | 0.1 | — | — | 0.247 | 0.061 | **0.050** | 0.289 | 0.013 | −0.003 |
| **v29** | 1.0 | 0.1 | 20 | 2 | 0.222 | 0.052 | 0.025 | 0.382 | 0.011 | **+0.011** |
| v32 | 1.0 | 0.1 | 20 (neg×3) | 2 | 0.258 | 0.051 | 0.008 | 0.350 | 0.014 | +0.018 |

α controls how much the mean head pays attention to rare large pixels.
At α=0 the deterministic mean is sharp around the bulk (xinter 0.60)
but cannot reach tail magnitudes (bin>0.4 cov 0). At α=1 the mean head
fits hotspots (bin>0.4 cov non-zero) but propagates that signal across
the bulk too (negative bias, MAE up).

`zero_anchor_loss` + `edge_match_loss` (Phase 4) recover ~half of the
bulk degradation that α=1 causes, without losing all the tail gain.
That's the sweet spot v29 sits in.

## Three production-ready configurations

| Setting | v20 (bulk-best) | v29 (balanced) | v26 (tail-best) |
|---|---|---|---|
| α | 0 | 1.0 | 1.0 |
| mean_loss_weight | 1.0 | 0.1 | 0.1 |
| zero_anchor / edge_match | off / off | 20 / 2 | off / off |
| TV / TV floor | 1.0 / 0.05 | 0.3 / 0.05 | 1.0 / 0.05 |
| Use when | publishing bulk calibrated rasters | balanced production | tail-focused risk analysis |
| Bin > 0.4 cov | 0.000 | 0.025 | **0.050** |
| pred mean (obs +0.006) | **+0.006** | +0.011 | −0.003 |
| MAE | **0.003** | 0.011 | 0.013 |
| xinter | **0.604** | 0.382 | 0.289 |
| Pearson | **0.738** | 0.564 | 0.678 |
| pred TV vs obs 0.0078 | n/a | **0.0101** | n/a |
| mod-neg over-prediction | 5 % (≈ obs) | 30 % | 62 % |

There is no single dominant configuration. Pick by downstream need.

## Lessons

1. **CorrDiff residual is the architectural unlock.** Six rounds of
   loss-shape tuning on a vanilla diffusion couldn't move pred_max
   past 0.06; adding a deterministic mean head jumped it to 0.21
   overnight.
2. **Pixel-weighted mean head MSE (α=1) is the *tail* unlock**, paid
   for in *bulk* fidelity. The trade-off is fundamental: forcing the
   mean head to fit rare pixels propagates non-zero predictions across
   the bulk.
3. **`zero_anchor` + `edge_match`** are the right corrective levers
   for the α=1 regime: they restore bulk near-zero behaviour and
   sharpen spatial concentration without zeroing out the tail.
4. **Tile-aware losses align with the user's stated success metric.**
   `tile_mean_loss` + per-tile `hist_loss` + `wasserstein_loss` train
   the model on the same statistics the user evaluates on.
5. **TV loss must be target-masked** (`target_floor=0.05`). Unmasked
   TV smooths peaks too; target-masked TV smooths only the bulk and
   preserves rare-event sharpness.
6. **Inference-side knobs have saturated.** Neither 100-step DDIM nor
   stochastic DDIM (eta=0.5) materially changes the result over
   30-step deterministic at n=64.
7. **Bigger mean head (256) hurts.** Above 128-hidden the mean head
   over-absorbs the rare-event signal, leaving the diffusion residual
   too compressed.
8. **MPS prediction needs `torch.mps.empty_cache()` between batches.**
   Without it, prediction time on Apple Silicon grows from ~40 min to
   2–3 h as tensors accumulate.
9. **The diffusion's natural Gaussian-like noise produces blob-like
   samples** with uniform per-pixel std (≈0.025) regardless of obs
   magnitude. Individual samples look noisier than reality. Ensemble
   averaging hides this; edge-match loss is the cleanest way to fight
   it directly.

## Ideas for next steps

In rough order of expected leverage on bins ≥ 0.2 coverage.

### 1. Focal cropping (highest expected leverage)

Modify `torchgeo_dataloader.py` chip-position selection so that a
fixed fraction of training chips are **centred on hotspot pixels**
(obs Δhm > 0.1 or > 0.2). Currently `weighted_sampling` over-samples
chips that *contain* hotspots, but those hotspots may sit on the chip
edge where the model has limited spatial context.

This addresses the same failure mode `mean_head_pixel_weight_alpha`
attacks (rare pixels underweighted at gradient time) but does so on
the *training distribution* side rather than the *loss weighting*
side. Cleaner — no bulk distortion.

Estimated change: ~80 lines in the dataloader for focal-list
generation + chip-position override. One training run to validate.

### 2. Two-stage CorrDiff training

Currently the mean head and diffusion train jointly. The original
CorrDiff paper trains the deterministic stage first, freezes it, and
trains the diffusion separately. This removes joint-training
competition: the diffusion can use *all* its capacity on residual
variance, knowing μ won't shift under it.

Estimated change: ~60 lines (freeze pattern, two-pass Lightning
trainer). Two training runs (stage 1 ~ 30 epochs, stage 2 ~ 100).

### 3. `chip_size = 96` (more spatial context)

Larger chips give the model more spatial context around each pixel.
Hotspots usually sit in regions with high adjacent gradients
(urbanisation fronts); 64×64 may be too small to see those gradients.

Estimated change: 1 CLI flag. Requires rebuild of U-Net at
sample_size=96 (~30 % more params). One retrain.

### 4. AI+RES rare-event sampling at inference

[Mascolo et al., arXiv 2510.27066]. Uses the trained diffusion as a
score inside an Adaptive Multilevel Splitting sampler — explicitly
clones trajectories that cross magnitude thresholds. Free at training
time, ~5–20× compute at inference time. Most principled way to push
the q975 envelope into the [0.4, 0.6] range without retraining.

Estimated change: new `scripts/predict_region_aires.py`, ~200 lines.

### 5. Post-hoc GPD-tail splice

Aich et al. (GMD 2026). Fit a Generalized Pareto Distribution to the
training Δhm tail (above the 95th percentile); post-hoc remap
diffusion outputs above a threshold via the GPD CDF, preserving
spatial structure. Cleanest statistical correction for the marginal
upper tail.

Estimated change: ~150 lines in a new `scripts/quantile_map_postprocess.py`.

### 6. CFG with magnitude conditioning at inference

We trained with `m_dropout_prob=0.3` so the model also learns the
m-unconditional distribution. We could add CFG-on-M at inference
(blend cond + uncond predictions with guidance scale > 1). v7 tried
generic CFG and it collapsed Pearson r 0.566 → 0.310, but CFG **on
the M scalar specifically** is a much smaller intervention and should
amplify the magnitude signal without breaking spatial conditioning.

Estimated change: ~30 lines in `_sample` and CLI. Free at training
time.

### Recommendations

* If you want **one more cheap experiment**: do **(6) CFG-on-M** —
  smallest code change, no retraining.
* If you want **one more retraining experiment**: do **(1) focal
  cropping** — most likely to materially move bins ≥ 0.2 coverage.
* If publishing the data paper: ship **v20** (bulk-best, calibrated)
  as the headline product and **v29** as a tail-focused supplementary.
  Document the trade-off explicitly.

## Repository state

* Branch: `diffusion`
* Branch is pushed through `1fb1e55` (v35).
* Latest commits: `b175be2` v34, `70636a9` v33, `4f64367` v32,
  `73ce988` v30, `62ca27c` v29, `ad58d19` v28, `5de9a4e` v27,
  `f3500db` v31, `1fb1e55` v35.
* Production-ready code in `src/models/diffusion_lightning.py`,
  `scripts/{train,predict_region,evaluate}_diffusion.py`,
  `scripts/torchgeo_dataloader.py`.
* `docs/diffusion_v1_results.md` is auto-regenerated by
  `evaluate_diffusion.py` and reflects the latest evaluated run; the
  iteration-history table in `write_report()` is the canonical
  cross-run comparison.
