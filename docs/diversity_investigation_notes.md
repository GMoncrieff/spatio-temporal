# Sample Diversity Investigation Notes

Working notes from the v36 → v37 → v37b → v37c → v38 push.

## Problem statement

User feedback after v26 ship: per-pixel std across the v26 ensemble is
**0.025** raw Δhm — narrow enough that median, q025, q975 maps are
nearly indistinguishable. "Samples too similar." Goal: produce a wider,
structured posterior so the ensemble actually disagrees on *which*
spatial pattern wins, not just on uniform per-pixel grain.

Constraints (from prior feedback):
- Pixel coverage at high-change bins is the primary success metric.
- Tile-level magnitude (Q-Q mean, Q-Q max) should approach 1:1.
- Pixel-precise matching of extremes is explicitly NOT a goal.

## Findings so far

### v36 / v36b (inference-only levers)

Three knobs added to predict_region_diffusion.py:
1. `--residual_scale_pos / --residual_scale_neg`: asymmetric scaling on
   CorrDiff residuals before adding back to μ.
2. `--m_sample_diverse + --m_sample_min/max`: each ensemble member draws
   its own m_target from a uniform prior.
3. `--cond_perturb_std`: additive Gaussian noise on the full conditioning
   per sample.

On tiny test region:
- **v36** (pos=2, neg=0.5, m∈[0.1, 0.8]): std 0.025 → 0.029 (+13 %).
- **v36b** (pos=4, neg=0.25, m∈[0.05, 1.5], cond_perturb=0.1):
  std 0.048 (+88 %), envelope_p95 0.197 (+124 %). Big diversity gain
  but the asymmetric pos/neg scaling shifts the bulk positive.

### v37 (training-side MSGAN-style diversity loss)

Added `_diversity_loss(x0_a, x0_b) = -clamp(mean|x0_a - x0_b|, 2.0)`
with a second U-Net forward at the same timestep with an independent
noise draw.

Trained 100 epochs with `diversity_loss_weight=0.5`, otherwise v26
recipe. Best val_loss 0.71 at epoch 70 (vs v26's 0.82).

**Result on tiny region (epoch 70 ckpt):**
- std_mean **unchanged** at 0.025
- envelope_p95 0.087 (similar to v26)
- BUT median_mean +0.003 (vs v26 +0.018) — best mean calibration ever.

The diversity loss *engaged* (val_loss dropped) but **did not translate
to inference-time sample variance**. Hypothesis: mean head μ is so
dominant that the diffusion residual's variance is structurally tiny.
The loss pushed the residual to differ between paired (noise_a, noise_b)
draws — but only by uniform iid noise, which the loss accepts as
"diverse" even though it doesn't read as diversity at inference.

Spatial diagnostic (hot/flat std ratio) confirms this:
- v26: hot/flat = 0.99 (uniform across the field)
- v37: hot/flat = 1.00 (uniform — no concentration at hotspots)

If diversity meant anything structurally, we'd expect *higher* std near
hotspots (more uncertainty about magnitude there) and *lower* std in
flat areas (the model is confident those are zero). The flat 1:1 ratio
says the model is just adding uniform grain.

### v37b (weak mean head + strong diversity, in progress)

Reduces `mean_loss_weight 0.1 → 0.02` so μ can't absorb all the signal,
AND doubles `diversity_loss_weight 0.5 → 1.0`. Hypothesis: a weaker μ
leaves more entropy budget for the residual to express, so the
diversity loss can actually drive sample-to-sample variance up.

Status (epoch 44): val_loss 0.51 — significantly lower than v37's 0.71
and v26's 0.82. The diversity term is contributing materially. Whether
this translates to inference diversity is the next test.

### v37c (tile_max_l1 diversity, queued)

`mean_l1` can be satisficed by uniform iid noise. `tile_max_l1` instead
penalises difference of *per-tile max* values:

    diff = mean over tiles of |max_a - max_b|

Uniform noise across all pixels has roughly the same max in each tile,
so it can't satisfy this — the model has to put different magnitudes
in different tiles across paired samples. This targets **structural**
diversity directly.

### v38 (latent z conditioning, infrastructure ready)

Adds an explicit z ~ N(0, I) channel block to conditioning. Different
z per ensemble member at inference → different output by design (if
the model has learned to use z). Pairs naturally with the diversity
loss — the loss now sees variation in (noise, z) jointly.

Risk: model may simply ignore z if the diffusion noise already provides
sufficient stochasticity (and currently it doesn't, but the loss might
push it to use z preferentially).

## Decision tree

```
v37b training done →
  std > 0.04?
    YES → predict on full region, evaluate Q-Q max / mean
    NO → launch v37c (tile_max_l1)
      v37c done →
        std > 0.04 AND hot/flat > 1.3?
          YES → predict on full region
          NO → launch v38 (latent z) on top of best loss formulation
```

## Open questions

1. Is uniform per-pixel noise (current behavior) actually useful for
   the user, even if it doesn't read as "diversity" visually? The
   envelope (q025 - q975) does still capture obs in many places.
2. Is the v37 mean-calibration win (+0.003 vs v26 +0.018) worth keeping
   on its own? Q-Q mean would be closer to 1:1 just from this.
3. Does CorrDiff fundamentally cap residual diversity? Maybe a
   sigma-head (heteroscedastic) is the right architectural fix.

## Pending experiments

1. v37b post-training prediction + diversity_diag (5 min).
2. v37c (if v37b uniform-noise pathology persists) — ~85 min train.
3. Full-region prediction with best v37* ckpt (~2-3h on MPS, depends
   on which approach wins).
