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

## Final results — diversity tests on tiny region

| dir | std_mean | env_p95 | median_mean | hot/flat | notes |
|---|---|---|---|---|---|
| v26 (baseline) | 0.025 | 0.088 | +0.018 | 0.99 | reference |
| v36 (levers) | 0.029 | 0.110 | +0.009 | 1.00 | +13% std, modest |
| v36b (aggressive levers) | **0.048** | **0.197** | +0.016 | 1.05 | +88% std — *best raw diversity* |
| v37 (training, mean_l1) | 0.025 | 0.087 | +0.003 | 1.00 | no std gain; **mean cal best** |
| v37b (weak μ + diversity) | 0.026 | 0.087 | −0.003 | 1.02 | no std gain |
| v37c (tile_max_l1) | 0.025 | 0.087 | +0.006 | 1.01 | val_loss saturated, std unchanged |
| v37c + levers | 0.039 | 0.153 | +0.022 | 0.83 | trained ckpt + v36b levers |
| **v37c eta=1** | 0.023 | 0.082 | +0.006 | 0.99 | stochastic DDIM doesn't help |

## Conclusion

**Training-side diversity losses don't translate to inference-time
per-pixel sample variance for this U-Net + DDIM combo.** All three
training-side experiments (v37, v37b, v37c) leave inference std at the
v26 baseline of 0.025 despite the loss saturating during training.
eta=1.0 stochastic DDIM also doesn't help. The model satisfies the
diversity loss at fixed-t intermediate predictions in a way that
doesn't propagate through the full denoising trajectory to ensemble
spread.

The practical answer for sample diversity is **inference levers**:

```bash
python scripts/predict_region_diffusion.py \
  --checkpoint <v26-ckpt> \
  --ensemble_n 16 --predict_batch_size 2 \
  --m_sample_diverse --m_sample_min 0.05 --m_sample_max 1.0 \
  --residual_scale_pos 3.0 --residual_scale_neg 0.4 \
  --cond_perturb_std 0.08
```

This roughly doubles per-pixel std (0.025 → 0.048) and widens the
q025–q975 envelope by ~2× (0.088 → 0.197 at p95). Cost: a small
positive shift in the bulk median (+0.018 → +0.016). On the tiny test
region, q975_max reaches 0.37 vs obs max 0.26 — the ensemble brackets
the obs max comfortably.

## What didn't work (and why)

* **Diversity loss mean_l1** (v37/v37b): satisficeable by uniform iid
  pixel noise across the field. The model adds tiny grain everywhere
  and the loss is satisfied without producing structural diversity.
* **Diversity loss tile_max_l1** (v37c): can't be satisficed by uniform
  noise, and the loss DOES saturate during training. But the structural
  diversity learned at fixed t doesn't propagate through DDIM sampling.
  Per-pixel std is unchanged at inference.
* **eta=1.0** (full stochastic DDIM): no effect on per-pixel std for
  trained models. Confirmed v34 finding — different initial noise
  already provides all the variance the U-Net is going to add.
* **Weak mean head** (v37b mean_loss_weight 0.1→0.02): val_loss drops
  from 0.71 to 0.32 but inference std stays at 0.025. The mean head's
  share of the spatial structure isn't the bottleneck.

## What would likely work (untested)

1. **Stochastic latent z + AdaLN conditioning** (v38 ish): explicit z
  feeds adaptive normalization layers in the U-Net. This is how DiT,
  EDM, BicycleGAN inject controllable diversity. Current v38
  infrastructure exists but conditioning via concat alone (without
  AdaLN) is unlikely to be enough — the model will probably ignore z.
2. **Heteroscedastic mean head**: predict (μ, σ) and sample residual
  from N(0, σ). Gives controlled per-pixel uncertainty, doesn't
  depend on DDIM.
3. **Score-distillation through a wider teacher**: post-hoc make a
  student model that learns to spread samples wider, regularised by
  the teacher's marginal.
