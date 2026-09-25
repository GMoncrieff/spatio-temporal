# Next Phase — Improve the ConvLSTM Itself

Branch `ensemble`, written 2026-08-16. Companion to `docs/next_phase_marginals.md` (the
post-hoc phase that just closed) and `docs/central_field_baseline.md` (the last time the
model itself was changed).

## 1. Why the post-hoc lever is exhausted

The marginal phase took the scorecard 90/126 → 101/127 and the per-member metric 7/20 → 15/20
without retraining anything. It also mapped the ceiling precisely, and the ceiling is the
model's.

**The binding constraint is structural.** `fit_residual_shape` normalizes each side by that
side's own 2.5/97.5 quantile — which is exactly what pins T5.2 — so the fitted marginal is the
residual's distribution *stretched to fill the published interval*. Measured stretch at h=20:
1.3-3.3× up, 1.3-7.1× down. **Any T5.2-preserving marginal therefore re-injects whatever
width error the published bounds carry, and no change to the shape can reach it.** Width
factors can rescale the bounds post-hoc, and do, but they are a per-class scalar applied to a
raster the model emitted — they cannot give the model information it did not use.

Four post-hoc ideas were tried and rejected on measurement, not taste:

| tried | verdict |
|---|---|
| per-band marginal *shapes* (the phase's own H1) | lost on every metric; the *tail bound* is what paid |
| central-forecast conditional bias correction | costs 10% of skill (median shift vs right-skewed residual); the rows it targeted became unscoreable |
| HM-conditional tail bound | rejected on **two** held-out protocols — temporal windows (0.775 vs 0.505) and spatial folds on 33× the data (0.932 vs 0.918) |
| extra conditioning axes on the shape (Δ̂, w_up) | nulled by construction — the fit renormalizes, so any axis acting on scale disappears |

## 2. What the model is measurably getting wrong

All from the reference configuration's own residuals, southern Africa, k=5, out-of-sample.

**The central forecast has a conditional bias that scales with what it predicts.** Median
standardized residual by predicted-change class:

| Δ̂ class | h=5 | h=10 | h=15 | h=20 |
|---|---|---|---|---|
| (−0.01, 0.001] | −0.03 | −0.02 | −0.02 | −0.03 |
| (0.001, 0.01] | −0.09 | −0.11 | −0.21 | −0.28 |
| (0.01, 0.05] | −0.47 | −0.29 | −0.29 | −0.30 |
| (0.05, 0.15] | — | — | **−0.79** | **−0.60** |

Where it predicts substantial change it lands 0.6-0.8 half-widths high. This is a *training*
defect: it cannot be corrected post-hoc without paying RMSE, because RMSE follows the mean and
this residual is strongly right-skewed (median −0.0233 vs mean −0.0117 in that class).

**The intervals are too wide, by a factor that varies strongly across three axes.** The
half-width that would make the published interval the residual's own 95% interval:

- by distance band at h=20: 0.77 / 0.70 / 0.54 near past change, and *above 1* in the far
  field at short lead (1.785 at h=5, 30-100 px — that interval is too **narrow**);
- by predicted change within the 0-1 px band at h=20: 0.75 for Δ̂ ∈ (0.01,0.05] against 0.95
  for (0.05,0.15];
- by HM level within a fixed (band × Δ̂) cell: still varies 1.4-81×.

A model whose width heads used these covariates properly would not need any of the post-hoc
factors.

**Two failures nothing post-hoc reaches:**

1. *Near-pristine land, HM < 0.01.* Rank of the observation among 400 members is 400/400 at
   +5/+10 yr and 24/400 at +20 yr — **the sign of the error flips with lead time**. No single
   width factor or tail bound can do that.
2. *The far field at short lead.* At h=5, 30-100 px, the +0.05 threshold sits at a **median of
   119 half-widths**. Only 0.17% of that band's pixels are within reach of the fitted shape at
   all. The interval there is far too narrow relative to what actually happens.

## 3. Scope for this phase

**Preserve:** the ConvLSTM architecture overall, and the input covariates. New covariates only
if *derived* from what already exists (past change, distance to past change at the five
`CONTEXT_RADII`, HM_t0, the multi-scale context rasters).

**In play:** minor architectural changes, loss functions and their weighting, hyperparameters,
training schedule and budget, head parameterisation, target transforms.

**Judge on:** the central field *and* the quantiles, both stratified — by distance to past
change, by baseline HM level, and by change magnitude. Then, decisively, **on the downstream
ensemble**, because that is what the project delivers. A model change that improves a
validation loss and not the scorecard or the per-member metric has not improved anything.

## 4. What the previous model change taught

From `docs/central_field_baseline.md`, the last time the model was touched:

- **The central head was solving the wrong problem.** Predicting absolute HM means reproducing
  HM_t0 through the trunk before anything useful can be added, at a cost (sd ≈ 0.0075) larger
  than the entire signal. `--central_residual` made "nothing happens" free and was the large
  effect.
- **The ~10 px trunk radius is the binding structural limit**, for the central head exactly as
  it was for the quantile heads. Supplying precomputed long-range context is what turned h=20
  from a regression into the largest gain. Widening the trunk itself — dilated convolutions,
  downsampled branches — is the general version and **has never been tried**.
- **Training budget is the cheapest untested lever.** 150 epochs × 13 steps × 8 chips ≈ one
  pass over the globe's valid pixels. Run-to-run variance between identically configured folds
  is comparable to the effects being chased (h=5 RMSE 0.01347 vs 0.01478).
- **`ModelCheckpoint` monitors `val_total_loss`, which includes pinball**, so a quantile-only
  change still selects a different epoch and therefore a different central field. Central-only
  A/Bs need a central-only monitor.

## 5. Scope caveat that applies to everything below

The reference configuration is calibrated on southern Africa, which changes **2-4× less than
Africa as a whole** at every HM level, and whose HM distribution is unrepresentative — the
`[0, 0.01)` stratum is 6% of southern Africa and 40% of Africa. Every "too hot" verdict in the
marginal phase was measured against an unusually quiet target. Africa-wide residuals exist at
`data/ensemble/exp/africa_k5/` (35.65M px per window × horizon, five-fold stitched) if a
finding ever needs checking against a fairer sample.
