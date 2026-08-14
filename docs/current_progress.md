# Current Progress — Spatiotemporal HM Uncertainty

Status as of 2026-08-14, branch `ensemble`. Companion to
`docs/ensemble_uncertainty_plan.md`, which holds the design and the full target definitions.
This document records what exists, what was learned, how we score against each target, and
where the remaining leverage is.

> **The central field was revisited and is substantially better.** See
> `docs/central_field_baseline.md` for the measurement, the per-flag attribution and the
> negative results. Headline, k=5 fold-stitched over the whole region in both cases:
> RMSE −36% / −17% / −9% / −8% across horizons, MAE −62% / −41% / −26% / −25%, and skill
> against persistence goes from **−1.23 / −0.24 / +0.01 / +0.05** to
> **+0.09 / +0.15 / +0.18 / +0.19** — positive at every horizon for the first time, and
> increasing with lead time rather than decreasing. The scorecard goes 73/137 to 91/128.
> The h=10 anomaly is explained (non-monotone interval width) and fixed.

**Development scale**: all iteration happens on the southern-Africa subregion by user
instruction. The global k=5 hindcast is complete and on disk; global runs happen only on
explicit instruction.

---

## 1. What has been built

### Phase 0 — out-of-sample residual harness
- Fixed a confirmed bug: `HumanFootprintChipDataset` grid mode ignored the split mask
  entirely, so val/test metrics were computed over the whole globe rather than held-out
  geography. **All `val_coverage_*` numbers from before this fix are not comparable to
  numbers after it.**
- `create_kfold_splits(k=5)` reusing the existing chip enumeration and seed; the production
  `split_mask_1000.tif` is untouched.
- `train_lightning.py` extended additively: `--predict_input_years`, `--predict_all_windows`,
  `--predict_output_prefix/_dir`, `--predict_max_target_year`, `--predict_restrict_mask`,
  `--fold_mask/--exclude_fold/--val_fold`, `--norm_stats_json`, `--val_stride`, `--devices`,
  W&B run naming, `--split_mask`.
- Global run: 5 folds x 4 windows, 421 min wall clock on 2 GPUs, producing 10 residual maps
  of 184,573,321 genuinely out-of-sample pixels each.

### Phase 1 — diagnostics
- RESOLVE Ecoregions2017 rasterised to the 1 km grid (846 ecoregions, area ratio median
  1.0000, 3.95% of valid HM pixels falling on `ECO_ID == 0`).
- Coverage vs aggregation scale, both propagation constructions; class-conditional coverage
  audit with separate tails and `n_eff` counted as distinct 128 px chips; biome-stratified
  variogram fitting via `gstools`.

### Phase 1.5 — recalibration
- Mondrian split conformal with hierarchical shrinkage (cell -> primary stratum -> horizon),
  isotonic smoothing, monotone *spread* across horizons, and leave-one-fold-out evaluation.
- Decision rule now uses held-out **interval score**, not coverage.

### Phase 2/3 — field generation and copula
- Circulant-embedding fields on GPU, Matérn or Gaussian kernel, longitude wrap, AR(1)
  horizon coupling.
- `fit_spectral_mixture`: NNLS fit of the mixture to the observed radial power spectrum.
- Median-spliced two-piece normal marginals reproducing lower/central/upper exactly in
  closed form; int16 zarr storage with seeds recorded for deterministic regeneration.

### Phase 4 — validation
- One script scoring T1–T8 into a pass/fail scorecard with per-stage fault isolation and
  partial-result flushing, W&B tables and figures, and the "which knob fixes which failure"
  diagnosis attached to every failure.

### Model change (constraint #1 lifted by the user)
- Quantile heads now receive multi-scale past-change context; trunk and central heads are
  frozen during retraining (verified live: both gradient norms exactly 0.000000).
- `scripts/prepare_change_context.py` builds the context rasters on the full grid.

### Central-field change (the central forecast is no longer sacred either)
Three flags on `train_lightning.py`, attributed separately on held-out folds:
- `--central_residual` — central heads predict change on top of HM_t0, output convolution
  zero-initialised so training starts at exact persistence. The large effect.
- `--central_context` — central heads read the same past-change context as the quantile
  heads. This is what fixes h=20, which the residual head alone left fractionally worse.
- `--monotone_quantile_width` — half-widths accumulate across horizons around the
  *detached* central forecast, making spread non-decreasing in lead time and
  `lower ≤ central ≤ upper` structural rather than a post-hoc clip.

Supporting harness: `scripts/diagnose_central_field.py` (stratified central-field error),
`scripts/compare_central_runs.py`, `scripts/run_central_experiment.sh` (one experiment:
train held-out folds, predict the region, stitch) and `scripts/run_region_loop.sh`
(Phase 1→4 for one experiment, re-deriving recalibration, spectrum and AR(1) from that
model's own residuals).

### Tests
101 ensemble/model tests pass. 7 failures in the pre-existing suite are stale (they assert a
4-channel output from a model that has emitted 12 since the independent-heads work) and fail
identically with this branch's changes stashed.

---

## 2. What we learned

1. **The quantile heads were structurally blind, not mis-trained.** ~10 px receptive radius
   cannot see whether change occurred 30–100 px away. Supplying that covariate fixed the
   dominant defect; no loss reweighting could have.
2. **Inverse-frequency class weighting is backwards for this problem** — it upweights the
   rare near field, widening the shared head.
3. **Long-range context must be computed on the full raster**; deriving it from a 128 px
   chip saturates every radius ≥ 30 px into "is there any change in this chip".
4. **With well-behaved heads, conformal recalibration does net harm** — and coverage alone
   cannot see that, because coverage always prefers the widest interval.
5. **Pooled coverage hides class failure**: 0.966 pooled coexisted with 0.335 in the
   high-change class.
6. **The motivating "coverage collapse" needs the right baseline.** Averaging published
   bounds over a block over-covers (0.98 → 1.00); independent propagation collapses
   (0.284 → 0.045 → 0.009). The truth is bracketed between them.
7. **Variograms miss mid frequencies and Gaussian kernels cannot make texture.** Fit the
   spectrum; use Matérn.
8. **A mean-subtracted spectrum is blind to k=0**, so aggregate spread has to be calibrated
   from aggregate error, not from the spectrum.
9. **Measurement bugs outnumbered model bugs.** Mismatched masks, a quantisation-scale
   tolerance on a sample median, globally-scattered sampling for structure scores, and two
   runs scoring the wrong checkpoint all produced plausible but meaningless numbers.
10. **The central head was solving the wrong problem.** Predicting absolute HM means
    reproducing HM_t0 through the trunk before anything useful can be added, and the cost of
    that reproduction (sd ≈ 0.0075 HM on pixels that did not change) exceeded the entire
    signal being predicted. Predicting change instead makes "nothing happens" — the answer
    for 53–70% of the map — free.
11. **Skill has to be scored against persistence, not against zero.** Pooled RMSE looked
    unremarkable while the model was losing to "nothing will change" by a factor of 2.2 in
    MSE at h=5. No coverage or interval metric could have surfaced that.
12. **The ~10 px trunk radius limits the central head exactly as it limited the quantile
    heads.** Supplying the same precomputed context is what turns h=20 from a regression
    into the largest skill gain.
13. **A metric pinned at 1.000 and insensitive to its own knob is under-powered, not
    mis-tuned.** `long_weight` was swept 0.15→0.70 against failing T2 rows; they never
    moved, while spread-skill ran 0.82→1.52. The rows needed more aggregation units, not a
    different field.
14. **Checkpoint selection couples the two heads.** `ModelCheckpoint` monitors
    `val_total_loss`, which includes pinball, so a quantile-only change still selects a
    different epoch and therefore a different central field. Central-only A/Bs should
    monitor a central-only metric.

---

## 3. Targets and current standing

Scored on southern Africa with the retrained heads, identity recalibration, and the
spectral+long-scale field unless noted. "—" means not re-scored in the final configuration.

| Target | What it asks | Status | Evidence |
|---|---|---|---|
| **T1.1** pooled coverage ±0.01 | per horizon | ~ | 0.951–0.957, h=10 at 0.892 |
| **T1.2/1.3** class-conditional | \|cov−0.95\| ≤ 0.03, tail ≥ 0.92 | partial | bulk bins 0.930–0.971; high-change bins 0.82–0.91 |
| **T1.5** sharpness ≤ +25% | width vs original heads | **pass** | mean width halved (0.141 → 0.065) |
| **T2.1** block coverage 0.95±0.05 | 10/100/1000 km | partial | 1 km 0.914, 10 km 0.842, 100 km 0.826–0.899 |
| **T2.2** ecoregion coverage | 0.95±0.05 | **pass** | 0.889 / 1.000 |
| **T2.7** interval score | beat both baselines | used as the decision rule | identity 0.089 vs stratified 0.194 |
| **T3.1** member variogram | sill/range/nugget vs fitted | partial | variance 0.40 (clipping-limited) |
| **T3.2/3.3** vs independent null | ≥30% lower / beat null | **fail / pass** | variogram score uninformative once the far field is near-degenerate |
| **T3.4** spectrum 10–1000 km | within 1.5× | **near** | 3–10 px 0.73×, 1–3 px 0.92×, >50 px 2.44× |
| **T4.1** between-horizon corr | ±0.10 | **pass** | 0.724/0.939/0.916 vs 0.737/0.943/0.922 |
| **T4.2** monotone spread | ≥99% | **fail** | 68.5% |
| **T5.1/5.2/5.3** hard gates | median≡central, tails, mask | **pass** | 99.76–99.87%, ~100%, 0 mismatches |
| **T6** change-sign realism | ratios in band | **mostly pass** | P(Δ<−0.05) 0.0049 vs 0.0030 observed; P(Δ<−0.15) 0.000015 vs 0.000108 |
| **T7.2** member diversity | < 0.98 | **pass** | 0.26–0.46 |
| **T7.3** spread-skill | 1.0±0.25 | **pass** | 1.038 |
| **T8.1–8.3** change clustering | ratio ≤ 2, remote ≈ 0 | **mostly pass** | remote band exactly 0.00000; near field 2.0–2.7× |

### After the central-field change (k=5, whole region, both configurations scored identically)

Both columns are fold-stitched over the same 4.36M pixels and pushed through the same
Phase 1→4 loop at M=100, with the recalibration, spectrum and AR(1) coupling re-derived
from each model's own residuals.

| | baseline | central-field winner |
|---|---|---|
| **scorecard rows passed** | 73/137 (0.533) | **91/128 (0.711)** |
| **T1.2** class-conditional coverage | 6/21 | **13/16** |
| **T1.3** high-change tail | 0/4 | **3/3** |
| **T2.1** block coverage 1/10/100 km | 11/12 | **12/12** |
| **T2.5** rank histogram | 2/4 | **4/4** |
| **T4.2** monotone spread | fail | **0.9973** |
| **T6.2** `P(Δ<−0.05)` realism | 2/4 | **4/4** |
| **T8.2** remote band | 0/1 | **1/1** |
| **T8.3** near/remote ratio | 0/1 | **1/1 (∞)** |
| **T8.4** lower-tail clustering | 2/5 | **4/5** |
| T1.1 pooled coverage | 2/4 | 0/4 (0.963–0.982, uniformly wide) |
| T7.3 spread-skill | fail | 1.275 (just over the 1.25 gate) |

T2.1 at 12/12 is the motivating problem of the project — aggregate coverage at every block
scale. T8.3 at ∞ means remote stable country receives exactly zero invented change.

The recalibration decision is **identity, and this time measured** (held-out interval score
identity 0.09667 vs global 0.09670 vs stratified 0.09790): the new heads need no post-hoc
rescaling. Note the identity/global margin is 0.03%, which is why T1.1 is the obvious next
target rather than a deep problem.

Headline before/after on the two defects that motivated the earlier head-only change:

| | original heads | retrained heads |
|---|---|---|
| upper-head width decay near→far | 1.74× | **51.8×** |
| P(Δ>0.05) beyond 100 px (observed 0.0000) | 0.0297 | **0.00000** |
| P(Δ<−0.15) at h=20 (observed 0.000108) | 0.00062 | **0.000015** |
| aggregate coverage at 100 km | 0.703 | 0.826–0.899 |
| pooled coverage | 0.990 | 0.952 |
| mean interval width | 0.141 | 0.065 |

---

## 4. Where the remaining leverage is

**Done since this list was written.** Items 1, 2 and 3 below are addressed — see
`docs/central_field_baseline.md`. The central field now beats persistence at every horizon;
the quantile heads are anchored to it; h=10 is explained (its interval was *narrower* than
h=5's while covering 1.5× the error) and fixed structurally.

**In the model (highest value).**
1. *Training budget.* 150 epochs × 13 steps × 8 chips ≈ one pass over the globe's valid
   pixels. The central field is under-trained, and run-to-run variance between identically
   configured folds is comparable to the effects being chased (h=5 RMSE 0.01347 vs 0.01478
   for the same architecture). Longer training is the cheapest untested lever.
2. *T1.1 is the last marginal-calibration gap* (0.963/0.974/0.979/0.982 against 0.95±0.01).
   The heads are 1.5–3 points too wide and identity recalibration won by a 0.03% margin on
   held-out interval score (identity 0.09667, global 0.09670). A mild global rescale should
   close T1.1 at almost no interval-score cost — worth re-running the decision with a
   coverage-aware tie-break.
3. *Near-field over-prediction* (T8 at 0–3 px) — the opposite error from the one we fixed,
   and now the largest T8 residual.
4. *Receptive field.* Supplying long-range context as a raster worked; widening the trunk
   itself (dilated convolutions, or downsampled branches) is the general version and has not
   been tried.

**In the uncertainty layer.**
5. ~~*T4.2 monotone spread* (68.5%)~~ — **done.** The constraint moved to the head level
   (`--monotone_quantile_width`); T4.2 is now 0.9973 at M=100. The earlier 0.976 at M=20 was
   Monte-Carlo noise in the sample sd, not a violated constraint.
6. ~~*Block coverage at 10–100 km*~~ — **T2.1 now passes 12/12** at 1/10/100 km.
7. *`long_weight = 0.40` is calibrated, not derived — and it survives a sweep.* Swept
   0.15–0.70: it moves **only** the spread-skill ratio (0.821 → 1.524, monotone), so it is
   identifiable from T7.3 alone. At k=5 the winner scores T7.3 = 1.275, just over the 1.25
   gate, and dropping to 0.25 fixes that (1.065) — but at the cost of pushing T2.3's
   *already near-nominal* rows from 0.944/0.889 down to 0.778/0.667, while the rows
   saturated at 1.000 do not move. Keep 0.40; T7.3 is a recorded near-miss, not a knob
   waiting to be turned. Re-derive before any global run.
8. *M = 50 limits tail-sensitive per-pixel products.* Seeds are recorded, so extending to
   M ≥ 200 is cheap and would clear several T1 rows that are Monte-Carlo-limited rather than
   wrong.
9. ~~*T3.2's null comparison* needs spread-weighted sampling~~ — **done, and the diagnosis
   was corrected in doing it.** Pairs were being drawn out to 500 px against a fitted
   practical range of 50.5 px, and beyond the correlation range the correlated ensemble and
   its independent null produce the same variogram term, so those pairs cancelled in the
   ratio while diluting it. Capping separation at half the fitted range took the improvement
   from 1.0% to 10.5%; spread-weighted sampling took it to 16.5%. **The distance cap was the
   dominant effect, not the far-field degeneracy.** T3.2 is now informative and still fails
   (16.5% against 30%); the residual is most likely the nugget fraction (0.345 — a third of
   member variance uncorrelated at zero lag), which is a Phase 2 field knob, and T3.1 agrees
   the field structure is off. The uniform-sampled score is retained as `T3.2b` so the
   change stays auditable.

**Process.** Re-run the whole regional loop after any model change — it is ~20 minutes and it
has repeatedly caught measurement errors that looked like findings.
