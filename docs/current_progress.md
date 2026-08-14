# Current Progress — Spatiotemporal HM Uncertainty

Status as of 2026-08-14, branch `ensemble`. Companion to
`docs/ensemble_uncertainty_plan.md`, which holds the design and the full target definitions.
This document records what exists, what was learned, how we score against each target, and
where the remaining leverage is.

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

Headline before/after on the two defects that motivated the model change:

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

**In the model (highest value).**
1. *The central field itself.* Every uncertainty product is built on it and it has never been
   revisited in this work. Its error is what the intervals must cover, so reducing it
   improves every target at once.
2. *Quantile heads beyond the current fix.* They now see past-change context, but only that.
   Candidates: give them the same context at more scales or as a learned embedding; let them
   see the central head's own output (the predicted change is a strong conditioner); train
   them on the global split rather than one region.
3. *h=10 is the weak horizon* (pooled 0.892 against ~0.95 elsewhere) and is unexplained.
4. *Near-field over-prediction* (T8 at 0–3 px, 2.0–2.7×) — the opposite error from the one
   we fixed, and now the largest T8 residual.

**In the uncertainty layer.**
5. *T4.2 monotone spread* (68.5%) — the guard now constrains ŝ·w, but the heads' own widths
   are not monotone in horizon. Likely wants a constraint at the head level.
6. *Block coverage at 10–100 km* (0.83–0.90) — falls between the fine structure and the
   long-scale component; probably wants an intermediate term, fitted the same way the long
   one was.
7. *`long_weight = 0.40` is calibrated, not derived.* Re-tune per region and certainly before
   any global run; it encodes how much systematic regional bias the model carries.
8. *M = 50 limits tail-sensitive per-pixel products.* Seeds are recorded, so extending to
   M ≥ 200 is cheap and would clear several T1 rows that are Monte-Carlo-limited rather than
   wrong.
9. *T3.2's null comparison* needs spread-weighted sampling to be meaningful now that the far
   field is near-degenerate.

**Process.** Re-run the whole regional loop after any model change — it is ~20 minutes and it
has repeatedly caught measurement errors that looked like findings.
