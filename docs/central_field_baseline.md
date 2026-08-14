# The Central Field — Baseline Measurement

Branch `ensemble`, southern Africa, 2026-08-14. Companion to `docs/current_progress.md`.

The uncertainty layer has been audited repeatedly; the central forecast it is built around
never has. This document measures where that forecast's error actually lives, using the
same stratification discipline Phase 1d applied to the quantile heads: not "how big is the
error" but "which pixels own it", cut only by covariates available at prediction time.

Produced by `scripts/diagnose_central_field.py` against the k=5 fold-stitched hindcast
rasters (`data/ensemble/region/southern_africa/stitched/`), which are genuinely
out-of-sample everywhere by fold rotation. 10 (window × horizon) pairs, 1.09M valid pixels
each.

---

## 1. The headline: the central forecast loses to persistence at short horizons

Persistence means Δ̂ = 0 — "nothing will change" — which for a variable whose median
20-year change is 0.0001 is a genuinely hard baseline, not a strawman.

| horizon | model RMSE | persistence RMSE | skill | corr(Δ̂, Δ) | slope of Δ on Δ̂ |
|---|---|---|---|---|---|
| +5yr | 0.01478 | **0.00990** | **−1.227** | 0.136 | 0.104 |
| +10yr | 0.01746 | **0.01565** | **−0.244** | 0.232 | 0.282 |
| +15yr | 0.02047 | 0.02054 | +0.006 | 0.297 | 0.445 |
| +20yr | 0.02416 | 0.02472 | +0.045 | 0.300 | 0.500 |

At +5yr the model's mean squared error is **2.2× worse than predicting no change at all**.
Skill only becomes positive at +15yr, and even at +20yr it is +4.5%.

The slope column is the second half of the story. Regressing observed change on predicted
change gives 0.10 at h=5: when the model predicts a change of +0.01, the expected observed
change is +0.001. The model's change signal needs to be shrunk roughly tenfold at h=5 to be
optimal, and roughly halved even at h=20.

## 2. Where the error comes from: change invented on pixels that did not change

53–70% of valid pixels have |observed Δ| < 0.001 across the horizons. On exactly those
pixels — where the correct answer is "emit HM_t0 unchanged" — the model still produces:

| window | horizon | share of valid px | RMSE | sd of predicted change |
|---|---|---|---|---|
| 1990-95-00 | 5 | 0.670 | 0.00749 | 0.00747 |
| 1990-95-00 | 10 | 0.580 | 0.00636 | 0.00634 |
| 1990-95-00 | 20 | 0.531 | 0.00810 | 0.00671 |
| 2005-10-15 | 5 | 0.643 | 0.00836 | 0.00829 |

`sd(predicted change) ≈ RMSE` on these pixels: the error there is *entirely* spurious
predicted change, not a bias or a level offset. Confirmed from the other direction —
`R²(residual ~ HM_t0)` is 0.00001–0.0074, so the error is not a level-dependent
reconstruction bias; and `R²(residual ~ Δ̂)` is **0.50–0.61 at h=5**, so half the residual
variance at the shortest horizon is the model's own predicted change coming back as error.

At h=5, static pixels are ~67% of the map and carry ~0.0075² of MSE each, which is about a
third of the model's entire excess over persistence. The rest sits near past change, where
the model over-shoots (see §4).

## 3. The change amplitude barely responds to lead time

| horizon | sd(predicted Δ) | sd(observed Δ) | variance ratio |
|---|---|---|---|
| +5yr | 0.0113–0.0130 | 0.0091–0.0104 | **1.18–1.96** |
| +10yr | 0.0116–0.0132 | 0.0151–0.0157 | 0.54–0.77 |
| +15yr | 0.0126–0.0141 | 0.0198–0.0201 | 0.39–0.50 |
| +20yr | 0.0144 | 0.0240 | 0.36 |

The truth's change spread grows 2.3× from +5yr to +20yr. The model's grows 1.3×. The four
central heads are independent modules on a shared trunk with nothing tying their output
scale to their lead time, and they have converged on nearly the same amplitude — which is
simultaneously **too much change at +5yr** (variance ratio up to 1.96) and **too little at
+20yr** (0.36).

This is why h=5 is the worst horizon: it receives a 20-year-sized change prediction.

## 4. Error by distance to past change

RMSE, pixel-weighted across windows:

| distance to past change | +5yr | +10yr | +15yr | +20yr |
|---|---|---|---|---|
| 0–1 px | 0.02714 | 0.03108 | 0.03606 | 0.04191 |
| 1–3 px | 0.01631 | 0.02032 | 0.02451 | 0.02973 |
| 3–10 px | 0.01055 | 0.01372 | 0.01765 | 0.02286 |
| 10–30 px | 0.00606 | 0.00828 | 0.01100 | 0.01500 |
| 30–100 px | 0.00424 | 0.00517 | 0.00604 | 0.00813 |
| >100 px | 0.00661 | 0.00576 | 0.00541 | 0.00563 |

Two things to read here. First, error concentrates near past change exactly as the T8 table
predicts — the same geography the quantile heads were blind to. Second, and less expected,
the **>100 px band does not continue the decline**: RMSE stays at 0.0054–0.0066 in country
where the observed probability of any change above 0.01 was measured at exactly zero over
493,240 pixels. Every bit of that is invented. Beyond 30 px the model is losing to
persistence by a factor of 3 (skill −3.24 in the 30–100 px band at h=5).

## 5. Error by predicted change magnitude

Bias = mean(observed − central), h=5, window 1990-95-00:

| Δ̂ bin | share | mean Δ̂ | mean observed Δ | bias |
|---|---|---|---|---|
| ≤ −0.01 | 5.7% | −0.0182 | +0.0019 | **+0.0201** |
| (−0.01, 0.001] | 48.0% | −0.0024 | +0.0006 | +0.0030 |
| (0.001, 0.01] | 34.7% | +0.0040 | +0.0007 | −0.0033 |
| (0.01, 0.05] | 11.0% | +0.0193 | +0.0032 | −0.0160 |
| (0.05, 0.15] | 0.58% | +0.0729 | +0.0115 | **−0.0614** |
| > 0.15 | 0.015% | +0.1763 | +0.0161 | **−0.1602** |

Every bin's bias points back toward zero, in proportion to the prediction — the graphical
form of the 0.10 regression slope. The model predicts a 0.176 increase and gets 0.016. It
also predicts *decreases* of −0.018 on 5.7% of pixels where the truth increases slightly;
HM decreases are real but rare, and the model is manufacturing them.

At +20yr the same table is far better behaved (Δ̂ > 0.15 bin: predicted 0.176, observed
0.077), which again points at lead-time-blind amplitude rather than a broken change signal.

---

## 6. The h=10 coverage anomaly is a non-monotone interval width

Separately measured on the retrained quantile heads
(`newheads_residuals/`, the configuration whose scorecard reports pooled coverage 0.892 at
h=10 against ~0.95 elsewhere).

Median interval half-widths, and the residual spread they are supposed to cover:

| horizon | median w_up | median w_lo | sd(residual) | coverage |
|---|---|---|---|---|
| +5yr | 0.0068–0.0076 | 0.0045–0.0057 | 0.0103–0.0118 | 0.946–0.963 |
| **+10yr** | **0.0050–0.0061** | **0.0040–0.0044** | **0.0152–0.0164** | **0.880–0.909** |
| +15yr | 0.0122–0.0138 | 0.0092–0.0098 | 0.0196 | 0.956–0.957 |
| +20yr | 0.0277 | 0.0124 | 0.0232 | 0.951 |

**The h=10 interval is narrower than the h=5 interval while the error it must cover is 1.5×
larger.** The dip appears in all three input windows that contribute to h=10, so it is not
a window-composition artifact of pooling. The misses are near-symmetric (4–6% below, 4–8%
above), so it is not a tail-shape problem either — the interval is simply too small.

The cause is structural: the four horizons' quantile heads are independent modules trained
by a per-horizon pinball loss with nothing coupling them, so nothing prevents h=10 from
landing below h=5. It is the same defect that T4.2 measures as 68.5% monotone spread; the
h=10 coverage failure is that number seen from the other side.

Corroborating detail: on *unchanged* pixels the h=10 coverage falls to 0.931 while h=5 and
h=15 hold 0.990 and 0.993 — the h=10 head collapses its interval hardest exactly where
there is nothing to predict.

---

## 7. What this implies

| finding | implied change |
|---|---|
| Spurious change on 53–70% of pixels; persistence beats the model at h=5/h=10 | **Parameterise the central head as a change on top of HM_t0** so "nothing happens" is free instead of something the trunk must reconstruct |
| Amplitude nearly constant across horizons | Follows from the same change — each head then predicts Δ directly against its own horizon's target scale |
| Invented change beyond 100 px from past change | **Give the central heads the past-change context** the quantile heads already get; the trunk's ~10 px radius cannot see it |
| h=10 interval narrower than h=5 | **Make interval width cumulative in horizon** so spread cannot shrink with lead time; this also makes `lower ≤ central ≤ upper` structural |

All four are implemented behind separate flags (`--central_residual`, `--central_context`,
`--monotone_quantile_width`) so they can be attributed individually.

---

## 8. Result — `--central_residual`, measured

Round 1 held out fold 1, trained on the other four, predicted southern Africa, and scored
the held-out fold's pixels only. Control and treatment differ in exactly one flag and are
scored on identical pixels (931,776 at h=5), so the deltas are attributable.

| horizon | RMSE control | RMSE residual | skill control | skill residual | slope control | slope residual |
|---|---|---|---|---|---|---|
| +5yr | 0.01347 | **0.00943** (−30%) | −0.875 | **+0.081** | 0.080 | **1.183** |
| +10yr | 0.01682 | **0.01447** (−14%) | −0.139 | **+0.157** | 0.314 | **1.009** |
| +15yr | 0.02028 | **0.01876** (−7%) | +0.041 | **+0.179** | 0.521 | 0.721 |
| +20yr | 0.02334 | 0.02366 (+1%) | +0.088 | +0.063 | 0.563 | 0.523 |

MAE falls 58% at h=5 and 39% at h=10. Skill against persistence is **positive at every
horizon for the first time**, and the change signal is correctly scaled at short lead times
where it was previously ten times too large.

The mechanism is the predicted one, not a coincidence:

| | reconstruction RMSE on unchanged pixels (h=5) | RMSE at 30–100 px from past change (h=5) |
|---|---|---|
| control | 0.00725 | 0.00529 |
| residual | **0.00121** (−83%) | **0.00155** (−71%) |

What disappeared is the invented change, in every distance band, most of all in the far
field. The residual parameterisation removes the error the baseline attributed to it.

**One regression.** Coverage at h=5 falls from 0.990 to 0.859. The quantile heads are
unanchored and were trained against a central field that moved under them for the whole
run, so their widths belong to a worse forecast than the one finally published. This is the
case for `--monotone_quantile_width`, which anchors the interval to the (detached) central
forecast rather than predicting bounds independently.

Also worth recording: the control's own h=5 RMSE (0.01347) beats the k=5 stitched baseline's
(0.01478) on the same architecture and budget. That gap is run-to-run variance between fold
models, and it sets the noise floor for reading any of these deltas — the residual effect at
h=5 is roughly three times it, the h=20 effect is well inside it.

---

## 9. Rounds 2 and 3 — attributing the rest

Round 2 added one flag each on top of the residual head; round 3 combined all three and
trained two folds instead of one. All configurations are scored on identical pixels
(931,776 at h=5 for the fold-1 comparison).

| | RMSE vs control, h=5/10/15/20 | skill h=20 | corr h=20 | raw coverage across horizons |
|---|---|---|---|---|
| `--central_residual` | 0.700 / 0.860 / 0.925 / **1.014** | +0.063 | 0.381 | 0.859 – 0.965 |
| ` + --central_context` | 0.697 / 0.855 / 0.904 / 0.947 | **+0.183** | **0.414** | 0.997 (uniformly wide) |
| ` + --monotone_quantile_width` | 0.704 / 0.858 / 0.944 / 0.951 | +0.176 | 0.373 | **0.965 – 0.971** |
| all three (`e4_all`) | 0.700 / 0.878 / 0.936 / **0.933** | **+0.207** | 0.404 | 0.968 – 0.980 |

Three separable effects:

1. **The residual parameterisation is the large one** — 30% RMSE at h=5, reproduced in every
   configuration that carries it. It leaves h=20 unimproved or fractionally worse.
2. **The past-change context is what fixes h=20**, in both configurations that carry it
   (1.014 → 0.947 and → 0.933), tripling h=20 skill. The trunk really cannot see that far.
3. **Monotone width does not affect the central field** and flattens raw coverage across
   horizons, which is what it was for.

### The k=5 result — like for like

`e5_all_k5` runs the combined configuration through all five folds, so it is stitched over
the whole region exactly as the original baseline was: 4,358,388 pixels at h=5 against the
baseline's 4,366,544. No pixel-set caveat applies to this comparison.

| horizon | baseline RMSE | k=5 winner | baseline MAE | k=5 MAE | baseline skill | k=5 skill | baseline slope | k=5 slope |
|---|---|---|---|---|---|---|---|---|
| +5yr | 0.01478 | **0.00947** (−36%) | 0.00759 | **0.00288** (−62%) | −1.227 | **+0.085** | 0.104 | **1.156** |
| +10yr | 0.01746 | **0.01444** (−17%) | 0.00859 | **0.00505** (−41%) | −0.244 | **+0.148** | 0.282 | **0.928** |
| +15yr | 0.02047 | **0.01855** (−9%) | 0.00988 | **0.00728** (−26%) | +0.006 | **+0.184** | 0.445 | **0.784** |
| +20yr | 0.02416 | **0.02222** (−8%) | 0.01255 | **0.00939** (−25%) | +0.045 | **+0.191** | 0.500 | **0.840** |

Correlation with observed change rises from 0.136 / 0.232 / 0.297 / 0.300 to
0.266 / 0.340 / 0.385 / 0.380.

The two-fold screening figures below are retained because they are what the flags were
attributed on; they agree with the k=5 numbers to within a percent at every horizon.

Scored on both held-out folds (1.54M px at h=5), against the original k=5 baseline:

| horizon | baseline RMSE | e4 RMSE | baseline skill | e4 skill | baseline slope | e4 slope |
|---|---|---|---|---|---|---|
| +5yr | 0.01478 | **0.00938** (−37%) | −1.227 | **+0.086** | 0.104 | **0.831** |
| +10yr | 0.01746 | **0.01458** (−17%) | −0.244 | **+0.137** | 0.282 | **0.743** |
| +15yr | 0.02047 | **0.01881** (−8%) | +0.006 | **+0.170** | 0.445 | **0.728** |
| +20yr | 0.02416 | **0.02188** (−9%) | +0.045 | **+0.198** | 0.500 | **0.825** |

Skill is positive at every horizon and now *increases* with lead time, which is the sensible
ordering — the baseline's ran the other way because its error was dominated by a
horizon-independent noise floor rather than by forecast difficulty.

### T1–T8 consequences

Scored through the same regional loop for every configuration (M=20 members, identity
recalibration — see the note below).

| target | control | +residual | +residual +monotone |
|---|---|---|---|
| **T1.1** pooled coverage h=5/10/15/20 | 0.975 / 0.965 / 0.967 / 0.957 (1/4 pass) | 0.903 / 0.979 / 0.975 / 0.967 (0/4) | **0.940 / 0.950 / 0.942 / 0.953 (4/4)** |
| **T4.2** monotone spread | 0.529 | 0.106 | **0.976** |
| **T7.3** spread-skill | 1.233 | 1.178 | 0.920 |

h=10 lands at 0.950 — the anomaly this document opened with is gone, and T1.1 passes at
every horizon for the first time.

The cost lands on **T2.1**: block coverage at 10–100 km falls to 0.71–0.86 on several rows,
because the pixel intervals are now genuinely tighter. The plan's knob table is unambiguous
about this case — raise the long-range weight in the field, do not re-widen the marginals.
It is a field-structure knob, not a model one, and `long_weight` was already flagged in
`current_progress.md` as calibrated rather than derived.

**T4.2's residual 2.4%** was Monte-Carlo noise, now confirmed: it is measured from the
*sample* standard deviation of the members, which at M=20 carries ~16% relative noise, so
adjacent horizons of similar true spread invert by chance. Re-scored at M=100 the same
ensemble gives **0.9997** — a pass, against 0.529 for the control and the 0.685 previously
documented. The head widths are monotone by construction anyway (asserted in
`tests/test_central_head_parameterisation.py` and verified through the real inference path
at 100% of pixels).

### The full scorecard, control vs the combined configuration at M=100

| target | control | e4 (all three flags) |
|---|---|---|
| **T1.2** class-conditional coverage | 9/18 | **14/15** |
| **T1.3** high-change tail | 0/1 | **2/2** |
| **T4.2** monotone spread | 0.529 ✗ | **0.9997 ✓** |
| **T3.3** energy score vs independent null | 0.769 vs null 0.754 ✗ | **0.738 vs null 0.792 ✓** |
| **T6.2 / T6.3** change-sign realism | 0/4, 3/4 | **4/4, 4/4** |
| **T8.2** remote band `P(Δ>0.05)` | NaN ✗ | **0.00000 ✓** |
| **T8.4** lower-tail clustering by band | 3/5 | **4/5** |
| T1.1 pooled coverage | 1/4 | 1/4 |
| T2.1 block coverage | 9/12 | 8/12 |
| **overall** | 69/131 = 0.527 | **88/126 = 0.698** |

T1.2 is the target the whole revision was written around, and it goes from half-failing to
14 of 15 cells. T3.3 beating the independent-pixel null is the first time that has happened:
that null has identical marginals and no spatial structure, so the margin is attributable to
the copula's correlation and nothing else.

### What did *not* work, and why it is worth recording

T1.1 stays at 1/4 (0.953 / 0.966 / 0.974 / 0.979) — coverage drifts wide with lead time. That
is a recalibration matter, and none of these runs could decide it (see the single-fold note
below); every one of them ran identity.

The remaining T2 failures are **over**-coverage, not under: T2.2 sits at exactly 1.000 across
all four horizons and every failing T2.1 row is a 100 km block at exactly 1.000. The plan's
knob table prescribes raising the long-range field weight when T2 under-covers, so the
opposite move was tested — `long_weight` 0.40 → 0.55 → 0.70:

| long_weight | overall | T2.1 | T2.2 | T2.3 | T7.3 spread-skill | T1.1 values |
|---|---|---|---|---|---|---|
| 0.15 | 87/126 | 9/12 | 0/4 | 0/8 | 0.821 | 0.953 / 0.966 / 0.974 / 0.979 |
| 0.25 | 88/126 | 9/12 | 0/4 | 1/8 | **0.936** | 0.953 / 0.966 / 0.974 / 0.979 |
| 0.40 | 88/126 | 8/12 | 0/4 | 1/8 | 1.113 | 0.953 / 0.966 / 0.974 / 0.979 |
| 0.55 | 87/126 | 8/12 | 0/4 | 1/8 | 1.304 | 0.953 / 0.966 / 0.975 / 0.979 |
| 0.70 | 87/126 | 8/12 | 0/4 | 1/8 | 1.524 | 0.952 / 0.966 / 0.975 / 0.979 |

Over a 4.7× range of the knob the overall score moves by one row and T1.1 is identical to
three decimals. **The only thing `long_weight` actually controls in this scorecard is the
spread-skill ratio** (0.821 → 1.524, monotone). T2.2 stays 0/4 throughout, pinned at 1.000.

That is the finding: the T2 rows are **saturated and under-powered rather than mis-tuned**. A
2-fold subset of the region leaves ~14 ecoregions and few 100 km blocks, and the plan's own
power analysis says a ±0.05 target is undecidable at n=14 (Wilson half-width ±0.114). No
setting of a field-structure knob can fix a metric that cannot resolve the difference it is
being asked about; deciding these rows needs the full fold set.

On the one thing it does control, `long_weight = 0.25` looked marginally better than the
incumbent 0.40 in the two-fold sweep (spread-skill 0.936 vs 1.113, T2.1 9/12 vs 8/12).

**The k=5 data reverses that.** Re-run on the full fold set:

| | T7.3 spread-skill | T2.3 area-above-threshold, per row | overall |
|---|---|---|---|
| 0.40 | 1.275 (just over the gate) | 1.0 · 1.0 · 1.0 · 1.0 · **0.944** · 1.0 · **0.889** · **0.944** | 91/128 |
| 0.25 | **1.065** (passes) | 1.0 · 1.0 · 1.0 · 1.0 · **0.778** · 1.0 · **0.667** · 0.889 | 90/129 |

Lowering the weight buys spread-skill by pulling the *already near-nominal* T2.3 rows below
target, while the rows saturated at 1.000 do not move at all. That is a real cost rather
than the wash the under-powered two-fold sweep implied. **`long_weight` stays at 0.40**, and
T7.3 = 1.275 is recorded as a near-miss with a known, undesirable remedy.

The general lesson is the one already learned about coverage: a sweep judged on a total pass
count hides which rows moved. Reading the per-row values is what made the trade-off visible.

### Measurement notes worth carrying forward

- The `newheads` central rasters and the fold-stitched rasters are **not comparable
  as-is**: the former is a single production-checkpoint model scored on the production
  val/test/calib splits (331k px/window), the latter is fold-stitched over the whole region
  (1.09M px/window). Restricting the baseline to the same splits changes its h=5 RMSE from
  0.0148 to 0.0157 — the residual gap to `newheads` (0.0111) is a genuine model difference,
  most plausibly run-to-run variance, since both were trained to a similar step budget.
- That budget is small: 150 epochs × 13 steps × 8 chips ≈ 15,600 chip presentations of
  128², roughly one pass over the globe's valid pixels. Treat the central field as
  under-trained until shown otherwise, and expect run-to-run variance comparable to the
  effect sizes being chased.
- **Checkpoint selection couples the two heads.** `ModelCheckpoint` monitors
  `val_total_loss`, which includes pinball, so a change that touches only the quantile heads
  still selects a different epoch and therefore a different central field. That is why
  `e3_res_mono`'s central RMSE differs from `e1_residual`'s despite an identical central
  path. Any future central-only A/B should monitor a central-only metric, or the comparison
  carries a confound.
- **Single-fold screening cannot make a recalibration decision.** Leave-one-fold-out has
  nothing to hold out, so all three interval scores come back non-finite. `select_recalibration`
  used to report `decision: identity` from that as though it were measured; it now says
  explicitly that it is a fallback. Every screening run above therefore used identity, which
  is at least uniform across configurations, so the comparisons hold — but the *decision*
  needs the full k=5 residual set.
- The regional loop for one configuration (`scripts/run_region_loop.sh`) is ~15 min end to
  end at M=20; a fold's train-plus-regional-predict is ~25 min. Global prediction, not
  training, was what made the original fold run take 121 min per fold.
