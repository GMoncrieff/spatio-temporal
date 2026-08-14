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

---

## 10. T3.2's scoring fix — and a corrected diagnosis

`current_progress.md` recorded T3.2's null comparison as needing **spread-weighted
sampling**, "now that the far field is near-degenerate". That was half right, and the
smaller half.

Two things were wrong with the scoring, both fixed:

1. **Pairs were drawn out to 500 px against a fitted practical range of 50.5 px.** Beyond
   the correlation range a correlated ensemble and an independent one produce the same
   variogram term, so those pairs contribute an identical quantity to numerator and
   denominator — they cancel in the ratio while diluting it. The cap is now half the member
   variogram's own fitted range (`--score_max_dist_px` to override).
2. **Points were drawn uniformly**, so the sample included pixels whose ensemble has no
   spread to structure. Sampling is now weighted by the marginal scale — which the ensemble
   and its null share *by construction*, so the weighting changes where both are measured
   without favouring either.

Measured on the k=5 ensemble:

| sampling | improvement vs null | pairs carrying spread |
|---|---|---|
| uniform, 500 px (the old scoring) | **1.0%** | — |
| uniform, 25 px | **10.5%** | 90.6% |
| spread-weighted, 25 px | **16.5%** | 100.0% |

**The distance cap did most of the work**, not the spread weighting: at 25 px with clustered
patches, uniform sampling already draws 90.6% live pairs. The far-field degeneracy was a
secondary effect, and the plan's stated diagnosis should be read as corrected.

T3.2 is now an informative measurement and it still **fails**, at 16.5% against a 30%
target. The obvious next move is to tune the field's short-scale content down until it
passes. That move is wrong, and the reason is measurable.

### T3.2 and T3.4 are in direct conflict, and T3.4 is the one grounded in data

An ensemble calibrated to the residual reproduces the residual's correlation structure, so
it can differ from an independent-pixel ensemble only in the variance still *correlated* at
the separation being scored. `scripts/t32_structure_budget.py` reads that off the
standardized residual's own empirical variogram as `gamma(d)/sill`:

| separation | residual variance already decorrelated | structure budget T3.2 can reward |
|---|---|---|
| 5 px | 71.1% | 28.9% |
| 10 px | 81.5% | 18.5% |
| **25 px** (where T3.2 is scored) | **86.0%** | **14.0%** |
| 50 px | 90.1% | 9.9% |

Consistent across all ten (window, horizon) pairs. **A 30% reduction in variogram score
against an independent-pixel null asks the ensemble to be better structured than the data it
is calibrated to.** The measured 16.5% is a reasonable showing against a 14% budget, not a
shortfall.

The confirming check: refitting the spectrum with the sub-4 px basis ranges deleted does not
remove the short-scale content, it relocates it — the explicit nugget goes 0.103 → 0.262,
because the power is in the data. The observed residual carries 12% of its spectral power at
1–3 px and 37–41% at 3–10 px. Passing T3.2 would mean generating a field markedly smoother
than the residual, which is precisely the failure T3.4 (spectrum within 1.5×) exists to
catch. **Do not tune the field to pass T3.2**; the target's 30% threshold was set a priori
and should be re-scoped — scored at short lags where structure exists, or set from the
residual's own correlation budget.

**A negative result recorded rather than buried.** The first attempt at this question built
a synthetic ensemble from the fitted spectrum and scored T3.2 on it to estimate an
achievable ceiling. It returned 2.9%, then 1.6% after the geometry was corrected to share a
common central forecast between truth, members and null — both far *below* the 16.5% that
production actually scores. A ceiling the real system exceeds is not a ceiling, so the
synthetic model was not a valid stand-in and its numbers were discarded rather than
reported. The variogram of the real residual answers the same question with no stand-in at
all, which is why the script that survives measures instead of simulating.

T3.1 separately says the field structure is imperfect (practical range 0.357 relative error,
nugget 0.119 absolute, both over target) — that remains worth attention on its own terms,
but it is not what T3.2 is measuring.

Supporting changes: `variogram_score` now returns a weighted **mean** rather than a sum, so
the value no longer depends on how many pairs survived the distance filter (ratios are
unaffected); `informative_pair_fraction` reports the diluting share directly; and the
uniform-sampled score is retained on the scorecard as `T3.2b` so the weighting's effect
stays auditable rather than silently replacing a published number.

Two implementation traps found while testing this, both fixed:
- Thresholding "has spread" against the *median* spread collapses in exactly the regime it
  is meant to detect, because the degenerate background is the majority and the median sits
  inside it. It uses the 99th percentile.
- `rng.choice(..., replace=False, p=w)` must return `n` distinct indices, so a patch that
  only clips the live region is forced to make up the difference from zero-weight pixels —
  silently reintroducing the degenerate points. Candidates are now filtered before the draw.

---

## 11. Parameter sweeps before complexity — three knobs, all negative

Before proposing any new machinery for the remaining failures (T1.1, T3.1, T3.2), every
existing parameter that could plausibly move them was swept. None does.

### Recalibration: identity is already the best of the three options

| variant | overall | T1.1 | T1.2 | T1.3 | T8.4 | T7.3 |
|---|---|---|---|---|---|---|
| **identity (chosen)** | **91/128 = 0.711** | 0/4 | **13/16** | **3/3** | **4/5** | 1.275 |
| global conformal | 83/127 = 0.654 | 0/4 | 13/15 | 2/2 | 2/5 | 2.129 |
| empirical width multiplier | 85/125 = 0.680 | **4/4** | 8/15 | 0/2 | 0/5 | **1.185** |

The third row is the interesting one. Coverage turns out to be remarkably *insensitive* to
interval width, because the central-field fix left most pixels with near-zero error, so
almost nothing sits near the interval boundary:

| horizon | ×1.0 | ×0.8 | ×0.6 | ×0.4 | ×0.2 | multiplier for 0.95 |
|---|---|---|---|---|---|---|
| +5yr | 0.9659 | 0.9533 | 0.9297 | 0.8802 | 0.7579 | 0.76 |
| +10yr | 0.9761 | 0.9661 | 0.9473 | 0.9061 | 0.7841 | 0.62 |
| +15yr | 0.9820 | 0.9714 | 0.9454 | 0.8888 | 0.7500 | 0.63 |
| +20yr | 0.9850 | 0.9732 | 0.9429 | 0.8742 | 0.6632 | 0.63 |

Cutting width 20% buys about one point of coverage. Applying the multiplier that *does*
reach 0.95 (24–38% narrowing) puts T1.1 at 4/4 — and costs class-conditional coverage
(13/16 → 8/15), the high-change tail (3/3 → **0/2**) and lower-tail clustering (4/5 →
**0/5**). That is exactly the trade T1.2 and T1.3 exist to guard. **T1.1's failure is the
price of those passing, and no scalar fixes it.**

Note also why the conformal global fit did *not* do this: its factors narrow the upper tail
(s_up 0.61–0.92) while widening the lower (s_lo 1.03–1.21), so the two-sided coverage barely
moves while realism degrades.

### Field structure: neither `long_weight` nor Matérn `nu` moves T3.1 or T3.2

`long_weight` over 0.15–0.70 (§ above) moves only the spread-skill ratio. Matérn smoothness
over 0.5–1.5:

| nu | T3.1 range error (≤0.25) | T3.1 nugget error (≤0.10) | T3.2 |
|---|---|---|---|
| **0.5** | **0.357** | 0.1185 | **16.5%** |
| 1.0 | 0.417 | 0.1019 | 15.9% |
| 1.5 | 0.447 | **0.1015** | 16.1% |

Raising `nu` trades a marginally better nugget for a clearly worse range, and T3.2 is flat
to within noise. Both T3.1 rows fail at every setting.

Worth noting about T3.1's range row specifically: the reference it compares against is a
fitted practical range whose variogram fits carry r² of 0.42–0.79, describing a field that
§10 measured as **71% decorrelated by 5 px**. A single "practical range" is not a
well-determined descriptor of a variogram that has already lost most of its structure inside
the first few pixels, so this row is weak evidence either way.

**Conclusion of the sweep.** Three independent knobs, eleven settings, no improvement on any
remaining failure. The residual gaps are structural, not parametric: T1.1 is a genuine
tension with T1.2/T1.3, and T3.2 is a target-threshold problem (§10). Keep identity
recalibration, `long_weight = 0.40`, `nu = 0.5`.

---

## 12. The marginal family — built, measured, not adopted

§11 established that no existing knob moves the remaining failures. The next question was
*which* model-level change, and the data answered it differently from the obvious guess.

### The guess was wrong: the widths are fine, the shape is not

The intervals look too wide, so the natural move is to retrain the width heads tighter.
The residual says otherwise. In units of the published 95% half-width:

| | P(\|e\|>0.25) | P(\|e\|>0.5) | P(\|e\|>1) | P(\|e\|>1.96) | kurtosis |
|---|---|---|---|---|---|
| Gaussian (what the marginal assumes) | 0.803 | 0.617 | 0.317 | 0.050 | 3 |
| observed h=5 | 0.342 | 0.194 | 0.088 | 0.034 | **20366** |
| observed h=20 | 0.492 | 0.254 | 0.082 | 0.015 | **928** |

At the 95% point the interval is close to nominal — that is T1.1's mild over-coverage. In
the *body* the residual is three times more concentrated than the Gaussian the two-piece
normal fills it with, and the members inherit exactly that. Narrowing the heads would fix
the body by breaking the tails, which is precisely what §11's `k*` experiment demonstrated.

### The fix, and what it achieved

`fit_residual_shape` (`src/ensemble/copula.py`) replaces the *shape* only: the normal score
is remapped through the residual's own standardized quantile function, normalized so
u = 0.025/0.5/0.975 still land exactly on lower/central/upper. Strictly monotone, so the
copula's rank structure and every spatial property are untouched; a genuinely Gaussian
residual returns the identity. Enabled by `--marginal_shape` on `generate_ensemble.py`.

| | overall | T6.1 ratios (target 0.5–2.0) | T6.5 | T5.2 |
|---|---|---|---|---|
| two-piece normal | **91/128 = 0.711** | 3.84 / 5.65 / 6.75 / 6.65 | 1/8 | **4/4** |
| empirical shape | 84/126 = 0.667 | **1.74 / 2.88 / 4.36 / 4.55** | 2/8 | 0/4 |

It halves the defect it was built for — h=5's T6.1 ratio passes for the first time — and
costs every percentile-estimated row.

### Why it is *not* the default

The percentile rows degrade because a spiky marginal is harder to estimate percentiles from
at fixed M: the mapping is steep near the bound and few members land there. Measured
directly, as |ensemble percentile − published bound| in units of the interval width:

| | M | lower | upper |
|---|---|---|---|
| two-piece | 100 | 0.0331 | 0.0518 |
| shaped | 100 | 0.0461 | 0.0974 |
| shaped | 400 | **0.0233** | 0.0665 |

At M=400 the shaped marginal reproduces the lower bound *better* than the two-piece does at
M=100. But T5.2's tolerance is MC-scaled as 1/√M under a Gaussian assumption the shaped
family violates, so its pass fraction went *down* at M=400 (0.847 → 0.773) even as the
absolute error halved. Scoring this family against that tolerance is not a fair test — and
"the metric is unfair" is not sufficient grounds to adopt a change that loses 91/128 to
84/126 as actually scored. So the metric was fixed first — see below.

### The T5 tolerances are now family-aware, and the incumbent is unaffected

The standard error of a sample p-quantile is `sqrt(p(1-p)/M) / f(x_p)`. In normal-score
units that is `sqrt(p(1-p)/M) / phi(z_p)`, and converting to value units costs `dx/dz`. For
the two-piece normal `x = loc + sigma*z`, so that factor is exactly `sigma` — which is what
the gates assumed. For any other marginal it is `sigma * S'(z_p)`, and the missing slope is
large:

| horizon | S'(−1.96) | S'(0) | S'(+1.96) |
|---|---|---|---|
| 5 | 1.91 | 0.27 | 2.05 |
| 20 | 1.33 | 0.83 | 1.87 |
| **two-piece normal** | **1.00** | **1.00** | **1.00** |

So T5.2's tolerance was 1.3–2.2× too tight for the shaped family and T5.1's was up to 3.7×
too *loose*. `shape_slope(None, ·)` returns exactly 1.0, so the incumbent is scored
bit-for-bit as before — asserted in a test and confirmed empirically by rescoring it
(99.820 / 99.785 / 99.696 / 99.694, identical to the previous run).

Scored correctly, the shaped marginal's tail agreement is **99.8–100%**: T5.2 goes 0/4 → 4/4,
and its apparent failure was entirely the wrong tolerance. The correction cuts both ways —
T5.1's median tolerance tightens and h=5 now marginally fails at 0.9936 against 0.995.

| | overall | T5.1 | T5.2 | T1.2 | T1.3 | T6.1 ratios |
|---|---|---|---|---|---|---|
| two-piece normal | **91/128 = 0.711** | 4/4 | 4/4 | 13/16 | 3/3 | 3.84 / 5.65 / 6.75 / 6.65 |
| shaped, Gaussian tol | 84/126 = 0.667 | 4/4 | 0/4 | 11/15 | 1/2 | **1.74 / 2.88 / 4.36 / 4.55** |
| shaped, shape-aware tol | 87/126 = 0.690 | 3/4 | **4/4** | 11/15 | 1/2 | **1.74 / 2.88 / 4.36 / 4.55** |

Fairly scored the shape closes most of the gap (0.667 → 0.690) but still loses 0.690 to
0.711 at M=100. The residual gap sat in T1.2/T1.3, which are also estimated from the
ensemble's own percentiles and so carry the same finite-M penalty T5.2 did — so the
comparison was run again with **both** families at M=400.

### At matched M=400 the ordering reverses

| | M=100 | M=400 |
|---|---|---|
| two-piece normal | 91/128 = 0.711 | 85/126 = 0.675 |
| empirical shape | 87/126 = 0.690 | **90/126 = 0.714** |

Net **+5 rows** to the shape at M=400: better on T1.1, T1.2, T2.1, T2.4, T2.5, T6.1, T6.5;
worse on T5.1 (one horizon, 0.9936 against a 0.995 gate) and T7.3 (1.44 against 1.25).

Note the two-piece gets *worse* at M=400 while the shape gets better. That is the same
mechanism as T5.2: MC-scaled tolerances tighten as 1/√M, so a higher member count exposes
error that a loose tolerance was hiding. **The M=100 comparison was itself confounded** — it
scored one family at a member count where it is under-resolved.

### Recommendation

Adopt the empirical shape **together with M ≥ 400**; the two are a package, since at M=100
the shape is worse. Cost is 3.1 min and 3.0 GB per ensemble against 0.7 min and 0.3 GB. This
is left as a one-line change (`--marginal_shape` plus the member count in
`run_region_loop.sh`) rather than applied silently, because it changes a published product
and the plan asks for those to be versioned deliberately.

Honest caveat on the margin: +5 of 126 rows, and scorecard rows are correlated. The stronger
evidence is not the count but that the shape more than halves the defect it was built for
(T6.1 ratios 3.84/5.65/6.75/6.65 → 1.74/2.88/4.36/4.55) while every hard gate still holds.

### Three scoring bugs, all the same shape

Each was a diagnostic that silently assumed the two-piece normal, and each was caught by a
result that could not be true rather than by inspection:

| bug | how it announced itself | effect |
|---|---|---|
| T5 tolerances assumed `dx/dz = sigma` | T5.2 got *worse* at M=400, impossible against an MC-scaled tolerance without a bias | tolerance 1.3–2.2× too tight at the bounds, 0.27–0.83× too loose at the median |
| `recover_z` undid only the scaling | T3.1, a *field* property, moved under a rank-preserving transform | normal-score variance read 0.613 instead of 0.96; variogram measured `S(z)`, not `z` |
| T3.2 sampled pairs to 500 px against a 50 px range | 1.0% improvement against a 30% target, with no plausible mechanism | score dominated by pairs where the two ensembles are identical by construction |

All three fixes are exactly inert for the two-piece normal — asserted in tests and, for the
T5 one, confirmed by rescoring the incumbent and getting bit-identical numbers.

### Two bugs of mine, and how each was caught

1. **Tail inherited the outliers.** The first version put the crossover back to unit slope
   at z = 3.72 rather than at the 95% bound, so the mapping took the residual's extreme
   standardized outliers at face value and sent z = 3 to **5.4 half-widths**. Those outliers
   are mostly pixels whose *width* is near-degenerate — the ratio explodes because the
   denominator collapsed. Cost: T8.2 crossed its gate, T7.3 1.28 → 1.66, T4.2 0.997 → 0.79.
   Fixed by putting the crossover at the bound; the tail is then the two-piece normal
   exactly (z = 3 → 1.53).
2. **A bug created by fixing a non-bug.** Suspecting an interpolation bias at the gate
   quantiles, I pinned them as exact knots — which made the u-grid non-uniform and broke the
   *torch* path, which computed its bucket index by uniform-grid arithmetic. That is the GPU
   path every ensemble is generated on, so it would have mis-mapped members silently while
   numpy stayed right. Caught by the numpy/torch parity test.

The bias hypothesis itself was **wrong**: with both fixes the ensembles differ by at most
3e-4 HM and the scorecard is unchanged to three decimals. Worth recording because
byte-identical scorecards are this project's signature for scoring the wrong artifact — here
the artifacts were verified genuinely distinct first, and the null result is real.

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
