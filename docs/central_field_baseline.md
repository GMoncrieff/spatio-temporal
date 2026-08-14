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
