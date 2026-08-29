# Distributional model scorecard — what each metric asks, and how the model does

Every metric `scripts/score_distributional_model.py` reports, explained in plain terms, with the
measured value beside it.

**Provenance.** Branch `dist-convlstm`, configuration **e1** — the floor plus the neighbourhood-HM
covariate (`--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean,max`,
twelve context channels instead of eight) — over **three seeds (45, 46, 47)**. Screened on
**Africa**, folds 1 and 2 of `fold_mask_b4_1000.tif`, trained on global chips, predicted on the
200-block screen (`--predict_subsample_seed 7`), 2,474,010 scored pixels at h=20. Measured
2026-08-28. Method and the full slate in `docs/dist_model_phase2.md`.

**This supersedes the southern-Africa edition**, which scored configuration D0 on a region whose
far-field band contains **zero pixels** — the defect this model exists to fix was unmeasurable
there. Numbers are not comparable between the two editions: Africa changes 2–4x more at every HM
level and its `[0,0.01)` stratum is 40% of the region against 6%.

**e1 is the best available configuration, not a validated improvement.** Ten variants were tested
against a six-seed floor and **none cleared the bar** (beat the floor's whole range by more than
the range's own width). e1 is the only one whose three seeds all landed outside the floor on the
better side, on five metrics, by 0.06–0.24 floor widths.

**Read the levels, not small differences.** Each table below carries the floor's own six-seed
range (`floor`, seeds 42, 43, 44, 45, 46, 47) as the noise band. Where the band is wide, the number cannot rank
anything — this is stated per section rather than left to be inferred.

---

## 1. Is the single best-guess map any good?

**`rmse`, `mae`, `bias`, `corr`, `slope`, `skill`**

Asks: how far is the central forecast from the truth, and does it beat "assume nothing changes"?
Persistence is the bar, not zero — the median 20-year HM change is 0.0001, so a forecast can look
accurate while being worse than doing nothing.

| h | RMSE | MAE | bias | corr | slope | skill vs persistence | floor's skill range |
|---|---|---|---|---|---|---|---|
| +5 | 0.01332 | 0.00468 | -0.00004 | 0.34 | 1.16 | **+0.1415** | 0.1328–0.1408 |
| +10 | 0.02075 | 0.00810 | +0.00003 | 0.41 | 1.20 | **+0.2112** | 0.1868–0.2088 |
| +15 | 0.02641 | 0.01121 | -0.00013 | 0.44 | 1.16 | **+0.2532** | 0.2154–0.2446 |
| +20 | 0.03108 | 0.01428 | -0.00064 | 0.43 | 1.14 | **+0.2523** | 0.2200–0.2468 |

**Verdict: clearly positive.** It beats persistence by 14% at +5 yr rising to 25% at +15/+20 yr,
and `corr` runs 0.34–0.45, so a real share of the change it predicts is where change happened.

`bias` is essentially zero (-0.00004 at h=5, -0.00064 at h=20)
— the systematic over-prediction the southern-Africa edition reported is not present here.
`slope` runs 1.16–1.20: when the model predicts a change of
1.0 the truth averages ~1.15, so it now *under*-reaches slightly rather than over-reaching.

RMSE is the most trustworthy number here — the floor's range is 0.03120–0.03175
at h=20, under 2%. Skill has a small denominator and a wider band; read it as a level.

---

## 2. Is the *whole distribution* any good?

**`crps`, `crps_skill`**

CRPS generalises absolute error to a distribution: it rewards being both accurate and
appropriately confident. A point forecast's CRPS is its absolute error, so persistence gives a
free baseline.

| h | CRPS | persistence | skill | floor's skill range |
|---|---|---|---|---|
| +5 | 0.00345 | 0.00440 | **+0.2126** | 0.2082–0.2130 |
| +10 | 0.00582 | 0.00786 | **+0.2587** | 0.2461–0.2581 |
| +15 | 0.00784 | 0.01097 | **+0.2847** | 0.2645–0.2818 |
| +20 | 0.00952 | 0.01325 | **+0.2816** | 0.2656–0.2785 |

**Verdict: the headline, and it is solid — 21% better than persistence at +5 yr and 28–29% at
+15/+20 yr.** As a full predictive distribution the model earns its keep at every horizon, and
the floor's own range is only 2–6% wide here, so the level is trustworthy.

---

## 3. Is the distribution the right *shape*?

**PIT — `pit_mean`, `pit_ks`, and the four tail fractions**

The sharpest test on the list. Ask each pixel: what percentile of my forecast did the truth land
at? If the distribution is right, those percentiles are uniform — as often at 3% as at 97%.

| h | pit_mean | P(u<0.001) | P(u<0.025) | P(u>0.975) | P(u>0.999) | KS | floor's KS range |
|---|---|---|---|---|---|---|---|
| +5 | 0.527 | 0.0034 | 0.0223 | 0.0201 | 0.0030 | 0.187 | 0.169–0.196 |
| +10 | 0.515 | 0.0031 | 0.0206 | 0.0215 | 0.0025 | 0.097 | 0.101–0.182 |
| +15 | 0.509 | 0.0038 | 0.0212 | 0.0278 | 0.0023 | 0.081 | 0.064–0.107 |
| +20 | 0.496 | 0.0044 | 0.0262 | 0.0300 | 0.0018 | 0.123 | 0.076–0.120 |
| **should be** | **0.500** | 0.0010 | 0.0250 | 0.0250 | 0.0010 | **0** | |

**Verdict: the body is well calibrated; both extreme tails are too thin.**

`pit_mean` runs 0.496–0.527 against a target of 0.500,
so the distribution is centred — the upward shift reported on southern Africa is gone, which is
the same fact as the near-zero bias in §1. The 2.5% tails are close to nominal:
`P(u<0.025)` 0.0223–0.0262 and
`P(u>0.975)` 0.0201–0.0300 against 0.025.

The **0.1% tails are 2–4x too populated**: `P(u<0.001)` runs
0.0031–0.0044 and
`P(u>0.999)` 0.0018–0.0030,
both against a nominal 0.001. So observations land beyond the deepest quantiles the model
expresses several times too often, on **both** sides. That is the defect `e4` (lower-tail
weighting) and `e10` (a coarser knot grid) were aimed at; e4 was a null and e10 was not run.

KS of 0.187 at h=5 on 9,896,040 pixels is still enormous —
the 5% critical value is about 0.001 — but the h≥10 rows sit at
0.081–0.097 against southern Africa's 0.21–0.27. Note the
floor's own KS range at h=10 is 0.101–0.182, which is 66% of its level:
that row is **under-powered** and a difference there means nothing.

---

## 4. Are the intervals the right *width*?

**`cov50/80/95/99` and the matching widths**

If you publish a 95% interval, the truth should fall inside 95% of the time.

| h | cov50 | cov80 | cov95 | cov99 | width50 | width95 | floor's cov95 range |
|---|---|---|---|---|---|---|---|
| +5 | **0.435** | 0.799 | 0.958 | 0.989 | 0.0029 | 0.0296 | 0.941–0.956 |
| +10 | **0.529** | 0.838 | 0.959 | 0.991 | 0.0061 | 0.0480 | 0.954–0.959 |
| +15 | **0.560** | 0.854 | 0.951 | 0.991 | 0.0094 | 0.0588 | 0.950–0.953 |
| +20 | **0.571** | 0.855 | 0.944 | 0.989 | 0.0113 | 0.0672 | 0.943–0.951 |

**Verdict: the outer intervals are honest; the 50% interval is wrong in opposite directions at
the two ends.** `cov95` runs 0.944–0.959 against 0.95 and
`cov99` 0.989–0.991 against 0.99 — both close. But `cov50`
covers 0.435 at +5 yr (too narrow) and 0.571 at +20 yr (too
wide). The middle of the distribution is the part that is mis-shaped, and it is mis-shaped in a
horizon-dependent way, which is why §3's KS is worst at h=5.

This is why coverage alone is not enough: look only at `cov95` and the model passes.

---

## 5. Does it predict the right *amount* of change?

**Exceedance: `P(Δ>0.01)`, `P(Δ>0.05)`, `P(Δ<−0.01)` against observed**

Read straight off the quantile function — no sampling, no marginal assumption. "How often does
the model say a pixel will gain more than 0.05?" against how often it actually did.

| h | P(Δ>0.01) pred / obs | P(Δ>0.05) pred / obs | P(Δ<−0.01) pred / obs |
|---|---|---|---|
| +5 | 0.0808 / 0.0871 = **0.93** | 0.01256 / 0.01468 = **0.86** | 0.0146 / 0.0195 = **0.74** |
| +10 | 0.1433 / 0.1449 = **0.99** | 0.03186 / 0.03523 = **0.90** | 0.0224 / 0.0259 = **0.87** |
| +15 | 0.1960 / 0.1860 = **1.05** | 0.05246 / 0.05518 = **0.95** | 0.0305 / 0.0303 = **1.01** |
| +20 | 0.2291 / 0.2112 = **1.08** | 0.06878 / 0.07093 = **0.97** | 0.0278 / 0.0301 = **0.92** |

**Verdict: pooled exceedance is close to right — every ratio between 0.86 and 1.08.** The
southern-Africa edition reported a factor-of-two over-prediction of increases; on Africa the
aggregate amount of change is well calibrated, and declines are within 15% at h≥10.

The summary statistic `exceedance_abs_log10` is **0.3399**
— but note what it summarises: the mean of |log₁₀(pred/obs)| **across distance bands**, not
pooled. Pooled agreement of 0.97 coexists with per-band errors of 1.6x and worse. §6 is where
that lives, and it is the section that matters for this model's purpose.

---

## 6. Can it imagine rare, large change in remote country?

**`tail_reach`, `halfwidth_p999`, and exceedance per distance band**

This is what the phase exists for. `tail_reach` = how far Q(0.999) sits above the median, in
units of the 95% half-width. A normal distribution gives **1.577**, and the model starts there by
construction, so anything above it was learned.

At +20 yr, by distance from existing change:

| band | n | 95% half-width | Q(0.999) − median | tail reach | P(Δ>0.05) pred / obs | +0.05 sits at |
|---|---|---|---|---|---|---|
| 0-1 px | 288,667 | 0.0857 | 0.3969 | **2.52** | 0.234700 / 0.261370 = **0.90** | 0.6 half-widths |
| 1-3 px | 287,156 | 0.0649 | 0.3662 | **3.23** | 0.148174 / 0.132280 = **1.12** | 0.8 half-widths |
| 3-10 px | 717,385 | 0.0355 | 0.2585 | **5.55** | 0.059951 / 0.069600 = **0.86** | 1.4 half-widths |
| 10-30 px | 647,409 | 0.0183 | 0.2066 | **8.03** | 0.023734 / 0.017371 = **1.37** | 2.7 half-widths |
| 30-100 px | 357,915 | 0.0056 | 0.0587 | **7.14** | 0.003867 / 0.002353 = **1.64** | 8.9 half-widths |
| >100 px | 175,478 | 0.0024 | 0.0217 | **5.83** | 0.000001 / 0.000137 = **0.01** | 21.3 half-widths |

**Verdict: the tail is genuinely learned out to 30 px and the far field is still cold — but the
far-field exceedance number is not an estimate, and this scorecard will not pretend otherwise.**

The model reaches 2.5–8.0 against the Gaussian 1.58 it started from, and the near bands are
within 10–40% of observed out to 30 px. Two bands then fail in opposite directions: 30–100 px is
**1.64x too hot**, and >100 px reads 0.01 of observed at the median.

**That far-field ratio is unusable at this sample size.** Across e1's three seeds it is
1.486 / 0.003 / 0.010; across the floor's six it is 0.076 / 0.533 / 0.001 / 0.011 / 0.001 / 1.502. Three orders of magnitude,
from the same configuration. The quantity is a tiny tail probability integrated over
175,478 pixels, and it is dominated by whether a handful of pixels'
quantile functions happen to reach past +0.05 at all. **Any single-number claim about far-field
exceedance — including "7.3% of observed", recorded earlier in this project — is one draw from
that spread, not a measurement.**

Use the two numbers in that band that *are* stable instead. `tail_reach` there is 6.04 / 5.37 / 5.83
across e1's seeds, and the 95% half-width is 0.00235, which puts +0.05 at **21.3
half-widths** above the median. To emit +0.05 at the observed rate the model would need a reach
of that order; it has ~6. That is the defect stated in the units where it can be measured — the
same statement the frozen product's scorecard made at ~12 half-widths, on a different extent.

Note also `crps_skill` in the far band is **-1.72**: where almost nothing happens,
persistence is very hard to beat, and the model loses to it badly.

---

## 7. Is the product internally consistent?

**`qf_vs_triple_max`, `central_outside_interval`**

Not a model-quality question — a "do the published files agree with each other" question. The
64-band quantile raster and the lower/central/upper triple must describe the same forecast, or
the scorecard and the ensemble would be scoring different models.

**`qf_vs_triple_max` = 1.54e-05**, half the int16 storage quantum — the two agree exactly before
rounding, at every horizon and every seed.

**`central_outside_interval` is not clean for this configuration.** Two of three seeds read
exactly 0, but `af_e1_s45` reads **0.00636** — 0.6% of pixels at +15 yr where the central
forecast falls outside its own published interval. The floor never does this. It is one seed and
the magnitude is small, but it is a consistency violation rather than a quality metric, and it
should be understood before e1 is adopted.

---

## The short version

| question | verdict |
|---|---|
| point forecast | **beats persistence by 14–25%**; unbiased; slightly under-reaches (slope ~1.15) |
| whole distribution | **+21% to +29% over persistence — the headline, and the floor's band is narrow** |
| distribution shape | body centred and the 2.5% tails near nominal; **both 0.1% tails 2–4x too thin** |
| interval width | 95% and 99% honest; the 50% interval too narrow at h=5 and too wide at h=20 |
| amount of change | **pooled exceedance within 15% at every horizon** — the southern-Africa 2x over-prediction is absent |
| rare remote change | reach 2.5–8.0 vs Gaussian 1.58; 30–100 px **1.6x too hot**; >100 px cold, and **not estimable at 3 seeds** |
| internal consistency | quantile function and triple agree exactly; one seed puts the centre outside its own interval on 0.6% of h=15 pixels |

**The two things to fix next, in order.** The deep tails are too thin on both sides while the
body is well calibrated — that is a shape problem at fixed centre, and it is what the knot grid
and the tail-weighted objectives were aimed at. And the far field needs a reach of ~20 where it
has ~6; nothing tested in this phase moved it, and the one lever that clearly *does* move it
moves it the wrong way (removing the cumulative-width constraint costs 35% of tail reach).

**A measurement caveat that outranks both.** The far-field exceedance row cannot be estimated
from three seeds, or six. Before any experiment is judged on it, the estimator needs fixing —
more seeds will not do it, because the spread is model variance rather than sampling error.

