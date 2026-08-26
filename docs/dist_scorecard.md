# Distributional model scorecard — what each metric asks, and how the model does

Every metric `scripts/score_distributional_model.py` reports, explained in plain terms, with the
measured value beside it.

**Provenance.** Branch `dist-convlstm`, configuration **D0** (the probabilistic loss alone:
CRPS, auxiliary MSE on `E[Q]`, no SSIM, no Laplacian, no histogram), averaged over its **three
replicates** at seeds 42/43/44. Screened on **southern Africa**, folds 1 and 2 of the 512 px
`fold_mask_b4_1000.tif` mask, trained on global chips and predicted regionally. Measured
2026-08-25. Method and slate in `docs/dist_model_phase.md`.

**Read the levels, not small differences.** The run-to-run bands measured in
`docs/dist_model_phase.md` §3.1 differ enormously by metric: RMSE and coverage are stable to
2%, CRPS skill to 18% at h=5 and 40%+ at longer horizons, and `tail_reach` moves by 4.3 across
replicates of one configuration. Each section below says which it is.

---

## 1. Is the single best-guess map any good?

**`rmse`, `mae`, `bias`, `corr`, `slope`, `skill`**

Asks: how far is the central forecast from the truth, and does it beat "assume nothing changes"?

Persistence is the bar, not zero — the median 20-year HM change is 0.0001, so a forecast can look
accurate while being worse than doing nothing.

| h | RMSE | bias | corr | slope | skill vs persistence |
|---|---|---|---|---|---|
| 5 | 0.00865 | −0.00069 | 0.22 | 0.78 | +0.036 |
| 10 | 0.01362 | −0.00167 | 0.28 | 0.70 | +0.049 |
| 15 | 0.01756 | −0.00244 | 0.30 | 0.73 | +0.064 |
| 20 | 0.02037 | −0.00355 | 0.31 | 0.74 | +0.060 |

**Verdict: positive but thin.** It beats persistence at every horizon, by 4–6%. `bias` is
negative throughout, meaning the central forecast sits *above* the truth — the model
over-predicts change. `slope` 0.74 means when it predicts a change of 1.0 it should have
predicted 0.74; it over-reaches. `corr` 0.2–0.3 says most of the change it predicts is not where
change actually happened.

RMSE is the most trustworthy number on this whole list — its seed-to-seed band is 2%. Skill is
the same quantity with a small denominator, which is why its band is 88–155% and it cannot be
ranked.

---

## 2. Is the *whole distribution* any good?

**`crps`, `crps_skill`**

CRPS is the natural generalisation of absolute error to a distribution: it rewards being both
accurate and appropriately confident. A point forecast's CRPS is just its absolute error, so
persistence gives a free baseline.

| h | CRPS | persistence | skill |
|---|---|---|---|
| 5 | 0.00171 | 0.00196 | **+0.13** |
| 10 | 0.00305 | 0.00352 | **+0.13** |
| 15 | 0.00418 | 0.00487 | **+0.14** |
| 20 | 0.00500 | 0.00578 | **+0.13** |

**Verdict: solidly positive, ~13% better than persistence** at every horizon. This is the
headline: as a full predictive distribution the model earns its keep. Band is 18% at h=5 and
40%+ at longer horizons, so treat the level as real and small differences as noise.

---

## 3. Is the distribution the right *shape*?

**PIT — `pit_mean`, `pit_ks`, and the four tail fractions**

The sharpest test on the list. Ask each pixel: what percentile of my forecast did the truth land
at? If the distribution is right, those percentiles are uniform — as often at 3% as at 97%.

| h | pit_mean | P(u<0.001) | P(u<0.025) | P(u>0.975) | P(u>0.999) | KS |
|---|---|---|---|---|---|---|
| 5 | 0.419 | 0.0015 | 0.0224 | 0.0154 | 0.0016 | 0.267 |
| 10 | 0.414 | 0.0027 | 0.0212 | 0.0114 | 0.0017 | 0.252 |
| 15 | 0.410 | 0.0040 | 0.0225 | 0.0124 | 0.0023 | 0.208 |
| 20 | 0.396 | **0.0076** | 0.0269 | 0.0122 | 0.0020 | 0.262 |
| **should be** | **0.500** | 0.0010 | 0.0250 | 0.0250 | 0.0010 | **0** |

**Verdict: this is the weakest area, and it says two specific things.**

`pit_mean` ≈ 0.40 instead of 0.50 — the truth lands *below* the forecast's middle far too often.
The distribution is shifted up. Same fact as the negative bias in §1.

`P(u>0.975)` is 0.012 against 0.025 — only half as many observations exceed the upper bound as
should. But `P(u<0.001)` at h=20 is 0.0076, **seven times** what it should be. So the upper bound
is too generous and the deep lower tail is too thin: the model is braced for growth that does not
come and blindsided by declines.

KS of 0.26 on 1.8M pixels is enormous — the 5% critical value is about 0.001.

---

## 4. Are the intervals the right *width*?

**`cov50/80/95/99` and the matching widths**

If you publish a 95% interval, the truth should fall inside 95% of the time.

| h | cov50 | cov80 | cov95 | cov99 | width95 |
|---|---|---|---|---|---|
| 5 | **0.415** | 0.834 | 0.962 | 0.995 | 0.0177 |
| 10 | 0.470 | 0.889 | 0.967 | 0.993 | 0.0334 |
| 15 | 0.615 | 0.901 | 0.965 | 0.991 | 0.0439 |
| 20 | **0.626** | 0.907 | 0.961 | 0.987 | 0.0509 |

**Verdict: the 95% and 99% intervals are close to honest; the middle of the distribution is
not.** The 50% interval covers only 42% at h=5 (too narrow) and 63% at h=20 (too wide) — it is
wrong in *opposite directions* at the two ends.

This is why coverage alone is not enough. Look only at `cov95` and this model passes. §3 shows
the shape inside is wrong, and §5 shows the consequence.

---

## 5. Does it predict the right *amount* of change?

**Exceedance: `P(Δ>0.01)`, `P(Δ>0.05)`, `P(Δ<−0.01)` against observed, summarised as
mean |log₁₀(pred/obs)|**

Read straight off the quantile function — no sampling, no marginal assumption. "How often does
the model say a pixel will gain more than 0.05?" against how often it actually did.

| h | P(Δ>0.01) pred / obs | P(Δ>0.05) pred / obs | P(Δ<−0.01) pred / obs |
|---|---|---|---|
| 5 | 0.047 / 0.027 = 1.76 | 0.0061 / 0.0041 = 1.46 | 0.0139 / 0.0133 = 1.05 |
| 10 | 0.094 / 0.051 = 1.85 | 0.0175 / 0.0102 = 1.71 | 0.0243 / 0.0225 = 1.08 |
| 15 | 0.127 / 0.067 = 1.88 | 0.0279 / 0.0163 = 1.71 | 0.0347 / 0.0297 = 1.17 |
| 20 | 0.151 / 0.077 = 1.95 | 0.0367 / 0.0205 = 1.79 | 0.0318 / 0.0362 = 0.88 |

**Verdict: the model over-predicts increases by about a factor of two, and gets declines roughly
right.** Summary error 0.44 in |log₁₀|, i.e. typically wrong by a factor of ~2.7 across the
distance bands. Band is 9%, so this number is reliable.

---

## 6. Can it imagine rare, large change in remote country?

**`tail_reach`, `halfwidth_p999`, and exceedance per distance band**

This is what the phase exists for. `tail_reach` = how far Q(0.999) sits above the median, in
units of the 95% half-width. A normal distribution gives **1.577**, and the model starts exactly
there by construction, so anything above it was learned.

At h=20, by distance from existing change:

| band | n | 95% half-width | Q(0.999) − median | tail reach | P(Δ>0.05) pred / obs |
|---|---|---|---|---|---|
| 0–1 px | 78,762 | 0.0893 | 0.330 | 2.27 | 0.227 / 0.177 = **1.28** |
| 1–3 px | 82,284 | 0.0565 | 0.267 | 2.75 | 0.101 / 0.040 = **2.53** |
| 3–10 px | 345,867 | 0.0349 | 0.212 | 4.18 | 0.047 / 0.018 = **2.54** |
| 10–30 px | 535,425 | 0.0156 | 0.113 | 5.37 | 0.014 / 0.007 = **2.02** |
| 30–100 px | 323,169 | 0.0081 | 0.0374 | 4.39 | 0.0010 / 0.0024 = **0.40** |
| >100 px | **0** | — | — | — | *southern Africa has no pixels in this band* |

**Verdict: the model learned a genuinely heavy tail — 2.3 to 5.4 against the Gaussian 1.58 it
started from — and it is still too cold in the far field and too hot near change.**

The mechanism is now within reach, which it was not before. In the 30–100 px band, +0.05 sits
**6.2 half-widths** above the median and the tail reaches **4.4** — a shortfall of about 1.4×.
The frozen product's far band needed ~12 half-widths and reached 3–4, a shortfall of 3–4×.
Different extent, so not a like-for-like score, but the *gap the model has to close* is much
smaller.

The remaining problem is placement, not capability: over-concentrated near existing change
(2.5× too hot at 1–10 px) and 0.40 of observed at 30–100 px. Note also that `crps_skill` is
**−0.098** in that furthest band — where nothing much happens, persistence is very hard to beat.

`tail_reach` has a 19% band across baseline seeds but **4.3 across replicates of one variant**,
so read the level, never a small difference.

---

## 7. Is the product internally consistent?

**`qf_vs_triple_max`, `central_outside_interval`**

Not a model quality question — a "do the published files agree with each other" question. The
64-band quantile raster and the lower/central/upper triple must describe the same forecast, or
the scorecard and the ensemble would be scoring different models.

**`qf_vs_triple_max` = 1.53e-5**, exactly half the int16 storage quantum — the two agree exactly
before rounding. **`central_outside_interval` = 0.00000.** Both perfect at every horizon.

---

## The short version

| question | verdict |
|---|---|
| point forecast | beats persistence, but weakly; over-predicts and over-reaches |
| whole distribution | **+13% over persistence at every horizon — solid** |
| distribution shape | **worst area** — shifted up, deep lower tail 7× too thin |
| interval width | 95% honest; the 50% interval wrong in opposite directions at the two ends |
| amount of change | over-predicts increases ~2×; declines about right |
| rare remote change | **tail genuinely learned (4–5× Gaussian)**; still 2.5× too cold at 30–100 px, 2.5× too hot near change |
| internal consistency | exact |

One thread runs through §1, §3 and §5: the model over-predicts change and is under-prepared for
decline. Fix that one location bias and three of the seven families move together.
