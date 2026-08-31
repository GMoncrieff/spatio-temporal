# Global distributional ConvLSTM scorecard — what each metric asks, and how the model does

Every metric `scripts/score_distributional_model.py` reports, explained in plain terms, with the
measured value beside it. **The ConvLSTM alone — no ensemble.**

**Provenance.** Branch `dist-convlstm`, configuration **e1**, seed 46. Trained and predicted
globally: k=5 fold cross-validation against `fold_mask_b4_1000.tif` (512 px blocks), all five
folds, `--stitch_mode holdout`, so every scored pixel comes from the fold model that never saw
it. Grid 17111 x 40000; **184,573,321 valid pixels per window-year**, 738 M scored rows at h=5
falling to 184.6 M at h=20. Ten (base year, target year) windows. Measured 2026-08-31 from
`data/ensemble/exp/g_e1_hind_score/`. Method in `docs/global_methodology.md`, runbook in
`docs/fitting_running_model.md`.

```
--head_family spline --central_residual True --central_context True --quantile_context True
--checkpoint_monitor val_crps --ssim_weight 0 --laplacian_weight 0 --histogram_weight 0
--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean,max
--weight_avg_last 20 --seed 46
```

**This supersedes the Africa edition of this scorecard.** That edition screened three seeds on a
200-block subsample of Africa (2.47 M pixels at h=20) and stated, correctly, that its far-field
exceedance row could not be estimated at that sample size — the ratio ranged over three orders of
magnitude across seeds of one configuration. **The global run resolves that**: the >100 px band
here holds 33.6 M pixels rather than 175 k, a factor of 191, and the far-field number is a
measurement rather than one draw from a wide spread.

**One run, not a replicate set.** CLAUDE.md rule 29 applies: a single global run cannot bound its
own noise. Nothing here should be read as ranking e1 against another configuration. The numbers
are levels for the shipped product, not evidence for a choice — the choice was made regionally
and is documented in `docs/dist_model_phase2.md`.

---

## 1. Is the single best-guess map any good?

**`rmse`, `mae`, `bias`, `corr`, `slope`, `skill`**

Asks: how far is the central forecast from the truth, and does it beat "assume nothing changes"?
Persistence is the bar, not zero — the median 20-year HM change is 0.0001, so a forecast can look
accurate while being worse than doing nothing.

| h | RMSE | MAE | bias | corr | slope | skill vs persistence |
|---|---|---|---|---|---|---|
| +5 | 0.01333 | 0.00409 | +0.000209 | 0.319 | 1.136 | **+0.1239** |
| +10 | 0.02046 | 0.00731 | +0.000286 | 0.412 | 1.128 | **+0.2069** |
| +15 | 0.02610 | 0.00997 | +0.000458 | 0.451 | 1.088 | **+0.2566** |
| +20 | 0.03100 | 0.01265 | +0.000247 | 0.434 | 1.050 | **+0.2499** |

**Verdict: clearly positive at every horizon.** It beats persistence by 12% at +5 yr rising to
26% at +15 yr. `corr` runs 0.32–0.45, so a real share of the change it predicts is where change
happened. `bias` is +0.0002 to +0.0005 — three orders of magnitude below RMSE, effectively
unbiased. `slope` falls from 1.14 to 1.05 with lead time: when the model predicts a change of
1.0 the truth averages 1.05–1.14, so it under-reaches slightly, and less so at longer horizons.

These are the same levels the Africa screen reported (skill 0.14 / 0.21 / 0.25 / 0.25), which is
the useful fact: the central field transferred from the screening region to the globe unchanged.

---

## 2. Is the *whole distribution* any good?

**`crps`, `crps_skill`**

CRPS generalises absolute error to a distribution: it rewards being both accurate and
appropriately confident. A point forecast's CRPS is its absolute error, so persistence gives a
free baseline.

| h | n | CRPS | skill vs persistence |
|---|---|---|---|
| +5 | 738,293,284 | 0.003049 | **+0.2000** |
| +10 | 553,719,963 | 0.005330 | **+0.2687** |
| +15 | 369,146,642 | 0.007239 | **+0.3065** |
| +20 | 184,573,321 | 0.008985 | **+0.2959** |

**Verdict: the headline. 20% better than persistence at +5 yr and 30% at +15 yr, as a full
predictive distribution.** This is what the model is trained on and it is where it is strongest.

Africa reported +0.21 / +0.26 / +0.28 / +0.28 on the same metric. The globe is **better at every
horizon**, most at +15 yr (0.28 → 0.31). A screening region that under-states the shipped
product's skill is the pleasant direction for that error to run.

---

## 3. Is the distribution the right *shape*?

**PIT — `pit_mean`, `pit_ks`, and the tail fractions**

The sharpest test on the list. Ask each pixel: what percentile of my forecast did the truth land
at? If the distribution is right, those percentiles are uniform — as often at 3% as at 97%.

| h | pit_mean | P(u<0.001) | P(u<0.025) | P(u>0.975) | P(u>0.999) | KS |
|---|---|---|---|---|---|---|
| +5 | 0.517 | 0.01306 | 0.0304 | 0.0204 | 0.00341 | 0.157 |
| +10 | 0.495 | 0.01361 | 0.0276 | 0.0212 | 0.00365 | 0.116 |
| +15 | 0.552 | 0.01467 | 0.0302 | **0.1019** | 0.00420 | 0.094 |
| +20 | 0.578 | 0.01633 | 0.0382 | **0.0970** | 0.00412 | 0.153 |
| **should be** | **0.500** | 0.0010 | 0.0250 | 0.0250 | 0.0010 | **0** |

**Verdict: this is the model's weakest area, and it fails in two distinct ways.**

**The lower tail is 13–16x too thin at every horizon.** `P(u<0.001)` runs 0.0131–0.0163 against
a nominal 0.001. Roughly 1.5% of observations land below the model's 0.1st percentile. The 2.5%
level is nearly right (0.028–0.038 against 0.025), so the defect is confined to the deep tail:
the model knows how far down to reach for a typical low outcome and badly under-reaches for an
extreme one.

**The upper tail breaks at h=15 and h=20 specifically.** `P(u>0.975)` is near nominal at h=5 and
h=10 (0.020, 0.021) and then jumps to **0.102 and 0.097** — four times nominal. Something about
the long horizons puts a tenth of observations above the published upper bound. This is the same
fact as §4's coverage collapse and is the single most actionable finding in this scorecard.

`pit_mean` drifts upward with lead time (0.517 → 0.578), so at long horizons the truth sits
higher in the forecast distribution than it should on average — the distribution is placed a
little low, not merely mis-shaped.

KS of 0.094–0.157 is enormous on this many pixels (the 5% critical value is ~1e-4), but KS is
meaningless as a significance test at n = 1e8; read it as an effect size. The h=5 and h=20 rows
being the worst matches the two failure modes above.

---

## 4. Are the intervals the right *width*?

**`cov50/80/95/99` and the matching widths**

If you publish a 95% interval, the truth should fall inside 95% of the time.

| h | cov50 | cov80 | cov95 | cov99 | width50 | width95 |
|---|---|---|---|---|---|---|
| +5 | 0.4477 | 0.8337 | 0.9604 | 0.9914 | 0.00271 | 0.02525 |
| +10 | 0.5872 | 0.8642 | 0.9623 | 0.9900 | 0.00590 | 0.04186 |
| +15 | 0.5196 | 0.7614 | **0.8790** | 0.9887 | 0.00846 | 0.05072 |
| +20 | 0.4801 | 0.7489 | **0.8759** | 0.9856 | 0.01024 | 0.05606 |

**Verdict: honest at +5 and +10 yr; the 95% interval is materially too narrow at +15 and +20.**

`cov95` is 0.960 and 0.962 at the short horizons — slightly conservative, which is the safe
direction — and then **0.879 and 0.876** at +15/+20 against a nominal 0.95. Roughly one
observation in eight falls outside a published 95% interval at the horizons the product is most
used for. `cov99` stays close throughout (0.986–0.991), so the failure is specific to the 95%
level rather than a general under-dispersion.

`cov50` is erratic (0.45 → 0.59 → 0.52 → 0.48) with no monotone pattern, which is the middle of
the distribution being mis-shaped rather than mis-scaled.

**This is the number the ensemble layer must be judged against.** The ensemble does not change
the marginal — it draws from this quantile function — so it cannot fix a 0.88 coverage. If the
global ensemble scorecard shows near-nominal coverage at h=20, that is evidence of a bug, not of
an improvement.

---

## 5. Does it predict the right *amount* of change?

**Exceedance: `P(Δ>0.01)`, `P(Δ>0.05)`, `P(Δ<−0.01)` against observed**

Read straight off the quantile function — no sampling, no marginal assumption.

| h | P(Δ>0.01) pred / obs | P(Δ>0.05) pred / obs | P(Δ<−0.01) pred / obs |
|---|---|---|---|
| +5 | 0.0727 / 0.0759 = **0.96** | 0.01058 / 0.01322 = **0.80** | 0.0142 / 0.0153 = **0.93** |
| +10 | 0.1355 / 0.1363 = **0.99** | 0.02886 / 0.03310 = **0.87** | 0.0209 / 0.0233 = **0.90** |
| +15 | 0.1789 / 0.1769 = **1.01** | 0.04799 / 0.05337 = **0.90** | 0.0281 / 0.0287 = **0.98** |
| +20 | 0.2044 / 0.1987 = **1.03** | 0.06131 / 0.06873 = **0.89** | 0.0283 / 0.0318 = **0.89** |

**Verdict: pooled exceedance is close to right — every ratio between 0.80 and 1.03.** The
+0.01 threshold is essentially exact (0.96–1.03). The +0.05 threshold is consistently 10–20%
under-predicted, and declines 7–11% under.

`exceedance_abs_log10` = **0.4335**, but note what it summarises: the mean of |log₁₀(pred/obs)|
**across distance bands**, not pooled. Pooled agreement of 0.89 coexists with per-band errors of
2x. §6 is where that lives.

---

## 6. Can it imagine rare, large change in remote country?

**`tail_reach`, `halfwidth_p999`, and exceedance per distance band**

This is what the whole distributional lineage exists for. `tail_reach` = how far Q(0.999) sits
above the median, in units of the 95% half-width. A normal distribution gives **1.577**, and the
model starts there by construction, so anything above it was learned.

At +20 yr, by distance from existing change:

| band | n | 95% width | Q(.999)−median | reach | P(Δ>0.05) pred / obs | ratio | +0.05 sits at |
|---|---|---|---|---|---|---|---|
| 0-1 px | 25,293,071 | 0.1639 | 0.3600 | **2.57** | 0.24570 / 0.26569 | **0.92** | 0.6 half-widths |
| 1-3 px | 17,054,286 | 0.1172 | 0.3208 | **3.04** | 0.13803 / 0.12056 | **1.14** | 0.9 half-widths |
| 3-10 px | 36,038,668 | 0.0656 | 0.2385 | **5.54** | 0.04942 / 0.06965 | **0.71** | 1.5 half-widths |
| 10-30 px | 35,848,186 | 0.0345 | 0.1911 | **8.23** | 0.02055 / 0.02620 | **0.78** | 2.9 half-widths |
| 30-100 px | 36,773,931 | 0.0123 | 0.0693 | **7.09** | 0.00517 / 0.01024 | **0.50** | 8.2 half-widths |
| >100 px | 33,565,179 | 0.0046 | 0.0445 | **5.30** | 0.00116 / 0.00250 | **0.47** | 21.8 half-widths |

**Verdict: the far field is alive. This is the result the lineage was built for.**

| product | far-field `P(Δ>0.05)` as a fraction of observed, h=20 |
|---|---|
| frozen `g1_foldb4` post-hoc chain | **0.034** |
| Africa distributional screen (3 seeds) | 0.001 – 1.50, unusable |
| **global e1, this scorecard** | **0.466** on 33.6 M pixels |

**13.7x the frozen product**, and measured on a sample large enough to state. The ratio degrades
monotonically with distance — 0.92, 1.14, 0.71, 0.78, 0.50, 0.47 — so the model is close in the
near field and increasingly conservative with remoteness, rather than silent beyond some
threshold. That is a different and much more tractable defect than the one this project started
with.

**But the far field is only alive at +20 yr.** The same row across horizons:

| h | >100 px pred / obs | ratio |
|---|---|---|
| +5 | 0.000000 / 0.00008 | **0.000** |
| +10 | 0.000000 / 0.00025 | **0.000** |
| +15 | 0.000030 / 0.00081 | **0.035** |
| +20 | 0.001160 / 0.00250 | **0.466** |

At +5 and +10 the model emits no development beyond 100 px at all. The observed rates there are
tiny (8e-5, 2.5e-4) so the absolute miss is small, but a user asking "where might new development
appear by 2030 in currently-remote country" gets nothing. **Only the +20 yr answer is usable.**

**A caution about which pixels carry it.** The band's *median* pixel has a 95% half-width of
0.0023, putting +0.05 at **21.8 half-widths** above its median, and a reach of only 5.3. A median
far-field pixel therefore cannot produce +0.05 at all. The 0.466 comes from a minority of
far-field pixels whose distributions are much wider. That is the mirror image of CLAUDE.md rule
26: the defect there lived in a thousandth of the pixels, and here the *capability* does. Any
future work on this row should be judged per pixel, not on the band mean.

`crps_skill` is positive in every band including the far field (+0.21 beyond 100 px) — unlike the
Africa screen, where it was **−1.72** there. On a large enough sample the model beats persistence
even in quiet country.

---

## 7. Is the product internally consistent?

**`qf_vs_triple_max`, `central_outside_interval`**

Not a model-quality question — a "do the published files agree with each other" question. The
64-band quantile raster and the lower/central/upper triple must describe the same forecast, or
the scorecard and the ensemble would be scoring different models.

**`qf_vs_triple_max` = 1.55e-05**, half the int16 storage quantum of 3.05e-05. The published
`lower` and `upper` are exactly Q(0.025) and Q(0.975) of the published quantile function, at
every horizon and every window. The COGs and the quantile store describe one forecast.

**`central_outside_interval` = 0.089 at h=15, 0.073 at h=20; ~1e-5 at h=5 and h=10.**

This is **intended behaviour, not a defect**, and it is worth understanding before it is
reported as one. The published `central` is `E[Q] = ∫₀¹ Q(u) du`, the *mean* of the quantile
function — not `Q(0.5)`, the median. The mean is the RMSE-optimal point estimate and is what §1
scores; the median would be optimal for MAE. A mean is not a percentile, so nothing constrains it
to lie between two percentiles.

For a strongly right-skewed pixel it will not. Consider remote land with a 97.6% chance of
staying at HM ≈ 0.01 and a 2.4% chance of a town arriving at HM ≈ 0.9: every one of Q(0.025),
Q(0.5) and Q(0.975) sits inside the "nothing happens" mass at ≈ 0.01, while E[Q] ≈ 0.031 — three
times Q(0.975). It appears only at h=15/20 because that is where the right tail gets long enough
(pooled reach 4.9–5.9 against a Gaussian 1.58).

`src/models/spatiotemporal_predictor.py::_forward_spline` documents this explicitly and the
scorer exists to report how often it happens. **Consumers must be told**: `central` is a mean,
not a percentile, and on rare-jump pixels it can fall outside `[lower, upper]`.

---

## The short version

| question | verdict |
|---|---|
| point forecast | **beats persistence by 12–26%**; unbiased; slightly under-reaches (slope 1.05–1.14) |
| whole distribution | **+20% to +31% over persistence — the headline, and better than the Africa screen** |
| distribution shape | **lower tail 13–16x too thin at every horizon; upper tail 4x too thin at h=15/20** |
| interval width | honest at +5/+10 (cov95 0.960/0.962); **too narrow at +15/+20 (0.879/0.876)** |
| amount of change | pooled exceedance 0.80–1.03 of observed; +0.05 consistently 10–20% under |
| rare remote change | **far field at 0.466 of observed, 13.7x the frozen product** — but only at +20 yr |
| internal consistency | quantile function and triple agree to 1.55e-05; `central` is a mean and may sit outside its own interval on 7–9% of h=15/20 pixels, by design |

**The two things to fix next, in order.**

1. **The 95% interval at +15/+20 yr.** Coverage 0.88 against nominal 0.95, and `P(u>0.975)` at
   four times nominal, are the same defect seen twice. It is horizon-specific — +5 and +10 are
   fine — which points at the cumulative width construction rather than at the shape head.
2. **The deep lower tail**, 13–16x too thin at every horizon while the 2.5% level is nearly
   right. This is a tail-shape problem at fixed body, and it is what `e4` (lower-tail weighting)
   and `e10` (coarser knots) were aimed at regionally; neither cleared the bar there and neither
   has been tried against a sample this size.

**And one that is now measurable for the first time.** The far field is at 0.466 of observed at
+20 yr and 0.000 at +5 and +10. The lineage's original defect is half-fixed rather than fixed,
and the remaining half is a horizon problem, not a distance problem. That is a new statement —
it could not be made from any regional screen, because no regional extent had the pixels to make
it.
