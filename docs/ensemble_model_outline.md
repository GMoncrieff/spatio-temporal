# The Ensemble Model — What We Built and Why

Branch `ensemble`, written 2026-08-17. This describes the whole approach as it now stands,
against the model on `main` as the reference point. It is written to be read by someone who
knows the project but not necessarily the statistics of spatial simulation; §2 in particular
explains the machinery from first principles.

---

## 0. Orientation — what the product is, and what problem the ensemble solves

The model forecasts **Human Modification (HM)**, a 0–1 index on a 1 km grid, at four lead
times: +5, +10, +15 and +20 years. For every pixel it publishes three numbers per horizon:

- a **central forecast** — the best single estimate,
- a **lower bound** at the 2.5th percentile and an **upper bound** at the 97.5th,

which together are a 95% interval. `main` produces exactly this and nothing more.

### Why that is not enough

Per-pixel intervals answer *"how uncertain is this one pixel?"*. They cannot answer
*"how uncertain is this district / this ecoregion / this country?"*, because to aggregate
you need to know **how the errors at different pixels relate to each other**.

Two extreme assumptions bracket the truth, and both are wrong:

- **Assume errors are perfectly correlated** — just average the bounds over the region. This
  over-states aggregate uncertainty: measured coverage went 0.98 → 1.00, i.e. the interval
  became so wide it was always right and therefore useless.
- **Assume errors are independent** — add them in quadrature. This under-states it badly:
  coverage collapsed 0.284 → 0.045 → 0.009 as the blocks got bigger. Independence makes
  errors cancel, so a 1000 km region looks almost perfectly known, which it is not.

The truth is in between, and *where* in between depends on the real spatial correlation of
the model's errors. **The ensemble exists to represent that correlation explicitly.**

An ensemble is a set of *M* complete maps ("members"). Each member is one plausible version
of what the world could look like, consistent with the published bounds. To get uncertainty
at any scale you simply compute your quantity on each member and look at the spread across
members. Aggregation stops being a modelling assumption and becomes arithmetic.

The chain, end to end:

```
inputs (HM at 3 timesteps + covariates)
   │
   ├─► ConvLSTM ─► central forecast + lower/upper bounds        ← §1
   │
   ├─► hindcast residuals (what the model actually got wrong)   ← §4
   │        │
   │        ├─► recalibration decision, per-class width factors
   │        ├─► marginal shape (the per-pixel error distribution)
   │        └─► spatial field parameters (how errors correlate)
   │
   └─► ensemble generator ─► M members ─► scorecard             ← §2, §3
```

---

## 1. Central and Quantile Models

### 1.1 What `main` has

A ConvLSTM over three input timesteps (t−10, t−5, t) producing twelve output channels —
four horizons × {lower, central, upper}.

| | |
|---|---|
| trunk | 4-layer ConvLSTM, `hidden_dim=64`, 3×3 kernels |
| inputs | HM plus its component layers (dynamic), 7 static rasters (elevation, climate, protection status), 8 learned location-encoder channels |
| heads | 12 independent heads: `Conv3×3 → ReLU → Conv1×1`, quantile heads at half width |
| central loss | MSE on change + 0.2·SSIM + 0.3·Laplacian pyramid + 1.0·histogram, all on **absolute HM** |
| quantile loss | pinball at 0.025 and 0.975, gradient-isolated from the trunk |
| optimiser | Adam 1e-3, no schedule, 150 epochs × 13 steps × 8 chips of 128² |

### 1.2 What changed on this branch

Four changes, all flag-gated, all measured individually on held-out folds.

**(a) `--central_residual` — the central head predicts *change*, not level.**
On `main` the head predicts absolute HM, which means it must first reproduce HM at t₀
through the trunk before it can add anything. That reproduction is not free: on the 53–70%
of pixels whose true change is below 0.001 — where the right answer is "emit t₀ unchanged" —
the model still emitted change with sd ≈ 0.0075 HM, which is **larger than the entire signal
being predicted**. With a skip connection and a zero-initialised output convolution, the
model starts at exact persistence and "nothing happens" costs nothing to say. This was the
large effect: −30% RMSE at h=5.

**(b) `--central_context` / `--quantile_context` — a new derived covariate.**
The trunk's receptive radius is about 10 px, set by its 3×3 kernels. It therefore **cannot
see whether change happened 30–100 px away** — which is the single best predictor of whether
change is possible at a pixel at all. Beyond 100 px from past change, not one of 493,240
measured pixels moved by more than 0.01 in twenty years.

`scripts/prepare_change_context.py` precomputes, on the **full raster**, the distance from
every pixel to the nearest pixel that changed by >0.01 in the previous decade. Both head
families receive 8 channels derived from it:

| channels | content |
|---|---|
| 5 | is there past change within 1 / 3 / 10 / 30 / 100 px? |
| 1 | `log1p(distance)/10` |
| 1 | the signed past change itself |
| 1 | HM at t₀ |

It must be precomputed globally: derived inside a 128 px chip, every radius ≥30 px saturates
into "is there any change anywhere in this chip".

**(c) `--monotone_quantile_width` — the interval is built around the central forecast.**
On `main` the three heads predict their values independently, so nothing prevents the h=10
interval from being *narrower* than the h=5 one. It was: the h=10 half-width was smaller than
h=5's while the error it had to cover was 1.5× larger, and h=10 coverage fell to 0.892.
Now each quantile head emits a non-negative **half-width increment** that accumulates across
horizons around the (detached) central forecast. Spread cannot shrink with lead time, and
`lower ≤ central ≤ upper` is structural rather than a clipped-afterwards property.

**(d) Data-loader and harness changes.** A confirmed bug is fixed: grid-mode sampling ignored
the split mask entirely, so validation and test metrics were computed over the whole globe
instead of held-out geography. **Any `val_coverage_*` number from before that fix is not
comparable to one after it.** Added: k-fold splits, fold-restricted prediction, normalisation
statistics as a JSON sidecar, per-window prediction, device pinning.

### 1.3 Measured effect (k=5, southern Africa, whole region, both scored identically)

| horizon | RMSE `main` | RMSE now | MAE `main` | MAE now | skill vs persistence |
|---|---|---|---|---|---|
| +5 yr | 0.01478 | **0.00947** (−36%) | 0.00759 | **0.00288** (−62%) | −1.227 → **+0.085** |
| +10 yr | 0.01746 | **0.01444** (−17%) | 0.00859 | **0.00505** (−41%) | −0.244 → **+0.148** |
| +15 yr | 0.02047 | **0.01855** (−9%) | 0.00988 | **0.00728** (−26%) | +0.006 → **+0.184** |
| +20 yr | 0.02416 | **0.02222** (−8%) | 0.01255 | **0.00939** (−25%) | +0.045 → **+0.191** |

"Skill vs persistence" compares the model against the forecast *"nothing will change"*.
Positive means it beats that; negative means it would be better to predict no change at all.
**On `main` the model lost to persistence at +5 and +10 years** — its squared error at +5 yr
was 2.2× worse than doing nothing. It now wins at every horizon, and by more at longer lead,
which is the sensible ordering.

Scorecard on southern Africa: **73/137 → 101/127**; the same chain scores **111/133** on Africa (§3).

### 1.4 Things found in the model that should be addressed

These were discovered while auditing. **None of them will change or degrade the model** —
they are latent defects, wasted work, or measurement hazards.

**(a) The histogram loss has never trained anything.** `compute_histogram` bins by boolean
comparison and sums into a plain `torch.zeros` buffer, so the returned loss carries no
gradient at all — verified: `requires_grad=False`, and `backward()` raises. It has been in
the central objective at weight 1.0 for the whole project without moving a single parameter.

It is not harmless, because it **is** inside `val_total_loss`, which is what `ModelCheckpoint`
selects on, and it swings by ~60× between adjacent epochs. So it has been adding a large
random term to epoch selection. Setting `--histogram_weight 0` changes nothing about
training and removes that noise. (Making it differentiable was tried and is *worse* — h=20
RMSE +6.4% — because its rarity weights favour rare large-change bins ~240×, chasing the
marginal distribution of change at the expense of the conditional mean.)

**(b) The four horizons are trained 4:3:2:1.** `end_year` is sampled from
(2000, 2005, 2010, 2015) and any target past 2020 is NaN-filled and masked. So h=5 gets a
target from all four sampled windows, h=10 from three, h=15 from two, and **h=20 from one**.
The largest-error horizon receives a quarter of h=5's gradient. Validation uses fixed years
and is balanced, so **nothing in `val_*` reveals this**. Compensating with loss weights was
tried and does not help — h=20 is intrinsically harder and all four heads share one trunk —
but the imbalance should be known when reading per-horizon numbers.

**(c) Checkpoint selection is close to a lottery.** The selected epoch ranged from 17 to 145
across five runs of the same configuration. It barely matters — epoch 17 and epoch 75 give
the same regional error — but only because **the model saturates within about five epochs**:
neither the training nor the validation curve trends afterwards. Tripling the budget to 450
epochs changes nothing measurable. The 150-epoch schedule is therefore ~145 epochs of
sampling noise, and `save_top_k=1` picks the luckiest draw on 34 validation chips.

**(d) Validation runs on 34 chips.** `--val_stride 2048` was chosen for speed. It is enough
to rank epochs roughly and not enough to distinguish good ones.

**(e) Training is not deterministic at a fixed seed.** Two runs with identical flags and
`--seed 42` differ in 54 of 62 tensors after three epochs (worst 2.1e-4), from cuDNN
algorithm selection. Over ~2,000 steps this compounds into visibly different predictions. Any
"is this change inert?" check must therefore be **distributional**, never bit-exact.

### 1.5 Fundamental issues that bear on validity

These are more serious — they constrain what the numbers can be claimed to mean.

**(a) Run-to-run variance exceeds every effect we have measured since the central-field
change.** Two k=5 trainings of one configuration — five folds, 4.36M pixels, identical
pipeline — differ by **7 scorecard rows out of 128**. Twenty-two model experiments were run
in a dedicated phase and none produced an effect larger than that band, which is why none
was adopted. The large historical gains (+18 and +11 rows) comfortably clear it and stand.
**But any future comparison inside ±7 rows requires replicate training runs to mean
anything**, and every scorecard number in this project to date is a single run.

**(b) Every h=20 number is in-sample in time.** Only the window ending in 2000 reaches
+20 yr within the observed record (2020). So the +20 yr horizon is validated on held-out
*geography* but never on held-out *time*. It is the horizon we care most about for a 2040
forecast and the one with the weakest validation.

**(c) Findings measured on southern Africa are provisional.** Southern Africa changes 2–4×
less than Africa at every HM level, and its near-pristine `[0, 0.01)` stratum is 6% of the
region against 40% of Africa. Measured directly: the interval-width error the post-hoc layer
exists to correct is roughly **twice as large on southern Africa as on Africa** — so a
substantial part of that machinery is fixing a regional artifact. Conversely the central
model is *more* skilful on Africa (skill 0.130/0.201/0.234/0.231 against
0.085/0.148/0.184/0.191) and its raw coverage is much closer to nominal (0.957–0.971 against
0.978–0.985).

**(d) There is no production model.** Every checkpoint carrying these improvements was
trained with a fold held out — they are hindcast instruments and cannot forecast forward. The
only artifact that has ever made global forward predictions is `artifacts/model-khrpthgy:v0`
(October 2025), which carries the **old** central head — the configuration measured at skill
−1.23 at h=5. See §4.3.

**(e) A production model cannot be validated directly.** It trains on everything, so no
held-out data exists for it. You validate the *configuration* via k-fold and then trust the
draw — which, given (a), carries an unmeasurable ±7-row lottery.

---

## 2. Ensembles

Everything in this section is new on this branch. `main` has none of it.

### 2.1 What we are trying to build, in plain terms

We want *M* maps. Each map must satisfy two things at once:

1. **At each pixel, it must be drawn from the right distribution.** If the model says a pixel
   is 0.30 with a 95% interval of [0.25, 0.42], then across the members that pixel should sit
   at 0.30 on average, be below 0.25 about 2.5% of the time and above 0.42 about 2.5% of the
   time. This is the **marginal** — the per-pixel distribution.

2. **Neighbouring pixels must be wrong together.** Real forecast errors are spatially
   organised: if the model under-predicts a city's expansion, it under-predicts it across the
   whole city, not at every second pixel. This is the **spatial correlation**.

The standard tool for building something that satisfies both is a **copula**, which is a
fancy name for a simple two-step recipe:

> **Step A.** Generate a spatially correlated random field of *standard normal* values —
> a map of numbers that are individually N(0,1) but correlated with their neighbours in the
> right way. Call this field **z**.
>
> **Step B.** At every pixel, feed that pixel's *z* through that pixel's own distribution to
> get an HM value.

Step A carries all the spatial structure. Step B carries all the per-pixel calibration. They
are completely separable, which is what makes the whole thing tractable — and it means we can
fix a marginal problem without touching the spatial structure, and vice versa.

An analogy: think of *z* as a map of "how unlucky is this pixel today", drawn so that nearby
places have similar luck. Step B translates each pixel's luck into an HM value using that
pixel's own uncertainty. Same luck map, different pixels, different outcomes.

### 2.2 Step 0 — residuals: measuring what the model actually gets wrong

Everything downstream is estimated from **hindcast residuals**: `observed − central`, on
predictions that are genuinely out-of-sample (§4.1). For each (window × horizon) we store the
raw residual, the standardised residual, the predicted change, HM at t₀, the two half-widths,
and the distance-to-past-change covariate.

The **standardised residual** is the key quantity:

```
e = (observed − central) / (half-width / 1.96)
```

using the upper half-width when the residual is positive and the lower when negative. If the
published interval were exactly right, `e` would look like a standard normal: about 95% of
values between −1.96 and +1.96. Departures from that are exactly what the rest of the machinery
corrects.

### 2.3 Step 1 — recalibration (and why it ends up doing nothing)

Classic conformal recalibration: measure coverage on held-out data, and if the interval is
too narrow or too wide, multiply it by a factor. We implemented Mondrian split conformal with
hierarchical shrinkage (cell → stratum → horizon), isotonic smoothing, and monotone spread
across horizons, decided by **held-out interval score** rather than coverage.

**The decision has come back `identity` — no rescaling — in every k=5 run.** Held-out interval
score: identity 0.09667, global 0.09670, stratified 0.09790. That is a real result: once the
central field was fixed, the heads no longer needed a global rescale. Coverage alone would
have chosen differently, because coverage always prefers the widest interval; the interval
score penalises width and is why the decision rule was changed.

### 2.4 Step 2 — per-class half-width factors

Although no *global* rescale is needed, the interval is wrong by different amounts in
different places. `fit_width_factors.py` computes, for each class, the multiplier that would
make the published interval exactly the residual's own 95% interval — call it `k`. `k = 1`
means the model needs no correction. Classes are nested on three axes, each shrunk toward its
parent by `n/(n + 2000)`:

```
distance band  →  predicted change  →  HM level
```

All three carry signal: within one distance band at h=20 the factor runs 0.75 for predicted
change in (0.01, 0.05] against 0.95 for (0.05, 0.15]; and within a fixed (band × change) cell
it still varies 1.4–81× across HM level.

Two hard-won details:

- **Only the near bands are corrected** (0–1, 1–3, 3–10 px). Narrowing the far field moves
  the +0.05 threshold from 3.5 to 11.5 shape units and makes it unreachable; leaving it alone
  scores better.
- **The factor must not be centred on the residual median.** A width is centred on the
  *central forecast*, so it must cover the residual's bias as well as its spread. Centring it
  (which is mandatory for the *shape*, §2.5) once read a class's 2.5 percentile as −1.28
  instead of −2.13, narrowed an already-tight bound, and dropped class coverage 0.95 → 0.73.

### 2.5 Step 3 — the marginal: what distribution sits at each pixel

The simplest choice that honours the published triple is a **median-spliced two-piece
normal**: a normal distribution with one standard deviation below the median and a different
one above, chosen so the 2.5th percentile lands exactly on the published lower bound, the
median exactly on the central forecast, and the 97.5th exactly on the upper bound. It is
closed-form and exact.

Its weakness is shape. Measured in units of the published half-width, the real residual is
far more concentrated in the middle and far heavier in the tails than any normal:

| | P(\|e\|>0.25) | P(\|e\|>0.5) | P(\|e\|>1) | kurtosis |
|---|---|---|---|---|
| Gaussian (what the two-piece assumes) | 0.803 | 0.617 | 0.317 | 3 |
| observed, h=5 | 0.342 | 0.194 | 0.088 | **20366** |
| observed, h=20 | 0.492 | 0.254 | 0.082 | **928** |

So we replace the *shape* while keeping the three anchor points exact. `fit_residual_shape`
builds the residual's own standardised quantile function on a 512-knot grid and normalises it
so that `u = 0.025 / 0.5 / 0.975` still map precisely to lower / central / upper. Because the
map is strictly increasing, it **cannot change the spatial structure at all** — it preserves
the rank order of every pixel, so the correlation field built in step A survives untouched. A
genuinely Gaussian residual returns the identity map.

Two things learned the hard way:

- **The three gate quantiles must be exact knots.** Interpolating near them leaves a bias
  that does not shrink with more members.
- **Beyond the 95% bound, revert to the two-piece normal.** The empirical tail out there is
  dominated by pixels whose *half-width* is near-degenerate, so the ratio explodes for reasons
  about the denominator. Taking it at face value once sent `z = 3` to 5.4 half-widths.

**The tail bound.** How far out the empirical shape is trusted before reverting is a knob
(`u_bound`), and it turned out to matter more than any other marginal choice. The settled
configuration is **asymmetric and band-dependent**: upper bound 0.999 out to 100 px, 0.975
beyond it, lower bound 0.025 everywhere. The reason is that the two tails fail in opposite
directions — the upper is far too thin, the lower already 3–28× too hot — so one symmetric
knob cannot serve both.

**A structural limit worth stating plainly.** Because the shape normalises each side by that
side's own 2.5/97.5 quantile — which is exactly what makes the bounds exact — the fitted
marginal is *the residual's distribution stretched to fill the published interval*. Measured
stretch at h=20: 1.3–3.3× up, 1.3–7.1× down. **So any marginal that preserves the published
bounds necessarily re-injects whatever width error those bounds carry.** No shape can fix a
width problem; that is what §2.4 is for, and it is why the post-hoc lever is considered
exhausted.

### 2.6 Step 4 — the spatial field, explained from scratch

This is the part that makes members look like maps rather than static.

#### What we need

A map of standard normal values where the correlation between two pixels depends on how far
apart they are. Close together → similar values. Far apart → unrelated.

#### Why you cannot just add noise

If you draw an independent N(0,1) at every pixel, the map looks like television static and
every aggregate is far too certain, because independent errors cancel when you average them.
Real error maps have patches. The whole point is to get the patch structure right.

#### Describing the structure: the power spectrum

Two standard ways to describe how a field varies with distance:

- **The variogram** — average squared difference between pairs of pixels a given distance
  apart, plotted against distance.
- **The power spectrum** — how much of the field's variance sits at each spatial *frequency*.
  Low frequency = big smooth blobs. High frequency = fine speckle.

We fit the **spectrum**, not the variogram, and that was a deliberate correction. A variogram
is evaluated at lags, and two long-range components can absorb the fit while leaving the
middle of the range empty. Measured against the real residual field, the variogram-fitted
mixture carried **4.9× too much power beyond 50 px and only 0.47× what it should at 3–10 px**
— the band holding 47% of the observed variance. Visually: smooth blobs where the truth is
structured speckle. Fitting the radial spectrum instead brings those to 0.94× and 0.88×.

The fit is a non-negative least squares mixture over a fixed basis of correlation ranges
(2, 5, 12, 30, 80, 200 px), so the result is "x% of the variance at ~2 px scale, y% at ~5 px,
…" plus a nugget (the purely pixel-level part).

#### The kernel: Matérn, not Gaussian

The correlation function's shape matters as much as its range. A Gaussian kernel produces
*infinitely smooth* fields — soft blobs. Real error fields have rough, textured edges. The
**Matérn** family has a smoothness parameter `nu`; at `nu = 0.5` it is the exponential
kernel, which is rough at short range. Swept over 0.5–1.5, `nu = 0.5` fits best. In short:
**Gaussian kernels cannot make texture.**

#### Building the field: circulant embedding

Generating a correlated field naively means factorising an N×N covariance matrix, where N is
the number of pixels — completely impossible at 63 million pixels.

Circulant embedding avoids it with a fact from Fourier analysis: **if a field's correlation
depends only on the separation between points, then in frequency space the different
frequencies are independent.** So instead of a huge correlated draw you can do:

1. Compute the target covariance's spectrum `S` — one number per frequency.
2. Draw independent white noise (easy — that is just N independent normals).
3. Fourier-transform the noise, **multiply each frequency by `sqrt(S)`**, transform back.

The result has exactly the covariance you asked for, and costs two FFTs. Multiplying by
`sqrt(S)` is "turn up the frequencies that should carry variance and turn down the ones that
should not" — the map comes out with the right mixture of large and small structures.

Two practical details: the grid is **padded** before the transform, because an FFT implicitly
treats the map as wrapping around, which would otherwise correlate the left edge with the
right; and for global runs `--wrap_lon` deliberately re-enables the wrap in longitude, where
it is physically correct.

#### The long-range component, and why it is added by hand

A power spectrum computed from a mean-subtracted field is **blind to frequency zero** — the
overall level. But frequency zero is precisely what controls whether a *whole region's*
average is uncertain. Fit the spectrum alone and every large aggregate comes out too
confident.

So a deliberate long-range component (range 1500 px) is added at weight `long_weight = 0.40`,
calibrated from aggregate error rather than from the spectrum. Swept over 0.15–0.70 it moves
essentially one thing — the spread-skill ratio (0.82 → 1.52, monotone) — so it is identifiable
from that metric alone. 0.40 is the settled value.

### 2.7 Step 5 — from field to member

For each member and horizon:

1. Draw a correlated standard-normal field **z** as above.
2. At each pixel, convert `z` to a probability and push it through that pixel's marginal —
   two-piece normal, reshaped by the fitted empirical shape, with the shape selected by that
   pixel's distance band.
3. Clip to [0, 1] and apply the model's validity mask.

Because the shape is a strictly increasing function, step 2 changes *values* but not *ranks*
— the spatial structure built in step 1 passes through unaltered.

### 2.8 Step 6 — coupling the horizons

The four horizons must not be independent. A member that shows a city expanding fast by 2005
should still show it expanding by 2020. The horizons are chained by an **AR(1)** process on
the normal-score fields:

```
z(h) = ρ · z(h−1) + sqrt(1 − ρ²) · ε(h)
```

with ρ estimated from the residuals' own between-horizon correlation (falling back to 0.9).
This keeps each `z(h)` marginally standard normal while making consecutive horizons
correlated. Measured result: between-horizon correlation 0.724/0.939/0.916 against an
observed 0.737/0.943/0.922.

### 2.9 Step 7 — storage and reproducibility

Members are stored as **int16** with scale 1/32767 — HM is bounded on [0, 1], so this is
lossless to about 3e-5, far below any quantity of interest. An M=400 ensemble is ~3 GB for
southern Africa; Africa is 33× larger, so large artifacts live on `/mnt/hdd1` behind
symlinks.

**Every member's random seed is recorded in the manifest**, so any ensemble regenerates
exactly, and a superseded one can be deleted and rebuilt in about three minutes.

### 2.10 The two guarantees

Two properties hold by construction and are asserted as hard gates:

- **T5.1** — the ensemble's median equals the published central forecast.
- **T5.2** — the ensemble's 2.5th and 97.5th percentiles equal the published bounds.

That is what makes the ensemble a *representation* of the published product rather than a
second, competing product. Whatever the ensemble says, the headline numbers are unchanged.

---

## 3. The Scorecard

`scripts/validate_ensemble.py` scores the generated ensemble against eight families of
targets and writes a pass/fail row per (target × horizon × scale), with a diagnosis attached
to each failure saying which knob would move it.

**Current state: Africa, k=5, M=400 — 111 of 133 scoreable rows pass.** The southern-Africa
run of the same chain, also at M=400, reads 101/127 and is kept below as the regional
comparison. Fourteen rows are **reported, not scored** (T2.6, T3.2b, T3.4, T4.1, T7.1) —
they produce numbers or figures for inspection but have no pass gate.

The two regions are not interchangeable, and the differences are informative in both
directions. Africa has **158 ecoregions against southern Africa's 18**, which turns several
aggregate targets from undecidable into real tests; it also has enough remote country to
populate the far distance bands, which exposes a failure the smaller extent could not show.
Read the families, not the totals.

| family | Africa | s. Africa | what moved |
|---|---|---|---|
| T1 pixel-scale coverage | 29/33 | 29/31 | pooled coverage flips from over- to under-covering |
| T2 aggregate coverage | **37/41** | 31/41 | ecoregion rows become decidable at n=158 |
| T3 spatial realism | 3/6 | 3/6 | unchanged — the correlation structure is regionally invariant |
| T4 temporal coherence | 0/1 | 0/1 | monotone spread worse on Africa (0.712 vs 0.937) |
| T5 hard gates | 11/12 | 12/12 | h=20 median gate misses by 0.0015 |
| T6 change realism | **19/24** | 14/22 | change quantiles 7/8 against 3/8 |
| T7 diversity / spread-skill | 1/2 | 1/2 | unchanged |
| T8 change placement | 11/14 | 11/12 | two new far-band failures Africa can see and s. Africa cannot |

### T1 — Is each pixel's uncertainty right? (29/33)

| row | plain meaning | result |
|---|---|---|
| **T1.1** pooled coverage | Of all pixels, does the truth fall inside the 95% interval 95% of the time? | 2/4 — 0.944 / 0.938 / 0.942 / 0.939, mildly **under** at every horizon |
| **T1.2** class-conditional coverage | Same question but *within* each class of pixel (by distance to past change, by predicted change, by HM level). Catches the case where being right on average hides being wrong everywhere in compensating directions. | 14/16 — worst cell deviation 0.033 against a 0.05 allowance |
| **T1.3** high-change tail | Coverage specifically on the rare pixels where a lot of change is predicted — the ones the project exists to get right. | **3/3** (0.966 / 0.965 / 0.950) |
| **T1.5** sharpness | Are the intervals no wider than they need to be? An interval can always be made to cover by being useless. | **10/10** — width ratios 0.81–1.09 against the original heads |

**The direction reversed between regions.** On southern Africa T1.1 was mildly *over*-covering
(0.963 at h=20); on Africa it is mildly *under* at all four horizons. The width machinery was
fitted on southern Africa, and about half of it turns out to be compensating for that
region's own error structure rather than the model's.

**Why T1.2 matters more than T1.1.** Pooled coverage of 0.966 once coexisted with 0.335 in the
high-change class. Averages hide exactly the failures you care about.

### T2 — Is uncertainty right at *larger* scales? (37/41)

This is the motivating problem of the whole project, and Africa is where it can finally be
measured properly.

| row | plain meaning | result |
|---|---|---|
| **T2.1** block coverage | Aggregate HM over 1 km, 10 km and 100 km blocks. Does the truth fall inside the ensemble's range 95% of the time at *every* scale? | **12/12** (0.924–0.978) |
| **T2.8** aggregate width | Is the aggregate interval narrower than naively averaging the per-pixel bounds — i.e. is the correlation actually buying anything? | **12/12** |
| **T2.5** rank histogram | Where does the truth rank among the members? A flat histogram means calibrated; a U-shape means too narrow, a hump too wide. | 3/4 — 2005 fails at p = 1.2e-6 |
| **T2.2 / T2.3 / T2.4** ecoregion coverage | Same test over real ecological regions rather than square blocks. | 2/4, **7/8**, **1/1** |

**T2.1 at 12/12 is the headline result of the project**, and it now holds on a continent as
well as a region. Aggregate coverage is correct at every block scale, which is precisely what
per-pixel intervals could not deliver.

**The ecoregion rows were a resolution problem, and Africa proves it.** With 18 units the
coverage can only take values k/18, so 18/18 = 1.000 fails a ±0.05 target while 17/18 = 0.944
passes — the metric was quantised too coarsely to land inside its own tolerance, and every
southern-Africa failure sat at exactly 1.000. Africa's 158 units give values of 0.981–0.994
and the family goes 3/8 → 7/8, with T2.4 (ecoregion mean *change* between horizons, the
hardest case) passing for the first time. The remaining T2.2 failures are still over-coverage.

**T2.5 is the mirror image: a test that gained power and started failing.** Southern Africa
returned p = 0.72 three times of four at n=18 — no power. At n=158 the 2005 rank histogram
fails decisively. That is a stronger test failing, not a worse model.

### T3 — Do the members look like real maps? (3/6)

| row | plain meaning | result |
|---|---|---|
| **T3.1** member variogram | Does a member's spatial structure match the structure fitted to the real residuals? | 1/3 — normal-score variance 1.151 (target 1.0 ± 0.15), practical range off by 0.148 (**passes**), nugget by 0.103 against a 0.10 allowance |
| **T3.2** vs an independent null | Compare against an ensemble with identical marginals but *no* spatial structure. The correlated one should be markedly better. | **fails** (0.031 against a ≥0.30 target) |
| **T3.3** energy score | A proper multivariate score against the same null. | **passes** |
| **T3.5** seam continuity | No visible discontinuity at the longitude wrap. | **passes** (regional grid) |

**T3 is the one family that does not move between regions** — 3/6 on both, and T3.2 reads
0.031 on Africa against 0.021 on southern Africa. The correlation structure is the live
defect and it is regionally invariant, which makes it the one thing southern-Africa iteration
was *not* flattering.

**T3.2's target is the problem, not the field, and this is measured.** An ensemble calibrated
to the residual can only differ from an independent one in the variance still *correlated* at
the separation being scored. Read off the residual's own variogram, **86% of its variance is
already decorrelated by 25 px**, leaving a 14% structure budget — so a 30% improvement asks
the ensemble to be better structured than the data it is calibrated to. Reaching it would mean
generating a field markedly smoother than reality, which is exactly what **T3.4 exists to
catch**. T3.2 and T3.4 are in direct conflict and T3.4 is the one grounded in data. The
threshold should be re-scoped rather than tuned toward.

### T4 — Are the four horizons coherent? (0/1 scored)

| row | plain meaning | result |
|---|---|---|
| **T4.1** between-horizon correlation | Members should carry their story forward in time. | reported: 0.900 / 0.901 / 0.900 — the AR(1) target file does not exist for this configuration, so ρ fell back to 0.9 and the row is not gated |
| **T4.2** monotone spread | Uncertainty should not *shrink* as you forecast further ahead. | 0.712 against a ≥0.99 gate (southern Africa: 0.937) |

T4.2 is the one real cost of raising the marginal's tail bound: extending the upper tail adds
spread at every horizon, and at h=5 that buys nothing while perturbing the ordering. It is
markedly worse on Africa. A bound that varies by horizon as well as by band is the obvious
next move.

### T5 — Hard gates: is the ensemble faithful to the published product? (11/12)

| row | plain meaning | result |
|---|---|---|
| **T5.1** | ensemble median == published central forecast | 3/4 — 0.9956 / 0.9989 / 0.9975 / **0.9935** against a ≥0.995 gate |
| **T5.2** | ensemble 2.5/97.5 percentiles == published bounds | **4/4** (0.986–0.992) |
| **T5.3** | ensemble validity mask == model validity mask | **4/4**, zero mismatched pixels |

These are non-negotiable. If they fail, the ensemble is telling a different story from the
published maps. Their tolerances are Monte-Carlo scaled (they tighten as 1/√M) and
**family-aware**: the standard error of a sample quantile depends on the density at that
quantile, which for a reshaped marginal is not the same as for a normal. Ignoring that made
the tolerance 1.3–2.2× too tight at the bounds and up to 3.7× too loose at the median.

**The member count is visible in this family.** At M=100 the same Africa ensemble scored T5.1
at 0.9923 / 0.9976 / 0.9968 / 0.9921 — 2/4. At M=400 it is 3/4, with h=20 missing by 0.0015.
Nothing changed but the number of draws.

### T6 — Is the *amount* and *sign* of change realistic? (19/24)

| row | plain meaning | result |
|---|---|---|
| **T6.1** | Does a member show decreases of >0.01 as often as reality? | 0/4 — over-predicted 2.5–4.5× |
| **T6.2 / T6.3** | Same for larger decreases (>0.05, >0.15). | **4/4, 4/4** |
| **T6.4** | Is the asymmetry between increases and decreases right? | **4/4** |
| **T6.5** | Do member change quantiles match observed ones? | **7/8** — ratios 1.01–1.86, only h=20 q05 misses at 4.25 |

**T6.5 is the clearest place Africa is genuinely better**, 7/8 against southern Africa's 3/8,
with ratios of 1.01–1.86 against 1.11–5.41. On a fairer HM distribution the members' change
distribution matches observation well.

T6.1 is the counterweight and it is worse on Africa. HM decreases are real but rare, and the
ensemble manufactures too many small ones. Most of this is not fixable by any per-pixel
marginal: the ideal calculation — using each band's own empirical distribution — still
over-predicts by 3.05×, so it reflects a dependence between the standardised residual and the
pixel's own threshold, which a marginal by definition cannot encode.

### T7 — Are the members diverse but not absurd? (1/2 scored)

| row | plain meaning | result |
|---|---|---|
| **T7.2** member diversity | Members must not be near-copies of each other. | **passes** (0.256 mean pairwise correlation) |
| **T7.3** spread-skill ratio | Where the ensemble is uncertain, the model should actually be more wrong. A ratio of 1 means spread predicts error correctly. | 1.891 against 1.0 ± 0.25 (southern Africa: 1.583) |

T7.3 above 1 means the ensemble is over-spread relative to the error it needs to explain, and
it is worse on Africa. Lowering `long_weight` fixes it but pulls near-nominal T2.3 rows below
target, so it is recorded as a known near-miss with an undesirable remedy.

### T8 — Is change put in the *right places*? (11/14)

This family is the geographic sanity check, it is where the project's largest failure
originally lived, and it is where Africa's extra area buys the most information.

| row | plain meaning | result |
|---|---|---|
| **T8.1** | Probability of >0.05 increase, per distance-to-past-change band, against observed. | 5/6 — bands 1–5 at 1.06–1.36, the **>100 px band at 0.107**, i.e. ten times too *little* |
| **T8.2** | Change invented in the remote band. | **passes** — 1.4e-06 against a ≤0.002 allowance |
| **T8.3** | Ratio of near-field to remote-field change. | **passes** — 2.3e+05 against ≥20 |
| **T8.4** | Same for decreases. | 4/6 — the 30–100 px band **7.8×** and the >100 px band **123×** the observed rate |

**T8.2 and T8.3 remain the clearest single improvement in the project.** The original heads
assigned a 3% chance of substantial change to country where change has never been observed;
the remote band now receives essentially none.

**But Africa exposes the opposite error on the same axis.** Southern Africa passes T8.4 5/5
because it has too little remote country to populate the far bands — its remote observed
change is exactly zero, so the rows are trivial. Africa has real far-field observations, and
there the ensemble produces far too many *decreases* (7.8× and 123×) while producing too few
*increases* (0.107×). Both are invisible at the smaller extent. This is the strongest single
argument for evaluating on the continent rather than the region.

### How to read the scorecard honestly

- **Read rows, not the total.** Two settings once tied on count while one had quietly pushed
  near-nominal rows below target.
- **Member count is part of the measurement.** Tolerances tighten as 1/√M, so comparisons
  must be at matched M. An M=100 comparison between two marginal families once *reversed* at
  M=400; a 101/127 at M=400 became 99/127 at M=800. On Africa, going M=100 → M=400 with
  nothing else changed moved T5.1 from 2/4 to 3/4 and T1.1 from 0/4 to 2/4.
- **Region is part of the measurement too, and not only through difficulty.** Several
  aggregate targets are limited by the *number of aggregation units*, not by the field: with
  18 ecoregions a ±0.05 coverage target is undecidable, and rows can fail on quantisation
  alone. Compare regions per family, never on the total.
- **A single scorecard has ±7 rows of run-to-run noise** (§1.5a). Differences smaller than
  that are not interpretable without replicate training runs.

---

## 4. Evaluation and Forecasting

### 4.1 How hindcasts are generated

A hindcast is a forecast for a date that has already happened, so it can be checked. But a
model must not be scored on data it trained on, and HM only exists from 1990 to 2020.

The solution is **five-fold cross-validation over geography**. The globe is divided into five
spatial folds. Five models are trained; each excludes one fold from training *and* uses
another as its validation split. Each model then predicts only the fold it never saw, and the
five predictions are stitched into one raster.

**The result is a complete map that is out-of-sample at every pixel** — 184.6M pixels
globally, 4.36M for southern Africa, 142.6M for Africa. This is what makes the residuals
honest, and every downstream statistic depends on it.

#### The seam this creates, and the recipe for it

Holding geography out has a visible consequence. The fold mask is a **128 px checkerboard** —
`create_kfold_splits` assigns each chip independently — so adjacent tiles of the stitched map
come from *different models*, joined with no blending. Wherever those models disagree, the
join shows.

They disagree on one head only. Measured on Africa at +20 yr, mean pairwise
|fold_i − fold_j| is 0.0053 for the central field and 0.0040 for the lower bound, against
**0.0383 for the upper bound**. The step across a fold boundary is 1.08× the step inside a
fold for central and lower, and **2.01× for the upper** — 47× the local background in quiet
country, where nothing else is happening. A single fold model's own large-area prediction is
seamless at every period from 64 to 1024 px, so neither the ConvLSTM nor the overlap blending
is involved: it is purely the mosaic.

**The recipe, deliberately split in two:**

- **Products and display rasters are stitched as the fold mean** (`--stitch_mode mean`):
  every fold averaged at every pixel. There is no mosaic and no seam — the upper bound's
  fold-boundary ratio goes 31.9× → 1.9× after recalibration, and an ensemble member 2.5× →
  1.1×. This mixes in-sample and out-of-sample predictions by construction.
- **Everything scored stays on the holdout mosaic** (`--stitch_mode holdout`, the default).
  A fold-mean raster is in-sample at every pixel and would score flatteringly.

The forward 2025–2040 forecast has no seam at all: nothing is held out, so no mosaic is built.

#### What was considered and not adopted

Two ways of removing the seam *within* the scored product were investigated and rejected on
measurement, not on cost alone:

- **Repeated k-fold CV** (each chip predicted out-of-fold R times from models trained on
  different subsets, then averaged). It reduces the seam by √R — at R=5 the upper bound goes
  from 2.01× to about 1.45×, and quiet country from 47× to ~21×. It does not remove it, and
  it costs 5× the training budget. It also changes what is being scored: an average of R
  out-of-fold predictions has lower error than any single model, so unless the average is
  also what ships, the hindcast becomes optimistic in a way that does not transfer.
- **Repeated seeds at a fixed fold mask.** Cheaper to reason about and still strictly out of
  sample, but it only averages the training-noise component and delivers 1.24–1.76× at R=5.

To decide between them, fold 1 was retrained twice with only `--seed` changed. On the upper
half-width the seed-to-seed spread is 0.0308 against a fold-to-fold 0.0335 at h=20; in
variance terms training noise is 85% of the disagreement at h=20 and 44–63% at shorter
horizons. So both schemes are largely averaging away *optimisation noise* rather than genuine
model diversity — which points at the training recipe as the cheaper lever. Switching the
checkpoint monitor to `val_central_loss` alone already cuts the fold-to-fold upper-width
spread from 0.0335 to 0.0275, an 18% reduction at no extra cost, and `--weight_avg_last`
exists and is untested for this purpose. Full detail in `docs/validator_scaling.md` §8.

A third option — **coarser fold blocks** (`--fold_block_chips 10`, giving contiguous 1280 px
fold territories instead of a 128 px checkerboard) — is implemented but not trained. It cuts
seam *density* about 12×, so a typical viewing window sits inside one fold, though the step at
the seams that remain is unchanged. Its real argument is not cosmetic: the residual field's
fitted practical range is 99–166 px, typically ~130, against a 128 px fold tile, so the
current split holds geography out at **one correlation length** and the held-out skill is
therefore optimistic. Adopting it means retraining all five folds and makes every existing
scorecard non-comparable.

Each model predicts four input windows (ending 2000, 2005, 2010, 2015) at four horizons,
giving ten (window × horizon) pairs within the observed record.

**This is entirely new on this branch.** `main` has a single train/val/test split, no fold
rotation, and therefore no way to produce an out-of-sample map.

### 4.2 The evaluation loop

`scripts/run_region_loop.sh` runs the whole chain for one configuration:

```
stitched predictions
  → central-field diagnostics    (stratified error, the cheapest signal)
  → residuals
  → coverage / variograms / class audit / recalibration decision
  → per-class half-width factors  → rebuild residuals against them
  → field spectrum
  → marginal shape
  → M members + an independent-pixel null
  → T1–T8 scorecard
```

**Everything downstream of the model is re-derived from that model's own residuals** — the
recalibration, the spectrum, the AR(1) coupling, the width factors and the marginal shape.
Carrying any of them over from another configuration scores a new central field through an old
model's error structure, which is the single most productive source of convincing wrong
numbers in this project.

Beyond the scorecard, two diagnostics carry particular weight:

- **`member_distance_relationship.py` — the per-member test.** Pooling all members together
  answers a weaker question than it appears to: a pool that is uniformly too hot still
  overlaps the truth. The honest test asks, for each member separately, where the observation
  *ranks* among the members. Ranks pinned at 0 or M are the specific failure. Currently 15 of
  20 (year × band) cells have the observation inside the member 5–95% range, up from 7/20.
- **`predict_change_rates.py` — a closed-form second implementation.** The member *mean* of
  P(Δ > threshold) is available analytically, because the spatial field changes how much
  members scatter around that mean, not the mean itself. It runs in 30 s against ~35 min for
  generate-and-validate, and it shares no code with the GPU sampler — so agreement is evidence
  and disagreement localises a bug to the generation path. It has agreed with sampled
  ensembles to 1–11% every time it was checked.

### 4.3 Preparing for a global 2025–2040 forecast

This is the step that has **not** been taken, and it is important to be precise about why.

**A hindcast model cannot forecast forward.** Each fold model deliberately never saw one
fifth of the world. Their value is that their predictions are out-of-sample; that same
property makes them unusable for production, where you want a model that has seen everything.

So two artifacts are needed, and they are different training runs:

| | purpose | fold exclusion | exists? |
|---|---|---|---|
| **k=5 hindcast** | out-of-sample residuals → every ensemble statistic | one fold held out per model | **yes** — southern Africa and Africa |
| **production model** | the forecast itself | **none** | **no** |

They must be the **same configuration**, or you apply one model's error structure to another
model's predictions.

**The forward path**, once the production model exists:

1. Train with the reference flags and **no** `--exclude_fold`, on the full production split.
   Chips are sampled globally either way, so this is the same model in every respect except
   that it uses all the geography.
2. Predict with inputs 2010 / 2015 / 2020 → targets **2025 / 2030 / 2035 / 2040**. The +5 /
   +10 / +15 / +20 horizon structure is identical to the hindcast, so the residual statistics
   transfer directly. All required inputs — HM 2010/2015/2020 and `change_context_w2020` —
   exist.
3. Apply the hindcast-derived statistics to those forward bounds:
   `apply_recalibration.py --targets production --production_years 2025,2030,2035,2040`.
4. Generate the forward ensemble with the same spectrum, shape and AR(1) coupling.

**The current state of production is the gap to close.** The only artifact that has ever made
global forward predictions is `artifacts/model-khrpthgy:v0` from October 2025 — old central
head, old quantile heads, no context covariate — and `data/predictions/prediction_2025..2040_*`
was made with it. That is the configuration measured at skill −1.23 at h=5, i.e. losing to
"nothing will change". **Applying the new ensemble statistics to those old predictions would
be exactly the error §4.2 warns about.**

### 4.4 Summary of differences to `main`

| | `main` | this branch |
|---|---|---|
| central head | predicts absolute HM | predicts **change** on top of HM at t₀ |
| long-range covariate | none | precomputed distance-to-past-change, 8 channels, to **both** head families |
| interval construction | three independent predictions | half-widths accumulating around the central forecast; monotone in lead time by construction |
| validation geography | grid mode ignored the split mask | fixed; fold-aware |
| out-of-sample maps | not possible | k=5 fold rotation |
| uncertainty product | per-pixel intervals only | + M-member ensemble with fitted spatial structure |
| recalibration | none | conformal, decided on held-out interval score (currently identity) |
| per-class width correction | none | three-axis factors, shrunk hierarchically |
| marginal | n/a | two-piece normal reshaped by the residual's own quantile function, asymmetric band-dependent tail bound |
| spatial structure | n/a | Matérn (`nu=0.5`) spectral mixture by circulant embedding + a calibrated long-range term |
| horizon coupling | n/a | AR(1) on the normal-score fields |
| validation | loss curves | T1–T8 scorecard, per-member rank diagnostics, closed-form cross-check |
| skill vs persistence | **negative at +5 and +10 yr** | positive at all four horizons |
