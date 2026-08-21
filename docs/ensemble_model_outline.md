# The Ensemble Model — What We Built and Why

Branch `ensemble`, written 2026-08-17, substantially revised 2026-08-21. This describes the
whole approach as it now stands. It is written to be read by someone who knows the project
but not necessarily the statistics of spatial simulation: §2 explains the machinery from
first principles, and §3 explains every row of the scorecard — what it measures, where its
target comes from, how the model is doing, and where it fails, why.

The reference run throughout is **`c1_foldb4`**: five folds on the 512 px fold mask, one
unified calibration layer, M = 400, scored on **Africa**. Its card is **107 of 136**.
`docs/improvement_plan.md` records how it was arrived at.

---

## 0. Orientation — what the product is, and what problem the ensemble solves

The model forecasts **Human Modification (HM)**, a 0–1 index on a 1 km grid, at four lead
times: +5, +10, +15 and +20 years. For every pixel it publishes three numbers per horizon:

- a **central forecast** — the best single estimate,
- a **lower bound** at the 2.5th percentile and an **upper bound** at the 97.5th,

which together are a 95% interval. That triple is the published product.

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

### 1.1 The starting point

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
Predicting absolute HM means the head must first reproduce HM at t₀ through the trunk before
it can add anything. That reproduction is not free: on the 53–70%
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
With the three heads predicting independently, nothing prevents the h=10
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

### 1.3 Where the central field stands

The right yardstick for a forecast is **skill against persistence** — against simply predicting
that nothing will change. That bar is higher than it sounds: the median 20-year HM change is
0.0001, so pooled RMSE can look unremarkable while the model is losing badly to doing nothing.

On Africa, k=5, scored on held-out geography:

| horizon | skill vs persistence |
|---|---|
| +5 yr | **+0.130** |
| +10 yr | **+0.196** |
| +15 yr | **+0.228** |
| +20 yr | **+0.222** |

Positive at every horizon and rising with lead time, which is the sensible ordering — the
further out you go, the more there is for a model to beat "nothing happens" at.

**These are honest held-out numbers, and that phrase is doing work.** The fold mask holds out
contiguous 512 px territories rather than a fine checkerboard, so 31.9% of held-out pixels sit
beyond one residual correlation length of any pixel their fold trained on. Under a finer mask
that figure is 0.0%, and skill measured there is optimistic — measurably so: within the
current mask, h=20 skill falls from +0.239 within 32 px of trained-on data to +0.188 beyond
192 px, a 21% relative decline that a finer mask cannot see at all.

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

Everything in this section is the ensemble machinery proper.

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

### 2.3 Step 1 — recalibration: one layer, and what it actually does

The quantile heads emit a lower and an upper bound. Nothing forces those bounds to be honest:
they can be systematically too narrow (the truth falls outside more than 5% of the time) or
too wide (it almost never does, and the interval is useless). Recalibration measures that on
held-out hindcast residuals and multiplies the **half-widths** by a correction factor. The
central forecast is never moved.

**One layer, three class axes.** Every pixel is sorted into a class, and each class carries
its own pair of multipliers `k_up` and `k_lo`:

```
distance to past change  ×  predicted change (Δ̂)  ×  HM level today
```

*Distance* because development happens near development, and the model's error out in
untouched country is a different animal from its error at a city edge. *Predicted change*
because a pixel where the model expects a lot of development has a different error
distribution from one where it expects none. *HM level* because error on already-built land
does not behave like error on wilderness.

`k = 1` means the heads were already right. `k = 0.5` means the published interval was twice
as wide as it needed to be; `k = 2` means half as wide. The two sides are fitted separately
because the error is skewed — there is far more room to be surprised upward than downward.

**How the factor is estimated.** For each class, the standardised residual
`e = (observed − central) / (half-width / 1.96)` is collected, and the factor is that class's
own 97.5th (or 2.5th) percentile expressed in half-width units. If the interval were exactly
right, `e` would look standard normal and the factor would come out at 1.

Four safeguards, each of which exists because leaving it out went wrong:

- **Not centred on the residual's median.** The marginal *shape* (§2.5) must be centred,
  because the median has to land exactly on the central forecast. A *width* must not: the
  interval is centred on the central forecast rather than on the residual's median, so it has
  to cover the model's bias as well as its spread. Copying the centring once read a class's
  2.5th percentile as −1.28 instead of −2.13 and dropped that class's coverage from 0.95 to
  0.73, with every miss on the low side.
- **Thin classes are shrunk toward their parent** by `n / (n + 2000)`, so a class with few
  pixels inherits rather than fitting noise. The artifact records which classes were fitted
  on their own pixels and which inherited, because a table of hundreds of classes otherwise
  looks far better resolved than it is.
- **A floor of 0.25**, so a degenerate class cannot collapse the interval to nothing.
- **Factors are made non-decreasing across horizons** within a class, so no class can claim
  more certainty at a longer lead time.

**Then two guards and a clip at apply time.** The lower bound is forced not to exceed the
centre and the upper not to fall below it, and both are clipped into [0, 1] because HM is an
index on that range.

**Finally, a per-pixel monotonicity pass.** After everything else, each pixel's half-widths
are made non-decreasing across horizons by a cumulative maximum. This is separate from the
constraint inside the fit, and it has to be: 2.6% of pixels **change class** along the horizon
sequence, because `Δ̂` grows with lead time, so they pick up a different factor at each horizon
and can invert even when every class is individually monotone.

The effect is decisive. Measured on the published bounds, monotonicity of the interval width
goes from **0.813 to 0.998**, and of the marginal's actual spread from 0.808 to **0.9996** —
which is what turns T4.2 (§3) from a failing row into a structural guarantee.

### 2.4 What a class-conditional factor really does — and why it is the far field's product feature

It is tempting to read the layer as "make intervals bigger or smaller". That is not what it
does, and the difference matters enough to be worth a section.

**Within one class the transformation is a pure scaling.** Every pixel in a given
`(band × Δ̂ × HM)` cell is multiplied by the same number, so that cell's median and its 99.9th
percentile move by exactly the same factor. Nothing is redistributed.

**But a distance band is not one class — it is a mixture**, and the other two axes split it.
The far band beyond 100 px at h = 20 is almost entirely one cell and a sliver of another:

| Δ̂ bin | HM bin | pixels | share of band | median half-width | p99.9 half-width | `k_up` |
|---|---|---|---|---|---|---|
| 1 | 1 | 23,041 | **0.61%** | 0.00372 | 0.00940 | **4.004** |
| 1 | 0 | 3,780,697 | **99.29%** | 0.00147 | 0.00412 | **0.760** |

The 99.3% is remote land at HM ≈ 0, where nothing can plausibly happen, and it is **narrowed
to 0.76×**. The 0.6% is remote land that already carries some development — whose intervals
are 2.5× wider to begin with — and it is **widened four-fold**. Those 23,041 pixels *are* the
far band's upper tail.

So at band level the same table produces:

```
p50   → 0.760      (the 99.3% class, narrowed)
p99   → 0.827
p99.9 → 3.977      (the 0.6% class, widened)
```

**The apparent within-band redistribution is really between-class selection.** The distance
band chooses which table of `(Δ̂ × HM)` factors applies; those two axes choose the multiplier;
and because width and class membership are correlated — the widest remote pixels are exactly
the ones with nonzero HM — a per-class constant lands very differently on the band's median
and on its tail.

**This is the mechanism behind the far-field result in §3's T8**, and it is fragile. A class
axis that mixes distances instead of separating them sits on top of both populations at once
and averages them together: keying the layer on *biome* rather than distance took the far
band's p99.9 half-width to 0.435× and `P(Δ>0.05)` beyond 100 px from 0.85 of observed down to
**0.04**, while the *average* far-field pixel got wider the whole time. Held-out interval
score preferred that arrangement by 0.8%, because the pixels it destroyed are a thousandth of
the band and a pooled average cannot see them.

**Which is the general warning.** The layer's job in the far field is not to widen or narrow
it. It is to *separate* the remote land where nothing can happen from the remote land where
something can. Any class definition that blurs that distinction will score well on average
and remove the product feature.

**What no calibration can do.** Both the width factor and the marginal correct different
things, and neither can substitute for the other. A width factor changes how wide the interval
is; it cannot change how probability is arranged inside it. That is §2.5's job — and because
the marginal is normalised so that 2.5%, 50% and 97.5% land exactly on the published bounds,
it re-injects whatever width error those bounds carry. **No shape can fix a width problem**,
which is precisely why this layer exists.

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

**The tail bound.** How far out the empirical shape is trusted before reverting to the
two-piece normal is a knob (`u_bound`), and it turned out to matter more than any other
marginal choice.

The settled configuration is now **0.999 on every band**, lower bound 0.025. It used to be
0.999 out to 100 px and **0.975 beyond**, on the reasoning that suppressing far-field change
was right — which it was, on the region that setting was tuned on, whose observed far-field
change rate is exactly 0.0000. On Africa and globally it is not: holding the remote band at
0.975 is what made the ensemble emit a ninth of the observed rate of new development in
remote country (§3, T8.1).

Swept on this model's own residuals, the pooled error `mean |log₁₀(predicted/observed)|`
across 24 (horizon × band) cells reads:

| tail bound | pooled error |
|---|---|
| two-piece normal (no shape) | 0.518 |
| 0.975 | 0.506 |
| 0.99 | 0.368 |
| **0.999** | **0.272** |
| 1.0 (no truncation at all) | 0.272 |

**0.999 and 1.0 are identical**, which is the important part: the lever is exhausted. Removing
the bound entirely adds nothing, so whatever far-field shortfall remains after this is not
reachable by reshaping a tail — it belongs to the model's width heads. The same sweep shows no
horizon axis worth exploiting either: the optimum is the same bound at every horizon and the
two candidates tie to the fourth decimal, so a per-horizon tail policy was measured and
dropped rather than built.

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

with ρ estimated from the residuals' own between-horizon correlation. This keeps each `z(h)`
marginally standard normal — which matters, because it means **ρ changes how the horizons
hang together without changing the spread at any one of them**. For `c1_foldb4`,
ρ = `{10: 0.650, 15: 0.461, 20: 0.433}` and the generator reproduces
0.658 / 0.475 / 0.441, within 0.014 everywhere.

**Measure it; do not let it fall back.** If `--rho_json` is missing the generator uses 0.9,
and that path was silently taken by *every* regional run in this project's history. The
fingerprint is unmistakable in hindsight — T4.1 reporting 0.900 / 0.901 / 0.900 against a NaN
target, three rows pinned to the constant they were fed and contributing nothing. Measuring ρ
is what turned T4 from 0/1 into 3/4 (§3).

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

`scripts/validate_ensemble.py` scores the ensemble against eight families and writes one
pass/fail row per (target × horizon × scale), each failure carrying a note saying which knob
would move it.

**Africa, k=5, M=400, `c1_foldb4` — 107 of 136 scored rows pass**, with 14 reported-only.

Three things to hold in mind before the tables:

- **Read rows, not the total.** Six gates were rewritten recently, so a flipped row is often
  the ruler moving rather than the model.
- **The member count is part of the measurement.** Several tolerances tighten as 1/√M, so
  comparisons must be at matched M.
- **A single card carries roughly ±7 rows of run-to-run noise** at k=5. Differences smaller
  than that are not interpretable.

### How a target gets its number

Targets come from four different places, and knowing which one you are looking at tells you
how seriously to take a near-miss.

| kind | where the number comes from | example |
|---|---|---|
| **definitional** | the interval is *defined* as 95%, so coverage must be 0.95 | T1.1, T2.1 |
| **derived from the data** | the observation is counted and the ensemble must match it | T6, T8 |
| **Monte-Carlo scaled** | what a finite sample of M members can achieve even when perfect | T5.1, T5.2 |
| **set a priori** | a judgement call made before the measurement existed | T7.3's ±0.25 |

The last kind is the one to distrust. Two a priori thresholds have already been re-derived
from data after they turned out to be asking for something the data cannot supply.

---

### T1 — Is each pixel's uncertainty right? (31/33)

**T1.1 pooled coverage — 2/4.** The fraction of pixels where the truth falls between the
published bounds. Target **0.95 ± 0.01**, definitional. Reads
**0.936 / 0.949 / 0.958 / 0.965** at h=5/10/15/20.

*Why the two fail, and note they fail in opposite directions.* h=5 under-covers at 0.936;
h=20 over-covers at 0.965. That is the per-pixel monotonicity pass (§2.3) doing exactly what
it must — a cumulative maximum only ever widens, and it widens later horizons hardest
(24.8% of pixels at h=20, none at h=5). So the constraint that makes T4.2 structural pushes
h=20 past nominal, and cannot help h=5 at all.

**T1.2 class-conditional coverage — 16/16.** The same question asked *inside* each class of
pixel, split by predicted change. Target **|coverage − 0.95| ≤ 0.03**. This exists because
pooled coverage can be perfect while every class is wrong in compensating directions — a
pooled 0.966 once coexisted with 0.335 in the high-change class. Reads 0.925–0.963 across
all sixteen cells, worst deviation 0.028 against a 0.05 allowance.

**T1.3 high-change tail — 3/3.** Coverage restricted to pixels where a *lot* of change is
predicted, `Δ̂ ∈ (0.05, 0.15]` — the pixels the product exists to get right. Target **≥ 0.92**,
deliberately looser because these classes are small. Reads **0.945 / 0.951 / 0.959**.
Passing here matters more than T1.1.

**T1.5 sharpness — 10/10.** Ratio of the final interval width to what the quantile heads
produced before calibration. Target **≤ 1.25**: calibration may widen by a quarter, no more,
because an interval can always be made to cover by being useless. Reads **0.946–1.066** —
close to the heads, occasionally slightly wider. Ten rows because the ratio is reported per
(window × horizon) pair.

---

### T2 — Is uncertainty right at *larger* scales? (34/41)

This family is the reason the project exists. Per-pixel intervals cannot answer "how uncertain
is this district?"; an ensemble can, and T2 checks whether the answer is right.

**T2.1 block coverage — 12/12.** Average HM over square blocks of 1 km, 10 km and 100 km,
compute that average on every member, and ask whether the observed block average falls inside
the members' 95% range. Target **0.95 ± 0.05**, definitional. Reads **0.928–0.964** across all
twelve (scale × horizon) combinations.

**This is the headline result of the whole project.** Aggregate coverage is correct at every
scale simultaneously, which neither naive alternative achieves: assuming errors are perfectly
correlated gives 0.97–1.00 (useless), assuming they are independent collapses to 0.09 at
100 km.

**T2.8 aggregate width — 12/12.** Coverage alone is not enough; the aggregate interval must
also be *narrower* than simply averaging the per-pixel bounds, or the correlation model is
buying nothing. Passes everywhere, by a margin that grows with block size — which is the
correlation doing its job.

**T2.2 ecoregion mean coverage — 2/4.** The same test over real ecological regions instead of
squares (158 on Africa). Reads **0.987 / 1.000 / 1.000 / 0.994**. Two fail by
**over**-covering: a coverage of exactly 1.000 means the truth never once fell outside the
members' range across 158 regions.

**T2.3 ecoregion area-above-threshold — 8/8.** Instead of the mean, the *fraction of the
ecoregion above HM 0.1 and 0.3* — harder than the mean because it depends on spatial
structure, not just level. Reads 0.981–0.994.

**T2.4 ecoregion mean change between horizons — 0/1.** Coverage of the *change* in ecoregion
mean from 2005 to 2020, the hardest aggregate case because it depends on the AR(1) coupling as
well as the spatial field. Reads **1.000** — over-covering.

**T2.5 rank histogram flatness — 0/4.** For each ecoregion, where does the observed value rank
among the 400 member values? If the ensemble is calibrated, the observation is equally likely
to land at any rank, so the histogram should be flat. A U-shape means too narrow; a dome means
too wide. Scored by χ² against uniform on pooled bins, target **p > 0.01**. Reads
**7.9e-18 / 1.2e-15 / 6.4e-11 / 8.5e-17** — rejected decisively at every horizon.

*Why:* the histograms are **domed** — the observation lands mid-ensemble far too often. That
is over-dispersion at ecoregion scale, the same thing T2.2, T2.4 and T7.3 report from other
directions. **This is the clearest unaddressed model defect on the card.**

*On the test itself:* it is scored on **pooled** rank bins. With 158 ecoregions against 401 raw
bins the unpooled χ² is valid but nearly powerless against a broad dome — measured, it rejects
such a dome 59% of the time against the pooled version's 90%, at the same size. Pooling is why
this family now fails honestly rather than passing for want of power.

**T2.6 biome / realm coverage — reported, not scored.** Eight rows, all 1.000. Not gated
because at ~14 units the confidence interval (±0.114) is wider than the ±0.05 tolerance, so
such a row can neither pass nor fail honestly.

---

### T3 — Do the members look like real maps? (5/6)

A member must not merely carry the right *amount* of uncertainty; it must have the right
*texture* — errors smooth where reality is smooth, rough where it is rough.

**T3.1 member variogram — 3/3.** Convert a member back to normal scores and measure its
spatial structure. Three checks: the variance of those scores (target **1.0 ± 0.15**, the
generator's contract), the **practical range** — the distance at which correlation effectively
dies — against the range fitted to the real residuals (**within 25% relative**), and the
**nugget fraction**, the share of variance that is pure pixel-scale noise (**within 0.10**).
Reads **1.000**, **0.215**, **0.044**. All three pass.

**T3.2 versus an independent null — 0/1.** Generate a second ensemble with *identical*
marginals but no spatial structure, and score both with the variogram score, which is
sensitive to correlation rather than marginals. The correlated one should win.

**The target is derived, not assumed.** An ensemble calibrated to the residual can only differ
from an independent one in the variance still *correlated* at the separation being scored —
everything already decorrelated in the data is identical for both by construction. That
ceiling is read off the residual's own fitted variogram as `1 − γ(d)/sill`, computed **in-run
at the pairs' realized mean separation**, and the gate is the *share of that budget* captured:
target **≥ 0.50**. Reads **0.104** — the ensemble captures a tenth of what is available.

Companion reported rows: `T3.2r` the raw improvement (0.043), `T3.2b` a uniform-sampled
reference (0.096), `T3.2s` the same comparison at short lag (**0.269** — markedly better,
which says the shortfall is specifically in long-range structure).

**T3.3 energy score — 1/1.** A proper multivariate scoring rule, checked against two
baselines: the independent null and a degenerate ensemble that is the central forecast
repeated. Target: **beat both**. Reads 0.764 against 0.794 and 0.923.

**T3.4 radial power spectrum — reported.** Catches the failure where range and sill both look
right but the shape between them is wrong.

**T3.5 longitude seam — 1/1.** On a global grid, checks for a discontinuity where longitude
wraps; not applicable on a regional grid and reported as such. *This row once failed a global
run on a NaN* — the antimeridian is open ocean for its whole length, so the metric had nothing
to compare and `isfinite(nan)` scored it as a discontinuity. It now marks itself reported-only
when the seam holds no valid pixels.

---

### T4 — Are the four horizons coherent? (4/4)

A member is one story about the future, so its four horizons must hang together.

**T4.1 between-horizon correlation — 3/3.** The correlation between consecutive horizons'
normal-score fields within a member, against the AR(1) coupling measured on the real
residuals. Target **within ±0.10 of the measured ρ**. ρ is `{10: 0.650, 15: 0.461, 20: 0.433}`;
the ensemble reproduces **0.653 / 0.472 / 0.439**, within 0.011 everywhere.

**This family was unscorable until ρ was measured.** Earlier runs coupled the horizons at a
fallback of 0.9 *without measuring* and scored T4.1 against a NaN — three rows pinned to the
constant they were fed, contributing nothing.

**T4.2 monotone spread — 1/1, and newly passing.** Uncertainty must not *shrink* as you
forecast further ahead. Target **≥ 0.99** of pixels non-decreasing. Reads **0.995**.

This row scores the **population** spread — the member marginal integrated against a standard
normal by quadrature, shape- and clip-aware — rather than the spread of a finite sample of
members. That distinction is load-bearing: the old sample-based version moved from 0.62 to
0.45 when ρ was measured rather than assumed, even though ρ *provably cannot change* the
spread at any single horizon. Two ensembles differing only in ρ now return an identical value
to sixteen digits.

It passes because of the per-pixel monotonicity pass in §2.3, and only because of it. Before
that pass the same chain read 0.802. The reported-only `T4.2s` row (0.886) carries the old
sample statistic.

---

### T5 — Hard gates: is the ensemble faithful to the published product? (10/12)

Non-negotiable. If these fail, the ensemble is telling a different story from the maps
published beside it.

**T5.1 median == central forecast — 2/4.** The member-wise median must equal the published
central forecast. Target **≥ 0.995** of pixels within a Monte-Carlo tolerance. Reads
**0.970 / 0.937 / 0.996 / 0.997** — h=5 and h=10 fail.

*Why, and how much to believe it.* The distributional median is exact **by construction** —
the marginal maps `u = 0.5` to the central forecast exactly. Only the *sample* median of 400
draws is not, so this row scores a population property through a finite sample, the same shape
of problem T4.2 had. Its tolerance is `3·(1.2533/√M)·σ·S′(0) + quantization`, where `S′(0)` is
the marginal shape's slope at the median; raising the tail bound stretches the shape's body
over a wider range of normal scores, which **shrinks `S′(0)` and tightens the tolerance**. So
the failure follows mechanically from a marginal choice acting on the gate's own scaling.
**It is not established that generation is wrong, nor that the gate is.** Flagged as open; the
field must not be tuned to pass it before that is settled.

**T5.2 tails == published bounds — 4/4.** The ensemble's 2.5th and 97.5th percentiles must
equal the published bounds. Target **≥ 0.95** within an MC tolerance. Reads **0.978–0.995**.
The tolerance is *family-aware*: the standard error of a sample quantile depends on the
density at that quantile, which for a reshaped marginal is not what it is for a normal.

**T5.3 validity mask identity — 4/4.** Every pixel the model calls valid must be valid in the
ensemble and vice versa. Target **0 mismatched pixels**. Reads 0 at all four horizons.

---

### T6 — Is the *amount* and *sign* of change realistic? (12/24)

T1–T2 ask whether the uncertainty is the right size. T6 asks whether a member, read as a map
of change, resembles a real one.

**T6.1 P(decrease > 0.01) — 0/4.** How often does a member show HM *falling* by more than
0.01, against how often reality did? Target **ratio in [0.5, 2.0]**, from the data. Reads
**2.05 / 3.73 / 3.52 / 5.02** — two to five times too many small decreases.

*Why:* the marginal's lower tail is too heavy for a quantity that is real but rare. Part of
this is not reachable by any per-pixel marginal — the ideal calculation using each band's own
empirical distribution still over-predicts, which means it reflects a dependence between the
standardised residual and the pixel's own threshold that a marginal cannot encode.

**T6.2 / T6.3 larger decreases — 4/4 and 4/4.** The same question at −0.05 and −0.15. Targets
are looser (ratio ≤ 3 and ≤ 5) *and* carry an absolute cap, so a class cannot pass on a
favourable ratio while emitting an implausible absolute rate. Read 0.26–2.61 and 0.00005–0.029.
**The excess is entirely in small decreases**, not extreme ones — which is diagnostic.

**T6.4 tail asymmetry — 0/4.** Real HM change is strongly asymmetric: increases are far more
common than decreases. This compares the ensemble's ratio of large increases to large
decreases against the observed ratio. Target **[0.5, 2.0]**. Reads 76,000 / 2,250 / 105 / 75.

*Why it fails, and why it now fails honestly.* This gate was one-sided — "at least half the
observed asymmetry" — which a member with an arbitrarily heavy upper tail passes trivially.
Scored two-sided it correctly reports that at large thresholds the upper tail is far too heavy
relative to the lower one.

**T6.5 change quantiles — 4/8.** The 1st and 5th percentile of member change against the
observed ones. Target **within a factor of 2**. Reads 0.88–10.4, failing at the deeper
percentile at three horizons — the same too-heavy lower tail as T6.1.

---

### T7 — Are the members diverse but not absurd? (1/2)

**T7.2 member diversity — 1/1.** Mean pairwise correlation between members. Target **< 0.98**:
near-copies would mean the ensemble has no spread to offer. Reads **0.191**.

**T7.3 spread-skill ratio — 0/1.** The deepest question on the card. Where the ensemble says
it is uncertain, the model should actually be more wrong. Computed at ecoregion scale as the
ensemble's own standard deviation divided by the RMSE of the ensemble mean. **1.0** means
spread predicts error correctly; above 1 means over-spread — claiming more uncertainty than is
needed. Target **1.0 ± 0.25**, set a priori. Reads **1.768**.

*Why:* the ensemble is too wide at aggregate scale — the same defect as T2.5's dome and T2.2's
saturated coverage. Three independent instruments agreeing.

---

### T8 — Is change put in the *right places*? (10/14)

The geographic sanity check, and the family the calibration's class structure exists to serve.
Every row is scored per **distance-to-past-change band**. Change is overwhelmingly concentrated
near change that already happened, and an ensemble that sprinkles development into untouched
country is wrong in a way no coverage target notices.

**T8.1 P(increase > 0.05) per band — 5/6.** Ratio of member rate to observed rate, target
**[0.5, 2.0]**:

| band | ratio | |
|---|---|---|
| 0–1 px | 1.54 | pass |
| 1–3 px | 1.44 | pass |
| 3–10 px | 1.32 | pass |
| 10–30 px | **2.28** | fail, too hot |
| 30–100 px | 1.32 | pass |
| **> 100 px** | **1.36** | **pass** |

**The far field is the result to notice.** An earlier arrangement of this chain emitted 0.107
of the observed rate beyond 100 px — an ensemble that could not imagine development appearing
anywhere new. It now sits within 36% of observed, and it does so because the calibration
separates the 0.6% of remote pixels that carry development potential from the 99.3% that do
not (§2.4). The 10–30 px row is the cost: extending the correction to every band overshot
there, and excluding band 3 is the obvious next adjustment.

**T8.2 remote-band realism, both directions — 0/1.** A single summary of the remote band: mean
**|log₁₀(member / observed)|** over *both* tails, target **≤ 0.301** (within a factor of two
either way). Reads **2.357**, and entirely on the decrease side — the increase side is 1.36.

**T8.3 near/remote contrast — 1/1.** Is change concentrated near past change to the same degree
reality is? Ratio of the member's near/remote contrast to the observed contrast, target
**[0.5, 2.0]**. Reads **1.138**, close to perfect.

**T8.4 P(decrease > 0.05) per band — 4/6.** Reads 2.16 / 1.03 / 1.37 / 1.91 / **8.22** /
**0.00**. The two far bands fail.

*Why the far one should not be chased.* The floor that prevents HM falling below zero is
**already exact** — the sampler clamps every member to [0, 1], so a pixel at HM ≤ 0.05
contributes exactly zero and cannot produce any excess. Measured on the far band, only
**0.14% of remote pixels can physically decrease past −0.05**, and all 88 observed decrease
events globally lie on them — **one** of which is on Africa. A ratio computed against a
denominator of one observed event is not a measurement, whether it reads 0.00 or 18.9.

---

### What the card says overall

Three model defects are named and unaddressed, all pointing the same way — the ensemble is
over-dispersed at aggregate scale (T2.5 rejected 4/4, T7.3 at 1.77, T2.2 and T2.4 saturated) —
plus long-range structure at a tenth of the residual's own budget (T3.2), where the short-lag
reading of 0.269 says the shortfall is specifically at long range. One instrument question is
open (T5.1). The far-field placement problem is substantially solved on the increase side and
rests on too few events to solve on the decrease side.

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

Holding geography out has a visible consequence. The fold mask **used to be** a 128 px
checkerboard (it is now 512 px blocks; see below) —
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

A third option — **coarser fold blocks** — was adopted, and it is the current mask.

The argument was never cosmetic. The residual field's fitted practical range is 99–166 px,
typically ~130, against a 128 px fold tile: the old checkerboard held geography out at
**one correlation length**, so every held-out tile was ringed by trained-on tiles well inside
the distance over which the residual is still correlated. Held-out skill was therefore
optimistic by an unknown amount.

`--fold_block_chips 4` gives contiguous **512 px** fold territories. Measured with a distance
transform on the Africa window — for each fold, how far every held-out pixel is from the
nearest pixel that fold *trained on*:

| mask | held-out pixels beyond 135 px of trained-on data |
|---|---|
| 128 px checkerboard | **0.0%** |
| **512 px blocks** | **31.9%** |

Under the old mask **not one** held-out pixel was beyond a correlation length. Under the new
one a third are, and Africa still gets 137–165 blocks per fold to average over.

**What it cost: at most 0.9 skill points.** Same architecture, same seed, only the mask
changed:

| horizon | 128 px | 512 px |
|---|---|---|
| h=5 | 0.1298 | **0.1301** |
| h=10 | 0.2006 | 0.1960 |
| h=15 | 0.2340 | 0.2277 |
| h=20 | 0.2313 | 0.2221 |

**And what it bought: the optimism is now measurable.** Within the new mask, h=20 skill falls
with distance from trained-on data — **+0.239** at 1–32 px, **+0.188** beyond 192 px, a 21%
relative decline. The old mask cannot see this at all, because its held-out pixels stop at
128 px. The leak was real, it is modest, and it is now a number instead of a worry.

Adopting it invalidated every earlier checkpoint and made every earlier scorecard
non-comparable. That is why it was done in one batch, and why 111/133 and 101/127 are
historical.

Each model predicts four input windows (ending 2000, 2005, 2010, 2015) at four horizons,
giving ten (window × horizon) pairs within the observed record.

Without fold rotation there is no way to produce an out-of-sample map at all, which is what
every number in §3 depends on.

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

