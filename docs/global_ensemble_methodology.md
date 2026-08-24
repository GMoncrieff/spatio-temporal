# Global Human Modification forecasting — methodology

This document describes the production system: a ConvLSTM that forecasts Human Modification
(HM) at four lead times, a calibration layer that makes its published intervals honest, and
an ensemble generator that turns per-pixel intervals into complete, spatially coherent maps.

**The product.** HM is a 0–1 index on a 1 km global grid (17,111 × 40,000 cells, 184.6 M of
them carrying valid land data). For every valid pixel the system publishes three numbers at
each of +5, +10, +15 and +20 years:

- a **central forecast** — the best single estimate,
- a **lower bound** at the 2.5th percentile and an **upper bound** at the 97.5th.

That triple is the headline product. Alongside it the system publishes an **ensemble** of 400
complete maps ("members"), each one a plausible version of the world consistent with those
bounds. The ensemble exists because per-pixel intervals cannot answer questions about areas:
to aggregate uncertainty over a district, an ecoregion or a country you must know how errors
at different pixels relate to one another. With an ensemble, aggregation stops being a
modelling assumption and becomes arithmetic — compute the quantity on each member and read
the spread.

The chain end to end:

```
inputs (HM at 3 timesteps + covariates)
   │
   ├─► ConvLSTM ──► central forecast + lower/upper bounds          ← §1
   │
   ├─► hindcast residuals (what the model actually gets wrong)
   │        │
   │        ├─► calibration layer: per-class half-width factors    ← §2
   │        ├─► marginal shape: the per-pixel error distribution   ← §2
   │        └─► spatial field parameters + horizon coupling        ← §3
   │
   └─► ensemble generator ──► 400 members                          ← §3
```

---

## 1. The ConvLSTM

### 1.1 What it consumes and emits

The model reads three input timesteps spaced five years apart (t−10, t−5, t) and emits twelve
output channels: four horizons × {lower, central, upper}.

| | |
|---|---|
| trunk | 4-layer ConvLSTM, `hidden_dim = 64`, 3×3 kernels |
| dynamic inputs | 11 channels — HM and its component layers, at each of the three timesteps |
| static inputs | 7 channels — elevation, climate, protection status |
| location encoding | 8 learned channels (spherical-harmonics + SIREN backbone, 10 Legendre polynomials) |
| heads | 12 independent heads, `Conv3×3 → ReLU → Conv1×1`; quantile heads at half trunk width (32) |
| optimiser | Adam, lr 1e-3, no schedule, no weight decay, no gradient clipping |
| schedule | 150 epochs × 13 steps × 8 chips of 128² px |

Three structural properties of the head design carry most of the model's behaviour, and each
is described below.

### 1.2 The central head predicts change, not level

The central head is a **residual** head: it predicts the *change* in HM from t₀ and adds it to
HM at t₀ through a skip connection, with a zero-initialised output convolution.

This matters because the signal is small and sparse. Across the globe, 53–73% of pixels change
by less than 0.001 over a five-to-twenty-year horizon — for those, the correct answer is "emit
t₀ unchanged". A head predicting absolute HM must first reproduce t₀ through the trunk before
it can add anything, and that reproduction is not free: it leaks noise on exactly the pixels
where the truth is "nothing happens". With the skip connection the model starts at exact
persistence and saying "nothing happens" costs nothing.

The right yardstick for the result is **skill against persistence** — against predicting that
nothing changes. That bar is higher than it sounds, because the median 20-year HM change is
0.0001, so pooled RMSE can look unremarkable while the model loses to doing nothing. Measured
globally on held-out geography:

| horizon | skill vs persistence |
|---|---|
| +5 yr | **+0.110** |
| +10 yr | **+0.183** |
| +15 yr | **+0.226** |
| +20 yr | **+0.222** |

Positive at every horizon and rising with lead time, which is the sensible ordering: the
further out you forecast, the more there is to beat "nothing happens" at.

### 1.3 The long-range context covariate

The trunk's receptive radius is roughly 10 px, set by its 3×3 kernels over 4 layers. It
therefore cannot see whether change happened 30–100 px away — which is the single best
predictor of whether change is possible at a pixel at all.

`scripts/prepare_change_context.py` precomputes two bands per input window, on the **full
raster**:

1. `past_change` — HM at the window's base year minus HM ten years earlier;
2. `dist_past_change` — Euclidean distance, in pixels, to the nearest pixel whose
   `past_change` exceeded 0.01.

Both head families receive 8 channels derived from these:

| channels | content |
|---|---|
| 5 | occupancy — is there past change within 1 / 3 / 10 / 30 / 100 px? (simply `dist ≤ r`) |
| 1 | `log1p(dist) / 10` |
| 1 | the signed past change itself |
| 1 | HM at t₀ |

Because occupancy is read off a precomputed distance, no pooling is involved and the chip
boundary plays no part. This is why the distance is computed globally rather than inside the
chip: derived from a 128 px chip, the 100 px radius saturates into "is there any change
anywhere in this chip", which is an artifact of framing rather than a fact about geography.

### 1.4 The interval is built around the central forecast, and cannot narrow with lead time

The quantile heads do not predict bounds directly. Each emits a non-negative **half-width
increment** through a softplus, and those increments **accumulate across horizons** around the
central forecast, which is detached before being used as the anchor.

Two consequences, both structural rather than enforced afterwards:

- **Spread cannot shrink with lead time.** A cumulative sum of non-negative increments is
  non-decreasing by construction.
- **`lower ≤ central ≤ upper` always holds**, rather than being clipped into place later.

The quantile loss is pinball at 0.025 and 0.975, and it is **gradient-isolated from the
trunk** — the central forecast enters the quantile heads detached, so the pinball objective
shapes the interval without moving the central field.

### 1.5 Loss

The central objective is computed on absolute HM:

```
MSE  +  0.2 · SSIM  +  0.3 · Laplacian-pyramid  +  1.0 · histogram
```

and the quantile objective is pinball at the two tail levels, isolated as described above.

### 1.6 How out-of-sample predictions are produced

HM exists only from 1990 to 2020, and a model must not be scored on data it trained on. The
system therefore uses **five-fold cross-validation over geography.**

The globe is partitioned into five spatial folds on a mask of contiguous **512 px blocks**
(`fold_mask_b4_1000.tif`). Five models are trained; each excludes one fold from training
entirely and uses another for validation. Each model then predicts only the fold it never saw,
and the five predictions are stitched into one raster where every pixel comes from the model
that held it out.

Block size is a deliberate choice. The residual field's fitted practical range — the distance
over which model errors remain correlated — is about 300 px globally. Fold territories must be
larger than that, or every held-out pixel sits inside the correlation length of pixels its own
model trained on, and held-out skill is optimistic by an unknown amount. At 512 px blocks a
substantial fraction of held-out pixels lie beyond one correlation length of any pixel their
fold trained on, and each fold still holds hundreds of blocks to average over.

Each model predicts four input windows (ending 2000, 2005, 2010, 2015) at four horizons, giving
ten (window × horizon) pairs that fall inside the observed record. **Everything downstream —
the calibration, the marginal, the spatial field, the horizon coupling — is estimated from the
residuals of these predictions**, and every one of those residuals is genuinely out of sample.

Two stitching modes exist and they serve different purposes:

- **`holdout`** — each pixel takes the value from the fold that held it out. Out of sample
  everywhere. **This is the only mode anything scored may use.** It is a hard mosaic: adjacent
  fold territories come from different models, so wherever those models disagree the join is
  visible, and they disagree mainly in the upper bound.
- **`mean`** — every fold averaged at every pixel. Seamless, and in-sample at every pixel by
  construction, since four of the five folds trained on any given location. This is a display
  product only.

### 1.7 The forward model

A fold model cannot forecast forward: each deliberately never saw a fifth of the world, which
is what makes its hindcast honest and what makes it unusable for production. The forward
product therefore comes from a **sixth model, trained on every chip with no fold excluded**,
in the identical configuration.

It reads HM at 2010 / 2015 / 2020 and predicts 2025 / 2030 / 2035 / 2040. The horizon structure
is identical to the hindcast, so all the residual statistics of §2 and §3 transfer directly to
its outputs.

Because it trains on everything, it has no held-out data of its own. What is validated is the
*configuration*, by the five-fold hindcast; the production draw is then trusted.

---

## 2. Calibration and marginal fitting

Everything in this section is estimated from **hindcast residuals**: `observed − central`, on
the out-of-sample predictions described in §1.6. For each (window × horizon) the system stores
the raw residual, the predicted change, HM at t₀, both published half-widths, and the
distance-to-past-change covariate.

The key derived quantity is the **standardised residual**:

```
e = (observed − central) / (half-width / 1.96)
```

using the upper half-width when the residual is positive and the lower when it is negative. If
the published interval were exactly right, `e` would look like a standard normal — about 95% of
values between −1.96 and +1.96. Every departure from that is what this section corrects.

Two distinct things can be wrong with a published interval, and they need different fixes:

- it can be **the wrong width** — the truth falls outside more than 5% of the time, or almost
  never does. §2.1 fixes this.
- it can have **the wrong shape inside** — the right width, but probability distributed wrongly
  within it. §2.2 fixes this.

Neither substitutes for the other, and the order matters: the shape is normalised to whatever
bounds it is fitted on, so it must be fitted *after* the width correction, on residuals rebuilt
against the corrected bounds.

### 2.1 The calibration layer

**One layer, three class axes.** Every pixel is sorted into a class, and each class carries its
own pair of multipliers on the half-widths:

```
distance to past change  ×  predicted change (Δ̂)  ×  HM level at t₀
```

with these bins:

| axis | bins |
|---|---|
| distance (px) | 0–1, 1–3, 3–10, 10–30, 30–100, >100 |
| predicted change Δ̂ | ≤−0.01, (−0.01, 0.001], (0.001, 0.01], (0.01, 0.05], (0.05, 0.15], >0.15 |
| HM at t₀ | [0, 0.01), [0.01, 0.1), [0.1, 0.3), [0.3, 0.6), [0.6, 1] |

Each axis earns its place. *Distance* because development happens near development, and the
model's error in untouched country is a different animal from its error at a city edge.
*Predicted change* because a pixel where the model expects heavy development has a different
error distribution from one where it expects none. *HM level* because error on already-built
land does not behave like error on wilderness.

**The central forecast is never moved.** Only half-widths are scaled. Central rasters are
copied byte for byte.

**How a factor is estimated.** Within a class, the factor is that class's own 97.5th (or 2.5th)
percentile of `e`, expressed in half-width units. If the interval were already right, `e` would
be standard normal and the factor would come out at 1. A factor of 0.5 means the published
interval was twice as wide as it needed to be; 2 means half as wide. The two sides are fitted
separately, because the error is skewed — there is far more room to be surprised upward than
downward.

The estimate is **not centred on the residual's median.** The interval is centred on the central
forecast, not on the residual's median, so it has to cover the model's bias as well as its
spread. (The marginal shape in §2.2 *is* centred, for a different reason, and conflating the two
is a real hazard.)

**Four safeguards:**

- **Shrinkage toward the parent.** The three axes form a hierarchy — distance band is the root,
  predicted change sits under it, HM level under that. Each cell is pulled toward its parent by
  `n / (n + 2000)`, so a thin class inherits rather than fitting noise. The artifact records
  which classes were fitted on their own pixels and which inherited, because a table of
  hundreds of classes otherwise looks far better resolved than it is.
- **A floor of 0.25**, so a degenerate class cannot collapse the interval to nothing.
- **Monotone in horizon within a class**: factors are made non-decreasing across horizons, so no
  class can claim more certainty at a longer lead time.
- **Guards and clipping at apply time**: the lower bound is forced not to exceed the centre, the
  upper not to fall below it, and both are clipped into [0, 1] because HM is an index on that
  range.

**A per-pixel monotonicity pass, after everything else.** Each pixel's half-widths are made
non-decreasing across horizons by a cumulative maximum. This is separate from the constraint
inside the fit and cannot be replaced by it: a pixel is not in one class. Because Δ̂ grows with
lead time, a pixel can move between predicted-change bins along the horizon sequence and pick up
a different factor at each, so it can invert even when every class is individually monotone. A
cumulative maximum is the smallest change that guarantees the property — it only ever widens,
only at the horizons that dipped, and leaves the shortest horizon untouched.

**What the class structure actually does, and why the distance axis is load-bearing.** It is
tempting to read the layer as "make intervals bigger or smaller". Within one class it is exactly
a scaling — every pixel in a `(band × Δ̂ × HM)` cell is multiplied by the same number, so that
cell's median and its 99.9th percentile move together.

But a distance band is not one class; it is a **mixture**, and the other two axes split it. The
band beyond 100 px is almost entirely one cell plus a sliver of another: the overwhelming
majority is remote land at HM ≈ 0, where nothing can plausibly happen, and it is *narrowed*; a
small fraction is remote land that already carries some development, whose intervals are far
wider to begin with, and it is *widened* several-fold. Those few pixels are the far band's
entire upper tail.

So the layer's job in the far field is not to widen or narrow it. It is to **separate the remote
land where nothing can happen from the remote land where something can.** Any class definition
that blurs that distinction — anything that averages across distance — sits on top of both
populations at once, scores well on a pooled average, and removes the product feature. This is
why the axis is distance and not, say, biome, and it is why the layer is judged on
class-conditional coverage and on the far band's extreme half-width percentiles, never on a
pooled score alone. A defect living in a thousandth of the pixels is invisible to an average.

Every distance band is corrected. The decision was taken by fitting on three folds, scoring on
the two held out, and reading both the interval score and the far-band tail — under which
correcting all six bands is what keeps the remote classes' coverage near nominal.

### 2.2 The marginal — what distribution sits at each pixel

Each pixel needs a full distribution, not just three numbers, and it must honour the published
triple exactly.

**The base family** is a **median-spliced two-piece normal**: a normal with one standard
deviation below the median and a different one above, chosen so the 2.5th percentile lands
exactly on the published lower bound, the median exactly on the central forecast, and the 97.5th
exactly on the upper bound. It is closed-form and exact.

**Its weakness is shape.** Measured in units of the published half-width, the real residual is
far more concentrated in the middle and far heavier in the tails than any normal — its kurtosis
runs into the hundreds or thousands, against 3 for a Gaussian.

So the *shape* is replaced while the three anchor points are kept exact. `fit_residual_shape`
builds the residual's own standardised quantile function on a 513-knot grid and normalises it so
that `u = 0.025 / 0.5 / 0.975` still map precisely to lower / central / upper. Because the map is
strictly increasing it **cannot change spatial structure at all** — it preserves the rank order
of every pixel, so the correlated field of §3 passes through untouched. A genuinely Gaussian
residual returns the identity map.

Two properties of the fit:

- **The three gate quantiles are exact knots.** Interpolating near them leaves a bias that does
  not shrink with more members.
- **Beyond the tail bound, the two-piece normal resumes.** The empirical tail far out is
  dominated by pixels whose *half-width* is near-degenerate, so the ratio explodes for reasons
  about the denominator rather than about the distribution.

**The tail bound** (`u_bound`) is how far out the empirical shape is trusted. Production uses
**0.999 on the upper side and 0.025 on the lower, on every distance band**, with the body pooled
across bands and a separate shape per (horizon × band). Swept on global residuals, the pooled
error across the 24 (horizon × band) cells falls monotonically from the plain two-piece normal
through 0.975 and 0.99 to 0.999 — and **0.999 and 1.0 are identical**, which is the important
part: the lever is exhausted. Removing the bound entirely adds nothing.

**A structural limit worth stating plainly.** Because the shape normalises each side by that
side's own 2.5/97.5 quantile — which is exactly what makes the bounds exact — the fitted marginal
is *the residual's distribution stretched to fill the published interval*. Any marginal that
preserves the published bounds necessarily re-injects whatever width error those bounds carry.
**No shape can fix a width problem**, which is precisely why §2.1 exists as a separate step.

---

## 3. Ensemble generation

### 3.1 What a member has to satisfy

We want 400 maps, each satisfying two things at once:

1. **At each pixel, the value is drawn from that pixel's own distribution** — the marginal of
   §2.2. Across members, a pixel should sit at its central forecast on average, fall below the
   published lower bound about 2.5% of the time and above the upper about 2.5% of the time.
2. **Neighbouring pixels are wrong together.** Real forecast errors are spatially organised: a
   model that under-predicts a city's expansion under-predicts it across the whole city, not at
   every second pixel.

The standard tool for satisfying both is a **copula**, which is a two-step recipe:

> **Step A.** Generate a spatially correlated random field of *standard normal* values — a map
> of numbers individually N(0,1) but correlated with their neighbours in the right way. Call it
> **z**.
>
> **Step B.** At every pixel, push that pixel's *z* through that pixel's own marginal.

Step A carries all the spatial structure; step B carries all the per-pixel calibration. They are
completely separable, which is what makes this tractable at 184.6 M pixels — and it means a
marginal problem can be fixed without touching spatial structure, and vice versa.

### 3.2 Describing the spatial structure

Two standard ways to describe how a field varies with distance are the **variogram** (average
squared difference between pairs a given distance apart) and the **power spectrum** (how much
variance sits at each spatial frequency — low frequency is big smooth blobs, high frequency is
fine speckle).

The system fits the **radial power spectrum**, not the variogram. A variogram is evaluated at
lags, and two long-range components can absorb the fit while leaving the middle of the range
empty — which produces smooth blobs where the truth is structured speckle. Matching the radial
spectrum instead reproduces the observed variance shares across scales.

The fit is a non-negative least-squares mixture over a fixed basis of correlation ranges —
1.5, 2.5, 4, 6, 9, 14, 25, 50, 120 and 300 px — plus a **nugget**, the purely pixel-level part.
The result reads as "x% of the variance at ~4 px scale, y% at ~25 px, …". Fitted on global
residuals, the nugget runs 0.03–0.06 and roughly 0.36–0.56 of variance sits beyond 50 px.

**The kernel is Matérn with `nu = 0.5`** — the exponential kernel, rough at short range.
Smoothness matters as much as range: a Gaussian kernel produces infinitely smooth fields, and
real error fields have rough, textured edges. Gaussian kernels cannot make texture.

**The long-range component is added by hand.** A power spectrum computed from a mean-subtracted
field is blind to frequency zero — the overall level. But frequency zero is precisely what
controls whether a *whole region's* average is uncertain, so fitting the spectrum alone leaves
every large aggregate over-confident. A deliberate long-range component at **range 1500 px and
weight 0.40** is therefore added, calibrated from aggregate error rather than from the spectrum.

### 3.3 Building the field

Generating a correlated field naively means factorising an N×N covariance matrix, where N is the
number of pixels — impossible at this scale. **Circulant embedding** avoids it using a fact from
Fourier analysis: if a field's correlation depends only on the separation between points, then
in frequency space the different frequencies are independent. So:

1. compute the target covariance's spectrum `S` — one number per frequency;
2. draw independent white noise;
3. Fourier-transform it, **multiply each frequency by `sqrt(S)`**, transform back.

The result has exactly the covariance requested and costs two FFTs. Multiplying by `sqrt(S)` is
"turn up the frequencies that should carry variance and turn down the ones that should not".

Two practical details. The grid is **padded** before the transform, because an FFT implicitly
treats the map as wrapping, which would otherwise correlate the left edge with the right. And on
the global grid **longitude wrap is deliberately re-enabled**, because there the wrap is
physically correct — the antimeridian is a real join, not an artifact.

### 3.4 From field to member

For each member and horizon:

1. draw a correlated standard-normal field **z**;
2. convert `z` to a probability and push it through that pixel's marginal — two-piece normal,
   reshaped by the fitted empirical shape, with the shape selected by the pixel's distance band;
3. clip to [0, 1] and apply the model's validity mask.

Because the shape is strictly increasing, step 2 changes *values* but not *ranks* — the spatial
structure built in step 1 passes through unaltered. The clip to [0, 1] also makes the physical
floor exact: a pixel at HM ≈ 0 cannot produce a spurious decrease.

### 3.5 Coupling the horizons

A member is one story about the future, so its four horizons must hang together: a member showing
a city expanding fast by 2025 should still show it expanding by 2040. The horizons are chained by
an **AR(1)** process on the normal-score fields:

```
z(h) = ρ · z(h−1) + sqrt(1 − ρ²) · ε(h)
```

with ρ estimated from the residuals' own between-horizon correlation. This construction keeps
each `z(h)` marginally standard normal, which matters: **ρ changes how the horizons hang together
without changing the spread at any one of them.**

ρ is measured, never assumed. The estimator reads evenly spaced full-width row stripes — it is
deterministic and seed-free, and independent stripe phases agree to within a few hundredths.
Measured globally: **ρ = {+10 yr: 0.749, +15 yr: 0.613, +20 yr: 0.706}**, and the generator
reproduces those values closely in the members it draws.

### 3.6 Two guarantees

Two properties hold by construction and are asserted as hard gates:

- the ensemble's **median equals the published central forecast**;
- the ensemble's **2.5th and 97.5th percentiles equal the published bounds**.

That is what makes the ensemble a *representation* of the published product rather than a
second, competing product. Whatever the ensemble says about aggregates, the headline maps are
unchanged.

Both are population properties that a finite sample of 400 members reproduces only up to Monte
Carlo error, so both are scored against tolerances that scale with M and with the marginal's own
density at the quantile being checked.

### 3.7 Storage and reproducibility

Members are stored as **int16 with scale 1/32767**. HM is bounded on [0, 1], so this is lossless
to about 3 × 10⁻⁵, far below any quantity of interest.

Ensembles are **icechunk repositories** containing one array, `members`, of shape
`(400, 4, 17111, 40000)`, chunked `(1, 1, 1024, 1024)` — one member per chunk. The chunk shape is
deliberate: a partial-chunk write in icechunk is a read-modify-write that retains every version
until garbage collection, so writing one member into a shared chunk would multiply the store size
several-fold. One member per chunk also means two writers can never share a chunk.

The write is a **single transaction**: each GPU worker writes through a forked session and the
parent merges and commits once, so a run killed part-way leaves no store at all rather than a
directory that reads back as plausible-looking sentinel values.

**Every member's random seed is recorded in the manifest**, so any ensemble regenerates exactly
and a superseded one can be deleted and rebuilt from the record.

---

## Summary of fitted parameters

Everything below is estimated from the global out-of-sample hindcast residuals. Nothing is
carried over from any other extent.

| parameter | value |
|---|---|
| fold mask | 512 px contiguous blocks, k = 5 |
| calibration class axes | distance band × predicted change × HM level, all six distance bands corrected |
| calibration shrinkage | `n / (n + 2000)` toward parent; floor 0.25; monotone in horizon; per-pixel cumulative max after apply |
| marginal | two-piece normal reshaped by empirical quantile function, 513 knots, per (horizon × band), body pooled across bands |
| marginal tail bounds | upper 0.999, lower 0.025, on every band |
| spectral basis (px) | 1.5, 2.5, 4, 6, 9, 14, 25, 50, 120, 300 |
| kernel | Matérn, `nu = 0.5` |
| long-range component | range 1500 px, weight 0.40 |
| residual practical range | ≈ 300 px |
| AR(1) ρ | +10 yr 0.749, +15 yr 0.613, +20 yr 0.706 |
| ensemble size | M = 400 |
| storage | int16 × 1/32767, icechunk, chunks (1, 1, 1024, 1024) |
