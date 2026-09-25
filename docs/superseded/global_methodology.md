# Global Human Modification forecasting — methodology

How the global product is built, end to end: a distributional ConvLSTM that emits a full
per-pixel quantile function, and an ensemble that turns those independent per-pixel
distributions into spatially coherent maps.

**Scope.** Branch `dist-convlstm`. This document describes the system that supersedes the frozen
`g1_foldb4` product — a triple-head ConvLSTM plus a four-stage post-hoc chain of width
calibration, empirical marginal reshaping, spatial spectrum and horizon coupling. **That entire
post-hoc chain is gone.** The model now emits the distribution directly, and the only thing left
downstream is the spatial dependence, which no per-pixel model can supply.

Companions: `docs/dist_global_scorecard.md` (what the model scores and what each metric means),
`docs/global_ensemble_scorecard.md` (the ensemble's, once the global run completes),
`docs/fitting_running_model.md` (the runbook, with measured cost),
`docs/dist_model_phase.md` and `docs/dist_model_phase2.md` (why the model looks like this),
`docs/dist_ensemble_phase.md` (why the ensemble looks like this).

**Grid.** 17111 × 40000 at 0.009° (about 1 km), EPSG:4326, **184,573,321 valid land pixels**. Horizons +5,
+10, +15, +20 yr. Hindcast base years 2000/2005/2010/2015; forward base year 2020 predicting
2025–2040.

---

## Part 1 — The model

### 1.1 What it consumes and emits

A ConvLSTM over three five-yearly input steps. Per pixel it sees the HM level and its component
layers, static covariates (elevation, climate, protection status), a location encoding, and two
families of neighbourhood context described in §1.3.

For each of four horizons it emits a **complete conditional quantile function** `Q_h(u | x)` —
not three numbers, but the whole distribution. Everything published is read off that function:

| published raster | definition |
|---|---|
| `lower` | `Q(0.025)` |
| `upper` | `Q(0.975)` |
| `central` | `E[Q] = ∫₀¹ Q(u) du` — the **mean**, not the median |
| `qf` | 64 bands, `Q(u)` on a fixed normal-spaced `u` grid with 0.025, 0.5 and 0.975 pinned |

Because the triple is *derived from* the quantile function rather than predicted alongside it,
the two agree by construction. Measured globally: `max|qf − published bound| = 1.55e-05`, half
the int16 storage quantum.

**`central` is a mean and is not constrained to lie inside `[lower, upper]`.** For a strongly
right-skewed pixel — 97.6% chance of staying undeveloped, 2.4% chance a town arrives — all three
of Q(0.025), Q(0.5), Q(0.975) sit in the "nothing happens" mass while the mean is dragged above
Q(0.975) by the rare jump. Measured on 7–9% of pixels at +15/+20 yr. This is intended: the mean
is the RMSE-optimal point estimate, which is what the central field is scored on. Consumers must
be told, because a reader treating `central` as a percentile will be wrong there.

### 1.2 The head: a monotone spline over absolute HM

`Q_h(u)` is a **monotone rational-quadratic spline** with **fixed, tail-dense knots** (14 bins,
15 knots, 29 parameters per horizon). Three properties come from the construction rather than
from a later repair pass:

- **Monotone in `u` by construction** — softmax bin heights and positive derivatives. `lower ≤
  central ≤ upper` cannot be violated and no sorting pass exists.
- **Anchored at persistence.** The spline's location is the existing zero-initialised residual
  head, so an untrained model predicts "nothing changes" and every departure is learned. Scoring
  against persistence rather than zero is not optional here — the median 20-year HM change is
  0.0001.
- **Cumulative scale across horizons.** The width parameter is a softplus increment accumulated
  over lead time, so the interval **cannot narrow** as the horizon grows. The frozen product
  enforced this with a post-hoc `cummax` pass that silently skipped the entire forward product
  because of a filename-prefix bug; here it is structural.

Tail-dense knots are what let the model express a long right tail at all. Measured pooled
`tail_reach = (Q(0.999) − Q(0.5)) / (Q(0.975) − Q(0.5))` runs **4.9–5.9** against **1.577** for a
Gaussian — the model starts at the Gaussian value and learns its way out.

### 1.3 Neighbourhood context

Two families of full-raster covariates, both precomputed. **Any long-range covariate must be
computed on the full raster**, never inside a training chip: radii ≥ 30 px saturate against a
128 px chip boundary and the covariate silently becomes a constant.

- **Past-change context** (`change_context_w{year}_1000.tif`): distance to past change, and
  past-change density at radii 3, 30, 100 px.
- **Neighbourhood HM** (`hm_context_w{year}_1000.tif`, built by `scripts/prepare_hm_context.py`):
  mean and max HM at radii 3, 30, 100 px.

Together: **12 context channels**. The banner `Context channels: 12 (radii 3,30,100, hm mean,max
@ 3,30,100)` is the fingerprint that the covariate reached the model; a run that silently fell
back to eight would read as "the covariate does nothing".

### 1.4 Loss

**CRPS alone**, with the auxiliary MSE on the central field. `--ssim_weight 0
--laplacian_weight 0 --histogram_weight 0`; checkpoint selection on `val_crps`.

Two findings behind that. Pure CRPS costs nothing centrally — dropping the MSE weight entirely
leaves RMSE inside a 2%-wide band — and **the trunk must hear the distributional objective**:
isolating the shape head's gradient leaves RMSE intact while `crps_skill` collapses from 0.13 to
0.059. NLL was tried and collapses to a point mass at the width floor.

**The loss weights must be named explicitly.** `run_hindcast_folds.py` injects the frozen
product's `--ssim_weight 0.2 --laplacian_weight 0.3 --histogram_weight 1.0` into every fold
command and `--extra_train_args` is appended last, so anything `BASE_ARGS` does not name is
silently inherited. `scripts/dist_base_args.sh` holds the baseline once and `verify_loss_weights`
reads it back out of each fold log. Every global fold was verified this way.

### 1.5 The shipped configuration

Experiment **e1**, seed 46:

```
--head_family spline --central_residual True --central_context True --quantile_context True
--checkpoint_monitor val_crps --ssim_weight 0 --laplacian_weight 0 --histogram_weight 0
--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean,max
--weight_avg_last 20 --seed 46
```

`e1` is **the best available configuration, not a validated improvement**: ten variants were
screened against a six-seed floor on Africa and none cleared the bar. See
`docs/dist_model_phase2.md`.

### 1.6 How out-of-sample predictions are produced

**k = 5 fold cross-validation** against `fold_mask_b4_1000.tif` — 512 px contiguous blocks, not
the 128 px checkerboard the frozen product used. The fold tile must exceed one residual
correlation length or held-out skill is optimistic; the residual's fitted practical range is
99–166 px, so a 128 px tile leaves every held-out pixel ringed by trained-on neighbours.

Each fold model trains with its own fold held out entirely *and* one further fold reserved for
validation, then predicts only its own fold's pixels. `stitch_fold_predictions(mode="holdout")`
keeps, at each pixel, the prediction from the model that never saw it.

**`holdout` for anything scored; `mean` for display.** A mean-stitched raster averages all five
folds everywhere — seamless, and in-sample at every pixel. The two are one flag apart and must
never be confused. The forward product has no mosaic and no seam.

The holdout mosaic is a hard join on the fold boundary. Measured on Africa: the step across a
boundary against the step inside a fold is 1.08x for the central field and the lower bound, and
**2.01x for the upper bound** — the folds agree on the centre and disagree on the width. Most of
that disagreement is optimisation noise rather than data: retraining one fold with only the seed
changed reproduces 85% of the fold-to-fold spread at h=20.

### 1.7 Row-banded prediction

The prediction path accumulates overlapping 128 px tiles with distance-to-edge weights, at stride
64. It holds `len(active_horizons) × (3 + n_qf_levels)` full-window float32 accumulators — **268**
for a four-horizon window at 64 quantile levels. On the global grid each is 2.55 GiB, and the
pages a fold's tiles actually touch come to **136 GiB for one fold**, against 125 GB of RAM with
two folds running at once. On Africa's 63.1 Mpx grid the same accumulators are ~22 GB, and the
only configuration ever run globally before this was the 12-accumulator triple head.

`--predict_row_chunk N` processes the region in bands of N rows. A band keeping rows `[a, b)`
accumulates every tile that covers one of them — every tile start in `(a − 128, b)` — so the
blend weights over the kept rows are complete and the written values are identical to an unbanded
run. Verified against an unbanded control: **max difference 1.19e-07** across all outputs
including the 64-band quantile raster, where merely changing `--predict_batch_size` moves results
by 1.15e-05. Production uses 512 rows: **25.6 GiB per fold**.

**Outputs must declare `BIGTIFF=YES`.** GDAL's `IF_NEEDED` default cannot promote a *compressed*
raster because it cannot predict the compressed size, so every prediction output was a classic
TIFF capped at 4 GiB. The global forward quantile rasters want ~14 GB and died at
`TIFFAppendToStrip: Maximum TIFF file size exceeded` partway through, leaving files that read
back as **finite zeros** past the failure point. The k=5 hindcast escaped by 27 MB — its largest
quantile raster is 4,267,991,319 bytes against a 4,294,967,296 ceiling.

### 1.8 The forward model

The forecast is **not** a fold model and **not** a mosaic. `--train_all_splits` trains on every
valid chip in the split mask rather than the 70% training split: a forward model has no held-out
geography to protect, so restricting it discards 30% of the world for nothing. The flag is
ignored under `--exclude_fold`, so it cannot pull a held-out fold back into training.

Validation still runs on split 2 and is **in-sample by construction**. That is unavoidable for a
production model and is the reason the configuration is validated by k-fold instead.

The fingerprint is the banner `PRODUCTION MODE: training on EVERY chip in the split mask` with
`FOLD-CV MODE` absent. Nothing else would reveal a forward model that had quietly trained on
split 1 — it has no held-out score.

---

## Part 2 — The ensemble

### 2.1 Why an ensemble at all

The model gives every pixel its own distribution, and that is enough to answer any question about
one pixel. It says nothing about how pixels co-vary, so it cannot answer a question about a
*region*: "how much new development might this catchment see?" needs to know whether neighbouring
errors are independent (they are not) or move together (they do).

Drawing each pixel independently would produce white-noise maps whose regional aggregates are
absurdly over-confident. The ensemble's only job is to supply that dependence.

### 2.2 What a member has to satisfy

400 maps, each satisfying two things at once:

1. **Each pixel's value is drawn from that pixel's own quantile function** — the model's, read
   directly, with no reshaping. Across members a pixel should reproduce `Q(u)` exactly.
2. **Neighbouring pixels are wrong together.** A model that under-predicts a city's expansion
   under-predicts it across the whole city, not at every second pixel.

The tool is a **copula**:

> **Step A.** Generate a spatially correlated field of standard-normal values — individually
> N(0,1), correlated with their neighbours in the right way. Call it **z**.
>
> **Step B.** At every pixel, convert that pixel's `z` to a probability and push it through that
> pixel's own `Q(u)`.

Step A carries all the spatial structure, step B all the per-pixel calibration. They are
completely separable, which is what makes this tractable at 184.6 M pixels — and it means the
ensemble **cannot fix a marginal defect**. If the model's 95% interval covers 0.88, so will the
ensemble's. An ensemble scorecard showing near-nominal coverage where the model showed 0.88 is
evidence of a bug, not of an improvement.

`--qf_dir` is what makes step B read the model's own quantile function. Without it the sampler
falls back to a two-piece normal the members were never drawn from, and T3/T5 score a marginal
that does not exist. `run_dist_ensemble_variant2.sh` passes it; the older loop script did not.

### 2.3 Describing the spatial structure — in PIT space

The dependence model is a copula, so the quantity whose spatial structure must be reproduced is
**the observation's probability rank under the forecast**, `Φ⁻¹(F_qf(y))` — not a residual in HM
units, and not a residual divided by an interval half-width.

The frozen product fitted `(y − central)/σ` with σ read off three of the sixty-four stored
levels, which assumes the rest is symmetric. HM is bounded below, a large share of the world sits
in `[0, 0.01)`, and the atom at HM = 0 is real, so that assumption fails hardest in the pixels
that dominate the count — and those pixels set the nugget and the shortest-range weight.
Measured: the two spaces disagree by 0.10–0.15 of total variance on the 4–50 px share against a
realisation-noise floor of 0.044, and the width-standardised field carries a 0.62σ region-wide
mean at h=20 where the PIT field carries 0.03.

The fit is a non-negative least-squares mixture over a fixed basis of correlation ranges —
**1.5, 2.5, 4, 6, 9, 14, 25, 50, 120, 300 px** — plus a **nugget**, matched to the **radial power
spectrum** rather than the variogram. A variogram is evaluated at lags and two long-range
components can absorb the fit while leaving the middle empty, producing smooth blobs where the
truth is structured speckle.

**Kernel: Matérn with `nu = 0.5`** — the exponential kernel, rough at short range. A Gaussian
kernel produces infinitely smooth fields and cannot make texture.

**No hand-added long-range component.** The frozen product appended a range-1500 px component at
weight 0.40 to represent the frequency-zero variance a mean-subtracted spectrum is blind to. That
weight was tuned on southern Africa; measured across forecast origins the k=0 variance it stands
in for is **0.002–0.009**, 45–200x smaller, and what the residual actually carries at k=0 is a
*bias*, not a spread — which a variance component cannot represent. `--long_weight 0`.

The ten-range basis is not over-specified: dropping the 4 px and 25 px components sends the
structure score below white noise.

### 2.4 Building the field

Factorising an N×N covariance matrix is impossible at this scale. **Circulant embedding** avoids
it: if correlation depends only on separation, frequencies are independent in Fourier space. So
compute the target spectrum `S`, draw white noise, Fourier-transform, multiply each frequency by
`sqrt(S)`, transform back. Exactly the requested covariance, two FFTs.

The grid is **padded** before the transform, because an FFT treats the map as wrapping and would
otherwise correlate the left edge with the right. On the global grid **longitude wrap is
deliberately enabled** — there the wrap is physically correct, the antimeridian being a real
join. `generate_ensemble.py` detects this from the grid (width ≥ 39000, origin at −180); an
explicit `--wrap_lon False` overrides the detection and would put a seam down the Pacific in
every member.

### 2.5 The copula is Student-t, not Gaussian

Gaussian copulas under-produce compound regional extremes: they make many places moderately bad
but rarely make one whole region very bad. Measured directly — the observation's standardised
position in the member ecoregion-mean distribution has **kurtosis 4.2–5.2** at h=10/15/20 against
3.0 for a Gaussian.

The field is therefore drawn as a **Student-t copula with `nu = 7`**, `nu` inverted from that
measured kurtosis via `3 + 6/(nu−4)`, which gives 6.7–8.9. The chi-square scale factor is drawn
**stratified** rather than i.i.d.: an i.i.d. draw at M=400 leaves the realised marginal biased
(+0.0033 at u=0.944, intervals ~6% narrow) purely as a finite-sample artifact.

The independent-pixel **null** ensemble stays Gaussian by design — it exists to isolate the
effect of spatial structure, so its marginal must be the only thing it shares with the members.

### 2.6 Coupling the horizons

A member is one story about the future, so its four horizons must hang together. They are chained
by an **AR(1)** process on the normal-score fields:

```
z(h) = ρ · z(h−1) + sqrt(1 − ρ²) · ε(h)
```

with ρ measured from the residuals' own between-horizon correlation, never assumed. The
construction keeps each `z(h)` marginally standard normal, so **ρ changes how horizons hang
together without changing the spread at any one of them**.

An exactly separable alternative was tested and rejected: the AR(1) chain's spectral distortion
is at the noise floor, while forcing a shared spectrum costs 0.110 of the 4–50 px variance share.

### 2.7 The one guarantee

The ensemble's realised quantile function **is** the model's published one — that is what
`--qf_dir` buys, and it is checked directly rather than assumed
(`scripts/check_qf_ensemble.py --mode marginal`). The frozen product's guarantee was different
and weaker: it pinned the median and the two bounds to the published rasters because its marginal
was a fitted approximation to them. Here there is nothing to pin, because there is nothing
fitted.

### 2.8 Storage and reproducibility

Members are stored as **int16 with scale 1/32767** — lossless to 3e-5 on a [0,1] quantity.

Ensembles are **icechunk repositories** holding one array `members` of shape
`(400, 4, 17111, 40000)`, chunked `(1, 1, 1024, 1024)` — one member per chunk. The chunk shape is
deliberate: a partial-chunk write in icechunk is a read-modify-write that retains every version
until garbage collection, so writing one member into a shared chunk multiplies the store size
several-fold. One member per chunk also means two writers can never share one.

The write is a **single transaction**: each GPU worker writes through a forked session and the
parent merges and commits once, so a killed run leaves no store at all rather than a directory
that reads back as plausible sentinel values. **Every member's seed is recorded in the manifest**,
so any ensemble regenerates exactly and a superseded one can be deleted and rebuilt.

---

## Part 3 — Published products

| product | format | contents |
|---|---|---|
| hindcast triple | 12 COGs | w2000 → 2005/2010/2015/2020 × lower/central/upper |
| forecast triple | 12 COGs | w2020 → 2025/2030/2035/2040 × lower/central/upper |
| hindcast quantiles | icechunk | `(time=4, quantile=64, latitude=17111, longitude=40000)` int16 |
| forecast quantiles | icechunk | same shape, base year 2020 |

All ten hindcast (base, target) pairs are stitched, scored and used to fit the residual spectrum.
**Only the w2000 window ships**: it is the only one reaching +20 yr, so it is the only window
whose store spans the full 5/10/15/20 horizon set. The others are measurement inputs.

The quantile stores carry named dimensions and real coordinates — `time` is the target year,
`quantile` the `u` level each band stands for, `latitude`/`longitude` the pixel centres, with
`base_year` and `horizon` as auxiliary coordinates along time. Chunks are `(1, 64, 512, 512)`:
all quantile levels together, so reading one pixel's whole distribution is a single chunk read.

---

## Summary of fitted parameters

Everything below is estimated from the global out-of-sample hindcast. Nothing is carried over
from any other extent.

| parameter | value |
|---|---|
| fold mask | 512 px contiguous blocks, k = 5 |
| spline knots | `default14` — 15 knots, 14 bins, 29 params/horizon, fixed and tail-dense |
| quantile output grid | 64 levels, normal-spaced, 0.025/0.5/0.975 pinned, `u ∈ [2.5e-4, 0.9999]` |
| width across horizons | cumulative softplus increments — cannot narrow with lead time |
| context channels | 12 — change context @ 3,30,100 px; HM mean,max @ 3,30,100 px |
| loss | CRPS + auxiliary central MSE; SSIM/Laplacian/Histogram all 0 |
| checkpoint | monitor `val_crps`, weights averaged over the last 20 epochs |
| post-hoc calibration | **none** |
| fitted marginal | **none** — the model's quantile function is the marginal |
| spectral fit space | PIT, `Φ⁻¹(F_qf(y))` |
| spectral basis (px) | 1.5, 2.5, 4, 6, 9, 14, 25, 50, 120, 300 |
| kernel | Matérn, `nu = 0.5` |
| long-range component | **none** (`--long_weight 0`) |
| copula | Student-t, `nu = 7`, chi² factor drawn stratified |
| null ensemble | Gaussian, independent pixels, same marginals |
| horizon coupling | AR(1) on normal scores, ρ measured per horizon |
| longitude wrap | enabled on the global grid (physically correct) |
| ensemble size | M = 400 |
| storage | int16 × 1/32767, icechunk, chunks (1, 1, 1024, 1024) |
