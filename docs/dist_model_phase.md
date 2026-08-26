# The distributional phase — an end-to-end quantile ConvLSTM

Branch `dist-convlstm`, off `ensemble`, started 2026-08-24. Screened on southern Africa.

**Status: the head works, sixteen runs are scored, and nothing is adopted.** The end-to-end
distributional model trains, publishes a quantile function that reproduces its own bounds
exactly, and beats persistence as a full predictive distribution. Three results are established
well outside any plausible noise band (§6). No *variant* survives replication, and §4.3 explains
why in a way that is more useful than a ranking would have been.

The goal is not to beat the frozen product. It is to replace four post-hoc layers — conformal
width scaling, per-class width factors, empirical marginal reshaping, and the horizon
monotonicity pass — with a model that emits the whole conditional quantile function
`Q_h(u | x)` and publishes it directly. Some loss of performance is acceptable for that.

The one place it might genuinely win is the far field. `docs/global_scorecard.md` records
`P(ΔHM > 0.05)` beyond 100 px at **0.034 of observed**, and
`docs/background/model_phase.md` §4.0 diagnosed it from the model side: *"the model is not
obviously wrong there; the deliverable cannot express what it knows. The lever is a
model-predicted tail quantity feeding the marginal's bound per pixel."* That is exactly what a
learned quantile function is.

---

## 1. What the data audit settled before anything was built

Measured on southern Africa (`region_to_predict_small.geojson`, 1,091,486 valid px), across
all four horizons:

| question | measurement | consequence |
|---|---|---|
| probability mass at exactly zero? | exact-zero Δ fraction **1.8e-6 to 7.3e-6**; smallest nonzero \|Δ\| is 3e-9 | **No hurdle.** A continuous spline, not the signed three-part model the spec held in reserve |
| atoms at the physical bounds? | HM is never exactly 0 or 1 (min 0.00029, max 0.950); `frac(Δ == −HM₀)` = **0** | The `[0,1]` support is a real constraint that is never *observed* active, so NLL has no atom to represent |
| how peaked is the target? | \|Δ\|<1e-4 for 25% of pixels at h=5 and 11.5% at h=20; \|Δ\|<1e-3 for 53% at h=20 | A very narrow central peak with heavy two-sided tails — the knot grid must be tail-dense |
| are decreases real? | `P(Δ<−0.01)` = 0.021 / 0.030 / 0.029 / 0.031 by horizon | Real and rare. T6.1's 4–9× over-emission is a property of the two-piece marginal's lower tail, and a support-bounded spline cannot reproduce it |
| is southern Africa blind to the far field? | the **>100 px band contains zero pixels** — not a zero rate, an empty band | No far-field verdict is obtainable here at all. Band 4 (30–100 px, 142,893 px) has a real observed `P(Δ>0.05)` of **0.00225** and is the furthest measurable proxy |
| does the 512 px fold mask work regionally? | fold coverage **20.5 / 21.3 / 26.3 / 17.8 / 14.1 %** | Use `fold_mask_b4_1000.tif`. Fold territory exceeds the residual correlation length, and coverage is balanced by luck of the window |

Observed per-band rates, h=20, the targets the exceedance rows are scored against:

| band | 0–1 px | 1–3 | 3–10 | 10–30 | 30–100 | >100 |
|---|---|---|---|---|---|---|
| `P(Δ>0.05)` | 0.1767 | 0.0606 | 0.0236 | 0.0077 | 0.00225 | *empty* |
| `P(Δ>0.01)` | 0.5908 | 0.2931 | 0.1176 | 0.0347 | 0.0111 | *empty* |
| `P(Δ<−0.01)` | 0.0380 | 0.0466 | 0.0504 | 0.0184 | 0.0022 | *empty* |

---

## 2. The model

### 2.1 The head

Per pixel and horizon, over the **absolute HM level** at the target year:

```
Q_h(u) = clamp( anchor_h + s_h · g_h(u),  0,  1 )
```

| piece | construction | why |
|---|---|---|
| `anchor_h` | `HM₀ + f_c,h(trunk, ctx)`, output conv zero-initialised | The existing `--central_residual` skip, unchanged. At initialisation the model is exact persistence, which is the right prior when 53% of pixels move by less than 0.001 |
| `s_h` | `cumsum_h softplus(·)` | Non-negative increments accumulated across horizons: the 95% width **cannot shrink with lead time**, structurally, which is what T4.2 exists to check |
| `g_h(u)` | rational-quadratic spline on **fixed** u-knots, normalised `g = (v(u) − v(0.5)) / (v(0.975) − v(0.025))` | `g(0.5) = 0` and unit 95% width by construction, so `s_h` *is* the 95% width and `lower ≤ upper` is structural |

**Level rather than change is a coordinate choice, not a modelling one.** HM₀ is a known input
and CRPS is translation-equivariant, so scoring `Q` over the level against the observed level
is *the same objective* as scoring it over Δ against the observed change — measured at 3.3e-16,
float64 round-off. Level was chosen because the support box is then a constant pair of numbers
instead of two per-pixel tensors, because the head's clamp and the ensemble sampler's clip to
`[0, 1]` become one operation rather than two that can drift, and because the emitted rasters
are already in published units. The exception is SSIM and the Laplacian pyramid, which are not
translation-equivariant; they act on absolute HM, matching the incumbent and matching E3a,
which measured the change-field version and lost h=5/h=10 RMSE.

**The knots are fixed and tail-dense, not learned:**

```
U_KNOTS = [0, 0.001, 0.005, 0.025, 0.05, 0.10, 0.25, 0.50,
           0.75, 0.90, 0.95, 0.975, 0.99, 0.999, 1.0]
```

A standard rational-quadratic spline learns its abscissae by softmax and, with 14 bins, will
not place a boundary anywhere near `u = 0.999`. The defect this phase is aimed at lives at
exactly that depth. Fixing the abscissae buys the tail resolution; the learned heights then
decide how much mass goes there. `0.025 / 0.5 / 0.975` are exact knots, so the published triple
is a lookup rather than an interpolation — the same reason `copula.fit_residual_shape` pins
those three points today.

The shape head's output bias is initialised to `log h` of the **standard normal's** bin
heights, so the model starts at an exactly Gaussian marginal — `Q(0.999)` at 1.577 half-widths,
the Gaussian value — and every departure from Gaussianity afterwards is something training paid
for. A uniform initialisation would instead start at a shape nothing chose, whose tail reach is
an artifact of the knot spacing.

### 2.2 The objective

```
CRPS(Q, y) = 2 ∫₀¹ ρ_u( y − Q(u) ) du
```

CRPS is the right objective for *this* residual rather than merely a defensible one.
`model_phase.md` §6.3 measured Gaussian NLL here and it inflated the fitted widths by up to 7×:
the residual's kurtosis runs 928–20366, the log score's quadratic term is dominated by a handful
of extreme pixels, and `log σ` resists only logarithmically. Pinball has bounded influence per
pixel and cannot be dragged that way, and CRPS is an integral of pinball losses.

The quadrature splits at three points, all closed-form: the two support crossings where the
clamp takes over, and `u* = Q⁻¹(y)` where the pinball kink sits. Outside the crossings `Q` is
constant, so those two pieces are integrated without evaluating the spline at all.

**Tail weighting is safe in a way sample reweighting is not.** `∫ w(u) ρ_u(y − Q(u)) du` is a
positive combination of pinball losses and each is minimised at its own true quantile
*independently of `w`*, so a u-weighting moves where the optimiser spends effort and never the
target. It needs no importance correction. Reweighting *samples* does move the target, which is
why that knob lives in the dataloader with a correction beside it.

The published central forecast is **`E[Q]`, not the median.** The mean is the RMSE-optimal point
estimate and this residual is strongly right-skewed, so the two differ. One consequence worth
stating: because `E[Q]` depends on the whole distribution including its tail, the central
objective now reaches the width and shape heads as well. In the incumbent it touched only the
central heads.

### 2.3 What the model writes

Per (window, target year), the three historical rasters plus one new one:

| file | content |
|---|---|
| `w{base}_prediction_{year}_{lower,central,upper}.tif` | `Q(0.025)`, `E[Q]`, `Q(0.975)` |
| `w{base}_prediction_{year}_qf.tif` | 64-band int16 × 1/32767, `Q` on a fixed normal-spaced grid with the three gates pinned |

The triple being *derived from* the quantile function is what lets every existing reader run
unchanged. Measured: `qf[u=0.025]` reproduces the `lower` raster to **1.53e-5**, exactly half
the int16 quantum, i.e. agreement is exact before storage rounding; monotone in `u` across all
66.6M differences.

---

## 3. How anything gets judged

Three replicates of the baseline measure **this configuration's own** run-to-run floor, and the
bar is that a variant's margin beyond the baseline range must exceed that range's own width.
`model_phase.md` §6.1: a single new draw falls outside the range of three base runs roughly half
the time under the null, so "outside the range" is barely a test, and the looser bar produced
two false positives that the stricter one removed. `scripts/compare_dist_runs.py` applies it and
refuses to rank at all with fewer than two replicates, so a floor cannot be borrowed.

Ranking metrics, pre-registered: `crps_skill` (pooled and per band), PIT KS and the four
tail-coverage rows, and the exceedance `mean|log₁₀(pred/obs)|`. Central RMSE is a guard, not a
target.

### 3.1 The measured floor, and what it decides

Three replicates of D0 (`--seed 42/43/44`, folds 1 and 2). The band is the min-max range; a
variant must beat it by more than its own width. **`rel` is the band as a fraction of the
metric's level, and it is the number that decides whether a row can decide anything.**

| metric | min | max | width | rel | |
|---|---|---|---|---|---|
| `rmse5` | 0.00858 | 0.00872 | 0.00014 | **2%** | powered |
| `cov95_5` | 0.95430 | 0.97351 | 0.01920 | **2%** | powered |
| `cov95_20` | 0.95941 | 0.96194 | 0.00253 | **0%** | powered |
| `pit_ks5` | 0.25941 | 0.27368 | 0.01427 | **5%** | powered |
| `rmse20` | 0.01990 | 0.02090 | 0.00100 | **5%** | powered |
| `exceedance_abs_log10` | 0.43528 | 0.47848 | 0.04320 | **9%** | powered |
| `crps_skill5` | 0.11593 | 0.13950 | 0.02357 | **18%** | powered |
| `tail_reach20` | 4.15966 | 5.02268 | 0.86302 | **19%** | powered |
| `pit_ks15` | 0.18159 | 0.23778 | 0.05619 | 27% | weak |
| `crps_skill10` | 0.10041 | 0.15390 | 0.05349 | 40% | weak |
| `crps_skill15` | 0.10265 | 0.16219 | 0.05954 | 42% | weak |
| `pit_ks10` | 0.18484 | 0.29281 | 0.10796 | 43% | weak |
| `crps_skill20` | 0.10784 | 0.16498 | 0.05714 | 43% | weak |
| `pit_ks20` | 0.15060 | 0.38481 | 0.23421 | **89%** | weak |
| `skill5` (central) | 0.02076 | 0.05269 | 0.03193 | 88% | weak |
| `skill20` (central) | 0.01096 | 0.10317 | 0.09221 | **155%** | weak |

Three things this settles.

**The central field is stable run to run and the distribution is not.** RMSE moves by 2–5%
across seeds while CRPS skill moves by 18–43% and PIT KS at h=20 by 89%. That is the same split
rule 20 found on the incumbent — seed-to-seed spread on the *upper half-width* was 0.0308
against a fold-to-fold 0.0335, while the central field was stable to ~0.0003. The width and
shape heads are the unstable part of this model too, and it is worth saying plainly that this
is a property of the configuration at this training budget, not of the harness.

**Central skill cannot be ranked here at all.** Its band is 88–155% of its own level. Any
statement of the form "variant X improved central skill" is unsupportable on one run; RMSE, which
is the same quantity without the persistence denominator amplifying it, is precise to 2%. Use
RMSE, and treat skill as a sign check. (`model_phase.md` reached the same conclusion for the
same reason: skill = 1 − MSE/MSE_persistence and the margin over persistence is small, so a 4%
RMSE move becomes a 38% skill move.)

**The long-horizon distributional rows are under-powered and the h=5 rows are not.** h=5 draws
1.82M pixels from four input windows; h=20 draws 455k from one, because only w2000 reaches
+20 yr. That is a fixed property of the record, not something more compute fixes.

The consequence is a scope decision, taken here rather than after the fact: the slate is
**ordered by the size of the mechanism**, and D9 (fewer spline knots, asking whether the tail
resolution is load-bearing) is **not run**. Its expected effect is a fraction of a band 19% wide
at best, so a null would mean "under-powered", not "no effect" — a distinction this project has
already paid for once.

---

## 4. Results

*(the floor and the slate are filled in as they complete)*

### 4.1 What the first baseline run shows about *where* it is wrong

One run, `d0_s42`, folds 1 and 2, so none of this is a ranking. It is a diagnosis, and the
diagnosis is sharp enough to be worth recording before the floor exists: **the miscalibration
is conditional, not global.** Pooled at h=20 the model looks close to right — `cov95` 0.961,
deep-tail exceedance 1.9x nominal — while PIT KS reads 0.25 on 455k points, where the 5%
critical value is ~0.001.

Split by observed change magnitude, h=20:

| observed Δ | n | CRPS skill | PIT mean | cov50 | cov95 | tail reach |
|---|---|---|---|---|---|---|
| `< −0.01` | 16,453 | −0.03 | **0.023** | 0.000 | **0.287** | 2.16 |
| `[−0.01, 0.001)` | 342,564 | **−1.25** | 0.320 | 0.768 | **0.996** | 4.35 |
| `[0.001, 0.01)` | 61,007 | +0.03 | 0.614 | 0.617 | 0.999 | 3.53 |
| `[0.01, 0.05)` | 25,801 | +0.49 | 0.706 | 0.507 | 0.956 | 2.22 |
| `≥ 0.05` | 9,344 | +0.32 | **0.905** | 0.083 | 0.650 | 2.11 |

The interval is far too wide on the quiet three-quarters of the map — `cov95` 0.996 and CRPS
**worse than persistence by 1.25** — and far too narrow on the pixels that actually moved, in
both directions: real decreases sit at PIT 0.023 with 29% coverage, real large increases at PIT
0.905 with 65%. Split by HM level the same thing appears from another angle: the low-HM bins
carry `tail_reach` 4.4–5.9 with `cov95` 0.98, the high-HM bins 1.5–2.4 with `cov95` 0.86. **The
model is putting its heavy tails on the pixels where nothing happens and its thin tails on the
pixels where things do.**

Pooled PIT mean is 0.38–0.42 rather than 0.50 at every horizon, and `P(u > 0.975)` is 0.010
against a nominal 0.025 while `P(u < 0.001)` runs to 0.0078 at h=20 against 0.001. So the whole
distribution is also shifted upward: the model over-predicts change, which is what the too-hot
near-band exceedance rates say from the other direction.

Two things follow for reading the slate. First, this is exactly the defect the incumbent's
per-class width factors existed to correct, and an end-to-end model has to learn it from the
covariates instead — so the variants that change *what the optimiser attends to* matter more
here than the ones that change capacity. Second, a pooled metric would have called this model
roughly calibrated; the conditional rows are what make it legible.

---

### 4.2 The slate

Each variant is one run against the three-replicate baseline band of §3.1, and clears only if
its margin beyond that band exceeds the band's own width. Rows marked *weak* in §3.1 cannot
decide anything and are not used for verdicts.

| run | crps_skill5 | pit_ks5 | exceedance | rmse5 | tail_reach20 | verdict |
|---|---|---|---|---|---|---|
| baseline band | 0.11593–0.13950 | 0.25941–0.27368 | 0.43528–0.47848 | 0.00858–0.00872 | 4.15966–5.02268 | — |
| **d1a** | 0.14759 | 0.26121 | 0.38264 | 0.00856 | 4.40666 | **clears** exceedance by 1.2x the band; nothing degrades |
| **d1b** | 0.13509 | 0.27625 | 0.49302 | 0.00863 | 4.80830 | worse on pit_ks5 and exceedance |
| **d2** | 0.09907 | 0.20089 | 0.53585 | 0.00859 | 1.63052 | clears pit_ks5 by 4.1x; **tail destroyed**, CRPS worse at every horizon |
| **d2u** | 0.12523 | 0.26018 | 0.43162 | 0.00871 | 5.45862 | all inside the bands; worse on four h=20 rows |
| **d3** | 0.13963 | 0.25616 | 0.55236 | 0.00855 | 4.02557 | exceedance clearly worse; SSIM does not earn its place |
| **d4** | 0.13951 | 0.24465 | 0.42535 | 0.00858 | 2.69690 | clears pit_ks5 by 1.03x; tail worse by 1.7x the band |
| **d6** | 0.13416 | 0.28649 | 0.40324 | 0.00862 | 5.21529 | central RMSE inside a 2% band with no auxiliary MSE at all |
| **d7** | 0.05907 | 0.37929 | 0.47172 | 0.00859 | 4.96216 | distribution collapses, central field intact |
| **d8** | 0.13078 | 0.22640 | 0.43223 | 0.00861 | 2.97990 | clears pit_ks5 by 2.3x; tail worse by 1.4x the band |
| **n1** | -11.14491 | 0.98117 | 1.31621 | 0.02710 | 1.00000 | point mass at the width floor; worse on every scored row |

Read the table by column, not by row count. **Every variant that improves PIT calibration does
it by shrinking the tail** — d2 to 1.63, d4 to 2.70, d8 to 2.98, against a baseline band of
[4.16, 5.02]. That is not a coincidence between three unrelated knobs; it is the same fact seen
three ways. The miscalibration is dominated by the over-wide quiet majority (§4.1), so anything
that narrows the body improves the PIT statistic and takes the tail with it. The tail is the
property the phase exists to buy, so a PIT improvement bought that way is a loss, and
`tail_reach` is reported without a direction precisely because "higher" is not unconditionally
better either.

**d1a is the only variant that improves anything without that trade.**

**D6 — `--mu_mse_weight 0`, pure CRPS: a powered null, and the most useful result of the
slate so far.** With the auxiliary MSE removed entirely, `rmse5` reads **0.00862 inside a
baseline band of [0.00858, 0.00872] that is 2% wide**, and `rmse20` 0.02052 inside
[0.01990, 0.02090]. The central field is not paying for the removal. Exceedance improves to
0.403 against a band minimum of 0.435 — better, though by less than the band's own width, so
not a clear. PIT KS at h=5 worsens (0.286 against a band of [0.259, 0.274]).

This is the answer to the question the term was introduced to hedge. CRPS presses on `E[Q]`
only indirectly, and the worry was that the central field would decay without a direct
squared-error term; measured, it does not. The elegant configuration — one proper scoring
rule, nothing else — is available at no measurable central cost.

**D2 — stratified chip sampling with importance correction: clears one row and destroys the
tail.** PIT KS at h=5 improves to 0.201 against a band of [0.259, 0.274], a margin of 4.1x the
band's width — the largest single move in the slate, on the best-powered calibration row. But
`tail_reach20` **collapses from the band [4.16, 5.02] to 1.63**, barely above the Gaussian 1.577,
and on the quiet stratum from 5.29 to 1.85. CRPS skill is worse at all four horizons and
exceedance worsens to 0.536.

The mechanism is visible in the sampler's own numbers. The correction is `uniform / stratified`,
so the chips the sampler favours enter the loss at weight 0.115 and the ones it neglects at 5.0
— it *undoes* the stratification inside the objective, leaving the target unchanged and the
gradient estimator far noisier (a 43.6x sampling ratio). The tail is the noisiest quantity in
the model to estimate, and it is what was lost. **That makes D2u, the uncorrected run, the
informative contrast rather than an afterthought**: it keeps the exposure without the variance,
at the price of targeting a utility-weighted distribution.

**N1 — NLL: collapses to a point mass, and the direction is the interesting part.** The
95% interval reads **1.5e-7 HM** against the baseline's 1.9e-2 — five orders of magnitude
narrower, pinned exactly at `MIN_SCALE`, with `width95 == width99` so the whole spline has
degenerated. `crps_skill5` is **-11.1**, coverage at every level is 0.000, and PIT sits at 0 or
1 with nothing in between. There is no noise band to argue about.

**A prediction registered before the run, and confirmed.** `model_phase.md` §6.3 measured
Gaussian NLL on this data and it inflated the fitted widths by up to 7x, because the quadratic
term is dominated by rare extreme residuals while `log sigma` resists only logarithmically. The
prediction here was that the same objective on a *flexible* head would fail the other way: a
spline can absorb an outlier in its tail instead of paying for it in the body, so the log
score's cheapest move is to shrink the body onto the modal value — and 53% of pixels move by
less than 0.001 over twenty years, so that mode is extremely sharp. It shrank until the floor
stopped it.

So §6.3's finding generalises in mechanism but not in sign: **the log score is unusable on this
residual in either parameterisation, and which way it fails is a property of how much shape
freedom the head has.** CRPS is not merely the safer choice, it is the only one of the two that
works here — which is the positive form of the result.

`MIN_SCALE` earned its place in the same run. Without a floor the widths would have gone to
zero, the log density to infinity, and the failure would have arrived as NaN rather than as a
number that says exactly what happened.

**D1b — tail-weighted CRPS at lambda = 4: null to negative.** `tail_reach20` reads 4.81, inside
the baseline band, so weighting the upper quantiles four-to-one bought no additional tail reach
at all; PIT KS (0.276) and exceedance (0.493) both land just outside the band on the wrong side.
Worth stating positively: the head's tail is not limited by how much loss weight the upper
quantiles receive.

### 4.3 Replication, and the finding that overturns the bar

`d1a` was the only variant to clear a powered row cleanly, so it was replicated at seeds 43 and
44. It does not reproduce.

| metric | d1a x 3 replicates | D0 x 3 replicates |
|---|---|---|
| `exceedance_abs_log10` | [0.38264, **0.47166**] | [0.43528, 0.47848] |
| `crps_skill5` | [**0.07377**, 0.14759] | [0.11593, 0.13950] |
| `pit_ks5` | [0.21466, 0.26121] | [0.25941, 0.27368] |
| `rmse5` | [0.00856, 0.00868] | [0.00858, 0.00872] |
| **`tail_reach20`** | **[1.60783, 5.92745]** | [4.15966, 5.02268] |

Every band overlaps D0's. The exceedance win was one draw of three; a second draw of the same
configuration gave 0.47166, inside the baseline band.

**The last row is the important one.** `tail_reach20` varies by **4.3 within a single
configuration**, against the 0.86 the three D0 replicates suggested. The baseline band was an
under-sample — three draws that all happened to land in the heavy-tail mode. So the *width* of
every bar applied in §4.2 was too narrow, and every marginal call made against it is
unsupported.

Looking back at the slate with that in hand, the trade named in §4.2 dissolves into one fact.
Across all thirteen non-NLL runs, `tail_reach20` and `pit_ks5` correlate at **r = 0.691**, and
the runs sort along that axis almost perfectly:

```
tail_reach20   1.61  1.63  2.70  2.98  4.03  4.16  4.33  4.41  4.81  4.96  5.02  5.22  5.46
pit_ks5       .224  .201  .245  .226  .256  .259  .269  .261  .276  .379  .274  .287  .260
run          d1a43   d2    d4    d8    d3  d0_42 d0_43  d1a   d1b   d7  d0_44  d6   d2u
```

A run either learns a heavy tail — and pays for it with a body too wide for the quiet 75% of
the map, which is what PIT KS measures — or it does not. **d2, d4 and d8 did not "trade PIT for
tail"; they landed on the low end of an axis every run sits somewhere on**, and so did one of
d1a's own three replicates. Three knobs looked like three findings and were one property of the
configuration seen three times.

**Checkpoint selection is not the mechanism**, tested rather than assumed: the correlation
between the selected epoch and `tail_reach20` is **0.150**, with clear counterexamples both ways
(d7 selected epoch 31 and reached 4.96; d1a_s43 selected epoch 139 and reached 1.61). The
bimodality is in the training trajectory, not in which point of it gets picked.

---

## 5. Defects found while building, and what generalises

Three, none of which announced itself as an error.

**A degenerate u-grid made CRPS NaN for every pixel.** The normal-spaced output grid already
contains a point indistinguishable from 0.5, and pinning the gate quantile beside it produced a
zero-width segment whose slope is 0/0. The writer's `%.8g` tag then collapsed the pair, so the
reader's grid differed from the writer's. Fixed at three points: the grid dedupes and guarantees
exactly `n` strictly-increasing levels, the tag is written at full float64 precision, and the
reader refuses a non-increasing grid rather than passing NaN downstream. *Generalises: a metric
failure and a model failure look identical from the outside. Put the check where the cause is
one line away.*

**The CRPS closed form was wrong, by a relative error of ~1.** The crossing point clips to a
segment end far more often than it lands inside one, and the second piece's error at its own
origin is then not zero. Caught only by a dense-integral second implementation sharing no code
with it; now exact to 7.6e-8, which is the *reference's* trapezoid error. *Generalises: the
second implementation is not a luxury. This is the third time in this project that keeping two
has localised a bug instead of leaving a plausible number to be believed.*

**The stitcher copied the profile but not the dataset tags.** The quantile-function raster
carries its u grid in a tag, and that tag is the contract; nothing else in this pipeline has
ever had tags, so the omission was invisible until the first multi-band stitch. *Generalises:
the same shape as the global-scale defects — a code path that only the never-yet-run branch
reaches is untested by construction.*

And one that was not a defect but a working set:

**The quadratures held 22.4 GB of a 24.6 GB card** at the production batch size — 91%, with an
OOM waiting for the first unlucky chip mix twenty minutes into a fold. Gradient checkpointing on
the two spline evaluations: **8.65 GB, and 35% faster**, because the allocator pressure cost
more than the recompute. `train_loss` is identical, so the arithmetic is unchanged.

### The configuration drift that nearly invalidated the floor

`run_hindcast_folds.py` injects the incumbent's own loss weights
(`--ssim_weight 0.2 --laplacian_weight 0.3 --histogram_weight 1.0`) into every fold command, and
`--extra_train_args` is appended last, so argparse takes the final occurrence. **Anything
`BASE_ARGS` does not name is silently inherited from the frozen triple-head product.** The first
floor run trained with the spatial terms active while the slate defines D0 as the probabilistic
loss alone; every D3/D4 comparison against it would have been meaningless while looking healthy.
Caught at 35 minutes by reading the run's own `LOSS WEIGHTS` banner.

`scripts/dist_base_args.sh` now holds the baseline once for both drivers, and
`verify_loss_weights` reads the effective weights back out of the fold log and refuses to score
a run that trained on the wrong ones. Proven discriminating against the dead run's log. *This is
CLAUDE.md rule 25 in a new costume: "no extra flags" was argparse defaults; here "the flags I
passed" was not the same as "the flags that took effect".*

---

## 6. Conclusion — what is established, and what is not

**Nothing is adopted.** That is the same verdict the previous model phase reached across 22
experiments, and it would be a poor result if the reason were the same. It is not: this time the
reason is identified, measured, and actionable.

### What is established

Four results sit far outside any plausible noise band, and three of them are positive.

1. **The end-to-end distributional model works.** It trains, it beats persistence as a full
   predictive distribution at every horizon (`crps_skill` +0.10 to +0.16), and the quantile
   function it publishes reproduces its own 2.5/97.5 bounds to 1.53e-5 — exactly half the int16
   storage quantum, i.e. exact before rounding. The four post-hoc layers it was built to replace
   are not needed for it to produce a coherent product.

2. **Pure CRPS costs nothing centrally.** With `--mu_mse_weight 0` and no auxiliary squared-error
   term at all, `rmse5` lands inside a baseline band **2% wide**. RMSE is the one metric whose
   band is genuinely tight and trustworthy — it does not sit on the unstable axis of §4.3. The
   maximally elegant configuration, one proper scoring rule and nothing else, is available.

3. **The trunk must hear the distributional objective.** Isolating it (`d7`) leaves `rmse5` and
   `rmse20` inside their bands while `crps_skill5` falls to 0.059 (band 0.116–0.140) and
   `pit_ks5` rises to 0.379 (band 0.259–0.274). The shape heads cannot do the job from a trunk
   trained on the central loss alone. This is the end-to-end design measured rather than assumed.

4. **NLL is unusable on this residual, and fails opposite to the way it failed before.** The
   spline collapses to a point mass: 95% width 1.5e-7 HM against the baseline's 1.9e-2, pinned
   at `MIN_SCALE`, `crps_skill5` = −11.1. `model_phase.md` §6.3 saw Gaussian NLL *inflate*
   widths 7x; a flexible head lets the log score buy its outliers with a tail instead and shrink
   the body onto the mode. Same objective, opposite sign, both unusable — and the prediction was
   registered before the run.

And one diagnosis, from §4.1: **the miscalibration is conditional, not global.** The interval is
far too wide on the quiet three-quarters of the map (`cov95` 0.996, CRPS worse than persistence
by 1.25) and too narrow on the pixels that moved in either direction (real decreases at PIT
0.023 with 29% coverage; real large increases at PIT 0.905 with 65%). Pooled, the same model
looks nearly calibrated. This is the defect the incumbent's per-class width factors existed to
correct, and the end-to-end model has to learn it from the covariates instead.

### What is not established, and why

**No ranking among the knobs.** `tail_reach20` moves by 4.3 across three replicates of one
configuration (§4.3), so the bands used to judge the slate were too narrow and every marginal
call against them is unsupported. Six variants — d1a, d1b, d2, d2u, d3, d4, d8 — are
**undecided**, not rejected. The one that looked like a win failed its own replication.

**The far field, by construction.** Southern Africa has zero pixels beyond 100 px from past
change. `tail_reach` in the 30–100 px band is the only proxy available, and no verdict on the
defect this phase targets can be reached at this extent.

### What to do next, in order

1. **Fix the instability before running another slate.** It is the binding constraint: no
   screening protocol resolves a knob whose effect is smaller than a 4.3-wide band. This is a
   training-recipe question, not a model-architecture one. `--weight_avg_last` already exists,
   is untested here, and is exactly the kind of lever that collapses a bimodal trajectory; a
   longer schedule and a less noisy checkpoint selector are the other two candidates. Judge them
   on the *spread across seeds*, not on any single run's score.
2. **Re-measure the floor with more than three replicates**, and on `tail_reach` specifically.
   Three draws all landed in one mode and made the band look four times tighter than it is.
3. **Then re-run the undecided six.** They cost ~50 min each and none of them is refuted.
4. **Take the far field to a larger extent.** Nothing about it is answerable here.

Two things are worth carrying forward whatever happens to the variants: pure CRPS is free
centrally, and the trunk has to hear the objective. Both are properties of the design rather
than of a knob, and both replicate by construction rather than by luck.
