# Next Phase — Marginals That Hold *Within* a Member

Branch `ensemble`, written 2026-08-15. Companion to `docs/current_progress.md` (status) and
`docs/central_field_baseline.md` (measurement record).

The previous phase fixed the central field: skill against persistence went from
−1.23/−0.24/+0.01/+0.05 to +0.09/+0.15/+0.18/+0.19 across horizons, and the scorecard from
73/137 to 91/128. This phase targets what that exposed.

---

## 1. The finding this phase exists to fix

Every realism diagnostic so far pooled all members together. Scored **per member** — which
is the right question, because a member is meant to be one plausible realisation of the
world — the ensemble fails badly.

`scripts/member_distance_relationship.py`, k=5, M=100, southern Africa:

> **The observation falls inside the member 5–95% range in 20% of (year × band × family)
> cells. A calibrated ensemble would give ~90%.**

The rank of the observation among 100 members, +20yr, `P(Δ > +0.05)`:

| band | observed | member mean | rank/100 |
|---|---|---|---|
| 0–1 px | 0.1767 | 0.3752 | **0/100** |
| 1–3 px | 0.0605 | 0.2561 | **0/100** |
| 3–10 px | 0.0234 | 0.1537 | **0/100** |
| 10–30 px | 0.0078 | 0.0254 | **0/100** |
| 30–100 px | 0.0024 | 0.0003 | **100/100** |

Rank 0 means *every* member produced more change than reality; rank 100 means every member
produced less. Not an off-centre spread — no member is on the right side at all. Pooling hid
this completely, because a pool that is uniformly too hot still overlaps the truth.

The **shape of the decay** is wrong too, and its error changes sign with lead time. Near/far
contrast `P(0–1px)/P(3–10px)`, observed vs per-member median [5–95%]:

| year | observed | two-piece normal | empirical shape |
|---|---|---|---|
| +5yr | 4.3 | 244.7 [135, 645] | 120.8 [68, 293] |
| +10yr | 7.2 | 9.2 [6.8, 14.2] | 7.0 [5.0, 11.1] |
| +15yr | 7.8 | 4.3 [3.1, 5.8] | 6.0 [4.0, 8.6] |
| +20yr | 7.5 | 2.5 [1.8, 3.2] | 4.4 [2.9, 6.3] |

Observed contrast is roughly flat at 4–8 across horizons. The members swing from 245 down to
2.5 — far too clustered at short lead times, too diffuse at long ones.

## 2. Why this is the marginals, and not the correlation field

An initial reading blamed the field's correlation structure. That was wrong, and the
arithmetic says so.

Per-band half-width against the residual it must cover, +20yr:

| band | median `w_up` | observed \|residual\| p97.5 | ratio | **+0.05 in units of `w`** |
|---|---|---|---|---|
| 0–1 px | 0.156 | 0.105 | 1.49 | **0.32×** |
| 1–3 px | 0.124 | 0.077 | 1.62 | 0.40× |
| 3–10 px | 0.091 | 0.058 | 1.58 | 0.55× |
| 10–30 px | 0.036 | 0.024 | 1.48 | 1.38× |
| 30–100 px | 0.0085 | 0.0047 | **1.83** | **5.88×** |

Two things follow.

**The width error is essentially uniform across bands** (1.48–1.83), so the *relative*
spatial pattern of the widths is roughly right and the field is not misplacing change. What
is wrong is the level — every interval is ~1.6× wider than the residual's own 97.5th
percentile, which is also what T1.1 reports as pooled coverage 0.963–0.982.

**But the last column explains the wildly non-uniform band errors.** The +0.05 threshold sits
at 0.32 half-widths in the near field and 5.9 in the far field. A uniform width error is
therefore mild where the threshold is inside the interval and catastrophic where it is far
outside — 2–7× too hot in the near/mid bands, 25× too cold beyond 30 px. One global width
factor cannot fix both, which is exactly what the `k*` experiment demonstrated: forcing
T1.1 to 4/4 destroyed T1.2 (13/16 → 8/15), T1.3 (3/3 → 0/2) and T8.4 (4/5 → 0/5).

Beyond 30 px the residual is a near-zero body with rare genuinely large values. The interval
is simultaneously too wide for the body (ratio 1.83) and unable to represent the tail
(`P(Δ>0.05)` at 0.04× observed). That is not a scale problem. **It is a shape problem, and
the shape needs to depend on distance to past change.**

## 3. What already exists to build on

`src/ensemble/copula.py` gained an empirical marginal shape this session
(`fit_residual_shape` / `apply_shape` / `invert_shape` / `shape_slope`, flag
`--marginal_shape`). It replaces the two-piece normal's Gaussian body with the residual's
own standardized quantile function, normalized so u = 0.025/0.5/0.975 still land exactly on
lower/central/upper — so T5.1 and T5.2 hold by construction, the map is strictly monotone
(the copula's rank structure and every spatial property are untouched), and a genuinely
Gaussian residual returns the identity.

**It is fitted once per horizon, pooled over all pixels.** That is the limitation this phase
should lift. Measured effect of the pooled version at matched M=400: scorecard 90/126 vs the
two-piece's 85/126, T6.1 ratios 3.84/5.65/6.75/6.65 → 1.74/2.88/4.36/4.55, and 12/40 vs 8/40
cells with the observation inside the member range. Better, but nowhere near calibrated.

Two properties are worth preserving in anything that replaces it:

- **the three gate quantiles must be exact knots** — interpolating near them leaves a bias
  that does not shrink with M (this cost a full M=400 run to discover);
- **beyond the 95% bound, revert to the two-piece normal.** The empirical tail there is
  dominated by pixels whose *width* is near-degenerate, so the ratio explodes for reasons
  about the denominator. Taking it at face value sent z = 3 to 5.4 half-widths.

## 4. The plan

**H1 — condition the marginal shape on distance to past change.** The primary hypothesis.
Fit `fit_residual_shape` per (horizon × distance band) instead of per horizon, apply per
pixel by band. Cheap: the residual rasters and the distance covariate are already on disk,
and the fit is a quantile grid. Success is the far-field band gaining a representable tail
without the near field inflating.

**H2 — fix the width level, per class rather than globally.** Widths are ~1.6× too wide
uniformly. A global factor is known to fail (§2). The interval-score decision rule chose
identity by a 0.03% margin over global, so it is nearly indifferent — worth re-running with
the class axis being *distance band* rather than Δ̂-bin, which is the axis the failures
actually organise along.

**H3 — check whether H1 and H2 are the same knob.** A distance-conditional shape changes the
effective spread as well as the shape. Fit both, and test whether the width correction is
still needed once the shape is conditional.

**H4 — if the above stalls, the widths are the model's.** The quantile heads already receive
the past-change context; they are producing intervals whose *level* is 1.6× too generous
across the board. That is a training-objective question (pinball at 0.025/0.975 on a spiky
target) and would mean reopening the head training, which is more expensive than H1–H3 and
should not be started first.

### How to judge it

**Primary metric: the fraction of (year × band) cells where the observation lies inside the
member 5–95% range, currently 20%.** Report the rank distribution, not just the fraction —
ranks pinned at 0 or M are the specific failure. `scripts/member_distance_relationship.py`
produces both.

Secondary: the near/far contrast per member against the observed 4–8, and the T1–T8
scorecard via `run_region_loop.sh`. Score at **M ≥ 400** — MC-scaled tolerances tighten as
1/√M, so a lower member count flatters whichever family is under-resolved, and the M=100
comparison between marginal families was confounded for exactly this reason.

Anything touching the marginal must keep T5.1/T5.2 exact and stay monotone in z. If a
diagnostic needs the normal score back, it must invert the shape (`invert_shape`), or it
measures a nonlinearly distorted field.

---

# 5. Results — H1 measured, 2026-08-15

**H1 is a negative result. The knob it exposed is the positive one.** Conditioning the
marginal shape on distance band does not improve the primary metric; raising the *tail
bound*, which only became adjustable because H1 forced an examination of the tail, does.

## 5.1 The measurement instrument came first

Before any code changed, `scripts/predict_change_rates.py` was written to answer the
scoring question in closed form. Under a per-pixel marginal the member *mean* of
`P(Δ > thr)` is exactly

    E_pixel[ 1 - Phi( S^-1( Z975 * (thr - dhat) / w ) ) ]

which needs no ensemble at all — the field's correlation changes how much members scatter
around that mean, not the mean. It runs in ~30 s against 3 min of generation plus 18–32 min
of validation, and it agrees with the sampled ensembles it predicts:

| | 0–1 px | 1–3 px | 3–10 px | 10–30 px | 30–100 px |
|---|---|---|---|---|---|
| two-piece, predicted | 0.38246 | 0.26227 | 0.15969 | 0.02805 | 0.000291 |
| two-piece, M=400 measured | 0.38005 | 0.25934 | 0.15533 | 0.02581 | 0.000277 |
| pooled shape, predicted | 0.28827 | 0.13256 | 0.07021 | 0.01534 | 0.000135 |
| pooled shape, M=400 measured | 0.28654 | 0.13054 | 0.06717 | 0.01368 | 0.000125 |

Two implementations that share no code — a closed-form CDF and 400 sampled int16 members
through the GPU copula — agreeing to 1–11% is the check that makes everything below
trustworthy. **Use it before generating anything.** It also caught the out-of-bounds gather
that a per-band stack introduces for the `>100 px` band, and it is what turned a day of
`u_bound` A/Bs into a thirty-second sweep.

## 5.2 The distance band had two definitions, and they disagreed

`right=True` at four call sites including the primary judge; `right=False` at four others
including T8. The distance raster comes from an exact Euclidean transform, so `dist == 1.0`
is one of the most populated values on the map, not a measure-zero edge case. Scored the
other way the 0–1 px band's observed `P(Δ>0.05)` reads **0.222 rather than 0.177** — a 26%
difference in the number the marginal is being fitted to reproduce, and §1's table and
`src/ensemble/validate.py`'s own docstring were quoting different conventions.

Now one definition: `src.ensemble.validate.distance_band`, `right=True`, used by all eight.

## 5.3 The primary metric, M=400

Observation inside the member 5–95% range, 20 (year × band) cells:

| configuration | cells inside |
|---|---|
| two-piece normal | 2/20 |
| pooled shape, bound 0.975 (shipped) | 7/20 |
| **pooled shape, bound 0.999** | **9/20** |
| per-band shape, bound 0.999 | 7/20 |

The rank distribution is the more informative half. Under both shipped families the mid and
far bands were pinned at 400/400 — every member too cold — in almost every cell. Raising the
bound un-pins them:

| year × band | two-piece | pooled 0.975 | pooled 0.999 | per-band 0.999 |
|---|---|---|---|---|
| 2005 · 3–10 px | 400 | 400 | **181** | **140** |
| 2005 · 10–30 px | 400 | 400 | **321** | **290** |
| 2010 · 10–30 px | 398 | 398 | **213** | **131** |
| 2015 · 10–30 px | 202 | 329 | **136** | **62** |
| 2020 · 10–30 px | 0 | **67** | 9 | 6 |
| 2020 · 30–100 px | 400 | 400 | **356** | **76** |

**The band axis is not additive.** At the same bound it makes every cell hotter — 2020
30–100 px goes from rank 356 to 76 — and costs two cells net. The analytic score agrees:
mean |log₁₀ ratio| over all cells is 0.607 pooled@0.975, **0.394 pooled@0.999**, 0.455
per-band@0.999.

Corroborated on a completely different spatial sampling: `compare_marginal_renders.py`
scores 768 px windows rather than distance bands, and pooled@0.999 is closer to the observed
rate than per-band@0.999 in **16 of 16** window × year cells, on both thresholds. Two
diagnostics that partition the map differently agreeing this cleanly is what makes the
negative result safe to act on.

## 5.4 Why the band axis cannot help the way §2 expected

`fit_residual_shape` normalizes each side by *that side's own* 2.5/97.5 quantile and
multiplies by `Z975`. That is precisely what pins T5.2 — and it means the fitted shape is
not the residual's distribution, it is the residual's distribution **stretched to fill the
published interval.** Measured stretch, `Z975/hi_ref`, at h=20:

| band | up | down |
|---|---|---|
| 0–1 px | 1.30 | 1.51 |
| 1–3 px | 1.42 | 1.56 |
| 3–10 px | 1.86 | 1.49 |
| 10–30 px | 3.10 | 2.41 |
| 30–100 px | 3.27 | 7.06 |

So **any T5.2-preserving marginal re-injects the width error it was meant to fix.** Two
consequences:

- Conditioning on any axis that acts mainly on *scale* is nulled by construction — which is
  why sub-conditioning on Δ̂ quintiles (0.335) or `w_up` deciles (0.342) moves nothing
  against the band shape's 0.344. Do not spend another pass on more covariates.
- The near bands stay pinned at rank 0 in every family, because the stretch is what makes
  them hot. **H2 is not optional, it is the only remaining lever on that failure.** The
  ideal-marginal calculation — the band's own ECDF against each pixel's own threshold —
  lands at 1.10/1.03/0.84 × observed in the near bands, against the fitted shape's
  1.98/2.95/3.93. The gap is entirely the stretch.

## 5.5 Two failures nothing here touches

**The lower tail.** `P(Δ < −0.01)` is over-predicted 2.8–4.3× in the near bands and **28×**
beyond 30 px at h=20 — the single worst cell in the table, and opposite in sign to the upper
tail in the same band. Most of it is not fixable by any per-pixel marginal: the ideal-ECDF
calculation still gives 3.05× at 0–1 px, so it is a dependence between the standardized
residual and the pixel's own threshold, which a marginal by definition cannot encode.

**The far band at short lead.** At h=5, 30–100 px, the +0.05 threshold sits at a *median of
119 half-widths* (p0.1 = 4.9). Only 0.17% of that band's pixels are within the fit's clip at
all. No marginal shape can bridge that; it says the published interval there is far too
narrow relative to what actually happens, which is H4 — the widths are the model's.

## 5.6 What to carry forward

1. **Ship the tail bound, not the band axis.** `--u_bound 0.999` with the pooled fit is the
   best-measured configuration. `--by_band` stays available and defaults on in
   `fit_marginal_shape.py`, but the loop should pass a pooled fit until something makes the
   band axis pay.
2. **H2 next, and it is now a quantified target, not a hypothesis.** The width factor that
   matches observed exceedance is 0.32/0.44/0.44/0.40/0.47 across bands at h=20 —
   near-uniform once the tail is representable, which *reverses* §2's reading (that argument
   was conditional on the two-piece marginal). But 95% pooled coverage wants ≈0.7, so a
   scalar cannot satisfy both, and the `k*` precedent is that width changes wreck
   T1.2/T1.3/T8.4. Score it as its own configuration with the T1 cost reported.
3. **Never sweep a marginal by generating ensembles again.** `predict_change_rates.py` is
   30 s and it was right every time it was checked.

## 5.7 The scorecard, and what the tail bound costs

Scored at M=400 against `validation_shaped400_fixed`, the recorded 90/126 baseline. **Score
that directory, not `validation_shaped400`** — the latter is an earlier run made *without*
`--marginal_shape`, so its `recover_z` never inverted the shape and T3.1 reads 0.61 instead
of 0.96. Comparing against it manufactures a T3.1 "gain" that does not exist; T3 is 3/6 for
every configuration tried.

**The binning fix alone moves nothing.** Re-scoring the same ensemble under `right=True`
gives 90/126 either way and flips zero rows. It shifts the T8.1/T8.4 band ratios by 5–20%
but nothing crosses a gate, so every family comparison below is clean.

| | baseline @0.975 | pooled @0.999 | per-band @0.999 |
|---|---|---|---|
| total | **90/126** | 84/129 | 83/129 |
| T5 hard gates | 11/12 | 11/12 | **9/12** |
| T8 | 7/12 | 6/12 | **5/12** |

Eleven rows flip between the baseline and pooled@0.999 — one gain, ten losses — and the
losses have three identifiable causes, which is what made them fixable:

| cause | rows |
|---|---|
| the **lower** bound was raised too | T6.1, T6.2, T6.3, T6.5, T8.4 |
| the remote band inherited a tail | T8.2 (0.0 → 0.00203, gate ≤0.002) |
| the tail extension itself | T4.2 (0.998 → 0.879) |
| a boundary moved under noise | T1.1 h=5 (0.9600 → 0.9616), T2.1 100 km ×2 |
| **the target** | T8.1 far band 0.0997 → **1.2508** |

## 5.8 The bound has to be asymmetric, and per band

Six of the ten losses were self-inflicted, and the predictor says why. At the −0.15 threshold
T6.3 scores, the **observed rate is 0.000000 in every band**, while bound 0.999 predicts
0.00165 — a thousandfold jump over 0.975. The two tails of this residual fail in opposite
directions: the upper is far too thin, the lower already 3–28× too hot. One symmetric knob
cannot serve both.

`u_bound` is therefore now two numbers per shape (`u_bound` / `u_bound_lo`), and the measured
configuration is:

> **pooled body; upper bound 0.999 out to 100 px and 0.975 beyond it; lower bound 0.025
> everywhere.** `--pooled_body true --u_bound 0.999,0.999,0.999,0.999,0.999,0.975
> --u_bound_lo 0.025`

Predicted, against the shipped 0.607 and the symmetric 0.394, it scores **0.377** — and the
per-quantity numbers show it is not a compromise but a strict dominance:

| | far band P(>+0.05) | remote band P(>+0.05) | 0–1 px P(<−0.15) |
|---|---|---|---|
| *observed* | *0.002410* | *0.000000* | *0.000000* |
| shipped @0.975 | 0.000135 | 0.000000 | 0.000001 |
| pooled @0.999 symmetric | 0.001530 | 0.001646 | 0.001647 |
| **asymmetric** | **0.001530** | **0.000000** | **0.000001** |

Confirmed on the generated M=400 ensemble: far band 0.001237, remote band exactly 0.000000,
`P(Δ<−0.15)` at most 3e-6.

On the primary metric it ties pooled@0.999 at **9/20 with every rank identical** (35, 93,
181, 321, 397 …). That is the point rather than a disappointment: the metric reads the upper
tail only, so an unchanged result is proof the asymmetry removed the collateral damage
without perturbing what was working.

Still not fixed, and not fixable here: the 30–100 px band stays pinned at rank ~400 at +5,
+10 and +15 yr. Only +20 yr comes inside. See §5.5 — at h=5 the threshold sits 119
half-widths out, which is a statement about the published width, not the marginal.

## 5.9 The asymmetric bound, scored

M=400, against the re-scored baseline so the binning is common to both:

| | baseline @0.975 | pooled @0.999 symmetric | **asymmetric** |
|---|---|---|---|
| **total** | 90/126 | 84/129 | **88/126** |
| T5 hard gates | 11/12 | 11/12 | 11/12 |
| T6 | 12/21 | 11/24 | **12/21** |
| T8 | 7/12 | 6/12 | **8/12** |
| T8.2 remote band | 0.0 | 0.00203 ✗ | **0.0** |
| T8.3 near/remote | ∞ | 141 | **∞** |
| T7.3 spread-skill | 1.439 | 1.528 | 1.456 |

The asymmetry recovered everything the symmetric bound had cost except one row. Against the
baseline only **four rows move**:

| | row | | |
|---|---|---|---|
| **GAIN** | T8.1 far band 30–100 px | 0.0997 → **1.2508** | the defect this phase exists to fix |
| LOST | T4.2 spread non-decreasing | 0.9981 → 0.8819 | the one real cost |
| LOST | T1.1 pooled coverage h=5 | 0.95999 → 0.96089 | a 0.0009 move across a gate edge |
| LOST | T2.1 block coverage 100 km (2010) | 0.9908 → 1.0000 | over-covering, i.e. wider than needed |

So the trade is one target row and the primary metric (7/20 → 9/20, mid bands un-pinned from
rank 400/400) against one genuine regression, T4.2, plus two rows that moved by less than
their own noise. **Read the rows, not the 90 → 88.**

**T4.2 is the open item, and it has a candidate cause.** Extending the upper tail adds member
spread at every horizon, and at h=5 that buys almost nothing — the far band there is
unreachable whatever the marginal does (§5.5), while the 3–10 px band does gain (0.09× →
1.27× observed). A bound that varies by *horizon* as well as by band is the obvious next
knob; the machinery is per band only today.

## 5.10 Standing recommendation

`SHAPE=measured` in `scripts/run_region_loop.sh` — pooled body, upper bound 0.999 out to
100 px and 0.975 beyond, lower bound 0.025 everywhere. Not the per-band *shape*: that was
H1, and it lost on every measurement taken.

---

# 6. The width level — H2 measured, 2026-08-15

§5.4 established that no T5.2-preserving marginal can fix the near bands, because the shape
normalizes to the published bounds and therefore re-injects their error. That makes the width
the only remaining lever. Four configurations were built and scored at M=400.

| configuration | primary metric | mean \|rank−200\| | scorecard | central skill h=20 |
|---|---|---|---|---|
| baseline, pooled shape @0.975 | 7/20 | — | 90/126 | 0.1909 |
| band-only widths | **16/20** | **110.3** | 90/125 | 0.1909 |
| **band × Δ̂ widths, uncentred** | 13/20 | 119.2 | **98/125** | 0.1909 |
| bias-corrected centre + widths | 14/20 | 128.9 | 91/122 | 0.1711 |

**Ship the band × Δ̂ uncentred widths.** 98/125 is the best scorecard this project has
recorded, and the rows it wins are the ones that matter: T1 goes 21/31 → 28/31, T8 8/12 →
11/12, T1.2's worst-cell deviation 0.209 → 0.033, T1.3 back over its gate.

## 6.1 What each attempt taught

**Narrow only where the tail survives it.** Narrowing every band scores *worse* than doing
nothing (0.702 against 0.458) — not a tuning failure but a mechanism: narrowing 30–100 px
moves the +0.05 threshold from 3.5 to 11.5 shape units while the fitted shape stops at the
clip, leaving 0.24% of the band reachable. Raising the clip recovers part of it (0.013 →
0.509) and still scores worse than leaving that band alone. 10–30 px overshoots at every
horizon (h=10 1.67× → 0.43×). **Bands 0–1, 1–3, 3–10 px only.**

**T1.1 was closable after all.** `docs/current_progress.md` recorded pooled coverage as "not
closable by recalibration… needs a model-level change to the width heads". Band-only widths
closed it at three of four horizons — 0.972/0.979/0.982 → 0.956/0.956/0.954 — with no change
to the heads. That entry should be revised.

**A width factor must not be centred on the residual median.** The first fit copied
`fit_residual_shape`'s centring, which is mandatory there (T5.1 pins the shape's median to
zero) and wrong for a width: the interval is centred on the *central forecast*, so it has to
cover the residual's bias as well as its spread. In the high-change class the standardized
residual has median −0.85, and centring read its 2.5 percentile as −1.28 (k_lo = 0.65)
instead of the true −2.13 (k_lo = 1.09). Acting on that narrowed a bound that was already
tight, and class coverage fell 0.95 → 0.73 with every miss on the low side (0.043 → 0.212 →
0.263). **It announced itself as a number that could not be true** — coverage falling while
the bound was widened — which is the third time this project has found a bug that way.

## 6.2 The central forecast has a conditional bias, and correcting it is the wrong move

Median standardized residual by predicted-change class:

| Δ̂ class | h=5 | h=10 | h=15 | h=20 |
|---|---|---|---|---|
| (−0.01, 0.001] | −0.03 | −0.02 | −0.02 | −0.03 |
| (0.001, 0.01] | −0.09 | −0.11 | −0.21 | −0.28 |
| (0.01, 0.05] | −0.47 | −0.29 | −0.29 | −0.30 |
| (0.05, 0.15] | — | — | **−0.79** | **−0.60** |

The bias is real and stable where it is large (h=15 (0.05,0.15] reads −0.0287 and −0.0255 on
its two windows) and unstable where it is small (h=5 (0.01,0.05] flips sign across windows).
`scripts/fit_central_bias.py` measures it and zeroes the unstable classes.

**Correcting it costs more than it buys, for three separate reasons.**

1. *RMSE follows the mean, coverage follows the median, and this residual is strongly
   right-skewed.* In the (0.05,0.15] class the median is −0.0233 and the mean −0.0117, so a
   median shift overshoots the mean by half its magnitude. Skill against persistence fell
   0.1909 → 0.1711 at h=20, giving back a tenth of what the previous phase won.
2. *The rows it was built to fix did not get fixed — they disappeared.* Shifting the centre
   moved those pixels out of the (0.05,0.15] class, so T1.2 and T1.3 at h=15/h=20 became
   unscoreable. 91/122 is 98/125 minus the four rows the correction was aimed at.
3. *The width factor already handles the bias correctly.* An uncentred factor makes the
   interval asymmetric about the unmoved centre (k_lo 1.07, k_up 0.62), which restores
   coverage at no cost to the central field. Displacing an RMSE-optimal centre to fix a
   quantile problem is paying in the wrong currency.

The machinery is kept, guarded and flag-gated (`--central_bias`, manifest records
`central_regenerated: true`), because the finding is worth having and a *mean*-based shift is
the untested middle — it would improve RMSE slightly and partly fix coverage. Not run.

## 6.3 Standing recommendation

Widths: `scripts/fit_width_factors.py` (bands 0,1,2; uncentred; shrinkage n/(n+2000) toward
the band factor) → `apply_recalibration.py --width_factors` → **rebuild residuals against
that output** → refit the shape → generate. The rebuild is not optional: the shape
normalizes to whatever half-width it is fitted on.

Marginal: unchanged from §5.10 — pooled body, upper bound 0.999 out to 100 px and 0.975
beyond, lower bound 0.025.

Still open: T4.2 (0.998 → 0.96), the 30–100 px band at h=5/10/15 (unreachable — that band's
interval is too *narrow* at short lead, unstretch factor 1.785 at h=5), and 10–30 px at
+20 yr. All three are width-head questions, i.e. H4.
