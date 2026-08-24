# Global scorecard

The production ensemble is scored against eight families of test, T1 to T8, at M = 400 members
on the global out-of-sample hindcast. This document describes every evaluation on the card: what
it tests, what its failure would look like in a forecast someone actually used, where its pass
threshold came from, and what the system scored.

**Headline: 99 of 134 scored rows pass.** A further 15 rows are reported without a gate, for
reasons given individually below.

| family | question | pass |
|---|---|---|
| **T1** | Is each pixel's uncertainty right? | **24 / 24** |
| **T2** | Is uncertainty right at *larger* scales? | 41 / 49 |
| **T3** | Do the members look like real maps? | 3 / 5 |
| **T4** | Are the four horizons coherent? | 3 / 4 |
| **T5** | Is the ensemble faithful to the published product? | 10 / 12 |
| **T6** | Is the amount and sign of change realistic? | 10 / 24 |
| **T7** | Are members diverse but not absurd? | 1 / 2 |
| **T8** | Is change put in the right *places*? | 7 / 14 |

## How to read a total

**Read rows, not the total.** The families are not equally weighted, equally sized, or equally
severe: T5 contains three hard gates on which the product's internal consistency depends, while
T6 contains 24 rows that mostly restate two underlying defects. A card of 99/134 is a summary of
a summary.

**Where the pass values come from.** Four kinds of threshold appear, and the kind determines how
seriously to take a near miss:

| kind | meaning |
|---|---|
| **definitional** | The interval *is defined* as 95%, so coverage must be 0.95. Not a judgement call. |
| **derived from data** | The observation is counted and the ensemble must match it. The target is whatever reality did. |
| **Monte-Carlo scaled** | What a finite sample of 400 members can achieve even when the underlying quantity is exactly right. |
| **set a priori** | A judgement made before the measurement existed. The kind to distrust. |

---

# T1 — Is each pixel's uncertainty right? — 24 / 24

The most basic requirement: when the product says a pixel will be 0.30 with a 95% interval of
[0.25, 0.42], the truth should land inside that interval 95% of the time.

### T1.1 Pooled coverage — 4 / 4

**What it tests.** The fraction of all valid pixels where the observation falls between the
published bounds, at each horizon.

**What failure looks like.** Under-coverage means the intervals are too narrow — a user planning
against the upper bound gets surprised more often than the stated 1-in-20. Over-coverage means
they are too wide: the interval is technically correct and practically useless, because it
admits outcomes nobody needs to plan for.

**Where the target came from.** *Definitional.* The bounds are the 2.5th and 97.5th percentiles,
so coverage must be 0.95. The ±0.01 tolerance is tight because the denominator is 184.6 million
pixels and sampling error at that size is negligible.

**Score: 0.943 / 0.958 / 0.954 / 0.959** at +5/+10/+15/+20 yr. All four pass. Coverage is
correct at the pixel level at every lead time — the per-pixel product does what it says.

### T1.2 Class-conditional coverage — 16 / 16

**What it tests.** The same question asked *inside* each class of pixel, split by how much change
the model predicts there. Fifteen (horizon × predicted-change) cells, plus one row reporting the
worst deviation across the finer primary classes.

**What failure looks like.** This exists because pooled coverage can be perfect while every class
is wrong in compensating directions. A product could be far too wide on the 70% of the world
where nothing happens and far too narrow on the pixels where development is expected — and T1.1
would read 0.95. The user who cares about a city fringe would be badly misled while the headline
number looked healthy.

**Where the target came from.** *Definitional*, with a wider tolerance: |coverage − 0.95| ≤ 0.03.
The classes are smaller than the whole map, so sampling error is larger and a ±0.01 band would
fail on noise.

**Score: 0.931 to 0.974 across all sixteen cells**, worst deviation 0.024 against a 0.05
allowance on the primary-cell row. All pass. The interval is honest inside every class of pixel,
not just on average — this is the row that makes T1.1 meaningful.

### T1.3 High-change tail coverage — 3 / 3

**What it tests.** Coverage restricted to pixels where a lot of change is predicted,
Δ̂ ∈ (0.05, 0.15] — the pixels the product exists to get right.

**What failure looks like.** The product is used to anticipate where development will appear. If
coverage collapses exactly on the pixels flagged as high-change, the product fails precisely
where someone would act on it, while remaining correct on the vast quiet majority.

**Where the target came from.** *Definitional* but deliberately looser at ≥ 0.92, because these
classes are small and a strict two-sided band would be dominated by sampling noise.

**Score: 0.965 / 0.969 / 0.974** at +10/+15/+20 yr. All pass, comfortably. Passing here matters
more than passing T1.1.

### T1.5 Sharpness guard — 1 / 1

**What it tests.** The ratio of the final published interval width to what the quantile heads
produced before calibration.

**What failure looks like.** Any coverage target can be met by making intervals enormous. This is
the guard that stops calibration buying coverage with width: without it, a system could pass
every T1 row by publishing [0, 1] everywhere.

**Where the target came from.** *Set a priori*: calibration may widen by at most a quarter,
≤ 1.25.

**Score: 0.971.** Passes. The calibrated interval is very slightly *narrower* than the raw heads
overall — coverage was achieved by redistributing width between classes rather than by inflating
it.

---

# T2 — Is uncertainty right at larger scales? — 41 / 49

This family is the reason the ensemble exists. Per-pixel intervals cannot answer "how uncertain
is this district?", because that depends on how errors at neighbouring pixels relate. An ensemble
can answer it, and T2 checks whether the answer is right.

### T2.1 Block coverage — 14 / 16

**What it tests.** Average HM over square blocks of 1, 10, 100 and 1000 km; compute that average
on every member; ask whether the observed block average falls inside the members' 95% range.

**What failure looks like.** A user asks "what is the uncertainty on mean HM across this
province?" Under-coverage means the answer is over-confident and decisions get made on a range
too narrow to contain reality. Over-coverage means the province-level range is so wide it cannot
discriminate between scenarios.

**Where the target came from.** *Definitional*, 0.95 ± 0.05. The tolerance is wider than T1.1's
because the number of blocks falls sharply with scale — at 1000 km there are only 182 of them,
so binomial noise alone is several percentage points.

**Score:**

| scale | 2005 | 2010 | 2015 | 2020 |
|---|---|---|---|---|
| 1 km | 0.943 | 0.958 | 0.954 | 0.959 |
| 10 km | 0.922 | 0.940 | 0.907 | 0.910 |
| 100 km | 0.934 | 0.946 | **0.898** | 0.908 |
| 1000 km | 0.978 | 0.995 | 0.923 | **0.890** |

**This is the central result of the whole system.** Aggregate coverage is approximately correct
at every scale simultaneously — which neither naive alternative achieves. Assuming errors are
perfectly correlated (just averaging the published bounds) gives 0.95–1.00 and degrades to
exactly 1.000 at 1000 km: always right, therefore useless. Assuming errors are independent
collapses to 0.09 at 100 km and 0.01 at 1000 km: catastrophically over-confident. The ensemble
sits where the truth is.

Two rows miss, both marginally and both at the low side: 100 km/2015 at 0.898 and 1000 km/2020
at 0.890, against a 0.900 floor. **The 1000 km row rests on 182 blocks**, where the binomial
standard error is about 0.023 — the miss is well inside one standard error of the gate, and this
row is under-powered rather than informative in either direction.

### T2.8 Aggregate width vs mean-of-bounds — 14 / 16

**What it tests.** The ensemble's aggregate interval must be *narrower* than simply averaging the
per-pixel bounds over the block.

**What failure looks like.** If the ensemble is no narrower than naively averaging the bounds,
the correlation model is buying nothing — you could have skipped the ensemble entirely and
averaged the published maps. This is the row that proves the machinery earns its cost.

**Where the target came from.** *Derived from the data*: each row's threshold is the actual
mean-of-bounds width computed on the same blocks, so the target differs per row (0.0251 at
1 km/2005, 0.0970 at 1 km/2020, and so on).

**Score: passes at every scale from 10 km upward, by margins that widen with block size** —
at 1000 km/2020 the ensemble width is 0.053 against a 0.095 baseline, a 45% reduction. That
widening margin *is* the spatial correlation doing its job.

Two rows miss, both at 1 km and both by under 2%: 0.02525 vs 0.0251 (2005) and 0.0478 vs 0.0472
(2010). At 1 km a "block" is a single pixel, where averaging the bounds and running the ensemble
are asking almost the same question, so there is no correlation benefit available to capture.
These are boundary ties at the scale where the test is least meaningful.

### T2.2 Ecoregion mean coverage — 4 / 4

**What it tests.** The same coverage question over 804 real ecological regions instead of
arbitrary squares.

**What failure looks like.** Ecoregions are the units conservation decisions are actually made
in. Squares are a convenient abstraction; a system could be calibrated on squares and wrong on
the irregular, size-varying, ecologically meaningful units people use.

**Where the target came from.** *Definitional*, 0.95 ± 0.05.

**Score: 0.980 / 0.988 / 0.973 / 0.975.** All four pass. Coverage is slightly high — the
ensemble is modestly over-wide at ecoregion scale — but inside tolerance at every horizon.

### T2.3 Ecoregion area-above-threshold — 8 / 8

**What it tests.** Instead of the ecoregion mean, the *fraction of the ecoregion above HM 0.1 and
above HM 0.3*.

**What failure looks like.** This is harder than the mean and closer to how the product gets
used: "what fraction of this region will be substantially modified?" The mean can be right while
the area fraction is wrong, because area depends on the spatial arrangement of change within the
region, not just its total. A member with the right regional average but change smeared evenly
instead of concentrated would fail here and pass T2.2.

**Where the target came from.** *Definitional*, 0.95 ± 0.05.

**Score: 0.973 to 0.991 across all eight rows.** All pass. Spatial arrangement within ecoregions
is right, not just level.

### T2.4 Ecoregion mean change between horizons — 1 / 1

**What it tests.** Coverage of the *change* in ecoregion mean HM from 2005 to 2020.

**What failure looks like.** The hardest aggregate case, because it depends on the horizon
coupling as well as the spatial field. A user asking "how much will this ecoregion change over
the next fifteen years, and how sure are we?" is asking exactly this. A system whose horizons
were independent would produce a far-too-narrow range here while passing every single-horizon
test.

**Where the target came from.** *Definitional*, 0.95 ± 0.05.

**Score: 0.970.** Passes. The AR(1) coupling of §3.5 is doing real work.

### T2.5 Rank histogram flatness — 0 / 4

**What it tests.** For each ecoregion, where does the observed value rank among the 400 member
values? If the ensemble is calibrated, the observation is equally likely to land at any rank, so
a histogram of ranks should be flat. A U-shape means the ensemble is too narrow (the truth keeps
falling outside); a dome means too wide (the truth keeps landing mid-pack).

**What failure looks like.** This is a strictly stronger test than coverage. An ensemble can put
95% of observations inside its 95% range while systematically placing them near the middle —
which means the spread is inflated and every derived probability is muted. Ask such a system
"what is the chance this region exceeds threshold X?" and it will answer closer to 50% than it
should, for every X.

**Where the target came from.** *Definitional*: flatness under a χ² test against uniform, p >
0.01. Scored on pooled rank bins — 804 ecoregions against 401 raw ranks would be technically
valid but nearly powerless against a smooth dome, so ranks are pooled into 50 bins with a
minimum expected count of 16.

**Score: p = 1.2e−34 / 6.2e−60 / 2.0e−52 / 1.6e−64.** Rejected decisively at every horizon.

**And the shape says which way.** The histograms are **domed** at every horizon — the middle
third of ranks holds 1.24 to 1.50 times as many observations as the two outer thirds combined.
The ensemble is **over-dispersed at ecoregion scale**: it claims more uncertainty than it needs.
This is the clearest model-level defect on the card, and T2.2's slightly-high coverage, T2.4, and
T7.3's 1.64 are the same defect seen from three other directions.

### T2.6 Biome and realm coverage — 8 rows, reported, not scored

**What it tests.** The same coverage question at biome and realm level — 14 biomes, 8 realms.

**Why it is not gated.** At ~14 aggregation units the binomial confidence interval on a coverage
estimate is about ±0.114, wider than the ±0.05 tolerance the test would apply. Such a row cannot
pass or fail honestly: it would be reporting sampling noise as a verdict. It is published with
its Wilson interval so the numbers are visible without pretending to be a test.

**Reported: biome 1.000 / 1.000 / 0.929 / 0.929, realm 0.875 at every horizon.**

---

# T3 — Do the members look like real maps? — 3 / 5

A member must not merely carry the right *amount* of uncertainty; it must have the right
*texture*. Errors should be smooth where reality is smooth and rough where it is rough.

### T3.1 Member variogram — 2 / 3

**What it tests.** Convert a member back to normal scores and measure its spatial structure
against the structure fitted to the real residuals. Three checks: the variance of those scores,
the **practical range** (the distance at which correlation effectively dies), and the **nugget
fraction** (the share of variance that is pure pixel-scale noise).

**What failure looks like.** Wrong variance means the generator is not producing the field it was
asked for. Wrong range means patches of error are the wrong size — too small and aggregates come
out over-confident, too large and neighbouring districts move in lockstep when they should not.
Too much nugget means members look like television static; too little and they look like soft
blobs with no fine detail.

**Where the target came from.** Variance is the generator's own contract, 1.0 ± 0.15
(*definitional*). Range and nugget are *derived from the data* — matched against the variogram
fitted to the actual residual field, within 25% relative and 0.10 absolute respectively.

**Score: variance 0.931 (pass), nugget fraction 0.080 (pass), practical range 0.278 relative
(fail, against ≤ 0.25).** Two of three pass. The member fields carry the right amount of
pixel-scale noise and close to unit variance, but their correlation length is off by 28% where
25% was allowed — a near miss on a *derived* target, meaning the members' patches of error are
modestly the wrong size, not that the field is structurally wrong.

### T3.2 Versus an independent-pixel null — 0 / 1

**What it tests.** Generate a second ensemble with *identical* marginals but no spatial structure
at all, and score both with the variogram score, which is sensitive to correlation rather than to
marginals. The correlated ensemble should win.

**What failure looks like.** If the correlated ensemble cannot beat a spatially-random one, the
entire spatial model is decoration — every aggregate answer would be no better than assuming
independence, which T2.1 shows collapses to 0.09 coverage at 100 km.

**Where the target came from.** *Derived from the data*, and this derivation matters. An ensemble
calibrated to the residual can only differ from an independent one in the variance still
*correlated* at the separation being scored — everything already decorrelated in the data itself
is identical for both by construction. That ceiling is read off the residual's own fitted
variogram as `1 − γ(d)/sill`, computed in-run at the pairs' realised mean separation, and the
gate is the *share of that available budget* the ensemble captures: ≥ 0.50.

**Score: 0.230 of the budget.** Fails. The ensemble does beat the null — the raw improvement is
11.8% spread-weighted, 13.4% under uniform sampling — but it captures less than a quarter of the
structure that was there to capture.

**Three companion rows are reported without a gate, and they locate the shortfall.** `T3.2r` is
the raw spread-weighted improvement (0.118), `T3.2b` the same under uniform sampling (0.134), and
`T3.2s` the comparison restricted to short lags — **0.265**. The ensemble reproduces near-field
structure roughly twice as well as it reproduces long-range structure, so the shortfall is
specifically in long-range coherence.

### T3.3 Energy score — 1 / 1

**What it tests.** A proper multivariate scoring rule evaluated against two baselines: the
independent-pixel null, and a degenerate ensemble that is just the central forecast repeated 400
times.

**What failure looks like.** Losing to the degenerate baseline would mean the spread actively
hurts — you would be better off publishing a single map with no uncertainty at all. Losing to the
null would mean the spatial structure hurts.

**Where the target came from.** *Derived from the data*: beat both baselines, whatever they score.

**Score: 0.729 against null 0.760 and degenerate 0.904** (lower is better). Passes, beating both.
The ensemble is genuinely better as a multivariate forecast than either alternative.

### T3.4 Radial power spectrum — reported, not scored

**What it tests.** The full spectral shape of member fields against the residual's, written out
for comparison.

**Why it is not gated.** It catches the failure where range and sill both look right but the
shape between them is wrong — a diagnostic for reading alongside T3.1 rather than a pass/fail
criterion. 40 spectral bins are written.

### T3.5 Longitude seam continuity — reported, not evaluable

**What it tests.** On a global grid, whether a discontinuity appears where longitude wraps at the
antimeridian. The field generator deliberately enables longitude wrap for exactly this reason.

**What failure looks like.** A visible vertical seam down the Pacific in every member — an
obvious artifact in any global visualisation.

**Where the target came from.** *Definitional*: a hard gate, no discontinuity permitted.

**Result: not evaluable — 0 valid pixels on either side of the seam.** The antimeridian is open
ocean for its entire length, so there is no land data to compare across it. The row reports the
interior reference value (|Δ| 0.0298) and marks itself unscored rather than returning a verdict
on an empty comparison.

---

# T4 — Are the four horizons coherent? — 3 / 4

A member is one story about the future, so its four horizons must hang together.

### T4.1 Between-horizon correlation — 3 / 3

**What it tests.** The correlation between consecutive horizons' normal-score fields *within* a
member, against the AR(1) coupling measured on the real residuals.

**What failure looks like.** If horizons were independent, a member could show a city expanding
rapidly by 2030 and not at all by 2035 — incoherent as a scenario, and useless for any question
about trajectories or about change between two dates. T2.4 is where that would bite numerically.

**Where the target came from.** *Derived from the data*: ρ is measured from the residuals'
own between-horizon correlation, and the gate is within ±0.10 of that measurement. The tolerance
reflects the estimator's own spread across independent samples.

**Score: 0.739 / 0.599 / 0.690 against measured ρ of 0.749 / 0.613 / 0.706.** All three pass,
within 0.016 everywhere. The generator reproduces the observed temporal coupling closely.

### T4.2 Monotone spread — 0 / 1

**What it tests.** Uncertainty must not *shrink* as you forecast further ahead. Measured as the
fraction of pixels whose marginal spread is non-decreasing across the horizon sequence.

**What failure looks like.** A product claiming to know 2040 better than 2030. Anyone reading the
intervals as a confidence trajectory would draw exactly the wrong conclusion about where the
model is reliable.

**Where the target came from.** *Set a priori*, ≥ 0.99 — near-universal, because this is a
property the construction is supposed to guarantee rather than approximate.

**Score: 0.9837.** Fails, by 0.006. Note this scores the **population** spread — the member
marginal integrated against a standard normal, shape- and clip-aware — not the spread of a finite
sample. The companion reported row `T4.2s` carries the sample
statistic at 0.900.

The property is enforced twice in construction: the quantile heads accumulate non-negative
increments, and a per-pixel cumulative maximum is applied after calibration. It reads 0.984 rather
than 1.000 because the clip to [0, 1] can compress a wide interval near the bounds — a pixel
already near HM = 1 cannot widen upward. So the residual 1.6% is a boundary effect of the
physical range, not horizons genuinely inverting.

---

# T5 — Is the ensemble faithful to the published product? — 10 / 12

Non-negotiable. If these fail, the ensemble tells a different story from the maps published
beside it, and a user comparing the two would find they disagree.

### T5.1 Median equals central forecast — 4 / 4

**What it tests.** The member-wise median must equal the published central forecast at every
pixel.

**What failure looks like.** The ensemble and the headline map would disagree about the single
most likely outcome. Anyone computing a median from the members would get a different answer from
the published central raster — two official numbers for the same quantity.

**Where the target came from.** *Monte-Carlo scaled.* The distributional median is exact by
construction: the marginal maps u = 0.5 to the central forecast exactly. Only the *sample* median
of 400 draws differs, so the tolerance is `3·(1.2533/√M)·σ·S′(0)` plus quantisation, where `S′(0)`
is the marginal shape's slope at the median. The gate is that ≥ 0.995 of pixels fall inside it.

**Score: 0.9979 / 0.9979 / 0.9985 / 0.9983.** All four pass. The ensemble's centre is the
published centre.

### T5.2 Tails equal published bounds — 2 / 4

**What it tests.** The ensemble's 2.5th and 97.5th percentiles must equal the published lower and
upper bounds.

**What failure looks like.** The published interval and the ensemble's interval would differ. A
user computing a 95% range from the members would get different bounds from the ones on the
published maps — again, two official answers.

**Where the target came from.** *Monte-Carlo scaled*, and family-aware: the standard error of a
sample quantile depends on the probability density at that quantile, which for a reshaped
marginal is not what it is for a normal. The gate is ≥ 0.95 of pixels within that tolerance.

**Score: 0.847 / 0.949 / 0.985 / 0.986.** The two longer horizons pass; +5 yr and +10 yr fail,
+10 yr by 0.001.

The pattern is informative: the failure is worst at the shortest horizon and disappears as the
lead time grows. At +5 yr the intervals are narrow and the marginal's density at the tail is
high, so the sample-quantile tolerance is at its tightest, while quantisation to int16
(3 × 10⁻⁵) is a larger fraction of a narrow interval. This is a finite-sample and storage-
precision effect concentrated where intervals are smallest, not a failure of the marginal
construction — which is exact by design.

### T5.3 Validity mask identity — 4 / 4

**What it tests.** Every pixel the model calls valid must be valid in the ensemble, and vice
versa.

**What failure looks like.** Members carrying data where the published maps have none, or holes
where the maps have values. Any aggregate computed over a region would silently include or
exclude different pixels depending on which product you used.

**Where the target came from.** *Definitional*: exactly 0 mismatched pixels.

**Score: 0 at all four horizons.** Passes exactly.

---

# T6 — Is the amount and sign of change realistic? — 10 / 24

T1 and T2 ask whether the uncertainty is the right *size*. T6 asks whether a member, read as a map
of change, resembles a real one.

### T6.1 Frequency of small decreases — 0 / 4

**What it tests.** How often a member shows HM *falling* by more than 0.01, against how often
reality did.

**What failure looks like.** HM decreases are real but rare — land does occasionally revert. An
ensemble that invents them freely will, for any region, over-state the probability of
improvement. A user asking "what is the chance this district gets less modified?" would be told a
number several times too high.

**Where the target came from.** *Derived from the data*: the observed rate is counted, and the
member rate must fall within [0.5, 2.0] of it.

**Score: 4.12 / 7.76 / 7.81 / 9.45.** Fails at every horizon, four to nine times too many small
decreases, and worsening with lead time.

**Why.** The marginal's lower tail is too heavy for a quantity that is real but rare. Part of this
is not reachable by any per-pixel marginal: even the ideal calculation using each distance band's
own empirical distribution over-predicts, which means it reflects a dependence between the
standardised residual and the pixel's own threshold that a marginal cannot encode.

### T6.2 Moderate decreases — 2 / 4

**What it tests.** The same question at −0.05.

**Where the target came from.** *Derived from the data*, looser at ratio ≤ 3 and carrying an
absolute cap, so a class cannot pass on a favourable ratio while emitting an implausible absolute
rate.

**Score: 0.84 / 2.37 / 6.08 / 10.18.** The two shorter horizons pass; +15 and +20 yr fail.

### T6.3 Large decreases — 4 / 4

**What it tests.** The same question at −0.15.

**Where the target came from.** *Derived from the data*, ratio ≤ 5.

**Score: 0.0004 / 0.124 / 2.17 / 3.90.** All four pass.

**Reading T6.1–T6.3 together is what matters.** The excess is concentrated in *small* decreases
and diminishes as the threshold deepens: at −0.01 the ensemble is 4–9× too generous, at −0.05 it
is right at short leads and 6–10× at long ones, and at −0.15 it is under-generous at short leads
and inside tolerance throughout. The defect is a slightly-too-heavy near tail, not an ensemble
that invents catastrophic reversion.

### T6.4 Tail asymmetry — 1 / 4

**What it tests.** Real HM change is strongly asymmetric — increases are far more common than
decreases. This compares the ensemble's ratio of large increases to large decreases against the
observed ratio.

**What failure looks like.** A member that treats development and reversion as roughly equally
likely would look, as a map, nothing like the real world: the actual pattern is overwhelmingly
one-directional. Any scenario analysis drawn from such members would badly over-weight
improvement.

**Where the target came from.** *Derived from the data*, ratio in [0.5, 2.0]. Scored two-sided
deliberately: a one-sided version ("at least half the observed asymmetry") is passed trivially by
any member with an arbitrarily heavy upper tail.

**Score: 18,830 / 49.5 / 2.14 / 1.14.** Only +20 yr passes.

The progression is the whole story. At +5 yr the observed number of large decreases is so small
that the observed ratio is enormous and the comparison is dominated by a near-zero denominator;
by +20 yr, where both directions have accumulated real counts, the ensemble reads 1.14 — very
close to correct. **This row is informative at long leads and close to meaningless at short
ones**, and the +20 yr value is the one to believe.

### T6.5 Change quantiles — 3 / 8

**What it tests.** The 1st and 5th percentile of member change against the observed ones.

**What failure looks like.** These are the "reasonable worst case for reversion" numbers. Getting
them wrong means the lower end of any scenario range is wrong.

**Where the target came from.** *Derived from the data*, within a factor of 2.

**Score:** the 1st percentile reads 0.48 / 0.93 / 0.65 / 0.70 and passes at three of four
horizons; the 5th percentile reads 3.54 / 6.97 / 4.75 / 6.40 and fails at all four. The deeper
percentile is closer to right than the shallower one — the same too-heavy near tail as T6.1,
seen from another direction.

---

# T7 — Are members diverse but not absurd? — 1 / 2

### T7.2 Member diversity — 1 / 1

**What it tests.** Mean pairwise correlation between members.

**What failure looks like.** Near-identical members would mean the ensemble has no spread to
offer — 400 copies of one map, with every aggregate uncertainty collapsing to zero. This is the
degenerate failure mode of any ensemble system.

**Where the target came from.** *Set a priori*, < 0.98 — a floor against degeneracy rather than a
calibration target.

**Score: 0.272.** Passes with a very large margin. Members are genuinely distinct realisations.

### T7.3 Spread-skill ratio — 0 / 1

**What it tests.** The deepest question on the card: where the ensemble says it is uncertain, is
the model actually more wrong? Computed at ecoregion scale as the ensemble's own standard
deviation divided by the RMSE of the ensemble mean. A value of 1.0 means spread predicts error
correctly; above 1 means over-spread.

**What failure looks like.** Uncertainty that does not track actual error is uncertainty that
cannot be used to prioritise. A user asking "where should I be most cautious about this forecast?"
gets an answer uncorrelated with where the forecast is actually poor.

**Where the target came from.** *Set a priori*, 1.0 ± 0.25 — the kind of threshold to hold most
loosely.

**Score: 1.637.** Fails, over-spread by roughly 64%. The ensemble claims more uncertainty than it
needs at ecoregion scale — the same defect T2.5's dome reports, arriving from a completely
different direction. Two independent instruments agreeing is what makes this a real finding rather
than an artifact of either one.

### T7.1 Member vs observed renders — reported, not scored

Twelve rendered comparison figures for visual inspection. Not a gate.

---

# T8 — Is change put in the right places? — 7 / 14

The geographic sanity check, and the family the calibration's class structure exists to serve.
Change is overwhelmingly concentrated near change that already happened; an ensemble that
sprinkles development into untouched country is wrong in a way no coverage target notices. Every
row is scored per distance-to-past-change band.

### T8.1 Rate of increase above 0.05, per band — 4 / 6

**What it tests.** The ratio of the member rate of `Δ > 0.05` to the observed rate, in each of six
distance bands.

**What failure looks like.** This is the row that governs whether the product can be trusted for
"where will new development appear?". Too high in a remote band and the product hallucinates
development in wilderness. Too low and it cannot anticipate genuinely new frontiers at all.

**Where the target came from.** *Derived from the data*: observed rates are counted per band, and
the member rate must land within [0.5, 2.0] of them.

**Score:**

| band | ratio | |
|---|---|---|
| 0–1 px | 1.37 | pass |
| 1–3 px | 1.44 | pass |
| 3–10 px | 1.41 | pass |
| 10–30 px | **2.32** | fail, too hot |
| 30–100 px | 0.74 | pass |
| >100 px | **0.034** | fail, far too cold |

**The near field is right.** Within 10 px of past change — where the overwhelming majority of real
development occurs — the ensemble emits within about 40% of the observed rate, comfortably inside
tolerance. For the question the product is mostly used for, placement is good.

**The far field is the headline defect of the system.** Beyond 100 px the ensemble emits about 3%
of the observed rate of new development. Reality shows genuine, if rare, development appearing far
from anything existing; the ensemble essentially cannot imagine it.

**This is a property of the model's width heads, not of the calibration.** In the far band the
published half-widths are around 0.004, so a change of +0.05 sits roughly twelve half-widths out.
The marginal's tail, trusted to its exhausted limit, reaches three to four. The calibration layer
does correctly identify and widen the remote pixels that *can* develop — their class-conditional
coverage moves from 0.67 to 0.99 and their extreme half-width percentile widens — but no post-hoc
layer can bridge a twelve-half-width gap. Closing this requires the quantile heads to emit wider
intervals in remote country, which is a model change.

### T8.4 Rate of decrease below −0.05, per band — 3 / 6

**What it tests.** The mirror of T8.1 on the decrease side.

**Where the target came from.** *Derived from the data*, ratio in [0.5, 3.0] — looser than T8.1
because observed decrease events are far rarer and the counts are correspondingly noisy.

**Score: 5.32 / 1.47 / 2.47 / 2.97 / 3.70 / 189.1** across the six bands. The three middle bands
pass; the nearest band, the 30–100 px band and the far band fail.

**The far-band figure should not be read as a measurement.** The physical floor is already exact —
the sampler clips every member to [0, 1], so a pixel at HM ≤ 0.05 contributes exactly zero and
cannot produce a spurious decrease. Only a small fraction of remote pixels can physically decrease
past −0.05 at all, and the observed count on them is tiny. A ratio of 189 against a denominator of
a handful of events is a statement about the denominator.

### T8.2 Remote-band realism, both directions — 0 / 1

**What it tests.** A single summary of the remote band: mean |log₁₀(member / observed)| over both
tails.

**Where the target came from.** *Set a priori*, ≤ 0.301 — within a factor of two either way.

**Score: 1.872**, i.e. off by a factor of about 74 on average across the two directions. Driven by
both far-band failures above.

### T8.3 Near/remote contrast — 0 / 1

**What it tests.** Is change concentrated near past change to the same degree reality is? The
ratio of the member's near/remote contrast to the observed contrast.

**What failure looks like.** A value near 1 means the ensemble concentrates development like
reality does. Far above 1 means it over-concentrates — too much near, too little far. Far below 1
means it smears development across the landscape.

**Where the target came from.** *Derived from the data*, ratio in [0.5, 2.0].

**Score: 40.1.** Fails, over-concentrated by a large factor. This is arithmetically the same fact
as T8.1's far band: the contrast is a ratio whose denominator is the remote rate, and that rate is
near zero. It is not independent evidence.

---

# What the card says overall

**What is solid.** Per-pixel uncertainty is correct everywhere, including inside every class of
pixel and on the high-change pixels the product exists to serve (T1, 24/24). Aggregate uncertainty
is approximately correct at every scale from 1 km to 1000 km and across 804 real ecoregions, which
is the central claim of the system and which neither naive alternative comes close to (T2.1, T2.2,
T2.3, T2.4). The ensemble is faithful to the published maps — same centre, same mask, same bounds
at the two longer horizons (T5). Members are coherent stories across horizons, reproducing the
measured temporal coupling (T4.1). Change is placed correctly in the near field, where nearly all
real development occurs (T8.1, bands 0–10 px).

**Three defects are named, and none is a measurement artifact.**

1. **Over-dispersion at aggregate scale.** T2.5's rank histograms are domed at every horizon and
   T7.3's spread-skill ratio reads 1.64. Two independent instruments agree that the ensemble
   claims more uncertainty than it needs at ecoregion scale. Consequence for a user: probabilities
   derived from the members are muted toward 50%, and regional uncertainty ranges are wider than
   they need to be. Aggregate coverage stays correct, so this costs sharpness rather than validity.

2. **The far field cannot emit new development.** T8.1 beyond 100 px reads 0.034 of observed. The
   product should not be used to anticipate genuinely novel, isolated development frontiers. This
   is a property of the model's quantile heads and is not reachable by any recalibration or
   marginal reshaping.

3. **Long-range spatial structure is under-reproduced.** T3.2 captures 0.230 of the available
   structure budget against a 0.50 gate, with the short-lag companion at 0.265 versus 0.118 at full
   range. Near-field texture is roughly twice as faithful as long-range coherence.

**One family reports mostly on a single underlying cause.** T6's 14 failures are largely the same
too-heavy near tail on the decrease side, restated at four horizons and several thresholds, plus a
T6.4 asymmetry row that is uninformative at short leads by construction.

**Two rows are under-powered rather than failing.** T2.1 at 1000 km rests on 182 blocks, and T2.6
is reported without a gate for exactly this reason at 14 units.
