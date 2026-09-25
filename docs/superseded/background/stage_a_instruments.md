# Stage A — the scoring instruments

Written 2026-08-20, branch `ensemble`. Stage A of `docs/improvement_plan.md`: fix the ruler
before anything is judged. Nothing here changes the model, the calibration or the marginal.
Four instruments changed, and two of the four changed for a different reason than the plan
gave, so the measurements that settled them are recorded alongside.

Read `docs/global_scorecard.md` first — it is what this stage responds to.

**Outcome: Africa 111/133 → 102/133 on the same ensemble**, verified bit-for-bit identical to
the published one. All nine lost rows are instruments. The card is
`data/ensemble/exp/africa_k5/validation_m400_stageA/scorecard_spliced.csv`, and its
`PROVENANCE.md` records the splice, the verification and the per-row causes.

| family | published | Stage A |
|---|---|---|
| T2 | 37/41 | **34/41** |
| T6 | 19/24 | **15/24** |
| T8 | 11/14 | **9/14** |
| T1 / T3 / T4 / T5 / T7 | 29/33, 3/6, 0/1, 11/12, 1/2 | unchanged |

---

## 1. T8.2, T8.3 and T6.4 are two-sided now (item 8)

**The problem.** All three were one-sided gates written when the failure mode was the
ensemble *inventing* change in country far from any past change: the original heads put
0.0297 of remote-band change against an observed 0.0000, and a ceiling was the right
instrument for that. The sign has since flipped. On the global card the ensemble emits
**2.3% of the observed increase rate and 28× the observed decrease rate beyond 100 px**, and
all three gates passed:

| gate | old form | global reading | verdict |
|---|---|---|---|
| T8.2 | remote `P(Δ>0.05)` ≤ 0.002 | 3.8e-06 | pass |
| T8.3 | near/remote contrast ≥ 20× | 90,596 (observed 1,527) | pass |
| T6.4 | tail asymmetry ≥ 0.5 × observed | 1343.9 (observed 5.3) | pass |

Each of them reads its best at the exact moment the far field goes silent, which is the
defect this round exists to remove. A one-sided gate satisfied by a value 254× the quantity
being checked is not evidence of health.

**The change.** All three are scored as a ratio to observed, which is what T8.1 and T8.4 have
always done:

- **T8.2** → mean `|log10(member / observed)|` over *both* remote-band tails, `P(Δ>0.05)`
  and `P(Δ<−0.05)`, passing at ≤ 0.301 (a factor of two either way).
- **T8.3** → `(near/remote)_member / (near/remote)_observed`, passing in [0.5, 2.0].
- **T6.4** → `asymmetry_member / asymmetry_observed`, passing in [0.5, 2.0].

**The zero-observed fallback is kept and is now explicit.** Southern Africa's observed remote
change rate is exactly 0.0000, where no ratio exists; there T8.2 falls back per-direction to
the original 0.002 ceiling and T8.3 scores the remote rate itself rather than an infinite
contrast. The note on the row records which branch was taken, so a reader can never mistake
one for the other. That branch is also why these gates never fired on 22 model experiments:
they were all screened on the one extent where the fallback is the only thing that runs.

`src.ensemble.validate.remote_band_error` and `near_far_contrast` hold the arithmetic, and
`tests/test_stage_a_gates.py` asserts both directions and both fallbacks.

**Measured on Africa at M=400** (`validation_m400_stageA`), against the same ensemble that
scored 111/133:

| row | old form | old verdict | new reading | new verdict |
|---|---|---|---|---|
| T8.2 | 1.4e-06 ≤ 0.002 | pass | mean \|log10 ratio\| **1.530** — `P(Δ>0.05)` at 0.107× observed, `P(Δ<−0.05)` at 123× | fail |
| T8.3 | 226,373 ≥ 20× | pass | contrast ratio **12.49** (member 2.26e+05, observed 1.81e+04) | fail |
| T6.4 | ≥ 0.5 × observed | pass ×4 | **91056 / 34.2 / 75.8 / 54.0** against [0.5, 2.0] | fail ×4 |

Six rows, all previously passing, all now failing on the same defect the round exists to fix.

## 2. T2.5's rank histogram is pooled before it is tested (item 10)

**The plan's premise was wrong, and so was the first replacement for it.** The plan recorded
T2.5 as a χ² over 35.65M pixels, over-powered and rejecting on deviations too small to
matter. It is not a pixel test at all: the aggregation unit is the ecoregion, **158 on Africa
and 804 globally**, against `M + 1 = 401` rank bins.

That looks like a textbook violation of the χ² conditions — 0.39 expected counts per cell on
Africa — and the first fix here was written on that basis. **Measured, it is not one.** For
*equiprobable* cells the approximation survives small expected counts, and 2000 uniform
replicates at (n=158, K=401) give:

| | statistic mean | statistic sd | rejection rate at nominal 0.01 |
|---|---|---|---|
| measured | 399.3 | 28.3 | **0.0105** |
| χ²(400) reference | 400 | 28.3 | 0.01 |

The published p-values are sound. A Monte-Carlo p-value agrees with the asymptotic one on
every row of both cards (Africa 2010: MC 0.370 against asymptotic 0.348).

**What is actually wrong is power.** The deviation these histograms carry is broad and smooth
— the observation lands in the middle of the ensemble far too often, centre bins at ~2×
uniform and the extremes at ~0.1× — and spreading that signal over 401 cells holding 0.39
counts each buries it. Pooling adjacent ranks into contiguous groups is lossless under the
null (uniform ranks stay uniform under any contiguous pooling) and much sharper against this
alternative:

| n | test | size at nominal 0.01 | power vs the observed dome |
|---|---|---|---|
| 158 | unpooled, 401 bins | 0.011 | **0.59** |
| 158 | pooled, 9 bins | 0.013 | **0.90** |
| 804 | pooled, 50 bins | 0.011 | 1.00 |

**The change.** `rank_histogram_test` pools to an expected occupancy of at least
`--rank_min_expected` (default 16) before testing, and reports `n`, `n_bins` and the achieved
minimum expected count on the row. The rank-histogram figure is drawn on the pooled bins too:
at 401 bins the picture is sampling noise and invites the opposite reading to the statistic
printed beside it.

**Measured on Africa at M=400**: p = 6.3e-31 / 3.4e-13 / 4.8e-14 / 9.8e-12 on 9 pooled bins
(n=158, min expected 17.3), against 1.2e-06 / 0.348 / 0.018 / 0.039 unpooled.

**Consequence: Africa's T2.5 goes from 3/4 passing to 0/4.** The three passes were
non-rejections at n=158, not evidence of flatness. The pooled Africa histograms are domed on
all four years, exactly as global's are, and T7.3's spread-skill ratio (1.89 Africa, 1.445
global) says the same thing from an independent direction. Ecoregion-scale over-dispersion is
real, it is not confined to the global extent, and it was previously being scored by an
instrument too blunt to see it on a regional card.

The reliability index is reported on the pooled bins as well. At `M+1` bins it is dominated
by how many aggregation units an extent has rather than by calibration — Africa read 1.38 and
global 0.77 off the same defect — so it is comparable only where `n_bins` matches, and
`n_bins` is now on the row.

## 3. T3.2 is scored against the structure the data actually has (item 9)

**The problem.** T3.2 asks the copula ensemble to score ≥ 30% below an independent-pixel
ensemble with identical marginals. That threshold was set a priori, and what the ensemble can
possibly win by is set by the data: calibrated to the residual, it reproduces the residual's
correlation structure, so it can differ from an independent-pixel ensemble only in the
variance still *correlated* at the separation being scored. `scripts/t32_structure_budget.py`
measured that budget as 14.0% at 25 px and the argument was recorded — in
`docs/current_progress.md` §4 item 9 and `docs/central_field_baseline.md` §10 — as "T3.2 and
T3.4 conflict, and T3.4 is the one grounded in data".

**That argument was measured on southern Africa and does not transfer.** Africa's own fitted
residual variogram (`data/ensemble/exp/africa_k5/diagnostics/variogram_fits.csv`, stratum
ALL, h=20: practical range 147.8 px against southern Africa's 50.5) gives a budget four times
larger at the same lag:

| separation | southern Africa budget | Africa budget |
|---|---|---|
| 25 px | 0.14 | **0.571** |
| 50 px | — | 0.440 |
| 78 px | — | 0.269 |
| 100 px | — | 0.157 |

This is the same blind spot as the far-field one, in a different family: a threshold set from
one extent's correlation length, applied to extents with a different one.

**It also explains a number that looked like a regression.** T3.2 read 16.5% when the
argument was written and 0.031 on the Africa M=400 card. Nothing about the field moved: the
scored separation did. T3.2 caps pair separation at half the *member* practical range, which
was 50.5 px then (cap 25 px) and ~148 px on Africa (cap ~74 px), where the budget is a
quarter of what it was.

**The change.** The budget is computed in-run from the fitted residual variogram —
`1 − gamma(d̄)/sill`, evaluated at the pairs' **realized mean separation**, not at the cap,
since the budget is steep enough between the two to overstate the difficulty by about a
factor of two — and T3.2 gates on the *share of the budget captured*, `improve / budget`,
at ≥ 0.50. Three rows are reported alongside so the published series stays readable:

- **T3.2r** — the raw improvement the ≥ 0.30 gate used to score.
- **T3.2b** — the uniform-sampled reference, unchanged.
- **T3.2s** — the same comparison at a short lag (pairs capped at ~8 px), with its own
  budget and share, reported and not gated.

T3.2s exists because "score it at short lags" is the other half of item 9 and should be
decided on measurements rather than on the argument that the long-lag reading is hard. It
costs one extra scoring call in a three-minute stage.

**Measured on Africa at M=400, and it reverses the recorded conclusion:**

| | improvement | budget | share |
|---|---|---|---|
| T3.2, spread-weighted, mean separation 40.1 px (gated) | 0.0312 | **0.4969** | **0.063** |
| T3.2s, short lag, mean separation 5.8 px (reported) | 0.290 | 0.8135 | 0.357 |

**Africa's budget at the scored separation is 0.497, and the old threshold was 0.30.** The
target was therefore *reachable in principle on this extent all along*, and the ensemble
captures 6% of what the data offers. "T3.2 asks the ensemble to be better structured than the
data it is calibrated to" is true of southern Africa and false here. The row stays red under
either instrument, but it is no longer a threshold artifact to be waived — it is a real
shortfall in long-range structure, and it should be read alongside T3.1 (nugget/sill 0.103
against ≤ 0.10, normal-score variance 1.151 against 1.0 ± 0.15) rather than dismissed.

The short-lag reading is the more interesting half: at 5.8 px the ensemble captures 0.357 of
a much larger budget. Structure is present at short range and missing at long range, which is
the same direction T2.5's dome and T7.3's spread-skill ratio point in.

**What has not changed: do not tune the field to pass T3.2.** The residual carries 12% of its
spectral power at 1–3 px and 37–41% at 3–10 px; generating a field smooth enough to win the
old gate is exactly the failure T3.4 exists to catch. Normalizing by the budget changes what
the number *means*, not what the field should be.

## 4. A global smoke tier (item 2)

`scripts/run_smoke_tier.sh <experiment>` generates a four-member ensemble and its null from
an experiment's existing calibration — nothing is re-fitted; a smoke tier that re-derives the
chain is testing a different chain from the one it protects — and runs **every stage at all
four block scales**, then `scripts/check_smoke.py` gates on the card.

Pass rates are meaningless at M=4 and are not checked. What is checked is what a
shape-and-scale bug breaks:

- every stage emitted the rows it owns, rather than dying and leaving the others to run;
- no `stage '<name>' completed` failure marker;
- no scored row carries a non-finite value — including one embedded in a *string*, which is
  how the global card's T3.5 failed a hard gate on
  `mean |Δ| across seam nan vs interior 0.02983`;
- every requested block scale produced its T2.1 rows, since the 1000 km path exists on no
  regional card and would otherwise first run inside a 17-hour stage.

The checker was verified in both directions before use: it clears the Africa M=400 card
(147 rows, 133 scored — the 111/133 denominator) and rejects the global card on the T3.5 NaN.

The three defects that cost the last round — a ρ estimator drawing seven blocks over a 73%
invalid grid, an FFT that outgrew a 24 GB card, a hard gate scored on a NaN — are all of this
shape. None needed 400 members to appear; each was instead found by a run that had already
spent hours.

**Proven on Africa**: 170 rows, 154 scored, **30.7 min** wall clock (plus ~2 min to generate
both stores), peak RSS 13.51 GB against a 20 GB budget — and the checker cleared it. Every
stage ran, including the 1000 km block path. The three new gates already fail in the right
direction at M=4 (T8.2 = 1.549, T8.3 = 9.70, T6.4 = 44.8/102.8/382.7), which is a useful
property: the far-field defect is large enough to see with four members.

Pass counts at M=4 are meaningless by construction and are not checked — the 2.5/97.5
percentiles of four draws are the min and max, so T1 coverage reads ~0.55 everywhere and T5.1
misses its MC-scaled gate. Read the checker's verdict, never the score.

Run it on Africa (`REGION_ROOT=data/ensemble/region/africa`) before any long regional run,
and on the global extent before any global one. Budget rather more than 30 min there: the
global grid is 5.2x Africa's valid pixels and 10.8x its raster, and T3's `build_z_field` and
T2's block streaming are the M-independent stages that will not shrink with the member count.

---

## 5. Two general lessons from this stage

**A gate written for one failure mode keeps passing after the mode inverts.** T8.2, T8.3 and
T6.4 were all correct instruments when they were written. Nothing about them decayed; the
system moved to the other side of them. Any one-sided gate should be re-read whenever the
thing it bounds changes sign.

**Sparse cells are not automatically an invalid χ².** The first version of the T2.5 fix was
written on the textbook "expected ≥ 5" rule and would have been recorded as a validity
correction. It is not one — measured size is nominal — and the change survives only because
the *power* argument is separately true. The rule that caught it is the project's own: verify
the measurement before believing the finding, including when the finding is your own
diagnosis of someone else's instrument.
