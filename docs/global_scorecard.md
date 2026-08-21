# The global scorecard

Status as of 2026-08-20, branch `ensemble`. The published T1–T8 evaluation of the global
hindcast: what was run, what it scored, and what the numbers mean. Companion to
`docs/validator_scaling.md` (how the validator was made to scale), `docs/ensemble_model_outline.md`
§3 (what each target asks), and `docs/central_field_baseline.md` (the central field).

**Headline: 112 of 146 scored rows pass, with 11 reported-only, at M = 400 on the full
17111 × 40000 grid.** Africa scored 111/133 and southern Africa 101/127. The three totals are
*not* comparable — see §7 — and the interesting content is not the total anyway. Most of the
34 failures are one defect wearing several hats.

---

## 1. What was scored

| | |
|---|---|
| grid | 17111 × 40000 = 684,440,000 px; **184,573,321 valid** (27.0%) |
| model | the five `africa_k5` fold checkpoints, re-predicted globally, no retraining |
| stitch | `holdout` — every pixel from the fold that held it out |
| members | 400 + a 400-member independent-pixel null |
| chain | recalibration, width factors, spectrum, AR(1) ρ and marginal shape all re-derived from *this* model's own residuals |
| wall clock | 29.7 h for the scorecard; 51.6 min per ensemble |

Artifacts: `data/ensemble/exp/global_k5/validation/` (`scorecard.csv`, 157 rows).

The chain that produced it, in the order it ran:

- **Central field.** Skill against persistence **0.110 / 0.189 / 0.239 / 0.231** at
  h = 5/10/15/20, against Africa's 0.130 / 0.201 / 0.234 / 0.231 and southern Africa's
  0.085 / 0.148 / 0.184 / 0.191. The model matches Africa at the long horizons the 2040 product
  depends on. Slope of observed on predicted 1.169 / 1.111 / 0.994 / 0.944.
- **Raw interval coverage 0.949 / 0.944 / 0.939 / 0.948** — the closest to nominal of any extent
  measured (Africa 0.957–0.971, southern Africa 0.978–0.985).
- **Recalibration: `stratified`**, held-out interval score 0.156840 against identity 0.158474 —
  a 1.03% margin, against Africa's 0.08%. The first run where recalibration does non-trivial
  work. `mean_s = 1.286`: it **widens**. Every regional run needed narrowing.
- **Residual practical range 135 px at h = 20**, against a 128 px fold tile. Independently
  reproduces the Africa reading (99–166 px, typically ~130) and reconfirms that held-out skill
  here is optimistic: every held-out tile is ringed by trained-on tiles well inside the range the
  residual is still correlated over.
- **Spectrum**: nugget 0.033–0.064, with 35–58% of residual variance beyond 50 px and 5–9%
  at 1–3 px. Long-scale dominated, consistent with the 135 px range.
- **AR(1) ρ = {10: 0.523, 15: 0.497, 20: 0.771}**, measured. See §5.

---

## 2. The result, by family

| family | global | Africa | s. Africa |
|---|---|---|---|
| T1 pixel-scale coverage | 28/35 | 29/33 | 29/31 |
| T2 aggregate coverage | 43/49 | 37/41 | 31/41 |
| T3 spatial realism | 3/6 | 3/6 | 3/6 |
| T4 temporal coherence | **3/4** | 0/1 | 0/1 |
| T5 hard gates | 11/12 | 11/12 | 12/12 |
| **T6 change realism** | **13/24** | 19/24 | 14/22 |
| T7 diversity / spread-skill | 1/2 | 1/2 | 1/2 |
| T8 change placement | 10/14 | 11/14 | 11/12 |

T2 at 43/49 is the motivating problem of the project — aggregate coverage at every block scale —
and it now holds at 1, 10, 100 **and** 1000 km, a scale Africa never carried. T4 goes from
unscorable to 3/4. The loss against Africa is concentrated almost entirely in T6.

---

## 3. One defect explains most of the failures

### 3.1 The far field is wrong in both directions

T8 scores `P(Δ > 0.05)` and `P(Δ < −0.05)` per distance-to-past-change band as a ratio of
ensemble to observed. Ratio 1.0 is perfect:

| band | P(Δ > 0.05) | P(Δ < −0.05) |
|---|---|---|
| 0–1 px | 1.34 | 1.36 |
| 1–3 px | 1.36 | 2.74 |
| 3–10 px | 1.36 | **4.19** |
| 10–30 px | 1.55 | 2.82 |
| 30–100 px | **2.17** | 2.46 |
| **> 100 px** | **0.023** | **27.9** |

**The near field is well calibrated in both directions. The far field is wrong in both: 44×
too few increases and 28× too many decreases.**

All eleven T6 failures are the same defect seen pooled rather than by band —
`P(Δ < −0.01)` 6.4–9.3× observed, `P(Δ < −0.05)` 2.5–10.2×, the 5th percentile of change
3.6–7.1× too deep. Note which lower-tail rows *pass*: `P(Δ < −0.15)` comes in at
0.04 / 0.85 / 0.38 / 2.04, i.e. fine. **The excess is entirely in small decreases**, not extreme
ones.

### 3.2 The mechanism: ~~a missing physical floor~~ — **corrected 2026-08-20, this was wrong**

> The paragraph below is retained because it drove a plan item. It is **false**, and the
> correction is the first entry of `docs/improvement_plan.md` Stage B, 4-diag.
>
> ~~About 40% of the globe sits in the `[0, 0.01)` HM stratum, and the remote band is
> overwhelmingly that stratum. On land whose HM is already zero a decrease is close to
> physically impossible, yet the marginal still assigns real probability to Δ < −0.05 there.
> That is exactly the signature above: a large excess of *small* decreases, no excess of large
> ones, concentrated where HM ≈ 0.~~

**The floor is already there and is exact.** `marginal_from_z_torch` clamps every member to
HM ∈ [0, 1], so Δ ≥ −HM_t0 holds by construction; a pixel at HM ≤ 0.05 contributes exactly
zero to `P(Δ < −0.05)` and cannot produce any excess. Measured on the far band at h=20,
**0.14% of remote pixels (48,438 of 33.6M) can decrease past −0.05 at all, and all 88 observed
decrease events lie on them, none below the floor.**

So the 28× excess is 88 observed events against ~2,460 expected, on 0.14% of the band — a real
signal, but far too thin to fit a correction against (rule 13), and Africa carries **one** such
event, so it cannot screen this at all.

**The far-field *increase* deficit is the target that is actually measurable**: 83,797 observed
events globally against the decrease side's 88, and ~346 on Africa. Read §3.1's two columns
with that asymmetry in mind — the `P(Δ > 0.05)` column is a measurement, the `P(Δ < −0.05)`
column in the remote row is a handful of events.

The validator's own knob table reaches the same conclusion without being told:

> Truncate the marginal's left tail at a physical floor (Phase 3 marginal family) — do **not**
> narrow the marginals globally, that breaks T1.

The upper side has the mirror problem from the opposite cause. The accepted `measured` recipe is
`--u_bound 0.999,0.999,0.999,0.999,0.999,0.975` — the tail bound is raised on the upper side out
to 100 px and **held down at 0.975 in the remote band**. That was tuned on southern Africa, where
the observed far-field rate genuinely *was* 0.0000, so suppressing far-field increase was correct
there. Globally it is not: 8,079 of 3,857,693 pixels beyond 100 px gained more than 0.05 HM
between 2000 and 2020.

### 3.3 It was predicted before generation, and confirmed three ways

This is not a scorecard artifact. Three independent paths agree:

1. `predict_change_rates.py`, closed form, run against the fitted marginal *before* the 434 GB
   ensemble existed: mean |log₁₀ ratio| **0.699** over the 24 (horizon × band) cells, with the
   far-band `P(>hi)` ratio at 0.00 and `P(<lo)` up to 63×.
2. Raw rasters — `HM_2000/HM_2020_AA_1000.tiff` plus band 2 of `change_context_w2000_1000.tif`,
   touching no pipeline artifact — reproducing the observed rates to 0.86–0.99.
3. The T8 rows above, from 400 sampled int16 members.

A closed-form CDF, a direct raster count, and a Monte-Carlo ensemble reaching the same answer is
about as much corroboration as this project can generate.

### 3.4 Two gates pass *because* of the defect

**T8.2** (change invented in the remote band ≤ 0.002) reads 3.8 × 10⁻⁶ and **T8.3** (near/remote
ratio ≥ 20×) reads 90,596. Both pass emphatically, and both reward an ensemble that is too quiet
remotely. They were written when remote-band *invention* was the failure mode — the original
heads put 0.0297 there against an observed 0.0000. Globally the sign has flipped and these two
gates no longer discriminate; they would look their best at the exact moment the far field went
completely silent.

**T6.4** (tail asymmetry) does the same thing more starkly: it passes at 1343.9 against a target
of "≥ 2.7 (observed 5.3)". A one-sided gate satisfied by a value 254× the observed quantity is
not evidence of health.

This is worth stating plainly because a reader scanning for red rows will conclude the remote
band is the best-behaved part of the map. It is the worst.

---

## 4. The other failures, by cause

**T1.5 ×4, T1.1 ×2, T1.2 ×1 — the cost of widening.** Global raw coverage was already near
nominal, so the stratified rescale widened rather than narrowed (`mean_s` 1.286, upper bounds
×1.13–1.26). T1.5 compares interval width against the *original heads*, a fixed historical
baseline, and four h = 5 rows land at 1.287–1.317 against ≤ 1.25. Every h = 10/15/20 row passes
(1.15–1.19). T1.1 is 0.9649 at h = 5 and h = 10 against 0.95 ± 0.01, and passes at h = 15 and
h = 20. These are all one coherent consequence, not seven independent problems.

**T2.5 rank histograms, 0/4**, χ² p from 10⁻²⁶ to 10⁻⁴². Either a real deviation or the test
being hopelessly over-powered at 184.6M pixels — a χ² over that many samples rejects on
deviations far too small to matter. **Not disentangled**, and the p-values are extreme enough
that I would not simply wave it away on power. `t2_rank_histograms.csv` and
`rank_histograms.png` hold the shapes.

**T4.2 = 0.5526 against ≥ 0.99** (Africa 0.7118, southern Africa 0.9973). **Suspected to be a
consequence of measuring ρ rather than a regression** — see §5. Untested.

**T3.1 practical range 0.342 against ≤ 0.25 relative.** The other two T3.1 rows pass:
member normal-score variance 0.919 (target 1.0 ± 0.15) and nugget/sill 0.074 (≤ 0.10).

**T3.2 = 0.045 against ≥ 0.30.** The known, regionally invariant correlation-structure defect —
Africa 0.031, southern Africa 0.021. Global is marginally the best of the three and still fails.
`docs/current_progress.md` §4 item 9 argues at length that T3.2's threshold conflicts with T3.4
and that T3.4 is the one grounded in data; nothing here changes that reading. The uniform-sampled
reference T3.2b is 0.112.

**T7.3 spread-skill 1.445 against 1.0 ± 0.25** — better than Africa's 1.89, still failing.
T7.2 member correlation 0.260 passes comfortably.

**T2.1 block coverage 1000 km (2020) = 0.885**, the single failing block-coverage row out of 16.
The 2020 column degrades generally (10 km 0.917, 100 km 0.913), which is the h = 20 horizon.

**T2.8 = 0.05032 against ≤ 0.0500** — over by 0.03%. One row of eight.

**T5.2 (2005) = 0.920 against ≥ 0.95 MC-scaled**, a hard gate. The other three horizons pass at
0.989 / 0.975 / 0.978, so it is the h = 5 horizon specifically, and **Africa failed the same
single row** — T5 is 11/12 in both. Its note is informative:
`shape slope at the bounds 0.50-0.50 / 2.48-4.08 over bands`, i.e. the marginal's slope at the
tail bound differs by a factor of eight across distance bands. That is the remote-band `u_bound`
of §3.2 showing up in a second place.

---

## 5. The AR(1) coupling, and what it cost and bought

The previous estimator turned its `n_sample_px` budget into a *block count* —
`2_000_000 // 512²` = **seven** 512 px blocks — and drew them uniformly over a grid that is 73%
invalid. Regionally those seven blocks covered most of southern Africa and looked stable. On the
global grid, ρ(h = 15) ranged **0.319 to 0.827** over three seeds at two sample sizes and did not
converge when the sample was doubled.

Targeting valid *pairs* rather than blocks was necessary but not sufficient — still 0.11–0.16
between seeds at 20M pairs, because ~130 blocks is an effective sample of *blocks* and the field
is correlated well inside one. The estimator now reads evenly spaced full-width row stripes:
deterministic, no seed, spanning every longitude. Independent stripe phases agree to 0.011–0.025.
Guarded by `tests/test_horizon_autocorrelation.py`.

**Measured ρ = {10: 0.523, 15: 0.497, 20: 0.771}** — far below the 0.9 that
`generate_ensemble.py` falls back to when `--rho_json` is missing.

`horizon_autocorrelation` is written only by `run_hindcast_folds.py --stage residuals`, which
`run_region_loop.sh` never calls. **So every regional ensemble ever scored in this project,
including Africa's 111/133, coupled its horizons at 0.9 without measuring.** The fingerprint is
in Africa's own card:

```
T4.1,between-horizon corr 2005->2010,0.8997,within +/-0.10 of nan,,
T4.1,between-horizon corr 2010->2015,0.9006,within +/-0.10 of nan,,
T4.1,between-horizon corr 2015->2020,0.9000,within +/-0.10 of nan,,
```

Three values pinned to the constant they were fed, scored against a NaN target, contributing
nothing. The global card instead reads 0.5155 / 0.4869 / 0.7590 against measured targets
0.523 / 0.497 / 0.771 and **passes all three to within 0.008**. The generator reproduces the
measured coupling essentially exactly.

**What it plausibly cost: T4.2.** ρ does not change the marginal spread at any single horizon —
under `z_h = ρ z_{h−1} + √(1−ρ²) ε` each `z_h` is standard normal whatever ρ is. But T4.2 scores
whether the *sample* spread over 400 members is non-decreasing in lead time, and at ρ = 0.9
successive horizons' sample spreads move together, so monotonicity is nearly automatic. At
ρ ≈ 0.5 they fluctuate independently and violations surface. If that is right, **Africa's 0.7118
was flattered by an unmeasured constant and 0.5526 is the honest number.** This is a hypothesis
with a mechanism, not a result: it is directly testable by regenerating a small ensemble at
ρ = 0.9 and re-scoring T4.2 alone, and that has not been done.

---

## 6. T3.5 is a spurious failure

`lon seam continuity` is a hard gate and it failed with `mean |Δ| across seam nan vs interior
0.02983`. A NaN is not a discontinuity.

Measured: **columns 0 and W−1 hold 0 valid pixels out of 17,111 rows.** The antimeridian is open
ocean for its entire length in this dataset, so the metric has nothing to compare, `nanmean` of an
empty selection is NaN, and `np.isfinite(nan)` scored it as a failed hard gate. The longitude wrap
itself is fine — `wrap_lon=True` is recorded in the store attributes and the field is generated
periodically in longitude.

`validate_ensemble.py` now marks the row reported-only when `n_seam == 0`, with the count in the
note. **The published row above is uncorrected**; re-running `--stages spatial` (~19 min) would
replace it in place, as Africa's T3 splice did. The corrected denominator is 112/145.

This is the fourth scoring bug in this project to announce itself as a result that could not be
true, after the three in `CLAUDE.md` rule 3. The pattern holds: when a number is impossible,
suspect the metric.

---

## 7. Why the three totals are not comparable

**Read rows, not totals.** Three things differ between this card and Africa's beyond the model:

1. **T4.1 is scored here and was not there** (§5), moving 3 rows into the denominator.
2. **846 ecoregions instead of 158.** `docs/validator_scaling.md` §7 records that with 18
   ecoregions southern Africa's T2.2–T2.5 were *quantised too coarsely to land in their own
   targets*, and that about half of Africa's margin over southern Africa was that effect rather
   than model quality. Global removes the remainder of it.
3. **A fourth block scale.** 1000 km rows exist here and nowhere else.

And the standing caveat from `CLAUDE.md`: a single scorecard carries **±7 rows of run-to-run
noise** at k = 5, which is larger than several of the differences above.

---

## 8. What this says about shipping

The central field is sound. Skill against persistence matches Africa at the horizons that matter,
raw coverage is the closest to nominal of any extent measured, and the aggregate-scale coverage
that motivated the whole project holds at every block scale from 1 km to 1000 km. Nothing here
argues against the central forecast or the interval bounds as published rasters.

The **far-field marginal is the one thing that should not ship unexamined.** An ensemble that
emits 2.3% of the observed increase rate and 28× the observed decrease rate beyond 100 px from
past change will, in a forward product, systematically under-state where new development can
appear and invent losses on pristine land. That is a visible, mappable error in exactly the
places a reader would look for surprises.

~~The fix is named and narrow — a physical floor on the marginal's left tail, plus a remote-band
tail bound chosen on global rather than southern-African evidence. Both are Phase 3 marginal
changes: they need no retraining, and they cost one regeneration and one re-score.~~

**Superseded by Stage B (2026-08-20).** Neither half survived measurement. The physical floor
is already implemented (§3.2). The tail bound **saturates**: swept on Africa, 0.999 and 1.0 give
identical far-field rates and even unbounded the far field emits 1.7–26% of the observed
increase rate. Raising it fixes the *mid* field (30–100 px: 0.109 → 0.449 at h=5) and is worth
taking, but the far field needs a **width-head change**, which does require retraining.
`scripts/score_model_experiment.py` adds the constraint that makes it non-obvious: far-field
intervals are already **2–4× too wide** and over-cover at 0.997 against 0.95, so the lever is
tail *shape* at fixed or reduced bulk width, not widening. What they
would disturb is T5.2 and T1, which is why `predict_change_rates.py` exists — it prices a
candidate marginal in 30 s against the sampled ensemble's hours.

Two things this card does **not** settle, and which should not be asserted until they are:
whether T4.2's regression is the ρ correction (§5), and whether T2.5's rejection is a real
deviation or an over-powered χ² (§4).

---

## 9. Cost, for planning the next one

| stage | wall clock | peak RSS |
|---|---|---|
| T5 hard gates | 6.0 h | 58.12 GB |
| T1 percentiles | 21 min | 13.37 GB |
| **T2 aggregate** | **17.2 h** | 59.60 GB |
| T2.5 rank | 2.6 s | 6.49 GB |
| T3 spatial | 19 min | **67.77 GB** |
| T4 temporal | 4.25 h | 16.79 GB |
| T6 change | 9.4 min | 6.99 GB |
| T7 + T8 | ~1.4 h | — |
| **total** | **29.7 h** | |

T2 is 58% of the run, and the fourth block scale is most of why. Dropping to `1,10,100` would
save several hours at the cost of the 1000 km rows.

`--mem_budget_gb 60` was honoured everywhere except **T3's `build_z_field`, which peaked at
67.77 GB — a 13% overshoot of its own budget**. It survived on a 125 GB box and would not have on
a smaller one. That stage is not bounded as tightly as `docs/validator_scaling.md` §3 implies, and
it is the one to fix before anyone runs this under a real memory ceiling.

Ensemble generation: 400 members × 4 horizons in **51.6 min, 434 GB**; the null 417 GB. Member
seeds are `seed + 1_000_003·m + 101·h`, deterministic and independent of M, so either store
regenerates exactly.
