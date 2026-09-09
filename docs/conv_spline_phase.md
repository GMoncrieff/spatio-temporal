# The conv-spline phase

Branch `conv-spline`. Started 2026-09-07. **No experiment has been run.** Everything below is
code that exists and a plan that has not been executed.

The phase evaluates the distributional ConvLSTM **alone**. There is no ensemble, no post-hoc
chain, no copula, no width calibration and no empirical marginal — those were deleted rather
than left dormant, because a dormant path is one nothing runs and nothing checks. The model's
own quantile function is the entire product.

---

## 1. What is being fixed

Two defects, both measured on the current model, both visible in the companion figures
(`marginal_change.png`, `fitted_densities.png`).

### The picket fence

Every one of twelve random pixels shows the same shape: a cluster of 2–6 needle-thin spikes
beside the persistence value, on an otherwise smooth broad hump.

| | measured | target |
|---|---|---|
| gaps ≤ 3 quantisation steps (~1e-4) | 4.5% | ~0 |
| gaps exactly zero | 0.7% | — |
| needle mass, median pixel | 4.5% | ~0 |
| needle mass, 90th-percentile pixel | 21% | ~0 |
| max implied density | 677 | ≤ 578 |

The 0.7% against 4.5% split is the tell. Levels collapsing *toward* each other without
collapsing *onto* each other is exactly what a strictly-positive-derivative monotonicity
constraint produces when the target wants a flat segment. It is not a malfunction — the head
is being asked to spend five bins on a region where nothing varies.

**Why the target does that.** Value range covered by each decile of the observed change
(10.3 M land pixels, 2000–2020):

| u | change | share of value range |
|---|---|---|
| 0.0–0.1 | −0.43158 → −0.00070 | 35.33% |
| 0.1–0.2 | −0.00070 → +0.00005 | 0.06% |
| 0.2–0.3 | +0.00005 → +0.00017 | 0.01% |
| 0.3–0.4 | +0.00017 → +0.00038 | 0.02% |
| 0.4–0.5 | +0.00038 → +0.00086 | 0.04% |
| 0.5–0.6 | +0.00086 → +0.00294 | 0.17% |
| 0.6–0.7 | +0.00294 → +0.00919 | 0.51% |
| 0.7–0.8 | +0.00919 → +0.02274 | 1.11% |
| 0.8–0.9 | +0.02274 → +0.05741 | 2.84% |
| 0.9–1.0 | +0.05741 → +0.78800 | 59.91% |

Flat from u ≈ 0.1 to 0.6, near-vertical above 0.9. The lower tail is *genuinely* wide — 35%
of the range, with real decreases to −0.43 — so density there is justified. The problem is
the middle. The head's tail-dense knots are **symmetric**, and the target is not.

The distribution is also trimodal (noise/persistence at +0.00015, small incremental change at
+0.0029, substantial change at +0.017), clearly separated in asinh space. **Any fix that
forces unimodality is wrong.**

**A point mass at zero is also wrong**, and measurably: exact zero occurs in 0.00% of land and
the core sits at +0.00031, not 0. There are two errors to avoid in opposite directions — a
spike at exactly zero, and the current needles at ~7× the sharpness the noise supports
(~1e-4 wide against a core σ of 0.00069). The target is a sharp *continuous* mode of
measurable width, and no sharper.

Not the cause: int16 storage. Only 2.0% of gaps fall within a single quantisation step.

### PIT structure

| | measured | target |
|---|---|---|
| PIT weighted mean | 0.5537 | 0.50 ± 0.01 |
| mass left of centre | 37.7% | 50% |
| spikes at 40–60 bins | ~0.25, 0.51, 0.73 | flat |
| −0.05→0 predicted/observed | 1.70 | ~1.0 |

The spikes are real, not noise: RMS deviation grows 0.577 → 0.735 → 0.750 at 20/40/60 bins
where noise predicts ×1.41 and ×1.22 (observed ×1.27 and ×1.02), and peak locations are stable
at every resolution. Between adjacent needles are near-zero-density gaps; an observation
landing in one takes a PIT set by the needle positions, and the pattern recurs across pixels
because the persistence anchor is shared.

About 3.5 M cells — 15% of the sample — are displaced one bin left, **across zero**, into a
region where change does not occur. HM does not meaningfully decrease.

### The incentive problem underneath both

CRPS in raw HM units is dominated by the tail. The persistence core spans ~0.0036 against a
range above 1.2, so getting the core badly wrong costs almost nothing. **That is very likely
why the head settled on a fence: nothing ever penalised it.** More knots alone will not fix
this — they give the head more room to do the same thing.

---

## 2. Step 1 — the b1 baseline

`b1` is `e1` plus exactly one change, accepted without question:

> **the neighbourhood context is injected into the trunk**, alongside elevation and climate,
> repeated across timesteps — not into the heads.

Until now the context reached `last_hidden` only, so the ConvLSTM never saw "can change happen
here at all" and could not combine it with its own spatial and temporal features; it could only
have the answer applied to its output. Nothing ever justified that. The precompute-on-the-full-
raster requirement (radii ≥ 30 px saturate inside a 128 px chip) is about where the covariate
is *derived*, not where it is *consumed*.

```bash
./scripts/run_conv_spline_baseline.sh          # three seeds, Africa
```

**Three seeds, not one.** The previous phase judged ten variants against a floor measured from
three replicates that all landed in one mode, read the band as 0.86, and adopted nothing when a
variant's own replicates spread by 4.3. A floor is a property of *this* configuration and it is
measured before anything is ranked against it.

**Nothing here is comparable to a dist-convlstm number.** The trunk change alone makes b1 a
different model, and the scorecard carries two gates e1 never had.

---

## 3. The scorecard

`scripts/score_distributional_model.py`, which now reports the gates beside CRPS rather than in
a side script — a diagnostic that has to be remembered is one that stops being run.

| what | metrics |
|---|---|
| accuracy | `crps`, `crps_skill` (vs persistence), `rmse`, `skill` |
| calibration | `pit_ks`, `pit_mean`, `cov50/80/95/99` |
| **gate 1: the fence** | `needle_mass_median/p90`, `max_density_p99`, `over_f_max_frac` |
| **gate 2: PIT structure** | `pit_rms_se_{20,40,60}`, `pit_growth_vs_noise`, `zero_leak_*` |
| placement | `exceedance_abs_log10`, `tail_reach` |
| **width vs lead time** | `width95_{h}`, `width_narrows_frac_{h,h'}` |

### Every fence statistic is reported twice, and this is load-bearing

A needle is a pair of *adjacent* quantile levels whose values collapsed together, and "adjacent"
is a property of the grid the raster was written on. A metric read off that grid moves when the
grid moves — so E0, which re-spaces the published levels and retrains nothing, would look like
it fixed the model.

- **`*_export`** — on the raster's own grid: what a consumer of the product sees. E0 moves this
  *by construction*, and that is the point of E0.
- **`*_ref`** — on `src/qf_diagnostics.REF_U`, a frozen 256-level grid. The raster is a
  piecewise-linear quantile function so it can be evaluated anywhere, and subdividing a segment
  leaves its implied density unchanged — which makes this a property of the forecast
  distribution rather than of the export. **E1–E5 must move this to have done anything. E0
  should barely move it.**

Reporting one alone is how a rendering fix gets recorded as a model fix.

### Reading the PIT structure statistics

`pit_rms_se_B` is the RMS deviation from uniform in units of its own standard error, so it is
~1 under a calibrated forecast **at every bin count** — which is what makes 20 and 60 bins
comparable at all. Raw RMS grows as √B and says nothing across resolutions.

`pit_growth_vs_noise` is secondary and has a measured blind spot: it reads 0.97 for features of
σ 0.004, 0.80 at σ 0.01, 0.67 at 0.02, 0.61 at 0.04. **A feature narrower than the finest bin
grows exactly like noise even though it is real.** Read growth only after `rms_se` has
established there is structure at all.

### The width-vs-lead-time row, added for the scale arms

`width95` narrowing with lead time is currently **unmeasurable, not absent**. The spline head
accumulates non-negative width increments across horizons (`cum = cum + softplus(step)` in
`SpatioTemporalPredictor.forward`), so `width95(20) ≥ width95(15) ≥ …` holds by construction and
no statistic could ever read otherwise. The free-scale arms in §4.1 — E0a, E1b, E1c, E2a —
remove that accumulation, which makes the quantity real for the first time, so it needs a row
before those arms run, not after.

**E1a is the control on this row, not a fourth free-scale arm.** Its learned tails act beyond
u = 0.001/0.999 while `width95` is interior (u = 0.025 to 0.975), so E1a on its own leaves the
accumulation intact and should read exactly 0.000 here. If it does not, either the tail
mechanism is reaching further in than intended or the row is measuring something else — and
either way that is worth knowing before E1c combines E1a with a freed scale.

`width_narrows_frac_{h,h'}` is the **fraction of pixels** where `width95(h) < width95(h′)` for
`h > h′`, reported for each of the six horizon pairs and **per row, never as a pass count**
(rule 3). A pooled mean width cannot express it: two horizons have to be compared *at the same
pixel*, and `score_distributional_model.py` currently pools each horizon independently, so this
row is a join the scorer does not yet do. Under the accumulating constraint it is 0.000 by
construction, which is also the control that proves the row discriminates (rule 5) — if it does
not read exactly zero on b1, the statistic is wrong before any arm is judged.

**The DEGEN rule binds harder here than anywhere else.** A freed scale can collapse toward
`MIN_SCALE` (1e-6) across the quiet majority of pixels — 68% of land is in the persistence core
— and a floor band of width ~0 clears every margin test it is given. Read this row beside
`width95_{h}` in absolute HM and beside `cov95`: a narrowing fraction that rises while `cov95`
falls is a collapse, not a finding.

### Save and show plot

For each experiment, save a plot of:
1) estimated distribution for 9 (3x3 grid) selected pixels.
2) The pit distribution for 0-1 with 20 bins
Embed these plots in the scorecard
---

## 4. The experiment menu

Every entry is one stated delta from b1, additive, defaulting to today's behaviour.
`./scripts/run_conv_spline_slate.sh` — refuses to start without b1's floor on disk.

| | flag | what it tests | cost |
|---|---|---|---|
| **E0** | `--u_grid_spacing skew` | 8 of the 64 published levels move out of u 0.1–0.6 into 0.6–0.9. **Export-only** — the same trained model, evaluated at different u, reversible by re-running the export. | no retrain |
| **E1** | `--head_family isqf` | Park et al. (2022) adapted for bounded support: fixed outer levels, learned values via cumulative positive increments, linear spline between. Exponential tails **dropped** (unbounded; HM is not). Cumulative-in-horizon scale **kept**. see https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.distributions.html gluonts.torch.distributions.ISQF for reference implementation| 16 params/h |
| **E2** | `--head_family pwl` | The incumbent's construction with linear pieces and **CRPS in closed form**. No quadrature, no nodes, nothing to checkpoint. Whether C1 was ever load-bearing has never been tested. see https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.distributions.html gluonts.torch.distributions.PiecewiseLinear for reference implementation that needs to be adapted for our context| 16 params/h |
| **E3** | `--crps_z_weight 1.0` | A second CRPS term in `z = asinh(change/s)`, summed with the raw term. Attacks the incentive problem directly. | same head |
| **E4** | `--head_family pwl --spline_gap_floor True` | No segment may imply a density above 578. Makes a fence structurally impossible rather than discouraged. | +0 params |
| **E5** | `--spline_knots skew14` | The same 14 bins, re-placed for the measured right skew. `skew11` is the strong form: 11 bins, testing whether the body needed resolution at all. | −0 / −6 params |
| **E0a** | `--spline_cumulative_width False` | The incumbent head, otherwise untouched, with the horizon accumulation removed so each horizon's 95% width comes from its own channel. **Already implemented and tested**; no new code. | 29 params/h |
| **E1a** | `--head_family isqf --isqf_tails True --isqf_space logit\|neglog` | Learned exponential tail rates β_L, β_R beyond u = 0.001/0.999, fitted on unbounded transformed support. ISQF's actual contribution, which E1 drops. CRPS still evaluated in HM. | +2 params/h |
| **E1b** | `--head_family isqf --free_scale True` | E1 minus the cumulative-in-horizon scale: increments carry the width directly, no renormalisation to unit 95% span. Park et al.'s own arrangement. | 16 params/h |
| **E1c** | E1a + E1b together | Park et al. (2022) as published, adapted only where the ConvLSTM forces it. **Conditional** — runs only if E1a or E1b moves. | +2 params/h |
| **E2a** | `--head_family pwl --free_scale True` | E2 with the anchor/scale factorisation removed: softmax heights replaced by unnormalised positive increments, so the 95% width is emergent rather than injected. | 16 params/h |

E0 and E2–E5 are the issue note's staged plan, ordered by how much has to be touched. E1 and E2
are the two literature-derived heads. The last five rows are a group rather than a continuation
of that ordering — they are §4.1 below, and they are listed after E5 because each is a delta
from an arm above it rather than from b1 directly.

**Flag status, so nothing runs inert.** `--spline_cumulative_width` (E0a) exists today. `--free_scale`,
`--isqf_tails` and `--isqf_space` (E1a, E1b, E1c, E2a) **do not exist yet** and are the names
proposed here, not names to pass at a shell before the code lands. Each needs a test that it
changed something against a seeded control before its arm is run, per the convention in
`tests/test_conv_spline_flags.py`.

### 4.1 The scale arms — one constraint, five ways of removing it

E0a, E1a, E1b, E1c and E2a share a target that E0–E5 never touch: **the two places this model
constrains its own quantile function from the outside.** The horizon-cumulative scale (the 95%
width cannot shrink with lead time) and the bounded clamp to [0, 1] are both imposed rather
than learned, and neither has ever been ablated. They are grouped here because they interact —
both decide where the outer knots may sit — and because the ordering below is what keeps a null
readable.

**E0a — the constraint alone, on the incumbent.** The cheapest possible read, and the one that
makes everything below separable. b1's rational-quadratic head, unchanged in every other
respect, with the horizon accumulation removed so each horizon's 95% width is set by its own
channel rather than accumulated across lead times. Running this on the incumbent rather than on
a new head is the whole point: if E0a's bands widen with lead time *unprompted*, the constraint
is free insurance, and every downstream free-scale arm inherits that finding instead of
re-establishing it against a different spline class at the same time.

> **This flag already exists.** `--spline_cumulative_width False` does exactly this, is
> documented in `train_lightning.py`, and is pinned by `tests/test_spline_cumulative_width.py`
> on both halves (the ablation ablates; `Q(u)` increasing in `u` is untouched). **Do not add a
> `--free_scale` alias for the spline family** — that is rule 2's predicate written twice, and
> the two spellings will disagree. `--free_scale` below names the *new* code the piecewise-linear
> families need; for the incumbent the flag is already there and E0a is a no-code run.
>
> One precision: for the incumbent, `scale_pre` is always read from the width head's channel and
> `QuantileSpline.from_channels`' `scale_pre=None` branch is unreachable from `forward`. The two
> paths read the same channel and differ only by an extra softplus, so the *meaningful* ablation
> for b1 is the accumulation, not the injection. E1b and E2a are where `scale_pre=None` becomes
> a real difference.

**E1a — a learned tail rate on unbounded support.** ISQF's actual contribution, which E1 drops.
The head fits in a transformed space where unbounded tails are admissible — `logit(HM)` for the
symmetric version, `−log(1−HM)` for the one-directional version that matches the product's tail
question — and beyond the outermost knots at u = 0.001/0.999 it extrapolates through exponential
tails whose rates β_L, β_R are **trainable** rather than pinned to the two outer knots. Mapping
back through the inverse transform respects [0, 1] structurally, so the clamp becomes redundant
rather than load-bearing. Observed HM runs 0.00029 to 0.950, so the transform needs no epsilon
fudge.

> **Evaluate CRPS in HM space, not in the transformed space.** The closed form in z is tempting
> and it silently changes the objective: weight per unit log-odds allocates very differently near
> u = 0.999 than weight per unit HM, and the point of E1a is to isolate the tail *mechanism*.
> z-space training is E3's question, and it is a follow-up here if and only if E1a moves the
> far-field statistics. Two knobs on one axis read as two findings (rule 10 in `CLAUDE.md`).

**E1b — E1 with the paper's own scale arrangement.** A pure subtraction, and the cheapest of the
head arms. `ISQFQuantile`'s docstring names the cumulative-in-horizon scale as the one departure
from Park et al. it deliberately keeps; E1b removes it. The increments carry the width directly,
the ladder is no longer renormalised to unit 95% span, and horizons carry independent parameters
exactly as the Seq2Seq setting does. Location still comes from the free first knot `q0` through
the same zero-init persistence skip. The gate levels stay exact knots and `triple()` stays a
gather — what goes is the reparameterisation and the constraint, not the lookup contract.

> `ISQFQuantile.from_channels` **already handles `scale_pre=None`** correctly (it skips the
> renormalisation block and uses `|·| + tol` directly). The missing piece is only that
> `SpatioTemporalPredictor._decode` passes `scale_pre=blk[..., h * p + 1]` unconditionally. E1b
> is a decode-site change, not a head rewrite.

**E1c — Park et al. (2022) transplanted.** E1a and E1b together: learned exponential tails on
unbounded support, free per-horizon scale, fixed outer quantile levels with values from
cumulative positive increments and linear interpolation between. This is the paper as published,
adapted only where the ConvLSTM architecture forces it. It exists so that a null on E1a or E1b
can be checked against the possibility that the two departures were load-bearing *jointly* — the
bounded clamp and the injected scale interact, since both constrain where the outer knots can
sit. **Run it last, and only if at least one of E1a or E1b shows movement**; if both are null,
E1c has nothing left to explain.

**E2a — the linear head with an emergent width.** E2 with the anchor/scale factorisation
removed, which requires more than deleting a channel. Softmax heights guarantee the ladder spans
exactly 0 → 1 (`_cumulative_v`), so stripping `scale` naively hands every pixel a span of 1.0 —
a broken model, not a free one. E2a therefore replaces the softmax with **unnormalised positive
increments** (`|·| + tol`, ISQF's construction), making the 95% width emergent from the
increments rather than injected.

> That convergence is worth stating plainly: **E2a and E1b end up as nearly the same head**,
> differing only in whether location arrives as a free first knot (`q0`) or as a location
> channel (`anchor`). If that holds in the code, **E1b is the primary arm and E2a is
> confirmation** — not two independent results. Check it before reading them as two.

**Order.** E0a first and alone, because it is the only arm that changes one thing. E1a and E1b
next, in either order, since they are independent subtractions from E1. E2a after E1b, as its
confirmation. E1c last and conditional. Each still needs a test that it changed something against
a seeded control (`tests/test_conv_spline_flags.py`) — a flag that is accepted, logged and inert
reads downstream as a null rather than as an experiment that never ran.

**E6–E8 are deliberately absent.** They are chosen from what E0–E5 and §4.1 measure. Writing them now
would be guessing, and a slate whose last third is guesswork is how twenty-two experiments got
screened against a blind spot. Candidates already visible:

- **`--crps_z_scale` sweep** — measured, at s = 0.001 the transform moves tail-vs-core
  weighting from 166× to 17.5×. The issue note claims it makes them "comparable"; it does not,
  it makes them closer. A smaller `s` compresses harder, and that is a one-parameter knob.
- **E2 × E5 together**, if both clear alone.
- **more knots (28–32)** — only if E4's floor visibly *binds* on a large share of pixels, which
  would mean the head genuinely lacks resolution for three modes.

Before locking in E6-E8, propose candidates, indicating the most promosing, and allow the eselection of 3.

## 5. Rules this phase inherits and must not relearn

Ordered by how much they cost when violated.

1. **Verify the measurement before believing the finding.** Measurement bugs have outnumbered
   model bugs throughout. Three bugs were already caught setting this phase up, each of which
   would have read as a null result: a closed-form CRPS returning *negative* values; a knot
   preset silently missing the gate levels; and `pwl`/`isqf` taking the triple-head code path
   because `head_family == 'spline'` was written out in three places.
2. **Score against persistence, not zero.** The median 20-year change is 0.0001.
3. **Judge on per-row values, never a total pass count.**
4. **A band that is too narrow is as unreadable as one too wide**, and it fails in the more
   dangerous direction: with width ~0 every variant clears "margin exceeds the band's width".
   `compare_conv_spline_runs.py` marks those rows `DEGEN` and reports movement as "floor
   unmeasured" rather than as a win.
5. **Prove a check fires on a control**, or it checks nothing. The trunk-context fingerprint was
   verified to discriminate before `conv_spline_base.sh` was allowed to grep for it.
6. **"The flags I passed" is not "the flags that took effect."** `run_hindcast_folds.py` injects
   the frozen product's loss weights and `--extra_train_args` is appended last, so anything
   `BASE_ARGS` does not name is silently inherited. `conv_spline_base.sh` names every weight and
   reads them back out of the log.
7. **Iterate on Africa.** Southern Africa's far-field band holds *zero pixels* and its
   `[0,0.01)` HM stratum is 6% of the region against 40% of Africa. A stratified finding
   measured there is provisional; this already cost the project a whole phase. The globe is for
   promotion, on instruction, never for iteration.
8. **The stitched hindcast is a quilt of five fold models.** Anything scored stays on
   `--stitch_mode holdout`; `mean` is for display rasters only. Never score a mean-stitched
   raster.
9. **Never edit a shell script while it is running.** Bash reads by byte offset. Copy to a new
   name and launch that.
