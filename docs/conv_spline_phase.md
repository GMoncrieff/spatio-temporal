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
(rule 3). **Compute it from `triple()`, not from a channel** — under `--free_scale` there is no
`scale` channel to read, so a channel-based implementation would work on b1 and E0a and silently
break on exactly the arms the row exists for. A pooled mean width cannot express it: two horizons have to be compared *at the same
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
| **E1a** | `--head_family isqf --isqf_tails True --isqf_space logit\|neglog` | Learned exponential tail rates β_L, β_R on unbounded transformed support. ISQF's actual contribution, which E1 drops. CRPS still evaluated in HM. **The tails' domain is unsettled** — see §4.1. | 18 params/h |
| **E1b** | `--head_family isqf --free_scale True` | E1 minus the cumulative-in-horizon scale: increments carry the width directly, no renormalisation to unit 95% span. Park et al.'s own arrangement. | 15 params/h |
| **E1c** | E1a + E1b together | Park et al. (2022) as published, adapted only where the ConvLSTM forces it. **Runs unconditionally** — it is the fourth cell of a 2x2 and completes it. | 17 params/h |
| **E2a** | `--head_family pwl --free_scale True` | E2 with the anchor/scale factorisation removed: softmax heights replaced by unnormalised positive increments, so the 95% width is emergent rather than injected. | 15 params/h |

E0 and E2–E5 are the issue note's staged plan, ordered by how much has to be touched. E1 and E2
are the two literature-derived heads. The last five rows are a group rather than a continuation
of that ordering — they are §4.1 below, and they are listed after E5 because each is a delta
from an arm above it rather than from b1 directly.

**Param counts assume `default14` (15 knots, 14 bins).** `--free_scale` drops the `scale`
channel, so `pwl`/`isqf` go 16 → 15 and the incumbent would go 29 → 28; E1a's two tail rates
are `+2`. Simplicity is a scoring criterion (`CLAUDE.md`), and these are the numbers reported
beside the metrics — E1b and E2a are the *cheapest* heads on the slate, not merely different
ones.

**Flag status, so nothing runs inert.** `--spline_cumulative_width` (E0a) exists today.
`--free_scale`, `--isqf_tails` and `--isqf_space` (E1a, E1b, E1c, E2a) **do not exist yet** and
are the names proposed here, not names to pass at a shell before the code lands. `--free_scale`
is one flag across all three head families with a per-family implementation — specified below,
and specified in one place for the reason rule 2 exists. Each flag needs a test that it changed
something against a seeded control before its arm is run, per the convention in
`tests/test_conv_spline_flags.py`; for `--free_scale` on the incumbent that test must cover
**all five** methods `scale` is read from, since two of them are the gates themselves.

### 4.1 The scale arms — one constraint, five ways of removing it

E0a, E1a, E1b, E1c and E2a share a target that E0–E5 never touch: **the three things this model
imposes on its own quantile function from the outside rather than learning.** The
horizon-cumulative scale (the 95% width cannot shrink with lead time), the anchor/scale
factorisation that makes width an injected channel at all, and the bounded clamp to [0, 1].
None has been ablated. They are grouped here because they are entangled — dropping the scale
channel drops the accumulation with it, and the clamp and the injected scale both constrain
where the outer knots may sit — and the ordering below is what keeps a null readable in spite
of that.

**E0a — the constraint alone, on the incumbent.** The cheapest possible read, and the one that
makes everything below separable. b1's rational-quadratic head, unchanged in every other
respect, with the horizon accumulation removed so each horizon's 95% width is set by its own
channel rather than accumulated across lead times. Running this on the incumbent rather than on
a new head is the whole point: if E0a's bands widen with lead time *unprompted*, the constraint
is free insurance, and every downstream free-scale arm inherits that finding instead of
re-establishing it against a different spline class at the same time.

> **This flag already exists.** `--spline_cumulative_width False` does exactly this, is
> documented in `train_lightning.py`, and is pinned by `tests/test_spline_cumulative_width.py`
> on both halves (the ablation ablates; `Q(u)` increasing in `u` is untouched). E0a is a no-code
> run.
>
> **`--free_scale` is not an alias for it, and E0a must not be respelled as one.** `--free_scale`
> (defined below) also removes the accumulation, but only as a consequence of removing the
> anchor/scale factorisation entirely — it is a strict superset. That is exactly why E0a exists
> as its own arm on the narrow flag: an arm carrying `--free_scale` answers *"was the
> anchor/scale machinery earning its keep"*, and the two questions are no longer separately
> attributable inside it. E0a is what makes the free-scale arms readable, and it is the only
> arm in §4.1 that changes one thing.
>
> One precision on the incumbent: `scale_pre` is always read from the width head's channel and
> `QuantileSpline.from_channels`' `scale_pre=None` branch is unreachable from `forward`. The two
> paths read the same channel and differ only by an extra softplus, so `--spline_cumulative_width`
> is the whole of the narrow ablation; there is no separate "injection" knob to turn.

#### `--free_scale`: one semantic change, three implementations

**The 95% width becomes emergent rather than injected** — whatever the fitted increments imply,
per horizon, with no cross-horizon accumulation. One meaning, implemented per family:

| family | what changes |
|---|---|
| `spline` (incumbent), `pwl` | unnormalised positive increments (`\|·\| + tol`) in place of the softmax heights; **no `scale` channel**; `Q(u) = anchor + (v(u) − v_mid)`, with the `/ v_span` division dropped |
| `isqf` | removing the `scale_pre` renormalisation, and nothing else — that head has no factorisation natively |

**What does not change.** The anchor stays at `Q(0.5)` and still receives `hm_t0` through the
zero-init persistence skip: adding persistence to the median is a defensible prior, and adding
it to the bottom of a ladder is not. The u-knots stay fixed. 0.025 / 0.5 / 0.975 stay exact
knots, so `triple()` stays a gather rather than an interpolation.

**Three things need porting rather than deleting**, and the third is not obvious from the
forward pass.

1. **`MIN_SCALE` loses its referent.** Re-express it as a floor on the span `v_hi − v_lo` in the
   same units, enforced at construction. Without it CRPS can drive the width to zero on the
   quiet majority of pixels and `anchor + (v − v_mid)` silently returns a constant — which reads
   downstream as a perfectly confident forecast rather than a degenerate one, and DEGEN only
   catches that after the fact. The failure mode is already written out at
   `quantile_spline.py:130`; move the comment with the constant. **Resolve the number while
   porting**: that comment justifies 3e-5 (members are int16 × 1/32767, so 3e-5 is the smallest
   width the product can represent at all) while `MIN_SCALE` is 1e-6. Inheriting the mismatch
   silently is how a floor stops binding.
2. **The horizon-narrowing row must be computed from `triple()`**, since width is no longer a
   readable channel. §3 says the same thing from the scorer's side.
3. **On the incumbent, `scale` and `_v_span` are read in five methods, not one** —
   `_g` (290), `ppf` (298), `triple` (303–305), `cdf` (319–320) and `log_dq_du` (362–364). The
   first three are the ones anyone would look for. The last two are the ones that matter most:
   `cdf` **is the PIT** and `log_dq_du` is the implied density, so they are gate 2 and gate 1
   respectively. Port all five or the arm's own gates read garbage and the run looks like a
   model failure rather than a porting bug — rule 2 in its exact form, and rule 1's "a metric
   failure and a model failure are indistinguishable from the outside". Under `--free_scale`
   they collapse cleanly (`_g` → `v − v_mid`, `cdf` → `v = (y − anchor) + v_mid`, `log_dq_du`
   → `log(dv)`), which is an argument for the change rather than against it — but only if all
   five move together.

   `pwl` and `isqf` do not have this problem: `_PWLBase` applies the factorisation once in
   `PWLQuantile.__init__` and every method downstream reads `q_knots`, so E2a and E1b are
   one-site changes. **The incumbent is the expensive implementation of this flag, and no arm
   on the slate currently needs it** — implement `spline` + `--free_scale` only if an E6–E8
   candidate asks for it.

**Scope this implies, stated so it is not re-derived later.** Dropping the scale channel entails
dropping the horizon-cumulative constraint. An arm carrying this flag therefore answers *"was
the anchor/scale machinery earning its keep"* and **not** the narrower *"was horizon
monotonicity binding"*. The two are not separately attributable inside such an arm — which is
the whole reason E0a runs first, alone, on `--spline_cumulative_width False`.

**E1a — a learned tail rate on unbounded support.** ISQF's actual contribution, which E1 drops.
The head fits in a transformed space where unbounded tails are admissible — `logit(HM)` for the
symmetric version, `−log(1−HM)` for the one-directional version that matches the product's tail
question — and on the outer tails, below u = 0.001 and above u = 0.999, it extrapolates through
exponential tails whose rates β_L, β_R are **trainable** rather than pinned to the two outer
knots. (Where exactly those tails attach is not settled; see the note below.) Mapping
back through the inverse transform respects [0, 1] structurally, so the clamp becomes redundant
rather than load-bearing. Observed HM runs 0.00029 to 0.950, so the transform needs no epsilon
fudge.

> **Evaluate CRPS in HM space, not in the transformed space.** The closed form in z is tempting
> and it silently changes the objective: weight per unit log-odds allocates very differently near
> u = 0.999 than weight per unit HM, and the point of E1a is to isolate the tail *mechanism*.
> z-space training is E3's question, and it is a follow-up here if and only if E1a moves the
> far-field statistics. Two knobs on one axis read as two findings (rule 10 in `CLAUDE.md`).
>
> **Settle the tail's domain before coding E1a, or the flag is inert.** As written, the tails
> attach "beyond the outermost knots at u = 0.001 / 0.999" — repeating a claim in
> `ISQFQuantile`'s docstring that **is not true of any grid this repo has**. `validate_knots`
> hard-requires `arr[0] == 0.0` and `arr[-1] == 1.0` (line 109), and `default14` runs
> `0.0, 0.001, …, 0.999, 1.0`. There is no u outside the outermost knots, so tails attached
> there would cover an empty set: accepted, logged, and inert — a null that never ran.
>
> Prefer having the tails **replace** the outermost bins `[0, 0.001]` and `[0.999, 1]`, anchored
> at the 0.001 / 0.999 knots, rather than relaxing `validate_knots`. That keeps one definition
> of the grid contract, gives each tail a real domain, and matches the paper's shape (their
> spline covers `[α, 1−α]` and the tails cover the rest; here α = 0.001). Relaxing the validator
> for one head family breaks the `[0, 1]` span invariant everywhere else for a grid property
> that is not the experiment.

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
adapted only where the ConvLSTM architecture forces it.

**E1c runs unconditionally.** E1, E1a, E1b and E1c are a complete 2×2 over {learned tails, free
scale} — neither, tails only, scale only, both — and three cells of a 2×2 cannot separate an
interaction from two main-effect nulls. The bounded clamp and the injected scale interact by
construction, since both constrain where the outer knots can sit, so a joint effect is a live
hypothesis rather than a remote one.

Gating E1c on movement in E1a or E1b would drop the fourth cell in **exactly the case that makes
it diagnostic**: if both are null, an interaction is the only remaining explanation, and the run
that would test it is the one the gate skips. Such a gate is an unstated assumption — that the
two departures are additive — wearing the clothes of a cost saving, and it fails silently: a
run that never happened reads downstream as a result, the same family of error as a flag that is
accepted, logged and inert. Two nulls and a skipped cell would get written up as "the paper's
mechanism does not transfer", which is a claim the design would not have earned. Run it last for
ordering, not for permission.

**E2a — the linear head with an emergent width.** E2 with the anchor/scale factorisation
removed, which requires more than deleting a channel. `_cumulative_v` softmaxes the heights and
then divides by `v[..., -1:]`, so the ladder spans exactly 0 → 1 twice over; stripping `scale`
naively hands every pixel a span of 1.0 — a broken model, not a free one. E2a is therefore the
`pwl` row of the `--free_scale` table above: unnormalised positive increments, no `scale`
channel, and `Q(u) = anchor + (v − v_mid)` with the `/ v_span` division dropped, making the 95%
width emergent from the increments rather than injected.

> **E2a and E4 do not compose as written.** `gap_floor_heights` computes its floor in *normalised
> v units* as `dp_k / (f_max * scale)`, deriving it from `w_k = scale * h_k / v_span` — the
> factorisation is load-bearing in that expression, and `--free_scale` deletes both terms. So
> `--spline_gap_floor True --free_scale True` is not a combination, it is a re-derivation: the
> floor has to be restated directly on the increments in absolute HM (`inc_k ≥ dp_k / f_max`),
> which is in fact the simpler statement of the same physical claim. Worth knowing before
> anyone reads E4 × E2a as a free crossing of the slate.

> **The convergence is exact, and one decision away from being total.** Written out, E2a is
> `anchor + cumsum(inc) − cumsum(inc)[mid]` and E1b is `q0 + cumsum(inc)`: the same ladder, up
> to which knot the free location parameter attaches to. So they differ in exactly one thing —
> **where persistence enters**. E2a's `anchor` is `Q(0.5)`; E1b's `q0` is `Q(u_knots[0])`, and
> `U_KNOTS_DEFAULT[0]` is **0.0**, so `hm_t0` is currently added to the 0th percentile, not the
> median.
>
> That is not cosmetic. The median 20-year change is 0.0001 against a core spanning ~0.0036, so
> the two placements put the persistence prior in materially different places. It is also the
> case the `--free_scale` "what does not change" clause rules out — and ISQF is the head that
> already does it.
>
> **Decide before coding, and prefer keeping E1b paper-faithful** (`q0 + hm_t0` at `Q(0.0)`).
> E1b's entire purpose is to test Park et al.'s arrangement; moving its skip to the median makes
> it E2a with a rename and leaves the paper's arrangement untested. Keeping it means **E2a and
> E1b are one comparison apart rather than the same head**, so both are needed and neither is
> mere confirmation — which supersedes the "E1b primary, E2a confirmation" reading this doc
> carried before the `--free_scale` semantics were fixed. What E2a-vs-E1b then measures is
> precisely the persistence-anchor placement, with everything else held identical: a cleaner
> question than either arm was originally scoped to ask.

**Order.** E0a first and alone, because it is the only arm that changes one thing, and because
every arm below it conflates the constraint with the factorisation. E1a and E1b next, in either
order — independent subtractions from E1. E2a paired with E1b rather than following it: with the
`--free_scale` semantics fixed, the pair is a controlled comparison of where persistence enters
the ladder, so reading either alone wastes the control. E1c last — last in sequence, not
contingent on what precedes it; it is the cell that closes the 2×2.

**Cost, since none of these is optional.** Five arms at ~50 min each (two folds in parallel, per
the loops table in `CLAUDE.md`) is ~4 h for the group, on top of the six-arm E0–E5 slate and b1's
three seeds. `run_conv_spline_slate.sh` currently knows E0–E5 only; §4.1 needs adding to it, and
it should refuse to start without b1's floor for the same reason the existing slate does.

Each arm still needs a test that it changed something against a seeded control
(`tests/test_conv_spline_flags.py`) — a flag that is accepted, logged and inert reads downstream
as a null rather than as an experiment that never ran. Two arms here have a specific way of
failing that test silently: E1a, if its tails are given no domain, and `--free_scale` on the
incumbent, if fewer than all five `scale` sites are ported.

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
