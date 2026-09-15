# The conv-spline phase

Branch `conv-spline`. Started 2026-09-07. **No experiment has been run.** Everything below is
code that exists and a plan that has not been executed.

> **2026-09-14 — the code met real data for the first time, and three things had to change
> before `b1` could run.** Recorded here because each would have been read as a result.
>
> 1. **The prediction writer could not decode two of the three head families.** It called
>    `splines_from_output` — the rational-quadratic decoder, 29 channels per horizon — for
>    every family, so `pwl` and `isqf` (16) raised `ValueError: expected 116 spline channels
>    after 12, got 64` at the first prediction batch. **E1, E2, E4 and all four of §4.1's
>    free-scale arms** would each have trained for fifty minutes and written no raster. It now
>    goes through `SpatioTemporalPredictor._decode`, which is the call the loss already makes.
> 2. **Both gates crashed every real scoring run.** `load_row` streams the quantile-function
>    raster in horizontal bands and does `del cell["qf"]` before concatenating them;
>    `gate_stats` read `cell["qf"]` back out, so the pooled stratum raised `KeyError: 'qf'`.
>    The unit test passed because it built `cell` itself with the quantile function still in
>    it. Both gates are now reductions over per-pixel quantities computed inside the band loop
>    (`qf_diagnostics.gate_per_pixel` / `gate_reduce`), which also removes the `64 x n_px` copy
>    per stratum that §22 of `CLAUDE.md` warns about. Pinned by an exact identity against the
>    whole-raster statistic and by an end-to-end control proved to fire on the reverted fix.
> 3. **`b1` was not `e1` plus one change.** `BASE_ARGS` named no context flags, so it trained
>    on argparse defaults — `--context_radii 1,3,10,30,100`, no HM summaries, **eight** trunk
>    channels — where `e1` is `--context_radii 3,30,100 --hm_context_radii 3,30,100
>    --hm_context_stats mean,max`, **twelve**. `verify_context_wiring` could not see it: it
>    compared the module against the same defaults and read 8 == 8. The flags are named now and
>    the count is pinned as `EXPECT_CTX_CHANNELS=12`.
>
> The suite stands at **425 passed / 7 failed**; all 7 fail identically on `dist-convlstm`.

## 1a. What b1_s42 actually measured, and what had to be fixed to measure it

Run 2026-09-14: Africa, folds 1+2, `fold_mask_b4`, 150 epochs, one seed. **The first scorecard
was unreadable, and three separate things were wrong with the instrument before anything could
be said about the model.**

**The fence gate was pinned at its own ceiling.** `max_density_p99` read **1531.8 at every
horizon** — which is exactly `dp_max / one int16 quantum` (0.046747 / 3.0518509e-05). An
identical value at four horizons is not a density, it is a statistic reporting that the widest
u-bin has collapsed onto a single stored code. The qf raster is now written **float32**
(`--predict_qf_dtype float32`, named in `BASE_ARGS`) and the reader takes the scale off the
raster's own tag instead of a hardcoded constant — that quantity used to be spelled in three
places. With the floor removed the same weights report a max density of **1.2e7**, and the
statistic varies by horizon again.

**The gate was discarding its own worst evidence.** `fence_reduce` filtered non-finite values
(`px_max[np.isfinite(px_max)]`) before every density statistic, so a pixel with a zero-width
segment — infinite implied density, the strongest possible needle — left no trace. On b1_s42
that was **91.5%** of pixels. They now count, and `px_degenerate_frac` sits beside
`max_density` so the latter is never read alone.

**The support boundary was being counted as a fence.** HM cannot leave [0, 1] and 40% of Africa
sits in `[0, 0.01)`, so the lower tail clamps flat at zero — correct behaviour. Measured: of
pixels whose two lowest levels were identical, **100%** had `Q = 0.0` exactly there, and the
clamp contributed **0.3%** of the needle mass. It is now excluded from the fence and reported
as `px_clamp_frac` / `gap_frac_clamp`, ranked "none": a column that is a large constant on
every arm cannot discriminate between them.

### The fence is real, it is in the core, and it is severe

With the instrument fixed, the same weights re-exported at float32 give needle mass p50 0.334
against int16's 0.344 and **p90 identical to four decimals**. Nothing about it was storage.
Attributing the needle mass by u-region:

| u region | share of needle mass | probability it holds |
|---|---|---|
| clamp, u < 0.01 | 0.3% | 0.010 |
| lower tail 0.01–0.1 | 0.3% | 0.097 |
| **core 0.1–0.6** | **73.3%** | 0.509 |
| upper body 0.6–0.9 | 24.3% | 0.297 |
| upper tail u ≥ 0.9 | 1.9% | 0.087 |

With the clamp separated out, the genuine degeneracy is confined to short lead times:
`px_degenerate_frac` reads 0.613 / 0.174 / **0.000** / 0.000 at h=5/10/15/20 while
`px_clamp_frac` sits at 0.49-0.59 throughout. **Beyond h=10 there is no zero-width segment
anywhere that is not HM resting on its floor.** The same split shows in coverage from a wholly
independent instrument -- `cov50` is 0.383 / 0.374 at h=5/10 against 0.525 / 0.504 at h=15/20 --
so the short-horizon core collapse is two measurements agreeing, not one metric's artefact.

**74.8% of core segments are needles**, and the median core segment is **7.93e-6 HM** wide —
87x narrower than the needle threshold and 87x narrower than the observation noise's own sigma
of 6.9e-4. Meanwhile `width95` is healthy (median 0.020, p1 4.6e-3). Half the probability mass
is packed into a few microunits of HM inside a wide interval. That is the phase's own sentence,
measured: *the head is being asked to spend five bins on a region where nothing varies.*

> **`MIN_SCALE` is not the cause, though the mismatch section 4.1 flags is real.** The comment
> at `quantile_spline.py:130` justifies 3e-5 while the constant is 1e-6 — which is 1.53e-7 in
> HM, 4,500x below the noise. Measured on b1_s42: **0.0%** of pixels are at that floor, below
> 3e-5, or even below the noise sigma. Resolve it before the `--free_scale` port as section 4.1
> says; it explains nothing here.

> **`max_density` is a lower bound, not a value, and is reported rather than ranked.** The
> int16 export pinned it at 1531.8; float32 moved the ceiling to ~1e7 without removing it.
> Measured on b1_s42's float32 raster: **47.9% of pixels have their sharpest segment within
> two float32 ULPs**, median 3.0 ULP. A `max_density_p99` that is byte-identical across four
> window-years (1.25e+07 at h=5 in every one) is the tell. The fence is therefore ranked on
> the statistics that are **bounded in [0, 1] and cannot saturate** -- `needle_mass_median` /
> `_p90`, `px_degenerate_frac`, `over_f_max_frac` -- with `max_density` printed beside them
> for scale. `max_density_p50` is the better-behaved of the two percentiles (8.1e3-1.9e4
> across window-years at h=5, against a p99 that does not move at all).

> **Do not compare any of this to e1's 4.5% / 677.** Different model, different fold mask
> (`fold_mask_b4`'s 512 px blocks against a 128 px checkerboard), and e1's number came from a
> side script whose density statistic had the same drop-the-degenerate-pixels defect. b1's
> fence IS the floor; there is no regression to explain.

**Costs, measured rather than projected.** Train 150 epochs + predict, two folds in parallel:
63.0 min. Stitch: 11.7 min. Score: ~57 min. **~2.2 h per experiment**, so the fourteen-arm
programme is ~31 h, not the ~10.75 h section 4.1 projects. float32 storage is 2.4x int16
(2.58 GB against 1.06 GB per stitched window-year; 66 GB per experiment).
>
> **And two costs the code's own estimates understated, both found by running it.** Prediction
> accumulators peak at **40.7 GB for one Africa fold** where `plan_row_bands` estimated ~22 GB
> — the restriction mask saves less than it looks like it saves, because `fold_mask_b4`'s
> 512 px blocks touch nearly every page of a full-region accumulator while keeping a fifth of
> the pixels. The scorer peaks at **59.1 GB on one fold**: `read_qf` pulls the whole 64-band
> raster (16.1 GB on Africa) and the ref-grid gate builds `[256, n_px]` and `[255, n_px]` on
> top of it — the float64 density array alone 13.6 GB. b1 runs *two* folds on a 125 GB box.
> `conv_spline_base.sh` passes `--predict_row_chunk 2048` for the accumulators; the gate is
> fixed instead by blocking it over pixels where the temporary is built
> (`qf_diagnostics.fence_per_pixel`), measured **59.1 → 32.1 GB with all 88 summary metrics
> bit-identical and no time cost**. `--row_chunk` would have bought the same memory for ~36%
> more wall clock, so it defaults off and stays available.
>
> Timings, measured rather than inherited: prediction is **~26 min per fold** (9:32 / 7:10 /
> 5:23 / 3:39 for the four windows, which shrink with the horizon count), stitching ~7 min, and
> **scoring ~33 min** — not the ~5 min the loop table in `CLAUDE.md` carried from an earlier
> phase. Training is on top of all of that.

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
> repeated across timesteps — and **not into the heads**.

Until now the context reached `last_hidden` only, so the ConvLSTM never saw "can change happen
here at all" and could not combine it with its own spatial and temporal features; it could only
have the answer applied to its output. Nothing ever justified that. The precompute-on-the-full-
raster requirement (radii ≥ 30 px saturate inside a 128 px chip) is about where the covariate
is *derived*, not where it is *consumed*.

**This is not a flag.** `--trunk_context`, `--central_context` and `--quantile_context` were
removed on 2026-09-11: distance to past change and the neighbourhood HM summaries are ordinary
covariates, handled like elevation and climate, and the wiring is part of the model. There is
nothing to pass, nothing for `--extra_train_args` to override back, and no configuration in
which a head receives the covariate a second time. What is still verified is that it *arrived*:
the run's log reports the channel count the trunk was built with, read off the module rather
than off an argument, and `verify_context_wiring` refuses a run where that count is zero,
disagrees with what `--context_radii` / `--hm_context_stats` imply, or names a head as a
consumer. A checkpoint trained before the change has a narrower trunk conv and is warm-started:
its trained weights are copied in and the context channels start at zero.

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

### The figures — built 2026-09-14, `src/qf_plots.py`

Every experiment writes both, beside its numbers, without being asked:

1. **`densities_<label>.png`** — the implied density of nine per-pixel forecasts on a 3x3
   grid. The x axis is *change* (`Q(u) - HM_t0`), because against absolute HM the whole
   persistence core collapses into one pixel of the plot; the y axis is logarithmic, because
   the core sits at several hundred and the tails well under one. Drawn as a **step** function,
   which is what the distribution is — the raster is a piecewise-linear quantile function, so
   its density is piecewise constant, and smoothing it would hide exactly the discontinuities
   the gate counts. `f_max = 578`, the observed change and zero are marked on every panel.
2. **`pit_<label>.png`** — the PIT histogram in twenty bins, one panel per horizon, plotted as
   a density so the calibrated reference is the line at 1.0 at every horizon whatever the pixel
   count. Same reason `pit_rms_se` is reported rather than raw RMS.

Both are embedded, base64, into **`scorecard_<label>.html`** beside the pooled-by-horizon and
gate tables.

**The nine pixels are drawn by a seeded walk over the fold mask**, not over the run's own
finite pixels, so two arms scored on the same folds draw the same nine and their panels stack.
A pixel an arm failed to predict is skipped and the walk continues — the only way the choice
can differ between arms, and worth seeing when it does. Both figures are read off the exported
raster, the same object the gates are computed from: a figure drawn from a different object
than the metric would be the fourth way this project has found to make a measurement disagree
with itself.
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
| **E0a** | `--spline_cumulative_width False --mu_mse_weight 0.0` | The incumbent head, otherwise untouched, with the horizon accumulation removed so each horizon's 95% width comes from its own channel. **Already implemented and tested**; no new code. | 29 params/h |
| **E1a** | `--head_family isqf --isqf_tails True --isqf_space logit\|neglog` | Learned exponential tail rates β_L, β_R on unbounded transformed support. ISQF's actual contribution, which E1 drops. CRPS still evaluated in HM. **The tails' domain is unsettled** — see §4.1. | 18 params/h |
| **E1b** | `--head_family isqf --free_scale True --mu_mse_weight 0.0` | E1 minus the cumulative-in-horizon scale: increments carry the width directly, no renormalisation to unit 95% span. Park et al.'s own arrangement. | 15 params/h |
| **E1c** | E1a + E1b together, `--mu_mse_weight 0.0` | Park et al. (2022) as published, adapted only where the ConvLSTM forces it. **Runs unconditionally** — it is the fourth cell of a 2x2 and completes it. | 17 params/h |
| **E2a** | `--head_family pwl --free_scale True --mu_mse_weight 0.0` | E2 with the anchor/scale factorisation removed: softmax heights replaced by unnormalised positive increments, so the 95% width is emergent rather than injected. | 15 params/h |

E0 and E2–E5 are the issue note's staged plan, ordered by how much has to be touched. E1 and E2
are the two literature-derived heads. The last five rows are a group rather than a continuation
of that ordering — they are §4.1 below, and they are listed after E5 because each is a delta
from an arm above it rather than from b1 directly.

**E0's "no retrain" is not what the slate does.** `--u_grid_spacing` is read only at
prediction time (`train_lightning.py:2289`, inside the `predict_qf_levels` export path) and has
no effect on training, so with the seed fixed E0's retrain reproduces `b1_s42`'s checkpoint
exactly. `run_conv_spline_slate.sh` says as much in its own comment — "export-only and needs no
retraining" — but still routes E0 through the full train path, ~50 min for a checkpoint that
already exists. Running it predict-only from b1's checkpoints (`--max_epochs 0`) makes E0 ~5 min.
The result is the same either way; only the cost differs.

**Param counts assume `default14` (15 knots, 14 bins).** `--free_scale` drops the `scale`
channel, so `pwl`/`isqf` go 16 → 15 and the incumbent would go 29 → 28; E1a's two tail rates
are `+2`. Simplicity is a scoring criterion (`CLAUDE.md`), and these are the numbers reported
beside the metrics — E1b and E2a are the *cheapest* heads on the slate, not merely different
ones.

**Flag status.** `--spline_cumulative_width` (E0a) exists. `--free_scale`, `--isqf_tails` and
`--isqf_space` were **implemented 2026-09-15** and are pinned by twelve tests in
`tests/test_conv_spline_flags.py`; `scripts/run_conv_spline_scale_arms.sh` runs the six arms in
the order below, with a test asserting that order. **Two things this section specified turned
out to be wrong, and neither was visible without running it:**

> **1. `--free_scale` as specified starts the ladder 330x too wide.** The spec is "unnormalised
> positive increments (`|.| + tol`) in place of the softmax heights". But every head feeds the
> shape channels *through a softmax*, so only their relative values have ever mattered and
> their absolute magnitude is arbitrary -- measured at init, `|raw|` averages **2.68**.
> `--free_scale` makes that magnitude *be* the 95% width: 8 bins x 2.68 = 3.3 HM against an
> intended 0.010. Every knot then saturates the [0, 1] clamp, and the two arms fail in
> opposite directions -- `pwl` spans the entire range, `isqf` piles its whole ladder onto the
> upper bound and reports a width of **exactly zero**. Both would have measured whether the
> optimiser can escape a hopeless initialisation. The fix gives one increment a defined size
> (`initial_width_normalized / n_95_bins`, `quantile_pwl.free_increments`): a **unit**, not a
> normalisation -- no pixel's width is tied to another's and nothing is rescaled to a target
> span, so the width stays emergent. Measured after: every arm starts at 0.020-0.028 HM
> against b1's own 0.0184.
>
> **2. The tails would have been a straight line to a distant point.** Writing the learned
> tail into the endpoint knot alone leaves everything between u = 0.999 and u = 1 -- which is
> exactly where the 64-level raster samples the far tail -- to linear interpolation.
> `ISQFQuantile.ppf` now evaluates the exponential wherever it owns the domain. Verified
> curved: `Q(0.9995)` = 0.0799 against a straight-line 0.1576. The numbers move either way,
> so this omission would have been invisible and E1a would have been judged on a mechanism it
> was not running.

**Decisions taken before coding, as this section demands.** The tails **replace** the outer
bins and anchor at the 0.001 / 0.999 knots (verified: the spline interior is bit-identical to
E1, so E1a remains the control on the width row). E1b stays paper-faithful with `q0` at
`Q(0.0)` while E2a anchors at `Q(0.5)` -- verified to produce identical widths, confirming they
are one ladder differing only in where persistence enters. `MIN_SCALE` is ported to a span
floor at the **3e-5** this doc resolved, not the mismatched 1e-6, and shared between both
families so E1b and E2a cannot drift onto different scales.

**`--isqf_space` is an experiment, not a setting.** b1's far-tail miss is two-sided and
near-symmetric -- `pit_lt_0001 / pit_gt_0999` = 1.017 / 1.254 / 0.971 across the three floor
seeds -- which `logit` can address and `neglog`, unbounded above only, structurally cannot. Both
get an arm.

**E1a would have been unreadable.** Its mechanism acts only beyond u = 0.001 / 0.999, and no
ranked metric measured that: the arm could have worked perfectly and scored as a null, which is
the inert-flag failure one level up. `pit_gt_0999`, `pit_lt_0001` and a pooled
`far_tail_excess` are now ranked. Their floor band is **44%** and they print `WEAK`, so E1a
must roughly halve the excess (2.5x -> ~1.3x) to be readable -- reported rather than hidden. `--free_scale`
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

#### The four free-scale arms drop the auxiliary MSE

**E0a, E1b, E1c and E2a run on CRPS alone — `--mu_mse_weight 0.0`.** E1a keeps it, because E1a
keeps the accumulation and is the control on the width row; b1 and E0–E5 keep it too.

The reason is that the term is not a neutral bystander to *this* question. The objective is

```
total = mu_mse_weight * MSE(E[Q], y) + CRPS
```

and an MSE term pins the distribution's first moment to the conditional mean. Every arm in this
group exists to ask where the width comes from once it is no longer injected — and a freed
scale that is simultaneously held by a squared-error term on its own mean is not free. Worse, it
is not *separably* unfree: a null would not distinguish "the factorisation was earning its keep"
from "the MSE term supplied what the factorisation used to". Since the arms were defined by a
subtraction, the subtraction has to be complete.

**This weight was an argparse default until now, on every arm including b1.** `BASE_ARGS` named
`--ssim_weight`, `--laplacian_weight` and `--histogram_weight` and not this one, and
`PRODUCTION_HPARAMS` does not carry it either, so a second objective nobody chose rode along at
1.0 — rule 16 in its exact form. It is now named in `BASE_ARGS` and read back by
`verify_loss_weights`.

> **The banner was printing a literal.** `train_lightning.py` printed `MSE weight: 1.0 (fixed)`
> regardless of the flag, which is the line `verify_loss_weights` greps and the only place a log
> reader could check. An arm defined by *not* having this term would have logged as though it
> did, and the previous phase already ran one (`d6`, `--mu_mse_weight 0.0`) against that banner.
> Fixed to print `args.mu_mse_weight`, and pinned by
> `tests/test_conv_spline_paths.py::test_banner_reports_the_flag_rather_than_a_literal` — without
> the fingerprint moving with the run, every other check on this weight passes vacuously.

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

**Cost, since none of these is optional.** ~55 min per arm — ~50 min to train two folds in
parallel and stitch, plus the ~5 min scoring step the driver scripts run separately — so ~4.6 h
for the five, on top of the six-arm E0–E5 slate (5.5 h) and b1's three seeds (2.75 h). Arms
cannot overlap: `GPUS=0,1` with `FOLDS=1,2` means one arm already occupies both cards, so the
total is a sum rather than a max. `run_conv_spline_slate.sh` currently knows E0–E5 only; §4.1
needs adding to it, and it should refuse to start without b1's floor for the same reason the
existing slate does.

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
