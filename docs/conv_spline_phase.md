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
| **E1** | `--head_family isqf` | Park et al. (2022) adapted for bounded support: fixed outer levels, learned values via cumulative positive increments, linear spline between. Exponential tails **dropped** (unbounded; HM is not). Cumulative-in-horizon scale **kept**. | 16 params/h |
| **E2** | `--head_family pwl` | The incumbent's construction with linear pieces and **CRPS in closed form**. No quadrature, no nodes, nothing to checkpoint. Whether C1 was ever load-bearing has never been tested. | 16 params/h |
| **E3** | `--crps_z_weight 1.0` | A second CRPS term in `z = asinh(change/s)`, summed with the raw term. Attacks the incentive problem directly. | same head |
| **E4** | `--head_family pwl --spline_gap_floor True` | No segment may imply a density above 578. Makes a fence structurally impossible rather than discouraged. | +0 params |
| **E5** | `--spline_knots skew14` | The same 14 bins, re-placed for the measured right skew. `skew11` is the strong form: 11 bins, testing whether the body needed resolution at all. | −0 / −6 params |

E0 and E2–E5 are the issue note's staged plan, ordered by how much has to be touched. E1 and E2
are the two literature-derived heads.

**E6–E8 are deliberately absent.** They are chosen from what E0–E5 measure. Writing them now
would be guessing, and a slate whose last third is guesswork is how twenty-two experiments got
screened against a blind spot. Candidates already visible:

- **a learned tail rate**, ISQF's actual contribution, adapted for bounded support (in a logit
  or `1−HM` space). Aimed at the far field rather than at the fence.
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
