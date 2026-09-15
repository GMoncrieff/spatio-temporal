# Working notes for this repository

Spatiotemporal Human Modification (HM) forecasting. **One system lives here now.**

Active branch: **`conv-spline`**. Started 2026-09-07.

## What this repository is, as of this phase

A distributional ConvLSTM that emits a full per-pixel quantile function `Q_h(u|x)` at
+5/+10/+15/+20 yr, and **that function is the entire product**. No ensemble, no post-hoc
chain, no conformal scaling, no width factors, no empirical marginal reshaping, no
horizon-monotonicity pass.

The four-stage post-hoc chain, ensemble generation, icechunk stores, the copula, the
validators and every script and test that served them were **deleted** on this branch, not
left dormant: a dormant path is one nothing runs and nothing checks. Two pieces survived as
their own modules because they are properties of the *data* rather than of the chain —
`src/strata.py` (the binning definitions) and `src/stitch.py` (the fold mosaic).

Prior phases are recorded in `docs/background/` and `docs/dist_model_phase.md`; those
documents describe systems that no longer exist here. Read them for method, not for state.

**Read `docs/conv_spline_phase.md` first.** It carries the measured defects, the experiment
menu, and why each experiment exists.

**The scorecard has pictures now.** `score_distributional_model.py` writes
`densities_<label>.png` (nine per-pixel implied densities on a 3x3 grid, log axis, with
`f_max`, the observed change and zero marked), `pit_<label>.png` (a 20-bin PIT histogram per
horizon) and `scorecard_<label>.html`, which embeds both beside the two metric tables. The nine
pixels come from a seeded walk over the **fold mask**, not over the run's own finite pixels, so
every arm draws the same nine and the panels stack. `src/qf_plots.py`.

## Where the phase is

**No experiment has been run.** The data *is* on this machine now, and as of 2026-09-14 the
code has met it: a one-epoch, one-fold smoke pass over the full Africa path. Three defects had
to be fixed before `b1` could run at all — the prediction writer could not decode `pwl`/`isqf`,
both gates raised `KeyError` on every real scoring run, and `b1` was scripted on the wrong
covariate. All three are written up in `docs/conv_spline_phase.md`.

1. **Step 1, not yet run** — establish `b1`. Three seeds, on Africa.
   `./scripts/run_conv_spline_baseline.sh`
2. **Then** the E0–E5 slate. `./scripts/run_conv_spline_slate.sh`
3. **Then §4.1's scale arms** — `./scripts/run_conv_spline_scale_arms.sh`. Written 2026-09-15:
   `--free_scale`, `--isqf_tails` and `--isqf_space` exist and are tested, and the runner holds
   the six arms in the doc's order (E0a first and alone, both `--isqf_space` arms, E1b paired
   with E2a, E1c last and unconditional) with a test asserting it. **Two things §4.1 specified
   were wrong and only running it showed that** — see the phase doc's flag-status block.
4. **E6–E8 are chosen from what E0–E5 and §4.1 measure**, not written in advance.

Nothing on this branch is comparable to a `dist-convlstm` number: the trunk change alone makes
`b1` a different model, and the scorecard carries two gates `e1` never had.

**The neighbourhood context is a covariate, not a setting.** Distance to past change and the
neighbourhood HM summaries go into the **trunk**, repeated across timesteps beside elevation and
climate, and into **no head**. There is nothing to pass: `--trunk_context`, `--central_context`
and `--quantile_context` were removed, and the wiring is part of the model. `e1` fed the heads
and never the trunk; everything from `b1` onward feeds the trunk and never the heads. The run's
log reports the channel count the trunk was actually built with — read off the module, not off a
flag — and `verify_context_wiring` refuses a run where that is zero, disagrees with what
`--context_radii` / `--hm_context_stats` imply, or says a head received it.

**Where it goes is not a setting; which covariate it is still is, and it is named.** `b1` is
"`e1` plus exactly one change", and `e1` is `--context_radii 3,30,100 --hm_context_radii
3,30,100 --hm_context_stats mean,max` — **twelve** trunk channels. `BASE_ARGS` named none of
them until 2026-09-14, so `b1` was scripted on argparse defaults: `--context_radii
1,3,10,30,100`, no HM summaries, **eight** channels — `e1`'s dropped fine radii back and the
neighbourhood-HM covariate simply absent. `verify_context_wiring` could not see it: it compared
the module against the same defaults and read 8 == 8. `conv_spline_base.sh` now names the flags
and carries `EXPECT_CTX_CHANNELS=12`, which the wiring check compares the trunk against —
`tests/test_conv_spline_paths.py` proves that line fires on the 8- and 9-channel controls.

## What b1 measures — first run, 2026-09-14

`b1_s42` on Africa, folds 1+2, `fold_mask_b4`, 150 epochs. `crps_skill` 0.202 / 0.261 / 0.285 /
0.289 and central `skill` 0.120 / 0.209 / 0.247 / 0.253 at +5/10/15/20 yr, both against
persistence. **The pooled skill is carried entirely by pixels near past change**: by distance
band at h=20 it runs +0.395 / +0.293 / +0.223 / +0.137 / **−0.106** / **−1.510**, so beyond
30 px the model loses to persistence and in the far field it loses badly.

**The fence is real, it is in the core, and it is severe.** 73.3% of the needle mass comes from
`u ∈ [0.1, 0.6]` and 0.3% from the clamp; 74.8% of core segments are needles; the median core
segment is **7.93e-6 HM**, 87x narrower than the observation noise's sigma of 6.9e-4 — while
`width95` is a healthy 0.020. `MIN_SCALE` is **not** the cause (0.0% of pixels near it). None of
this is comparable to e1's numbers, for the reasons in `docs/conv_spline_phase.md` §1a.

## The two defects the phase exists to fix

Both measured, both in `docs/conv_spline_phase.md` with the full tables.

**The picket fence.** Every sampled pixel shows 2–6 needle-thin spikes beside the persistence
value. Needle mass is 4.5% at the median pixel and 21% at the 90th percentile; max implied
density reaches 677 where the observation noise supports at most 578. The tell that this is a
monotonicity artefact rather than a real spike: 4.5% of gaps are within three quantisation
steps while only 0.7% are exactly zero. Levels collapsing *toward* each other without
collapsing *onto* each other is what a strictly-positive-derivative constraint does when the
target wants a flat segment.

**PIT structure.** Weighted mean 0.5537 against 0.50; only 37.7% of mass left of centre;
stable spikes at ~0.25 / 0.51 / 0.73 that survive a resolution test. About 15% of cells are
displaced one bin left, across zero, into a region where change does not occur.

Underneath both: **CRPS in raw HM units is dominated by the tail.** The persistence core spans
~0.0036 against a range above 1.2, so placing the core badly costs almost nothing — and 68% of
land is in that core. Nothing ever penalised the fence. More knots alone would only give the
head more room to do the same thing.

Two things that are *not* the fix, both measurably: a point mass at exactly zero (exact zero
occurs in 0.00% of land and the core sits at +0.00031), and anything that forces unimodality
(the target is trimodal at +0.00015 / +0.0029 / +0.017).

## Questions this phase reopens

The previous CLAUDE.md closed these. They are open again, and several are experiments.

- **Is C1 smoothness load-bearing?** The rational-quadratic spline has no closed-form CRPS and
  costs 168 evaluations per horizon; a piecewise-linear head has one. Never tested (E2).
- **Are the knots in the right places?** The grid is symmetric and the target is not (E5).
- **Are 14 bins the right number?** `lean9` and `skew11` ask the strong form.
- **Should the trunk see the neighbourhood context?** Now settled as yes — `b1`
- **Should the objective be scored in raw HM at all?** (E3.)
- **Is the anchor/scale factorisation right**, or the free-first-knot arrangement ISQF uses (E1)?
  E1b and E2a ask the strong form, and are expected to converge on nearly the same head.
- **Is the horizon-cumulative scale load-bearing, or merely tidy?** The 95% width cannot shrink
  with lead time by construction, so the question has never been measurable. E0a is the
  one-thing-changed read, on the incumbent, using the `--spline_cumulative_width False` flag that
  already exists and is already tested. `--free_scale` is **not** a second spelling of it: it is
  a strict superset that drops the anchor/scale factorisation too, so an arm carrying it cannot
  separate "was the factorisation earning its keep" from "was horizon monotonicity binding".
  That is why E0a runs first and alone.
- **Does the far field need a learned tail rate** rather than more knots? Now E1a: trainable
  β_L, β_R on unbounded transformed support, with CRPS still scored in HM. E1c is E1a + E1b, the
  paper as published, and **runs unconditionally**: E1/E1a/E1b/E1c are a complete 2×2 over
  {tails, free scale}, and gating the fourth cell on movement in the other two drops it exactly
  when a joint effect is the only remaining explanation.
- **`--crps_nodes 6` may be thin under extreme shapes.** Measured 2026-09-07: at `shape_mag=3`
  the split quadrature's gradient error rises to 4.2e-2, *worse* than a naive uniform rule at
  the same budget. Cheap to check, never checked.

## Environment and scale

- Conda env **`spatio-temporal-dl`**, not base:
  `/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python`.
- Two GPUs (RTX A5000, 24 GB each). Track experiments on **W&B**.
- **Iterate on Africa** (63.1 Mpx), `config/region_africa.geojson`. Not southern Africa: its
  far-field band contains **zero pixels** (not a zero rate — no pixels), and its `[0,0.01)` HM
  stratum is 6% of the region against 40% of Africa. A stratified finding measured there is
  provisional, and this already cost the project a whole phase. `guard_region` in
  `conv_spline_base.sh` enforces it.
- **The globe (17111 x 40000, 184.6M valid px) is for promotion only, on instruction.**
  `ALLOW_GLOBAL=1` is required and should never be set casually. Projected from the Africa run:
  ~121 min prediction per fold, ~10 h for k=5.
- Artifacts live under `data/conv_spline/`. Large ones belong on the HDD
  (`/mnt/hdd1/spatio-temporal/data`), and output roots are passed as the HDD path spelled out
  rather than through a symlink. **The reason this rule used to give no longer exists**: the
  `shutil.rmtree` that refuses on a symbolic link lived in `migrate_zarr_to_icechunk.py`,
  which commit `60c6a9e` deleted with the rest of the ensemble layer — `git grep rmtree` finds
  nothing on this branch, and `run_hindcast_folds.py` only does `mkdir(exist_ok=True)` and
  `Path.unlink()`. Keep the practice (there is then no resolution to reason about when a
  path is cleared or moved), drop the dead justification. `EXP_ROOT` therefore defaults to
  `/mnt/hdd1/spatio-temporal/data/conv_spline/exp` spelled out, while `LOG_DIR` and `SCORE_DIR`
  stay on the SSD under `data/conv_spline/` — they are small and every verifier greps them.
- Monitor free space on the HDD and SSD. A stitched Africa window-year is 64 int16 bands over
  63.1 Mpx and there are **ten** of them per experiment, plus one set per fold under
  `--keep_fold_rasters`. Deflate-compressed, so measure rather than project from the raw size.
- When long running tasks are underway e.g. training, scoring, prediction. periodically check progress and health  (every 30 mins). 

## The loops

| what | command | cost |
|---|---|---|
| one experiment (train held-out folds, predict Africa, stitch) | `BASE_ARGS=... ./scripts/run_central_experiment.sh <name> 0,1 1,2 "<flags>"` | **prediction alone is ~26 min/fold** (measured 2026-09-14: 9:32 + 7:10 + 5:23 + 3:39 for the four windows), folds in parallel, + stitch ~7 min. Training is on top of that. |
| **establish the baseline and its floor** (3 seeds) | `./scripts/run_conv_spline_baseline.sh` | 3x the above |
| **the experiment slate** (refuses without a floor) | `./scripts/run_conv_spline_slate.sh` | 6x the above |
| **score a model**, no ensemble, both gates included, two figures and an HTML scorecard | `scripts/score_distributional_model.py --stitched_dir … --folds 1,2` | **~31 min on Africa**, not the ~5 min this table claimed — measured 2026-09-14 over ten window-years, one fold |
| **rank against the measured floor** | `scripts/compare_conv_spline_runs.py --floor_prefix b1` | seconds |
| **predict-only from frozen checkpoints** | `run_hindcast_folds.py --fold_checkpoints … --max_epochs 0` | ~121 min/fold globally |
| build the neighbourhood-HM covariate (once) | `scripts/prepare_hm_context.py` | 2:45 per base year |

## Rules learned the hard way

Kept only where they still apply. Numbering is fresh; the old file's numbers are gone.

1. **Verify the measurement before believing the finding.** Measurement bugs have outnumbered
   model bugs in this project throughout. Byte-identical numbers across supposedly different
   configurations is the tell — but check the artifacts are genuinely distinct before
   concluding it, since two immaterial fixes look identical too.
2. **Bugs caught before a single run, each of which would have read as something else.** A
   closed-form CRPS returning *negative* values; a knot preset silently missing the 0.5 and
   0.975 gate levels; and `pwl`/`isqf` taking the triple-head two-pass code path because
   `head_family == 'spline'` was spelled out in three separate places. The general form:
   **a predicate written more than once is a predicate that will disagree with itself.**
   A fourth, found 2026-09-14 and the same shape: the prediction writer called
   `splines_from_output` — the *rational-quadratic* decoder, 29 channels per horizon — for
   every family, so `pwl` and `isqf` (16) raised `ValueError: expected 116 spline channels
   after 12, got 64` at the first prediction batch. E1, E2, E4 and all four §4.1 free-scale
   arms would each have trained for fifty minutes and written no raster. There is now one
   decoder, `SpatioTemporalPredictor._decode`, and the loss and the writer both go through it
   (`tests/test_conv_spline_flags.py`, with the old call pinned as the control).
3. **The second implementation earns its keep, fourth time now.** The rational-quadratic closed
   form was wrong by a *relative* error of ~1; the piecewise-linear one returned negatives.
   Both were caught only by a dense numerical reference sharing no quadrature code. Keep one.
4. **A metric failure and a model failure are indistinguishable from the outside.** Put the
   check where the cause is one line away: `validate_knots` refuses a bad grid at definition
   time, the qf reader refuses a non-increasing u-grid rather than passing NaN downstream.
5. **Prove a check fires on a control, or it checks nothing.** Before `conv_spline_base.sh` was
   allowed to grep for the context fingerprint, that fingerprint was verified to discriminate.
   **ON is not the same as ONLY**: the first version checked that the trunk context was on, and
   passed `b1`'s own misconfiguration, because `e1`'s head consumers were on too. The check now
   reads the trunk's channel count off the module and refuses a head consumer. **A check can
   also fail *open*, which is silent.**
   `verify_loss_weights` passed on a log file that did not exist: every expected weight in this
   phase is 0.0, `grep` on a missing file returns nothing, and awk coerces `""` to 0, so all
   three comparisons read 0==0 and the check reported success having read nothing. When a
   check's expected value is a default (0, empty, absent), test it against *no input* as well
   as against a wrong one — `tests/test_conv_spline_paths.py`.
6. **Score against persistence, not zero.** The median 20-year HM change is 0.0001.
7. **Judge sweeps on per-row values, never a total pass count.**
8. **A metric pinned and insensitive to its own knob is under-powered, not mis-tuned** — and
   the *inverse* is more dangerous. A baseline band of width ~0 means the floor is unmeasured,
   not that the metric is infinitely sensitive; every variant then clears "margin exceeds the
   band's own width". `compare_conv_spline_runs.py` marks those rows `DEGEN`.
9. **Three replicates can all land in one mode**, and the band you measure is then far too
   narrow. Three baseline seeds once gave a spread of 0.86 where three replicates of a
   *variant* of the same configuration gave 4.3. If a floor's replicates agree suspiciously
   well, suspect the sample before believing the band.
10. **Two metrics that look independent can be one axis.** `tail_reach20` and `pit_ks5`
    correlate at r = 0.691 across 13 runs; three separate knobs were all landing at different
    points on one axis. Correlate the metrics across runs before reading N knobs as N findings.
11. **A binning convention is a measurement.** `right=True` vs `right=False` on the distance
    bands changed the 0–1 px band's observed `P(Δ>0.05)` from 0.177 to 0.222. One definition
    now: `src.strata.distance_band`. Use it; do not re-derive the edges. The same rule caught
    `OBS_MAG_BINS` living in two files with two conventions during this phase's setup.
12. **A diagnostic can encode an assumption it never states**, and announces itself as a result
    that *could not be true* rather than by inspection. When a number is impossible, suspect
    the metric.
13. **A gate written for one failure mode keeps passing after the mode inverts.** Re-read any
    one-sided gate whenever the thing it bounds changes sign, and prefer a ratio to observed,
    which cannot be satisfied from either direction alone.
14. **A pooled average cannot see a defect that lives in a thousandth of the pixels.** Judge a
    change on the pooled score AND a stratified check.
15. **Hold something out before believing a class-conditional fit.** A correction fitted to the
    metric it is scored on will look good and not generalise.
16. **"No extra flags" is not the production architecture — it is argparse defaults**, and
    **"the flags I passed" is not "the flags that took effect."** `run_hindcast_folds.py`
    injects the frozen product's loss weights and `--extra_train_args` is appended last, so
    anything `BASE_ARGS` does not name is silently inherited. `conv_spline_base.sh` names all
    four loss weights — including `--mu_mse_weight`, which was an argparse default on every arm
    until 2026-09-11 — and reads them back out of the run's own log. **Hardwiring something
    into the model exempts only what was hardwired.**
    The context's *destination* is no longer a flag, so nothing appended later can turn it off —
    but its *content* is still `--context_radii` / `--hm_context_stats`, and `BASE_ARGS` named
    neither until 2026-09-14. `b1` was therefore scripted on eight default channels while the
    phase doc called it `e1` (twelve) plus one change. `EXPECT_CTX_CHANNELS` now names the count
    and `verify_context_wiring` compares the trunk against it.
17. **Any long-range covariate must be precomputed on the full raster**, never derived inside a
    128 px chip (radii ≥ 30 px saturate against the chip boundary). This is about where the
    covariate is *derived*, not where it is *consumed* — which is why b1 can feed the same
    precomputed tensor to the trunk.
18. **The stitched hindcast is a quilt of five fold models, and the seam is real.** Anything
    scored stays on `--stitch_mode holdout`. **`mean` through `run_hindcast_folds.py` is not a
    five-fold average**: `--predict_restrict_mask` is passed unconditionally, so each fold
    predicts only tiles overlapping its own territory and a block's interior has exactly one
    prediction to average. A real mean needs a prediction pass without the restriction. The
    forward product is neither — it is a separate `--train_all_splits` model
    (`run_global_dist_forecast.sh`), which is why the configuration is validated by k-fold.
19. **The fold tile is one correlation length, so held-out skill is optimistic.** The residual
    field's practical range on Africa is 99–166 px against a 128 px fold tile. `fold_mask_b4`
    (512 px blocks) is what this phase uses.
20. **Most fold disagreement is optimisation noise, not data** — 85% of the variance at h=20.
    Central-field seed noise is ~0.0003, so a large move in the central field cannot be
    optimisation noise; the *upper half-width* is where the seed spread lives.
21. **`ModelCheckpoint` monitors what you tell it**, and a quantile-only change still selects a
    different epoch through a shared monitor. This phase uses `--checkpoint_monitor val_crps`
    so a run setting `--mu_mse_weight 0` is not selecting on a different quantity from the rest
    of the slate. E0a, E1b, E1c and E2a are those runs: the free-scale arms score on CRPS
    alone, because an MSE term pinning `E[Q]` is not a bystander to an experiment about where
    the width comes from.
22. **A regional working set can hide a quadratic.** Something `(M, H, W)` and unremarkable on
    1.86 Mpx is dead on 63.1 Mpx. Check peak memory against the region you will actually run.
    **Measured on Africa, 2026-09-14, both worse than the estimates in the code:** prediction
    accumulators peak at **40.7 GB per fold** where `plan_row_bands` estimated ~22 GB, and the
    scorer peaked at **59.1 GB on one fold** — `read_qf` pulls the whole 64-band raster
    (16.1 GB) and the ref-grid gate built `[256, n_px]` and `[255, n_px]` on top of it, the
    float64 density array alone 13.6 GB. `conv_spline_base.sh` passes
    `--predict_row_chunk 2048` for the first; the second is fixed by blocking the gate over
    pixels inside `qf_diagnostics.fence_per_pixel` — **59.1 → 32.1 GB, 31 min against 33, all
    88 summary metrics bit-identical** on the same rasters. **Bound the temporary where it is
    built, not by shrinking the caller's slice**: `--row_chunk 512` bought the same memory and
    cost ~36% wall clock, four hours across this phase's fourteen arms, so it now defaults off
    (`SCORE_ROW_CHUNK=0`) and stays available. **A restriction mask is not a memory saving when
    the kept pixels are scattered** — `fold_mask_b4`'s 512 px blocks touch nearly every 4 KiB
    page of a full-region accumulator while keeping a fifth of the pixels.
23. **Never edit a shell script while it is running.** Bash reads a script incrementally by byte
    offset, so inserting lines mid-run makes it resume at a stale offset and execute a
    fragment. Copy to a new name and launch that. Likewise `pkill -f <pattern>` matches the
    shell running it — use `ps -eo pid,cmd | grep "patt[e]rn"` and kill by pid.
24. **A leak shows up as a gradient, not a level.** Removing a leak must make error rise *with*
    distance from the fold's own trained-on data; a flat profile that is simply worse
    everywhere is a training signature.
25. **A metric is only a metric while its instrument can resolve the thing.** `max_density_p99`
    read **1531.8 at all four horizons** on b1_s42 — exactly `dp_max / one int16 quantum`. An
    identical value at four horizons is a ceiling, not a density. Three defects compounded and
    each had to be fixed before anything could be said about the model: the int16 export pinned
    the statistic (now `--predict_qf_dtype float32`, and the reader takes the scale off the
    raster rather than a constant); `fence_reduce` **dropped** every pixel with infinite density
    before computing the density gate, which on b1 was 91.5% of them; and the clamp at HM=0 — a
    physical boundary, 40% of Africa sits in `[0, 0.01)` — was counted as a fence. **Separate
    the boundary from the pathology and rank only the pathology**: a column that is a large
    constant on every arm cannot discriminate between them (`px_clamp_frac`, ranked "none").
    **The ceiling moves, it does not leave.** float32 lifted `max_density` from 1531.8 to ~1e7
    and 47.9% of pixels still sit within two float32 ULPs of their sharpest segment, so it is
    now reported rather than ranked. Rank the fence on what is bounded in [0, 1] and cannot
    saturate: `needle_mass`, `px_degenerate_frac`, `over_f_max_frac`.
26. **A parameterisation is only meaningful relative to what consumes it.** Every head feeds
    its shape channels through a *softmax*, so their absolute magnitude has never mattered --
    measured, `|raw|` averages 2.68 at init. `--free_scale` makes that magnitude *be* the
    width, and the spec as written (`|.| + tol`, unnormalised) therefore started the ladder
    **330x too wide**, saturating the clamp: pwl spanned all of [0, 1] and isqf reported a
    width of exactly zero. Both arms would have measured optimiser escape from a hopeless
    init. When you remove a normalisation, the quantity underneath does not keep its old
    meaning — give it a **unit** (a constant), which is not the same thing as giving it back
    a **normalisation** (a per-pixel rescale) and does not undo the experiment.
27. **A unit test that builds its own input tests the function, not the path.** `gate_stats`
    had a passing test that constructed `cell` by hand with the quantile function in it. The
    real caller does not: `load_row` streams the raster in bands and does `del cell["qf"]`
    before concatenating, so **every real scoring run raised `KeyError: 'qf'` on the pooled
    stratum** and the two gates the phase exists to measure never ran once. Nothing between the
    two was tested. Pair every unit test of a reduction with one end-to-end call on the path
    that feeds it, and prove the pair fires by reverting the fix
    (`tests/test_score_dist_row_chunk.py`).
28. **A banner that prints a literal is not a fingerprint.** `MSE weight: 1.0 (fixed)` was
    hardcoded while `--mu_mse_weight` was a live flag the previous phase had already ablated,
    so the one line a log reader checks could not distinguish the two runs — and the check that
    greps it passed either way. Print the argument, then verify it; a check on a constant
    string checks nothing (rule 5, and rule 2's predicate written twice).

## Conventions

- Flat argparse, no config framework. New flags are additive and default to today's behaviour;
  the boolean idiom is
  `type=lambda x: (str(x).lower()=='true'), nargs='?', const=True, default=…`.
- Tests are flat in `tests/`, pytest, `sys.path.insert(0, parent)` + absolute imports,
  synthetic tensors. **Measured 2026-09-15 on this machine: 451 passed / 7 failed.** Not the 15
  the phase began with — the fixture-needing ones pass now that the data is here. All 7 fail
  identically on `dist-convlstm` (checked in a worktree): stale legacy tests asserting a
  4-channel output from a model that emits 12, forwards that pass no `lonlat` to a trunk built
  with a location encoder, and a `target` key the loader no longer emits. Any other failure is
  yours. One flake to know about: `test_dataloader_targets_different` samples chips with
  `mode="random"` and no seed, so it fails perhaps one run in three — it passes 3/3 in
  isolation, and it is not a regression.
- **A new experiment needs a test that it changed something**, against a seeded control. A flag
  that is accepted, logged and inert reads downstream as "the experiment was a null" rather
  than "the experiment never ran" — see `tests/test_conv_spline_flags.py`.
- Scripts write to `data/conv_spline/exp/<name>/` so configurations never collide.
- Simplicity is a scoring criterion, not a preference. A simpler configuration that gives back
  a little is preferred to a more complex one that does not; the parameter count per horizon is
  reported alongside the metrics for that reason (spline 29, pwl/isqf 16, skew11 23,
  free-scale 15, E1a 18, E1c 17).
