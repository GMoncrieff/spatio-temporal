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

## Where the phase is

**Nothing has been run.** The code is in place; the data is not on this machine.

1. **Step 1, not yet run** — establish `b1`: `e1` plus one accepted modification, the
   neighbourhood context injected into the **trunk** alongside elevation and climate rather
   than into the heads. Three seeds, on Africa. `./scripts/run_conv_spline_baseline.sh`
2. **Then** the E0–E5 slate. `./scripts/run_conv_spline_slate.sh`
3. **E6–E8 are chosen from what E0–E5 measure**, not written in advance.

Nothing on this branch is comparable to a `dist-convlstm` number: the trunk change alone makes
`b1` a different model, and the scorecard carries two gates `e1` never had.

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
  (`/mnt/hdd1/spatio-temporal/data`) behind symlinks — but pass the HDD path directly as an
  output, never a symlink, since output directories get cleared with `shutil.rmtree` and that
  refuses on a symbolic link.
- Monitor free space on the HDD and SSD. 

## The loops

| what | command | cost |
|---|---|---|
| one experiment (train held-out folds, predict Africa, stitch) | `BASE_ARGS=... ./scripts/run_central_experiment.sh <name> 0,1 1,2 "<flags>"` | ~50 min, 2 folds in parallel |
| **establish the baseline and its floor** (3 seeds) | `./scripts/run_conv_spline_baseline.sh` | 3x the above |
| **the experiment slate** (refuses without a floor) | `./scripts/run_conv_spline_slate.sh` | 6x the above |
| **score a model**, no ensemble, both gates included | `scripts/score_distributional_model.py --stitched_dir … --folds 1,2` | ~5 min on Africa |
| **rank against the measured floor** | `scripts/compare_conv_spline_runs.py --floor_prefix b1` | seconds |
| **predict-only from frozen checkpoints** | `run_hindcast_folds.py --fold_checkpoints … --max_epochs 0` | ~121 min/fold globally |
| build the neighbourhood-HM covariate (once) | `scripts/prepare_hm_context.py` | 2:45 per base year |

## Rules learned the hard way

Kept only where they still apply. Numbering is fresh; the old file's numbers are gone.

1. **Verify the measurement before believing the finding.** Measurement bugs have outnumbered
   model bugs in this project throughout. Byte-identical numbers across supposedly different
   configurations is the tell — but check the artifacts are genuinely distinct before
   concluding it, since two immaterial fixes look identical too.
2. **Three bugs were caught setting up this phase, each of which would have read as a null
   result.** A closed-form CRPS returning *negative* values; a knot preset silently missing the
   0.5 and 0.975 gate levels; and `pwl`/`isqf` taking the triple-head two-pass code path
   because `head_family == 'spline'` was spelled out in three separate places. The third is the
   general form: **a predicate written more than once is a predicate that will disagree with
   itself.**
3. **The second implementation earns its keep, fourth time now.** The rational-quadratic closed
   form was wrong by a *relative* error of ~1; the piecewise-linear one returned negatives.
   Both were caught only by a dense numerical reference sharing no quadrature code. Keep one.
4. **A metric failure and a model failure are indistinguishable from the outside.** Put the
   check where the cause is one line away: `validate_knots` refuses a bad grid at definition
   time, the qf reader refuses a non-increasing u-grid rather than passing NaN downstream.
5. **Prove a check fires on a control, or it checks nothing.** Before `conv_spline_base.sh` was
   allowed to grep for the trunk-context fingerprint, that fingerprint was verified to
   discriminate ON from off. **A check can also fail *open*, which is silent.**
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
    anything `BASE_ARGS` does not name is silently inherited. `conv_spline_base.sh` names every
    weight and reads them back out of the run's own log.
17. **Any long-range covariate must be precomputed on the full raster**, never derived inside a
    128 px chip (radii ≥ 30 px saturate against the chip boundary). This is about where the
    covariate is *derived*, not where it is *consumed* — which is why b1 can feed the same
    precomputed tensor to the trunk.
18. **The stitched hindcast is a quilt of five fold models, and the seam is real.** Products and
    display rasters use `--stitch_mode mean` (seamless, in-sample); **anything scored stays on
    `holdout`**. Never score a mean-stitched raster.
19. **The fold tile is one correlation length, so held-out skill is optimistic.** The residual
    field's practical range on Africa is 99–166 px against a 128 px fold tile. `fold_mask_b4`
    (512 px blocks) is what this phase uses.
20. **Most fold disagreement is optimisation noise, not data** — 85% of the variance at h=20.
    Central-field seed noise is ~0.0003, so a large move in the central field cannot be
    optimisation noise; the *upper half-width* is where the seed spread lives.
21. **`ModelCheckpoint` monitors what you tell it**, and a quantile-only change still selects a
    different epoch through a shared monitor. This phase uses `--checkpoint_monitor val_crps`
    so a run setting `--mu_mse_weight 0` is not selecting on a different quantity from the rest
    of the slate.
22. **A regional working set can hide a quadratic.** Something `(M, H, W)` and unremarkable on
    1.86 Mpx is dead on 63.1 Mpx. Check peak memory against the region you will actually run.
23. **Never edit a shell script while it is running.** Bash reads a script incrementally by byte
    offset, so inserting lines mid-run makes it resume at a stale offset and execute a
    fragment. Copy to a new name and launch that. Likewise `pkill -f <pattern>` matches the
    shell running it — use `ps -eo pid,cmd | grep "patt[e]rn"` and kill by pid.
24. **A leak shows up as a gradient, not a level.** Removing a leak must make error rise *with*
    distance from the fold's own trained-on data; a flat profile that is simply worse
    everywhere is a training signature.

## Conventions

- Flat argparse, no config framework. New flags are additive and default to today's behaviour;
  the boolean idiom is
  `type=lambda x: (str(x).lower()=='true'), nargs='?', const=True, default=…`.
- Tests are flat in `tests/`, pytest, `sys.path.insert(0, parent)` + absolute imports,
  synthetic tensors. **The suite stands at 355 passed / 15 failed.** Those 15 fail identically
  on `dist-convlstm` (stale legacy tests asserting a 4-channel output from a model that emits
  12, plus fixtures needing data that is not on this machine). Any other failure is yours.
- **A new experiment needs a test that it changed something**, against a seeded control. A flag
  that is accepted, logged and inert reads downstream as "the experiment was a null" rather
  than "the experiment never ran" — see `tests/test_conv_spline_flags.py`.
- Scripts write to `data/conv_spline/exp/<name>/` so configurations never collide.
- Simplicity is a scoring criterion, not a preference. A simpler configuration that gives back
  a little is preferred to a more complex one that does not; the parameter count per horizon is
  reported alongside the metrics for that reason (spline 29, pwl/isqf 16, skew11 23).
