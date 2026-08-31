# Working notes for this repository

Spatiotemporal Human Modification (HM) forecasting. **Two systems live here and it matters
which one you are working on.**

Active branch: **`dist-convlstm`**.

> **SUPERSEDED 2026-08-30.** The global production phase replaces this product; its *artifacts*
> (`g1_foldb4`, `c1_foldb4`, `africa_k5`, `data/ensemble/hindcast/`) are authorised for deletion.
> The three documents below remain the authority on method and stay. See the
> `global-production-phase` memory before deleting anything.

## 1. The published product — frozen, complete, do not change

Branch `ensemble`, experiment **`g1_foldb4`**. A ConvLSTM emitting `(lower, central, upper)` at
+5/+10/+15/+20 yr, plus a four-stage post-hoc chain (`src/ensemble/`) — width calibration,
empirical marginal, spatial spectrum, AR(1) horizon coupling — and a 400-member ensemble.
**99/134 scorecard rows.** Artifacts on the HDD at `data/ensemble/exp/g1_foldb4`; checkpoints in
`data/ensemble/production/FOLD_CHECKPOINTS.json`.

Three documents supersede everything else and stand on their own:

- **`docs/global_ensemble_methodology.md`** — the method. Start here.
- **`docs/global_scorecard.md`** — every evaluation: what it tests, what failure looks like,
  where its threshold came from, what it scored.
- **`docs/fitting_running_model.md`** — the runbook, with measured cost per stage.

Its three named defects: aggregate over-dispersion, long-range structure at 0.230 of budget, and
**the far field emitting 0.034 of the observed rate of new development beyond 100 px**. The last
is a property of the quantile heads — far-band half-widths are ~0.004, so +0.05 is ~12
half-widths out — and no post-hoc layer reaches it. That defect is why system 2 exists.

One open question, never settled: `predict_change_rates.py` and the sampled M=400 ensemble
disagree on the far band (0.000000 against 0.034), where they had agreed to 1-11% every previous
time. It blocks far-field work *on the frozen product* only.

## 2. Active work — the end-to-end distributional model

Branch **`dist-convlstm`**. **Model phase and ensemble phase are both CLOSED** (2026-08-28,
2026-08-30). `e1` is the model; `V4b` is the ensemble — PIT-space spectrum, `--long_weight 0`,
`--copula t --copula_df 7 --copula_w_draw stratified`. Full writeup `docs/dist_ensemble_phase.md`.
Current work is the **global production hindcast + 2025-2040 forecast**.

The model emits a full per-pixel quantile function `Q_h(u|x)` and
that function *is* the product: no conformal scaling, no width factors, no empirical marginal
reshaping, no horizon-monotonicity pass. A monotone rational-quadratic spline over absolute HM
with **fixed tail-dense knots**, anchored at persistence through the existing zero-init residual
head, with a cumulative scale so spread cannot shrink with lead time, trained on CRPS.

- **`docs/dist_model_phase.md`** — round 1: the method, every result, every null.
- **`docs/dist_scorecard.md`** — every metric explained in plain terms with its measured value.
- **`docs/superpowers/specs/2026-08-25-distributional-round-2-design.md`** — round 2's design.

**Round 1 adopted nothing but established three things**, all properties of the design rather
than of a knob: the model beats persistence as a full distribution by ~13% at every horizon and
its published triple reproduces its own quantile function exactly; **pure CRPS costs nothing
centrally** (`--mu_mse_weight 0` leaves `rmse5` inside a 2%-wide band); and **the trunk must hear
the distributional objective** (isolating it leaves RMSE intact while `crps_skill5` collapses
0.13 → 0.059). Separately, NLL collapses to a point mass at the width floor — the *opposite*
direction to the incumbent's Gaussian-NLL failure, which inflated widths 7x.

**Why nothing was adopted, and the round's real finding:** `tail_reach20` varies by **4.3 within
one configuration**, so six of ten variants were undecidable. Worse, three baseline replicates
all landed in one mode and made the band read 0.86 — four times too tight. And `tail_reach20`
and `pit_ks5` correlate at **r = 0.691** across 13 runs, so three knobs that each looked like
"trades calibration for tail reach" were one axis seen three times.

**Round 2 is implemented, tested and PAUSED before its Phase 1 stability gate.** Nothing is
committed. Resume with `./scripts/run_dist_gate.sh` — see §The loops.

## Environment and scale

- Conda env **`spatio-temporal-dl`**, not base:
  `/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python`.
- **Screening is regional, and southern Africa's far-field band is EMPTY.** Measured
  2026-08-24: the >100 px distance band contains **zero pixels**, not merely a zero change
  rate. So a far-field row there is structurally absent, not quiet, and no experiment aimed at
  what the model does beyond ~100 px can report anything. Band 4 (30-100 px, 147,721 px) is the
  furthest measurable proxy, observed `P(Δ>0.05)` = 0.00225 at h=20. Southern Africa (1.86 Mpx)
  remains the right fast A/B for everything else. **Screen far-field work on Africa**
  (63.1 Mpx), which has a real far-field rate. The globe (17111 x 40000 = 684 Mpx grid, 184.6M
  valid px) stays for the published product only.
- **Global cost, projected from the Africa run** (which is the only continental measurement
  in hand, so treat these as estimates to be checked, not facts):
  prediction ~121 min per fold, ~10 h for k=5; ensemble generation ~1 h per ensemble at
  M=400; **~450 GB per ensemble on disk**, so members + null is ~0.9 TB; and the T1-T8
  scorecard is **the better part of a day** (Africa took 190 min at 18.4 GB peak, and the
  globe is 5.2x its valid pixels and 10.8x its grid). Budget one global scorecard run, not
  an iteration loop.
- Two GPUs (RTX A5000, 24 GB each). Track experiments on **W&B**.
- Large global artifacts live on the HDD (`/mnt/hdd1/spatio-temporal/data`) behind symlinks
  in `data/ensemble/`. **2.1 TB free after the 2026-08-20 cleanup**, which removed the whole
  pre-`--central_residual` lineage (its ensembles, residuals and validation), the global null,
  and Africa's three M=400 stores — every one of them regenerates from recorded seeds.
  **`data/ensemble/hindcast/stitched/` was deliberately kept**: it is the *main-branch
  architecture* global hindcast and the baseline for the old-vs-new comparison. Root has **~201 GB free** (2026-08-25);
  southern-Africa ensembles at M=400 are ~3 GB
  each, so delete superseded ones (they regenerate from recorded seeds in ~3 min). Africa at
  M=400 is ~85 GB per ensemble and belongs on the HDD — pass the HDD path directly as
  `--out`, never a symlink, since the store's directory gets cleared with `shutil.rmtree`
  and that refuses on a symbolic link.
- **Ensembles are icechunk repositories, not plain zarr** (`*.icechunk`, one array named
  `members`). `src.ensemble.aggregate.open_ensemble` opens them; everything downstream is
  unchanged because it still returns `(array, attrs)`. The write is one transaction: each
  GPU worker writes through a forked session and the parent merges and commits once, so a
  killed run leaves no store instead of a directory that reads back as sentinel.
  `scripts/migrate_zarr_to_icechunk.py --verify` converts an old store and proves the copy
  byte-identical.

## The loops

| what | command | cost |
|---|---|---|
| one model experiment (train held-out folds, predict the region, stitch) | `./scripts/run_central_experiment.sh <name> <gpus> <folds> "<flags>"` | ~25 min/fold, 2 folds in parallel |
| Phase 1→4 for one experiment | `./scripts/run_region_loop.sh <name> <members>` | ~15 min at M=20, ~45 min at M=400 |
| central-field error, stratified | `scripts/diagnose_central_field.py --stitched_dir …` | seconds |
| compare configurations | `scripts/compare_central_runs.py label=dir …` | seconds |
| **score a marginal without generating anything** | `scripts/predict_change_rates.py --manifest … [--sweep_u_bound …]` | ~30 s |
| per-member realism (the honest marginal test) | `scripts/member_distance_relationship.py --ensembles label=path … --members 400` | ~10 min |
| convert an old plain-zarr store | `scripts/migrate_zarr_to_icechunk.py --src … --dst … --verify` | ~1 min/GB |
| **prove the chain before a long run** (every stage at M=4, all four block scales) | `./scripts/run_smoke_tier.sh <name>` | ~33 min on Africa, longer globally |
| **predict-only from frozen checkpoints** (no retraining) | `run_hindcast_folds.py --fold_checkpoints … --max_epochs 0` | ~121 min/fold globally |
| **score calibration variants on held-out folds** | `scripts/score_width_variants.py --score_folds 4,5 --variants …` | ~15 min |
| **check a calibration keeps the far-field tail** | `scripts/band_tail_audit.py --variants …` | seconds |
| **score a screened configuration**, central *and* width, no ensemble | `scripts/score_model_experiment.py` | ~40 s (5 min on Africa) |
| **one distributional experiment** (train held-out folds, predict, stitch) | `BASE_ARGS=... ./scripts/run_central_experiment.sh <name> 0,1 1,2 "<flags>"` | ~50 min, 2 folds in parallel |
| **score a distributional model**, no ensemble | `scripts/score_distributional_model.py --stitched_dir … --folds 1,2` | ~2 min |
| **rank runs against the measured floor** | `scripts/compare_dist_runs.py` | seconds |
| **the stability gate** (4 configs x 3 seeds, judged on *spread*) | `./scripts/run_dist_gate.sh` | ~10 h |
| **the round-2 slate** (needs `GATE_FLAGS`; refuses without it) | `GATE_FLAGS=… SEEDS=… ./scripts/run_dist_phase2.sh` | ~13-20 h |
| **build the neighbourhood-HM covariate** (once) | `scripts/prepare_hm_context.py` | 2:45 per base year, 0.75 GB each |

`run_region_loop.sh` re-derives the recalibration, spectrum and AR(1) coupling from *that
model's own* residuals. Never carry them over between configurations. `SHAPE=<u_bound>`
adds the empirical marginal, re-fitted from those same residuals; `SUFFIX=<tag>` keeps two
marginal families side by side under one experiment.

**Never sweep a marginal by generating ensembles.** The member *mean* of `P(Δ > thr)` is
available in closed form — the field's correlation moves the members' scatter, not their
mean — so `predict_change_rates.py` answers in 30 s what a generate-and-validate cycle
answers in 35 min. It agreed with three M=400 ensembles to 1–11%. Its second job is being a
second implementation: it shares no code with the GPU sampler, so agreement is evidence and
disagreement localises the bug to the generation path.

## Rules learned the hard way

1. **Verify the measurement before believing the finding.** Measurement bugs have
   outnumbered model bugs in this project throughout. Byte-identical numbers across
   supposedly different configurations is the tell — but check the artifacts are genuinely
   distinct before concluding it, since two immaterial fixes look identical too.
2. **Score against persistence, not zero.** The median 20-year HM change is 0.0001. Pooled
   RMSE looked unremarkable while the central forecast was losing to "nothing will change"
   by 2.2× in MSE at h=5.
3. **A diagnostic can encode an assumption it never states.** Three separate scoring bugs
   this session all assumed the two-piece normal marginal. Each announced itself as a result
   that *could not be true* (a gate getting worse at higher M; a field property moving under
   a rank-preserving transform), not by inspection. When a number is impossible, suspect the
   metric.
4. **Judge sweeps on per-row values, never a total pass count.** Two settings tied on count
   while one had quietly pushed near-nominal rows below target.
5. **A metric pinned at 1.000 and insensitive to its own knob is under-powered, not
   mis-tuned.** Check the number of aggregation units before turning anything.
6. **Pooling across members answers a weaker question than it appears to.** Per-member
   statistics with the observation's *rank* among members is the honest test; a pool that is
   uniformly too hot still overlaps the truth.
7. **`ModelCheckpoint` monitors `val_total_loss`, which includes pinball** — so a
   quantile-only change still selects a different epoch and therefore a different central
   field. Central-only A/Bs need a central-only monitor.
8. **Any long-range covariate must be precomputed on the full raster**, never derived inside
   a 128 px chip (radii ≥ 30 px saturate against the chip boundary).
9. **A binning convention is a measurement.** `right=True` vs `right=False` on the distance
   bands lived at four call sites each; the distance is an exact Euclidean transform, so
   `dist == 1.0` is one of the most populated values on the map and the 0–1 px band's
   observed `P(Δ>0.05)` reads 0.222 one way and 0.177 the other. One definition now:
   `src.ensemble.validate.distance_band`. Use it; do not re-derive the edges.
10. **When a change has two parts, build the control that separates them.** Conditioning the
    marginal on distance band and raising its tail bound were introduced together and looked
    like a win. The pooled fit at the *same* raised bound beat both, so the band axis was
    carrying nothing.
13. **Hold something out before believing a class-conditional fit.** A correction fitted to
    the metric it is scored on will look good and not generalise: per-class tail bounds beat a
    flat bound in sample and lost on two independent held-out protocols. Quantile *estimates*
    on 15k+ pixels survived the same test; *metric minimisations* over ~60 events did not.
14. **Southern Africa changes 2-4x less than Africa at every HM level**, and its `[0,0.01)`
    stratum is 6% of the region against 40% of Africa. Regional iteration is right for speed,
    but a stratified finding measured only there is provisional. Only w2000 reaches +20 yr
    whatever the geography, so every h=20 number is in-sample in time.
    **This cost the project a whole phase.** Southern Africa's far-field band holds **zero
    pixels** (measured 2026-08-24; earlier notes said "rate 0.0000", which understated it), so
    "the ensemble emits zero change beyond 100 px" was recorded as a *win* there — and is a 44x
    under-prediction globally, where the observed rate is 0.0021 at h=20. All 22 model
    experiments were screened against that blind spot. A metric with no denominator on the
    screening region cannot rank anything.
15. **A regional working set can hide a quadratic.** `validate_ensemble.py` allocated
    `(M, H, W)` float64 for the block-mean pass, which at `--block_sizes 1,10,100` is the
    pixel grid: 6 GB on southern Africa's 1.86 Mpx at M=400 and unremarkable, 50 GB on
    Africa's 63.1 Mpx at M=100, dead. Every scored statistic there was a count or a mean
    over blocks, so none of it ever needed to be resident. Peak memory is now set by
    `--mem_budget_gb` and the stages are checked against it with `--mem_trace`.
16. **Sample memory faster than the thing you are watching for.** The OOM went from steady
    to killed in under 60 s; a 120 s poll saw nothing. `src/ensemble/memtrace.py` samples
    `/proc/self/statm` at 0.25 s and attributes each sample to the innermost labelled
    section, which is what turned "T2 died" into "`block_member_stats` at B=1".
17. **In icechunk, a partial-chunk write keeps both versions.** Writing one member into a
    ten-member chunk is a read-modify-write; plain zarr overwrote the file, icechunk retains
    every version until garbage collection. The first M=400 null came to 16.7 GB against
    2.9 GB for the identical array, and took 4x as long. Chunks are `(1, 1, 1024, 1024)` now
    — one member per chunk, which also means two workers can never share one.
18. **The stitched hindcast is a quilt of five models, and the seam is real.** The k=5 fold
    mask is a 128 px checkerboard, so adjacent tiles come from different fold models and
    `stitch_fold_predictions` joins them with no blending. They agree on the central field
    and the lower bound (mean pairwise |fold_i - fold_j| 0.0053 and 0.0040) and not on the
    upper (0.0383), so the join shows in the upper bound only — 2.01x the within-fold step,
    47x the local background in quiet country. A single fold's own prediction is seamless at
    every period from 64 to 1024 px; neither the model nor the overlap blending is involved.
    **The settled recipe: products and display rasters use `--stitch_mode mean` (every fold
    averaged, seamless, in-sample); anything scored stays on `holdout` (the default).** Never
    score a mean-stitched raster. The forward 2025-2040 product has no mosaic and no seam.
19. **The fold tile is one correlation length, so held-out skill is optimistic.** The residual
    field's fitted practical range on Africa is 99-166 px, typically ~130, against a 128 px
    fold tile — every held-out tile is ringed by trained-on tiles well inside the range the
    residual is still correlated over. `create_validity_mask.py --fold_block_chips 10` builds
    contiguous 1280 px fold territories instead; implemented, **not trained**, because
    adopting it retrains all five folds and makes every existing scorecard non-comparable.
20. **Most of the fold disagreement is optimisation noise, not data.** Retraining one fold
    with only the seed changed gives a seed-to-seed spread of 0.0308 on the upper half-width
    against a fold-to-fold 0.0335 at h=20 — 85% of the variance at h=20, 44-63% at shorter
    horizons. Repeated k-fold CV or repeated seeds would average this away at 5x the budget
    and still only shrink the seam ~2.2x; **both were rejected**. The cheap lever is the
    training recipe: `--checkpoint_monitor val_central_loss` alone cuts the spread 0.0335 ->
    0.0275 for free, and `--weight_avg_last` exists and is untested for it.

21. **A gate written for one failure mode keeps passing after the mode inverts.** T8.2, T8.3
    and T6.4 were correct instruments when written — remote-band *invention* was the failure,
    and a ceiling bounded it. Nothing about them decayed; the system moved to the other side.
    All three then read their best at the moment the far field went silent, which is the
    defect. Re-read any one-sided gate whenever the thing it bounds changes sign, and prefer
    a ratio to observed, which cannot be satisfied from either direction alone.
22. **Sparse cells are not automatically an invalid chi-square.** The first T2.5 fix was
    written on the textbook "expected >= 5" rule and would have been recorded as a validity
    correction. Measured, the size is nominal; the change survives only because the *power*
    argument is separately true. Verify the measurement before believing the finding —
    including when the finding is your own diagnosis of someone else's instrument.

23. **A fold-mask change moves what the validation stride samples — but that was not the
    bug, and the correction is recorded because I asserted it before testing it.**
    `VAL_STRIDE=2048` draws 34 fold-2 chips under the 1x1 checkerboard (mean RMS 5-yr change
    0.0114); under 512 px blocks the same stride hits 30 of ~530 blocks holding mean RMS
    **0.0003**, 13x quieter. That difference is real and worth knowing. **It is not what
    broke the run** (rule 25 is), and fixing it in isolation made things slightly *worse*:
    at defaults, stride 2048 scored h=5 skill −0.641 on fold-1 territory and stride 1024
    scored −0.842. The stride's effect in the *correct* configuration has never been
    isolated — that needs a prod-flags run at stride 2048, which has not been done.
    `VAL_STRIDE=1024` (131 chips, mean RMS 0.0075) is what the accepted run uses, on the
    argument that a selector should see change, not on measured benefit. Dead run:
    `exp/c1_foldb4_valstride2048/`.
24. **A leak shows up as a gradient, not a level.** The check that killed the leak reading in
    one step: RMSE against distance from the fold's own trained-on data. Removing a leak must
    make error rise *with* that distance; the bad run was flat (0.01415 at 1-32 px, 0.01475
    beyond 192 px) and simply worse everywhere, which is a training signature. Seed replicates
    bounded it from the other side — three seeds on the old mask give skill +0.0442 / +0.0422
    / +0.0452, so **central-field seed noise is ~0.0003** and a 0.7 move cannot be optimisation
    noise. (Rule 20's large seed spread is the *upper half-width*; the central field is stable.)

25. **"No extra flags" is not the production architecture — it is argparse defaults.**
    `run_central_experiment.sh` used to echo `<production architecture>` for an empty flag
    string while `--central_residual` defaults to **False**, under which the central head
    predicts absolute HM and reconstructs the baseline through the trunk. The shipped set is
    `--central_residual True --central_context True --monotone_quantile_width True
    --quantile_context True` (`PHASE_REF` in `run_model_slate.sh` /
    `promote_model_experiment.sh`; the model phase later added `--histogram_weight 0
    --checkpoint_monitor val_central_loss`, which `africa_k5` did **not** carry). A k=5 run
    launched on defaults scored h=5 skill −0.50 against `africa_k5`'s +0.13 and read exactly
    like a fold-mask finding. The runner now refuses to start without `BASE_ARGS` unless
    `ALLOW_DEFAULT_ARCH=1`. **Verify the fingerprint before trusting a retrain:** with
    `--central_residual` the startup diagnostic prints `ConvLSTM grad norm: 0.000000` (the
    trunk gets no gradient from the central loss); on defaults it prints ~0.04 and the
    initial central loss is 20x higher. Dead run: `exp/c1_foldb4_noflags/`.

26. **A pooled average cannot see a defect that lives in a thousandth of the pixels.**
    Keying the calibration on *biome* rather than distance band won the held-out interval
    score by 0.8% and destroyed the far field: `P(Δ>0.05)` beyond 100 px went 0.845 → **0.042**
    of observed, because biome classes average across distance and regressed the far band's
    p99.9 half-width to **0.435×** — while the *median* far-field pixel got 1.65× **wider** the
    whole time. Those pixels are ~0.1% of the band. **Judge a calibration on the interval score
    AND a tail check** (`scripts/band_tail_audit.py`); the far-band p99.9 half-width is the
    quantity that carries T8. The two axes were never redundant: distance separates remote land
    that can develop from remote land that cannot, and any axis that blurs that scores well on
    average and removes the product feature.
27. **A restriction inherited from an older model can invert.** The width layer corrected only
    bands 0-2 because far-band narrowing once made +0.05 unreachable. On this model's residuals
    the far-band fit *widens* (p99.9 at 3.9× the raw heads), so the restriction was still being
    applied for a reason that had stopped being true — the southern-Africa blind spot one layer
    down. Re-measure the reason, not just the setting.
28. **Never edit a shell script while it is running.** Bash reads a script incrementally by
    byte offset, so inserting lines mid-run makes it resume at a stale offset and execute a
    fragment. It cost a stitch stage this round (`odel_experiment.sh: command not found`), and
    the same edit made twenty minutes earlier would have destroyed the run. Copy to a new name
    and launch that. Likewise `pkill -f <pattern>` matches the shell running it — use
    `ps -eo pid,cmd | grep "patt[e]rn"` and kill by pid.

11. **A width factor must not be centred on the residual median.** `fit_residual_shape`
    centres because T5.1 pins the shape's median to zero; a *width* is centred on the
    central forecast and must cover the residual's bias too. Copying the centring cost a
    full cycle: class coverage fell 0.95 → 0.73 with every miss on the low side.
12. **The centre is RMSE-optimal; do not move it to fix a quantile problem.** This residual
    is strongly right-skewed, so a median bias-shift overshoots the mean and gives back
    skill (0.191 → 0.171 at h=20). Make the interval asymmetric about the unmoved centre
    instead — an uncentred width factor does exactly that, at no cost to the central field.

29. **Three replicates can all land in one mode, and the band you measure is then far too
    narrow.** Three baseline seeds gave a `tail_reach20` spread of 0.86; three replicates of a
    *variant* of the same configuration gave **4.3**. Every bar applied against the first number
    was wrong, and the round's only apparent winner failed its own replication. If a floor's
    replicates agree suspiciously well, suspect the sample before believing the band. This is
    rule 20's problem one level deeper: not "is the floor large" but "did I sample it".
30. **Two metrics that look independent can be one axis.** `tail_reach20` and `pit_ks5` correlate
    at **r = 0.691** across 13 runs. Three separate knobs each appeared to trade calibration for
    tail reach; they were all landing at different points on one axis every run sits on.
    Correlate the metrics across runs before reading N knobs as N findings.
31. **A metric failure and a model failure are indistinguishable from the outside.** A degenerate
    u-grid — two output levels equal after the writer's `%.8g` tag rounding — made a segment slope
    0/0 and CRPS `NaN` for every pixel. Rule 3 again, and the fix is to put the check where the
    cause is one line away: the reader now refuses a non-increasing grid rather than passing NaN
    downstream.
32. **"The flags I passed" is not "the flags that took effect."** `run_hindcast_folds.py` injects
    the incumbent's own loss weights into every fold command and `--extra_train_args` is appended
    last, so **anything `BASE_ARGS` does not name is silently inherited**. A floor run trained
    with SSIM and Laplacian active while the slate defined its baseline as probabilistic-alone;
    caught at 35 min by reading the run's own `LOSS WEIGHTS` banner. `scripts/dist_base_args.sh`
    holds the baseline once and `verify_loss_weights` reads it back out of the log. Rule 25 in a
    new costume.
33. **A lever whose signature only reaches W&B cannot be verified from a log.** A cosine-schedule
    fingerprint would have false-alarmed on every legitimate run until an explicit print was
    added. Before writing a check, confirm the thing it greps for is actually emitted — and
    prove the check fires on a control, or it checks nothing.
34. **The second implementation earns its keep, third time now.** A closed-form CRPS was wrong by
    a *relative* error of ~1: the crossing point clips to a segment end far more often than it
    lands inside one, so the second piece's error at its own origin is not zero. Only a dense
    numerical reference sharing no code with it caught this.
35. **Check a filename before writing to it.** `tests/test_round2_flags.py` already existed — the
    incumbent's round 2 — and was overwritten by picking the obvious name. Restored from git; the
    distributional one is `test_dist_round2_flags.py`.

## Conventions

- Flat argparse, no config framework. New flags are additive and default to today's
  behaviour; the boolean idiom is
  `type=lambda x: (str(x).lower()=='true'), nargs='?', const=True, default=…`.
- Tests are flat in `tests/`, pytest, `sys.path.insert(0, parent)` + absolute imports,
  synthetic tensors. **The suite stands at 318 passed / 7 failed**; those 7 are stale legacy
  tests (they assert a 4-channel output from a model that emits 12) and fail identically with
  this branch stashed. Any other failure is yours.
- Scripts write to `data/ensemble/exp/<name>/` so configurations never collide.
- **The distributional lineage keeps its own drivers and scorers**, none of which touch the
  frozen product's: `run_dist_gate.sh`, `run_dist_phase2.sh`, `run_dist_floor.sh`,
  `run_dist_replicate.sh` (all sourcing `dist_base_args.sh`), `score_distributional_model.py`,
  `compare_dist_runs.py`, `prepare_hm_context.py`. `score_model_experiment.py` is deliberately
  left untouched and still worth running — its `k_up` table is an independent width check on
  the emitted triple, and two instruments sharing no code is how a bug gets localised.
