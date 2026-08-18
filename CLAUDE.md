# Working notes for this repository

Spatiotemporal Human Modification (HM) forecasting: a ConvLSTM producing a central forecast
and 2.5/97.5 quantile intervals at +5/+10/+15/+20 yr, plus a post-hoc ensemble layer
(`src/ensemble/`) that adds spatial and temporal coherence on top of the per-pixel
marginals.

Active branch: **`ensemble`**. Read `docs/current_progress.md` for status,
`docs/ensemble_model_outline.md` (and its illustrated HTML edition) for the end-to-end
method and the current scorecard, `docs/validator_scaling.md` for the evaluation harness,
and `docs/central_field_baseline.md` plus `docs/next_phase_marginals.md` §5-6 and
`docs/model_phase.md` for the measurement record — including the negative results, which are
load-bearing and are most of the record.

## Where the project is

Two phases are **closed with nothing further to take from them**:

- **Post-hoc marginals** — scorecard 90/126 → 101/127, per-member 7/20 → 15/20, model
  untouched. Exhausted structurally: T5.2 pins the marginal to the published bounds, so any
  shape re-injects their width error.
- **The ConvLSTM model phase** — **22 experiments, nothing adopted.** Training budget, trunk
  capacity in both directions, receptive field, loss composition, horizon weights, LR
  schedule, gradient clipping, weight averaging, deeper heads, the asinh transform, the
  differentiable histogram loss, `width_head_mode power/power_plus` and
  `quantile_dhat_context` are all measured dead or worse. See [[model-phase-results]] and
  `docs/model_phase.md` before proposing any of them again.

**Evaluation is now continental and the harness scales.** Africa k=5 M=400 scores
**111/133** (`data/ensemble/exp/africa_k5/validation_m400/`), against southern Africa's
101/127. The validator's peak memory follows `--mem_budget_gb`; ensembles are icechunk.

**There is still no production model.** Every checkpoint carrying the current configuration
was trained with a fold held out.

## The remaining plan

1. **Hyperparameter sweep at global extent** (W&B sweeps), screening cheaply, then
2. **promote the best configuration** to the full scorecard, then
3. **global hindcast** — published scorecard and global hindcast maps — and finally the
   **production 2025-2040 forecast**.

The binding constraint on step 1 is the noise floor, not compute: **two runs of one
configuration differ by 7 scorecard rows out of 128 at k=5**, which exceeds every effect the
model phase measured. A sweep cannot be ranked on the scorecard. Screen on
`scripts/score_model_experiment.py` (~40 s, stratified central *and* width, primary quantile
metric `mean|ln k|`), and treat any difference smaller than the base band as noise until it
replicates across seeds.

## Environment and scale

- Conda env **`spatio-temporal-dl`**, not base:
  `/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python`.
- **Screening stays regional; the phase is global.** Southern Africa (1.86 Mpx) is still the
  right extent for a fast A/B, Africa (63.1 Mpx) for a verdict, and the globe
  (17111 x 40000 = 684 Mpx grid, 184.6M valid px) for the published product. Do not screen
  globally — it buys nothing a regional read does not already say, and costs 300x.
- **Global cost, projected from the Africa run** (which is the only continental measurement
  in hand, so treat these as estimates to be checked, not facts):
  prediction ~121 min per fold, ~10 h for k=5; ensemble generation ~1 h per ensemble at
  M=400; **~450 GB per ensemble on disk**, so members + null is ~0.9 TB; and the T1-T8
  scorecard is **the better part of a day** (Africa took 190 min at 18.4 GB peak, and the
  globe is 5.2x its valid pixels and 10.8x its grid). Budget one global scorecard run, not
  an iteration loop.
- Two GPUs (RTX A5000, 24 GB each). Track experiments on **W&B**.
- Large global artifacts live on the HDD (`/mnt/hdd1/spatio-temporal/data`) behind symlinks
  in `data/ensemble/`. Root has ~90 GB free; southern-Africa ensembles at M=400 are ~3 GB
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
| **screen one configuration** (train 2 folds, predict, stitch, stratified read) | `./scripts/run_model_slate.sh <slate> 1,2` | ~35 min each |
| **score a screened configuration**, central *and* width, no ensemble | `scripts/score_model_experiment.py` | ~40 s |
| **promote a winner**: k=5, whole downstream chain, scorecard, per-member | `./scripts/promote_model_experiment.sh <name> "<flags>"` | ~2 h |

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

## Sweeping the model (the current phase)

The machinery already exists and should be driven, not rebuilt: `run_model_slate.sh` queues
screening runs on folds 1-2 with a fixed reference; `score_model_experiment.py` reads each
one in ~40 s; `promote_model_experiment.sh` takes a winner through k=5 and the whole
downstream chain. A W&B sweep should drive `train_lightning.py` with the same fixed folds and
log the `score_model_experiment.py` metrics as the sweep objective.

**The reference for any new run** is the shipped configuration plus the two corrections the
model phase established: `--histogram_weight 0` (the term carries no gradient, and made
differentiable it is decisively worse, but at weight 1.0 it swings ~60x between epochs and
makes checkpoint selection a lottery) and `--checkpoint_monitor val_central_loss` (so a
quantile-only change must leave the central field bit-identical — a free correctness check
rather than a confound).

**Objective.** Not the scorecard: it carries +/-7 rows of run-to-run noise at k=5. Screen on
`mean|ln k|` over classes (the multiplier that would make the published interval the
residual's own 95% interval — a model whose width heads are right needs k = 1 everywhere),
with within-20% and central skill-vs-persistence alongside. Central skill is much the least
noisy of the three.

**Do not sweep what is already measured dead** (see "Where the project is"). Twenty-two
hand-run experiments covered most of the obvious axes and adopted none of them, so a sweep
over the same space will confirm the incumbent at best. Axes that were *not* covered and are
worth the budget: the learning rate value itself (only the *schedule* was tried), batch size,
`--quantile_class_weighting`, `--width_parameterisation exp` vs softplus,
`--initial_width_normalized`, chip sampling density, and `--weight_avg_last` — which the seam
work independently motivated, since 44-85% of the between-fold width disagreement is
optimisation noise.

**Expect the sweep to find nothing, and design for that being an acceptable answer.** The
value of running it is a defensible statement that the configuration is at a local optimum
before a global production run, not a promise of improvement.

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

11. **A width factor must not be centred on the residual median.** `fit_residual_shape`
    centres because T5.1 pins the shape's median to zero; a *width* is centred on the
    central forecast and must cover the residual's bias too. Copying the centring cost a
    full cycle: class coverage fell 0.95 → 0.73 with every miss on the low side.
12. **The centre is RMSE-optimal; do not move it to fix a quantile problem.** This residual
    is strongly right-skewed, so a median bias-shift overshoots the mean and gives back
    skill (0.191 → 0.171 at h=20). Make the interval asymmetric about the unmoved centre
    instead — an uncentred width factor does exactly that, at no cost to the central field.

## Conventions

- Flat argparse, no config framework. New flags are additive and default to today's
  behaviour; the boolean idiom is
  `type=lambda x: (str(x).lower()=='true'), nargs='?', const=True, default=…`.
- Tests are flat in `tests/`, pytest, `sys.path.insert(0, parent)` + absolute imports,
  synthetic tensors. 7 pre-existing failures in the legacy suite are stale (they assert a
  4-channel output from a model that emits 12) and fail identically with this branch stashed.
- Scripts write to `data/ensemble/exp/<name>/` so configurations never collide.
