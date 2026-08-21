# Working notes for this repository

Spatiotemporal Human Modification (HM) forecasting: a ConvLSTM producing a central forecast
and 2.5/97.5 quantile intervals at +5/+10/+15/+20 yr, plus a post-hoc ensemble layer
(`src/ensemble/`) that adds spatial and temporal coherence on top of the per-pixel
marginals.

Active branch: **`ensemble`**. Read **`docs/global_scorecard.md` first** — it is the current
state of the product and names the live defect. Then `docs/current_progress.md` for history,
`docs/ensemble_model_outline.md` (and its illustrated HTML edition) for the end-to-end
method and the current scorecard, `docs/validator_scaling.md` for the evaluation harness,
and `docs/central_field_baseline.md` plus `docs/next_phase_marginals.md` §5-6 and
`docs/model_phase.md` for the measurement record — including the negative results, which are
load-bearing and are most of the record.

## Where the project is

**The global product is measured.** The five Africa fold checkpoints were re-predicted over the
full 17111 x 40000 extent (no retraining), stitched holdout *and* fold-mean, and pushed through
the whole chain re-derived from their own residuals. **The published global scorecard is
112/146 at M=400** — `docs/global_scorecard.md` is the full reading and is the document to start
from. Africa's 111/133 and southern Africa's 101/127 are superseded as references and are not
directly comparable to it (different denominators; see that doc §7).

**The live target is the width heads in the far field**, and Stage B (2026-08-20) narrowed it
to one side. Beyond ~100 px the ensemble emits **2.3% of the observed rate of increase**; that
is 83,797 observed global events and ~346 on Africa, so it is measurable and screenable.
The mirror figure — 28x the observed rate of decrease — is **88 events on 0.14% of the band**
and should not be fitted against (rule 13); Africa carries exactly one, so it cannot screen it
at all.

**Post-hoc is exhausted, measured not assumed.** Sweeping the remote tail bound on Africa,
0.999 and 1.0 give *identical* far-field rates — the lever saturates — and even unbounded the
far field reaches only 1.7-26% of the observed increase rate. Raising the bound is still worth
taking for the **mid** field (30-100 px: 0.109 -> 0.449 at h=5; pooled mean |log10 ratio|
0.587 -> 0.297).

**Do not fix the far field by widening it.** `score_model_experiment.py` on the Africa stitch
puts `k_up` at **0.24-0.75** beyond 100 px with coverage **0.997** against a 0.95 target: the
far-field intervals are already 2-4x too wide in the bulk while being too short-tailed in the
extreme. The lever is tail *shape* at fixed or reduced bulk width.

**The physical floor is already implemented.** `marginal_from_z_torch` clamps members to
HM in [0, 1], so Δ >= −HM_t0 holds by construction and every one of the 88 observed far-band
decreases lies above it. `docs/global_scorecard.md` §3.2's "missing floor" mechanism is
corrected in place and the plan item is dropped.

**Do not treat the model phase as closed on this.** Its 22 experiments were all screened on
southern Africa, **where the observed far-field change rate genuinely is 0.0000** — so none of
them could have detected this defect, and none of them is evidence against fixing it. See
[[model-phase-results]] for what *is* dead and `docs/global_scorecard.md` §3 for what the
screening was blind to. **Screen far-field work on Africa or global, never southern Africa.**

Still closed with nothing further to take from them: post-hoc marginal *shape* (T5.2 pins it to
the published bounds), the southern-Africa-screened model levers, and the Africa dress rehearsal.

**There is still no production model.** Every checkpoint carrying the current configuration was
trained with a fold held out. The 2025-2040 forecast has not been run; when it is, the decision
already taken is a sixth model trained on all chips, with the fold mean reserved for display.

## The plan for this round

**The round is complete. `docs/reference_card.md` is the current state of the product.**
`exp/c1_foldb4/validation_m400/` — **103/136 at M=400 on Africa**, the first card of the
honest-fold-mask lineage. `africa_k5`'s 111/133 and the Stage A re-score's 102/133 are
superseded and **not comparable** to it: the mask, five gates, the calibration and the
marginal all changed.

**The far field is fixed on the increase side, which was the round's target.** Remote-band
`P(Δ>0.05)` goes **0.107 → 0.845** of observed (T8.1 passes there for the first time) and the
near/remote contrast lands at 1.63 against [0.5, 2.0]. The lever was `--u_bound 0.999` on
every band including the remote one. Two caveats: the 30-100 px band tips to 2.03, just past
its bound, so a band-varying bound is the cheap follow-up; and the *decrease* side is still
18.9x, which 4-diag showed rests on 88 global events and cannot be fitted against.

**What is left is mostly model work, and it is named:** ecoregion-scale over-dispersion
(T2.5 0/4, corroborated by T7.3 = 1.64) and long-range structure at an eighth of the
residual's own budget (T3.2 = 0.125). **T4.2 = 0.802 is NOT model work** — decomposed, the raw
heads are 0.980 monotone and the *calibration* takes it to 0.813 (the conformal layer smooths
isotonically across horizons, the width layer does not, and 2.6% of pixels change class
between horizons). Fix is a cumulative max on the half-widths after both layers: 1.000 by
construction for +1.2-2.5% width, and it should help T1.1's under-coverage. **Item 1 (unify
the two calibration layers) was never done** — Stage D measured the axes and stopped; the
biome axis it turns on is still untested. Plus one open instrument question — **T5.1 fails at h=5/h=10 and it is not
settled whether that is generation or the gate's own tolerance**, which tightens mechanically
when `u_bound` rises. See the reference card §4.

**Stage C is done: the 4x4 fold mask is adopted and it is nearly free.** `exp/c1_foldb4/`
(mask `fold_mask_b4_1000.tif`, `VAL_STRIDE=1024`, shipped flags) scores **0.1301 / 0.1960 /
0.2277 / 0.2221** against `africa_k5`'s 0.1298 / 0.2006 / 0.2340 / 0.2313 — **at most 0.9
skill points, none at h=5**. Every card from here uses it; `africa_k5` and the 102/133 card
are the last of the 1x1 lineage.

**The leak is now measurable and modest.** Within the new mask, h=20 skill falls **+0.239
(1-32 px) → +0.188 (>192 px)** with distance from the fold's own trained-on data, a 21%
relative decline. The 1x1 mask stops at 128 px and cannot see it. Rule 19 is confirmed and
quantified rather than feared.

**Stages A and B are done.** `docs/stage_a_instruments.md` records the instruments;
`docs/improvement_plan.md` Stage B records the four diagnostics and what they changed:
item 4 **dropped** (the floor already exists), item 7 **moved to Stage D** (h=5's width error
is class-concentrated, so it is calibration not the head), item 5 **demoted to a mid-field
fix**, and Stage C **confirmed to need a far-field width-head change**.

**T4.2 is fixed too** (a fifth instrument, found in Stage B). It scored the *sample* spread
over M members, so it moved 0.62 -> 0.45 under rho and 0.71 -> 0.62 under M — neither of which
can change the quantity it claims to measure. It now scores the **population** spread,
integrated against N(0,1) by Gauss-Hermite (`validate_ensemble.population_spread`,
shape- and clip-aware); the sample statistic survives as reported-only **T4.2s**. Two
ensembles differing only in rho return **0.7304648026222341 to every digit**. The gate still
fails, correctly: **the width heads shrink with horizon at 27% of the map**, which is a model
defect for the far-field width work and cannot be reached post-hoc.

**Africa's measured rho is `{10: 0.374, 15: 0.347, 20: 0.769}`**, now on disk at
`exp/africa_k5/residuals/horizon_autocorrelation.json`. With it T4.1 becomes scorable and
passes all three rows to <= 0.025; every regional card to date, including 111/133, ran at the
0.9 fallback against a NaN target.

**Stage A is done** — `docs/stage_a_instruments.md` is the record. The scoring instruments
are fixed and the Africa baseline is re-scored: **111/133 → 102/133 on the same ensemble**
(`data/ensemble/exp/africa_k5/validation_m400_stageA/scorecard_spliced.csv`). All nine lost
rows are instruments — T8.2, T8.3 and T6.4×4 were passing *because* the far field is too
quiet, and T2.5×3 were non-rejections at n=158 rather than flatness. **111/133 is superseded;
compare everything from here against 102/133.**

Two premises in the plan did not survive measurement, and both corrections are load-bearing:

- **T2.5 is not a pixel test and its χ² was never invalid.** Its units are ecoregions (158 on
  Africa, 804 global) against 401 rank bins, and at that sparsity the equiprobable-cell χ²
  holds its size (0.0105 measured at nominal 0.01). The defect is *power* — 0.59 against a
  smooth dome, 0.90 pooled — so the ranks are pooled to ≥16 expected per bin. Ecoregion-scale
  over-dispersion is real on Africa too, not just globally.
- **T3.2's "unreachable target" argument is southern-Africa-only.** The structure budget is
  0.14 at 25 px there and **0.497** at Africa's scored separation, so the old 0.30 threshold
  sat *below* the budget and the target was always reachable here. T3.2 now gates on
  `improve / budget ≥ 0.50`; Africa reads 0.063 at 40 px and 0.357 at 6 px. Treat it as a
  genuine long-range structure shortfall, not a threshold to waive.

**Run `./scripts/run_smoke_tier.sh <exp>` before any long validation run** — every stage at
M=4, all four block scales, ~33 min on Africa, and `scripts/check_smoke.py` gates on stage
completeness and non-finite values rather than on pass counts.

**`docs/improvement_plan.md` is the sequence.** Ten changes in six stages, all evaluated on
Africa. The ordering is fixed by three constraints: fix the scoring gates before judging anything
(T8.2/T8.3/T6.4 currently pass *because* the far field is too quiet), change the fold mask only
once (it invalidates every checkpoint and every residual-fitted calibration), and run the cheap
diagnostics first because five of the ten items have an experiment that decides their own scope.

Two findings already settled inside that plan:

- **Fold blocks: 2x2 does not work, use 4x4.** The residual's practical range is 135 px and a
  pixel in a 128·B block is at most 64·B px from trained-on data, so at 2x2 (256 px) **0%** of
  held-out pixels are beyond one correlation length; at 4x4 (512 px) 22.3% are, with ~48 blocks
  per fold still left on Africa. 10x10 was sized for the global grid and leaves ~7 per fold here.
- **The two recalibration layers are keyed on different axes**, not simply redundant: conformal
  is `horizon × dhat × HM × biome`, width factors are `horizon × band × dhat × HM` out to 10 px.
  Unifying them means *measuring* which axis earns its place, on held-out data.

## Environment and scale

- Conda env **`spatio-temporal-dl`**, not base:
  `/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python`.
- **Screening is regional, but southern Africa is blind to the far field.** Southern Africa
  (1.86 Mpx) is a fast A/B *for near-field questions only*: its observed far-field change rate is
  0.0000, so any experiment aimed at what the model does beyond ~100 px from past change will
  read as "no effect" there whatever it actually did. **Screen far-field work on Africa**
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
  architecture* global hindcast and the baseline for the old-vs-new comparison. Root has
  ~90 GB free; southern-Africa ensembles at M=400 are ~3 GB
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
| **prove the chain before a long run** (every stage at M=4, all four block scales) | `./scripts/run_smoke_tier.sh <name>` | ~33 min on Africa |
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
    **This cost the project a whole phase.** Southern Africa's observed far-field change rate is
    0.0000, so "the ensemble emits zero change beyond 100 px" was recorded as a *win* there — and
    is a 44x under-prediction globally, where the observed rate is 0.0021 at h=20. All 22 model
    experiments were screened against that blind spot. A metric that reads 0 on the screening
    region cannot rank anything.
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
