# Working notes for this repository

Spatiotemporal Human Modification (HM) forecasting: a ConvLSTM producing a central forecast
and 2.5/97.5 quantile intervals at +5/+10/+15/+20 yr, plus a post-hoc ensemble layer
(`src/ensemble/`) that adds spatial and temporal coherence on top of the per-pixel
marginals.

Active branch: **`ensemble`**. Read `docs/current_progress.md` for status,
`docs/next_phase_model.md` for the current phase, and `docs/central_field_baseline.md` plus
`docs/next_phase_marginals.md` §5-6 for the measurement record — including the negative
results, which are load-bearing.

**The post-hoc marginal phase is closed** (scorecard 90/126 → 101/127, per-member 7/20 →
15/20, model untouched). The lever is exhausted structurally: T5.2 pins the marginal to the
published bounds, so any shape re-injects their width error. The current phase improves the
ConvLSTM itself.

## Environment and scale

- Conda env **`spatio-temporal-dl`**, not base:
  `/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python`.
- **All iteration is on the southern-Africa subregion.** Global runs only on explicit
  instruction; the global k=5 hindcast already exists on disk.
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
