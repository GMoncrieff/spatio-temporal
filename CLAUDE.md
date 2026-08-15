# Working notes for this repository

Spatiotemporal Human Modification (HM) forecasting: a ConvLSTM producing a central forecast
and 2.5/97.5 quantile intervals at +5/+10/+15/+20 yr, plus a post-hoc ensemble layer
(`src/ensemble/`) that adds spatial and temporal coherence on top of the per-pixel
marginals.

Active branch: **`ensemble`**. Read `docs/current_progress.md` for status and
`docs/central_field_baseline.md` for the measurement record — including the negative
results, which are load-bearing.

## Environment and scale

- Conda env **`spatio-temporal-dl`**, not base:
  `/home/glenn/miniforge3/envs/spatio-temporal-dl/bin/python`.
- **All iteration is on the southern-Africa subregion.** Global runs only on explicit
  instruction; the global k=5 hindcast already exists on disk.
- Two GPUs (RTX A5000, 24 GB each). Track experiments on **W&B**.
- Large global artifacts live on the HDD (`/mnt/hdd1/spatio-temporal/data`) behind symlinks
  in `data/ensemble/`. Root has ~230 GB free; ensembles at M=400 are ~3 GB each, so delete
  superseded ones (they regenerate from recorded seeds in ~3 min).

## The loops

| what | command | cost |
|---|---|---|
| one model experiment (train held-out folds, predict the region, stitch) | `./scripts/run_central_experiment.sh <name> <gpus> <folds> "<flags>"` | ~25 min/fold, 2 folds in parallel |
| Phase 1→4 for one experiment | `./scripts/run_region_loop.sh <name> <members>` | ~15 min at M=20, ~45 min at M=400 |
| central-field error, stratified | `scripts/diagnose_central_field.py --stitched_dir …` | seconds |
| compare configurations | `scripts/compare_central_runs.py label=dir …` | seconds |

`run_region_loop.sh` re-derives the recalibration, spectrum and AR(1) coupling from *that
model's own* residuals. Never carry them over between configurations.

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

## Conventions

- Flat argparse, no config framework. New flags are additive and default to today's
  behaviour; the boolean idiom is
  `type=lambda x: (str(x).lower()=='true'), nargs='?', const=True, default=…`.
- Tests are flat in `tests/`, pytest, `sys.path.insert(0, parent)` + absolute imports,
  synthetic tensors. 7 pre-existing failures in the legacy suite are stale (they assert a
  4-channel output from a model that emits 12) and fail identically with this branch stashed.
- Scripts write to `data/ensemble/exp/<name>/` so configurations never collide.
