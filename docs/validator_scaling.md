# Making the scorecard scale — the Africa OOM, and what it cost to fix

Status 2026-08-17, branch `ensemble`. Companion to `docs/current_progress.md`. This records
the diagnosis, the rewrite, the regression that proves the rewrite moved nothing, and the
storage switch to icechunk. The negative and surprising results are load-bearing and are
kept here for the same reason the rest of this project's are.

---

## 1. The failure

`scripts/validate_ensemble.py` was OOM-killed at ~77 GB RSS in stage T2 while scoring the
Africa hindcast ensemble (63.1 Mpx, 8112 × 7778, M=100). T5 and T1 had completed. The
obvious suspects — `compute_block_coverage`'s stripe budget, `zonal_member_stats`'s tables,
the zarr chunk size — had all been checked and were not it.

## 2. The diagnosis, from a control rather than an argument

Instrumentation first: `src/ensemble/memtrace.py` samples `/proc/self/statm` at 0.25 s on a
daemon thread and tags each sample with the innermost labelled section, exposed as
`--mem_trace`. This matters at more than the margin — **the process went from steady to
killed in 61 seconds**, so an external poll at 120 s could never have seen the peak, and the
earlier attempt to locate the stage by watching from outside was structurally unable to
succeed.

Run against the M=100 southern-Africa store, the trace put the peak in two places:

```
5.93 GB  aggregate/scores[2010,B=1]
5.67 GB  aggregate/block_member_stats_multi[2020]
2.52 GB  aggregate/scores[2010,B=10]        <- same run, same year, same members
2.52 GB  aggregate/scores[2010,B=100]
```

`src/ensemble/aggregate.py::block_member_stats` allocated

```python
sums = np.zeros((M, H // B, W // B), dtype=np.float64)
```

and `means = sums / np.maximum(cnt, 1)` allocated a second array of the same size while the
first was still live; `block_member_stats_multi` then made a third. The runs pass
`--block_sizes 1,10,100`, and **at B=1 the block grid is the pixel grid**:

| region | pixels | M | one array | demanded |
|---|---|---|---|---|
| southern Africa | 1.86 Mpx | 400 | 6.0 GB | ~18 GB — survived, unnoticed, for a year |
| Africa | 63.1 Mpx | 100 | 50.5 GB | ~101 GB — killed at 77 GB |
| global | 684 Mpx | 400 | 2.2 TB | not remotely |

Confirmed at scale before anything was changed: the unmodified code at HEAD, Africa extent,
M=50 (half the member count that died), one year, B=1 only, climbed **0 → 30.6 GB in 61 s**
monotonically before the watchdog stopped it. Doubling that to M=100 is the 101 GB above.

**The lesson worth keeping** is not "an array was too big". It is that a regional working
set can hide a quadratic: the same line of code, on the region all iteration happens on, is
6 GB and invisible. Nothing about southern Africa's numbers was wrong; they simply could not
reveal this.

## 3. The rewrite

Every statistic T2 computes — coverage, mean width, the Winkler interval score, CRPS — is a
count or a mean over blocks. None of them needs the blocks resident.

`aggregate.block_score_streaming` walks tiles whose dimensions are multiples of the coarsest
block size, so a block never straddles a tile and no carry-over is needed, and folds each
block row into nine scalars. The tile comes from `--mem_budget_gb`, split between the read
working set (`score_tile_shape`) and the scoring workspace (`score_batch_blocks`), which are
concurrent. The same treatment went through the rest of the scorecard:

| stage | was | now |
|---|---|---|
| T2 blocks | `(M, H/B, W/B)` float64 ×3 | streaming tiles, nine scalars per scale |
| T2 baseline | six `(n_bi, n_bj)` grids, re-read per scale | per-block-row accumulation, one call for all scales |
| T2 zonal | `(M, 65536)` per statistic | compacted to the ~164 zones present |
| T4 | `zs` = 4 × `(M, 2000, 2000)` float64 — **51 GB at M=400** | pooled cross-products, per-pixel Welford |
| T3 | full-raster `z`, plus a full-raster double `cumsum` | memmap-backed `z`, streamed patch weights |
| T3 sampling | whole horizon read once *per member* for ~1500 points | `members_at_points` reads the dozen tiles they fall in |
| T7 | four full-resolution rasters stacked for a figure | streaming co-moments, decimated render reads |
| T8 | six full rasters, members re-read once per band | one pass, a (band × member) counter table |

T4 deserves its own line: it would have OOM'd at M=400 **on southern Africa too**, the moment
T2 stopped failing first. It was never reached.

## 4. Two speed defects found on the way

- **`np.nanpercentile` on a 2-D array has no C fast path.** numpy falls back to
  `apply_along_axis`, a Python-level loop over the columns. Measured on `(400, 200000)`:
  two `nanpercentile` calls take 20.3 s where one `np.percentile(m, [2.5, 97.5], axis=0)`
  takes 1.51 s — **13.5×**, bit-identical. The block means are sums of finite pixels over a
  non-empty count and cannot be NaN, so the NaN handling was buying nothing. Left in place
  this would have made Africa's T2 about sixteen hours. (`stage_gates` had already found
  this and said so in a comment; T2 had not.)
- **`energy_score` materialised `(M, M, D)`** — 1.9 GB at M=400 on a 1500-point sample,
  7.7 GB at M=800. It is the only allocation in the scorecard that scaled with the *square*
  of the member count and answered to no budget. Now accumulated in row blocks;
  `variogram_score` got the same treatment over pairs.

## 5. The regression

The rewrite is exact arithmetic, so it has to reproduce the published numbers. Judged per
row, never on the total:

| control | rows | result |
|---|---|---|
| southern Africa M=100 vs `validation_m100` | 141 | 87/127, **0 pass/fail flips**, max rel diff 2.3e-14 |
| southern Africa M=400 vs `validation_hm` | 141 | **101/127**, **0 flips**, max rel diff 4.1e-14 |
| M=400 re-run after the T3 fixes | 141 | 101/127, 0 flips, max rel diff 4.1e-14 |
| same run at `--mem_budget_gb` 2 vs 8 | 141 | identical to the last bit (max rel diff 0.0) |

Peak RSS 5.7–8.4 GB against budgets of 2–8 GB, against 77 GB and dead before.

Two things fell out of this that are worth recording separately. The M=400 control needed
`null_hm400`, which had been deleted; regenerating it from the recorded seed reproduced
T3.2 and T3.3 to 1e-14, so **member seeds are deterministic across runs and independent of
M**. And the unit test in `tests/test_ensemble_aggregate.py` runs the streaming path against
the arrays-in-memory path it replaced, at two budgets, so the old functions are kept as the
reference the new one is checked against rather than deleted.

## 6. Storage: icechunk

Ensembles are icechunk repositories now (`*.icechunk`, one array `members`); the zarr read
path is gone. `open_ensemble` still returns `(array, attrs)`, so no consumer changed.
`icechunk 2.1.2` installs into `spatio-temporal-dl` from conda-forge on its own and works
with the zarr 3.1.3 already there.

The write is one transaction: `assign_members` already partitioned members on the chunk
boundary, so each GPU worker takes a `session.fork()`, writes its own members, and returns
the change record; the parent merges and commits once. A killed run now leaves no store
rather than a directory that looks complete and reads back as sentinel — which is exactly
what the OOM'd run would have left behind.

**The trap**: chunks were `(10, 1, 1024, 1024)` and a member is generated and written whole,
so each write was a read-modify-write of a ten-member chunk — and icechunk keeps every
version until garbage collection. The first M=400 null came to **16.7 GB against 2.9 GB**
for the identical array written in chunk-aligned blocks, and took **4× as long**. Plain zarr
hid this completely, because the last write simply overwrote the file. Chunks are
`(1, 1, 1024, 1024)` now, which also means two workers can never share one and reading a
single member stops decompressing nine others.

Old stores come across with `scripts/migrate_zarr_to_icechunk.py --verify`, which re-reads
both and asserts the arrays are byte-identical. That check earns its place: the migrated
M=400 store is what the validator rewrite was then checked against, and without it a
difference in the scorecard would have been ambiguous between "storage changed" and
"validator changed".

---

## 7. The Africa scorecard at M=400

Both ensembles regenerated at M=400 straight onto the HDD (85.7 GB + 87.7 GB, 10.4 + 6.8
min), nothing refitted. Scored in 190 min at a peak of 18.4 GB against `--mem_budget_gb 20`.
See `data/ensemble/exp/africa_k5/validation_m400/PROVENANCE.md` — the card is spliced from
the main run and a T3 re-run, because T3 hit a *second* latent scale bug: the summed-area
table over the spread raster was accumulated in float32, and over 63.1M pixels the
four-corner difference for a low-weight patch comes out negative, which `rng.choice`
rejects. In float64 it is fine, and southern Africa's eight T3 rows are bit-identical
before and after, so the fix moves nothing already published.

**Africa 111/133. Southern Africa 101/127.** Both at M=400, same model, same chain.

First, the member-count effect, measured directly on Africa rather than argued:

| | M=100 | M=400 |
|---|---|---|
| T5.1 median≡central | .9923 / .9976 / .9968 / .9921 → 2/4 | .9956 / .9989 / .9975 / .9935 → **3/4** |
| T1.1 pooled coverage | .9322 / .9250 / .9287 / .9261 → 0/4 | .9436 / .9383 / .9417 / .9388 → **2/4** |

Then the regional comparison, per test id. The headline count is the least interesting part
of it; the composition is where the information is.

**Africa is genuinely better where the members' own change distribution is tested.**
T6.5 (change quantiles vs observed) goes 3/8 → 7/8, with Africa's ratios 1.01–1.86 against
southern Africa's 1.11–5.41. T2.4 (ecoregion mean change 2005→2020) passes for the first
time.

**Several aggregate gates were under-powered on southern Africa and only become real tests
on Africa.** Southern Africa has **18** ecoregions, Africa has **158**. With 18 units the
coverage can only take values k/18, so 18/18 = 1.000 fails the ±0.05 tolerance and 17/18 =
0.944 passes — the metric was quantised too coarsely to land inside its own target. Africa's
T2.3 values sit at 0.981–0.994 and pass 7/8 against 3/8. The same applies to T2.5: southern
Africa returned chi2 p = 0.72 three times out of four (no power at n=18), while Africa's
rank histogram fails 2005 at p = 1.2e-6. **That is a stronger test failing, not a worse
model** — CLAUDE.md rule 5 in its natural habitat.

**And Africa exposes defects the smaller region could not show.** T8.4, spurious HM
*decreases* by distance-to-past-change band, passes 5/5 on southern Africa and fails the two
far bands on Africa at 7.8× and 123× the observed rate — there is not enough remote country
in southern Africa to populate those bands. T4.2 (spread non-decreasing in horizon) is 0.712
against 0.937. T6.1 (P(Δ < −0.01)) fails 4/4 against 1/4.

**Shared failures, essentially unmoved:** T3.2 against the independent-pixel null (0.031 vs
0.021, target ≥0.30) and T7.3 spread-skill at ecoregion scale (1.89 vs 1.58, target 1.0
±0.25). These are correlation-structure problems, and they are the same size in both
regions — the one place where iterating on southern Africa has not misled.

The prior expectation was that Africa would look better. On the count it does, 83.5% against
79.5%. But roughly half of that margin is southern Africa's aggregate metrics being too
coarse to pass rather than Africa's model being better, and Africa surfaces three defects
that were invisible at the smaller extent. The useful conclusion for the global decision is
narrower than the headline: **the central field and the marginals travel; the correlation
structure and the far-field change realism do not, and were being flattered by the region
they were tuned on.**
