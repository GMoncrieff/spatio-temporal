# The global production run — E1v, both products

**No longer the production model: E2a was chosen on 2026-09-25** (`docs/global_production_e2a.md`).
This document remains the reference for the runner, its verifiers and the defects found building it.

Started 2026-09-17, on branch `conv-spline`, after the screening phase closed and E1v was
promoted. Read `CLAUDE.md` for the configuration and the rules; this document is the state
and the reasoning for the global run itself.

## What is being produced

Two products, each in two formats. The spec is in the `global-export-products` memory and
implemented by `scripts/export_products.py`; it is not restated here.

| | base year | targets | model |
|---|---|---|---|
| **hindcast** | 2000 | 2005 / 2010 / 2015 / 2020 | k=5 fold mosaic, holdout, out of sample |
| **forecast** | 2020 | 2025 / 2030 / 2035 / 2040 | one model, `--train_all_splits`, no geography held out |

The hindcast years are not free. The model consumes a three-year window `(t-10, t-5, t)` and
HM exists for 1990–2020, so the earliest usable window is `(1990, 1995, 2000)` and the
earliest targets are 2005+. Only that one window is predicted: it is the delivered product,
and scoring it is scoring what is delivered.

## One runner: `scripts/run_global_model.sh`

    ALLOW_GLOBAL=1 ./scripts/run_global_model.sh <stage>
    stages: args | smoke | hindcast | stitch | score | forecast
            | export_hindcast | export_forecast | all

### Why it exists rather than the two scripts that were already there

`run_global_dist_hindcast.sh` and `run_global_dist_forecast.sh` source
`scripts/dist_base_args.sh`, whose `BASE_ARGS` hardcodes `--head_family spline` — the
29-parameter rational-quadratic head — plus `--seed 46` and e1's context flags. They carry
none of E1v's. Pointed at this phase they would have trained **e1**, logged like e1 and
scored like e1, and none of their verifiers could have said so: that file's
`verify_loss_weights` never reads `--mu_mse_weight` (E1v sets it to 0.0) and its
`verify_context_channels` greps for a *string* instead of reading the trunk's channel count
off the module. The new runner sources `conv_spline_base.sh`, where E1v's baseline lives,
and names the delta from it in exactly one place.

### What is verified before any result is believed

Per fold, and for the forward model, read back out of the run's own log:

| check | what it catches |
|---|---|
| `verify_loss_weights` | `run_hindcast_folds.py` injects the frozen product's `--ssim 0.2 --laplacian 0.3 --histogram 1.0` into every fold command. All four weights, `--mu_mse_weight` included. |
| `verify_context_wiring` | the **trunk's** channel count off the module against `EXPECT_CTX_CHANNELS=12`, and that no head consumes it |
| `verify_head_fingerprint` | **new.** family `pwl` **and** 17 params/horizon **and** `, tails neglog` **and** `, free scale`. 17 with the tails, 15 without — the count is the only thing separating E1v from a tailless arm, and `--isqf_tails` has already once been accepted, allocated and never read. |
| `verify_row_banding` | **new.** the *number* of rows, not merely that banding happened. See below. |
| `verify_weight_averaging` | the mean of the last 20 epochs, and prediction repointed at it |
| `check_qf_raster.py` | the written raster read back through the scorer's own reader: monotone, finite, gate levels present, and **the raster's own `head_family` / `head_params` tags** |
| `check_prediction_complete.py` | **new.** valid pixels against a two-sided bound, and BigTIFF where a raster needs it |
| `verify_products.py` | **new.** the delivered COGs and icechunk store against the rasters they came from |

Every one of these was proved to fire on a control before being trusted
(`tests/test_global_e1v_paths.py`, 32 tests).

### `--predict_row_chunk` is not a detail

Prediction holds `len(active_horizons) * (3 + 64)` full-window float32 accumulators — 268 on
a four-horizon window. On the 17111 x 40000 grid `BASE_ARGS`' Africa-sized 2048 rows is
0.305 GiB each, **81.8 GiB for one fold**, and two folds run at once against 125 GB of DRAM.
512 rows is 20.4 GiB. The runner appends 512 after `BASE_ARGS` and argparse keeps the last
occurrence; `verify_row_banding` reads the number back, because a check that only asks
whether banding happened passes the 2048 case.

## The smoke gate

`ALLOW_GLOBAL=1 ./scripts/run_global_model.sh smoke` — one epoch, one fold, one window, one
target year and a few hundred prediction blocks, **on the real global grid**, through every
stage: train, predict, `check_qf_raster`, stitch, score, export, verify products, and the
`--train_all_splits` forward path. It writes `smoke_ok.stamp`, and **every long stage refuses
to start without a receipt whose code hash matches the code on disk.** Change any of the
twelve files the hash covers and the receipt is void.

Nothing in the smoke is a measurement. What it proves is that the code paths exist and
survive global width — which is the failure mode this project actually meets: eight of the
twelve defects found on first contact with real data were invisible to a green test suite.

## Measured on this box, 2026-09-17 — not projected

From the smoke (one fold, one horizon, sparse):

| | measured |
|---|---|
| fold train (1 epoch) + predict, 1 horizon, 400 blocks | 12.7 min |
| stitch, one target year, global grid | 10.0 min |
| five COGs + one icechunk year, global grid | ~18 min |
| **peak python RSS, whole smoke** | **35.5 GiB — and it is the STITCH, not prediction** |
| prediction RSS at 67 accumulators | 11.5 GiB, so ~31 GiB at 268; two folds fit |
| land | 184.5 M px of the 684.4 M grid (measured, matches the 184.6 M constant) |
| icechunk shard plan on the real shape | chunks (1, 64, 256, 320) = 10.00 MB, shards (1, 64, 1024, 1280) = 160 MB, **544 shard files per year** |
| writing one all-NaN 64-band global raster | 12.0 min — the floor on any full-grid write |

The stitch peak is worth keeping: `stitch_fold_predictions` holds `out` and one fold's `vals`
at `(64, 1024, 40000)` float32, 10.5 GB each, plus the boolean-selected copy. It does **not**
grow with fold count — folds are read one at a time — so five folds peak where one did.

## Costs, projected from those measurements

- hindcast training: ~23 min/fold (13 steps/epoch at ~1.4 it/s, 150 epochs), two folds per GPU round
- hindcast prediction: ~100 min GPU + ~50 min of full-grid raster writes per fold
- **the forward model is the longest single job**: it predicts the whole globe from one
  model, where each hindcast fold predicts about a third of it
- the two products share nothing, so they can run at once on one GPU each

## Defects found building this, all of the family this project keeps meeting

1. **`head_family="spline"` was a literal in the qf raster's `update_tags`**, for every head,
   so E1v's own promoted Africa rasters are tagged `spline` while the head is `pwl`. Harmless
   to the scorer, which does not read it; wrong metadata on a delivered product. It now comes
   off the constructed module — the same place the banner reads its parameter count — with
   `head_params` beside it. The control is real: the existing E1v raster fails the new check.
2. **A completeness bound that could not be calibrated would have refused a good run.** The
   first version of the per-fold check expected "about a fifth of the land". Prediction is
   restricted to the tiles that *overlap* a fold and keeps every pixel of each, so a fold
   raster covers well over its own territory. **CORRECTED 2026-09-23:** the 67.4–71.6 M px
   figure below is wrong — this run's own `data/conv_spline/logs/global/hindcast_run.log`
   records 50,586,161 / 54,098,518 / 52,230,930 / 53,176,705 / 51,659,610, and the E2a global
   run reproduced those five byte-identically (identical is correct: the extent is fixed by the
   fold mask and the prediction stride, so only the values may differ between heads). The
   calibrated `[40 M, 120 M]` bound the check carries is unaffected. Original text: measured,
   67.4–71.6 M px against an
   own-land share of 35.8–38.4 M. It would have failed all five folds after nine hours. The
   check now takes an absolute range, and the bound that does the real work is the **ceiling**:
   a raster truncated at the classic-TIFF 4 GiB limit reads its unwritten rows back as finite
   zeros and reports close to the whole 684.4 M grid.

3. **`${SCORE_ROW_CHUNK:-512}` never fired, because `"0"` is not empty.**
   `conv_spline_base.sh` exports `SCORE_ROW_CHUNK=0` — Africa's 7778-wide rasters are
   affordable unbanded and banding cost 36% wall clock there — and `:-` substitutes only when
   a variable is *unset or empty*. The global scorer therefore ran unbanded and died on
   `Unable to allocate 163. GiB for an array with shape (64, 17111, 40000)`. **The smoke
   caught it**; without the smoke it would have appeared three hours into the scoring stage
   of a nine-hour run. A global default needs its own name: `GLOBAL_SCORE_ROW_CHUNK`. This is
   rule 16 again — "the flags I passed" is not "the flags that took effect", and here the
   flag that took effect came from a file the runner sourced.

4. **The forward product's triple rasters declared a nodata value they do not use.**
   `train_lightning.py`'s prediction writer starts from the HM reference raster's profile,
   which carries `nodata=3.4e38`, updates the dtype to float32 — and never updates the
   nodata. The quantile raster overrides it explicitly; the triple does not. So
   `prediction_<year>_{lower,central,upper}_blended.tif` say `3.4e38` while filling unwritten
   pixels with **NaN**, and `gdal_translate` would have carried that straight into the
   delivered COGs, where a consumer masking on the declared nodata masks nothing and reads
   NaN as data.

   Only the forward product is exposed: `stitch_fold_predictions` sets `nodata=np.nan`
   explicitly, so every hindcast deliverable is correct. That is global-scale defect 2's shape
   again — *a code path only the never-yet-run branch reaches is untested by construction*.

   Caught by `check_prediction_complete.py`, which reported 370.8% of land: with a sentinel
   nodata the check counts `a != nodata`, and `NaN != 3.4e38` is true, so every pixel of the
   684.4 M grid counted. **It failed closed and named the right rasters.** Fixed in place with
   a tag-only rewrite (no pixel touched); all sixteen then read 184,608,551 px = 100.0% of
   land.

   **Two follow-ups are owed and are NOT yet done**, because both files are hashed by the
   smoke receipt and editing them mid-run would block the remaining stages:
   - `train_lightning.py` must set `nodata=np.nan` on the triple's profile, as it already does
     for the quantile raster.
   - `check_prediction_complete.py` must count a float raster's valid pixels as
     `isfinite(a) & (a != nodata)`, not one or the other.
   Do both, add the controls, and **re-smoke** before the next global run.

5. **The scorer's row loop held two window-years at once — 170+ GiB — and OOMed.**
   `u, cell, pp, cons = load_row(...)` rebinds those names only when the call *returns*, so
   row 2's compacted per-pixel arrays are built while row 1's are still referenced.
   **`--row_chunk` does not help**: it bounds the transient band reads, not the compacted
   arrays, which are the bulk. MEASURED on the global grid: one window-year of 184,573,321 px
   peaks at **92.4 GiB**, so row 2 ran past 170 GiB against 125 GB of DRAM. On Africa the
   same row is ~29 GiB and two fit, which is why twenty-three screening runs never saw it.

   Fixed by releasing the row's arrays at the end of the loop body. **Confirmed on the real
   run**: RSS fell 92.4 -> 50.4 GiB at the row-1/row-2 boundary, where before it would have
   kept climbing. `tests/test_score_row_release.py` holds a weak reference to the previous
   row's `observed` array and fails if it is alive when the next `load_row` is entered —
   proved to fire by reverting the `del`.

   **Operational consequence: the scorer needs the box to itself.** Three jobs sharing 125 GB
   (scorer + two exports) drove swap to 6 of 7 GB, which is what exposed this in the first
   place. Run `score` alone.

6. **A peak that belonged to a different run.** `monitor_resources.sh` *appends*, so one CSV
   per stage accumulates every invocation of it, and `stop_monitor` took the max over the
   whole file. The hindcast export reported **99.0 GiB**; it actually peaked at **24.6 GiB**,
   and the 99.0 was the scorer, sampled hours earlier by an aborted run of the same stage
   whose monitor was still alive. Two compounding faults: the summary did not bound itself to
   its own invocation, and killing a stage externally left its monitor running (ten hours, in
   this case). Fixed by recording the start timestamp and filtering on it, plus an
   `EXIT INT TERM` trap that takes the monitor down with the stage. Rule 25 in the place it
   is easiest to believe a wrong number: a resource measurement nobody cross-checks.

7. **Real data written as "missing": the uint16 encoding collided with its own fill value.**
   `write_icechunk` did `rint(clip(q, 0, 1) * 65535)` while declaring `fill_value = 65535`,
   so **any HM >= 0.99999237 encoded to exactly the fill value** and read back as nodata. The
   store's attributes already said `valid_range [0, 65534]` and "the representable maximum is
   65534/65535" — the contract was right and the code never implemented it, which is rule 2's
   shape: a predicate written twice disagreed with itself.

   It concentrates at high percentiles because E1v's upper tail is exponential on
   `-log(1-HM)` support, so the far upper quantiles saturate at HM = 1 over a large share of
   land, and more so at +40 yr than +20. MEASURED globally on every land pixel:

   | | 99.99th pct | band 63 | band 62 | band 61 |
   |---|---|---|---|---|
   | forecast 2040 | **31.73%** (58.6 M px) | 22.62% | 9.10% | 4.45% |
   | hindcast 2020 | **24.46%** (45.2 M px) | 16.85% | 10.95% | 3.86% |

   **Found by the user inspecting the delivered store, not by any check of mine.**
   `verify_products.py` compared the decoded value against the source and passed: 65535 x
   1/65535 decodes to 1.0 and equals a source of 1.0, so the *value* was right and only its
   *meaning* was wrong. A round trip cannot see a fill-value collision. The check now also
   asserts the other half of the encoding — a cell may hold the fill value only where the
   source is not finite — and was proved on the broken store (2,464 collisions in a
   4,000-pixel sample, with every other check still green).

   **Do not sample-and-extrapolate for a number like this.** Three random-window estimates
   gave 0.14%, 46% and 0.31% before the global count settled it at 31.7%: the quantity varies
   with how much heavily-modified land a window holds, so windows are the wrong instrument.
   The source rasters are band-interleaved, so counting one percentile level over the whole
   globe costs a 2.7 GB read — cheaper than the guessing.

   Fixed by clamping the code to 65534. Both stores re-exported; the COGs are float32 and
   were never affected.

## RUN COMPLETE — 2026-09-18

Both products are built, verified and on disk under
`/mnt/hdd1/spatio-temporal/data/conv_spline/products/`.

| | COGs | icechunk |
|---|---|---|
| hindcast (base 2000 -> 2005/2010/2015/2020) | 20 files, 11 GB | 67 GB, `[4, 64, 17111, 40000]` uint16 |
| forecast (base 2020 -> 2025/2030/2035/2040) | 20 files, 11 GB | 69 GB, `[4, 64, 17111, 40000]` uint16 |

157 GB total. Every COG verified `LAYOUT=COG`, single-band Float32, 7 overview levels; both
stores at chunks `(1, 64, 256, 320)` and shards `(1, 64, 1024, 1280)` = 2,176 shards, uint16
round trip 7.66e-06 against a half step of 7.63e-06; passthrough bit-exact and the exceedance
round trip `Q(1-p) -> x` within 2e-07 on every year of both products.

### The scorecard — out of sample, all 184,573,321 land pixels

`data/conv_spline/scores/global/scorecard_g_E1v_hind.html`

| h | CRPS | crps_skill | central skill | cov95 | PIT KS |
|---|---|---|---|---|---|
| +5 | 0.002972 | 0.1565 | 0.0424 | 0.9529 | 0.0735 |
| +10 | 0.005284 | 0.2656 | 0.1914 | 0.9535 | 0.1379 |
| +15 | 0.007242 | 0.3002 | 0.2342 | 0.9253 | 0.0997 |
| +20 | 0.008958 | 0.2980 | 0.2414 | 0.9653 | 0.0846 |

`crps_skill` by distance band at h=20: 0.420 / 0.275 / 0.189 / 0.114 / 0.093 /
**0.217 beyond 100 px**. b1 on Africa ran +0.395 / +0.293 / +0.223 / +0.137 / **-0.106** /
**-1.510** over the same bands.

**The far field, stated precisely.** At **+20 yr** the >100 px band's CRPS skill is
**+0.217** — a real reversal of b1's −1.510 on Africa. But the same band's **RMSE skill is
−0.0070**, and at **+5 yr** its CRPS skill is **−0.136**. So the far-field win is
*distributional and long-horizon*: the predicted spread is better than persistence's at
+20 yr, while the point forecast E[Q] still is not, and at +5 yr neither is. The band's PIT
mean is 0.622 against 0.50, the worst centring on the card. "The far field no longer loses to
persistence" is true only of CRPS at +20 yr and must not be quoted unqualified.

The detailed scorecard (`scripts/build_detailed_scorecard.py`,
`data/conv_spline/scores/global/scorecard_detailed_g_E1v_hind.html`) is where the qualifier
came from: it prints RMSE skill and PIT mean beside CRPS skill per band, and the pooled
CRPS-skill row alone does not show that the point forecast is still not beating persistence
out there. Rule 14's shape, one level down — a *band* average hid a defect inside the band.

`qf vs published triple, max |diff|: 0.00e+00`. `E[Q]` outside its own 95% interval on 4.39%
of pixels. Exceedance `mean|log10(pred/obs)|` over distance bands 0.2294.

### What did NOT get fixed, and must not be read as fixed

**The PIT structure defect survives.** PIT mean 0.527 / 0.563 / 0.543 / 0.541 against 0.50,
with mass visibly heavy right of centre — the model is centred too low. Stable spikes at
~0.22 / ~0.52 / ~0.77 / ~0.92, spacing ~0.25, at every horizon. **E1v's Africa scorecard has
the same spike family and the same right shift** (0.543 / 0.502 / 0.531 / 0.525), so this is a
property of the head, not of the globe. The second of the two defects the conv-spline phase
opened with is still open.

Do not read the smaller global `pit_ks` (0.074-0.138) against Africa's (0.107-0.196) as an
improvement: the strata weights and n differ. The histogram shape is the honest comparison and
it is the same shape.

**The fence gates read very high and need interpreting, not quoting.** needle p50 0.786 at
h=5, `max_density_p99` 748,671 against a 578 target, `over_f_max` 0.838. But beside them
`degen = 0.000`, `0-gap = 0.000` and `clamp = 0.988`. By the scorecard's own legend that says
the sharpness is the physical HM=0 boundary — most of the globe is wilderness sitting on it —
rather than the monotonicity artefact. Rule 25 says rank the pathology, not the boundary;
nobody has yet separated them at global scale, so these numbers are **unresolved**, not good
and not bad.

### Measured costs, end to end

| stage | wall clock | peak RSS |
|---|---|---|
| smoke | ~57 min | 36.8 GiB |
| hindcast, 5 folds train+predict (2 GPUs) | 106.5 / 109.7 / 108.7 / 108.2 / ~110 min | ~31 GiB per fold |
| stitch, 16 rasters | 197 min | 35.9 GiB |
| score, 4 window-years | 5 h 04 min | 99 GiB (92-93 per row) |
| forward model train+predict (1 GPU, concurrent with folds 4-5) | ~3 h 50 min | ~31 GiB |
| export per product (20 COGs + icechunk) | ~2 h | 24.6 GiB |

The hindcast and the forward model ran concurrently on one GPU each, which is what the
runner's header documents. **The scorer must run alone**: three jobs sharing 125 GB drove
swap to 6 of 7 GB and is what exposed defect 5.

### Owed before the next global run

The last fix (defect 6) changed `run_global_model.sh`, so **the smoke receipt is void** — the
runner refuses every long stage until `smoke` is run again, which is correct and was verified.
Nothing in this run is affected; all seven stages completed before the edit.

## Decisions taken

- **All five folds are trained fresh**, though E1v's folds 1 and 2 already exist as
  global-trained checkpoints in `models/production/E1v/` (training was never restricted to
  Africa — only `--predict_region` was). Reuse would have saved ~25 min of a multi-hour run
  and kept folds 1 and 2 bit-identical to the promoted checkpoints; one code path and five
  folds trained identically on current code was preferred. Decided by the user, 2026-09-17.
- **Only window 2000 is predicted for the hindcast**, not all four. It is the delivered
  product; the other three windows would quadruple prediction and stitching for years that
  are not shipped.
- **Nothing is reprojected.** The source grid is already EPSG:4326, 40000 x 17111 at 0.009°,
  N 83.9970 / S -70.0020 — the requested 84.00 / -70.00 box to within a third of a pixel.
