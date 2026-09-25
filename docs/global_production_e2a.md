# The global production run — E2a, both products

**E2a is the production model (decided 2026-09-25).** See the last section for the decision
and the limitation it carries.

Run 2026-09-22/23 on branch `conv-spline`, immediately after the same run for E1v. Read
`CLAUDE.md` for the configuration and the rules, and `docs/global_production_e1v.md` for the
runner, the verifiers and the defects found building them — none of that is restated here.
This document is the E2a run and the **A/B against E1v on the learned tails alone**.

## What E2a is, and what it is not

    MODEL=E2a ALLOW_GLOBAL=1 ./scripts/run_global_model.sh <stage>

    --head_family pwl --free_scale True --mu_mse_weight 0.0

Head as built and as verified out of all six run logs: `family pwl, knots default14 (n=15,
bins=14), slopes learned, free scale, 15 params/horizon`. Against E1v's 17 the difference is
exactly `--isqf_tails True --isqf_space neglog` — two learned exponential tail rates on
`-log(1-HM)` support.

**This is not the E2a on the Africa promotion page.** That arm was scored at **16**
params/horizon because `--free_scale` did not actually remove the scale channel until
2026-09-16, so it carried a dead one. Its scorecard is not comparable to this run and must not
be quoted beside it. This run *is* the "E2a re-run on current code" that page said was needed:
same family, same free scale, differing from E1v only by the tails.

Published:
- E2a's detailed scorecard, base 2000 -> 2020: https://claude.ai/artifact/1dmSdZZcTB1jMDUyCXVY71
  (`data/conv_spline/scores/global/scorecard_detailed_g_E2a_hind.html`)
- the tails A/B, with the distance-band charts: https://claude.ai/artifact/36H1cgSmeHqdSPJABsH6oT
- E1v's detailed scorecard, for comparison: https://claude.ai/artifact/1ovxawvRr1JXyBqfrkserd

## The result: the tails trade far-field sharpness for far-field calibration

### Central accuracy is a tie

Out of sample on all 184,573,321 land pixels, base 2000 → 2005/2010/2015/2020.

| h | CRPS | crps_skill | E1v crps_skill | RMSE | central skill | cov95 | PIT KS |
|---|---|---|---|---|---|---|---|
| +5 | 0.002980 | 0.1542 | 0.1565 | 0.013336 | 0.0386 | 0.9528 | 0.0760 |
| +10 | 0.005284 | 0.2655 | 0.2656 | 0.020288 | 0.1906 | 0.9611 | 0.1227 |
| +15 | 0.007225 | 0.3018 | 0.3002 | 0.026386 | 0.2325 | 0.9478 | 0.1204 |
| +20 | 0.008905 | 0.3021 | 0.2980 | 0.031134 | 0.2434 | 0.8962 | 0.1147 |

Pooled `crps_skill` moves at most 0.0042 in either direction (E2a ahead at +15/+20, behind at
+5) and pooled RMSE is within 5e-05 at every horizon. **Two parameters per horizon buy nothing
in the body of the distribution.** Simplicity is a scoring criterion in this project, and on
the central columns alone the 15-parameter head wins it.

### The far field is where the tails earn or lose their keep, and it cuts both ways

At **+20 yr, distance band >100 px, 33,565,179 px**:

| | E1v (17 p, tails) | E2a (15 p, no tails) |
|---|---|---|
| CRPS skill | 0.2168 | **0.3003** |
| RMSE skill | −0.0070 | **+0.0310** |
| cov95 | 0.9950 | **0.7087** |
| P(u > 0.999) | 0.00155 | **0.12743** |

Read together, not separately:

- **E2a's far-field point forecast beats persistence, and E1v's does not.** `+0.0310` against
  `−0.0070` RMSE skill. Nothing in this project had previously beaten persistence out there on
  the point forecast — the E1v run's far-field win was explicitly CRPS-only and
  long-horizon-only.
- **E2a's far-field distribution is badly overconfident.** Its 95% interval covers **70.9%**
  of observations where 95% is nominal, and **12.7% of pixels fall above its 99.9th
  percentile — 127x nominal**, against E1v's 1.6x.

The mechanism is the same fact seen twice: without learned tail rates the far-field
distribution is much narrower. Narrow pays in CRPS and RMSE there, because change beyond
100 px is rare and small, and it destroys tail calibration and interval coverage.

**Neither model is calibrated in the far field. They fail in opposite directions** — E1v
conservative (cov95 0.995 where 0.95 is wanted), E2a overconfident (0.709). For a delivered
product carrying an uncertainty interval, over-wide is the safer error.

CRPS skill by distance band at h=20, both models:

| band | n | E1v | E2a | diff |
|---|---|---|---|---|
| 0–1 px | 25,293,071 | 0.4197 | 0.4212 | +0.0015 |
| 1–3 | 17,054,286 | 0.2752 | 0.2775 | +0.0023 |
| 3–10 | 36,038,668 | 0.1886 | 0.1920 | +0.0034 |
| 10–30 | 35,848,186 | 0.1140 | 0.1170 | +0.0030 |
| 30–100 | 36,773,931 | 0.0932 | 0.1191 | +0.0259 |
| **>100** | 33,565,179 | **0.2168** | **0.3003** | **+0.0835** |

E2a is ahead in every band, and the margin is concentrated where the width differs most.
`P(u>0.999)` by the same bands runs 0.00268 / 0.00450 / 0.00956 / 0.01282 / 0.03422 /
**0.12743** for E2a against 0.00022 / 0.00121 / 0.00709 / 0.00751 / 0.00554 / **0.00155** for
E1v — the miscalibration grows monotonically with distance, which the pooled number cannot
show.

### The pooled columns carry the same signal, diluted

`far_tail_excess_20` is **22.31 against E1v's 7.89**; `pit_gt_0999_20` is **0.0351** (35x
nominal) against 0.0044; `cov95_20` is 0.8962 against 0.9653. Rule 14 one level down: a pooled
`cov95` of 0.896 *is* a far-field 0.709 averaged against five better bands, and reporting only
the pooled figure would describe a mild narrowness rather than a band where three observations
in ten fall outside the interval.

### What is unchanged from E1v, and therefore is not about the tails

**The PIT defect is not fixed and is not the tails' doing.** PIT mean 0.509 / 0.560 / 0.553 /
0.552 against 0.50 (E1v 0.527 / 0.563 / 0.543 / 0.541), same right shift, and the detailed
scorecard's subsampled pass over 48,911,382 px reads 0.5607 against E1v's 0.5541. The
~0.25-spaced spike family is present in both. It was already known to be a property of the
head rather than of the globe; it is now also known not to be a property of the tails.

**Central coverage is still too narrow at the middle levels**, and the lower far tail is still
over-populated: `pit_lt_0001` 0.0095–0.0130 against 0.001, essentially E1v's 0.0114–0.0117.
The learned tails did not cause that and removing them did not fix it.

**The fence gates read high on both and remain unresolved at global scale**, for the reason
E1v's write-up gives: `degen` is 0.000 at three horizons and 0.029 at h=20, `0-gap` is 0.000,
and `clamp` is 0.689–0.784, so the sharpness is dominated by the physical HM=0 boundary that
most of the globe sits on. Rule 25 — rank the pathology, not the boundary; nobody has
separated them at global scale yet.

## The products

Both built and verified, at `/mnt/hdd1/spatio-temporal/data/conv_spline/products/E2a/`.

| | COGs | icechunk |
|---|---|---|
| hindcast (base 2000 → 2005/2010/2015/2020) | 20 files, 9.7 GB | 69 GB, `[4, 64, 17111, 40000]` uint16 |
| forecast (base 2020 → 2025/2030/2035/2040) | 20 files, 9.8 GB | 71 GB, same shape |

159 GB total (E1v: 157 GB). Chunks `(1, 64, 256, 320)`, shards `(1, 64, 1024, 1280)`, 2,650
files per store including manifests. Every COG `LAYOUT=COG`, single-band Float32, 7 overview
levels; per-year COGs 194–746 MB, `gt04` smallest and `mean`/`upper` largest, as on E1v.

`verify_products.py` green on both: passthrough bit-exact (`max |d| = 0`), the exceedance round
trip `Q(1-p) → x` within 3.2e-07 on every year of both products, **`code mismatches 0`** and
**`fill/finite collisions 0`**.

### One number that needs its own paragraph, because it reads like a regression and is not

Both stores report `round trip max |err| 1.53e-05` where E1v's logged export reported
7.66e-06 against a half step of 7.63e-06. **Nothing is wrong.** The decisive assertion in
`verify_products.py` is bitwise, not a tolerance — it recomputes
`min(rint(clip(q,0,1) * 65535), 65534)` in the source's own float32 and requires the store to
hold exactly that — and it reported zero mismatches. The printed figure is
decoded-versus-source, and it is **one full uint16 step** because a source of exactly 1.0
clamps to 65534 by design; the bound is `1.05 x` one full step and the code says so in place.

E1v's 7.66e-06 came from its export log written *before* the fill-value clamp existed: HM = 1.0
then encoded to 65535 and decoded back to exactly 1.0, so the error read as half a step and the
defect surfaced as a **collision** instead. `fill/finite collisions 0` here is that fix in
force. Two runs, two numbers, one encoding — and the smaller number is the broken one.

### The observed store, built 2026-09-25 to sit beside both products

    /mnt/hdd1/spatio-temporal/data/conv_spline/products/observed/observed_hm.icechunk   1.5 GB

Overall HM (`HM_{year}_AA_1000.tiff`, the file the scorer calls "observed"), one array `hm`
over **(year, latitude, longitude)**, years 1990/1995/2000/2005/2010/2015/2020, no percentile
axis. Written by `scripts/export_observed.py`, verified by `scripts/verify_observed.py`,
tests in `tests/test_export_observed.py`. Model-independent, so it lives beside `E2a/`, not in it.

The same as the prediction stores by construction: grid, latitude/longitude, `crs`, uint16
encoding, fill 65535 clamped to 65534, Blosc zstd-5, and the chunk (256 x 320) and shard
(1024 x 1280) FOOTPRINT are copied off the delivered hindcast store and then checked equal
against both. So every observed chunk covers exactly the map window of a prediction chunk. A
chunk is 160 KiB rather than 10 MB only because the 64-level axis is gone; 3,808 shard files.

Verified on every pixel of every year, not a sample: 184,610,103 px each, **0** code
mismatches, **0** fill-on-data, **0** data-where-missing. Range exactly [0, 1]; one pixel per
year at HM = 1.0, clamped to 65534.

**The nodata trap, measured.** The HM rasters declare nodata = 3.4e38 and do not fill with NaN.
The prediction exporter's `isfinite` test would have treated that as data, clipped it to 1.0
and written the ocean as fully modified. The read goes through the scorer's own
`_read_like_band`; the verifier's built-in control shows the naive encoding disagreeing with
the store on **33,025,297 px of the first 1024-row band alone**.

**Land counts differ between stores, and that is real.** Observed has 184,610,103 valid px per
year; the forecast rasters 184,608,551; the scored hindcast 184,573,321. A consumer pairing the
stores will find a few thousand to ~37 k observed pixels with no prediction behind them.

**A defect in the delivered prediction stores, found building this one: no array declares
`dimension_names`.** `xarray.open_zarr` refuses both with `Zarr object is missing the
dimension_names metadata`, measured on the E2a hindcast store. The dims are only implied by
shape. The observed store declares them and opens in xarray. The prediction stores are NOT yet
fixed: a fix is either a metadata-only commit to each store or a change to
`export_products.py`, which is hashed by the smoke receipt.

## Measured costs, end to end — this box, this run

| stage | wall clock | peak python RSS |
|---|---|---|
| smoke | 62.5 min | 35.6 GiB |
| hindcast, 5 folds train+predict (2 GPUs) | **313.6 min**; folds 103.5 / 103.8 / 103.8 / 105.8 / 106.8 | 36.1 GiB |
| stitch, 16 rasters | **197.9 min** | 35.9 GiB |
| forward model train+predict (1 GPU) | ~3 h 54 | 36.1 GiB |
| score, 4 window-years | **5 h 03 min** | **98.4 GiB** |
| export hindcast (20 COGs + icechunk) | ~2 h 20 | 26.6 GiB |
| export forecast | ~2 h 25 | 35.9 GiB |

Every figure is within a few minutes of E1v's, including the scorer's 99 GiB peak. Swap never
exceeded 2 GiB, and was 0 for the scorer.

Disk during the run: `g_E2a_hind` 308 GB, `g_E2a_fc` 129 GB, smoke 1.8 GB, products 159 GB —
598 GB, against E1v's ~590 + 157. HDD ended at 46% used, 2.0 TB free.

### Scheduling, and one thing worth keeping

The forward model was started on GPU 1 the moment fold 4 released it, so it overlapped fold 5
— E1v's pattern. **The stitch then ran concurrently with the forward model's prediction, and
the forecast export concurrently with the stitch.** Both pairs fit: 36 + 26 GiB against 125 GB
of DRAM, swap ≤ 2 GiB. The scorer ran alone, as rule 5 of the E1v write-up requires. Two
exports together were also fine (26 + 36 GiB); it was scorer-plus-exports that drove E1v into
swap, not exports alone.

That overlapping saved roughly four hours of a ~24 hour run and cost nothing measurable: the
stitch came in at 197.9 min against E1v's 197 while sharing the disk the whole way.

## A stale number in the E1v documentation, corrected here

`docs/global_production_e1v.md` and the `global-scale-defects` memory both say a fold's
prediction raster covers **67.4–71.6 M px**, 1.9x its own territory. E1v's own
`data/conv_spline/logs/global/hindcast_run.log` records **50,586,161 / 54,098,518 /
52,230,930 / 53,176,705 / 51,659,610**, and E2a produced those five figures **byte-identical**.

Identical is the correct expectation, not a measurement bug: the covered extent is fixed by the
fold mask and the prediction stride, so only the pixel *values* may differ between heads. The
1.9x figure appears to predate the current mask or stride; the calibrated range the check
actually carries — `[40 M, 120 M]` — is unaffected and passed all twenty rasters. The stitched
rasters then read 184,573,321 px each, the land exactly once, which settles it independently.

## Where this leaves the two models

The A/B the promotion page asked for is answered: **the learned tails cost nothing centrally
and buy far-field tail calibration at the price of far-field sharpness.** Which head is
preferable is a product decision rather than a measurement, and the measurement is now on the
table for it:

- If the deliverable is read as a point forecast with an interval attached, E2a is better where
  it has always been worst — it beats persistence beyond 100 px on RMSE — and its interval out
  there is not trustworthy.
- If the deliverable is read as a distribution, which is what this phase says it is ("that
  function is the entire product"), a band covering 70.9% at nominal 95% and exceeding its
  99.9th percentile 127x as often as it should is disqualifying, and on that reading E1v
  would be the product.

**Decision, 2026-09-25: E2a is the production model.** Chosen by the user with the trade-off
above on the table: the simpler head (15 parameters against 17), a far-field point forecast
that beats persistence, and centrally indistinguishable skill, accepting the far-field tail
miss. That miss is a **known limitation of the delivered product**, stated in the README, and
must not be described as calibrated: beyond 100 px at +20 yr the 95% interval covers 70.9% and
12.7% of pixels exceed the 99.9th percentile. The run was not set up to promote E2a; it was
set up as the A/B, and the decision was made on its result.
