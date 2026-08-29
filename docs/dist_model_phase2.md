# The distributional model, round 2: southern Africa, then Africa

Branch `dist-convlstm`. Written 2026-08-28, from
`data/ensemble/exp/dist_scores/summary_*.json` — every number below is generated from those
files by `scripts/gen_dist_phase2_doc.py`, so the prose and the tables cannot drift apart.
Regenerate it after any rescore; do not hand-edit the numbers.

**What is being tested.** A ConvLSTM whose head emits a full per-pixel quantile function
`Q_h(u|x)` — a monotone rational-quadratic spline on fixed tail-dense knots, anchored at
persistence, trained on CRPS. The quantile function *is* the product: no conformal scaling, no
width factors, no empirical marginal, no horizon-monotonicity pass. This document covers the
second round of experiments on it.

**What one run is.** Two folds (1 and 2) of the k=5 checkerboard mask, trained on global chips,
predicted and stitched over the region, scored on held-out fold territory only. A *variant* is
three such runs differing only in seed; the *floor* is six.

---

## Part 1 — Southern Africa: what the screen taught us, and why we left it

### 1.1 Two mechanical bugs explained round 1's "instability"

Round 1 concluded that the model was too unstable to rank anything: `tail_reach20` varied by
**4.3 within one configuration**. Both causes turned out to be mechanical.

**A `sqrt` derivative singularity silently killed 6 of 16 runs.** `quantile_spline.cdf` took
`sqrt(disc)` with `disc` floored at zero. `sqrt` is finite at zero and its derivative is not, and
a masked-out pixel supplies exactly-zero incoming gradient, so `0/0 = NaN`. **One pixel in
131,072** poisoned 38 of 94 parameter tensors in a single step — permanently, with a finite loss
and an ordinary gradient norm. It was invisible because `ModelCheckpoint` never selects a NaN
epoch: a dead run kept its last healthy checkpoint, finished, and **scored normally**. Fixed by
evaluating `sqrt` away from the singularity and selecting afterwards; bit-identical forward,
verified 38 → 0, and now guarded by `--abort_on_nonfinite` at three points.

Audited by grepping every fold log for `train_loss=nan`:

| run | flags | dead fold | verdict on record, now void |
|---|---|---|---|
| `d2` | `--chip_sampling stratified --chip_sampling_correct True` | fold 2, NaN 149/149 epochs | "tail destroyed, CRPS worse at every horizon" |
| `d8` | `--spline_slopes fritsch` | fold 2, NaN 84 (died ~epoch 65) | "clears pit_ks5 by 2.3x; tail worse by 1.4x the band" |
| `d1a` | `--crps_tail_weight 1.0` | fold 2, NaN 31 (died ~118) | the round's only apparent winner |
| `d1a_s43` | replicate of the above | fold 2, NaN from epoch 0 | supplied the 4.3 spread |
| `d3` | `--ssim_weight 0.2` | fold 2, NaN 30 | "SSIM does not earn its place" |
| `d6` | `--mu_mse_weight 0.0` | fold 2, NaN 41 | "pure CRPS costs nothing centrally" |
| `d7` | `--isolate_shape_grad True` | fold 1 NaN 108, fold 2 NaN 4 | "the trunk must hear the objective" |

Both folds are scored, so each of those verdicts was read off a half-dead raster. Clean:
`d0_s42–44`, `d1b`, `d2u`, `d4`, `n1`. **Round 1's two surviving findings — `d6` and `d7` — are
both on that void list** and are owed a redo.

**`val_crps` is flat to 1.0–2.4% across 150 epochs**, so checkpoint selection was a lottery: it
once picked epoch 19 of 150, and `r(selected epoch, tail_reach20) = +0.773` on clean runs. Round
1 had recorded that correlation as 0.150 and rejected the hypothesis; that measurement was made
on NaN-contaminated runs.

Two consequences for the record: round 2's design spec rests on artifacts of these bugs (the 4.3
spread is 1.52 without the epoch-4 death), and the `r = 0.691` correlation between `tail_reach20`
and `pit_ks5` that was written up as "two metrics are one axis" falls to +0.236 without the two
dead runs and is **−0.964** on six clean ones.

### 1.2 The stability gate could not work, and was never run

It proposed judging four configurations on the *spread* of three seeds. Measured on a six-seed
floor, a three-seed **range** varies by 17.5x across draws from one configuration (0.121 → 2.117)
while the median varies by 1.1x. A range is not a statistic. The gate was skipped and its lever
found directly:

- **`--weight_avg_last 20` — ADOPTED.** Beat the southern-Africa floor on all eight metrics and
  cut the `tail_reach20` spread 2.117 → 0.263. Carried by every Africa run since.
- **`--checkpoint_select final` — REJECTED.** Buys tail reach and drives `skill20` negative in
  2 of 3 seeds (`fin_s45` −0.0475, `fin_s46` −0.0077).
- `--lr_schedule cosine` — still untested.

### 1.3 Every southern-Africa run

Numbers here are **not comparable to Part 2**: southern Africa changes 2–4x less than Africa at
every HM level and its `[0,0.01)` stratum is 6% of the region against 40% of Africa, so the
skill denominators differ. Rows marked void above are included for completeness.

| run | crps_skill5 | crps_skill20 | pit_ks5 | skill5 | skill20 | tail_reach20 |
|---|---|---|---|---|---|---|
| `d0_s42` | 0.1274 | 0.1078 | 0.2594 | 0.0357 | 0.0110 | 4.16 |
| `d0_s43` | 0.1395 | 0.1300 | 0.2688 | 0.0527 | 0.0647 | 4.33 |
| `d0_s44` | 0.1159 | 0.1650 | 0.2737 | 0.0208 | 0.1032 | 5.02 |
| `d1a` | 0.1476 | 0.1565 | 0.2612 | 0.0580 | 0.0861 | 4.41 |
| `d1a_s43` | 0.0738 | -0.0665 | 0.2241 | 0.0387 | 0.0328 | 1.61 |
| `d1a_s44` | 0.1226 | 0.1531 | 0.2147 | 0.0307 | 0.0928 | 5.93 |
| `d1b` | 0.1351 | 0.1209 | 0.2762 | 0.0424 | 0.0311 | 4.81 |
| `d2` | 0.0991 | 0.0374 | 0.2009 | 0.0520 | 0.0760 | 1.63 |
| `d2u` | 0.1252 | 0.0870 | 0.2602 | 0.0226 | -0.0109 | 5.46 |
| `d3` | 0.1396 | 0.1447 | 0.2562 | 0.0594 | 0.0756 | 4.03 |
| `d4` | 0.1395 | 0.1554 | 0.2447 | 0.0539 | 0.0751 | 2.70 |
| `d6` | 0.1342 | 0.1276 | 0.2865 | 0.0431 | 0.0460 | 5.22 |
| `d7` | 0.0591 | 0.0782 | 0.3793 | 0.0514 | 0.1109 | 4.96 |
| `d8` | 0.1308 | 0.1380 | 0.2264 | 0.0464 | 0.0594 | 2.98 |
| `e1_s45` | 0.1346 | 0.1293 | 0.2413 | 0.0586 | 0.1113 | 5.26 |
| `e1_s46` | 0.1297 | 0.0995 | 0.2695 | 0.0372 | -0.0044 | 5.25 |
| `e1a_s45` | 0.1335 | 0.1242 | 0.2233 | 0.0440 | 0.0601 | 5.66 |
| `e1a_s46` | 0.1404 | 0.1263 | 0.2788 | 0.0553 | 0.0721 | 5.65 |
| `e2_s45` | 0.1390 | 0.1354 | 0.2621 | 0.0515 | 0.0558 | 6.06 |
| `e2_s46` | 0.1389 | 0.1497 | 0.2421 | 0.0505 | 0.0617 | 6.02 |
| `e4_s45` | 0.1345 | 0.1101 | 0.2737 | 0.0495 | 0.0512 | 6.80 |
| `e4_s46` | 0.1395 | 0.1502 | 0.2532 | 0.0545 | 0.0794 | 7.01 |
| `e4_s47` | 0.1374 | 0.1502 | 0.2359 | 0.0497 | 0.0775 | 6.52 |
| `e5_s45` | 0.1350 | 0.1239 | 0.2585 | 0.0514 | 0.0483 | 6.22 |
| `e5_s46` | 0.1349 | 0.1357 | 0.2483 | 0.0545 | 0.0797 | 5.72 |
| `e6_s45` | 0.1284 | 0.1175 | 0.2772 | 0.0449 | 0.0355 | 7.65 |
| `e6_s46` | 0.1341 | 0.1169 | 0.2905 | 0.0454 | 0.0371 | 5.60 |
| `e7_s45` | 0.1362 | 0.1350 | 0.2355 | 0.0615 | 0.0787 | 5.74 |
| `e7_s46` | 0.1389 | 0.1546 | 0.2439 | 0.0517 | 0.0820 | 6.90 |
| `f0_s42` | 0.1226 | 0.1333 | 0.2633 | 0.0389 | 0.0662 | 5.55 |
| `f0_s43` | 0.1338 | 0.1216 | 0.2818 | 0.0515 | 0.0557 | 5.28 |
| `f0_s44` | 0.1278 | 0.1395 | 0.2601 | 0.0405 | 0.0830 | 5.40 |
| `f0_s45` | 0.1357 | 0.1772 | 0.3293 | 0.0501 | 0.1150 | 3.43 |
| `f0_s46` | 0.1405 | 0.1385 | 0.2788 | 0.0555 | 0.0732 | 5.33 |
| `f0_s47` | 0.1148 | 0.1137 | 0.2703 | 0.0408 | 0.0309 | 5.55 |
| `fin_s45` | 0.1285 | 0.1097 | 0.2681 | 0.0296 | -0.0475 | 7.38 |
| `fin_s46` | 0.1180 | 0.1219 | 0.2660 | 0.0340 | -0.0077 | 6.53 |
| `fin_s47` | 0.1268 | 0.1746 | 0.2477 | 0.0422 | 0.1132 | 6.20 |
| `n1` | -11.1449 | -6.1778 | 0.9812 | -8.4828 | -6.7557 | 1.00 |
| `wa_s45` | 0.1387 | 0.1317 | 0.2649 | 0.0535 | 0.0803 | 5.54 |
| `wa_s46` | 0.1383 | 0.1468 | 0.2617 | 0.0548 | 0.0816 | 5.76 |
| `wa_s47` | 0.1388 | 0.1669 | 0.2448 | 0.0506 | 0.0965 | 5.81 |

`n1` is `--dist_loss nll`: the log score collapses the distribution to a point mass at the width
floor (`tail_reach20` exactly 1.0, `crps_skill5` −11.1). It is the mirror of the incumbent's
Gaussian-NLL failure, which inflated widths 7x, and it is why CRPS is the objective.

### 1.4 Why the region had to be abandoned

**Southern Africa's far-field band contains zero pixels.** Not a zero rate — an empty band. The
model exists to fix one named defect: the far field emits 3.4% of the observed rate of new
development beyond 100 px. That defect is *unmeasurable* at this extent, so every one of the 22
model experiments screened here was blind to the thing they were for. Africa's >100 px band holds
**175,478 px** at h=20, and the distributional model's first measurement there is
`P(ΔHM>0.05)` predicted 0.000010 against observed 0.000137 — **7.3% of observed**, against the
frozen product's 3.4%. Still 13.7x short, but measurable at last.

**Verdicts do not transfer between the regions.** Measured, not assumed:

| variant | southern Africa | Africa |
|---|---|---|
| `e1` / `e1a` (covariate) | worse on 7 of 8; monotone `floor > e1a > e1` | **inverts**: `e1 > e1a > floor` on every skill metric |
| `e4` (lower-tail weight) | the round's only winner, replicated at n=3 | **null at n=3**; the headline inverted at n=1 and then vanished |
| `e5` (horizon weights) | worse on everything | worse on everything — the one verdict that transferred |
| `e2` (body knots) | neutral | neutral |
| `e6` (deeper head) | the slate's worst, 0 better / 7 worse | null, one weak-row cost |

A regional screen ranked some of these backwards. That is the phase's most expensive lesson.

---

## Part 2 — The Africa programme

**The screen.** `--predict_subsample_blocks 200 --predict_subsample_seed 7` predicts 200 of 965
candidate 128 px blocks, intersected into the fold mask so every tile overlapping a kept pixel is
still processed. Blended values on kept pixels match a full-density run to float32 (measured
1.3e-6). Back-tested on existing rasters: at 41% of area, r = 0.990 on `tail_reach20` with 97%
ranking agreement. Every run in Part 3 uses the same blocks, so the comparisons are paired.

**The floor.** Six seeds of the identical configuration. Anything inside this range is seed
noise, not a result:

| metric | floor min | floor max | width | width / level |
|---|---|---|---|---|
| `crps_skill5` | 0.2082 | 0.2130 | 0.0049 | 2% |
| `crps_skill10` | 0.2461 | 0.2581 | 0.0120 | 5% |
| `crps_skill15` | 0.2645 | 0.2818 | 0.0173 | 6% |
| `crps_skill20` | 0.2656 | 0.2785 | 0.0130 | 5% |
| `skill5` | 0.1328 | 0.1408 | 0.0080 | 6% |
| `skill10` | 0.1868 | 0.2088 | 0.0219 | 11% |
| `skill15` | 0.2154 | 0.2446 | 0.0292 | 12% |
| `skill20` | 0.2200 | 0.2468 | 0.0268 | 11% |
| `pit_ks5` | 0.1688 | 0.1963 | 0.0275 | 15% |
| `pit_ks10` | 0.1009 | 0.1816 | 0.0807 | 66% |
| `pit_ks15` | 0.0643 | 0.1066 | 0.0423 | 48% |
| `pit_ks20` | 0.0762 | 0.1198 | 0.0435 | 42% |
| `cov95_5` | 0.9408 | 0.9557 | 0.0149 | 2% |
| `cov95_10` | 0.9540 | 0.9590 | 0.0050 | 1% |
| `cov95_15` | 0.9499 | 0.9534 | 0.0035 | 0% |
| `cov95_20` | 0.9429 | 0.9515 | 0.0086 | 1% |
| `rmse5` | 0.01333 | 0.01339 | 0.00006 | 0% |
| `rmse10` | 0.02079 | 0.02107 | 0.00028 | 1% |
| `rmse15` | 0.02657 | 0.02708 | 0.00051 | 2% |
| `rmse20` | 0.03120 | 0.03175 | 0.00055 | 2% |
| `tail_reach5` | 4.54 | 6.20 | 1.65 | 32% |
| `tail_reach10` | 4.13 | 5.24 | 1.11 | 23% |
| `tail_reach15` | 4.75 | 5.96 | 1.21 | 23% |
| `tail_reach20` | 5.23 | 6.67 | 1.45 | 23% |
| `exceedance_abs_log10` | 0.3361 | 0.3925 | 0.0564 | 16% |

Two rows deserve care. `tail_reach20`'s width is 23% of its level and its spread is *model*
variance rather than sampling error (between-seed 0.860 against within-seed 0.369 across four
windows of one model), so it does not tighten with more pixels. The `pit_ks` rows at h ≥ 10 run
42–66% of their own level and are marked **under-powered**: a null there means "cannot see",
not "no effect".

**The bar.** A variant must beat the floor's range by **more than the range's own width** — a
single new draw falls outside a three-run range about half the time under the null, so "outside
the range" is barely a test. Variants are ranked on the **median** of their three seeds, never on
their range.

*One correction to the instrument, made during this phase:* the WORSE half of that test was
measured from the floor's **best** edge, so it fired the instant a value left the range while
"better" needed a full extra width — two halves of one test at bars one width apart. Seven of the
eight WORSE verdicts on this slate sat 0.04–0.78 widths past the floor and evaporate under the
symmetric rule. Any "N better / M worse" count recorded before 2026-08-27 was scored on the lax
half.

**How to read Part 3.** Each model gives the median across its seeds for every metric at every
horizon, then the list of metrics where **all** seeds fall outside the floor's whole range. That
unanimity count is the honest signal in this phase: the margins are mostly small, and three
independent draws landing on the same side of a six-seed range is roughly a 1-in-343 event under
the null.

---

## Part 3 — Every Africa model


### afw — the floor

**Changed from the floor:** *(the comparator; six seeds)*
**Seeds:** 42, 43, 44, 45, 46, 47 (n=6)

The shipped distributional configuration: spline head, CRPS alone, `--central_residual`, `--central_context`, `--monotone_quantile_width`, `--quantile_context`, `--checkpoint_monitor val_crps`, `--weight_avg_last 20`, `--spline_knots default14`, learned slopes, cumulative width. Six seeds (42–47), identical in every other respect.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00346 | 0.2115 | 0.01334 | 0.1399 | 0.1806 | 0.00287 | 0.9513 | 5.27 |
| +10 yr | 0.00585 | 0.2549 | 0.02084 | 0.2044 | 0.1106 | 0.00253 | 0.9564 | 5.00 |
| +15 yr | 0.00794 | 0.2753 | 0.02668 | 0.2383 | 0.0886 | 0.00277 | 0.9516 | 5.24 |
| +20 yr | 0.00961 | 0.2747 | 0.03133 | 0.2407 | 0.1065 | 0.00154 | 0.9454 | 6.37 |

`exceedance_abs_log10` 0.3558 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2718 / 0.2759 / 0.2785 / 0.2781 / 0.2734 / 0.2656; `skill20` 0.2380 / 0.2426 / 0.2443 / 0.2468 / 0.2387 / 0.2200; `pit_ks5` 0.1930 / 0.1963 / 0.1775 / 0.1837 / 0.1723 / 0.1688; `tail_reach20` 6.26 / 6.67 / 5.23 / 6.43 / 6.32 / 6.42

*This is the comparator, so it has no "outside the floor" line.*


### e1 — the neighbourhood covariate

**Changed from the floor:** `--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean,max`
**Seeds:** 45, 46, 47 (n=3)

Twelve context channels instead of eight. Before this the model knew where past *change* happened but never the *level* of development around a pixel. Note it changes two things: the covariate **and** the dropped fine radii — `rad` below separates them.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00345 | 0.2126 | 0.01332 | 0.1415 | 0.1866 | 0.00296 | 0.9576 | 4.82 |
| +10 yr | 0.00582 | 0.2587 | 0.02075 | 0.2112 | 0.0966 | 0.00254 | 0.9592 | 4.36 |
| +15 yr | 0.00784 | 0.2847 | 0.02641 | 0.2532 | 0.0811 | 0.00232 | 0.9507 | 5.29 |
| +20 yr | 0.00952 | 0.2816 | 0.03108 | 0.2523 | 0.1232 | 0.00176 | 0.9439 | 5.99 |

`exceedance_abs_log10` 0.3399 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2827 / 0.2799 / 0.2816; `skill20` 0.2523 / 0.2513 / 0.2549; `pit_ks5` 0.1814 / 0.1866 / 0.1966; `tail_reach20` 5.84 / 6.01 / 5.99

**Reported, not judged:** `tail_reach20` median 5.99 vs floor 5.23–6.67 — inside the floor; `cov95_20` median 0.9439 vs floor 0.9429–0.9515 — 0 above / 1 below the floor

**Unanimous against the floor:** **crps_skill10** better 3/3 (+0.06 widths); **crps_skill20** better 3/3 (+0.24 widths); **skill5** better 3/3 (+0.08 widths); **skill10** better 3/3 (+0.11 widths); **skill15** better 3/3 (+0.29 widths); **skill20** better 3/3 (+0.21 widths); **rmse10** better 3/3 (+0.13 widths); **rmse15** better 3/3 (+0.32 widths); **rmse20** better 3/3 (+0.21 widths)  

**Split:** crps_skill5 better 1/3; crps_skill15 better 2/3; rmse5 better 2/3; pit_ks5 worse 1/3; pit_ks10 better 2/3; pit_ks15 better 1/3; pit_ks20 worse 2/3; exceedance_abs_log10 better 1/3


### e1a — the covariate, mean only

**Changed from the floor:** `--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean`
**Seeds:** 45, 46, 47 (n=3)

Isolates whether the `max` bands add anything over `mean`.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00346 | 0.2123 | 0.01332 | 0.1416 | 0.1880 | 0.00258 | 0.9543 | 4.99 |
| +10 yr | 0.00586 | 0.2547 | 0.02082 | 0.2059 | 0.0937 | 0.00234 | 0.9597 | 4.43 |
| +15 yr | 0.00791 | 0.2782 | 0.02661 | 0.2418 | 0.0840 | 0.00257 | 0.9514 | 5.82 |
| +20 yr | 0.00956 | 0.2785 | 0.03130 | 0.2421 | 0.0958 | 0.00247 | 0.9450 | 6.38 |

`exceedance_abs_log10` 0.3822 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2788 / 0.2785 / 0.2692; `skill20` 0.2421 / 0.2488 / 0.2305; `pit_ks5` 0.1880 / 0.1996 / 0.1636; `tail_reach20` 5.59 / 6.38 / 6.43

**Reported, not judged:** `tail_reach20` median 6.38 vs floor 5.23–6.67 — inside the floor; `cov95_20` median 0.9450 vs floor 0.9429–0.9515 — 0 above / 1 below the floor

**Unanimous against the floor:** *none*  

**Split:** crps_skill5 better 1/3; crps_skill10 better 1/3; crps_skill20 better 2/3; skill5 better 2/3; skill10 better 1/3; skill15 better 1/3; skill20 better 1/3; rmse5 better 2/3; rmse10 better 1/3; rmse15 better 1/3; rmse20 better 1/3; pit_ks5 better 1/3; pit_ks10 better 2/3; pit_ks15 worse 1/3; exceedance_abs_log10 worse 1/3


### rad — the context radii alone

**Changed from the floor:** `--context_radii 3,30,100`
**Seeds:** 45, 46, 47 (n=3)

The control e1 was owed: the radii change without the covariate. `e1 − rad` is the covariate; `rad − floor` is the radii.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00346 | 0.2119 | 0.01331 | 0.1429 | 0.1668 | 0.00268 | 0.9571 | 4.56 |
| +10 yr | 0.00583 | 0.2574 | 0.02076 | 0.2095 | 0.1400 | 0.00272 | 0.9583 | 4.08 |
| +15 yr | 0.00790 | 0.2793 | 0.02658 | 0.2440 | 0.0967 | 0.00243 | 0.9486 | 5.08 |
| +20 yr | 0.00958 | 0.2775 | 0.03127 | 0.2435 | 0.1112 | 0.00181 | 0.9444 | 5.60 |

`exceedance_abs_log10` 0.3968 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2775 / 0.2690 / 0.2825; `skill20` 0.2435 / 0.2370 / 0.2549; `pit_ks5` 0.1564 / 0.1668 / 0.1745; `tail_reach20` 5.60 / 5.40 / 6.42

**Reported, not judged:** `tail_reach20` median 5.60 vs floor 5.23–6.67 — inside the floor; `cov95_20` median 0.9444 vs floor 0.9429–0.9515 — 0 above / 1 below the floor

**Unanimous against the floor:** **skill5** better 3/3 (+0.25 widths); **rmse5** better 3/3 (+0.21 widths)  

**Split:** crps_skill5 better 1/3; crps_skill10 better 1/3; crps_skill15 better 1/3; crps_skill20 better 1/3; skill10 better 2/3; skill15 better 1/3; skill20 better 1/3; rmse10 better 2/3; rmse15 better 1/3; rmse20 better 1/3; pit_ks5 better 2/3; pit_ks15 worse 1/3; exceedance_abs_log10 worse 2/3


### e2 — knots in the body

**Changed from the floor:** `--spline_knots body_dense`
**Seeds:** 45, 46, 47 (n=3)

Adds knots at u = 0.35/0.45/0.55/0.65. The default grid has three knots between u = 0.10 and u = 0.90 while 53% of pixels move by less than 0.001 in twenty years.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00346 | 0.2109 | 0.01336 | 0.1367 | 0.1766 | 0.00312 | 0.9532 | 4.58 |
| +10 yr | 0.00586 | 0.2547 | 0.02087 | 0.2021 | 0.1087 | 0.00261 | 0.9570 | 4.53 |
| +15 yr | 0.00797 | 0.2733 | 0.02678 | 0.2328 | 0.0728 | 0.00229 | 0.9485 | 5.62 |
| +20 yr | 0.00961 | 0.2745 | 0.03140 | 0.2369 | 0.0884 | 0.00235 | 0.9444 | 6.15 |

`exceedance_abs_log10` 0.3646 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2745 / 0.2744 / 0.2754; `skill20` 0.2295 / 0.2402 / 0.2369; `pit_ks5` 0.2348 / 0.1766 / 0.1370; `tail_reach20` 6.15 / 5.47 / 7.27

**Reported, not judged:** `tail_reach20` median 6.15 vs floor 5.23–6.67 — 1 above / 0 below the floor; `cov95_20` median 0.9444 vs floor 0.9429–0.9515 — 0 above / 1 below the floor

**Unanimous against the floor:** *none*  

**Split:** pit_ks5 better 1/3; pit_ks10 better 1/3; pit_ks15 better 1/3; pit_ks20 better 1/3; exceedance_abs_log10 worse 1/3


### e4 — lower-tail weighted CRPS

**Changed from the floor:** `--crps_tail_weight_lo 1.0`
**Seeds:** 45, 46, 47 (n=3)

Weights the pinball terms below u = 0.05. `P(u<0.001)` read ~7x nominal at h=20, i.e. the model was blindsided by declines. A u-weighting moves where the optimiser spends effort and not the target, so it needs no importance correction.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00346 | 0.2112 | 0.01335 | 0.1389 | 0.1773 | 0.00343 | 0.9520 | 4.62 |
| +10 yr | 0.00584 | 0.2563 | 0.02080 | 0.2076 | 0.1195 | 0.00251 | 0.9556 | 4.47 |
| +15 yr | 0.00788 | 0.2812 | 0.02651 | 0.2479 | 0.0938 | 0.00255 | 0.9535 | 5.64 |
| +20 yr | 0.00957 | 0.2781 | 0.03120 | 0.2466 | 0.1358 | 0.00155 | 0.9473 | 6.70 |

`exceedance_abs_log10` 0.3622 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2821 / 0.2781 / 0.2695; `skill20` 0.2578 / 0.2466 / 0.2268; `pit_ks5` 0.1974 / 0.1773 / 0.1651; `tail_reach20` 5.71 / 6.86 / 6.70

**Reported, not judged:** `tail_reach20` median 6.70 vs floor 5.23–6.67 — 2 above / 0 below the floor; `cov95_20` median 0.9473 vs floor 0.9429–0.9515 — inside the floor

**Unanimous against the floor:** *none*  

**Split:** crps_skill5 better 1/3; crps_skill10 better 1/3; crps_skill15 better 1/3; crps_skill20 better 1/3; skill5 better 1/3; skill10 better 1/3; skill15 better 2/3; skill20 better 1/3; rmse5 better 1/3; rmse10 better 1/3; rmse15 better 2/3; rmse20 better 1/3; pit_ks5 better 1/3; pit_ks10 better 1/3; pit_ks15 worse 1/3; pit_ks20 better 2/3


### e5 — horizon-weighted loss

**Changed from the floor:** `--horizon_loss_weights 1,1.33,2,4`
**Seeds:** 45, 46, 47 (n=3)

h=20 receives a quarter of h=5's gradient because exposure is 4:3:2:1 by horizon.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00348 | 0.2074 | 0.01343 | 0.1283 | 0.1802 | 0.00244 | 0.9552 | 5.39 |
| +10 yr | 0.00591 | 0.2475 | 0.02103 | 0.1902 | 0.1429 | 0.00232 | 0.9573 | 4.93 |
| +15 yr | 0.00803 | 0.2677 | 0.02692 | 0.2250 | 0.1398 | 0.00264 | 0.9546 | 4.87 |
| +20 yr | 0.00969 | 0.2688 | 0.03149 | 0.2325 | 0.1149 | 0.00220 | 0.9512 | 5.54 |

`exceedance_abs_log10` 0.3584 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2688 / 0.2702 / 0.2680; `skill20` 0.2326 / 0.2325 / 0.2247; `pit_ks5` 0.2053 / 0.1802 / 0.1684; `tail_reach20` 6.00 / 5.54 / 5.24

**Reported, not judged:** `tail_reach20` median 5.54 vs floor 5.23–6.67 — inside the floor; `cov95_20` median 0.9512 vs floor 0.9429–0.9515 — 1 above / 0 below the floor

**Unanimous against the floor:** **crps_skill5** WORSE 3/3 (−0.17 widths)  

**Split:** skill5 worse 2/3; skill10 worse 1/3; rmse5 worse 2/3; rmse10 worse 1/3; pit_ks5 better 1/3; pit_ks15 worse 2/3; pit_ks20 worse 1/3; exceedance_abs_log10 worse 1/3


### e6 — deeper shape head

**Changed from the floor:** `--shape_head_hidden_layers 2`
**Seeds:** 45, 46, 47 (n=3)

Tests whether the shape head is under-capacity.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00347 | 0.2099 | 0.01337 | 0.1355 | 0.1842 | 0.00350 | 0.9492 | 5.34 |
| +10 yr | 0.00588 | 0.2516 | 0.02091 | 0.1985 | 0.1438 | 0.00304 | 0.9547 | 4.34 |
| +15 yr | 0.00800 | 0.2699 | 0.02682 | 0.2305 | 0.1494 | 0.00253 | 0.9489 | 5.44 |
| +20 yr | 0.00963 | 0.2730 | 0.03136 | 0.2388 | 0.1425 | 0.00197 | 0.9447 | 6.40 |

`exceedance_abs_log10` 0.3427 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2730 / 0.2685 / 0.2766; `skill20` 0.2388 / 0.2291 / 0.2406; `pit_ks5` 0.1842 / 0.1758 / 0.2168; `tail_reach20` 6.89 / 6.40 / 5.23

**Reported, not judged:** `tail_reach20` median 6.40 vs floor 5.23–6.67 — 1 above / 0 below the floor; `cov95_20` median 0.9447 vs floor 0.9429–0.9515 — 0 above / 1 below the floor

**Unanimous against the floor:** **pit_ks20** WORSE 3/3 (−0.52 widths)  

**Split:** pit_ks5 worse 1/3; pit_ks15 worse 2/3; exceedance_abs_log10 worse 1/3


### e7 — wider shape head

**Changed from the floor:** `--shape_head_width 64`
**Seeds:** 45, 46, 47 (n=3)

The same question through width rather than depth.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00346 | 0.2117 | 0.01333 | 0.1407 | 0.1692 | 0.00223 | 0.9508 | 4.95 |
| +10 yr | 0.00585 | 0.2553 | 0.02087 | 0.2019 | 0.1136 | 0.00228 | 0.9554 | 4.79 |
| +15 yr | 0.00794 | 0.2760 | 0.02671 | 0.2367 | 0.0901 | 0.00213 | 0.9479 | 5.30 |
| +20 yr | 0.00958 | 0.2775 | 0.03134 | 0.2401 | 0.1256 | 0.00225 | 0.9435 | 5.30 |

`exceedance_abs_log10` 0.3919 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2800 / 0.2755 / 0.2775; `skill20` 0.2468 / 0.2401 / 0.2388; `pit_ks5` 0.1692 / 0.1737 / 0.1603; `tail_reach20` 5.13 / 6.13 / 5.30

**Reported, not judged:** `tail_reach20` median 5.30 vs floor 5.23–6.67 — 0 above / 1 below the floor; `cov95_20` median 0.9435 vs floor 0.9429–0.9515 — 0 above / 1 below the floor

**Unanimous against the floor:** *none*  

**Split:** crps_skill5 better 1/3; crps_skill10 better 1/3; crps_skill20 better 1/3; skill5 better 1/3; skill10 better 1/3; skill15 better 1/3; skill20 better 1/3; rmse5 better 1/3; rmse10 better 1/3; rmse15 better 1/3; rmse20 better 1/3; pit_ks5 better 1/3; pit_ks20 better 2/3; exceedance_abs_log10 better 1/3


### nomono — no horizon-monotone width

**Changed from the floor:** `--spline_cumulative_width False`
**Seeds:** 45, 46, 47 (n=3)

Each horizon gets an independent scale instead of a cumulative sum of non-negative increments, so the 95% width may **shrink** with lead time. Q(u) increasing in u is untouched — that is structural to the spline.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00346 | 0.2113 | 0.01335 | 0.1383 | 0.1885 | 0.00225 | 0.9499 | 8.07 |
| +10 yr | 0.00586 | 0.2538 | 0.02087 | 0.2022 | 0.1324 | 0.00280 | 0.9499 | 5.91 |
| +15 yr | 0.00794 | 0.2752 | 0.02666 | 0.2398 | 0.1051 | 0.00379 | 0.9549 | 4.53 |
| +20 yr | 0.00966 | 0.2707 | 0.03143 | 0.2356 | 0.1102 | 0.00444 | 0.9530 | 4.09 |

`exceedance_abs_log10` 0.4184 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2707 / 0.2713 / 0.2576; `skill20` 0.2356 / 0.2376 / 0.2082; `pit_ks5` 0.1952 / 0.1885 / 0.1883; `tail_reach20` 3.99 / 4.33 / 4.09

**Reported, not judged:** `tail_reach20` median 4.09 vs floor 5.23–6.67 — **all 3 seeds BELOW** the floor, median 0.78 widths past the edge; `cov95_20` median 0.9530 vs floor 0.9429–0.9515 — 2 above / 1 below the floor

**Unanimous against the floor:** **exceedance_abs_log10** WORSE 3/3 (−0.46 widths)  

**Split:** crps_skill5 worse 1/3; crps_skill10 worse 1/3; crps_skill15 worse 1/3; crps_skill20 worse 1/3; skill5 better 1/3; skill10 better 1/3; skill15 better 1/3; skill20 worse 1/3; rmse5 worse 1/3; rmse10 better 1/3; rmse15 better 1/3; rmse20 worse 1/3; pit_ks15 worse 1/3; pit_ks20 worse 1/3


### e9 — no slope freedom

**Changed from the floor:** `--spline_slopes fritsch`
**Seeds:** 45, 46, 47 (n=3)

Every knot slope is derived from the adjacent secants (monotonicity-preserving, zero parameters) instead of learned: 29 head parameters per horizon fall to 16.

| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |
|---|---|---|---|---|---|---|---|---|
| +5 yr | 0.00345 | 0.2125 | 0.01331 | 0.1424 | 0.1416 | 0.00230 | 0.9564 | 5.19 |
| +10 yr | 0.00583 | 0.2572 | 0.02080 | 0.2070 | 0.1025 | 0.00162 | 0.9605 | 5.19 |
| +15 yr | 0.00789 | 0.2799 | 0.02648 | 0.2495 | 0.0716 | 0.00155 | 0.9562 | 5.61 |
| +20 yr | 0.00959 | 0.2763 | 0.03118 | 0.2477 | 0.0689 | 0.00088 | 0.9528 | 6.59 |

`exceedance_abs_log10` 0.3870 · `central_outside_interval` 0.00000 · `qf_vs_triple_max` 1.54e-05

**Per seed:** `crps_skill20` 0.2769 / 0.2752 / 0.2763; `skill20` 0.2482 / 0.2477 / 0.2443; `pit_ks5` 0.1416 / 0.1316 / 0.1668; `tail_reach20` 7.57 / 6.43 / 6.59

**Reported, not judged:** `tail_reach20` median 6.59 vs floor 5.23–6.67 — 1 above / 0 below the floor; `cov95_20` median 0.9528 vs floor 0.9429–0.9515 — 2 above / 0 below the floor

**Unanimous against the floor:** **pit_ks5** better 3/3 (+0.99 widths)  

**Split:** crps_skill5 better 1/3; skill5 better 2/3; skill10 better 1/3; skill15 better 2/3; skill20 better 2/3; rmse5 better 2/3; rmse10 better 1/3; rmse15 better 2/3; rmse20 better 2/3; pit_ks10 better 1/3; pit_ks20 better 2/3


---

## Part 4 — What this phase established

**Nothing cleared the bar.** Ten variants, three seeds each, against a six-seed floor: not one
median beat the floor's range by more than the range's own width. Two results are nonetheless
solid, because they rest on unanimity across seeds rather than on margin.

**1. The horizon-monotone width constraint is load-bearing — keep it.** `nomono` puts all three
seeds below the floor's entire range on `tail_reach20` (3.99 / 4.09 / 4.33 against a floor of
5.23–6.67), 0.78 widths past the edge, at no measurable central cost. A **second, independent**
metric agrees: `exceedance_abs_log10` — the error in predicted exceedance rates, which is what
the far-field defect is actually about — is worse in all three seeds as well (median 0.4184
against a floor of 0.3361–0.3925, 0.46 widths past). Two unanimous rows pointing the same way is
a great deal harder to explain as noise than one. `tail_reach` is
`(Q(0.999) − Q(0.5)) / (Q(0.975) − Q(0.5))`; the Gaussian value is 1.577 and the far field needs
~12 to put +0.05 inside its interval, so a 35% lighter tail is movement directly away from the
defect the model exists to fix. An external design review recommended *not* forcing width to grow
with lead time; on this model that recommendation is measurably wrong.

*Mechanism, offered as inference rather than measurement:* `tail_reach` is scale-invariant, so
this is not the scale parameterisation showing through. Under the cumulative sum, h=20's spread
is built from non-negative increments over h=5's and part of it has to be expressed through
shape; freed of the constraint, the optimiser buys h=20 spread with the scale alone and leaves a
lighter shape.

**2. Parsimony is the most promising lever tested.** `e9` — slopes from secants instead of
learned, 29 head parameters per horizon down to 16 — puts all three seeds below the floor on
`pit_ks5` at **0.99 floor widths**, one hair short of the bar, on a row that is properly powered
(width 15% of level). It costs nothing: `skill5`, `skill20`, `rmse5`, `rmse20` are each better in
2 of 3 seeds, and `tail_reach20` is if anything heavier (median 6.59 against a floor median of
6.37). This is the design review's §2.1 hypothesis — that the spline is more flexible than the
effective sample size supports — surviving its first real test. Its round-1 predecessor `d8` was
one of the NaN-killed runs, so it had never been tested cleanly before.

**3. e1's gain is two separable effects, and the control found them.** Dropping the fine context
radii buys the +5 yr central field and nothing beyond it (`skill5` and `rmse5` unanimous, h ≥ 10
all noise); the covariate buys h ≥ 10 and nothing at h=5. Adding the covariate even gives a
little of the h=5 gain back and costs `pit_ks5`. Two levers, two jobs, no overlap.

**4. Capacity and loss-shape knobs are inert; one is harmful.** `e2`, `e6`, `e7` — body knots,
deeper head, wider head — move nothing. `e4` is a null at n=3 after being southern Africa's only
winner. `e5` (horizon weights) is the single clear loser, worse on four metrics, mostly at h=5:
it takes gradient from the horizon with four times the data and spends it on the least-powered
one, and both ends get worse.

The pattern across the whole slate: **the model is not short of capacity or of a cleverer
objective — it is short of an input it never had.** The only levers that moved anything were a
new covariate and *removing* head parameters.

### Not run, and why

- **e8** (`--chip_sampling stratified --chip_sampling_correct True`) — the redo `d2` is owed,
  since `d2`'s verdict came from a fold that was NaN for all 149 epochs. Machinery and weight
  table are built. Dropped from this phase by choice.
- **e10** (`--spline_knots lean9`, a strict subset of the default grid: 14 bins → 8) — implemented
  and tested, never trained. Dropped by choice.
- **e3** (`--spline_knots deep_lower`) — **structurally excluded.** Knots 1e-4 apart drive
  boundary derivatives, pinned to secants `h/width`, to ~1e4 against ~1e3, and the shape-head
  biases go non-finite on the first optimiser step. Not a null result.
- **`d6` and `d7` redone** — round 1's two surviving findings, both from runs with a dead fold.
- **`--lr_schedule cosine`** — the last untested lever of the weight-averaging family.

### Instrument changes made during this phase

- `--abort_on_nonfinite` (default on) at three points in training.
- `compare_dist_runs.py`: `--group_seeds` (rank on the per-metric median of a variant's
  replicates), `--include` (keep another region's runs out of a floor's table), and the symmetric
  WORSE bar.
- `_spline_head_banner` prints `Spline head: knots <preset> (n=, bins=), slopes <mode>, N
  params/horizon`, and `verify_spline_head` greps it, so a head-capacity flag that silently
  failed to engage cannot read as "the lever does nothing". The printer and the check are tested
  against each other, and the check is proven to fire on three controls.
- The block screen (`--predict_subsample_blocks`) and blend-on-write, which took the two-fold
  prediction peak from 196 GB to 53 GB.

### A note on process

Two failures in this phase were not model failures:

*A gate written for a three-seed spread.* The judging statistic was never checked against the
thing it would be applied to. One measurement on the six-seed floor showed it varies 17.5x.

*A wait loop keyed on process-table text.* Every chained stage waited with
`ps -eo cmd | grep -q "<previous script>.s[h]"`. An unrelated shell that merely *quoted* the
script name in a comment kept matching after the stage had exited, and the queue sat idle for
3.2 hours with both GPUs free. The `[h]` bracket trick stops a grep matching itself; nothing
stops a third process from containing the string. Every wait is now `while kill -0 <pid>`.
