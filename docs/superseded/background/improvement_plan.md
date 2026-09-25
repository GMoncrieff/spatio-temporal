# Improvement plan — the round after the global scorecard

Written 2026-08-20, branch `ensemble`. Ten changes, sequenced. **All evaluation and testing on
Africa** (`config/region_africa.geojson`, 8112 × 7778, 35.65M valid px) by user instruction.

Start from `docs/global_scorecard.md` — it is the measurement this plan responds to.

---

## Why this order

Three constraints fix the sequence. Everything else is preference.

1. **Fix the ruler before measuring anything.** T8.2, T8.3 and T6.4 currently pass *because* the
   far field is too quiet — T8.2 reads 3.8e-06 against a `<= 0.002` ceiling, T8.3 reads 90,596
   against `>= 20x`, T6.4 clears a `>= 2.7` bar with 1343.9 against an observed 5.3. Any
   experiment scored against those is being graded in the wrong direction, and the whole point of
   this round is to make the far field louder.
2. **Do the invalidating thing once.** Changing the fold mask invalidates every checkpoint and
   every calibration fitted to the old residuals. So all model changes go in one batch, and
   everything fitted to residuals happens *after* it.
3. **Diagnose before committing.** Five of these items have a cheap experiment that determines
   their own scope. All of them run on artifacts already on disk, in under a day, and they decide
   what the expensive stage contains.

## Three corrections to the item list as written

- **Items 1's two layers are not purely redundant — they are keyed on different class axes.**
  Conformal `scale_factors.csv` is `horizon × dhat × HM × biome`; `width_factors.json` is
  `horizon × band × dhat × HM`, restricted to bands 0–2 (out to 10 px). They overlap on
  horizon/dhat/HM. The merge therefore has to *decide* whether biome or distance band earns its
  place, which is a held-out measurement, not a refactor.
- **Item 4's fix addresses half of item 4.** The physical floor fixes the 28× excess of
  *decreases*. It does nothing for the 2.3% deficit of *increases*. That half is either item 5's
  remote bound or a width-head change, and **which one is still unknown** — the `--sweep_u_bound`
  run OOM'd before it produced an answer. That single cheap experiment decides whether this round
  contains a model change for the far field at all.
- **The global smoke tier belongs to item 2**, not item 9 (it was attached to 9 in the list).

---

## Stage A — Fix the instruments — **done, see `docs/stage_a_instruments.md`**

Code only. Nothing was fitted or trained. Two of the four items changed for a different
reason than the one written here, and both corrections are recorded in that document.

| item | change | as built |
|---|---|---|
| **8** | T8.2 / T8.3 / T6.4 become two-sided: score remote-band change as a **ratio to observed**, like T8.1, instead of against a fixed ceiling. | as written. T8.2 is the mean `|log10 ratio|` over *both* remote tails at ≤ 0.301; T8.3 and T6.4 are ratio-to-observed in [0.5, 2.0]. The zero-observed fallbacks are kept and named on the row — southern Africa is the extent where they are the only branch that runs. |
| **10** | ~~T2.5 scored on `n_eff` (distinct 128 px chips) rather than raw pixel count — a χ² over 35.65M px rejects on deviations far too small to matter.~~ | **premise was wrong.** T2.5 is not a pixel test: its units are ecoregions, 158 on Africa and 804 globally, against 401 rank bins. The χ² is *valid* there (measured size 0.0105 at nominal 0.01) but nearly powerless against the broad dome these histograms carry. Ranks are pooled to ≥ 16 expected per bin before testing: same size, power 0.59 → 0.90 at n=158. **Africa's T2.5 goes 3/4 → 0/4**; the passes were non-rejections, and the over-dispersion is real. |
| **9** | Settle T3.2: re-scope its threshold or score it at short lags. | **the settled argument was southern-Africa-only.** The 14% structure budget at 25 px that justified "T3.2 conflicts with T3.4" is 57% on Africa's own residual fit, and the scored separation moved from 25 px to ~74 px between the two runs, which is the whole of the 16.5% → 0.031 "regression". T3.2 now gates on `improve / budget ≥ 0.50` with the budget computed in-run at the pairs' realized mean separation; raw improvement, the uniform reference and a short-lag reading are reported beside it. Still: do not tune the field to pass it. |
| **2** | Global smoke tier: every stage once at M=4, all four block scales, before any long run. | `scripts/run_smoke_tier.sh` + `scripts/check_smoke.py`. Built and proven on Africa. The checker was verified in both directions first — it clears the Africa M=400 card and rejects the global card on T3.5's NaN. |

**Then the baseline re-score (confirmed: do it).** Regenerate Africa's M=400 pair from its
recorded seeds (~17 min; the manifests survive at
`/mnt/hdd1/.../exp/africa_k5/{members,null}_m400_manifest.json`) and re-run only the affected
stages.

**`--stages change,clustering,rank,spatial` as written would have silently dropped T2.5**:
`stage_rank` takes its input from `stage_aggregate`'s return value in-process, so asking for
`rank` without `aggregate` scored an empty dict and emitted no rows at all. The aggregate
stage is now splittable — `--aggregate_parts zonal` runs the 31-minute half T2.5 needs
instead of the 111-minute whole — and it caches the zonal stats, so later rank-only re-scores
are free. The command is:

```
--stages aggregate,rank,spatial,change,clustering --aggregate_parts zonal
```

**Regenerate with `--rho_json` pointing at a path that does not exist.** The Africa M=400
card coupled its horizons at the 0.9 fallback (`validation_m400/PROVENANCE.md`); measuring ρ
now would change the ensemble and the comparison would no longer be apples-to-apples.

**Verification invariant:** the rows Stage A does not touch — T8.1, T8.4, T6.1/2/3/5 — must
come back byte-identical from the regenerated store. Anything else means the regeneration
differs, not the gates.

≈ 1 h total. This gives an apples-to-apples "what did the gate fix change" number against the
existing 111/133. Without it the final card in Stage F has no comparable predecessor.

**Exit criterion — met.** `data/ensemble/exp/africa_k5/validation_m400_stageA/` holds the
re-scored stages, the spliced full card and a `PROVENANCE.md` naming every row that moved.

**Africa 111/133 → 102/133**, same denominator, on an ensemble verified bit-for-bit identical
to the published one (regenerated manifests match on all 400 seed sets, ρ and field params;
`t8_change_clustering.csv` and `t6_change_distribution.csv` byte-identical). T2 37→34, T6
19→15, T8 11→9; T1, T3, T4, T5, T7 unchanged. All nine are instruments.

Cost as run: 21 min to regenerate the pair, 73 min to re-score, and 33 min for the Africa
smoke proof — against ~1 h projected. The zonal pass ran at ~16 min/year against 7.5 in the
original card, entirely contention from another user's GPU job.

One finding came out of it that belongs to Stage C/D rather than Stage A: **T3.2's structure
budget on Africa is 0.497 at the scored separation, against the 0.14 recorded from southern
Africa, so the old 0.30 threshold was below the budget and the target was reachable here all
along.** The ensemble captures 6% of it at 40 px and 36% at 6 px. That is a real long-range
structure shortfall, not a threshold artifact, and it points the same way as T2.5's dome and
T7.3's spread-skill ratio.

---

## Stage B — Cheap diagnostics

All on artifacts already on disk. No retraining, no generation beyond one small ensemble.

| # | question | how | cost |
|---|---|---|---|
| **5-diag** | Is the far-field *increase* deficit reachable post-hoc, or does it need the width head? | `predict_change_rates.py --sweep_u_bound 0.975,0.99,0.999,1.0` on Africa `residuals_w`. Read the far-band `P(Δ>0.05)`. **Needs ~84 GB — run it with nothing else on the box** (it was OOM-killed at 83.98 GB last time, competing with a validator). | minutes |
| **6-diag** | Is T4.2's regression caused by measuring ρ? | Generate two small ensembles (M≈50) identical but for ρ = 0.9 vs measured, score **T4.2 only**. | ~30 min |
| **7-diag** | Is the h=5 width head under-dispersed at model level, or is 1.3× a calibration artifact? | Read `k_up`/`k_lo` by horizon from `score_model_experiment.py` on the existing Africa stitch. If h=5's k is uniformly high across *all* classes it is the head; if it is class-concentrated it is calibration. | 40 s |
| **4-diag** | Does Δ ≥ −HM_t0 actually remove the 28× decrease excess? | Price it with `predict_change_rates.py` before implementing. | 30 s |
| **3-diag** | (largely settled — see below) | | |

**Exit criterion — met.** All four answered below; Stage C contains a far-field width change,
Stage D absorbs item 7, and **Stage E loses item 4 entirely**.

### Answers (measured 2026-08-20, Africa, box idle)

**5-diag — post-hoc cannot reach the far field. Stage C contains a width-head change.**
`predict_change_rates.py --sweep_u_bound 0.975,0.99,0.999,1.0` on `residuals_w`, 41 min,
completed with the box to itself after the earlier OOM at 83.98 GB. Far band (>100 px),
`P(Δ>0.05)` as a ratio to observed:

| tail bound | h=5 | h=10 | h=15 | h=20 |
|---|---|---|---|---|
| 0.975 (current) | 0.000 | 0.001 | 0.007 | 0.021 |
| 0.99 | 0.004 | 0.006 | 0.018 | 0.136 |
| 0.999 | 0.017 | 0.025 | 0.258 | 0.183 |
| 1.0 (no bound) | 0.017 | 0.025 | 0.258 | 0.183 |

**0.999 and 1.0 are identical — the lever saturates**, and even unbounded the far field emits
1.7-26% of the observed rate. Three further readings from the same run:

- the far-band *lower* tail is untouched by the bound (h=20 stays at 9.7-10.3x observed at
  every setting), so **item 4's physical floor is independent of item 5**, not a side effect;
- the **30-100 px band does respond**: 0.109 -> 0.449 at h=5 and 1.06 at h=20, so item 5-fix
  is still worth taking — it fixes the mid field;
- pooled over all 24 cells, mean |log10 ratio| goes 0.587 -> 0.297 (upper side 0.743 ->
  0.293), which is worth adopting on its own.

**6-diag — the rho hypothesis is confirmed, and T4.2 is a broken instrument underneath it.**
Two M=50 ensembles, identical but for the coupling. Africa's measured rho is
`{10: 0.374, 15: 0.347, 20: 0.769}` (now written to
`exp/africa_k5/residuals/horizon_autocorrelation.json`), against the 0.9 fallback every
regional card used:

| arm | T4.1 | T4.2 |
|---|---|---|
| rho = 0.9 | 0.900/0.902/0.901 vs a NaN target — unscorable | 0.6194 |
| rho measured | 0.386/0.372/0.773, passes all three to <= 0.025 | 0.4540 |

Measuring rho *lowers* T4.2 by 0.165 on identical generators, so global's 0.5526 is the
honest number and Africa's 0.7118 was flattered — `docs/global_scorecard.md` §5 confirmed.

**But T4.2 cannot reach its target for a reason unrelated to the copula.** Each `z_h` is
marginally N(0,1) whatever rho is, so rho cannot move the *population* spread at any horizon;
that is set by the width heads alone. Measured directly on the published bounds, the interval
width is non-decreasing across all three steps at **78.94%** of pixels (per step 90.5% /
99.1% / 89.3%). That is T4.2's ceiling at infinite M, against a >= 0.99 target:

| | T4.2 |
|---|---|
| population ceiling (published widths) | **0.789** |
| rho 0.9, M=400 (the 111/133 card) | 0.712 |
| rho 0.9, M=50 | 0.619 |
| rho measured, M=400 (global card) | 0.553 |
| rho measured, M=50 | 0.454 |

**4-diag — the physical floor is already implemented, and item 4 is a no-op.** This was
checked before writing any code, and the code was never written.

`marginal_from_z_torch` clamps every member to HM in [0, 1], so **Δ >= −HM_t0 holds by
construction in the sampler**, and `predict_change_rates.thresholds` prices it the same way
(`unreachable = hm0 + thr <= 0`). A pixel at HM <= 0.05 contributes exactly zero to
`P(Δ < −0.05)`; it cannot be the source of any excess.

Measured on the far band at h=20, on the holdout residuals:

| | Africa | global |
|---|---|---|
| far-band valid px | 3,807,633 | 33,565,179 |
| **can decrease past −0.05 (HM > 0.05)** | 9,704 (**0.25%**) | 48,438 (**0.14%**) |
| observed decrease events | **1** | **88** |
| — of those, on HM > 0.05 | 1 | **88 of 88** |
| observed *increase* events | ~346 | **83,797** |

Every observed far-band decrease sits above the floor and none below it, which is the floor
working exactly.

**Three consequences.**

1. `docs/global_scorecard.md` §3.2's mechanism — "on land whose HM is already zero a decrease
   is close to physically impossible, yet the marginal still assigns real probability to
   Δ < −0.05 there" — **is impossible by construction**. The excess lives on the 0.14% of
   remote pixels that *can* decrease, not on the HM ~ 0 majority.
2. The 28x global excess is **88 observed events**, and Africa's 123x is **one**. By rule 13
   that is not enough to fit a correction against, and **Africa cannot screen the far-field
   decrease at all**. Screen it globally or not at all.
3. **The far-field increase deficit is the real target**: 83,797 global events against 88,
   three orders of magnitude better powered, and Africa carries ~346 of them, which is enough
   to screen on.

What remains of the lower-tail story is a narrow one: on the 48k remote pixels with
HM > 0.05 the marginal is too fat downward. That is a conditional-on-HM shape question in a
0.14% stratum — worth a note against Stage D's class axes, not a stage item.

**7-diag — the h=5 width is calibration, not the head; and the far field is over-dispersed in
the bulk while being too short-tailed in the extreme.**
`score_model_experiment.py` on the Africa stitch (~5 min there, not the 40 s southern Africa
takes). Artifacts in `exp/africa_k5/diag7_width/`. `k_up` is the multiplier that would make
the published interval the residual's own 95% interval, so **k = 1 means the head is right**:

| band | h=5 | h=10 | h=15 | h=20 | coverage h=20 |
|---|---|---|---|---|---|
| 0-1 | 0.858 | 0.881 | 0.859 | 0.819 | 0.955 |
| 1-3 | 0.808 | 0.841 | 0.893 | 0.886 | 0.960 |
| 3-10 | 1.089 | 0.882 | 0.893 | 0.929 | 0.958 |
| 10-30 | 1.266 | 0.970 | 0.794 | 0.802 | 0.971 |
| 30-100 | 1.174 | 0.925 | 0.575 | **0.274** | 0.991 |
| >100 | 0.507 | 0.545 | 0.752 | **0.241** | **0.997** |

h=5 is **not** uniformly high — 0.81-0.86 near, 1.09-1.27 at 3-30 px — so it is
class-concentrated and **item 7-fix belongs in Stage D's calibration, not in the model batch**.

The far-field row is the one to carry forward. **Far-field intervals are 2-4x too wide and
over-cover at 0.997 against 0.95, while the far field emits 44x too few large increases.**
That is not a contradiction: `k` scales the 95% interval against the residual's *bulk*, and
`P(Δ>0.05)` is a rare-event property of the tail *beyond* it. The far field is over-dispersed
in the bulk and too short-tailed in the extreme.

**Consequence for Stage C and E: do not fix the far field by widening it.** Widening pushes
T1 coverage further past 0.997 and does almost nothing for the rare-event rate, which is what
5-diag showed the tail bound already saturating against. The lever has to change tail *shape*
at fixed or reduced bulk width.

Pooled: `mean|ln k|` 0.3906 leaf / 0.3198 band, with only 42.1% of 200 classes within 20%.

So T4.2 measures three things at once — head monotonicity, rho and M — and gates on a target
none of them reaches.

**Fixed, 2026-08-20.** T4.2 now scores the **population** spread: the member marginal
integrated against N(0,1) by Gauss-Hermite (`validate_ensemble.population_spread`), which is
shape-aware and sees the [0,1] clip, so it is the quantity a finite ensemble estimates without
the finite ensemble's noise. The old sample statistic is retained as reported-only **T4.2s**,
because its shortfall is Monte-Carlo noise modulated by M and rho and there is no
M-independent threshold to put on it.

Verified on the two M=50 arms, which differ only in the coupling:

| arm | T4.2 (population) | T4.2s (sample) |
|---|---|---|
| rho = 0.9 | **0.7304648026222341** | 0.6194 |
| rho measured | **0.7304648026222341** | 0.4540 |

**Identical to every digit**, which is what rho-invariance means and what the old gate did not
have. The ceiling is 0.7305 on the validator's own 2000x2000 sample — the earlier 78.94% came
from a raster strip and ignored the shape and the clip, so 0.7305 is the number to quote.

**The remaining consequence is a model finding:** the width heads shrink with horizon at 27%
of the map. That is not fixable post-hoc and belongs with the far-field width work, not with
item 7 (which 7-diag moved to Stage D).

---

## Stage C — Model batch (one retrain, ~1.5 h on Africa)

Everything that requires retraining goes here, together, because each batch costs five folds and
invalidates comparability.

### Item 3 — fold blocks. **Done: 4x4 adopted, and it is nearly free.**

`exp/c1_foldb4/` — five folds on `fold_mask_b4_1000.tif`, `VAL_STRIDE=1024`, the shipped
architecture flags, 89.7 min train + 1.9 min stitch. Verified against `africa_k5`'s startup
fingerprint (`ConvLSTM grad norm: 0.000000`) before anything downstream was read.

**Pooled skill against persistence, same pixels, same architecture, only the mask differs:**

| h | africa_k5 (1x1) | c1_foldb4 (4x4) | cost |
|---|---|---|---|
| 5 | 0.1298 | **0.1301** | **+0.000** |
| 10 | 0.2006 | 0.1960 | −0.005 |
| 15 | 0.2340 | 0.2277 | −0.006 |
| 20 | 0.2313 | 0.2221 | −0.009 |

**Honest held-out geography costs at most 0.9 skill points, and nothing at h=5.** The mask is
adopted: every later card is more defensible at negligible cost.

**And the leak is now measurable, which is the point of the change.** Within the new mask,
skill falls with distance from the fold's own trained-on data — visible only because this mask
*has* pixels beyond a correlation length:

| h=20, distance to trained-on data | 1-32 px | 32-64 | 64-128 | 128-192 | 192+ |
|---|---|---|---|---|---|
| skill | +0.239 | +0.239 | +0.224 | +0.211 | **+0.188** |

**0.239 → 0.188, a 21% relative decline.** The 1x1 mask cannot see this: its held-out pixels
stop at 128 px. So rule 19's "held-out skill is optimistic" is confirmed and, for the first
time, *quantified* — the optimism is real but modest, and it is concentrated in exactly the
far-field territory the rest of this round is about. At h=5 the same profile is flatter
(+0.054 near, +0.037 far).

### Item 3 — the sizing argument. **2×2 does not work; use 4×4.**

The residual's fitted practical range is 135 px at h=20. A pixel in a `128·B` block is at most
`64·B` px from the block edge, i.e. from trained-on data:

| block | px | max dist to edge | % held-out px beyond 135 px | blocks on Africa |
|---|---|---|---|---|
| 1×1 (today) | 128 | 64 | **0.0%** | 3,851 (~770/fold) |
| **2×2** | 256 | 128 | **0.0%** | 962 (~192/fold) |
| 3×3 | 384 | 192 | 8.8% | 427 (~85/fold) |
| **4×4** | 512 | 256 | **22.3%** | 240 (~48/fold) |
| 6×6 | 768 | 384 | 42.0% | 106 (~21/fold) |
| 10×10 | 1280 | 640 | 62.3% | 38 (~7/fold) |

**At 2×2 not one held-out pixel is beyond a correlation length of trained-on data.** It doubles
the mean distance (64 → 128 px at block centre), which genuinely reduces the leak, but nothing
crosses the threshold, so the honest-skill number it produces is still optimistic everywhere.
**4×4 is the smallest block that buys any clean held-out area** and still leaves ~48 blocks per
fold on Africa. 10×10 — the value already implemented — was sized for the global grid's 41,496
chips and would leave ~7 blocks per fold here, which is too few to average over.

**Decision rule if this is revisited:** pick the smallest `B` with ≥ 20% of held-out pixels
beyond the fitted practical range *and* ≥ 40 blocks per fold on the evaluation region. On Africa
at range 135 px that is `B = 4`. Recompute if the fitted range moves.

`create_validity_mask.py --folds_only --k 5 --fold_block_chips 4 --fold_mask_out <new>` builds
it. **This is a new fold mask: every existing checkpoint becomes unusable and every scorecard
before it becomes non-comparable.** That is the cost, and it is why this round is the cheapest
moment to pay it.

**Two k=5 runs on this mask were thrown away before the real one, and the cause was mine,
not the mask's.** Both were launched without `BASE_ARGS`, i.e. on argparse defaults, where
`--central_residual` is False and the central head predicts absolute HM. Its documented cost
is sd ~0.0075 HM of spurious change on pixels that did not move; measured, the unchanged-pixel
reconstruction was 0.0065-0.0067 against 0.0019 for `africa_k5`. Skill read −0.64 and −0.84 at
h=5 and looked exactly like "honest held-out geography is expensive".

It was not believed, and four checks said so before the cause was found: seed replicates put
central-field noise at ~0.0003 (the gap was 60x that), the stitch reproduced the fold's own
raster to five decimals, the training pool matched in size (24,908 vs 24,897) and HM
composition (to 0.004), and **RMSE was flat with distance from trained-on data** — which a
leak cannot be. See CLAUDE.md rules 24 and 25; dead runs at `exp/c1_foldb4_noflags/` and
`exp/c1_foldb4_valstride2048/`.

A side observation from those runs, **not** established as a cause: at 2048 the stride draws
30 fold-2 chips holding mean RMS 5-yr change 0.0003 under this mask, against 34 chips at
0.0114 under the 1x1 one. Changing it to 1024 (131 chips, 0.0075) did not help at defaults —
skill went −0.641 to −0.842 — and its effect under the correct flags has not been isolated.

**Built and verified, 2026-08-20** — `data/raw/hm_global/fold_mask_b4_1000.tif` (production
`fold_mask_1000.tif` untouched). 2,652 super-blocks of 512 px globally, folds balanced to
within 0.1% (8,292-8,304 chips each).

The table above is geometric. Measured with a Euclidean distance transform on the Africa
window — for each fold, the distance from every held-out pixel to the nearest pixel the fold
*trained on* — the mask does better than the estimate:

| mask | held-out px beyond the 135 px range, per fold | mean | 512 px blocks per fold on Africa |
|---|---|---|---|
| 1x1 (current) | 0.0 / 0.0 / 0.0 / 0.0 / 0.0 % | **0.0%** | — |
| **4x4 (new)** | 32.4 / 32.9 / 30.6 / 32.1 / 31.7 % | **31.9%** | **137-165** |

31.9% against the 22.3% the geometry predicted, because adjacent blocks sometimes fall in the
same fold and form larger contiguous territories than a single block. Blocks per fold on
Africa are 137-165 against the >= 40 the decision rule asks for, three times the estimate.
Africa fold sizes are less even than the 1x1 mask (10.96M-13.82M px, +/-12%, against +/-4%),
which is the expected cost of larger blocks and is well inside what k=5 averaging tolerates.
Total valid pixels are identical (63,095,136), so the two masks cover the same ground.

### Item 7-fix — the h=5 width head, if 7-diag says model-level.

### Far-field width head — 5-diag says post-hoc cannot reach it, but only for part of it.

**Measured 2026-08-20, and it splits the defect into two disjoint populations.** The far band
on Africa at h=20 holds 346 observed `Δ > 0.05` events. Stratified by HM level:

| HM stratum | px | gainers | observed rate | residual q99.9 | median `w_up` | half-widths to reach +0.05 |
|---|---|---|---|---|---|---|
| [0, 0.01) | 3,780,806 | **236** | 6.2e-05 | 0.0028 | 0.00145 | **34.5** |
| [0.01, 0.05) | 17,123 | 55 | 3.2e-03 | 0.0758 | 0.00738 | 6.8 |
| [0.05, 0.15) | 8,410 | 38 | 4.5e-03 | 0.1167 | 0.00911 | 5.5 |
| [0.15, 0.40) | 1,294 | 17 | 1.3e-02 | 0.0122 | 0.01193 | 4.2 |

The fitted shape's support is ~8 z-units, so the last column is the test of reachability.

- **110 of 346 gainers (32%) are reachable post-hoc.** In the three upper strata +0.05 sits
  4.2-6.8 half-widths out, inside the shape's support, and the residual's own 99.9th
  percentile there is 0.076-0.122 — the data supports the tail. **An HM axis on the far
  band's width/shape buys these with no retraining**, which is a Stage D change, not Stage C.
- **236 of 346 (68%) are not.** In the `[0, 0.01)` stratum +0.05 is **34.5 half-widths** away.
  No reshaping of a tail reaches that; it needs the width head to emit a far larger width on
  the tiny developable subset of near-zero-HM remote pixels.

**And the model has no signal for that subset.** Separability of the 346 gainers from the rest
of the far band, by AUC:

| covariate | AUC |
|---|---|
| `dhat`, the model's own predicted change | **0.465** — worse than random |
| `w_up`, the model's own half-width | **0.688** |
| `hm0` | **0.699** |
| distance to past change | 0.374 (inverted; gainers sit closer, median 131 vs 164 px) |

The width head has already learned the *ordering* — its half-width is 3.5x larger at gainers —
while the central head has not. So the model change is a **scale** problem on a head that
already ranks correctly, not a representation problem. But `hm0` alone beats the model, which
is why the post-hoc half is available at all.

**Consequence for scope.** These are disjoint populations, so both changes are justified and
neither substitutes for the other. It also bounds the outcome: a perfect Stage D fix moves the
far-field increase ratio from 0.107 to about 0.4 of observed, not to 1.0. The remaining 68%
is the retrain's to win, and it is the harder half.

The far field is ~73% of valid pixels but carries almost no change events, so any loss averaging
over pixels barely sees it. That imbalance is the mechanism to attack. Note that
`width_head_mode power/power_plus` is recorded dead in `docs/model_phase.md` — **but it was
scored on southern Africa, whose observed far-field rate is 0.0000**, so its verdict inherits the
blind spot and is worth re-reading rather than citing.

**Write the Stage D code in parallel with this stage** — it is model-independent, and having it
ready means the new model is calibrated through the final chain on the first pass.

---

## Stage D — item 1 — **NOT DONE. The axes were measured; the layers were never unified.**

**Read this heading literally.** The plan asked for one mandatory class-specific half-width
rescaling replacing both layers. That did not happen. `apply_recalibration.py` still applies
the conformal `scale_factors.csv` and then multiplies `width_factors.json` on top, keyed on
different axes, exactly as before this round.

What was done is the *measurement* the plan said the unification depended on — which axes
earn their place, on held-out data — and the result made unification look less urgent rather
than settling it. The margins below are the reason; they were not put to the user as a
decision at the time, and the item should be treated as open.

**What unification actually turns on, and is still untested: does the conformal layer's
`biome` axis earn its place?** That axis is the main reason the two layers are keyed
differently. `fit_width_factors.py` has no biome axis, so the comparison was never run.

**And there is now a concrete cost to leaving them split.** T4.2 fails at 0.802 because the
calibration destroys monotonicity the heads had (0.980 → 0.813; see the reference card). The
conformal layer is isotonically smoothed across horizons for exactly this reason; the width
layer is not, and a pixel can change class between horizons besides. A single layer could
carry the constraint internally. Split, the only robust fix is a cumulative maximum applied
to the half-widths after both layers have run.

### What was measured

Fitted on `c1_foldb4`'s own residuals, folds 1-3, scored on **held-out folds 4-5**
(25,833,824 px) by interval score. `scripts/score_width_variants.py` evaluates straight from
the residual manifest — a width factor rescales the published half-widths, so coverage and
the interval score follow in closed form and one raster pass scores every variant.

| variant | coverage | mean width | interval score | worst class dev |
|---|---|---|---|---|
| all (bands 0-5) | 0.938 | 0.0516 | **0.08996** | 0.198 |
| **near (0-2)** — retained | 0.961 | 0.0530 | 0.09054 | 0.158 |
| far_only (3-5) | 0.945 | 0.0559 | 0.09080 | 0.198 |
| identity | 0.968 | 0.0574 | 0.09138 | 0.158 |

**`near` is kept.** `all` wins the pooled score by 0.6% while pushing a class at h=10 down to
0.752 coverage (worst deviation 0.198, shared by both far-band variants) and pulling pooled
coverage below nominal to 0.938. That is a per-row loss bought with a total-score gain, which
rule 4 exists to refuse. **The distance axis earns its place only out to 10 px** — the same
conclusion `fit_width_factors.py`'s docstring reached from a pooled-over-HM argument, now
confirmed under a held-out protocol on the new model.

**The result that matters more is the scale.** The entire width layer is worth **0.9%** of
held-out interval score against identity, and extending its distance reach a further 0.6%.
Unification is therefore a tidiness question, not a skill one, and it should not be allowed to
consume the budget the far field needs. Note this does not contradict Stage B: the far-field
increase deficit is a *tail* problem (Stage E), and a width factor rescales the body.

**Not tested:** whether the conformal layer's *biome* axis earns its place. `fit_width_factors`
has no biome axis, adding one is real work, and with every margin here under 1% it is hard to
justify ahead of the far-field tail. Recorded as open rather than answered.

### The original framing

One **mandatory** class-specific half-width rescaling. Central forecast untouched. Sole source of
the published bounds. The identity/global/stratified decision disappears.

Held-out marginal coverage, class-conditional coverage, interval score and sharpness stay — as
**reported diagnostics of the mandatory step, not as a gate deciding whether to apply it.**

The substantive question is which class axes survive. Fit three ways — distance band, biome, and
both — and keep whichever earns its place **on held-out data**. The precedent matters here:
per-class *tail bounds* were rejected twice on held-out protocols (temporal windows on southern
Africa, spatial folds on Africa) while quantile *estimates* on 15k+ pixels survived the same
test. Metric minimisations over ~60 events do not generalise; quantile estimates on large classes
do. Use a held-out protocol, not an in-sample fit.

Expect this to absorb item 7's h=5 widening as a class factor rather than a global stretch.

---

## Stage E — Marginal, on the new chain

| item | change | note |
|---|---|---|
| ~~**4**~~ | ~~Clamp the left tail at a physical floor, Δ ≥ −HM_t0~~ — **dropped: already implemented.** | 4-diag: the sampler clamps members to HM ∈ [0,1], so the floor is exact and all 88 observed far-band decreases lie above it. The premise (~40% of the globe at HM ≈ 0 driving the 28× excess) is impossible by construction. |
| **5-fix** — **done** | `--u_bound 0.999` uniformly, **including the remote band**, replacing `0.999,…,0.975`. | Swept on `c1_foldb4`'s own residuals: mean \|log10 ratio\| **0.506 → 0.272**, and the far band at h=20 goes 0.014 → 0.162 of observed. 1.0 is identical to 0.999, so the lever is exhausted — the rest is the model's (Stage B, 5-diag). |
| ~~**6-fix**~~ — **dropped, measured** | ~~Let the tail policy vary by horizon.~~ | **The horizon axis carries nothing.** Per-horizon mean \|log10 ratio\| at 0.999 vs 1.0: 0.2840/0.2840, 0.2658/0.2656, 0.2056/0.2056, 0.3308/0.3311. The optimum is the same bound at every horizon and the two candidates tie to the fourth decimal, so the flag was never implemented. |

Price every candidate with `predict_change_rates.py` (~30 s) before regenerating anything. It
shares no code with the GPU sampler, so agreement is evidence and disagreement localises a bug to
the generation path.

**Guardrail:** widening the far field without the floor pushes the *left* tail into physically
impossible territory. Watch far-band `P(Δ<−0.05)` alongside `P(Δ>0.05)` on every candidate.

---

## Stage F — One reference card — **done: 103/136, `docs/reference_card.md`**

Ran with the M=4 smoke tier as a gate first (13 min, passed) — the first time a long run in
this project was gated rather than launched hopefully. Generation 9.7 + 9.9 min, scorecard
237 min inside a 20 GB budget.

Headline: **remote-band `P(Δ>0.05)` 0.107 → 0.845 of observed.** Full reading, including the
three model defects that remain and the one open instrument question (T5.1), is in the
reference card rather than repeated here.

### The original framing

Full T1–T8 at M=400 on Africa: new fold mask, new model, unified calibration, new marginal, fixed
instruments. This becomes the reference. **111/133 and 112/146 are historical and not comparable
to it** — the fold mask, the gates, the denominators and the calibration chain have all changed.

---

## Cost

| stage | wall clock |
|---|---|
| A instruments + baseline re-score | ~1 day (mostly code) + 1 h |
| B diagnostics | ~half a day |
| C model batch | ~1.5 h compute |
| D calibration refactor | ~1 day (code) + ~40 min/fit |
| E marginal | ~half a day, minutes per candidate |
| F reference card | ~3 h |

Three to four days, of which only the retrain and one scorecard are expensive. Everything else is
minutes to an hour, which is the point of the ordering.

## Deferred beyond this round

The hindcast COG deliverables, the old-vs-new-ConvLSTM comparison
(`data/ensemble/hindcast/stitched/` is kept on disk as the main-branch baseline), and the
2025-2040 production forecast. The 2040 decision is already taken: a sixth model trained on all
chips, fold mean reserved for display.
