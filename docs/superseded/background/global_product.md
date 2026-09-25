# The global product — what was built and what it measures

Written 2026-08-24, branch `ensemble`, experiment **`g1_foldb4`**. This executes
`docs/global_final_phase.md`. The model was frozen throughout; the only fitting was
re-deriving, on global data, parameters previously estimated on Africa or southern Africa.

Artifacts: `/mnt/hdd1/spatio-temporal/data/ensemble/exp/g1_foldb4`, symlinked at
`data/ensemble/exp/g1_foldb4`.

---

## 0. One correction to the plan's ordering

`global_final_phase.md` numbers "tune to the global extent" as step 1 and "global hindcasts"
as step 2. They must run the other way round: every parameter in step 1 is fitted on *global
residuals*, which do not exist until the hindcast is predicted and stitched. The document's
numbering is a list of concerns, not an execution order.

## 1. The hindcast

Predict-only from the five existing `c1_foldb4` checkpoints — they were already trained on
global chips; only the prediction extent had been Africa.

- **152.5 min** for five folds on two GPUs, against a 218 min budget. The 512 px mask makes
  each fold's territory contiguous, so `--predict_restrict_mask` skips more tiles than it did
  under the 128 px checkerboard.
- All five checkpoints logged `0 warm-started convs`; a main-branch checkpoint was included as
  a control and correctly **rejected** on `head_in 64` against the required `hidden + 8 = 72`.
- **Every one of the 30 stitched rasters reports exactly 184,573,321 valid px.** That figure
  was computed *before* the run from `fold_mask_b4 ∈ {1..5} ∧ valid HM`, and the method was
  validated first by reproducing the same number on the old mask. Per fold: 35.79 / 38.48 /
  36.74 / 37.49 / 36.08 M.
- Provenance asserted, not assumed: 136.7M sampled pixels, **0** disagree with the value from
  the fold that held them out.
- Distinct from the main-branch baseline (mean |diff| 0.0090 over all 184.6M px, 0.05%
  identical) — byte-identical output would have meant the wrong checkpoints loaded.

### The mean stitch was dropped, deliberately

Prediction runs with `--predict_restrict_mask`, so each fold's raster covers its own territory
plus a ≤128 px tile halo. `--stitch_mode mean` over those averages only in the fringe: it is
holdout-plus-halo, not the fold mean rule 18 describes. A true fold mean needs unrestricted
prediction at roughly 5× the cost. The scored product is the holdout either way and the forward
product has no mosaic, so nothing was lost.

## 2. Parameters re-derived globally

| parameter | Africa | **global** |
|---|---|---|
| AR(1) ρ | {10: 0.650, 15: 0.461, 20: 0.433} | **{10: 0.749, 15: 0.613, 20: 0.706}** |
| residual practical range | 99–166 px | **303 px** |
| spectrum | — | Matérn, nugget 0.032–0.064; 0.36–0.56 of variance beyond 50 px |
| calibration bands | all six | **all six** (re-decided; see §3) |
| marginal `u_bound` | 0.999 | **0.999**, identical to 1.0 — lever exhausted |
| block sizes | 1,10,100 | **1,10,100,1000** |
| ecoregions | 158 | **804** |

ρ was cross-checked across four independent stripe phases: spread 0.028 / 0.043 / 0.036, inside
T4.1's own ±0.10 tolerance. Neither Africa's values nor the old model's global values transfer.

Central field, held out, skill against persistence: **+0.110 / +0.183 / +0.226 / +0.222** at
h=5/10/15/20 — positive everywhere, rising with lead time, converging on Africa's numbers by
h=15. Raw coverage 0.948–0.955 before any calibration.

## 3. The band decision — reversed from what Africa implied

The open question was whether to exclude band 3 (10–30 px), which overshot T8.1 to 2.28× on
Africa. Globally the answer is **no: correct all six bands.**

Six `--bands` variants were priced from **one fit**. Each distance band is the root of its own
shrinkage hierarchy (`fit_width_factors.py:276-296`) and the horizon-monotonicity pass is keyed
on `(band, dhat, HM)`, so restricting the JSON reproduces a restricted fit exactly. Proven, not
assumed: an independently fitted `--bands 0,1,2` control matched the derived variant on all 360
cells, 0 disagreements. Saved ~3 h.

| variant | held-out interval score | worst class dev |
|---|---|---|
| identity | **0.110688** | 0.355 |
| b01245 (no band 3) | 0.110985 | 0.282 |
| b012345 | 0.111738 | 0.282 |

**Identity wins the pooled score, and is wrong.** Class-conditional coverage names the reason:

| class | n | identity | b012345 |
|---|---|---|---|
| >100 px × HM [0.01,0.1), h=15 | 378,581 | **0.674** | 0.974 |
| >100 px × HM [0.01,0.1), h=20 | 378,581 | **0.707** | 0.993 |
| 30–100 px × [0.01,0.1), h=15 | 2,280,792 | 0.883 | 0.983 |
| 10–30 px × [0.1,0.3), all h | 939,303 | 0.867–0.892 | 0.928–0.969 |

Remote land carrying some development — 0.2% of the grid — has an interval covering 67% of the
time under identity, and a pooled average cannot see it. This is the biome lesson (rule 26)
running the other way: last round a pooled metric preferred a bad option, here it prefers doing
nothing. b012345 is chosen over b01245 because only it repairs the band-3 class, at a cost of
0.68% pooled score — rule 4, the same call that kept `near` over `all` last round.

Band 3's Africa overshoot does not reproduce: closed-form `P(Δ>0.05)` for that band goes
0.60 → **0.93** of observed at h=20 when corrected, i.e. toward observed, not past it. Rule 27
— a restriction inherited from an older model can invert — measured.

The tail audit vetoed nothing: every corrected band gets **wider**, the far band's p99.9
half-width by 1.307×. The far-band factor table reproduces the §2.4 mechanism globally — the
bulk remote class narrowed to 0.897 while remote-land-that-can-develop is widened 2.43–2.68×
(Africa: 0.760 and 4.004).

## 4. The scorecard — 99 of 134 scored rows

M=400, all four block scales, `--mem_budget_gb 60`. **Not comparable to Africa's 107/136**: the
denominator is different, T4.1 is scored rather than reported-only, and ecoregions went
158 → 804.

| family | pass | |
|---|---|---|
| T1 per-pixel uncertainty | **24/24** | |
| T2 aggregate scale | 41/49 | |
| T3 spatial realism | 3/5 | |
| T4 temporal coherence | 3/4 | |
| T5 hard gates | 10/12 | |
| T6 change realism | 10/24 | |
| T7 diversity / spread-skill | 1/2 | |
| T8 placement | 7/14 | |

**Better than Africa.** T1 is perfect — pooled coverage 0.943/0.958/0.954/0.959, where Africa
failed two of four. **T5.1 passes 4/4 at 0.998–0.999**; that gate was the standing open
instrument question and it simply passes at global scale. T4.1 reproduces measured ρ to within
0.016. T2.2 ecoregion coverage reads 0.973–0.988 on 804 regions instead of saturating at 1.000
on 158 — the coarse-quantisation that inflated about half of Africa's margin is gone. T3.2 more
than doubled, 0.104 → 0.230, though still short of its 0.50 gate.

**The headline defect is the far field.** T8.1 beyond 100 px reads **0.034** of observed against
a [0.5, 2.0] gate, and T8.3's near/remote contrast is **40.1** where Africa read 1.138 — purely
because the remote denominator is near zero. This is the frozen width heads, not the
calibration: far-band half-widths are ~0.004, so +0.05 sits ~12 half-widths out and a 0.999 tail
reaches 3–4. The calibration did its job there (coverage 0.674 → 0.993, p99.9 widened 1.31×) and
cannot bridge that gap. **Africa's 1.36 does not transfer** — rule 14 in the opposite direction
from the usual one.

Unchanged known defects: aggregate over-dispersion (T2.5 rejected 4/4, T7.3 at 1.637) and the
too-heavy decrease tail (T6.1 0/4 at 4.1–9.4×), the latter worse globally than on Africa.

**Open discrepancy, not resolved.** `predict_change_rates.py` puts the >100 px band at exactly
0.000000 at every horizon; the sampled M=400 ensemble reads 0.034. Both fail the gate and the
conclusion is unchanged, but these two implementations have historically agreed to 1–11%, so the
gap localises a problem to either the sampler or the closed form. Worth taking up before the
far field is worked on.

Under-powered rows to read with care: the 1000 km block scale has only **182 blocks**, so a
reading near 1.000 there is a power ceiling, not a result (rule 5).

## 5. Deliverables

| artifact | path | size |
|---|---|---|
| hindcast ensemble, M=400 | `members_m400.icechunk` | 398 GB |
| hindcast COGs (central/lower/upper × 2005/10/15/20) | `cogs_hindcast/` | 8.7 GB |
| forecast ensemble, M=400 | `forecast_members_m400.icechunk` | 421 GB |
| forecast COGs (× 2025/30/35/40) | `cogs_forecast/` | 8.7 GB |
| calibrated hindcast rasters | `recal_u/` | 17 GB |
| calibrated forecast rasters | `forecast_recal/` | 6.8 GB |

All 24 COGs verified individually: `LAYOUT=COG`, seven overview levels, and transform, CRS,
nodata and sampled values byte-identical to source.

The production model is `spatio-temporal-convlstm/6zkppztt/checkpoints/epoch=12-step=169.ckpt`
— `c1_foldb4`'s exact configuration, seed 42, **no fold excluded**, trained on every chip. It
carries `--histogram_weight 1.0` and the default checkpoint monitor deliberately: `c1_foldb4`
did not carry the model-phase flags, and the hindcast residual statistics only transfer to the
same configuration.

**What cannot be validated:** ŝ is fitted on 2000–2020 residuals and applied to 2025–2040, so
the error structure is assumed stationary in time. Every h=20 number is in-sample in time,
because only the w2000 window reaches +20 yr inside the observed record. And a production model
has no held-out geography by construction — its validation split is in-sample, which is why the
*configuration* is validated by k-fold and the draw then trusted (±7 rows of run-to-run noise).

## 6. Three defects found and fixed

All three were invisible at regional scale, and two would have shipped silently.

1. **`enforce_horizon_monotonicity` OOM-killed the global run.** It held `3×H` full-grid float64
   arrays plus their `np.stack` copies plus four accumulators — over 200 GB at 17111 × 40000,
   about 20 GB on Africa. Rewritten to stream by row block; peak went to **14.4 GB**. The cummax
   runs along the horizon axis and never couples two pixels, so blocks are exactly equivalent —
   asserted against the old implementation at four block sizes in
   `tests/test_horizon_monotonicity_streaming.py`. Rule 15 again.

2. **The monotonicity pass silently skipped the entire forward product.** Its glob required a
   `w{base}_` prefix that production rasters (`prediction_2025_central_recal.tif`) do not have,
   and the stem parse would have raised `ValueError` on `int('rediction')` even if it matched.
   The forecast would have shipped with intervals free to *shrink* with lead time — the property
   T4.2 exists to guarantee. Never caught because no production model had ever existed. Now
   handles both conventions; on the real forecast it lifted **22.9%** of pixels.

3. **One distance raster was applied to all four windows.** The accepted Africa chain keys every
   window's calibration off `w2000_dist_past_change.tif`, but only **49–61%** of valid pixels
   keep the same distance band between windows, and 25–30% of far-band pixels move. About half
   the pixels in three of four windows had the wrong class looked up. It does not touch Africa's
   scored w2000 rasters, but it reaches its marginal shape through `residuals_u`. The global run
   applies each window's own covariate.

A fourth was caught by review rather than by the machine: **`--train_all_splits`**. With no
`--exclude_fold`, `train_split_value` stays at 1, so the production model would have trained on
the 70% train split and discarded ~30% of the world's chips for no benefit — the forward model
has no held-out geography to protect. The flag is additive, defaults to today's behaviour, and
is ignored under `--exclude_fold` so it can never pull a held-out fold back into training.
Covered by `tests/test_train_all_splits.py`.
