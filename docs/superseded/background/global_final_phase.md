# The final phase — global extent

Written 2026-08-21, branch `ensemble`. **The model is accepted and frozen.** This phase does
not change it. The only fitting permitted is re-deriving, on global data, the parameters that
were previously estimated from Africa or southern Africa subsamples.

Read `docs/ensemble_model_outline.md` for what the system is and `CLAUDE.md` for the rules.

---

## 0. What is frozen, and what is not

**Frozen — do not touch.** The ConvLSTM architecture and its four flags, the fold mask, the
class structure of the calibration, the marginal family, the field model, and every gate
definition in the scorecard.

```
--central_residual True --central_context True --monotone_quantile_width True --quantile_context True
fold mask: data/raw/hm_global/fold_mask_b4_1000.tif   (512 px blocks, k=5)
calibration: ONE layer, distance band × predicted change × HM level, monotone in horizon
marginal: empirical shape per (horizon × band), pooled body, u_bound 0.999 / 0.025
```

**The five fold checkpoints already exist and are already global.** They were trained on
global chips with the 4x4 mask; only the *prediction extent* was Africa.
`data/ensemble/exp/c1_foldb4/FOLD_CHECKPOINTS.json` holds the exact paths.

**This is the single most important fact for planning this phase: step 2 needs no retraining.**
`run_hindcast_folds.py --fold_checkpoints '1=…,2=…' --max_epochs 0` is the predict-only path
and it exists for exactly this. Budget ~121 min per fold for global prediction, ~5 h for k=5
across two GPUs — against ~10 h if anyone retrains by reflex.

## 1. Step 1 — tune to the global extent

Everything below was estimated on a regional subsample and must be re-derived from global
data. Nothing else changes.

| parameter | where it came from | how to re-derive | why it matters |
|---|---|---|---|
| **AR(1) ρ** | Africa residuals: `{10: 0.650, 15: 0.461, 20: 0.433}` | `horizon_autocorrelation` on the global residual manifest | Every regional card before the last one silently used the 0.9 fallback and scored T4.1 against a NaN. The estimator now reads evenly spaced full-width row stripes and is seed-free; `tests/test_horizon_autocorrelation.py` guards it. |
| **Field spectrum / variogram fits** | `fit_field_spectra.py` on Africa residuals | re-fit on global residuals | The residual's practical range was 99–166 px on Africa and 135 px globally; close, but the spectrum feeds T3 directly. |
| **Calibration factors** | one layer fitted on Africa residuals, bands 0–5 | re-fit on global residuals | 846 ecoregions and 5.2× the valid pixels means classes that were thin on Africa are populated globally. **Re-check which bands to correct** — see §1.1. |
| **Marginal `u_bound`** | swept on Africa: 0.999, and 1.0 is identical | re-sweep with `predict_change_rates.py --sweep_u_bound` on global residuals | 30 s per candidate in closed form. The lever saturated on Africa; confirm it saturates globally too. |
| **`VAL_STRIDE`** | 1024, chosen for the 4x4 mask on Africa | not needed unless retraining | Only relevant if anyone retrains. If so, check the validation fold's *change content*, not its chip count (rule 23). |
| **`--mem_budget_gb`** | 20 on Africa | 60 globally, and **verify** | `docs/global_scorecard.md` §9 records T3's `build_z_field` overshooting its own budget by 13% at 67.77 GB. That stage is not bounded as tightly as claimed. |
| **`--block_sizes`** | 1,10,100 on Africa | **1,10,100,1000** | The 1000 km rows exist on no regional card. |
| **`--score_points`, T3 sampling** | tuned regionally | verify the clustered sampler still finds live pairs at global width | Sampling 1500 points uniformly over 17111 × 40000 leaves ~26 of 20000 pairs within 500 px. |

### 1.1 The one calibration decision that must be re-taken globally

On Africa the layer corrects **all six distance bands**. That decision rests on a measurement
that does not transfer cleanly:

- Correcting every band gave far-band `P(Δ>0.05)` = **1.36×** observed — good.
- But it overshot at **10–30 px, 2.28×**, which is the one T8.1 row that fails.
- And the far-band *decrease* rows rest on **one observed event on Africa** against 88 globally.

So globally there is both more evidence and a different balance. **Re-run the band sweep**
(`--bands` variants, fitted on 3 folds and scored on 2) with `scripts/band_tail_audit.py`
alongside, and choose on two criteria, never one:

1. held-out interval score, and
2. **the far-band p99.9 half-width must not collapse** — that is the tail carrying the entire
   far-field change signal, and no pooled metric can see it.

The obvious candidate to test is excluding band 3 (`0,1,2,4,5`), which was never tried.

## 2. Step 2 — global hindcasts

Predict the five existing checkpoints over the full 17111 × 40000 grid, stitch on `holdout`.

```
run_hindcast_folds.py --stage train --max_epochs 0 \
    --fold_checkpoints "$(paths from FOLD_CHECKPOINTS.json)" \
    --fold_mask data/raw/hm_global/fold_mask_b4_1000.tif \
    --region config/region_global.geojson --windows all
run_hindcast_folds.py --stage stitch --fold_mask <same> ...
```

Verification invariants, from `docs/global_hindcast_execution.md`: the stitched raster must
cover 184,573,321 valid pixels, and every pixel must come from the fold that held it out.

## 3. Step 3 — the global scorecard

Full T1–T8 at M=400. **Run the M=4 smoke tier first** — it is 13 min and it has already caught
one broken chain this round.

Budget, from the previous global run: ~30 h for the scorecard, 52 min per ensemble, ~450 GB
per ensemble on disk. T2 is 58% of the run and the fourth block scale is most of why.

## 4. Step 4 — hindcast deliverables

From the 1990–2000 input window, targets 2005/2010/2015/2020:

- the **ensemble** as icechunk (`members`, one member per chunk, `(1,1,1024,1024)`);
- **central, lower and upper** as cloud-optimized GeoTIFF.

These are the *calibrated* ConvLSTM outputs — after the unified calibration layer, which is
what the ensemble is built from. `scripts/make_cogs.py` exists for the COG conversion.

**Use `--stitch_mode holdout` for anything scored.** Products and display rasters may use
`mean`, which is seamless but in-sample; never score a mean-stitched raster.

## 5. Step 5 — the 2025–2040 forecast

The forward product has no fold mosaic and no seam, because there is no held-out geography to
stitch. **The decision already taken: a sixth model trained on all chips**, with the fold mean
reserved for display.

Same deliverables as step 4: ensemble as icechunk, central/lower/upper as COG.

---

## What this phase must not do

- **Do not tune the model to pass a gate.** Three gates are known to be failing for reasons
  that are not the model's: T5.1 (an open instrument question), T8.4 in the far band (88
  observed events globally), and T3.2 (a target now derived from the data's own structure
  budget, at 0.104 of it).
- **Do not carry a regional fit forward.** Every parameter in §1 must be re-derived from global
  residuals. Carrying one over is the failure mode that produced the most convincing wrong
  numbers in this project.
- **Do not screen anything on southern Africa.** Its observed far-field change rate is exactly
  0.0000, so any far-field measurement reads as "no effect" there whatever it did.
