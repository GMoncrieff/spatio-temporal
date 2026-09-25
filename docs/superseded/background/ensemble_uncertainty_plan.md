# Spatiotemporal Residual Ensemble — Implementation Plan

## Context

The ConvLSTM model (`src/models/spatiotemporal_predictor.py`, `src/models/lightning_module.py`) already produces a deterministic central forecast and per-pixel 2.5/97.5% quantile intervals for global Human Modification (HM) at four horizons (+5/+10/+15/+20yr). Those intervals are calibrated *per pixel on average* (pooled validation coverage ~93–97% — an unconditional figure whose adequacy this revision no longer takes on trust; see the amended constraint below), but nothing about the model enforces spatial or temporal coherence across pixels. When these forecasts are aggregated — country means, area above an HM threshold, 2025→2040 change — independent per-pixel noise mostly cancels out, so naively propagating pixel-level uncertainty into an aggregate statistic understates the true uncertainty badly (coverage collapses toward 0 at large spatial scales). This is the motivating problem the ensemble layer fixes.

The fix is a **post-hoc statistical layer**, not a retrained model: the frozen checkpoint's central forecast is preserved exactly (the median of the new ensemble must reproduce it), and *joint spatial/temporal structure* is added on top by sampling realistic correlated error fields — calibrated from the model's own historical out-of-sample errors — and pushing them through per-pixel quantile marginals via a Gaussian copula. No changes to `train_lightning.py`'s loss functions or `src/models/*` architecture; all new code lives in a new `src/ensemble/` package plus a few new/extended scripts.

### Amended constraint on the quantile heads (user decision, this revision)

The original brief required the ensemble's pixelwise spread to reproduce the existing 2.5/97.5 heads *unchanged*. That is now **conditional on those heads passing a class-conditional coverage audit**.

The reported ~93–97% validation coverage is a single pooled number over all valid pixels per horizon (`src/models/lightning_module.py` lines 403–407 — `within_interval / mask_h.sum()`, no stratification anywhere in the repo). It says nothing about whether coverage holds *within* subpopulations, and the arithmetic says it can hide a severe failure. Measured on the real data (18.9M sampled pixels, 2000→2020): only **6.8%** of pixels change by more than 0.05 HM and only **2.7%** by more than 0.1; the median 20-year change is 0.0001. So a pooled 95% coverage is *exactly consistent* with 50% coverage on the high-change 2.7% — `0.973 × 0.9625 + 0.027 × 0.50 = 0.95`. The pooled number is dominated by a static majority and is nearly blind to the subpopulation the forecast is actually used for.

There is also direct evidence the error scale is not constant across classes: the conditional standard deviation of 20-year ΔHM rises monotonically with baseline HM — 0.020 (HM<0.01) → 0.037 → 0.050 → 0.048 → 0.052 (HM>0.6), a **2.6× range**. A single globally-trained pinball head has no mechanism to track that, and pooled pinball loss gives it no incentive to.

**Phase 1.5 (new)** therefore measures per-class coverage on the out-of-sample hindcast residuals, compares quantile-head half-width to hindcast residual spread per horizon × class, and decides **keep / global rescale / stratified rescale** from that comparison. Any rescale is a documented, auditable, few-parameter multiplicative transform of the *existing* heads (the identity if the audit says keep) — not a retrained head, no touching the loss code, and the central forecast is never modified. The constraint becomes: **ensemble median ≡ current central forecast (unchanged); ensemble pixelwise 2.5/97.5 ≡ the recalibrated quantile rasters, with the recalibration itself justified by a measured coverage table.**

Decisions confirmed with the user for this first build pass:
- **k=5** spatial folds for the out-of-sample residual harness (Phase 0).
- **Add `gstools`** as a new dependency for variogram fitting (Phase 1), rather than hand-rolling.
- **Source RESOLVE Ecoregions2017** and rasterize it to the project's 1km/EPSG:4326 grid (Phase 1a) — now a critical-path dependency, not a nice-to-have.
- **Ecological reporting framing**: ecoregion / biome / realm are the primary aggregation units for validation, and biome/realm remain the variogram and calibration strata. Coverage is *scored* at ecoregion level (n≈846, where the statistics support a ±0.05 target) and *reported* at biome/realm level with explicit confidence intervals — same raster, same pixels, different roll-up. Country-level is kept only as an optional unscored cross-check.
- **Single-tier field generation, 50 members** to start (Phase 2/3) — skip the two-tier 10km/1km optimization for now; validate correctness first, scale up later.
- **Prototype everything in the southern-Africa region** (`config/region_to_predict_small.geojson`) before running at global scale, at every phase.
- **The existing quantile heads are modifiable** if the class-conditional audit says so — post-hoc rescale only, never retraining or loss-code changes.
- **Class-conditional coverage is a first-class target metric**, not a diagnostic footnote — see the target-metrics section below.

---

## Grounded facts this plan relies on (verified directly against the code)

- **Model I/O**: `SpatioTemporalPredictor.forward(input_dynamic [B,3,C_d,H,W], input_static [B,C_s,H,W], lonlat=None)` → `[B,12,H,W]`. Channel order: `[lower_5,central_5,upper_5, lower_10,central_10,upper_10, lower_15,central_15,upper_15, lower_20,central_20,upper_20]`. All 4 horizons are produced from a single forward pass regardless of which 3-year input window is used — only the *target years* differ per window.
- **Gradient isolation** (`src/models/lightning_module.py`, manual optimization, `automatic_optimization=False`) is out of scope — do not touch.
- **Frozen-checkpoint inference pattern**: `SpatioTemporalLightningModule.load_from_checkpoint(ckpt_path).eval()` under `torch.no_grad()`.
- **Normalization stats gap**: `hm_mean/hm_std`, `comp_means/comp_stds`, `static_means/static_stds` are plain attributes copied post-hoc from a live `HumanFootprintChipDataset` (`train_lightning.py` lines ~435–440, ~489–493), **not** persisted in the `.ckpt`. Every new inference entrypoint must either re-derive them (same `random_seed=42`, reproducible) or persist them once as a JSON sidecar per checkpoint — **this plan persists them as a sidecar** for the new ensemble scripts, to avoid repeatedly paying the raster-sampling cost.
- **Global grid**: 17111 rows × 40000 cols, EPSG:4326, ~0.009°/px (~1km), bounds lon `[-180,180]` lat `[-70,84]`. Source rasters: `data/raw/hm_global/HM_{year}_AA_1000.tiff` (target), `HM_{year}_{VAR}_1000.tiff` (10 covariates), `hm_static_{var}_1000.tiff` (7 static vars).
- **HM distribution confirmed** (measured: 400 random 256×256 windows, 18.9M valid pixels, `HM_2000_AA_1000.tiff` / `HM_2020_AA_1000.tiff`): definitionally bounded `[0,1]`; sampled max 0.975 in a 300-window draw, with rare pixels reaching 1.0 in a larger draw — **use 1.0 as the quantization bound, not the 0.94 figure an earlier smaller sample suggested**. Exact-zero mass **1.38%** (real atom, not sensor floor). Heavy right skew. No existing transform code anywhere in the repo (no logit/rank-Gaussian/hurdle) — this is genuinely new work.
- **HM change distribution confirmed** (same sample, 2000→2020, i.e. the h=20 analogue): mean Δ = +0.0090, sd 0.0344, median Δ = 0.0001. Tail fractions: `|Δ|>0.001` → 45.3%, `>0.01` → 20.7%, `>0.05` → **6.8%**, `>0.1` → **2.7%**, `>0.2` → 0.68%. Quantiles: q90 = 0.029, q99 = 0.167, q99.9 = 0.344. **These numbers set the change-bin edges in Phase 1.5 and are the quantitative basis for doubting the pooled coverage figure.**
- **Error scale is class-dependent** (same sample): conditional sd of 20-year ΔHM by baseline-HM bin — `[0,0.01)` n=53.3% sd=0.0199 · `[0.01,0.1)` n=24.1% sd=0.0375 · `[0.1,0.3)` n=11.9% sd=0.0504 · `[0.3,0.6)` n=9.8% sd=0.0479 · `[0.6,1]` n=0.89% sd=0.0517. A **2.6× spread range** across classes that a single pooled pinball objective has no mechanism to track.
- **Coverage is currently computed pooled and unconditional**: `src/models/lightning_module.py::validation_step` lines 403–407 computes one coverage scalar per horizon over all valid pixels, logged as `val_coverage_{5,10,15,20}yr`. `scripts/train_lightning.py` has no coverage computation of its own and no stratified evaluation of any kind (`grep` for `stratif|decile|by_bin` → nothing). Class-conditional coverage has therefore **never been measured** on this model — Phase 1.5 measures it for the first time.
- **CONFIRMED BUG**: `HumanFootprintChipDataset.__getitem__` (`scripts/torchgeo_dataloader.py`) only applies split-mask filtering (`valid_split_positions`) in **random** mode (line ~283). In **grid** mode — the default for val/test (`--val_mode grid`) — `chip_positions` is built by walking the *entire* H×W grid with no split-mask filtering at all (lines ~229–236). Today, grid-mode val/test evaluation silently runs over the whole globe, not the held-out split. **Must fix before Phase 0** or fold-holdout residuals won't actually be out-of-sample.
- **CONFIRMED gap**: `_predict_region_and_write` (`scripts/train_lightning.py`, ~lines 1229–1662) hardcodes only two input-year windows via `--predict_final_year ∈ {2020, 2040}` (lines ~1268–1277). Phase 0 needs all 4 windows: `[1990,95,00]`, `[1995,00,05]`, `[2000,05,10]`, `[2005,10,15]`.
- **CONFIRMED output collision risk**: GeoTIFFs are written as `data/predictions/prediction_{target_year}_{quantile}_blended.tif` — keyed only by calendar year and quantile, with no window/fold discriminator. Multiple windows produce predictions for the same target year (e.g. year 2020 is a target of all 4 windows); multiple folds each produce a full global raster. Needs a filename prefix/discriminator or runs will clobber each other.
- **Region files**: `config/region_to_predict.geojson` and `config/region_to_predict_small.geojson` are **identical** (southern Africa bbox, lon `[13.57,32.31]` lat `[-35.30,-27.25]`) — verified byte-for-byte equivalent geometry. `config/region_to_predict_large.geojson` bbox is exactly the full global raster extent (`[-180,-70,180,84]`) — this is the one to use for "global" runs, not the unsuffixed file.
- **No biome/ecoregion/realm data anywhere in the repo** (confirmed via grep + filename search) — must be sourced externally (RESOLVE Ecoregions2017, per user decision) and rasterized to the 1km grid.
- **No config framework** — plain argparse in `train_lightning.py` (~180 `add_argument` calls); `config/config.yaml` is vestigial (only `inference.region_geojson`/`inference.do_prediction` are actually read). New scripts should follow the same flat-argparse idiom, including the boolean-flag pattern `type=lambda x: (str(x).lower()=='true'), nargs='?', const=True, default=...` and the additive-backward-compatible-flag precedent set by commit `204553f`.
- **Dependencies already available**: numpy, scipy (incl. `scipy.fft`, `scipy.optimize`, `scipy.stats`), rasterio, xarray, zarr 3.1, torch, torchgeo, geopandas, shapely, pyproj, pandas, matplotlib, scikit-learn, statsmodels. **Not present** (to be added): `gstools` (variogram fitting, per user decision), and whatever's needed to read the RESOLVE Ecoregions2017 shapefile (geopandas already covers this).
- **`src/` conventions**: docstring-only `__init__.py` per package, relative imports internally, absolute `from src.ensemble.X import Y` externally via `sys.path.insert`. `src/evaluation/`, `src/utils/`, `src/training/` are empty stub packages following this exact pattern — `src/ensemble/` should match.
- **Test conventions**: flat `tests/` dir, pytest, `sys.path.insert(0, str(Path(__file__).parent.parent))` + absolute imports, synthetic-tensor-based tests. Good templates: `tests/test_pinball_gradient_isolation.py`, `tests/test_baseline_mae.py`.
- **Reusable fixtures already on disk**: `data/predictions/prediction_{year}_{central,lower,upper}_blended.tif` for years 2005–2040 across 3 region variants — direct ground truth for Phase 4's "ensemble median ≈ current central forecast" sanity check, no regeneration needed.

---

## Target metrics (the definition of done)

Every target is evaluated on **out-of-sample hindcast** data (Phase 0's fold-rotated residuals), **per horizon**, never on training pixels. "Class" always means a stratum defined from quantities available *at prediction time* — predicted change `Δ̂ = central_h − HM_t0`, observed `HM_t0`, biome, horizon — **never from the observation**, or the calibration cannot be applied to a real forward forecast.

Targets are stated as numbers so that "did this work?" has an answer. They are intentionally aggressive on class-conditional coverage (the point of this revision) and deliberately paired with a sharpness guard so the trivial solution — widen everything — fails.

### T1 · Per-pixel marginal calibration — owner Phase 1.5

| id | Metric | Target |
|---|---|---|
| T1.1 | Pooled coverage of the 95% interval, per horizon | `0.95 ± 0.01` |
| T1.2 | **Class-conditional coverage**, every primary cell (horizon × Δ̂-bin) with `n_eff ≥ 100` chips | `\|cov − 0.95\| ≤ 0.03`; **worst cell ≤ 0.05** |
| T1.3 | **High-change tail** (Δ̂ > 0.05 bin, and Δ̂ > 0.15 bin) at every horizon | `cov ≥ 0.92` |
| T1.4 | Secondary strata (HM-level bin, biome), cells with `n_eff ≥ 100` | `\|cov − 0.95\| ≤ 0.05` |
| T1.5 | **Sharpness guard** — mean interval width vs today's heads | `≤ +25%` globally, **and** pinball loss at 0.025/0.975 (already computed in `_compute_horizon_losses`) degrades `≤ 5%` relative |
| T1.6 | Monotonicity/bounds: `lower' ≤ central ≤ upper'`, both in `[0,1]` | 100% of valid pixels — **hard gate** |

T1.2 is the headline new target. T1.5 exists because inflating every interval until coverage passes is not a fix — if width is about to blow past +25%, the correct move is a *finer stratification*, not a bigger global factor.

### T2 · Aggregate-scale calibration — owner Phase 4 (the reason the project exists)

The reporting framing is **ecological** (user decision): ecoregion / biome / realm are the primary aggregation units. Because coverage is a binomial rate over polygons, the unit that gets *scored* and the unit that gets *reported* must differ — see the note below the table.

| id | Metric | Target |
|---|---|---|
| T2.1 | Coverage of block-mean HM at 10 km / 100 km / 1000 km | `0.95 ± 0.05` at each scale |
| T2.2 | **Scored** polygon coverage: ecoregion-mean HM (RESOLVE `ECO_ID`, n≈846) | `0.95 ± 0.05` |
| T2.3 | Coverage of **area-above-threshold** (`HM > 0.1`, `HM > 0.3`) at ecoregion level | `0.95 ± 0.05` |
| T2.4 | Coverage of **change** statistics (ecoregion-mean ΔHM between two horizons) | `0.95 ± 0.05` |
| T2.5 | Rank histogram at aggregate scale (ecoregion-mean, M members → M+1 bins) | flat: χ² GoF `p > 0.01`, no visible U-shape or dome; report reliability index alongside |
| T2.6 | **Reported, not scored**: biome (n=14) and realm (n=8) coverage | report point estimate **with Wilson CI**; no pass/fail threshold |

**Why ecoregion is the scored unit and biome is not.** Coverage over `n` polygons has a 95% CI half-width of roughly `1.96·√(0.95·0.05/n)`: **±0.114 at n=14 (biome)**, ±0.062 at n=8 (realm), **±0.015 at n=846 (ecoregion)**. A ±0.05 target is therefore meaningless at biome level — the CI is wider than the tolerance, so a biome-level number can neither pass nor fail honestly — but entirely meaningful at ecoregion level. Ecoregion is fully ecological, so this costs nothing in framing. Biome and realm remain the headline *reporting* units (T2.6) with explicit CIs attached; they are summaries for the reader, not gates. All three come from the same `ECO_ID` raster plus a lookup table, so there is no extra data cost.

Polygons are spatially correlated, so effective `n` is below the polygon count and all of the above is optimistic. Mitigate by (a) attaching Wilson intervals at every level rather than bare point estimates, and (b) bootstrapping over ecoregions *within* biome to put an honest band on each biome-level number in T2.6.

### Coverage and width are scored together, never coverage alone

T1.5 guards sharpness at the *pixel* scale, and only on the global mean — which a
stratified rescale can satisfy while inflating one class threefold, because the classes
that narrow pay for the classes that widen. Nothing above scores width at the aggregate
scale at all, so an ensemble could pass every T2 coverage target by being enormously wide.
These close that hole:

| id | Metric | Target |
|---|---|---|
| T1.5b | Per-class width ratio vs the original heads, every cell with `n_eff ≥ 100` | `≤ 2.5` (no class inflated without bound) |
| T2.7 | **Interval score** (Winkler, α=0.05) of the aggregate interval at each block scale and at ecoregion level | lower than **both** pixelwise baselines |
| T2.8 | Mean aggregate interval width, reported beside every coverage number | reported; must not exceed the mean-of-bounds baseline |
| T2.9 | **CRPS** of the ensemble's aggregate distribution vs observed | lower than both baselines |

The interval score `(u−l) + (2/α)(l−y)·1{y<l} + (2/α)(y−u)·1{y>u}` is a proper scoring
rule: widening is only rewarded when it buys back more miscoverage penalty than it costs
in width. It is the metric that makes "optimise coverage and width at the same time"
operational, and it is why T2 coverage alone is never sufficient evidence.

### T7 · Member realism and diversity — owner Phase 4

Coverage and scores are aggregates; they can all pass while individual members look
nothing like a plausible HM field. Members are therefore inspected directly, both by eye
and by number.

| id | Metric | Target |
|---|---|---|
| T7.1 | Rendered member change fields beside the observed change field, at global and regional extent, logged to W&B | visual: members show the same texture and spatial scale of change as the observation, no tiling, no isotropic blur where the truth is structured |
| T7.2 | Mean pairwise correlation between member change fields | `< 0.98` — members must actually differ |
| T7.3 | Spread-skill ratio: ensemble sd vs RMSE of the ensemble mean | `1.0 ± 0.25` (under-spread ensembles score < 1) |

Always report the **pixelwise-independent-propagation baseline** at the same scales as the before/after — the expected baseline is collapse toward 0 coverage, and that contrast *is* the motivating figure for the whole project. Global-extent aggregation is `n=1` per horizon: report it, don't score it. T2.3 and T2.4 matter more than T2.1/T2.2 for the downstream use and are strictly more sensitive to spatial structure than a mean; T2.4 is the hardest case because it depends on getting the *between-horizon* error correlation right, which is exactly the AR(1) coupling's job.

**Optional secondary (not scored, skip if it costs anything):** country-level (Natural Earth admin-0) coverage as a cross-check. The original brief named "country/ecoregion means", the zonal-stats code path is identical to the ecoregion one, and the download is small — so it is close to free once T2.2 works. Explicitly not a target metric, and not a reason to delay anything.

### T3 · Spatial realism — owner Phase 2, verified in Phase 4

| id | Metric | Target |
|---|---|---|
| T3.1 | Ensemble-member semivariogram vs the fitted residual variogram, on ≥3 biome strata | sill within **15%**, practical range within **25%**, nugget/sill ratio within **0.10 absolute** |
| T3.2 | **Variogram score** (Scheuerer & Hamill, p=0.5) vs held-out hindcast truth | **≥30% lower** than an independent-pixel ensemble with identical marginals |
| T3.3 | **Energy score** | beats both the independent-pixel null and a degenerate single-member baseline |
| T3.4 | Radially-averaged power spectrum of members vs residual spectrum, 10–1000 km band | within a factor of **1.5** (log-log) |
| T3.5 | Artifacts: ±180° lon seam, tile edges, calibration-bin-boundary discontinuities | none visible on a global render or a seam-differenced render — **hard gate** |

The **independent-pixel ensemble is the honest null** for T3.2/T3.3: same marginals, zero spatial structure, so any improvement is attributable purely to the copula's correlation and nothing else. T3.4 catches the failure where sill and range both nominally "fit" but the spectral shape between them is wrong.

### T4 · Temporal coherence — owner Phase 2

| id | Metric | Target |
|---|---|---|
| T4.1 | `corr(z_h, z_{h−5})` of the residual field vs hindcast residual horizon autocorrelation | within `±0.10` at each step |
| T4.2 | Spread non-decreasing with horizon: `σ(5) ≤ σ(10) ≤ σ(15) ≤ σ(20)` | ≥99% of valid pixels |

T4.2 is a physical sanity constraint that a per-horizon-independent recalibration can easily violate — worth checking explicitly after Phase 1.5, not just after Phase 2.

### T6 · Change-sign realism — owner Phase 3, verified in Phase 4 (added after the global audit)

The T1/T2 targets score whether the *observation* falls inside the interval. Nothing in
them scores whether the members themselves are physically plausible, and there is a
specific way this project can fail that test.

HM decreases are real but rare, and the large decreases are very rare — measured on 300
random 256×256 windows (5.8M valid px per horizon):

| horizon | P(Δ<−0.01) | P(Δ<−0.05) | P(Δ<−0.15) | P(Δ>+0.15) | tail asymmetry |
|---|---|---|---|---|---|
| +5yr | 1.91% | 0.231% | 0.0132% | 0.05% | 3.8× |
| +10yr | 2.13% | 0.335% | 0.0234% | 0.22% | 9.4× |
| +15yr | 4.40% | 0.360% | 0.0285% | 0.58% | 20× |
| +20yr | 4.48% | 0.399% | 0.0309% | 0.93% | **30×** |

The Phase 1.5 audit found that the high-change classes miss almost entirely on the *lower*
side (66% of misses at h=5 in the Δ̂>0.15 bin fall below the lower bound), so conformal
recalibration widens σ_left exactly where change is large. The Phase 3 marginal is a
median-spliced two-piece normal — Gaussian tails on both sides — so an inflated left scale
mints large negative changes that the real world produces ~30× less often than the
positive ones. Coverage targets alone would applaud this.

| id | Metric | Target |
|---|---|---|
| T6.1 | `P(Δ_member < −0.01)` vs observed, per horizon | ratio in `[0.5, 2.0]` |
| T6.2 | `P(Δ_member < −0.05)` vs observed | ratio `≤ 3`, and absolute `≤ 1.5%` |
| T6.3 | `P(Δ_member < −0.15)` — the very rare tail | ratio `≤ 5`, and absolute `≤ 0.2%` |
| T6.4 | Tail asymmetry `P(Δ>+0.15) / P(Δ<−0.15)` | `≥ 0.5 ×` the observed ratio |
| T6.5 | 1st and 5th percentile of the member change distribution vs observed | within a factor of 2 |

Δ is always `member − HM_t0` for the member, `observed − HM_t0` for the reference, on
identical pixels. If T6 fails, the fix is **not** to narrow the marginals globally (that
breaks T1) — it is to make the lower tail of the marginal respect the physical floor, e.g.
truncating the left side at a horizon-dependent maximum plausible decrease, which changes
the marginal family rather than its calibration.

### T8 · Change is clustered near past change — owner Phase 1.5/3, verified in Phase 4

Measured on southern Africa (1.86M valid px), future change (2000→2020) as a function of
distance to the nearest pixel that changed by more than 0.01 in the past (1990→2000):

| distance | n_px | P(Δ>0.01) | P(Δ>0.05) | P(Δ>0.15) |
|---|---|---|---|---|
| 0–1 px | 91,164 | 0.676 | 0.222 | 0.0229 |
| 1–3 px | 183,371 | 0.355 | 0.080 | 0.0069 |
| 3–10 px | 332,821 | 0.116 | 0.023 | 0.0024 |
| 10–30 px | 419,443 | 0.031 | 0.0068 | 0.0009 |
| 30–100 px | 341,269 | 0.005 | 0.0010 | 0.0001 |
| >100 px | 493,240 | **0.0000** | **0.0000** | **0.0000** |

Beyond ~100 px from any past change, not one of 493,240 pixels moved by more than 0.01 in
twenty years. Change is overwhelmingly a near-neighbour phenomenon: hard to place exactly,
but scattered around where it has already happened.

The ensemble has no mechanism that knows this. The correlated field is stationary and the
per-pixel spread comes from classes defined by predicted change, HM level and biome — none
of which encode proximity to past change — so members sprinkle change into remote stable
country where the real world produces none.

**Distance to past change is computable from the input years alone**, so it is admissible
as a Phase 1.5 stratum under the same rule as every other class ("available at prediction
time, never from the observation").

| id | Metric | Target |
|---|---|---|
| T8.1 | `P(Δ_member > 0.05)` by distance-to-past-change band, vs observed | ratio in `[0.5, 2.0]` in every band with `n_eff ≥ 100` |
| T8.2 | Remote band (>100 px): `P(Δ_member > 0.05)` | `≤ 0.002` (observed is 0) |
| T8.3 | Ratio of `P(Δ>0.05)` between the nearest and remote bands | `≥ 20×` (observed: unbounded; the pixelwise heads must not flatten it) |
| T8.4 | Same three, for the *lower* tail `P(Δ < −0.05)` | ratio in `[0.5, 3.0]` per band |

If T8 fails, the fix is to add the distance band to the Phase 1.5 class definition so the
conformal factors can collapse intervals in remote stable areas — not to shrink the
correlation range, which would break T2.

### T5 · Consistency / regression gates — hard gates, owner Phase 3/4

| id | Metric | Target |
|---|---|---|
| T5.1 | Ensemble median vs central forecast raster | equal to within int16 quantization (`≤ scale/2`) at **100%** of valid pixels |
| T5.2 | Ensemble 2.5/97.5 percentiles vs the (possibly recalibrated) quantile rasters | equal to within Monte-Carlo error for M members (see the M=50 tail-resolution note in Phase 3) |
| T5.3 | NaN/nodata leakage | ensemble valid-pixel mask **identical** to the central forecast's |

### Which knob fixes which failure

The marginals and the correlation structure are **orthogonal** controls, and the order of tuning matters: the copula preserves marginals exactly, so once T1 passes, aggregate-scale coverage is a pure function of the correlation structure. Tune marginals first (Phase 1.5), then correlation (Phase 1c → Phase 2). Never re-widen marginals to fix an aggregate-scale miss — that double-counts and breaks T1.

| Symptom | Turn this knob | Do **not** |
|---|---|---|
| T1 fails (pixel/class coverage off) | Phase 1.5 rescale factors `ŝ(class, horizon)` | widen by changing the variogram |
| T1 passes, T2 under-covers (aggregate too narrow) | raise long-range weight / range in the fitted variogram | re-widen the marginals |
| T2 over-covers (aggregate too wide) | lower long-range weight; check the nugget fraction isn't under-estimated | narrow the marginals |
| T3 fails while T1/T2 pass | spectral shape wrong — add a third range or change kernel family (Phase 2) | add nugget to compensate |
| T4.1 fails | re-estimate AR(1) `ρ_h`; consider `ρ` varying by stratum | |
| T4.2 fails | enforce monotone spread across horizons as a constraint on `ŝ` in Phase 1.5 | |
| T1.5 sharpness blows out | use a finer stratification | use a larger global factor |
| T6 fails (too many large decreases) | truncate the marginal's left tail at a physical floor (Phase 3 marginal family) | narrow the marginals globally — that breaks T1 |
| T2 coverage passes but T2.7 interval score loses to a baseline | the interval is buying coverage with width; tighten the correlation structure | accept it because coverage passed |
| T7.2 members nearly identical | nugget fraction too low / field seeds correlated | add width to compensate |
| T7.3 spread-skill < 1 (under-spread) | marginals too narrow *for the aggregate* — check the correlation structure first | widen pixel marginals before checking T2 |

---

## Phase 0 — Residual harness

**New**: `src/ensemble/__init__.py`, `src/ensemble/residuals.py`, `scripts/run_hindcast_folds.py`
**Extended (additive, backward-compatible)**: `scripts/torchgeo_dataloader.py`, `scripts/train_lightning.py`, `scripts/create_validity_mask.py`

### 0a. Fix grid-mode split-mask filtering (prerequisite)

In `HumanFootprintChipDataset.__init__` (`scripts/torchgeo_dataloader.py`), the grid-mode `chip_positions` precompute (~lines 229–236) currently ignores the split mask entirely. Fix by checking `(split_data[i:i+chip_size, j:j+chip_size] == self.split_value).any()` directly against the split raster for each candidate `(i,j)` at the given `stride` (mirrors the existing random-mode logic at line ~221, but works at arbitrary stride instead of requiring `stride == chip_size` alignment). No effect when `split_mask_file is None` (existing callers unaffected). This changes what current val/test metrics *mean* going forward (shrinks from whole-globe to true held-out geography) — worth a one-line note in the PR description, not a behavior anyone should be relying on today.

### 0b. Extend the split mask to k=5 rotating folds

In `scripts/create_validity_mask.py`, factor the chip-position enumeration in `create_spatial_splits` (~lines 100–110: `CHIP_SIZE=128`, `MIN_VALID_RATIO=0.2`) into a shared helper, and add `create_kfold_splits(valid_mask, profile, transform, crs, k=5)`: same chip enumeration, same fixed `RANDOM_SEED=42` shuffle, but sliced via `np.array_split` into 5 groups instead of ratio cuts. Writes `data/raw/hm_global/fold_mask_1000.tif` (uint8: `0=invalid, 1..5=fold id`) and a `fold_manifest.csv` (fold_id, n_chips, n_pixels). Leaves the existing `split_mask_1000.tif` (70/10/10/10) untouched — that's a separate artifact for the production checkpoint already trained.

Add an `exclude_split_values` kwarg to `HumanFootprintChipDataset.__init__` (optional, default `None`, fully backward compatible) so a fold-CV training run can select "everything except fold f" as its training pool: `keep = np.isin(chip, exclude_split_values, invert=True).all()` in the split-position precompute, alongside the existing `split_value`-based include logic.

### 0c. Generalize `_predict_region_and_write` to arbitrary input windows + avoid filename collisions

In `scripts/train_lightning.py`, add two new CLI args:
- `--predict_input_years` (comma-separated 3 years, e.g. `"1995,2000,2005"`) — when set, overrides the legacy `--predict_final_year` binary branch (~lines 1268–1277); `target_years = tuple(base_year + h for h in (5,10,15,20))` filtered to `<= 2020`.
- `--predict_output_prefix` — prepended to output GeoTIFF filenames (default `None` reproduces today's exact filenames).
- `--predict_all_windows` (bool) — when set, loops `_predict_region_and_write` over all 4 windows within a single process/checkpoint load, so fold orchestration doesn't reload the model 4x per fold.

All additive; default behavior is byte-identical to today. This follows the same pattern as commit `204553f` (new flags + explicit backward-compat).

### 0d. `run_hindcast_folds.py` — orchestration

Subprocess-based (not an import-internals reimplementation): for each fold `f` in `1..5`, shell out to `python scripts/train_lightning.py` with `--fold_mask data/raw/hm_global/fold_mask_1000.tif --exclude_fold f --run_full_set_evaluation False --run_large_area_prediction True --predict_all_windows True --predict_region config/region_to_predict_large.geojson --predict_output_prefix fold{f}`. Reusing the *actual* training CLI (not a slimmed-down reimplementation) is deliberate: it guarantees fold-CV models are trained with identical hyperparameters/loss weights/LR schedule to whatever produced the production checkpoint, which matters for later median/spread consistency checks.

After all 5 folds complete:
- `stitch_fold_predictions()`: for each `(window, horizon, quantile)`, each fold's GeoTIFF covers the *whole* global extent (fold membership only controlled training data, not prediction extent) — stitch by taking, at each pixel, the prediction from the one fold whose training excluded that pixel's fold id (read `fold_mask_1000.tif`, select fold `f`'s raster values only where `fold_mask == f`). Produces one genuinely-out-of-sample global raster per `(window, horizon, quantile)`.
- `compute_residuals()`: writes **two** residual rasters per `(window, horizon)`, and this distinction is load-bearing —
  - `R_native = Y_observed − Y_pred_central` in **raw HM units**. This is what Phase 1.5 calibrates against, because the rescale is applied to the published quantile rasters, which live in HM units.
  - `R_z = transform(Y_observed) − transform(Y_pred_central)` in **rank-Gaussian space** (see 0e). This is what Phase 1c's variogram and Phase 2's field generator use, because correlated-Gaussian-field machinery needs an approximately Gaussian, homoscedastic residual.
  
  Keeping both is cheap and avoids the classic error of fitting a spread correction in one space and applying it in another. Both written to `data/ensemble/residuals/` as GeoTIFF, NaN where inputs are invalid (mirroring `_predict_region_and_write`'s own nodata handling).
- Alongside each residual raster, write the **class covariate rasters** needed by Phase 1.5 so it never has to recompute them: `dhat = central_h − HM_t0` (predicted change), `HM_t0` (baseline level), and the interval half-widths `w_up = upper − central`, `w_lo = central − lower` from the same stitched fold predictions. All four are already in memory at this point in the loop.
- Appends to `data/ensemble/residuals/manifest.csv` (fold_id, input_years, target_year, horizon, n_valid_px, path_native, path_z, path_dhat, path_wup, path_wlo).

**Southern-Africa smoke test first**: run with `--k 2`, one window, `--predict_region config/region_to_predict_small.geojson`, `--max_epochs 1` to validate the full pipeline (fold splitting, subprocess orchestration, stitching, transform, residual writing) cheaply before committing to the full k=5 × 4-window × global-extent run.

### 0e. Rank-Gaussian transform with explicit zero-atom handling

`src/ensemble/residuals.py`:
```
fit_rank_gaussian_transform(observed_hm_sample) -> RankGaussianTransform
  .forward(x) -> z   # HM [0,1] with exact zeros -> standard normal
  .inverse(z) -> x
```
Two-part (hurdle) design:
1. Estimate `p0 = P(X=0)` from a sampled fit (reuse the same windowed-random-read sampling pattern as `HumanFootprintChipDataset.__init__` lines ~105–127, not a full 150M-pixel load).
2. Exact-zero pixels: map to `z = Φ⁻¹(u)` where `u ~ Uniform(0, p0)`, seeded deterministically per-pixel (e.g. from row/col) so the mapping is reproducible, not resampled every call. (A fixed constant `Φ⁻¹(p0/2)` was considered but rejected — it creates a degenerate spike in the residual distribution that would break Phase 1's variogram fitting at zero-lag.)
3. Positive pixels: empirical CDF over positive values only (`F_pos`, via an interpolated quantile grid, ~1000–2000 knots — distribution-free, no parametric assumption), rescaled into `(p0, 1)`: `u = p0 + (1-p0)*F_pos(x)`, `z = Φ⁻¹(u)`.
4. `inverse`: if `Φ(z) <= p0` → `0.0`; else invert the rescaled quantile through `F_pos`.
5. Explicitly **not** raw logit (undefined at exact 0/1, doesn't address the atom) — matches the hard constraint.

### Testing (Phase 0)

`tests/test_ensemble_residuals.py` (pytest, `sys.path.insert` convention):
- Roundtrip: `inverse(forward(x)) ≈ x` off the zero atom; exact-zero pixels roundtrip to exactly 0.
- Zero-atom quantile: fraction of `z < Φ⁻¹(p0)` matches `p0` on synthetic data.
- `create_kfold_splits`: folds non-overlapping, cover 100% of valid chips exactly once.
- Grid-mode split filtering: synthetic raster + split mask, assert post-fix `chip_positions` only include chips overlapping the requested split value.

---

## Phase 1 — Diagnostics

**New**: `src/ensemble/validate.py`, `src/ensemble/variogram.py`, `scripts/prepare_ecoregions.py`
**New dependency**: `gstools` (add to environment)
**New data acquisition**: RESOLVE Ecoregions2017 shapefile, rasterized to the project's 1km/EPSG:4326 grid

### 1a. Ecoregion/biome/realm raster prep — **on the critical path**

This step went from a nice-to-have to a dependency of T2.2–T2.6, Phase 1c's stratification, and Phase 1.5's secondary stratum. Schedule it **first, running concurrently with Phase 0's fold retrainings** — Phase 0 is the long pole and occupies GPUs, while this is CPU/IO work that can proceed in parallel. Nothing downstream of Phase 1 can start without it.

`scripts/prepare_ecoregions.py`: download/read the RESOLVE Ecoregions2017 shapefile (geopandas), rasterize onto the same 17111×40000 grid as the HM rasters (`rasterio.features.rasterize` against the reference transform from `HM_2020_AA_1000.tiff`).

Write **one raster plus a lookup**, not three rasters:
- `data/raw/hm_global/ecoregion_id_1000.tif` — `uint16` of `ECO_ID` (846 terrestrial ecoregions; `0` = nodata/ocean). uint16 covers the ID range with room to spare.
- `data/raw/hm_global/ecoregion_lookup.csv` — `ECO_ID, ECO_NAME, BIOME_NUM, BIOME_NAME, REALM`.

Aggregating to biome (14) or realm (8) is then a join, not a re-rasterization — which matters because T2.2 scores at ecoregion level while T2.6 reports at biome/realm level, and both must come from exactly the same pixels or the numbers won't reconcile. It also keeps a single ~1.4 GB uint16 raster instead of three.

Validate the rasterization before relying on it: assert every non-zero `ECO_ID` appears in the lookup, report the count of valid HM pixels falling on `ECO_ID == 0` (coastline mismatch between the HM grid and the RESOLVE polygons is expected and needs to be quantified, not discovered later as mystery missing area), and check the rasterized area per ecoregion against the shapefile's own geometry area within a few percent.

One-time prep step; not part of the hindcast loop.

### 1b. Coverage vs. aggregation scale

`src/ensemble/validate.py::compute_block_coverage(pred_lower_path, pred_upper_path, observed_path, block_sizes, mask_path=None)`:
- Numeric block sizes (10km, 100km, 1000km): block-mean aggregation of the existing 1km central/lower/upper rasters and the observed raster (post-hoc spatial aggregation, not re-running the model).
- Zonal aggregation (ecoregion → biome → realm → global): accumulate per-`ECO_ID` sums in a single pass over the 1a raster (`np.bincount` on the ID raster weighted by the value raster — far cheaper than 846 separate `rasterio.mask` calls at this grid size), then roll up to biome/realm through `ecoregion_lookup.csv`. Same pass yields the area-above-threshold counts for T2.3.
- Empirical coverage per scale = fraction of blocks/polygons where `block_mean(observed)` falls in `[block_mean(lower), block_mean(upper)]`, reported **with a Wilson interval** so the n=14 biome number is visibly less certain than the n=846 ecoregion number. Expected: ~0.95 at pixel scale, collapsing toward 0 at large scale — this *is* the motivating figure, computed once on Phase 0's stitched hindcast rasters.
- Optional secondary: the same routine against Natural Earth admin-0 for a country-level cross-check (not a target metric — see T2's optional-secondary note).

### 1c. Variogram estimation and multi-scale fit (via gstools)

`src/ensemble/variogram.py`:
- `sample_pixel_pairs(residual_raster, n_pairs, stratify_by=None)`: random pair sampling (not exhaustive — infeasible at ~150M pixels), distance-binned; when `stratify_by` is a stratum raster, restrict pairs to same-stratum.
- `fit_nugget_multirange_model(binned_variogram)`: use `gstools`'s variogram-fitting utilities for a nugget + 2-range (short/long) model, per `(horizon, stratum, HM-level bin)`. Extract `σ(h)`, `range(h)` for `h ∈ {5,10,15}`.

**Stratify variograms by biome (14), not ecoregion (846).** These are different jobs with different optimal granularity: aggregation coverage wants *many* polygons for statistical power (hence ecoregion in T2.2), while variogram fitting wants *few, internally homogeneous, large* strata — each fit needs enough same-stratum pixel pairs across the full distance range out to the long-range scale, and an 846-way split starves the long lags and overfits noise. Biome is the right size; **realm (8) is the fallback** for biomes too thin or too spatially fragmented to support a stable fit (flag any biome whose fit fails to converge or whose range exceeds the stratum's own spatial extent, and roll it up to realm). Both come from the single `ECO_ID` raster via the lookup, so this is a `groupby` choice, not extra data.
- `extrapolate_h20_check(fitted_params_by_horizon, h20_residual_raster)`: fit simple `σ(h)`/`range(h)` growth curves from the 3 points, predict at `h=20`, compare against the single observed h=20 residual map's empirical variogram. This is a **single validation point**, not a distribution — report it as weak evidence, not proof, in any writeup (only one h=20 residual map exists across all folds, since it's only observed from the first input window).

### 1d. Class-conditional coverage audit (new — the evidence Phase 1.5 acts on)

`src/ensemble/validate.py::compute_class_conditional_coverage(residual_manifest, class_spec, alpha=0.05)`:
- Streams the Phase 0 stitched hindcast rasters + class covariate rasters, bins each valid pixel into its `(horizon, Δ̂-bin, HM-level-bin, biome)` cell, and accumulates `n`, `n_covered`, `n_below_lower`, `n_above_upper` per cell. Two counters, not one — **asymmetric miscoverage is the expected finding** (a right-skewed change distribution under-covers the upper tail far more than the lower), and a single coverage number would hide which side is broken.
- Reports **effective** sample size per cell as the count of distinct 128px chips contributing, not the raw pixel count. Residuals are spatially correlated at ranges far exceeding 1 km, so pixel counts overstate `n` by orders of magnitude and would make every cell look precisely estimated. Wilson intervals on the coverage estimate use `n_eff`.
- Emits `data/ensemble/calibration/coverage_audit.csv` and a heatmap figure (horizon × Δ̂-bin, cell colour = coverage, cell annotation = `n_eff`) — this is the second motivating figure, alongside 1b's coverage-vs-scale curve.

Bin edges, fixed from the measured change distribution (see grounded facts) so that no bin is degenerate:
- `Δ̂-bin`: `(−∞,−0.01] , (−0.01,0.001] , (0.001,0.01] , (0.01,0.05] , (0.05,0.15] , (0.15,∞)` → population shares roughly 3 / 65 / 11 / 14 / 4 / 1 % at h=20. The top two bins are small but are the entire point of the exercise.
- `HM-level-bin`: `[0,0.01) , [0.01,0.1) , [0.1,0.3) , [0.3,0.6) , [0.6,1]` → measured shares 53 / 24 / 12 / 10 / 0.9 %.
- `biome`: from 1a; `realm` as a coarser fallback where biome cells are too thin.

**Southern-Africa smoke test first**: run diagnostics against the Phase 0 small-region hindcast outputs before the full global residual rasters. Note the small region will not populate every class cell — the audit must degrade gracefully (report `n_eff` and suppress, not crash) on empty cells.

### Testing (Phase 1)

`tests/test_ensemble_variogram.py`, `tests/test_ensemble_validate.py`:
- Synthetic Gaussian random field with known nugget/range → assert `fit_nugget_multirange_model` recovers parameters within tolerance.
- Synthetic miscalibrated data with known scale-dependence → assert `compute_block_coverage` produces the expected collapse shape.
- Synthetic data with a *planted* class-conditional failure (95% coverage everywhere except one Δ̂-bin at 50%) → assert `compute_class_conditional_coverage` recovers ~0.95 pooled and ~0.50 in the planted cell. This test is the direct check that the audit can see what the pooled number hides.
- Empty/thin cells → assert `n_eff` reporting and suppression, no crash, no divide-by-zero.

---

## Phase 1.5 — Class-conditional recalibration of the quantile heads

**New**: `src/ensemble/calibrate.py`, `scripts/apply_recalibration.py`
**Owner of target metrics**: T1.1–T1.6, and T4.2's cross-horizon monotonicity constraint.

This phase did not exist in the original brief. It is added because the pooled coverage figure the project has been relying on cannot distinguish "calibrated" from "calibrated on average while badly under-covering every pixel anyone cares about" (see the arithmetic in Context). It runs **after** Phase 1's audit and **before** Phase 2, because the copula preserves whatever marginals it is given — fixing them afterwards is impossible without redoing Phase 3.

### 1.5a. Decision: keep / global rescale / stratified rescale

Per horizon `h` and cell `c`, over out-of-sample hindcast pixels, compute the **spread ratio** — observed residual spread over model half-width, per tail:

```
s_up(c) = quantile_{0.975}( R_native | c ) / median( w_up | c )
s_lo(c) = |quantile_{0.025}( R_native | c )| / median( w_lo | c )
```

Explicit decision rule, evaluated on the LOFO-CV estimates from 1.5d (not in-sample):

| Condition | Decision |
|---|---|
| all cells `s ∈ [0.9, 1.1]` **and** all cell coverages in `[0.93, 0.97]` | **keep** — no change, original constraint #2 stands as written, Phase 1.5 becomes a no-op identity transform |
| `s` roughly constant across cells (relative spread of `s` across cells < 0.1) but `≠ 1` | **global rescale** — one `(s_up, s_lo)` per horizon, 8 numbers total |
| `s` varies systematically with `Δ̂-bin` (the expected outcome) | **stratified rescale** — `(s_up, s_lo)` per cell |

Recording the decision and the table it came from matters as much as the rescale itself: "we checked and it was fine" is a publishable result, and if it is *not* fine, the table is the justification for having changed a published product.

### 1.5b. Mechanism: Mondrian (class-conditional) split conformal

Rather than fitting a spread ratio and hoping, take the conformal quantile directly — it gives a finite-sample coverage guarantee *within each class*, needs no distributional assumption, and is three lines of numpy. Nonconformity score is the **normalized** residual, so the model's own spatial pattern of interval width is retained and only its level is corrected:

```
E_i = (Y_i − central_i) / w_up_i      if Y_i > central_i
      (central_i − Y_i) / w_lo_i      otherwise
```

Per class `c` and tail, `ŝ(c) = ⌈(n_c + 1)(1 − α)⌉ / n_c` empirical quantile of `E` over the out-of-sample residuals in that class, with `α = 0.05` handled per-tail so the correction stays asymmetric. Then:

```
upper'(c) = central + ŝ_up(c) · w_up
lower'(c) = central − ŝ_lo(c) · w_lo
```

Four guards, all necessary:
1. **Effective sample size.** `n_c` in the conformal quantile is the count of independent 128px chips in that class, not pixels. Using pixel counts here would produce a spuriously precise quantile from what is effectively a handful of independent spatial blocks.
2. **Shrinkage.** `ŝ_final(c) = λ_c · ŝ(c) + (1 − λ_c) · ŝ_global(h)` with `λ_c = n_eff(c) / (n_eff(c) + n₀)`, `n₀ ≈ 200` chips. Without this, thin cells (HM > 0.6 is 0.9% of pixels) get a factor driven by noise.
3. **Smoothness across ordered bins.** `Δ̂-bin` and `HM-level-bin` are ordered, so fit `ŝ` monotonically (isotonic, or a low-order fit in `Δ̂`) across them rather than accepting a step function. A step function in `ŝ` produces visible discontinuities at bin boundaries in the output map — a real artifact, and T3.5 is a hard gate against it.
4. **Bounds and monotonicity.** Enforce `lower' ≤ central ≤ upper'`, clip to `[0,1]`, floor `ŝ` so no class can collapse to a zero-width interval, and enforce `ŝ` non-decreasing in horizon per class so T4.2 (spread grows with lead time) is not violated by independently-fitted horizons.

Note what this deliberately does **not** do: it does not touch `PinballLoss`, `_compute_horizon_losses`, the manual-optimization gradient isolation, or any architecture — consistent with hard constraint #1. A stratified pinball objective would be the "proper" fix and is explicitly out of scope for exactly that reason; post-hoc conformal rescaling is the constraint-compatible route, and for a frozen model it is close to as good.

### 1.5c. Applying it

`scripts/apply_recalibration.py` reads `data/ensemble/calibration/scale_factors.csv` (columns: `horizon, dhat_bin, hm_bin, biome, n_eff, s_up_raw, s_lo_raw, s_up_shrunk, s_lo_shrunk, lambda`) plus the class covariate rasters, and writes `data/predictions/recal/prediction_{year}_{lower,upper}_recal.tif`. Central rasters are **copied unmodified**, not regenerated — the guarantee that the central forecast is byte-identical should be structural, not something to verify later.

Applies to both the hindcast rasters (so Phase 4 can re-score them) and the production 2025–2040 rasters (so the ensemble is built on recalibrated marginals).

### 1.5d. Honest evaluation — leave-one-fold-out

Fitting `ŝ` and reporting coverage on the same residuals is circular and will always look good. Fit `ŝ` on 4 folds' residuals, evaluate class-conditional coverage on the held-out fold, rotate all 5, and report the **pooled LOFO-CV coverage table** as the headline. The in-sample table can be reported alongside; the gap between them is itself informative about how much the stratification is overfitting.

This costs nothing extra in compute — the fold structure from Phase 0 already exists and the conformal quantile is trivial to recompute 5 times.

### Testing (Phase 1.5)

`tests/test_ensemble_calibrate.py`:
- Synthetic data with a known planted per-class spread inflation (e.g. class A residuals 2× the model's stated width) → assert `ŝ_up(A) ≈ 2.0` and post-rescale class coverage ≈ 0.95.
- Conformal quantile: on exchangeable synthetic data with `n` points, assert empirical coverage ≥ `1 − α` (the finite-sample guarantee) across many seeds.
- Shrinkage: `λ → 1` as `n_eff → ∞`, `λ → 0` as `n_eff → 0`; a thin cell with a wild raw `ŝ` ends up near `ŝ_global`.
- Guards: monotone-in-horizon enforcement; `lower' ≤ central ≤ upper'` and `[0,1]` clipping hold on adversarial synthetic input.
- **Identity case**: on perfectly-calibrated synthetic input, assert the whole pipeline returns rasters equal to the originals within floating-point tolerance — i.e. "keep" is genuinely a no-op.

---

## Phase 2 — Residual field generator

**New**: `src/ensemble/fields.py`

Single-tier generation (per user decision — no 10km/1km split for this pass):
```
generate_correlated_field(H, W, ranges_px, weights, nugget, rng, wrap_lon=True) -> np.ndarray
```
FFT convolution of white noise with a multi-scale Gaussian-kernel mixture (`scipy.fft.rfft2`/`irfft2`, already available). `ranges_px`/`weights` sourced from Phase 1's `fit_nugget_multirange_model` output, converted km→px using the confirmed ~0.009°/px grid resolution. Longitude wraps periodically for free (FFT circular convolution, no zero-padding on that axis, matching the global lon `-180..180` grid); latitude is *not* periodic (poles) — pad/taper that axis to avoid wraparound artifacts near `+84/-70`.

```
apply_ar1_horizon_coupling(z_by_horizon, rho_by_horizon, rng) -> dict
```
`z_h = ρ_h·z_{h-5} + sqrt(1-ρ_h²)·ε_h`, applied sequentially `5→10→15→20`; `ρ_h` from horizon-autocorrelation of hindcast residuals (small helper computed from Phase 0's residual manifest, pooled across fold/window pairs where both horizons were observed).

**Prototype in southern Africa first**: generate a handful of fields (~5 members) at the small-region extent before scaling to the full global grid, given the global FFT at `17111×40000` (~684M px, ~2.7GB/field float32) is nontrivial even single-tier.

### Testing (Phase 2)

`tests/test_ensemble_fields.py`:
- Periodicity: circular continuity across the lon-wrap seam on a small synthetic grid.
- Unit variance of generated fields (pre-biome-scaling).
- AR(1) coupling: empirical `corr(z_h, z_{h-5})` over many realizations ≈ `ρ_h` within sampling tolerance.

---

## Phase 3 — Copula coupling to existing quantiles

**New**: `src/ensemble/copula.py`, `scripts/generate_ensemble.py`

```
fit_marginal_two_piece_normal(lower, central, upper) -> {loc, scale_left, scale_right}
```
Closed-form (no per-pixel optimizer — 684M pixels rules out `curve_fit`-per-pixel): `scale_left = (central-lower)/z_{0.025}`, `scale_right = (upper-central)/z_{0.975}`. Two-piece normal chosen over skew-normal specifically to keep this closed-form.

```
marginal_ppf(u, params) -> value          # vectorized inverse CDF
copula_sample_member(z_field, marginal_params) -> member_raster
  # u = Phi(z_field); member = marginal_ppf(u, marginal_params)
  # property: z=0 -> u=0.5 -> member == central forecast exactly (constraint: median preserved)
```

`scripts/generate_ensemble.py`: for each horizon, fit marginals from the **recalibrated** production rasters (`data/predictions/recal/prediction_*_{lower,upper}_recal.tif` from Phase 1.5, plus the unmodified `prediction_*_central_blended.tif`) — not the raw heads, and not the Phase 0 hindcast rasters. If Phase 1.5's decision was "keep", the recal rasters are byte-identical copies and this path is unchanged. Then generate 50 AR(1)-coupled correlated fields via Phase 2, sample 50 members, write to zarr.

**Storage** (single-tier, 50 members, per user decision): `data/ensemble/members.zarr`, shape `(50, 4, H, W)`, dtype `int16`, chunked `(10, 1, 1024, 1024)`. Quantization: `scale = 1.0/32767`, `offset = 0`, stored as zarr array attrs (not hardcoded) — note `1.0`, not the `0.94` an earlier undersized sample suggested; HM is definitionally bounded at 1 and the observed maximum reaches it. Sentinel `-32768` for invalid/masked pixels (int16 has no native NaN). `manifest.json` records each member's `(z_field_seed, ar1_chain)` for deterministic single-member regeneration.

**Member count vs tail resolution (affects T5.2).** With `M = 50`, the empirical 2.5th percentile falls between the 1st and 2nd order statistics, so the ensemble's estimate of its own 2.5/97.5 quantiles carries real Monte-Carlo noise — T5.2 must therefore be scored with a tolerance scaled by that MC error, not as an exact match, and the check should be framed as "consistent with" rather than "equal to". 50 members is fine for the development loop and for T2's aggregate statistics (aggregation averages the noise down), but tail-sensitive per-pixel products may want `M ≥ 200` before publication. Flag this at the Phase 3 → Phase 4 boundary rather than discovering it during validation; the `manifest.json` seed record makes extending an existing 50-member ensemble to 200 cheap and reproducible.

**Southern-Africa smoke test first**: 20-member, 1-horizon ensemble over the small region, verify zarr read/write correctness, before the full 50-member × 4-horizon global run.

### Testing (Phase 3)

`tests/test_ensemble_copula.py`:
- `marginal_ppf(0.5, params) == central` for synthetic pixel quantile triples.
- `marginal_ppf(0.025,...) ≈ lower`, `marginal_ppf(0.975,...) ≈ upper`.
- int16 roundtrip precision bound (`< scale/2`).

---

## Phase 4 — Aggregation and validation

**New**: `src/ensemble/aggregate.py`, `scripts/validate_ensemble.py`

```
aggregate_region_statistic(zarr_store, region_geom_or_mask, horizon, statistic_fn, threshold=None) -> np.ndarray[n_members]
summarize_ensemble(stat_values) -> {median, p2_5, p97_5}
```
Region selection reuses the same bbox+geometry-mask logic as `_predict_region_and_write` (`train_lightning.py` ~lines 1282–1355) for consistency.

`scripts/validate_ensemble.py` — one script that scores **every** target metric and emits a single pass/fail scorecard (`data/ensemble/validation/scorecard.csv` + a summary figure), so "is it done?" is answerable without re-deriving thresholds each time:

1. **T2.1–T2.4** — re-run `compute_block_coverage` against ensemble percentiles instead of raw pixelwise quantile heads, at every block size and polygon set, plus area-above-threshold and between-horizon change statistics. Report the pixelwise-independent baseline in the same table as the before/after. Headline result.
2. **T1.2–T1.4** — re-run `compute_class_conditional_coverage` (Phase 1d) against the *recalibrated* intervals, and against the ensemble's own per-pixel 2.5/97.5. These must agree with each other (they are the same marginals seen two ways); disagreement means the copula is not preserving marginals and is a bug, not a calibration issue.
3. **T2.5 + marginal check** — rank histograms at both pixel and aggregate scale, with the χ² GoF test and reliability index.
4. **T3.1–T3.4** — semivariogram of ensemble members vs the fitted residual variogram; variogram score (Scheuerer & Hamill) and energy score, each scored against the **independent-pixel null ensemble** built from the same marginals. Build that null explicitly in this script — it is a 5-line function and the only way T3.2/T3.3 mean anything.
5. **T4.1–T4.2** — between-horizon field correlation vs hindcast residual autocorrelation; per-pixel spread monotonicity in horizon.
6. **T5.1–T5.3 + T3.5** — hard gates: ensemble median vs `prediction_*_central_blended.tif` (already on disk), ensemble tails vs the recal rasters within MC tolerance, valid-mask identity, and the seam/tile/bin-boundary artifact renders.

Failures should print the diagnosis from the **"which knob fixes which failure"** table in the target-metrics section rather than just a red X — the whole point of separating the knobs is that a given failure has one correct response.

### Testing (Phase 4)

`tests/test_ensemble_aggregate.py`: synthetic ensemble array with known properties → assert percentile math and rank-histogram uniformity test on synthetic well-calibrated data. Also assert the scorecard logic itself: a synthetic ensemble constructed to fail exactly one target must produce exactly one failed row (guards against a scorecard that silently passes everything). Smoke test against Phase 2/3's southern-Africa ensemble first.

---

## Housekeeping

`docs/simple_model_architecture_and_training.md`, lines ~104–105: fix "4 consecutive timesteps" / `[1990,1995,2000,2005]→[2010,2015,2020,2025]` to match the actual 3-input-timestep behavior (matches README and `torchgeo_dataloader.py`'s `fixed_input_years=(1990,1995,2000)`): "3 consecutive timesteps to predict 4 future horizons (+5/+10/+15/+20yr)", example `[1990,1995,2000] → predict [2005,2010,2015,2020]`. Trivial, independent of the ensemble work — do any time before the Zenodo deposit.

---

## Deferred (explicitly out of scope for this build)

- EOF/PCA residual modes (too few residual maps with k=5 to estimate real components).
- Deep ensemble (retrain K≈20 seeds with spatially blocked bootstrap).
- Horizon-conditioned prediction head (FiLM/embedding).
- Two-tier (10km long-range + 1km on-demand nugget) field generation/storage — revisit once single-tier 50-member correctness is validated end-to-end.
- **Stratified / change-weighted pinball loss** (training the quantile heads to be class-conditionally calibrated by construction, rather than correcting them post-hoc). This is the principled fix for what Phase 1.5 patches, and it is deferred purely because hard constraint #1 forbids touching the loss code. Worth revisiting if the Phase 1d audit shows the miscalibration is severe enough that a multiplicative correction can't reach T1.2 without blowing the T1.5 sharpness guard — that outcome would be the evidence needed to justify reopening the training pipeline.
- **Distributional regression / conditional-variance head** as an alternative to rescaling — same reasoning, same constraint.

## Open risks to keep visible during execution

1. **k=5 full retrainings** is still the largest compute cost in the plan — size actual GPU-hours from a rehearsal run before committing to all 5.
2. **h=20 has one real out-of-sample data point** across all folds (only observed from the `[1990,95,00]` window) — the extrapolation check in Phase 1 is weak evidence, not proof; say so in any writeup.
3. **RESOLVE Ecoregions2017 acquisition is on the critical path**, not optional — T2.2–T2.6, Phase 1c's variogram strata, and Phase 1.5's secondary stratum all depend on it, and the ecological reporting framing is a settled decision. Start it immediately and run it concurrently with Phase 0's retrainings (CPU/IO work alongside GPU work). Failure modes to watch: coastline mismatch between the RESOLVE polygons and the HM grid (quantify the valid-HM-on-`ECO_ID==0` pixel count, don't discover it as missing area later), and any ecoregion whose rasterized area diverges from the shapefile geometry area by more than a few percent.
4. **`gstools` is a new dependency** in a repo with no pinned requirements file — install into the `spatio-temporal-dl` conda env and note it somewhere (even informally) so it's not lost.
5. **Recalibration is fit on 2000–2020 hindcast residuals and applied to 2025–2040 forecasts.** This assumes the error structure is stationary in time. It cannot be validated directly — there is no future data — and the assumption gets weaker the further out the forecast runs. State it explicitly wherever recalibrated intervals are published; do not silently present a 2040 interval as having the same empirical backing as a 2020 one.
6. **Thin classes are the whole point and also the weakest estimates.** The Δ̂ > 0.15 bin is ~1% of pixels and the HM > 0.6 bin is 0.9%; both will have modest `n_eff` after chip-level deduplication. The shrinkage in 1.5b keeps them from going wild, but the honest reading is that these cells have wide error bars on their coverage estimate — report Wilson intervals on the coverage table, not bare point estimates, or the audit will look more decisive than it is.
7. **Recalibration changes a published product.** If the decision is "stratified rescale", the intervals in any existing figure, deposit, or draft become stale. Version the outputs (`recal/` directory, factors CSV committed) so the before/after is reconstructible, and budget for regenerating downstream figures.
8. **The tightest coupling in the plan is Phase 1.5 → Phase 3.** The copula preserves whatever marginals it is handed, so a marginal change after the ensemble is generated means regenerating the ensemble. Do not start the global 50-member run until the recalibration decision is final.

## Development scale (user decision, superseding the global-first reading)

**All development and iteration happens on the southern-Africa subregion**
(`config/region_to_predict_small.geojson`). Global runs happen only on explicit
instruction. The global k=5 hindcast has already been produced and its residual and
prediction rasters are on disk, so the regional working set is a *crop* of those artifacts
(`scripts/make_region_subset.py`) rather than a retraining — the fold models are global and
genuinely out-of-sample everywhere, so cropping loses nothing but wall-clock time.

A full regional iteration of Phase 1 → Phase 4 takes minutes rather than the ~10 hours the
global equivalent takes, which is the difference between testing a change and guessing at it.

## Verification plan (end-to-end)

At every phase, validate against the southern-Africa region (`config/region_to_predict_small.geojson`) before scaling to the full global grid (`config/region_to_predict_large.geojson`) — this is cheap, fast, and exercises the full pipeline shape. Concretely: Phase 0's k=2/1-window/1-epoch smoke test → Phase 1 diagnostics (including the 1d class-conditional audit) on that smoke test's residuals → Phase 1.5's recalibration decision and LOFO-CV coverage table → Phase 2's 5-member regional fields → Phase 3's 20-member regional ensemble on recalibrated marginals → Phase 4's scorecard against that regional ensemble, checking the T5 hard gates first (cheapest, most direct signal something is wrong). Only after all of that passes regionally, proceed to the full k=5 × 4-window global hindcast and the 50-member × 4-horizon global ensemble. Run `pytest tests/test_ensemble_*.py` at each phase boundary as new test files land.

**Run Phase 1a concurrently with Phase 0.** The RESOLVE acquisition and rasterization is CPU/IO work with no dependency on the fold retrainings, and Phase 0's five retrainings are the long pole occupying the GPUs. Everything from Phase 1b onward is blocked on 1a, so starting it late serialises two independent bottlenecks for no reason.

Two ordering constraints are not negotiable, both for the same reason (the copula preserves marginals exactly, so anything upstream of it must be settled before it runs):
1. **Phase 1d's audit must complete before Phase 1.5 decides**, and Phase 1.5's decision must be final before the global Phase 3 run starts.
2. **Marginals are tuned before correlation structure.** T1 first, then T2. Never fix an aggregate-scale miss by re-widening marginals.

The regional smoke test cannot populate every class cell, so the *decision* in 1.5a must be taken on the global hindcast residuals even though the *code path* is validated regionally. Expect the regional run to exercise correctness and the global run to produce the numbers.
