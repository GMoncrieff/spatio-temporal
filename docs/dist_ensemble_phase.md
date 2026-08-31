# The ensemble phase — spatial dependence for the distributional model

Branch `dist-convlstm`, started 2026-08-29. Screened on Africa (63.1 Mpx, 21.85% of the grid
finite under the holdout fold checkerboard). Model frozen at `e1` seed 46 throughout; **nothing
about the model changes in this phase.** What is under test is how the ensemble's spatial
dependence is fitted and generated.

The phase was opened by an external review of the dependence model, which accepts the two-stage
architecture — neural marginals plus a Gaussian copula — as well precedented, and argues that the
risk is **over-specifying the dependence structure**. It names five specific things this pipeline
does. This document records what each of them measured.

**Status: CLOSED. Seven ensembles built and scored. Two instrument bugs fixed (+4 rows on the
baseline with no model change). Of the review's ten actionable points: three adopted, two
measured and rejected as wrong for this field, three measured and found inside the estimator's
own noise floor, two left untested. The promoted configuration is `V4` — PIT-space dependence,
no hand-added broad component, and a Student-t copula at ν = 7 — which more than doubles the
spatial-structure score and takes ecoregion coverage from "1.000 by being 1.7× too wide" to
0.918–0.966 at near-nominal dispersion. The model itself did not change.**

---

## 1. Two instrument bugs, and a baseline that was wrong before anything ran

### 1.1 The loop never scored the marginal it generated

`run_dist_ensemble_loop.sh` placed `--qf_dir` in the `COMMON` array, which only
`generate_ensemble.py` consumes. The Phase 4 `validate_ensemble.py` invocation builds its own
argument list and omitted it. So **running the committed loop reproduces `scorecard_twopiece/`
(77/46), not `scorecard_qf/` (83/40)** — the frozen quantile-function card had been produced by a
separate hand-run. A flag present in a file is not a flag reaching a stage.

The replacement driver is `scripts/run_dist_ensemble_variant.sh`, which passes it and adds
`SHARED_ROOT`, `SKIP_DIAGNOSTICS`, `SPECTRA_FLAGS` and `GEN_FLAGS` so a variant is a flag rather
than an edit. `run_dist_ensemble_variant2.sh` adds `SPECTRA_JSON`.

### 1.2 T4 was scoring a marginal the members were never drawn from

`--qf_dir` had been added to `validate_ensemble.py` and reached T3 and T5 — but `stage_temporal`
called `recover_z` unconditionally (`:1801-1802`) and fed `population_spread` the two-piece
`(cen, sl, sr)` (`:1855`). Recovering a normal score through the wrong marginal *attenuates* a
correlation, and the frozen card's four T4 rows all failed in exactly that direction.

Fixed by giving `stage_temporal` the same branch T3 already had, backed by one shared kernel
(`src/ensemble/residuals.qf_normal_score`, which `validate_ensemble.qf_recover_z` now delegates
to) and a closed-form `qf_population_spread` for T4.2 — exact for the piecewise-linear,
end-clamped law the sampler actually draws from, and centred on the stored median because HM sits
on `[0,1]` and an uncentred second moment cancels to a spurious 1e-8 spread.

| row | frozen card | corrected | target |
|---|---|---|---|
| T4.1 corr 2005→2010 | 0.512 | **0.676** | within ±0.10 of 0.692 |
| T4.1 corr 2010→2015 | 0.580 | **0.734** | within ±0.10 of 0.740 |
| T4.1 corr 2015→2020 | 0.710 | **0.826** | within ±0.10 of 0.843 |
| T4.2 spread monotone | 0.9877 | **0.9985** | ≥ 0.99 |

**0 of 4 became 4 of 4, and the baseline went 83/40/14 to 87/36/14 with no model change at all.**
The AR(1) horizon coupling had never been broken. The corrected rows live in
`data/ensemble/exp/dist_baseline_af_e1/scorecard_qf_t4fix/`; the frozen directory is untouched,
and `scripts/compare_scorecards.py --patch` is how they enter a comparison.

### 1.3 Two further defects, reported and not fixed

`T2.7` never emits a row: `validate_ensemble.py:658-660` filters on an `interval_score` column
that `src/ensemble/validate.py:297-309` never writes, so **T2.8 is the only width guard actually
running**. And `validate_ensemble.py`'s `main` ends `return 0 if len(failed) == 0 else 0`, so a
failing scorecard never fails the shell pipeline. Neither changes any comparison in this phase.

---

## 2. Measuring the review's criticisms before spending GPU time

Each rule below was written before its measurement was taken. Scripts:
`diag_mask_spectrum_bias.py`, `diag_latent_normalisation.py`, `diag_pit_vs_width_residual.py`,
and `fit_field_spectra.py --report_windows`.

**The realisation-noise floor is 0.044.** Refitting the same synthetic field three times moves the
4–50 px variance share by that much. That is the number every other measurement is read against,
and it is why four of the five criticisms do not lead anywhere.

| # | review § | question | measured | verdict |
|---|---|---|---|---|
| D1 | 4.3 | does the fitter recover a known spectrum through the real 21.85% / 512 px mask? | coastline −0.029, checkerboard +0.051, **net +0.022** | inside the noise floor |
| D2 | 4.1, 4.2 | is the mixture identifiable across forecast origins? | 4–50 px share sd **0.047** (h=5, n=4), **0.055** (h=10, n=3) | identifiable |
| D3 | 4.5 | are members and their null on the same latent scale? | 0.9308 vs 0.9471, **0.016** apart | fine |
| D4 | 4.8 | does T4.1 still fail once measured through Q? | **4 of 4 pass** | never broken |
| D5 | 4.8 | does the AR(1) mixture distort each horizon's spectrum? | max shift **0.021–0.033** | second-order |
| D6 | 3.1, 5.1 | is the spectrum fitted in the space the copula samples in? | 4–50 px share **+0.145 (h=5) / +0.099 (h=20)** | **real** |

Two further measurements decided the rest.

**The k=0 variance the appended component stands in for.** `--long_weight 0.40` at 1500 px is
documented as supplying the frequency-zero variance a mean-subtracted FFT cannot see, and was
hand-set on **southern Africa** against the *post-hoc product's* residuals. That quantity is
directly measurable as the across-forecast-origin variance of the region-wide mean standardised
residual: it is **0.002–0.009**, 45–200× smaller. Worse, what the residual carries at k=0 is a
**bias**, not a spread — the domain mean runs −0.11σ at h=5 to **−0.62σ at h=20** — and a variance
component cannot represent a bias.

**The review's §5.2 single-range recommendation, grid-searched.** The best single range is 25 px
(h=20) / 20 px (h=5), fitting **22–37× worse** than the ten-range basis and putting **58–65% of
the variance in the nugget**. Adopting it would make members near-pure speckle. Measurably the
wrong model for this field, and D2 had already shown the mixture is supported by the data.

### T3.1's variance row is a single-member draw

T3.1 "member normal-score variance" reads 1.117 against a `1.0 ± 0.15` gate. An independent
implementation reproduces it — member 0 is **1.1159** — but across 120 members the value is
**0.940 ± 0.075**, min 0.799, max 1.241, and **9.2% of members fall outside the gate**. The
tolerance is almost exactly two sigma of the quantity being measured, so the row passes or fails
on which draw happens to be member 0. The population value is 0.940. Reported, not acted on.

---


---

## 3. What each variant changed, and why

Every variant below alters **only how pixels agree with each other**. The model's per-pixel
quantile function `Q_h(u|x)` is byte-identical in all seven runs, and every member is still made
the same way: draw a correlated field, turn each pixel's value into a probability rank `u`, and
read that pixel's own `Q` at `u`. **What any single pixel is allowed to do never changes.** What
changes is which pixels are extreme *together*.

That is worth holding onto, because it is what makes the scorecard readable: rows that describe
one pixel at a time (T1, T5, T6, most of T8) should stay put, and rows that describe many pixels
at once (T2, T3, T7) are where a dependence change can show.

### The baseline, in plain terms

Two ingredients decide how neighbouring pixels co-vary.

**First, a measurement.** To learn "how much do neighbours agree, and over what distance", the
pipeline looks at where the observation fell relative to the forecast, and fits a spatial
spectrum to that field. The baseline builds that field as `(observed − central) / σ`, where `σ`
comes from the interval half-width. That is a shortcut: it reads **3 of the 64 stored quantile
levels** and assumes the rest is symmetric. HM is not symmetric — it is floored at zero, 40% of
Africa sits in `[0, 0.01)`, and 5.4% of member-pixels land exactly on the floor.

**Second, a hand-set constant.** On top of the fitted spectrum, **40% of the field's variance is
assigned by hand to a single very smooth component 1500 px (~1500 km) wide.** Its stated purpose
is to supply the "whole-region is a bit busier or quieter" uncertainty that a mean-subtracted FFT
cannot see. Its weight was tuned on **southern Africa**, against the *previous* post-hoc
product's residuals, to make one metric come out near 1.

So a baseline member differs from its neighbours in two ways: fine-grained texture from the
fitted spectrum, and a continent-wide mood swing worth 40% of the variance.

### V1 — delete the hand-set continental component

`--long_weight 0`. Nothing else. **Zero new code.**

*Why:* the quantity that component claims to represent is directly measurable — the variance of
the region-wide mean residual across forecast origins. It is **0.002–0.009**, not 0.40: between
45× and 200× too large. And what the residual actually carries at that scale is a **bias**, not a
spread (its domain mean runs −0.11σ at h=5 to **−0.62σ at h=20**), which a variance component
cannot represent at all.

*What happened:* the members stopped disagreeing at region scale almost entirely. In the
high-change crop at h=20 the member envelope collapsed from a 0.095-wide span to **0.016, never
crossing zero** — every member telling nearly the same story. Ecoregion coverage fell 0.99 → 0.74.
**76/123.**

*What it established:* the component is load-bearing, but **not for the reason its docstring
gives**. It is not a frequency-zero reservoir; it is the only thing supplying structure between
50 px and the domain, because the fitted mixture puts *exactly zero* weight on the 120 px and
300 px basis elements at every horizon. Ecoregions live exactly in that gap.

### V2 — measure the dependence in the space the copula actually samples in

`--fit_space pit`. Fit the spectrum to `Φ⁻¹(F_qf(y))` — the observation's true probability rank
under the forecast — instead of `(observed − central) / σ`.

*Why:* the copula's latent variable *is* a probability rank. Fitting its structure in a
different, approximate space measures the wrong thing. This is the review's §5.1, and it uses all
64 stored levels and handles the floor correctly.

*What happened:* the change is real and it is local. At h=20 the generator's loaded structure
went from `nugget 0.041, 1.5 px 0.181, 50 px 0.131` to `nugget 0.031, 1.5 px 0.132, 50 px 0.265`
— **double the 50 px weight and a quarter less pixel-scale speckle.** Ecoregion coverage was
untouched to four decimals, because the hand-set 40% still dominated that scale. **86/123.**

*What it established:* the two axes are close to independent — the fit space buys local texture
and touches the aggregate scale not at all.

### V3 — both of the above

`--fit_space pit --long_weight 0`. The candidate configuration of round 1.

*What happened:* best-on-values of round 1 — spatial-structure score doubled (T3.2s 0.0156 →
0.0312), spread-skill 1.713 → 1.051, rank histogram from p ≈ 0 everywhere to three of four
passing. But ecoregion coverage sat at **0.837** where 0.95 is wanted. **82/123.**

*The finding that drove round 2:* V3 has the **right dispersion and too little coverage** —
spread-skill 1.051 but coverage 0.837. `t2_zonal_coverage.csv` showed the misses are **symmetric**
(below/above 2.67 / 0.62 / 0.60 / 1.67 across years), so this is not the model's real region-wide
bias showing through, which would be one-sided. What is left is *shape*: at matched dispersion
the ensemble's ecoregion-mean distribution is **lighter-tailed than the observed errors**. Real
regional errors have fat tails — most regions fine, occasionally one badly wrong — while the
average of a Gaussian copula tends back toward a bell curve.

### V4 — give each member one shared "boldness" draw *(promoted)*

`--copula t --copula_df 7`, on top of V3.

*The intuition:* in every Gaussian-copula run, the 400 members are drawn independently, so the
regional averages across members cluster like a bell curve. A **Student-t copula** hands each
member a single extra number drawn at random — a boldness dial — and applies it to that member
everywhere. A timid member hugs the central forecast across the whole map; a bold member pushes
*every* pixel toward its own extreme. Members now go extreme **together**, which is exactly the
compound-regional behaviour that was missing.

Formally `u = T_ν(z / √W)` with one `W ~ χ²_ν/ν` per member, shared across all pixels and all
four horizons — a member is one story, so it gets one dial. Because `z/√W` is marginally `t_ν`,
`T_ν` of it is **exactly uniform**: every pixel's marginal is untouched by construction, and only
the joint behaviour changes. (`torch` has no Student-t CDF, so it is a 240k-point table plus
linear interpolation, agreeing with scipy to **1.15e-6** on the worker's own path.)

*ν was measured, not chosen.* The observation's standardised position within the member
ecoregion-mean distribution has kurtosis **4.2 / 5.2 / 4.9 at h=10/15/20** against a Gaussian's
3.0; inverting `kurtosis = 3 + 6/(ν−4)` gives ν = 8.9 / 6.7 / 7.1. Hence **ν = 7**.

The independent-pixel null stays Gaussian: a shared χ² factor *is* dependence, so applying it to
the null would give the null domain-scale structure and quietly flatter the correlated ensemble
in T3.2/T3.3.

*What happened:* every ecoregion coverage row moved toward nominal **from below**, without
re-inflating width — 2020 went 0.837 → 0.918, 2015 0.891 → 0.959 — and the spatial-structure
score rose again. **92/123 as run, 90 honest** (below).

### V4b — the same configuration, with the boldness cards dealt properly

`--copula_w_draw stratified`. **Not a fourth variant: a sampler correction to V4.**

*The bug:* the tail factor is **one scalar per member**, so 400 members sample the χ² law with
only 400 draws — and because those same 400 values are reused at all 13.8M pixels, the error does
not average away. The realised mixing mean came out **0.966** against a target of 1.0. That
biased the ensemble's own marginal (realised CDF **+0.0033** above target at u = 0.944), narrowed
every published interval ~6%, and flipped two T2.8 rows from fail to pass.

*How it was caught:* T2.8 at 1 km is a block of **one pixel**, so it reads the marginal width and
nothing else. **Every configuration in the phase reads exactly 0.02142 there — except V4, at
0.02003.** The one run whose marginal was measurably off was the only one that moved a pure
marginal row.

*The fix:* draw `w_m = χ²_ν.ppf((m + 0.5)/M)/ν` in random order, so the realised law matches the
target exactly at any M while staying independent of the field. Realised mean **0.9997**.

*What happened:* marginal check **0.00328 → 0.00004** (an 82× tightening, back to Gaussian
levels); the two T2.8 rows reverted to failing exactly as predicted; ecoregion coverage held or
rose (0.932 → 0.939, 0.952 → 0.966, T2.4 0.925 → 0.939); and T3.2s reached **0.0413, the best of
the phase**. **90/123 — V4's honest score.**

### V5 — replace the AR(1) horizon chain with an exact separable covariance

`--horizon_corr` plus a pooled spatial spectrum. Review §5.4.

*The intuition:* the four horizons of one member are currently chained — h=10 is built from h=5
plus fresh noise, h=15 from h=10, and so on. That keeps them coherent but means horizon *h*
inherits a *mixture* of the earlier horizon's spatial structure and its own, rather than the
structure that was fitted for it. The separable alternative draws four independent fields from
**one shared** spatial spectrum and combines them with `chol(R)`, where `R` is the full 4×4
correlation matrix. Both properties then hold exactly: each horizon carries the shared spectrum,
and the horizons come back with correlation `R`.

*Verified, not assumed:* `scripts/check_separable_horizon.py` measured the delivered matrix from
the stored members — max |delivered − R| = **0.0425** against a 0.05 tolerance, with the residual
~3.6% shortfall fully explained by the int16/quantile-function round trip, which loses 5.3% of
latent variance and attenuates a correlation by that same reliability factor (predicting 0.798
for the h=15→20 pair against a measured 0.800).

*What happened:* the best practical-range match of the phase (**0.030**), and a *worse*
spatial-structure score (T3.2s 0.0312 → 0.0241). **83/123.**

*What it established:* the AR(1) chain was not costing anything worth recovering. Its spectral
distortion measures **0.0438**, right at the estimator's own 0.044 realisation-noise floor — and
the exact construction that removes it *requires* one shared spectrum, whose cost is **0.110** of
the 4–50 px variance share. The fix is more expensive than the defect.

### V6 — three ranges instead of ten

`--ranges 1.5,6,50`, refitted. Review §5.3's "J = 2 or 3, not ten by default."

*What happened:* a decisive negative. **T3.2 went to −0.0023 — the correlated ensemble scored
worse than an independent-pixel null** — and T3.2s collapsed 0.0312 → 0.0046, a 7× loss.
**81/123.**

*What it established:* the ten-range basis is not over-specified. The 4 px and 25 px components
carry mean weights of only 0.111 and 0.071 and are load-bearing anyway. **And a screening trap
worth keeping:** the J=3 fit was judged equivalent on four-band variance shares — 0.617 against
0.618 at h=5 — and those agreed almost exactly while the field's variogram behaviour diverged
completely. A sparse mixture is a lumpy comb, not a smooth spectrum, and a four-band summary
cannot see comb teeth.

---

## 4. Results

| | `long_weight` | fit space | copula | horizons | ranges | **pass /123** | T3.2s | T3.2b | T3.1 range | T7.3 | ecoregion 2020 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.40 | width | Gaussian | AR(1) | 10 | 87 | 0.0156 | 0.0448 | 0.145 | 1.713 ✗ | 1.000 ✗ |
| V1 | **0** | width | Gaussian | AR(1) | 10 | 76 | 0.0225 | 0.0561 | 0.359 ✗ | 0.879 | 0.742 |
| V2 | 0.40 | **PIT** | Gaussian | AR(1) | 10 | 86 | 0.0185 | 0.0495 | 0.392 ✗ | 1.796 ✗ | 1.000 ✗ |
| V3 | **0** | **PIT** | Gaussian | AR(1) | 10 | 82 | 0.0312 | 0.0601 | 0.114 | 1.051 | 0.837 |
| V4 | **0** | **PIT** | **t(7)** i.i.d. | AR(1) | 10 | 92\* | 0.0360 | 0.0685 | 0.097 | 1.181 | 0.918 |
| **V4b — promoted** | **0** | **PIT** | **t(7)** stratified | AR(1) | 10 | **90** | **0.0413** | 0.0607 | 0.128 | 1.265 ✗ | **0.918** |
| V5 | **0** | **PIT** | Gaussian | **separable** | 10 | 83 | 0.0241 | 0.0568 | **0.030** | 1.051 | 0.844 |
| V6 | **0** | **PIT** | Gaussian | AR(1) | **3** | 81 | 0.0046 | 0.0526 | 0.282 ✗ | 1.008 | 0.871 |

\* two of V4's rows are the finite-M χ² artifact; its honest score is 90.

**What the promoted configuration buys, against the T4-corrected baseline:**

- **spatial structure more than doubles** — T3.2s 0.0156 → **0.0413**, 2.6×, at matched sampling;
- **ecoregion coverage becomes meaningful** — 1.000 at a spread-skill of 1.71 (covering by being
  1.7× too wide) becomes **0.918–0.966 at 1.27**, i.e. covering because it is closer to right;
- **the rank histogram unflattens** — p ≈ 0 at all four horizons becomes 0.056 / 0.002 / 0.217 /
  0.171, three of four passing;
- **T4 goes 0/4 to 4/4**, from the instrument fix alone;
- **the model did not change.** Same checkpoints, same quantile function, same hindcast.

**What it costs:** seven `T2.3` ecoregion-*area* rows sit below target in every configuration
without the hand-set component (V4b 0.79–0.91 against 0.95 ± 0.05), and `T7.3` lands at 1.265,
**0.015 outside** its gate.

**Reading the two rows that look like regressions.** `T3.1 member normal-score variance` reads
0.607 in V4b and 1.242 in V4 — failing on opposite sides — because that row scores **member 0
alone**, and under a t-copula member 0 carries its own boldness draw. It measures `w_0`, not the
ensemble. (Independently: across 120 members the statistic is 0.940 ± 0.075, and 9.2% of members
fall outside its ±0.15 gate, so the row is a coin flip even under a Gaussian copula.) And T8's
outer bands use **members 0:8 only**, which is why V4's T8.3 reads 83.5 against V4b's 10.6.

**The review, scored on measurement.** Adopted: §5.1 (fit the dependence in PIT space), §4.1/§5.3
(do not append a fixed broad term), §4.7 (Gaussian copulas under-produce compound regional
extremes). Rejected on evidence: §5.2's single range (22–37× worse fit, 58–65% nugget), §5.3's
small mixture (J=3 loses to white noise). Measured and found inside the estimator's own 0.044
noise floor, so not worth acting on: §4.3 masked FFT (0.022), §4.5 per-realisation
standardisation (0.016), §4.8 AR(1) spectral mixing (0.021–0.044). Left untested: §4.4 (PIT
clipping — the stored grid truncates at u = 1e-4, so testing it needs new predictions, not a new
ensemble) and §4.6 (physical-distance ranges — real, but Africa's E-W pixel only shrinks to
0.79× at 38°N and it touches no failing row).

**Open, and the order to take them in.** `ν = 8` or `9` — both inside the measured 6.7–8.9 range —
should pull T7.3 back inside its gate at little coverage cost. Then **replicate**: every number
here is one run per configuration, and this project's rule 29 is that three replicates can land
in one mode. Then price the `T2.3` trade.

---

## 5. The promoted ensemble — V4, full scorecard

**Configuration.** Model `e1` seed 46, unchanged, checkpoints `final_fold{1,2}_387829{7,8}.ckpt`.
Hindcast `data/ensemble/exp/af_e1_hind/stitched/` (Africa, folds 1+2, holdout stitch, w2000
base). Marginal = the model's own 64-band quantile function, untouched. Dependence = a spectrum
fitted in **PIT space** with **no appended broad component**, sampled through a **Student-t
copula at ν = 7** with a **stratified** χ² draw. M = 400 plus an independent-pixel (Gaussian)
null.

```bash
env SHARED_ROOT=data/ensemble/exp/af_e1_hind SKIP_DIAGNOSTICS=1 \
    SPECTRA_JSON=<pit_lw0.json> \
    GEN_FLAGS="--copula t --copula_df 7 --copula_w_draw stratified" \
    ./scripts/run_dist_ensemble_variant2.sh data/ensemble/exp/dist_v4b_tcopula_strat 400
```

**Which store to ship.** `V4` and `V4b` are the *same configuration*; they differ only in how the
per-member χ² factor is drawn. **Ship `dist_v4b_tcopula_strat`** — `V4`'s stored ensemble has a
measurably biased marginal (realised CDF +0.0033 off at u = 0.944, intervals ~6% narrow) and two
of its scorecard rows are that artifact. Both columns are shown below so the difference is
auditable.

### Artifacts

| what | where |
|---|---|
| **members, M=400** | `/mnt/hdd1/spatio-temporal/data/ensemble/exp/dist_v4b_tcopula_strat/members_m400.icechunk` (31 GB) |
| independent-pixel null | `…/dist_v4b_tcopula_strat/null_m400.icechunk` (32 GB) |
| per-member seeds and χ² factors | `…/dist_v4b_tcopula_strat/members_m400_manifest.json` |
| scorecard | `data/ensemble/exp/dist_v4b_tcopula_strat/validation/scorecard.csv` |
| **3×5 forecast panels** | **`data/ensemble/exp/dist_v4b_tcopula_strat/floor_panels/`** |
| spectrum | `data/ensemble/exp/dist_v4b_tcopula_strat/spectral_fits.json` (`fit_space: pit`, `long_weight: 0.0`) |
| marginal-recovery proof | `…/dist_v4b_tcopula_strat/qf_marginal_check.txt` (worst deviation 0.00004) |
| the superseded i.i.d. run | `…/dist_v4_tcopula/` — kept for audit, **not for publication** |

### The 3×5 panels

`floor_panels/forecast_panel_{high_change,quiet}_{2005,2010,2015,2020}.png` — eight figures,
built with the baseline's exact invocation so the crops are identical and comparable across every
run in the phase (the window picker is deterministic on the *observed* field):

```
row 1   observed dHM | central | lower 2.5% | upper 97.5%      (the published product)
row 2   3 random members | highest-change member | lowest-change member
row 3   each member's change distribution, observed overlaid
```

```bash
python scripts/plot_forecast_panel.py \
  --ensemble /mnt/hdd1/spatio-temporal/data/ensemble/exp/dist_v4b_tcopula_strat/members_m400.icechunk \
  --recal_dir data/ensemble/exp/dist_v4b_tcopula_strat/stitched --suffix "" \
  --base_year 2000 --years 2005,2010,2015,2020 --min_land 0.35 \
  --out_dir data/ensemble/exp/dist_v4b_tcopula_strat/floor_panels --disable_wandb
```

`--min_land 0.35` because a holdout-stitched hindcast is a fold checkerboard — only 21.85% of the
Africa grid is finite, and the 0.6 default rejects every candidate window while reporting success.
The checkerboard visible in the maps is the held-out fold geometry, not a defect.

**What the h=20 high-change panel shows**, against the same crop in the frozen baseline: the
member envelope spans **+0.0714 … +0.0125** where the baseline's spans +0.0875 … −0.0076 and V3's
only +0.0353 … +0.0133. The promoted ensemble sits between "every member tells the same story"
(V3, 0.022 wide) and "the extreme member is nothing like anything observed" (baseline, 0.095
wide) — 0.059, arrived at from a measured ν rather than a tuned knob.

### Full scorecard

Every row, including the ones that did not move. Read values, not the pass count.

#### T1 — per-pixel marginal coverage

A property of the model's quantile function. The copula reorders which pixels are extreme together; it cannot change what any one pixel may do. These rows should, and do, move only by Monte-Carlo noise.

| row | metric | target | baseline | V4 | **V4b** | |
|---|---|---|---:|---:|---:|:--:|
| `T1.1` | pooled coverage (h=5) | 0.95 +/- 0.01 | 0.9392 | 0.9347 | **0.9397** | **✗** |
| `T1.1` | pooled coverage (h=10) | 0.95 +/- 0.01 | 0.9522 | 0.9483 | **0.9525** | ✓ |
| `T1.1` | pooled coverage (h=15) | 0.95 +/- 0.01 | 0.9536 | 0.9511 | **0.9542** | ✓ |
| `T1.1` | pooled coverage (h=20) | 0.95 +/- 0.01 | 0.9473 | 0.9443 | **0.9478** | ✓ |
| `T1.2` | class coverage h=5 (-0.01,0.001] | \|cov-0.95\| <= 0.03 | 0.9411 | 0.9366 | **0.9415** | ✓ |
| `T1.2` | class coverage h=5 (0.001,0.01] | \|cov-0.95\| <= 0.03 | 0.9172 | 0.9117 | **0.9178** | **✗** |
| `T1.2` | class coverage h=5 (0.01,0.05] | \|cov-0.95\| <= 0.03 | 0.9417 | 0.9444 | **0.9487** | ✓ |
| `T1.2` | class coverage h=10 (-0.01,0.001] | \|cov-0.95\| <= 0.03 | 0.9618 | 0.9584 | **0.9619** | ✓ |
| `T1.2` | class coverage h=10 (0.001,0.01] | \|cov-0.95\| <= 0.03 | 0.9101 | 0.9056 | **0.9125** | **✗** |
| `T1.2` | class coverage h=10 (0.01,0.05] | \|cov-0.95\| <= 0.03 | 0.9536 | 0.9442 | **0.9495** | ✓ |
| `T1.2` | class coverage h=15 (-0.01,0.001] | \|cov-0.95\| <= 0.03 | 0.9714 | 0.9698 | **0.9716** | ✓ |
| `T1.2` | class coverage h=15 (0.001,0.01] | \|cov-0.95\| <= 0.03 | 0.9083 | 0.9044 | **0.9103** | **✗** |
| `T1.2` | class coverage h=15 (0.01,0.05] | \|cov-0.95\| <= 0.03 | 0.9339 | 0.9293 | **0.9350** | ✓ |
| `T1.2` | class coverage h=15 (0.05,0.15] | \|cov-0.95\| <= 0.03 | 0.9555 | 0.9425 | **0.9517** | ✓ |
| `T1.3` | high-change tail h=15 (0.05,0.15] | >= 0.92 | 0.9555 | 0.9425 | **0.9517** | ✓ |
| `T1.2` | class coverage h=20 (-0.01,0.001] | \|cov-0.95\| <= 0.03 | 0.9682 | 0.9658 | **0.9680** | ✓ |
| `T1.2` | class coverage h=20 (0.001,0.01] | \|cov-0.95\| <= 0.03 | 0.9087 | 0.9044 | **0.9100** | **✗** |
| `T1.2` | class coverage h=20 (0.01,0.05] | \|cov-0.95\| <= 0.03 | 0.9096 | 0.9067 | **0.9129** | **✗** |
| `T1.2` | class coverage h=20 (0.05,0.15] | \|cov-0.95\| <= 0.03 | 0.9531 | 0.9410 | **0.9488** | ✓ |
| `T1.3` | high-change tail h=20 (0.05,0.15] | >= 0.92 | 0.9531 | 0.9410 | **0.9488** | ✓ |
| `T1.2` | worst primary cell deviation | <= 0.05 | 0.0417 | 0.0456 | **0.0400** | ✓ |

#### T2 — aggregate-scale coverage and width

Where the whole phase was decided. Coverage is the fraction of ecoregions (n=147) or blocks whose observed value falls inside the members' 2.5-97.5 range.

| row | metric | target | baseline | V4 | **V4b** | |
|---|---|---|---:|---:|---:|:--:|
| `T2.1` | block coverage 1km (2005) | 0.95 +/- 0.05 | 0.9392 | 0.9347 | **0.9397** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 1km (2005) | <= 0.0204 | 0.0214 | 0.0200 | **0.0214** | **✗** |
| `T2.1` | block coverage 10km (2005) | 0.95 +/- 0.05 | 0.9482 | 0.9401 | **0.9454** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 10km (2005) | <= 0.0204 | 0.0162 | 0.0147 | **0.0157** | ✓ |
| `T2.1` | block coverage 100km (2005) | 0.95 +/- 0.05 | 0.9732 | 0.9441 | **0.9472** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 100km (2005) | <= 0.0199 | 0.0119 | 0.0086 | **0.0092** | ✓ |
| `T2.1` | block coverage 1km (2010) | 0.95 +/- 0.05 | 0.9522 | 0.9483 | **0.9525** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 1km (2010) | <= 0.0379 | 0.0394 | 0.0374 | **0.0397** | **✗** |
| `T2.1` | block coverage 10km (2010) | 0.95 +/- 0.05 | 0.9606 | 0.9527 | **0.9576** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 10km (2010) | <= 0.0380 | 0.0298 | 0.0273 | **0.0290** | ✓ |
| `T2.1` | block coverage 100km (2010) | 0.95 +/- 0.05 | 0.9816 | 0.9602 | **0.9655** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 100km (2010) | <= 0.0371 | 0.0220 | 0.0160 | **0.0171** | ✓ |
| `T2.1` | block coverage 1km (2015) | 0.95 +/- 0.05 | 0.9536 | 0.9511 | **0.9542** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 1km (2015) | <= 0.0497 | 0.0539 | 0.0514 | **0.0542** | **✗** |
| `T2.1` | block coverage 10km (2015) | 0.95 +/- 0.05 | 0.9653 | 0.9586 | **0.9619** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 10km (2015) | <= 0.0498 | 0.0434 | 0.0394 | **0.0417** | ✓ |
| `T2.1` | block coverage 100km (2015) | 0.95 +/- 0.05 | 0.9824 | 0.9625 | **0.9678** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 100km (2015) | <= 0.0487 | 0.0320 | 0.0232 | **0.0246** | ✓ |
| `T2.1` | block coverage 1km (2020) | 0.95 +/- 0.05 | 0.9473 | 0.9443 | **0.9478** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 1km (2020) | <= 0.0603 | 0.0661 | 0.0632 | **0.0664** | **✗** |
| `T2.1` | block coverage 10km (2020) | 0.95 +/- 0.05 | 0.9658 | 0.9574 | **0.9616** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 10km (2020) | <= 0.0604 | 0.0570 | 0.0516 | **0.0547** | ✓ |
| `T2.1` | block coverage 100km (2020) | 0.95 +/- 0.05 | 0.9855 | 0.9587 | **0.9609** | ✓ |
| `T2.8` | aggregate width vs mean-of-bounds 100km (2020) | <= 0.0591 | 0.0426 | 0.0318 | **0.0338** | ✓ |
| `T2.2` | ecoregion mean (2005) | 0.95 +/- 0.05 | 0.9932 | 0.9320 | **0.9388** | ✓ |
| `T2.3` | ecoregion area>0.1 (2005) | 0.95 +/- 0.05 | 0.9932 | 0.9048 | **0.9320** | ✓ |
| `T2.3` | ecoregion area>0.3 (2005) | 0.95 +/- 0.05 | 0.9864 | 0.8707 | **0.8844** | **✗** |
| `T2.6` | biome (reported) mean (2005) | reported with CI (not scored) | 1.0000 | 0.8182 | **0.8182** | — |
| `T2.6` | realm (reported) mean (2005) | reported with CI (not scored) | 1.0000 | 1.0000 | **1.0000** | — |
| `T2.2` | ecoregion mean (2010) | 0.95 +/- 0.05 | 0.9932 | 0.9524 | **0.9660** | ✓ |
| `T2.3` | ecoregion area>0.1 (2010) | 0.95 +/- 0.05 | 0.9864 | 0.9116 | **0.9184** | ✓ |
| `T2.3` | ecoregion area>0.3 (2010) | 0.95 +/- 0.05 | 0.9864 | 0.9116 | **0.9388** | ✓ |
| `T2.6` | biome (reported) mean (2010) | reported with CI (not scored) | 1.0000 | 1.0000 | **1.0000** | — |
| `T2.6` | realm (reported) mean (2010) | reported with CI (not scored) | 1.0000 | 1.0000 | **1.0000** | — |
| `T2.2` | ecoregion mean (2015) | 0.95 +/- 0.05 | 1.0000 | 0.9592 | **0.9592** | ✓ |
| `T2.3` | ecoregion area>0.1 (2015) | 0.95 +/- 0.05 | 0.9864 | 0.9184 | **0.9320** | ✓ |
| `T2.3` | ecoregion area>0.3 (2015) | 0.95 +/- 0.05 | 0.9728 | 0.9048 | **0.9048** | ✓ |
| `T2.6` | biome (reported) mean (2015) | reported with CI (not scored) | 1.0000 | 1.0000 | **1.0000** | — |
| `T2.6` | realm (reported) mean (2015) | reported with CI (not scored) | 1.0000 | 1.0000 | **1.0000** | — |
| `T2.2` | ecoregion mean (2020) | 0.95 +/- 0.05 | 1.0000 | 0.9184 | **0.9184** | ✓ |
| `T2.3` | ecoregion area>0.1 (2020) | 0.95 +/- 0.05 | 1.0000 | 0.8980 | **0.8980** | **✗** |
| `T2.3` | ecoregion area>0.3 (2020) | 0.95 +/- 0.05 | 0.9796 | 0.8980 | **0.9252** | ✓ |
| `T2.6` | biome (reported) mean (2020) | reported with CI (not scored) | 1.0000 | 1.0000 | **1.0000** | — |
| `T2.6` | realm (reported) mean (2020) | reported with CI (not scored) | 1.0000 | 1.0000 | **1.0000** | — |
| `T2.4` | ecoregion mean change 2005->2020 | 0.95 +/- 0.05 | 1.0000 | 0.9252 | **0.9388** | ✓ |
| `T2.5` | rank histogram flatness (2005) | chi2 p > 0.01 | 0.0000 | 0.1916 | **0.0564** | ✓ |
| `T2.5` | rank histogram flatness (2010) | chi2 p > 0.01 | 0.0000 | 0.0016 | **0.0016** | **✗** |
| `T2.5` | rank histogram flatness (2015) | chi2 p > 0.01 | 0.0000 | 0.1699 | **0.2168** | ✓ |
| `T2.5` | rank histogram flatness (2020) | chi2 p > 0.01 | 0.0000 | 0.2058 | **0.1705** | ✓ |

#### T3 — spatial realism of the field

Does the generated field carry the structure the residual actually has. **Read T3.2s and T3.2b, not T3.2**: the gated row divides by a structure budget evaluated at a lag set by the member field's own practical range, so it scores each run against its own difficulty and is not comparable across configurations.

| row | metric | target | baseline | V4 | **V4b** | |
|---|---|---|---:|---:|---:|:--:|
| `T3.1` | member normal-score variance | 1.0 +/- 0.15 | 1.1167 | 1.2424 | **0.6073** | **✗** |
| `T3.1` | member practical range vs fitted | <= 0.25 relative | 0.1446 | 0.0967 | **0.1276** | ✓ |
| `T3.1` | member nugget/sill vs fitted | <= 0.10 absolute | 0.0783 | 0.0683 | **0.0657** | ✓ |
| `T3.2` | variogram-score improvement as a share of the structure budget | >= 0.50 of budget | 0.0606 | 0.0597 | **0.0666** | **✗** |
| `T3.2r` | variogram score vs independent-pixel null (spread-weighted) | reported, not gated | 0.0184 | 0.0211 | **0.0240** | — |
| `T3.2b` | variogram score vs null (uniform sampling, reference) | reported, not gated | 0.0448 | 0.0685 | **0.0607** | — |
| `T3.2s` | variogram score vs null at short lag | reported, not gated | 0.0156 | 0.0360 | **0.0413** | — |
| `T3.3` | energy score beats null and degenerate | < min(null 0.7319, degenerate 1.004) | 0.7171 | 0.7002 | **0.7033** | ✓ |
| `T3.4` | radial power spectrum written | reference comparison | 40.0000 | 40.0000 | **40.0000** | — |
| `T3.5` | lon seam continuity | no discontinuity (hard gate) | regional grid; lon seam  | regional grid; lon seam  | **regional grid; lon seam ** | ✓ |

#### T4 — coherence across horizons

Fixed this phase: `stage_temporal` was inverting a two-piece normal the members were never drawn from, which attenuates a correlation. All four rows pass once measured through Q.

| row | metric | target | baseline | V4 | **V4b** | |
|---|---|---|---:|---:|---:|:--:|
| `T4.1` | between-horizon corr 2005->2010 | within +/-0.10 of 0.692 | 0.6756 | 0.6727 | **0.6733** | ✓ |
| `T4.1` | between-horizon corr 2010->2015 | within +/-0.10 of 0.740 | 0.7343 | 0.7303 | **0.7302** | ✓ |
| `T4.1` | between-horizon corr 2015->2020 | within +/-0.10 of 0.843 | 0.8265 | 0.8229 | **0.8232** | ✓ |
| `T4.2` | population spread non-decreasing in horizon | >= 0.99 | 0.9985 | 0.9985 | **0.9985** | ✓ |
| `T4.2s` | sample spread non-decreasing in horizon | reported, not gated | 0.9520 | 0.9609 | **0.9669** | — |

#### T5 — hard gates: does the ensemble reproduce its own published quantiles

A failure here is a generation bug, not a calibration issue. T5.1 compares the 400-member sample median to Q(0.5); the misses are Monte-Carlo and quantisation, not bias.

| row | metric | target | baseline | V4 | **V4b** | |
|---|---|---|---:|---:|---:|:--:|
| `T5.1` | median==Q(0.5) (2005) | >=0.995 (MC-scaled, hard gate) | 0.9928 | 0.9945 | **0.9938** | **✗** |
| `T5.2` | tails within MC tolerance (2005) | >=0.95 (MC-scaled) | 0.9657 | 0.9666 | **0.9724** | ✓ |
| `T5.3` | valid-mask identity (2005) | 0 pixels | 0.0000 | 0.0000 | **0.0000** | ✓ |
| `T5.1` | median==Q(0.5) (2010) | >=0.995 (MC-scaled, hard gate) | 0.9918 | 0.9898 | **0.9886** | **✗** |
| `T5.2` | tails within MC tolerance (2010) | >=0.95 (MC-scaled) | 0.9850 | 0.9844 | **0.9871** | ✓ |
| `T5.3` | valid-mask identity (2010) | 0 pixels | 0.0000 | 0.0000 | **0.0000** | ✓ |
| `T5.1` | median==Q(0.5) (2015) | >=0.995 (MC-scaled, hard gate) | 0.9942 | 0.9937 | **0.9930** | **✗** |
| `T5.2` | tails within MC tolerance (2015) | >=0.95 (MC-scaled) | 0.9904 | 0.9915 | **0.9912** | ✓ |
| `T5.3` | valid-mask identity (2015) | 0 pixels | 0.0000 | 0.0000 | **0.0000** | ✓ |
| `T5.1` | median==Q(0.5) (2020) | >=0.995 (MC-scaled, hard gate) | 0.9912 | 0.9923 | **0.9912** | **✗** |
| `T5.2` | tails within MC tolerance (2020) | >=0.95 (MC-scaled) | 0.9924 | 0.9912 | **0.9927** | ✓ |
| `T5.3` | valid-mask identity (2020) | 0 pixels | 0.0000 | 0.0000 | **0.0000** | ✓ |

#### T6 — the change distribution

Marginal properties of dHM, largely a model matter.

| row | metric | target | baseline | V4 | **V4b** | |
|---|---|---|---:|---:|---:|:--:|
| `T6.1` | P(change < -0.01) vs observed (2005) | ratio in [0.5, 2.0] | 1.1166 | 0.9847 | **0.8963** | ✓ |
| `T6.2` | P(change < -0.05) vs observed (2005) | ratio <= 3 | 0.4496 | 0.2807 | **0.2032** | ✓ |
| `T6.3` | P(change < -0.15) vs observed (2005) | ratio <= 5 | 0.0000 | 0.0000 | **0.0000** | ✓ |
| `T6.5` | change q01 vs observed (2005) | within a factor of 2 | 0.5581 | 0.8198 | **0.7744** | ✓ |
| `T6.5` | change q05 vs observed (2005) | within a factor of 2 | 0.5270 | 1.0400 | **1.0419** | ✓ |
| `T6.1` | P(change < -0.01) vs observed (2010) | ratio in [0.5, 2.0] | 1.5011 | 1.3714 | **1.2870** | ✓ |
| `T6.2` | P(change < -0.05) vs observed (2010) | ratio <= 3 | 0.9721 | 0.6720 | **0.5141** | ✓ |
| `T6.3` | P(change < -0.15) vs observed (2010) | ratio <= 5 | 0.1684 | 0.1305 | **0.0875** | ✓ |
| `T6.4` | tail asymmetry vs observed (2010) | ratio in [0.5, 2.0] | 14.6171 | 14.4556 | **17.7921** | **✗** |
| `T6.5` | change q01 vs observed (2010) | within a factor of 2 | 0.7154 | 0.9306 | **0.8782** | ✓ |
| `T6.5` | change q05 vs observed (2010) | within a factor of 2 | 0.9539 | 1.5477 | **1.5555** | ✓ |
| `T6.1` | P(change < -0.01) vs observed (2015) | ratio in [0.5, 2.0] | 1.4793 | 1.5351 | **1.4681** | ✓ |
| `T6.2` | P(change < -0.05) vs observed (2015) | ratio <= 3 | 0.4909 | 0.4961 | **0.4019** | ✓ |
| `T6.3` | P(change < -0.15) vs observed (2015) | ratio <= 5 | 0.1163 | 0.1212 | **0.0984** | ✓ |
| `T6.4` | tail asymmetry vs observed (2015) | ratio in [0.5, 2.0] | 15.4480 | 12.4448 | **13.8688** | **✗** |
| `T6.5` | change q01 vs observed (2015) | within a factor of 2 | 0.5564 | 0.8620 | **0.8096** | ✓ |
| `T6.5` | change q05 vs observed (2015) | within a factor of 2 | 1.3049 | 2.2684 | **2.2774** | **✗** |
| `T6.1` | P(change < -0.01) vs observed (2020) | ratio in [0.5, 2.0] | 1.5717 | 1.5418 | **1.4885** | ✓ |
| `T6.2` | P(change < -0.05) vs observed (2020) | ratio <= 3 | 0.5871 | 0.4803 | **0.3897** | ✓ |
| `T6.3` | P(change < -0.15) vs observed (2020) | ratio <= 5 | 0.0217 | 0.0244 | **0.0200** | ✓ |
| `T6.4` | tail asymmetry vs observed (2020) | ratio in [0.5, 2.0] | 88.4464 | 62.1814 | **70.2567** | **✗** |
| `T6.5` | change q01 vs observed (2020) | within a factor of 2 | 0.8391 | 1.0595 | **1.0048** | ✓ |
| `T6.5` | change q05 vs observed (2020) | within a factor of 2 | 1.6406 | 2.5989 | **2.6150** | **✗** |

#### T7 — member diversity and spread-skill

T7.3 = (member spread) / RMSE over ecoregion means. 1.0 is calibrated; the baseline's 1.71 is over-dispersion, which is how it reached 1.000 coverage.

| row | metric | target | baseline | V4 | **V4b** | |
|---|---|---|---:|---:|---:|:--:|
| `T7.2` | mean pairwise member correlation | < 0.98 | 0.2041 | 0.1977 | **0.2423** | ✓ |
| `T7.3` | spread-skill ratio (ecoregion) | 1.0 +/- 0.25 | 1.7127 | 1.1810 | **1.2650** | **✗** |
| `T7.1` | member vs observed change renders | visual inspection | 12.0000 | 12.0000 | **12.0000** | — |

#### T8 — the far field

A property of the model's quantile heads: `stage_clustering` reads only the distance-band raster, HM_2000, the observation and raw member values. It also uses **members 0:8 only**, so the outer bands carry real Monte-Carlo scatter that a correlation change modulates — V4's T8.3 reads 83.5 against V4b's 10.6 for exactly that reason. Suspect the measurement before the model.

| row | metric | target | baseline | V4 | **V4b** | |
|---|---|---|---:|---:|---:|:--:|
| `T8.1` | P(Δ>0.05) band 0-1 | ratio in [0.5, 2.0] | 1.0627 | 0.9990 | **1.0179** | ✓ |
| `T8.4` | P(Δ<-0.05) band 0-1 | ratio in [0.5, 3.0] | 0.7755 | 0.7014 | **0.5902** | ✓ |
| `T8.1` | P(Δ>0.05) band 1-3 | ratio in [0.5, 2.0] | 1.1783 | 1.1372 | **1.1645** | ✓ |
| `T8.4` | P(Δ<-0.05) band 1-3 | ratio in [0.5, 3.0] | 0.4108 | 0.3279 | **0.2542** | **✗** |
| `T8.1` | P(Δ>0.05) band 3-10 | ratio in [0.5, 2.0] | 0.8181 | 0.7575 | **0.7452** | ✓ |
| `T8.4` | P(Δ<-0.05) band 3-10 | ratio in [0.5, 3.0] | 0.1492 | 0.0952 | **0.0724** | **✗** |
| `T8.1` | P(Δ>0.05) band 10-30 | ratio in [0.5, 2.0] | 1.3966 | 1.1915 | **1.0781** | ✓ |
| `T8.4` | P(Δ<-0.05) band 10-30 | ratio in [0.5, 3.0] | 0.0668 | 0.0343 | **0.0259** | **✗** |
| `T8.1` | P(Δ>0.05) band 30-100 | ratio in [0.5, 2.0] | 2.4792 | 1.5456 | **1.2241** | ✓ |
| `T8.4` | P(Δ<-0.05) band 30-100 | ratio in [0.5, 3.0] | 0.0017 | 0.0000 | **0.0000** | **✗** |
| `T8.1` | P(Δ>0.05) band >100 | ratio in [0.5, 2.0] | 0.1017 | 0.0120 | **0.0957** | **✗** |
| `T8.4` | P(Δ<-0.05) band >100 | ratio in [0.5, 3.0] | 0.0000 | 0.0000 | **0.0000** | **✗** |
| `T8.2` | remote-band change realism (both directions) | mean \|log10 ratio\| <= 0.301 (within  | 2.7874 | 3.2521 | **2.8006** | **✗** |
| `T8.3` | near/remote contrast vs observed | ratio in [0.5, 2.0] | 10.4537 | 83.5356 | **10.6389** | **✗** |

#### Totals

| configuration | pass | fail | unscored | of |
|---|---:|---:|---:|---:|
| baseline (T4-corrected) | 87 | 36 | 14 | 137 |
| V4 (i.i.d. χ² draw) | 92 | 31 | 14 | 137 |
| **V4b (stratified χ²) — promoted** | 90 | 33 | 14 | 137 |

---

## 6. Reproducing the phase

```bash
COMMON="SHARED_ROOT=data/ensemble/exp/af_e1_hind SKIP_DIAGNOSTICS=1"
QF="--fit_space pit --qf_dir data/ensemble/exp/af_e1_hind/stitched"
SC=<scratch>

# the PIT spectrum is fitted ONCE (~80 min) and reused; long_weight is post-hoc arithmetic
python scripts/fit_field_spectra.py --manifest data/ensemble/exp/af_e1_hind/residuals/manifest.csv \
  --out $SC/pit_lw0.json $QF --long_weight 0 --report_windows
python scripts/derive_long_component.py --src $SC/pit_lw0.json --out $SC/pit_lw040.json --long_weight 0.40
python scripts/pool_horizon_spectra.py  --src $SC/pit_lw0.json --out $SC/pit_lw0_pooled.json

env $COMMON SPECTRA_FLAGS="--long_weight 0" \
  ./scripts/run_dist_ensemble_variant.sh  data/ensemble/exp/dist_v1_long000 400
env $COMMON SPECTRA_JSON=$SC/pit_lw040.json \
  ./scripts/run_dist_ensemble_variant2.sh data/ensemble/exp/dist_v2_pitz 400
env $COMMON SPECTRA_JSON=$SC/pit_lw0.json \
  ./scripts/run_dist_ensemble_variant2.sh data/ensemble/exp/dist_v3_long000_pitz 400
env $COMMON SPECTRA_JSON=$SC/pit_lw0.json GEN_FLAGS="--copula t --copula_df 7 --copula_w_draw stratified" \
  ./scripts/run_dist_ensemble_variant2.sh data/ensemble/exp/dist_v4b_tcopula_strat 400   # promoted
env $COMMON SPECTRA_JSON=$SC/pit_lw0_pooled.json GEN_FLAGS="--horizon_corr $SC/horizon_corr.json" \
  ./scripts/run_dist_ensemble_variant2.sh data/ensemble/exp/dist_v5_separable 400
env $COMMON SPECTRA_JSON=$SC/pit_lw0_j3.json \
  ./scripts/run_dist_ensemble_variant2.sh data/ensemble/exp/dist_v6_j3 400
```

Compare row by row, never on a pass count (rule 4). The `--patch` is not optional: the frozen
card's T4 rows were scored through the wrong marginal.

```bash
python scripts/compare_scorecards.py \
  --baseline base=data/ensemble/exp/dist_baseline_af_e1/scorecard_qf/scorecard.csv \
  --patch    data/ensemble/exp/dist_baseline_af_e1/scorecard_qf_t4fix/scorecard.csv \
  --variant  v4b=data/ensemble/exp/dist_v4b_tcopula_strat/validation/scorecard.csv
```

### What this phase added to the toolchain

| script | what it is for |
|---|---|
| `run_dist_ensemble_variant.sh` / `variant2.sh` | the loop with `--qf_dir` reaching Phase 4, shared residuals, and spectrum reuse |
| `compare_scorecards.py` | row-by-row diff with `--patch` for partial re-scores |
| `derive_long_component.py` | re-derives a spectrum at a different `long_weight` without refitting (exact: reproduces the frozen fit to 0.000e+00) |
| `pool_horizon_spectra.py` | one shared spatial spectrum, printing the per-horizon spread it averages away |
| `check_separable_horizon.py` | asserts a separable run reproduces its own `R` |
| `diag_mask_spectrum_bias.py` | D1 — spectral-fit bias through the real mask |
| `diag_latent_normalisation.py` | D3 — members vs null on the same latent scale |
| `diag_pit_vs_width_residual.py` | D6 — the two fit spaces, same estimator, same pixels |
| `fit_field_spectra.py --report_windows` | the per-origin spread `fits.setdefault` throws away |

**Measured cost.** One variant loop from the shared hindcast is **~2 h 15 min** — generation
9 min for M=400 (33 GB), the null the same, T1–T8 **133 min**. A PIT-space spectral fit is ~80 min
on its own, which is why the PIT variants share one. Residuals and diagnostics are reused via
`SKIP_DIAGNOSTICS=1`, saving 25 min and 2.8 GB per variant. ~78 GB on the HDD per variant, of
which 7.3 GB is an orphaned smoke-marginal cache safe to delete after the run.
