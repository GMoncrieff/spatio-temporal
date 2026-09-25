# The reference card — `c1_foldb4`, Stage F

Written 2026-08-21, branch `ensemble`. The first card of the honest-fold-mask lineage, and
the one every later run should be compared against. Supersedes `africa_k5`'s 111/133 and the
Stage A re-score's 102/133, **neither of which is comparable to it** — the fold mask, the
gates, the calibration and the marginal have all changed.

**Headline: 103 of 136 scored rows pass, 14 reported-only, at M = 400 on Africa.**

Read the total last. Five gates were rewritten in Stage A and a sixth (T4.2) in Stage B, so a
row that flips is usually the ruler moving, not the model.

---

## 1. What produced it

| | |
|---|---|
| model | five folds on `fold_mask_b4_1000.tif` (512 px blocks), `VAL_STRIDE=1024` |
| flags | `--central_residual True --central_context True --monotone_quantile_width True --quantile_context True` |
| stitch | `holdout` |
| calibration | `stratified` conformal + `near` width factors (bands 0-2), both re-derived here |
| marginal | empirical shape per (horizon × band), pooled body, **`u_bound 0.999` on every band** |
| AR(1) | measured: **ρ = {10: 0.650, 15: 0.461, 20: 0.433}** |
| members | 400 + a 400-member independent-pixel null, 9.7 + 9.9 min, 80 + 82 GB |
| scorecard | 237 min, peak RSS inside a 20 GB budget |
| gate | the M=4 smoke tier ran first and passed (172 rows, all stages, all block scales) |

Artifacts: `data/ensemble/exp/c1_foldb4/validation_m400/`.

## 2. By family, against the old lineage

| family | africa_k5 (1×1) | c1_foldb4 (4×4) |
|---|---|---|
| T1 pixel coverage | 29/33 | 27/33 |
| T2 aggregate | 37/41 | 33/41 |
| **T3 spatial** | 3/6 | **5/6** |
| **T4 temporal** | 0/1 | **3/4** |
| T5 hard gates | 11/12 | 10/12 |
| T6 change realism | 19/24 | 14/24 |
| T7 diversity | 1/2 | 1/2 |
| T8 change placement | 11/14 | 10/14 |
| **total** | 111/133 | **103/136** |

T4 goes 0/1 → 3/4 because ρ is measured here; every regional card before this coupled its
horizons at an unmeasured 0.9 and scored T4.1 against a NaN. T3 goes 3/6 → 5/6.

## 3. The far field is fixed on the increase side

This is the round's target and the result is unambiguous. `P(Δ > 0.05)` as a ratio to
observed, by distance band:

| band | africa_k5 | **c1_foldb4** |
|---|---|---|
| 0-1 px | 1.33 | 1.38 |
| 1-3 px | 1.21 | 1.37 |
| 3-10 px | 1.06 | 1.27 |
| 10-30 px | 1.27 | 1.34 |
| 30-100 px | 1.36 | **2.03** ✗ |
| **> 100 px** | **0.107** ✗ | **0.845** ✓ |

**The remote band goes from 9× too quiet to within 16% of observed**, and T8.3's near/remote
contrast lands at 1.63 against a [0.5, 2.0] target where the old one-sided gate read 226,373.
T8.2's two-sided error falls 1.530 → 0.675 — still failing its ≤ 0.301 bar, now entirely on
the decrease side.

The cost is the 30-100 px band tipping to 2.03, just past its 2.0 bound: `u_bound 0.999`
applied uniformly is slightly too hot there. A band-varying bound would recover it, and 5-diag
already shows the mid field responds to the bound, so this is a cheap follow-up rather than a
defect.

**The lower tail is not fixed and mostly cannot be**: `P(Δ<−0.05)` in the remote band is 18.9×
observed. Per Stage B 4-diag that ratio rests on **88 observed events globally and one on
Africa**, on the 0.14% of remote pixels that can physically decrease, so it should not be
fitted against.

## 4. What still fails, and which of it is the model

**Real model defects.**

- **T2.5, 0/4.** Ecoregion-scale over-dispersion, p = 1e-8 to 1e-14 on pooled bins. Confirmed
  independently by T7.3 (spread-skill 1.64, target 1.0 ± 0.25). This is the clearest
  unaddressed defect on the card.
- **T3.2 = 0.125 of the structure budget** (up from 0.063). The ensemble captures an eighth of
  the long-range structure the residual has at the scored separation.
- **T6.1/T6.4/T6.5**, 10 rows. The lower tail is too heavy pooled — the same defect as §3's
  decrease side.

**Calibration defects, not model ones — and the cheapest thing on this list to fix.**

- **T4.2 = 0.802.** Decomposed stage by stage, the raw quantile heads are **0.980** monotone
  (`--monotone_quantile_width` does its job); the post-hoc layers take it to **0.813**, the
  marginal shape to 0.808 and the [0,1] clip to 0.802. Two mechanisms: the conformal layer is
  isotonically smoothed across horizons and the **width layer is not**, and 2.6% of pixels
  change Δ̂ class between horizons, picking up a different factor. **Fix: a cumulative maximum
  on the final half-widths after both layers** — 1.000 by construction, robust to
  class-switching, costing +1.2% / +1.5% / +2.5% of mean width at h=10/15/20 and nothing at
  h=5. That cost is plausibly a gain, since T1.1 under-covers. Check it does not reverse the
  `stratified` decision, whose margin is 0.7%.
- This is also the concrete cost of **item 1 never being done**: with one calibration layer the
  constraint could live inside the fit, as it already does for the conformal half. Split across
  two layers keyed on different axes, the only robust fix is applied to their combined output.

**Consequences of choices made here.**

- **T1.1 under-covers** (0.932-0.941 against 0.95 ± 0.01) where `africa_k5` over-covered. The
  `near` width factors narrow bands 0-2, which Stage D chose on held-out interval score. The
  narrowing is slightly too strong pooled; it was selected on the score, not on coverage.
- **T2.2/T2.4 read exactly 1.000** at three horizons — over-coverage at ecoregion scale, the
  mirror of T2.5's dome.

**One open instrument question.**

- **T5.1 fails at h=5 and h=10** (0.990 and 0.925 against ≥ 0.995), a hard gate. Its own note
  says the distributional median is exact by construction and only the *sample* median of
  M=400 is not — so, like T4.2 before Stage B, it scores a population property through a
  finite sample. Its tolerance is `3·(1.2533/√M)·σ·S′(0) + quantization`, and raising
  `u_bound` to 0.999 stretches the body over a wider z-range, which *shrinks* `S′(0)` and
  tightens the tolerance. So the failure follows mechanically from the marginal change acting
  on the gate's own scaling. **It is not established that generation is wrong, and it is not
  established that the gate is.** Do not tune the field to pass it before that is settled; the
  test is whether the sample-median deviation exceeds what the marginal's own density implies,
  which needs the median SE computed from the shape rather than from a local-normal
  approximation.

## 5. What this card is not

It is **one run**. `CLAUDE.md` records ±7 rows of run-to-run noise at k=5, which is larger
than several differences in §2. Central-field seed noise is small (~0.0003 in skill, measured
over three seeds), but the interval rows are not: rule 20 puts 44-85% of fold-to-fold
upper-width variance down to optimisation noise.

And it is Africa, not the globe. Southern Africa's far-field rate is 0.0000 and cannot screen
any of §3; the global grid has 5.2× the valid pixels and a far-field decrease population 88×
Africa's.
