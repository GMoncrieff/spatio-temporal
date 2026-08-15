# Next Phase — Marginals That Hold *Within* a Member

Branch `ensemble`, written 2026-08-15. Companion to `docs/current_progress.md` (status) and
`docs/central_field_baseline.md` (measurement record).

The previous phase fixed the central field: skill against persistence went from
−1.23/−0.24/+0.01/+0.05 to +0.09/+0.15/+0.18/+0.19 across horizons, and the scorecard from
73/137 to 91/128. This phase targets what that exposed.

---

## 1. The finding this phase exists to fix

Every realism diagnostic so far pooled all members together. Scored **per member** — which
is the right question, because a member is meant to be one plausible realisation of the
world — the ensemble fails badly.

`scripts/member_distance_relationship.py`, k=5, M=100, southern Africa:

> **The observation falls inside the member 5–95% range in 20% of (year × band × family)
> cells. A calibrated ensemble would give ~90%.**

The rank of the observation among 100 members, +20yr, `P(Δ > +0.05)`:

| band | observed | member mean | rank/100 |
|---|---|---|---|
| 0–1 px | 0.1767 | 0.3752 | **0/100** |
| 1–3 px | 0.0605 | 0.2561 | **0/100** |
| 3–10 px | 0.0234 | 0.1537 | **0/100** |
| 10–30 px | 0.0078 | 0.0254 | **0/100** |
| 30–100 px | 0.0024 | 0.0003 | **100/100** |

Rank 0 means *every* member produced more change than reality; rank 100 means every member
produced less. Not an off-centre spread — no member is on the right side at all. Pooling hid
this completely, because a pool that is uniformly too hot still overlaps the truth.

The **shape of the decay** is wrong too, and its error changes sign with lead time. Near/far
contrast `P(0–1px)/P(3–10px)`, observed vs per-member median [5–95%]:

| year | observed | two-piece normal | empirical shape |
|---|---|---|---|
| +5yr | 4.3 | 244.7 [135, 645] | 120.8 [68, 293] |
| +10yr | 7.2 | 9.2 [6.8, 14.2] | 7.0 [5.0, 11.1] |
| +15yr | 7.8 | 4.3 [3.1, 5.8] | 6.0 [4.0, 8.6] |
| +20yr | 7.5 | 2.5 [1.8, 3.2] | 4.4 [2.9, 6.3] |

Observed contrast is roughly flat at 4–8 across horizons. The members swing from 245 down to
2.5 — far too clustered at short lead times, too diffuse at long ones.

## 2. Why this is the marginals, and not the correlation field

An initial reading blamed the field's correlation structure. That was wrong, and the
arithmetic says so.

Per-band half-width against the residual it must cover, +20yr:

| band | median `w_up` | observed \|residual\| p97.5 | ratio | **+0.05 in units of `w`** |
|---|---|---|---|---|
| 0–1 px | 0.156 | 0.105 | 1.49 | **0.32×** |
| 1–3 px | 0.124 | 0.077 | 1.62 | 0.40× |
| 3–10 px | 0.091 | 0.058 | 1.58 | 0.55× |
| 10–30 px | 0.036 | 0.024 | 1.48 | 1.38× |
| 30–100 px | 0.0085 | 0.0047 | **1.83** | **5.88×** |

Two things follow.

**The width error is essentially uniform across bands** (1.48–1.83), so the *relative*
spatial pattern of the widths is roughly right and the field is not misplacing change. What
is wrong is the level — every interval is ~1.6× wider than the residual's own 97.5th
percentile, which is also what T1.1 reports as pooled coverage 0.963–0.982.

**But the last column explains the wildly non-uniform band errors.** The +0.05 threshold sits
at 0.32 half-widths in the near field and 5.9 in the far field. A uniform width error is
therefore mild where the threshold is inside the interval and catastrophic where it is far
outside — 2–7× too hot in the near/mid bands, 25× too cold beyond 30 px. One global width
factor cannot fix both, which is exactly what the `k*` experiment demonstrated: forcing
T1.1 to 4/4 destroyed T1.2 (13/16 → 8/15), T1.3 (3/3 → 0/2) and T8.4 (4/5 → 0/5).

Beyond 30 px the residual is a near-zero body with rare genuinely large values. The interval
is simultaneously too wide for the body (ratio 1.83) and unable to represent the tail
(`P(Δ>0.05)` at 0.04× observed). That is not a scale problem. **It is a shape problem, and
the shape needs to depend on distance to past change.**

## 3. What already exists to build on

`src/ensemble/copula.py` gained an empirical marginal shape this session
(`fit_residual_shape` / `apply_shape` / `invert_shape` / `shape_slope`, flag
`--marginal_shape`). It replaces the two-piece normal's Gaussian body with the residual's
own standardized quantile function, normalized so u = 0.025/0.5/0.975 still land exactly on
lower/central/upper — so T5.1 and T5.2 hold by construction, the map is strictly monotone
(the copula's rank structure and every spatial property are untouched), and a genuinely
Gaussian residual returns the identity.

**It is fitted once per horizon, pooled over all pixels.** That is the limitation this phase
should lift. Measured effect of the pooled version at matched M=400: scorecard 90/126 vs the
two-piece's 85/126, T6.1 ratios 3.84/5.65/6.75/6.65 → 1.74/2.88/4.36/4.55, and 12/40 vs 8/40
cells with the observation inside the member range. Better, but nowhere near calibrated.

Two properties are worth preserving in anything that replaces it:

- **the three gate quantiles must be exact knots** — interpolating near them leaves a bias
  that does not shrink with M (this cost a full M=400 run to discover);
- **beyond the 95% bound, revert to the two-piece normal.** The empirical tail there is
  dominated by pixels whose *width* is near-degenerate, so the ratio explodes for reasons
  about the denominator. Taking it at face value sent z = 3 to 5.4 half-widths.

## 4. The plan

**H1 — condition the marginal shape on distance to past change.** The primary hypothesis.
Fit `fit_residual_shape` per (horizon × distance band) instead of per horizon, apply per
pixel by band. Cheap: the residual rasters and the distance covariate are already on disk,
and the fit is a quantile grid. Success is the far-field band gaining a representable tail
without the near field inflating.

**H2 — fix the width level, per class rather than globally.** Widths are ~1.6× too wide
uniformly. A global factor is known to fail (§2). The interval-score decision rule chose
identity by a 0.03% margin over global, so it is nearly indifferent — worth re-running with
the class axis being *distance band* rather than Δ̂-bin, which is the axis the failures
actually organise along.

**H3 — check whether H1 and H2 are the same knob.** A distance-conditional shape changes the
effective spread as well as the shape. Fit both, and test whether the width correction is
still needed once the shape is conditional.

**H4 — if the above stalls, the widths are the model's.** The quantile heads already receive
the past-change context; they are producing intervals whose *level* is 1.6× too generous
across the board. That is a training-objective question (pinball at 0.025/0.975 on a spiky
target) and would mean reopening the head training, which is more expensive than H1–H3 and
should not be started first.

### How to judge it

**Primary metric: the fraction of (year × band) cells where the observation lies inside the
member 5–95% range, currently 20%.** Report the rank distribution, not just the fraction —
ranks pinned at 0 or M are the specific failure. `scripts/member_distance_relationship.py`
produces both.

Secondary: the near/far contrast per member against the observed 4–8, and the T1–T8
scorecard via `run_region_loop.sh`. Score at **M ≥ 400** — MC-scaled tolerances tighten as
1/√M, so a lower member count flatters whichever family is under-resolved, and the M=100
comparison between marginal families was confounded for exactly this reason.

Anything touching the marginal must keep T5.1/T5.2 exact and stay monotone in z. If a
diagnostic needs the normal score back, it must invert the shape (`invert_shape`), or it
measures a nonlinearly distorted field.
