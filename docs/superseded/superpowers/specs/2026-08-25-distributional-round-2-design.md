# Distributional ConvLSTM, round 2 — design

Branch `dist-convlstm`. Written 2026-08-25, after round 1 (`docs/dist_model_phase.md`,
`docs/dist_scorecard.md`).

## Why

Round 1 built an end-to-end distributional head and established three things well outside any
noise band: the model works and beats persistence as a full distribution by ~13%; pure CRPS
costs nothing centrally; and the trunk must hear the distributional objective. It adopted
nothing, because **run-to-run instability made six of ten variants undecidable** —
`tail_reach20` varies by 4.3 across three replicates of one configuration.

Round 2 has two jobs, in order. First make the harness able to decide things. Then attack the
named defects with levers that have distinct mechanisms.

The defects, from `docs/dist_scorecard.md`:

| defect | measurement |
|---|---|
| run-to-run instability | `tail_reach20` spread 4.3 within one configuration |
| location bias | `pit_mean` 0.40 against 0.50; `bias` negative at every horizon; increases over-predicted ~2x |
| deep lower tail too thin | `P(u<0.001)` = 0.0076 at h=20 against a nominal 0.001 |
| body width wrong, and wrong in opposite directions | `cov50` 0.42 at h=5, 0.63 at h=20 |
| change put in the wrong places | `corr` 0.2-0.3, `slope` 0.74; 2.5x too hot at 1-10 px, 0.40 of observed at 30-100 px |

## The information gap this round closes

The model's 8 context channels are: five occupancy indicators (`dist <= r` to past *change*),
`log1p(dist)/10`, the signed past change, and `hm_now`. **There is no neighbourhood HM
information at all.** The model knows where change happened out to 100 px, and knows the
*level* of development only at the pixel itself plus whatever the trunk's ~10 px reach supplies.

Development spreads from development, not only from recently-changed ground. That is a
covariate the model has never had, and it matches round 1's own conclusion — and the previous
ConvLSTM phase's — that the central field needs new information rather than new architecture.

The covariate is **threshold-free**: neighbourhood *means* and *maxima* of HM, not a distance to
an HM cutoff. A mean at radius r encodes both how close development is and how much of it there
is; a distance-to-threshold encodes only the first, and needs a cutoff nothing justifies.

---

## 1. Order of work

1. Build and run the HM-context precompute (CPU/IO bound, ~1.5 h).
2. Launch the Phase 1 stability gate; implement the Phase 2 flags while it trains (the gate is
   GPU-bound, the flag work is not).
3. Read the gate, adopt the winner, re-measure the floor from the winner's replicates.
4. Run Phase 2 against the new floor.

Heavy CPU work does not run alongside prediction: a single concurrent analysis job has been
measured cutting prediction throughput from 170 to 135 tiles/s.

## 2. Phase 1 — the stability gate

**Four** configurations, three seeds each, judged **only on the spread across seeds** and never
on the level. The comparator is D0's existing three replicates.

| id | flags | mechanism |
|---|---|---|
| **S0** | `--checkpoint_select final` | **the control.** Stops selecting an epoch, without averaging |
| S1 | `--weight_avg_last 20` | averages the last 20 epochs' weights; the canonical lever against a bimodal trajectory. Exists in the codebase, never tested on a distributional head |
| S2 | `--lr_schedule cosine` | ends training at ~1% of the base learning rate, settling into a flatter basin |
| S3 | S1 + S2 | |

**S0 was added after reading the code, and it is what makes S1 interpretable.** Weight averaging
rewrites the weights in `on_train_end`, and `train_lightning.py:1099` then repoints prediction at
the end-of-training checkpoint — so under S1 and S3 `--checkpoint_monitor val_crps` stops
mattering entirely. Those configurations therefore change *two* things at once: they average,
**and** they abandon epoch selection. Without S0 a win could be either, and this project has
already paid once for introducing two changes together and reading the result as one
(CLAUDE.md rule 10). S0 isolates "stop selecting" from "average" at the cost of three runs.

**Acceptance:** a configuration is adopted if the spread of `tail_reach20` across its three
seeds is materially smaller than D0's 0.86 *and* — because D0's own band was an under-sample —
smaller than the 4.3 seen across d1a's replicates, while its median `crps_skill5`, `pit_ks5` and
`rmse5` are no worse than D0's median.

**Fingerprinted, not assumed.** `verify_gate_flags` in `scripts/dist_base_args.sh` reads each
run's own log back and refuses to score it unless the lever left its signature: the
`[weight averaging] wrote the mean of the last N epochs` line, the
`Prediction will use the end-of-training checkpoint` line, and a `[lr schedule] cosine` line
added for the purpose — `lr` was previously logged only to W&B, so a cosine run that silently
failed to engage would have been indistinguishable from one where the schedule did nothing.
Proven discriminating in both directions against a baseline log.

**Not tested:** whether checkpoint selection *noise* drives the instability. Round 1 measured the correlation between selected epoch
and `tail_reach20` at **0.150**, with counterexamples in both directions (d7 selected epoch 31
and reached 4.96; d1a_s43 selected epoch 139 and reached 1.61). The bimodality is in the
trajectory, not in which point of it is picked. A run spent on it would be a run spent on a
hypothesis already rejected.

**If none wins:** Phase 2 proceeds against the round-1 bands with more seeds per configuration,
and the writeup says the harness could not be improved. That is a worse outcome but a known one.

## 3. Phase 2 — the suite

Eight configurations. Seeds per configuration are set by what the gate buys: two if the gate
tightened `tail_reach20` spread below ~1.0, three otherwise.

| id | change | defect targeted |
|---|---|---|
| **E1** | the covariate — 12 context channels | `corr`, far-field coldness, near-field heat |
| E1a | covariate, `mean` only | isolates whether `max` adds anything over `mean` |
| **E2** | `--spline_knots body_dense` | `cov50`; PIT shape |
| E3 | `--spline_knots deep_lower` | `P(u<0.001)` 7x too thin |
| E4 | `--crps_tail_weight_lo 1.0` | same |
| E5 | `--horizon_loss_weights 1,1.33,2,4` | `cov50` diverging across horizons |
| E6 | `--shape_head_hidden_layers 2` | shape head may be under-capacity |
| E7 | `--shape_head_width 64` | same |

E2's motivation is worth stating because it corrects a choice made in round 1. The knot grid is
**tail-dense and body-sparse**: between u = 0.10 and u = 0.90 there are exactly three knots,
while 53% of pixels move by less than 0.001 over twenty years. The model must represent a spike
holding half its mass with three knots and spend six on the tails. `cov50` is the worst
calibrated coverage level. The grid was chosen to buy far-field tail resolution and
under-resourced the body.

E5 revisits a lever the previous ConvLSTM phase rejected (E2 there). That rejection was measured
under pinball on the central field; the exposure imbalance is 4:3:2:1 by horizon and this is a
different objective on a different head, so the verdict does not transfer.

---

## 4. New component: `scripts/prepare_hm_context.py`

Per base year, one raster of six bands:

| band | content |
|---|---|
| 1-3 | mean HM within r = 3, 30, 100 px |
| 4-6 | max HM within r = 3, 30, 100 px |

**Computed on the full raster**, never inside a chip. A 201x201 window cannot be evaluated
inside a 128 px chip; the measured precedent is the past-change context, where a 100 px radius
derived from a chip degenerated into "is there any change in this chip" at 0.752 mean occupancy
for a single changed pixel.

Streamed by row block with an `r`-row halo, using separable filters
(`scipy.ndimage.uniform_filter1d` and `maximum_filter1d`). Written int16 x 1/32767 with nodata
-32768: HM is bounded on [0, 1], so this is lossless to 3e-5. Output
`data/raw/hm_global/hm_context_w{year}_1000.tif`, beside the other model inputs.

**Measured, and an order of magnitude cheaper than this spec first estimated: 0.75 GB and 2:45
per base year, 22.9 GB peak RAM** — 3.8 GB and ~14 minutes for all five, against the projected
41 GB and 1.5 h. The estimate assumed the raw float32 volume; deflate compresses a smoothed
field very hard, and the separable filters are far faster than a 201x201 convolution.

Sanity, on southern Africa: the means smooth monotonically with radius (sd 0.095 -> 0.087 ->
0.084), `max >= mean` at every radius, `max` non-decreasing in radius, and the correlation with
the pixel's own HM decays **0.907 -> 0.728 -> 0.620**. That last number is the one that matters:
at r = 100 nearly 40% of the covariate's variance is not present in the channel the model
already had.

Base years 2000, 2005, 2010, 2015, 2020 — the four hindcast windows plus the forward product's.

## 5. Changes to existing code

### 5.1 The context channel budget becomes a function, not a constant

`N_CONTEXT_CHANNELS` is currently `len(CONTEXT_RADII) + 3` = 8, a module constant read at two
places in `train_lightning.py`. It becomes `context_channel_count(context_radii,
hm_context_stats, hm_context_radii)`.

Channel layout, in order:

```
occ(dist_change <= r)   for r in context_radii      len(context_radii)
log1p(dist_change) / 10                             1
past_change                                         1
hm_now                                              1
mean_HM at r            for r in hm_context_radii   len(hm_context_radii)   if 'mean' in stats
max_HM  at r            for r in hm_context_radii   len(hm_context_radii)   if 'max'  in stats
```

New flags, all additive, **every default reproducing today's 8 channels exactly** so D0 stays
reproducible and the gate's results stay comparable:

| flag | default | E1 sets |
|---|---|---|
| `--context_radii` | `1,3,10,30,100` | `3,30,100` |
| `--hm_context_stats` | `` (none) | `mean,max` |
| `--hm_context_radii` | `3,30,100` | — |
| `--hm_context_pattern` | `data/raw/hm_global/hm_context_w{year}_1000.tif` | — |

### 5.2 Missing context stops being silent

`SpatioTemporalPredictor._with_context` substitutes **zeros** when the context tensor is absent,
so a mis-wired run trains on a zeroed covariate and degrades silently. With the channel count
about to become variable that is a live hazard rather than a latent one.

It will raise instead, on both a `None` context when channels are expected and on a channel-count
mismatch. The expected count is stored as a buffer so it survives a checkpoint round trip, and the
discriminating control is that an 8-channel checkpoint must be **rejected** by a 12-channel model.

One call site needs fixing rather than exempting: `train_lightning.py`'s full-set evaluation calls
the model without `change_context`, so under `--central_context`/`--quantile_context` its W&B test
metrics have been computed with the context channels zeroed. It will pass the context.

### 5.3 Knot presets

`--spline_knots {default14, body_dense, deep_lower}`, default `default14`.

| preset | knots | bins | params/horizon |
|---|---|---|---|
| `default14` | the round-1 grid, 15 knots | 14 | 29 |
| `body_dense` | + 0.35, 0.45, 0.55, 0.65 | 18 | 37 |
| `deep_lower` | + 0.0001, 0.9999 | 16 | 33 |

Every preset must be strictly increasing and contain 0.025, 0.5 and 0.975 as exact knots, since
the published triple is a lookup rather than an interpolation.

### 5.4 Two-sided tail weighting

```
w(u) = 1 + lam_hi * ((u - u0_hi) / (1 - u0_hi))_+ ^ p
         + lam_lo * ((u0_lo - u) / u0_lo)_+ ^ p
```

New flags `--crps_tail_weight_lo` (default 0.0) and `--crps_tail_u0_lo` (default 0.05). The
existing upper-side flags are unchanged. A u-weighting changes where the optimiser spends effort
and never the target, so no importance correction is involved.

### 5.5 Shape head capacity

`--shape_head_hidden_layers` (default 1) and `--shape_head_width` (default 0, meaning
`hidden_dim // 2`). These affect the shape head only — `--head_hidden_layers` applies to every
head through the shared factory and would confound a shape-head experiment with a change to the
central head.

## 6. Testing

The halo is the bug surface in the precompute, and a working set that only the never-yet-run
branch reaches has bitten this project three times. So:

- mean and max against brute-force `scipy.ndimage.uniform_filter` / `maximum_filter` on a
  synthetic raster, including NaN handling;
- **block-streaming equivalence at several block sizes**, asserted against a single-block
  computation;
- int16 round-trip within 3e-5.

And for the model changes:

- `context_channel_count` for every flag combination, with the default asserting exactly 8;
- the channel builder's output ordering, and that defaults are byte-identical to today;
- a `None` context and a wrong-width context each raise, with an 8-channel state dict rejected
  by a 12-channel model as the discriminating control;
- each knot preset strictly increasing, containing the three gate quantiles, and `default14`
  identical to `U_KNOTS_DEFAULT`;
- lower-side tail weight symmetric to the upper, and `lam_lo = 0` a no-op;
- shape-head flags change the shape head's parameter count and nothing else.

The whole suite must end at the seven documented pre-existing failures and no others.

## 7. Cost

| stage | cost |
|---|---|
| HM-context precompute | **14 min measured** (est. 1.5 h) |
| implementation | ~3-4 h |
| Phase 1 gate, **12** runs (4 configs x 3 seeds) | ~10 h |
| Phase 2, 16-24 runs | 13-20 h |
| **total compute** | **~27-35 h** |

## 8. Out of scope

- The far field. Southern Africa has zero pixels beyond 100 px from past change, so no verdict
  on the round-1 headline defect is reachable at this extent whatever is built.
- A distance-to-HM stratification axis for the scorer. Considered and declined: the model input
  is threshold-free, and adding a thresholded axis to the scorecard alone was not worth the
  scope.
- Ensemble promotion. Round 2 ends at a candidate, as round 1 did.
