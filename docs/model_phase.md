# The Model Phase — Improving the ConvLSTM Itself

Branch `ensemble`, southern Africa, started 2026-08-16. Companion to
`docs/next_phase_model.md` (the plan), `docs/next_phase_marginals.md` §5-6 (the closed
post-hoc phase) and `docs/central_field_baseline.md` (the last time the model was changed).

Baseline to beat: scorecard **101/127**, per-member **15/20**, from
`data/ensemble/exp/e5_all_k5` with the post-hoc marginal and width factors.

---

## 1. What reading the model code found

Three things, none of them previously recorded. All were found by reading, and all were
confirmed by measurement before being acted on.

### 1.1 The histogram loss has never trained anything

`compute_histogram` (`src/models/histogram_loss.py:13`) builds its counts by boolean
comparison and `.sum()` into a `torch.zeros` buffer, so the returned loss is a constant with
respect to the prediction. Confirmed directly: `requires_grad` is `False`, `grad_fn` is
`None`, and `backward()` raises *"element 0 of tensors does not require grad"*. Pinned as
`tests/test_model_phase_flags.py::test_histogram_loss_carries_no_gradient`.

It has been in the central objective at weight 1.0 for the whole project. The cost is not
the wasted compute; it is that the term **is** in `val_total_loss`, which is what
`ModelCheckpoint` selects on, and it swings by ~60× between adjacent epochs (h=20: 0.01047
at epoch 147, 0.00017 at 148). Epoch selection has therefore been partly a lottery, which
is a candidate mechanism for the run-to-run variance §2 measures.

### 1.2 The horizons are trained 4:3:2:1

`end_year` is sampled uniformly from `(2000, 2005, 2010, 2015)` (`train_lightning.py:465`)
and any target year past 2020 is NaN-filled and masked out
(`torchgeo_dataloader.py:487`). So a training sample yields an h=5 target from all four
windows, h=10 from three, h=15 from two, and **h=20 from one**. The largest-error horizon
receives a quarter of h=5's gradient.

Validation uses fixed years (`use_temporal_sampling=False`), so it is balanced across
horizons and **nothing in `val_*` shows this**. `--horizon_loss_weights` compensates.

### 1.3 The quantile heads have no channel carrying the predicted change

Under `--monotone_quantile_width` the interval is built around `pred_central.detach()`, but
that anchor is never an *input* to the head. One of the three axes along which the measured
width error varies is predicted change — 0.75 for Δ̂ ∈ (0.01,0.05] against 0.95 for
(0.05,0.15] inside a single distance band — and the head cannot see it.
`--quantile_dhat_context` supplies it, derived from tensors already in the forward pass.

### 1.4 Other survey notes

- The trunk is a **4-layer** ConvLSTM (`PRODUCTION_HPARAMS`, not the argparse default of 2),
  `hidden_dim=64`, 3×3 kernels, T=3. The ~10 px radius follows from the kernel size alone.
- SSIM (0.2) and the Laplacian pyramid (0.3) are computed on **absolute HM** — the
  parameterisation the central head abandoned when `--central_residual` landed.
- The optimizer is bare Adam at 1e-3: no schedule, no weight decay, no clipping.
- Checkpoint selection runs on **35 validation chips** (`--val_stride 2048`).

---

## 2. The noise floor, measured before anything was compared

Three seeds of the reference configuration, folds 1 and 2, 150 epochs, scored on identical
pixels (1,542,872 at h=5).

| metric | seed 42 | seed 43 | seed 44 | spread | relative |
|---|---|---|---|---|---|
| RMSE h=5 | 0.00941 | 0.00936 | 0.00935 | 0.00005 | 0.6% |
| RMSE h=10 | 0.01447 | 0.01438 | 0.01437 | 0.00010 | 0.7% |
| RMSE h=15 | 0.01849 | 0.01872 | 0.01890 | 0.00041 | 2.2% |
| RMSE h=20 | 0.02168 | 0.02204 | 0.02259 | 0.00092 | 4.2% |
| skill h=20 | 0.2132 | 0.1867 | 0.1453 | **0.0679** | **38%** |
| slope h=20 | 1.018 | 0.807 | 0.653 | 0.365 | — |
| mean\|ln k\| | 0.7621 | 0.7805 | 0.7900 | 0.0279 | 3.6% |
| within 20% | 0.152 | 0.124 | 0.157 | 0.0328 | 22% |

**RMSE is a usable A/B metric; skill and slope are not.** Skill is `1 − MSE/MSE_persistence`
and the model's margin over persistence is small, so a 4% RMSE move becomes a 38% skill
move. Every judgement in this phase is therefore made on RMSE, with skill reported as a
derived quantity rather than used as a decision rule. The published k=5 headline (skill
0.1909 at h=20) sits inside this two-fold band, which is a statement about the band, not
about the headline.

The floor is worst exactly where §1.2 predicts: h=20, the horizon trained on a quarter of
the data and evaluated on a single window.

**The edits are not the cause.** The current model at default flags was checked against the
pre-edit code (`41f8a7e~1`) directly: same 56 parameter tensors, bit-identical
initialisation under a common seed, and `max |old − new| = 0.0` on a forward pass.

### 2.1 Where the variance actually comes from — and a standing hypothesis that is wrong

The first explanation tried was checkpoint selection, and the first measurement of it was
**wrong**: the selected epoch was read off the per-epoch printed "Total Loss", which is the
batch-0 print for the 20 yr horizon, not the epoch-aggregated quantity `ModelCheckpoint`
monitors. The real selected epochs, taken from the checkpoint filenames:

| run | fold 1 | fold 2 | monitor |
|---|---|---|---|
| e5_all_k5 | 85 | 68 | `val_total_loss` |
| nf_s43 | 134 | 138 | `val_total_loss` |
| nf_s44 | 145 | 81 | `val_total_loss` |
| ref_s42 | 137 | 75 | `val_central_loss` |
| ref_s43 | 139 | **17** | `val_central_loss` |

Selection really is spread over almost the whole run — 17 to 145. But the conclusion drawn
from that was also wrong, and the check that broke it is the one that matters:

**Epoch 17 and epoch 75 produce the same regional error.** Scored per fold, `ref_s43` fold 2
(epoch 17) against `ref_s42` fold 2 (epoch 75): h=5 0.00930 vs 0.00929, h=10 0.01441 vs
0.01416, h=15 0.01848 vs 0.01833, h=20 0.02204 vs 0.02208.

The validation curve says why. h=20 validation MSE, by epoch:

| run · fold | ep 5 | ep 10 | ep 20 | ep 40 | ep 60 | ep 100 | ep 149 | argmin |
|---|---|---|---|---|---|---|---|---|
| ref_s42 · f1 | 0.00430 | 0.00457 | 0.00474 | 0.00399 | 0.00581 | 0.00386 | 0.00598 | ep 146 |
| ref_s43 · f1 | 0.00435 | 0.00401 | 0.00454 | 0.00536 | 0.00432 | 0.00422 | 0.00354 | ep 33 |
| nf_s43 · f1 | 0.00395 | 0.00352 | 0.00344 | 0.00330 | 0.00408 | 0.00342 | 0.00406 | ep 51 |
| ref_s42 · f2 | 0.08871 | 0.08716 | 0.08246 | 0.09947 | 0.07560 | 0.08317 | 0.07138 | ep 75 |

**There is no downward trend after epoch 5.** The model reaches a plateau almost immediately
and then oscillates on it for 145 epochs, and `save_top_k=1` selects the luckiest draw from
that oscillation on 34 validation chips — which is selection on noise, and does not transfer.

Two consequences.

1. **`docs/next_phase_model.md` §4 and `current_progress.md` item 1 say the central field is
   under-trained and that training budget is "the cheapest untested lever". That is not what
   the data shows.** The budget experiment is still being run, as the direct test of this,
   but the prediction is now that it does nothing.
2. **The variance is in the training trajectory, not in the selection.** Averaging the tail
   epochs of an oscillating plateau is the natural response, which is what
   `--weight_avg_last` does — better motivated by this measurement than by the wrong one it
   replaced.

---

## 3. The measurement instruments

`scripts/score_model_experiment.py` reads a stitched prediction directory and returns, in
~40 s and with no ensemble, recalibration or residual stage:

- the central error against persistence, per horizon and stratified by distance band,
  predicted change and HM level;
- the **median standardized residual by predicted-change class** — the named central-head
  defect;
- the per-class **unstretch factors** `k_up`/`k_lo`: the multiplier that would make the
  published interval the residual's own 95% interval. A model whose width heads are right
  needs `k = 1` everywhere, so `mean|ln k|` over leaf (band × Δ̂ × HM) classes is a single
  number for *how much post-hoc width correction the model is still asking for*. That is
  this phase's primary quantile metric.

Conventions are imported rather than re-derived — `distance_band` (right=True), `DHAT_BINS`,
`HM_BINS`, `Z975` — and `e` is standardized exactly as `fit_width_factors.py` standardizes
it, so the two agree by construction.

The reference configuration reads:

| band | k_up h=5 | h=10 | h=15 | h=20 |
|---|---|---|---|---|
| 0–1 px | 0.745 | 0.702 | 0.655 | 0.611 |
| 1–3 px | 0.567 | 0.544 | 0.546 | 0.524 |
| 3–10 px | 0.468 | 0.321 | 0.372 | 0.399 |
| 10–30 px | 0.864 | 0.452 | 0.333 | 0.232 |
| 30–100 px | **2.233** | 1.044 | 0.669 | 0.252 |

Every class wants a narrower interval except the far field at short lead, which wants one
**2.2× wider** — the sign flip `docs/next_phase_model.md` §2 records, reproduced here from
raw stitched rasters with no recalibration in the path. Read down the last row: the
far-field width grows far too fast with lead time.

`mean|ln k|` = 0.762, with only 15% of 92 leaf classes within 20% of correct.

### 3.1 A sharper statement of the width defect, and a prediction registered before the test

`k_up(h=20) / k_up(h=5)` is the factor by which the model's interval growth over lead time
is wrong, with the published widths cancelling out:

| band | 0–1 px | 1–3 px | 3–10 px | 10–30 px | 30–100 px |
|---|---|---|---|---|---|
| ratio | 0.82 | 0.92 | 0.85 | **0.27** | **0.11** |

In the near field the model's width grows with lead time about correctly. **In the far field
it grows 3.7× to 8.9× too fast.** Under `--monotone_quantile_width` the width is a cumulative
sum of per-horizon softplus increments, each head initialised so its increment is ≈0.065; the
far field's increments should collapse toward zero after h=5 and evidently do not.

The mechanism this points at is the pinball gradient. `d(loss)/d(raw)` is a constant times
`sigmoid(raw)`, which at the far field's width is ~0.0085 against ~0.14 in the near field —
so the far-field width parameters learn ~18× slower, on exactly the pixels furthest from
right, and in a 150-epoch budget they barely leave their initialisation.

**Registered prediction: E6b (scale-normalised pinball) is the experiment that should move
the far field**, and E6a (multiplicative width) should not move it much on its own, because
it changes `d(step)/d(raw)` in the same proportion as the loss it is divided into. Recorded
before either was run so the result reads as a test rather than a story.

---

## 5. E5 at k=5, and the trade it makes

Trained on all five folds and scored over the whole region (4,358,388 px at h=5), against the
shipped configuration scored identically:

| | baseline k=5 | E5 k=5 | Δ |
|---|---|---|---|
| RMSE h=5 | 0.00947 | 0.00945 | −0.3% |
| RMSE h=10 | 0.01444 | 0.01432 | −0.8% |
| RMSE h=15 | 0.01855 | 0.01823 | −1.7% |
| RMSE h=20 | 0.02222 | 0.02206 | −0.7% |
| **mean\|ln k\|** | 0.732 | **0.686** | **−6.3%** |
| within 20% | 0.186 | 0.193 | +3.8% |

The central moves are all in the right direction and all small; with no k=5 seed replicate they
are not claimed past noise. The width metric reproduces the screening result.

**The post-hoc layer is asked to do less**, which is the phase's actual objective. Over the
same 360 leaf classes, the fitted half-width factors move from `mean|ln k|` 0.550 to 0.470,
and the median upper factor from 0.540 to 0.693 — a third less narrowing demanded of the
correction the model should not need.

### 5.1 The trade, found before the scorecard by the closed-form predictor

`scripts/predict_change_rates.py` shares no code with the GPU sampler and answers in 30 s.
`P(Δ > +0.05)`, observed against predicted:

| band, horizon | observed | baseline | E5 |
|---|---|---|---|
| 30–100 px, h=5 | 0.000155 | 0.000011 | **0.000000** |
| 30–100 px, h=10 | 0.000776 | 0.000055 | **0.000003** |
| 30–100 px, h=15 | 0.001107 | 0.000193 | **0.000044** |
| 30–100 px, h=20 | 0.002356 | 0.001185 | **0.001622** ✓ |
| 10–30 px, h=20 | 0.007747 | 0.020641 | **0.013339** ✓ |

Mean \|log₁₀ ratio\| 0.300 baseline against **0.381** E5 — worse overall, and concentrated
entirely in the far field at short lead.

The mechanism is E5's own. The width now depends on Δ̂; in the far field at short lead Δ̂ ≈ 0,
so the head narrows there and pushes the +0.05 threshold further out of reach. **E5 buys
per-class width calibration across the many near and mid classes and pays for it in the
far-field short-lead tail** — which §5.5 of `docs/next_phase_marginals.md` already measured as
unreachable (median 119 half-widths) and §4.0 above identified as not a width-head problem.

*Prediction registered before the scorecard: T1/T2 class-coverage rows improve, T6/T8 far-band
rows at short lead regress.*

### 5.2 Downstream — the decisive test

`SHAPE=measured WIDTHS=measured run_region_loop.sh e5k5 400`, everything re-derived from E5's
own residuals. Recalibration came back **identity**, as for the baseline.

| | baseline | E5 |
|---|---|---|
| **scorecard** | 101/127 | **104/128** |
| per-member cells inside the 5–95% range | 15/20 | 15/20 |
| **mean \|rank − M/2\|** | 115.6 | **103.3** |

Six rows moved, four up and two down:

| row | baseline | E5 | |
|---|---|---|---|
| T2.3 area above threshold | 3/8 | **5/8** | +2 |
| T3.1 member variogram | 1/3 | **2/3** | +1 |
| T6.4 change-sign realism | 2/2 | **3/3** | +1 |
| T8.1 change clustering by band | 4/5 | **5/5** | +1 |
| T2.5 rank histogram | 4/4 | 3/4 | −1 |
| **T5.1 median ≡ central (hard gate)** | 4/4 | **3/4** | −1 |

**The registered prediction was wrong, in the favourable direction.** T8.1 and T6.4 — the
far-band realism rows E5's narrowing was expected to damage — both *improved*, T8.1 to 5/5.
The closed-form predictor's warning was real but did not reach the gates it was expected to.

**Both losses are at the 2015 horizon and both are percentile-estimated from the member
sample.** T5.1 goes 0.9958 → 0.9888 against a 0.995 hard gate; T2.5's chi-squared p goes 0.164
→ 0.0075. h=15 is also where E5's central field improved most (−1.7% RMSE), and a sharper
central field gives a more concentrated residual, hence a spikier fitted shape, which
`docs/next_phase_marginals.md` measured as harder to read percentiles from at fixed M. That is
a plausible mechanism, not a demonstrated one — it rests on a single k=5 run. **A hard gate
failing at one horizon is the item to resolve before this ships**, most cheaply by rescoring at
higher M.

### 5.3 The two losses have an evidenced mechanism, not a story

The finite-M explanation was checked rather than asserted. `shape_slope(shape, 0)` is the
factor a Monte-Carlo tolerance needs at the median; a smaller value means a spikier marginal
there and a noisier sample median in value units.

| horizon | baseline `S'(0)` | E5 `S'(0)` | ratio |
|---|---|---|---|
| 5 | 0.2438 | 0.1383 | 0.567 |
| 10 | 0.2608 | 0.1866 | 0.716 |
| **15** | 0.3640 | 0.2411 | **0.662** |
| 20 | 0.6200 | 0.4700 | 0.758 |

**E5's residual is more concentrated at every horizon** — a sharper central field, which is
what it was meant to be. Spikiness alone does not single out h=15 (h=5 is spikiest and passes),
but h=15 was already the horizon closest to the gate in the baseline, 0.9958 against 0.995
where the others sat at 0.9982 / 0.9985 / 0.9998. It is the one that crossed.

**This means 104/128 probably understates E5.** Both configurations were scored at M=400, so
the comparison is matched and fair — but E5 is *more* penalised by finite M than the baseline
is, because the rows it loses are the ones its own sharper residual makes harder to estimate
from 400 members. This is the same confound `docs/next_phase_marginals.md` hit when the M=100
comparison between marginal families reversed at M=400. Rescoring both at M=800 would settle
the hard gate and the row count together, and is the obvious next measurement.

Honest scale of the result: +3 rows of 128, scorecard rows are correlated, and there is no k=5
replicate to put a band on any of it. The per-member count is unchanged at 15/20; what moved
is how far the observation sits from the member centre, 115.6 → 103.3. This is nonetheless the
first time in this project that a change to **the model** rather than to the post-hoc layer has
moved these numbers.

### 5.4 Africa-wide — the gain does not generalise

Trained k=5 on the Africa extent (35,650,217 px per window × horizon, 63.1 Mpx raster, 88.7
min) and scored against the existing Africa-wide run of the shipped configuration.

| | Africa | southern Africa |
|---|---|---|
| RMSE h=5 | **+0.16%** | −0.26% |
| RMSE h=10 | −0.12% | −0.82% |
| RMSE h=15 | −0.03% | −1.69% |
| RMSE h=20 | **+0.81%** | −0.72% |
| `mean\|ln k\|` | **−1.20%** | −6.31% |
| within 20% | **−0.18%** | +3.76% |

**Every Africa number is inside ±1.2%. E5's gain is southern-Africa-specific.**

The absolute values say why. On Africa *both* configurations are far better width-calibrated
than either is on southern Africa:

| | `mean\|ln k\|` | within 20% |
|---|---|---|
| Africa, baseline | 0.391 | 0.421 |
| Africa, E5 | 0.386 | 0.420 |
| southern Africa, baseline | 0.732 | 0.186 |
| southern Africa, E5 | 0.686 | 0.193 |

**The width defect this whole phase optimised against is largely a southern-Africa
artifact.** On a fairer sample of HM level the model's intervals are roughly twice as
well calibrated to begin with, so there is much less for a Δ̂-conditional width to fix.
CLAUDE.md's rule 14 — "a stratified finding measured only there is provisional" — has now
cost a real result, and it should be read as applying to everything measured in this phase.

Note the central error is *larger* on Africa (RMSE h=5 0.01204 against 0.00947), so the
central field has more room there, not less. The regional screening protocol is right for
speed and wrong for target selection.

### 5.5 The noise floor was under-estimated, and it withdraws the decision

Round 2 began by running the E5 configuration twice more as its own base. Three runs of the
**identical** configuration, same folds, same pixels:

| run | RMSE h=15 | `mean\|ln k\|` | within 20% |
|---|---|---|---|
| `e5_dhat` (the screening run) | 0.01845 | **0.681** | **0.236** |
| `r0_base_s42` | 0.01919 | 0.744 | 0.181 |
| `r0_base_s43` | 0.01831 | 0.769 | 0.179 |
| **spread** | **0.00087** | **0.088** | **0.057** |

The floor in §2 was measured on three seeds of the *incumbent* and gave 0.028 and 0.033. **The
E5 configuration's own spread is three times that**, and rmse15's is twice its earlier value.
A floor is not a property of the harness; it is a property of the configuration, and one
configuration's spread does not bound another's.

Three consequences, all against the result:

1. **E5's screening win was a lucky draw.** `e5_dhat` is the best of three runs of that
   configuration, and the other two score *worse* on `mean|ln k|` than both reference seeds.
2. **The k=5 gain (0.732 → 0.686, −0.046) is inside the 2-fold same-configuration spread of
   0.088.** k=5 averages five fold models over four times the pixels so its floor is smaller,
   but there is no k=5 replicate to say by how much.
3. **The Africa-wide null becomes the most trustworthy single measurement** of E5's effect —
   same k=5 protocol, eight times the pixels, −1.2%.

### 5.6 Decision — withdrawn; E5 is not established

**E5 is not adopted.** The claimed effect is smaller than the run-to-run spread of its own
configuration, it does not reproduce Africa-wide, and it costs a hard gate. What survives is
that it is cheap, harmless, and moved the downstream in the right direction on a single k=5
run — enough to keep it as a candidate, not enough to make it the reference.

What would settle it: two or three k=5 replicates of E5 and of the baseline, scored downstream,
which is the only comparison at the scale the product is delivered at. That is ~105 min per
replicate.

**The general lesson, which is the more valuable output:** this phase's screening protocol —
one run per configuration on two folds — cannot resolve effects of the size being chased. Every
verdict in §4 that rests on a single run and a margin near the floor should be read as
provisional. The clear results (E3b, E7, E1, E9) are the ones whose margins are several times
the floor or whose stratification localises a mechanism; those stand.

### 5.7 Superseded decision (kept for the record) — adopt, narrowly

**E5 becomes the reference model.** It improves the delivered numbers in the scope the
product is calibrated for (scorecard 101/127 → 104/128, per-member centring 115.6 → 103.3),
its mechanism is confirmed rather than inferred, it costs two extra input channels on the
quantile heads and nothing at inference, and it is neutral rather than worse on the fairer
Africa sample.

**It is not shippable as it stands.** T5.1, a hard gate, fails at one horizon (0.9888 against
0.995). §5.3 gives an evidenced mechanism and predicts that rescoring at M ≥ 800 recovers it;
that measurement should be made before this configuration is published.

Honest summary of the size of the win: small, region-specific, and resting on a single k=5
run per configuration.

## 6. Round 2 — substantial changes to architecture and objective

Round 1 established that no *optimisation* lever moves this model. Round 2 therefore changes
what the model can express. Base is the round-1 candidate (E5), run three times so the bar is
that configuration's own spread.

### 6.1 How round 2 is judged, and why the round-1 bar was wrong

A single new draw falls outside the range of three base runs roughly half the time under the
null, so "outside the base range" is barely a test at all. **The bar used here is that the
margin beyond the base range must exceed the range's own width.** Applying the looser test
first produced two false positives that the stricter one removes (`r2_wjoint`, `r8_asinh`),
which is exactly the failure mode round 1 fell into with a floor borrowed from a different
configuration.

| run | rmse5 | rmse10 | rmse15 | rmse20 | mean\|ln k\| | within20 | growth (far) |
|---|---|---|---|---|---|---|---|
| base ×3 | .00930–.00938 | .01434–.01441 | .01831–.01919 | .02178–.02225 | .681–.769 | .179–.236 | .077–.354 |
| **r1_wpower** | .00932 | **.01426** | .01829 | .02178 | .787 | .184 | **0.812** |
| r2_wjoint | .00936 | .01432 | .01832 | .02172 | .653 | .255 | .128 |
| r3_headdeep | .00936 | .01448 | .01872 | .02165 | .724 | .237 | .427 |
| r6_nll | .00933 | .01436 | .01838 | .02176 | **1.008** | .126 | .392 |
| r8_asinh | .00931 | .01433 | .01888 | .02172 | .768 | .157 | .365 |
| r4_wide (dim 128) | .00934 | .01435 | .01844 | .02180 | .713 | .151 | .140 |
| r5_shallow (2 layers) | .00937 | .01430 | .01866 | .02185 | .857 | .150 | .302 |

Clearing the bar:

| run | result |
|---|---|
| **r1_wpower** | growth ratio **better by 1.7×** the base spread; rmse10 better by 1.1× |
| r6_nll | `mean\|ln k\|` **worse by 2.7×** |
| r3_headdeep | rmse10 worse by 1.0× |
| r5_shallow | `mean\|ln k\|` worse by 1.0× |
| r2_wjoint, r8_asinh, r4_wide | nothing clears the bar |

### 6.2 The power-law width head fixes the defect it was built for

`k_up(20)/k_up(5)` is how much too fast the interval grows with lead time; 1.0 is correct.

| run | 10–30 px | 30–100 px |
|---|---|---|
| base s42 | 0.414 | 0.354 |
| base s43 | 0.252 | 0.105 |
| e5_dhat | — | 0.077 |
| **r1_wpower** | **1.230** | **0.812** |

The base has the far-field width growing three to nine times too fast. Parameterising it as
`w(h) = w0·(h/5)^γ` — two per-pixel parameters instead of four free increments — brings both
bands to essentially correct. Its `mean|ln k|` is *not* better (0.787 against .681–.769), so
the two parameters buy a correct growth **shape** at some cost in per-horizon level freedom.
That trade is what `power_plus` was added to test.

### 6.3 Why the log score fails, which is more useful than that it fails

`r6_nll` fits (lower, central, upper) as a two-piece normal by log score. It is the worst
result of either round on width calibration, and **in the opposite direction to the
expectation** — the intervals got far *wider*, not narrower:

| band, h=5 | base `k_up` | NLL `k_up` |
|---|---|---|
| 10–30 px | 0.743 | 0.157 |
| 30–100 px | 1.517 | 0.198 |

Gaussian NLL is not robust. Its quadratic term is dominated by rare extreme residuals — this
residual has kurtosis 928–20366 — while the `log(σ_lo+σ_up)` penalty resists only
logarithmically, so a handful of outliers inflate the fitted scale. **Pinball at 0.025/0.975
has bounded influence per pixel and cannot be dragged that way.** This is a positive
justification for the incumbent objective rather than merely a failed alternative.

### 6.4 The trunk is not the bottleneck in any dimension

| change | result |
|---|---|
| training budget ×3 (E1) | null |
| receptive field, dilations 1/2/4/8 (E7) | worse |
| width, `hidden_dim` 64 → 128 (r4) | null, at 2× the wall clock |
| depth, `num_layers` 4 → 2 (r5) | null centrally |
| head depth 1 → 3 (r3) | rmse10 worse |

Doubling the trunk changes nothing and halving its depth costs nothing measurable centrally.
Together with round 1 this says the trunk is **over-specified for what these inputs support**,
and that the central field is at an information ceiling rather than a capacity or
optimisation one. The Africa comparison (§5.4) points the same way from the other side: the
central error is 27% larger there, so the ceiling is a property of the data available, not of
the model.

### 6.5 A bug worth recording

`r7_softhist` produced rasters with **zero valid pixels** after 35 minutes. `compute_histogram`
masks by indexing, so invalid pixels never reach the arithmetic; the soft version is dense and
applied the mask *after* the sigmoid, and NaN × 0 is still NaN. It announced itself as `0 px`
in every window-horizon row — a number that cannot be true — rather than by inspection, which
is now the fifth time in this project that has been the detection mechanism. Fixed, pinned by
a test, and re-queued.

## 7. Round 3 — replication, and what survives it

Round 2's two apparent width-head wins were each run again at a second seed. **This is the
step that decides everything**, given §5.5.

| group (n) | mean\|ln k\| | within 20% | growth mid (10–30 px) | growth far (30–100 px) |
|---|---|---|---|---|
| base (3) | .681–.769 | .179–.236 | .252–.414 | .077–.354 |
| power (2) | **.787–.810** ✗ | .179–.184 | **.899–1.230** ✓ | .337–.812 |
| joint (2) | .653–.702 | .190–.255 | .422–.533 | .128–.148 |
| power_plus (2) | .752–.876 | **.139–.148** ✗ | **.759–1.097** ✓ | **.521–.600** ✓ |

Growth ratio is `k_up(20)/k_up(5)`; 1.0 means the interval grows with lead time exactly as the
residual's spread does.

**What replicates.** Both power modes fix the lead-time growth in the mid field, with ranges
that do not overlap the base at all — and `power_plus` also fixes the far field without
overlap. That is the only replicated, non-overlapping improvement in the whole phase, and it
is on the quantity §3.1 identified as most wrong.

**What does not.** `joint`'s round-2 numbers (`mean|ln k|` .653, `within 20%` .255) were a
favourable draw: the replicate lands at .702 and .190, both inside the base range. The
stricter bar of §6.1 had already called it null; replication confirmed the bar rather than the
first result.

**What it costs.** `power` is worse on `mean|ln k|` with no overlap; `power_plus` is worse on
`within 20%` with no overlap. Neither restructuring is free.

**And there is a reason to expect the cost to matter more than the gain.**
`fit_width_factors.py` fits its factors per (horizon × band × Δ̂ × HM) — *per horizon* — so a
wrong lead-time growth profile is precisely the error the post-hoc layer already corrects,
while `within 20%` measures the correction still outstanding. On the metric that predicts
downstream benefit, `power_plus` is worse than base. The downstream run exists to settle that
conflict rather than to confirm a guess.

### 7.1 The histogram loss should stay off, now for a measured reason

Fixed and re-run, the differentiable histogram is **decisively worse**: RMSE at h=20 is
0.02368 against a base range of 0.02178–0.02225, outside by 3.1× the spread (+6.4%), with
h=15 worse too. Its rarity weights upweight the rare large-change bins by roughly 240× over
the no-change bin, so a live version pushes the model to reproduce the marginal *distribution*
of change at the expense of the conditional mean — which is what RMSE measures.

So the term was inert by accident and should be off by choice. Two independent reasons now
support the same setting.

### 7.2 Round-3 nulls

`joint` with 3-layer heads: `mean|ln k|` .703, `within 20%` .210, both inside the base range.
Extra head capacity does not rescue the joint parameterisation.

## 8. The downstream verdict, and the phase conclusion

`power_plus` promoted to k=5 and pushed through the full chain, against E5 and the shipped
configuration scored identically.

| | baseline | E5 | E5 + power_plus |
|---|---|---|---|
| **scorecard** | 101/127 | **104/128** | **97/128** |
| T1 / T2 / T8 | 29 / 31 / 11 | 29 / 32 / **12** | 27 / 29 / 9 |
| per-member cells inside | 15/20 | 15/20 | **16/20** |
| mean \|rank − M/2\| | 115.6 | **103.3** | 106.6 |
| `mean\|ln k\|` (k=5) | .732 | **.686** | .823 |
| within 20% (k=5) | .186 | **.193** | .152 |
| growth ratio, mid | .391 | .565 | **.875** |

**The prediction registered in §7 is confirmed.** `power_plus` fixes the lead-time growth
profile and loses four scorecard rows for it, because the growth error is what the per-horizon
post-hoc factors already correct while the per-class calibration cost is what they cannot.
`within 20%` and `mean|ln k|` predicted a 105-minute downstream result in 40 seconds, which is
the instrument earning its place.

### 8.1 Conclusion — nothing is adopted

**The shipped configuration stands at 101/127.** Across 22 experiments in three rounds:

- every *optimisation* lever is dead — budget, schedule, loss weights, horizon balance, loss
  composition, weight averaging;
- every *capacity* lever is dead — trunk width, trunk depth, head depth, receptive field;
- the two *objective* alternatives are worse, and one of them (Winkler) was provably the same
  objective;
- the one defect that was genuinely fixed and replicated — the lead-time growth profile — is
  one the post-hoc layer already handles, and fixing it in the model costs more than it buys;
- the one candidate that improved the scorecard (E5, 104/128) is inside its own
  configuration's run-to-run spread, does not reproduce Africa-wide, and fails a hard gate.

### 8.2 What this phase actually established

1. **The central field is at an information ceiling, not a capacity or optimisation one.**
   Twelve configurations varied h=5 RMSE by under 2%, tripling the budget did nothing, and
   doubling or halving the trunk did nothing. The error is 27% larger on Africa, so the
   ceiling belongs to the inputs.
2. **The width defect that motivated the phase is largely a southern-Africa artifact.** Both
   configurations are about twice as well calibrated Africa-wide (`mean|ln k|` 0.39 vs 0.73).
   The regional protocol is right for speed and wrong for target selection.
3. **The screening protocol used in round 1 could not resolve the effects being chased.** One
   run per configuration against a floor borrowed from a different configuration produced a
   win that replication removed. Rounds 2 and 3 fixed this and it changed the answers.
4. **Two objective facts about the loss**: the histogram term has never trained anything and
   is harmful when made live; and pinball's bounded influence is what makes it right for a
   residual with kurtosis ~10³, where Gaussian NLL inflates the widths by 7×.

### 8.3 What to do next, in order

1. **Rescore the E5 k=5 ensemble at M ≥ 800.** §5.3 shows E5's residual is spikier at every
   horizon, and its two lost rows are both percentile-estimated. Cheap, and it decides
   whether 104/128 was an understatement.
2. **Replicate E5 at k=5** (~105 min per replicate). It is the only candidate with a positive
   downstream reading and the only thing blocking it is that one run cannot establish it.
3. **Stop optimising against southern-Africa width statistics.** If the width heads are
   revisited, judge them Africa-wide, where the defect is smaller and the central error larger.
4. **If the central field is to improve, it needs new information, not new architecture.**
   Every internal lever is now measured and dead. That is a covariate question, which this
   phase was scoped out of.

## 9. Rescored at M=800 — the finite-M penalty was real, and it was hiding a larger gap

§5.3 predicted from a measured mechanism that E5's two lost rows were finite-M artifacts:
its fitted marginal is spikier at the median at every horizon (`S'(0)` down 24–43%), which
makes the sample median a noisier estimator in value units, and h=15 was the horizon already
closest to the gate. Both configurations were regenerated at M=800 from their own existing
recal rasters, spectrum, width factors and marginal shape — nothing retrained — and rescored.

**The prediction holds, and the control confirms the mechanism.** T5.1, gain from M=400 to
M=800:

| horizon | baseline | E5 |
|---|---|---|
| 2005 | 0.9982 → 0.9985 (+0.0003) | 0.9960 → 0.9982 (+0.0022) |
| 2010 | 0.9985 → 0.9989 (+0.0004) | 0.9991 → 0.9994 (+0.0003) |
| **2015** | 0.9958 → 0.9984 (+0.0026) | **0.9888 → 0.9957 (+0.0069)** |

The baseline, with the flatter marginal, barely moves; E5 gains two to three times as much and
**clears the hard gate**. T2.5's rank histogram at 2015 also recovers (p 0.0075 → 0.660). E5's
T5 goes 11/12 → 12/12.

**And the matched comparison moves the other way from what M=400 suggested:**

| | M=400 | M=800 |
|---|---|---|
| baseline | 101/127 | **99/127** |
| E5 | 104/128 | **105/128** |
| E5 lead | +3 rows | **+6 rows** |

By family at M=800 the lead is T2 +3, T3 +1, T6 +1, T8 +1.

**The baseline gets worse with more members while E5 gets better.** That is exactly the
mechanism `docs/next_phase_marginals.md` recorded when an M=100 comparison between marginal
families reversed at M=400: MC-scaled tolerances tighten as 1/√M, so a higher member count
exposes error that a looser tolerance was hiding. **The M=400 comparison was understating E5,
as §5.3 predicted.**

What this does **not** settle is replication — both k=5 numbers remain single runs, and §5.5's
objection stands until the seed-43 replicate lands.

## 10. The k=5 replicate — E5 is not an improvement, and the downstream itself is noisy

§5.5 said E5 needed k=5 replicates before it could be believed. Seed 43 was trained through
the identical chain.

| | baseline | E5 seed 42 | E5 seed 43 |
|---|---|---|---|
| **scorecard** | 101/127 | 104/128 | **97/128** |
| per-member cells inside | 15/20 | 15/20 | 14/20 |
| mean \|rank − M/2\| | 115.6 | 103.3 | 124.5 |
| T2 / T8 | 31 / 11 | 32 / 12 | 30 / 9 |

**On every downstream metric the two E5 seeds bracket the baseline.** The effect does not
replicate. E5 is not an improvement.

### 10.1 The measurement that matters more than the verdict

**Two runs of one configuration differ by 7 scorecard rows out of 128 at k=5.** That is the
project's decisive test, on all five folds and 4.36 million pixels, and its run-to-run spread
**exceeds every effect this phase measured** (+3 rows for E5 at M=400, +6 at M=800, −4 for
power_plus).

The stratified metrics behaved better than the scorecard: the *central* improvement did
replicate (the baseline is worse than both E5 seeds at h=5, h=10 and h=15) and `within 20%`
did, while `mean|ln k|` and every downstream metric did not. So the cheap instruments were not
the weak link — the scorecard is noisier than the thing it is being asked to resolve.

This applies retrospectively. Every scorecard comparison in this project has been one run per
configuration. The large historical moves — 73/137 → 91/128 for the central-field phase,
90/126 → 101/127 for the marginal phase — are +18 and +11 rows and comfortably clear a
7-row band. **Anything within ±7 rows is not resolvable without replicates**, which
includes the whole of this phase.

### 10.2 What the M=800 result still supports, and what it does not

**Stands:** the hard gate was a finite-M artifact. It has a mechanism measured in advance
(E5's `S'(0)` down 24–43%), a control (the baseline gains 3–8× less from the same change),
and a replication of the phenomenon (seed 42 fails T5.1 at 2015, seed 43 at 2020 — a
different horizon, which is what a near-gate marginal does and what a lead-time-specific
defect does not).

**Does not stand:** the "+6 rows at M=800" reading. That compared one E5 seed against one
baseline run, and 6 is inside the 7-row spread — using the favourable seed.

### 10.3 Final verdict

**Nothing from the model phase is adopted. The shipped configuration stands at 101/127.**
E5 was the last candidate and replication removed it.

## 4. Results

Screening: folds 1 and 2, scored on identical pixels (1,542,872 at h=5). The phase reference
is `--histogram_weight 0 --checkpoint_monitor val_central_loss` on top of the shipped
configuration; two seeds of it bracket every comparison. Judgement is on RMSE (floor 0.6% /
0.7% / 2.2% / 4.2% by horizon) and on `mean|ln k|` (floor 0.028), never on skill.

| run | RMSE h=5 | h=10 | h=15 | h=20 | mean\|ln k\| | within 20% | verdict |
|---|---|---|---|---|---|---|---|
| incumbent, 3 seeds | .00935–.00941 | .01437–.01447 | .01849–.01890 | .02168–.02259 | .762–.790 | .124–.157 | — |
| **phase ref, 2 seeds** | **.00931–.00933** | **.01424–.01437** | **.01828–.01831** | .02184–.02185 | **.693–.746** | **.181–.196** | adopted |
| E3b MSE only | .00939 | .01489 | .01853 | .02232 | .730 | .151 | **worse** |
| E2 horizon weights | .00934 | .01449 | .01883 | .02228 | .690 | .173 | **no gain** |
| E1 budget ×3 (450 ep) | .00931 | .01426 | .01842 | .02184 | .761 | .183 | **null** |
| E3a loss on change | .00946 | .01478 | .01871 | .02231 | .676 | .188 | **worse** |
| E4 cosine + clip | .00931 | .01434 | .01852 | .02188 | .764 | .157 | **null** |
| **E5 Δ̂ to quantile heads** | .00930 | .01436 | .01845 | .02221 | **.681** | **.236** | **kept** |
| E6b scale-norm pinball | .00942 | .01435 | .01844 | .02161 | .743 | .144 | **worse** |
| E6a multiplicative width | .00935 | .01454 | .01885 | .02188 | .722 | .185 | marginal |
| E7 dilated trunk (1,2,4,8) | .00940 | .01463 | .01886 | .02217 | .760 | .175 | **worse** |
| E8 weight avg (last 20) | .00934 | .01436 | .01838 | .02173 | .782 | .152 | null / worse |
| E8b weight avg + cosine | .00931 | .01424 | .01827 | .02169 | .740 | .128 | null / worse |
| E9 = E5 + E6a | .00932 | .01441 | .01912 | .02220 | .877 | .153 | **worse than either** |
| E10 = E5 + weight avg | .00930 | .01421 | .01824 | .02172 | .741 | .189 | worse than E5 |

**Phase reference — adopted.** Better than all three incumbent seeds on five of six metrics
with disjoint ranges. At two seeds against three the honest claim is "no worse, and it
removes a confound", not "proven better": the histogram term carries no gradient, so nothing
about the optimisation changed, only which epoch is selected.

**E3b (MSE only) — the control fired.** Dropping SSIM and the Laplacian pyramid costs 4.4%
of h=10 RMSE, six times its noise floor, and raises the number of leaf classes whose residual
sits entirely on one side from 0–1 to 8. Those terms earn their weight even computed on
absolute HM, which is what makes E3a (moving them onto the change field) worth testing rather
than assuming.

**E5 (feed Δ̂ to the quantile heads) — the first positive, on its own mechanism.** The
fraction of leaf classes whose width is within 20% of correct goes .181–.196 → **.236**,
clearing its 0.033 floor; `mean|ln k|` beats both reference seeds but stays inside its floor,
so it is not decisive alone. What makes the result readable is the axis it was built for:
`k_up` spread across the predicted-change axis inside a fixed (horizon × band) cell falls
from a median max/min of 1.408 to 1.226, worst case 3.308 → 2.751. The central field is
unchanged within the floor, as a quantile-only change should be.

**E6b (scale-normalised pinball) — the registered prediction was wrong, and it failed in the
opposite direction.** §3.1 predicted this was *the* experiment that should move the far field.
It moved it the other way:

| | ref s42 | ref s43 | E6b |
|---|---|---|---|
| 30–100 px, `k_up` at h=5 | 1.834 | 2.534 | **3.824** |
| 30–100 px, `k_up(20)/k_up(5)` | 0.308 | 0.209 | **0.164** |
| 10–30 px, same ratio | 0.594 | 0.538 | **0.220** |
| within 20% | .196 | .181 | **.144** |

The gradient argument behind the prediction may still describe the optimisation correctly —
`d(step)/d(raw) = sigmoid(raw)` really is ~18× smaller at the far field's width — but the fix
does not follow from it. Dividing the pinball by the pixel's own detached half-width is not a
learning-rate change: the weighted mean is renormalised, and the half-width varies *within*
each class, so the objective stops being "the 97.5 percentile of the residual" and becomes
"the 97.5 percentile of the residual-to-width ratio". Those are different estimands with
different optima, and the second is not the published quantity.

Recorded as a loss rather than reinterpreted, because the prediction was written down before
the run.

**E6a (multiplicative width head) — marginal, and it confirms the other half of the
prediction.** §3.1 said E6a should *not* move the far field, and it does not: the 30–100 px
lead-time ratio goes 0.308 / 0.209 → 0.133, if anything slightly worse. Its one real gain is
band-level width calibration, `mean|ln k|` over (horizon × band) cells 0.604–0.747 → **0.527**,
past floor; the leaf-level metric (0.722) and `within 20%` (0.185) both stay inside the
reference band. Not a clear keep on its own.

**E7 (dilated trunk, per-layer dilations 1/2/4/8) — worse, and the stratification says why.**
RMSE regresses past floor at h=5, h=10 and h=15. By distance band at h=15:

| band | ref s42 | ref s43 | E7 |
|---|---|---|---|
| 0–1 px | .03129 | .03119 | **.03278** (+4.8%) |
| 1–3 px | .02146 | .02146 | .02183 |
| 3–10 px | .01554 | .01554 | .01577 |
| 10–30 px | .01034 | .01033 | .01041 |
| 30–100 px | .00417 | .00418 | .00418 |

**The far bands do not move at all** — the thing the wider receptive field was for — **and the
near field gets worse**, which is where the error lives. The explanation is this project's own
history: `scripts/prepare_change_context.py` already supplies the long-range information as a
full-raster covariate at five radii, and both head families already read it. Once that is in
the heads there is nothing left for a wider trunk to discover, while dilated kernels sample a
sparse grid and cost local resolution exactly where change happens.

**Both levers `docs/next_phase_model.md` §4 named as never tried are now tried and rejected**
— training budget (null, E1) and trunk receptive field (worse, E7). Per this phase's own
rejection rule the downsampled-branch variant was not run: the hypothesis under test was
"receptive field is the binding limit", and it is not.

**E8 / E8b (weight averaging) — null centrally, worse on width, in both variants.** The
plateau of §2.1 motivated averaging the tail epochs rather than picking one of them. Centrally
it does nothing: E8b matches or beats both reference seeds at all four horizons at once, which
is mildly suggestive, but every margin is inside its floor. On the quantile side both variants
regress past floor — `within 20%` .152 and **.128** against .181–.196, the two worst scores of
the twelve runs. Averaging in parameter space is not averaging the widths, which are a
softplus of an accumulating sum, but that is a hypothesis and it rests on two runs.

**E9 and E10 (combinations) — both lose to E5 alone, and E9 loses to both its parts.**
E5 + multiplicative width gives `mean|ln k|` .877, the worst of the phase, against .681 for E5
and .722 for E6a. E5 + weight averaging gives .741 / .189 against E5's .681 / .236. E5's gain
is the head modulating its width with Δ̂, and it does not survive either changing how that
modulation is expressed or averaging it over epochs.

### 4.-1 Summary of the twelve screening runs

Ordered by the phase's primary quantile metric. "Floor" columns mark whether the change
clears the run-to-run band of §2 (RMSE 0.6 / 0.7 / 2.2 / 4.2% by horizon; `mean|ln k|` 0.028;
`within 20%` 0.033).

| # | what changed | central field | quantile widths | verdict |
|---|---|---|---|---|
| — | **phase ref** (histogram off, central-only monitor) | better than all 3 incumbent seeds, h=5/10/15 | `ln k` .693–.746, w20 .181–.196 | **adopted** |
| E5 | **Δ̂ fed to the quantile heads** | unchanged within floor | **`ln k` .681, w20 .236**; Δ̂-axis spread 1.41 → 1.23 | **kept** |
| E6a | multiplicative width head | h=10/15 worse past floor | band `ln k` .527 (best), leaf .722, w20 .185 | marginal |
| E2 | horizon loss weights 1/1.33/2/4 | h=10, h=15 worse past floor | `ln k` .690, w20 .173 | no gain |
| E1 | training budget ×3 (450 ep) | identical within floor at all four | `ln k` .761, w20 .183 | **null** |
| E4 | cosine LR + gradient clip | inside floor at all four | `ln k` .764, w20 .157 | **null** |
| E8 | weight averaging, last 20 | inside floor | `ln k` .782, **w20 .152** | null / worse |
| E8b | weight averaging + cosine | best of the phase, all inside floor | `ln k` .740, **w20 .128** | null / worse |
| E10 | E5 + weight averaging | inside floor | `ln k` .741, w20 .189 | worse than E5 |
| E6b | scale-normalised pinball | h=5 worse | `ln k` .743, **w20 .144**; far field 1.83 → 3.82 | **worse** |
| E7 | dilated trunk 1/2/4/8 | h=5/10/15 worse; 0–1 px +4.8% | `ln k` .760, w20 .175 | **worse** |
| E3a | SSIM/Laplacian on the change field | h=5/10 worse; damage all in 0–1 px | `ln k` .676, w20 .188 | **worse** |
| E3b | MSE only | **h=10 worse by 4.4%**, 6× floor | `ln k` .730, w20 .151 | **worse** |
| E9 | E5 + multiplicative width | h=15 worse | **`ln k` .877 — worst of the phase** | **worse than either part** |

### 4.0 The far field is not a width-head problem

Both width experiments leave the far-field lead-time defect exactly where it was, one having
been predicted to fix it and one predicted not to. Neither the head's parameterisation nor
the loss's scaling reaches it.

That is worth stating positively. At h=5 in the 30–100 px band the residual is a near-zero
body with rare large values, so a 95% interval fitted to it is **correctly** tiny — and the
+0.05 threshold `docs/next_phase_marginals.md` §5.5 measures at a median of 119 half-widths
out is a statement about a two-bound product, not about a mis-trained head. The model is not
obviously wrong there; the deliverable cannot express what it knows. The lever is a
model-predicted tail quantity feeding the marginal's bound per pixel, not more width work.

### 4.1 A correctness check that could not be run, and the reason

E5 was designed to carry its own check: a quantile-only change, at the same seed, under a
central-only checkpoint monitor, should leave the central rasters **identical**. The trunk
and central heads are constructed before the quantile heads so their initialisation cannot
shift; the pinball gradients that reach the trunk are overwritten from the saved central
gradients; Adam is per-parameter; and `val_central_loss` does not depend on the quantile
heads. It failed — 100% of pixels differed, by up to 0.059.

The premise was wrong, not the design. **Training here is not deterministic at a fixed
seed.** Two runs with identical flags and `--seed 42`, three epochs each, differ in 54 of 62
tensors with a worst weight difference of 2.1e-4 — cuDNN algorithm selection and
non-deterministic atomics. Over 1,950 optimizer steps that compounds into the observed
difference.

Two things follow. The invariance check has to be **distributional** — a quantile-only change
must leave the central metrics inside the run-to-run band, which E5's do — not bit-exact.
And the "noise floor" of §2 is not seed variance but **run-to-run variance**, which is the
right comparator anyway and makes the measured floor if anything conservative.

**E1 (training budget) — null, as predicted in §2.1.** Three times the budget, 450 epochs
against 150, at 52 min against 18: **every metric lands inside the two-seed reference band**,
and h=5 and h=20 RMSE are identical to the reference's to five decimals. The selected epochs
were 248 and 175 of 450 — mid-run plateau selection again.

This is the first of the two levers `docs/next_phase_model.md` §4 named as never tried, and
it is now tried and rejected on evidence. `current_progress.md` item 1 ("the central field is
under-trained… longer training is the cheapest untested lever") should be revised: the model
saturates within a handful of epochs on 15.6k chip presentations, and giving it three times
as many changes nothing measurable. Whatever limits this model, it is not optimisation time.

**E3a (SSIM and Laplacian on the change field) — worse, and the stratification localises it.**
h=5 and h=10 RMSE regress by three to four times their floor. Every distance band is
identical to the reference to five decimals **except 0–1 px**, which degrades 6.8% at h=10
and 5.0% at h=20 — so the damage is entirely in the high-change near field.

The mechanism is SSIM's local normalisation. Against absolute HM the target carries real
spatial structure and the local statistics are well posed. Against the change field the
target is near-zero almost everywhere with rare spikes, so most windows have σ ≈ 0, the
score collapses onto its stabilising constants, and the gradient pushes toward smoothness
exactly where change concentrates.

**Taken with E3b, the loss composition is at a local optimum.** Both directions out of it —
removing the two terms, and moving them onto the change field — cost h=10 RMSE. The weights
were tuned for a central head that predicted absolute HM, and they survive the change to a
residual head.

**E2 (horizon loss weights) — negative, with a reason.** Compensating the measured 4:3:2:1
exposure with weights 1 / 1.33 / 2 / 4 made h=10 and h=15 worse (both just past floor) and
h=20 no better, and pushed the far-field width defect further out (h=5, 30–100 px: k_up 1.83
→ 2.10). The imbalance is not a deficiency to correct: h=20 is intrinsically harder, all four
heads share one trunk, and upweighting the hardest horizon simply trades the near ones away.

*(filled in as the slate completes)*
