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
