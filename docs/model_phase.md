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

---

## 4. Results

*(filled in as the slate completes)*
