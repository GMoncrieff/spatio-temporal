# Model architecture and training

The ConvLSTM that produces this branch's central forecast and 2.5/97.5 intervals at
+5/+10/+15/+20 years. Adapted from the `ensemble` branch's methodology document, restricted to
the model itself — that branch's calibration and 400-member ensemble layers are not here, so
the products on this branch are the heads' own output, uncalibrated.

---

## 1. What it consumes and emits

Input is three timesteps at 5-year spacing over a 128 px chip:

- **11 dynamic channels** — HM itself (`AA`) plus its ten component layers
  (`AG, BU, EX, FR, HI, NS, PO, TI, gdp, population`), z-scored against one pooled set of
  statistics held in `data/norm_stats.json`.
- **7 static channels** — elevation, temperature (mean and min), precipitation, dpi/dsi, and
  IUCN strict / non-strict protection — repeated across the time axis.
- **8 location-encoder channels**, optional, from spherical harmonics through a SIREN.

The trunk is a 4-layer ConvLSTM, `hidden_dim 64`, 3×3 kernels. Its last hidden state feeds
twelve heads: a central head and a lower/upper pair per horizon. Output is `[B, 12, H, W]`
ordered `[lower, central, upper]` per horizon, in normalised space; inference multiplies back
through `hm_std`/`hm_mean` and clips to [0, 1].

`hm_mean` and `hm_std` are plain attributes and are **not in the `.ckpt`**. A checkpoint alone
is not enough to denormalise — `--norm_stats_json` is required at prediction time as well as
training time.

## 2. The central head predicts change, not level

The central head is a **residual** head: it predicts the *change* in HM from t₀ and adds it to
HM at t₀ through a skip connection, with a zero-initialised output convolution.

This matters because the signal is small and sparse. Across the globe, 53–73% of pixels change
by less than 0.001 over a five-to-twenty-year horizon — for those, the correct answer is "emit
t₀ unchanged". A head predicting absolute HM must first reproduce t₀ through the trunk before
it can add anything, and that reproduction is not free: it leaks noise on exactly the pixels
where the truth is "nothing happens". With the skip connection the model starts at exact
persistence and saying "nothing happens" costs nothing.

The right yardstick for the result is **skill against persistence** — against predicting that
nothing changes. That bar is higher than it sounds, because the median 20-year HM change is
0.0001, so pooled RMSE can look unremarkable while the model loses to doing nothing. Measured
globally on held-out geography:

| horizon | skill vs persistence |
|---|---|
| +5 yr | **+0.110** |
| +10 yr | **+0.183** |
| +15 yr | **+0.226** |
| +20 yr | **+0.222** |

Positive at every horizon and rising with lead time, which is the sensible ordering: the
further out you forecast, the more there is to beat "nothing happens" at.

## 3. The long-range context covariate

The trunk's receptive radius is roughly 10 px, set by its 3×3 kernels over 4 layers. It
therefore cannot see whether change happened 30–100 px away — which is the single best
predictor of whether change is possible at a pixel at all.

`scripts/prepare_change_context.py` precomputes two bands per input window, on the **full
raster**:

1. `past_change` — HM at the window's base year minus HM ten years earlier;
2. `dist_past_change` — Euclidean distance, in pixels, to the nearest pixel whose
   `past_change` exceeded 0.01.

Both head families receive 8 channels derived from these:

| channels | content |
|---|---|
| 5 | occupancy — is there past change within 1 / 3 / 10 / 30 / 100 px? (simply `dist ≤ r`) |
| 1 | `log1p(dist) / 10` |
| 1 | the signed past change itself |
| 1 | HM at t₀ |

Because occupancy is read off a precomputed distance, no pooling is involved and the chip
boundary plays no part. This is why the distance is computed globally rather than inside the
chip: derived from a 128 px chip, the 100 px radius saturates into "is there any change
anywhere in this chip", which is an artifact of framing rather than a fact about geography.

## 4. The interval is built around the central forecast, and cannot narrow with lead time

The quantile heads do not predict bounds directly. Each emits a non-negative **half-width
increment** through a softplus, and those increments **accumulate across horizons** around the
central forecast, which is detached before being used as the anchor.

Two consequences, both structural rather than enforced afterwards:

- **Spread cannot shrink with lead time.** A cumulative sum of non-negative increments is
  non-decreasing by construction.
- **`lower ≤ central ≤ upper` always holds**, rather than being clipped into place later.

**Both guarantees are properties of the model's output, and the second survives everything
downstream. The first does not quite survive the write.** Predictions are clipped to [0, 1]
before they are written, because HM is bounded there. Interval width is `w_lower + w_upper` —
the central term cancels — so it is non-decreasing in model space by construction, and
averaging (overlap blending, fold-mean stitching) is linear and preserves that. The clip is
not linear. Where the upper bound has already saturated at 1.0 and the central estimate keeps
rising, the interval can narrow.

Measured on the shipped products over a 2.07 Mpx land window: **3,149 pixels (0.15%) narrow at
some horizon, worst magnitude 0.02, and every one of them touches the clip** — 100% have
`upper ≥ 0.999` at +20 yr, against 0.4% of all valid pixels. Among the 2,000,387 pixels that
never touch a clip at any horizon, narrowing occurs **zero** times. So this is the bound of the
HM range asserting itself, not a defect in the heads; an interval whose top is already at 1
cannot widen upward.

The quantile loss is pinball at 0.025 and 0.975, and it is **gradient-isolated from the
trunk** — the central forecast enters the quantile heads detached, so the pinball objective
shapes the interval without moving the central field.

## 5. Loss

The central objective is computed on absolute HM:

```
MSE  +  0.2 · SSIM  +  0.3 · Laplacian-pyramid  +  1.0 · histogram
```

and the quantile objective is pinball at the two tail levels, isolated as described above.

## 6. How out-of-sample predictions are produced

HM exists only from 1990 to 2020, and a model must not be scored on data it trained on. The
system therefore uses **five-fold cross-validation over geography.**

The globe is partitioned into five spatial folds on a mask of contiguous **512 px blocks**
(`fold_mask_b4_1000.tif`). Five models are trained; each excludes one fold from training
entirely and uses another for validation. Each model then predicts only the fold it never saw,
and the five predictions are stitched into one raster where every pixel comes from the model
that held it out.

Block size is a deliberate choice. The residual field's fitted practical range — the distance
over which model errors remain correlated — is about 300 px globally. Fold territories must be
larger than that, or every held-out pixel sits inside the correlation length of pixels its own
model trained on, and held-out skill is optimistic by an unknown amount. At 512 px blocks a
substantial fraction of held-out pixels lie beyond one correlation length of any pixel their
fold trained on, and each fold still holds hundreds of blocks to average over.

Each model predicts four input windows (ending 2000, 2005, 2010, 2015) at four horizons, giving
ten (window × horizon) pairs that fall inside the observed record. **Everything downstream —
the calibration, the marginal, the spatial field, the horizon coupling — is estimated from the
residuals of these predictions**, and every one of those residuals is genuinely out of sample.

Two stitching modes exist and they serve different purposes:

- **`holdout`** — each pixel takes the value from the fold that held it out. Out of sample
  everywhere. **This is the only mode anything scored may use.** It is a hard mosaic: adjacent
  fold territories come from different models, so wherever those models disagree the join is
  visible, and they disagree mainly in the upper bound.
- **`mean`** — every fold averaged at every pixel. Seamless, and in-sample at every pixel by
  construction, since four of the five folds trained on any given location. This is a display
  product only.

## 7. The forward model

A fold model cannot forecast forward: each deliberately never saw a fifth of the world, which
is what makes its hindcast honest and what makes it unusable for production. The forward
product therefore comes from a **sixth model, trained on every chip with no fold excluded**,
in the identical configuration.

It reads HM at 2010 / 2015 / 2020 and predicts 2025 / 2030 / 2035 / 2040. The horizon structure
is identical to the hindcast, so anything estimated from the hindcast residuals transfers directly to its outputs.

Because it trains on everything, it has no held-out data of its own. What is validated is the
*configuration*, by the five-fold hindcast; the production draw is then trusted.


---


## 8. What the flags mean

The shipped architecture is four flags, and **argparse defaults are not it**:

```
--central_residual True --central_context True --monotone_quantile_width True --quantile_context True
```

| flag | default | off | on |
|---|---|---|---|
| `--central_residual` | False | central head predicts absolute HM | predicts Δ, added to HM at t₀ through a zero-initialised skip |
| `--central_context` | False | central head in-channels 64 | 72 — the 8 change-context channels |
| `--quantile_context` | False | quantile head in-channels 64 | 72 |
| `--monotone_quantile_width` | False | heads predict bounds directly; isolation is a manual gradient save/restore | heads emit cumulative softplus half-widths around a detached centre; isolation is structural |

A k=5 run launched on defaults scored h=5 skill −0.50 against +0.13 with these flags, and read
exactly like a fold-mask finding. The fingerprint that separates them is one startup line:
with `--central_residual` the diagnostic prints `ConvLSTM grad norm: 0.000000` — the trunk gets
no gradient from the central loss — and on defaults it prints ~0.04 with an initial central
loss 20× higher.

The flags are required even under `--max_epochs 0`, because the model is built from the CLI args
before the state dict is loaded into it.

## 9. Flags that exist and are not used

The `ensemble` branch ran 22 model experiments and adopted none of them. Their flags remain,
all defaulting to off, and turning one on is re-opening a settled question rather than picking
up an improvement:

`--horizon_loss_weights`, `--loss_on_change`, `--pinball_scale_norm`, `--quantile_dhat_context`,
`--width_parameterisation`, `--convlstm_dilations`, `--head_hidden_layers`, `--width_head_mode`,
`--central_target_transform`, `--quantile_loss`, `--histogram_soft`, `--lr_schedule`,
`--grad_clip`, `--weight_avg_last`, `--checkpoint_select`, `--quantile_class_weighting`,
`--freeze_trunk`.

Two caveats worth carrying:

- **`--checkpoint_monitor` defaults to `val_total_loss`, which includes pinball.** A
  quantile-only change therefore selects a different epoch and so a different *central* field.
  A central-only A/B needs `--checkpoint_monitor val_central_loss`. The shipped models were
  trained on the default.
- **The k=5 scorecard's own run-to-run spread was 7 rows out of 128**, and training is not
  deterministic at a fixed seed on this box (cuDNN algorithm selection). Any comparison within
  ±7 rows needs replicates before it means anything, and invariance checks must be
  distributional, never bit-exact.

## 10. Known defect

`P(Δ > 0.05)` beyond 100 px from past change reads about 0.034 of observed. This is a property
of the quantile heads: far-band half-widths are ~0.004, so +0.05 is roughly 12 half-widths out,
while a 0.999 marginal tail reaches 3–4. No post-hoc layer reaches it; fixing it means changing
the width heads.

It is also the reason far-field work must not be screened on southern Africa, whose observed
far-field change rate is 0.0000 — a metric that reads 0 on the screening region cannot rank
anything. Screen on Africa (63.1 Mpx) or globally.
