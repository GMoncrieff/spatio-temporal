# What changed from `main`

`main` sits at `204553f`, which is exactly where this branch started, so everything below is
the difference between the model `main` fits and the one that produced the products in
`data/global/`.

The short version: the trunk is unchanged, and almost all of the gain comes from **not asking
the trunk to do work it does not need to do**. The central head no longer reconstructs the
baseline before it can predict a change; the quantile heads no longer have to discover an
ordering that can be built into the parameterisation; and the model is no longer blind beyond
its own ~10 px receptive radius. The training recipe changed at least as much as the model:
`main` has no cross-validation, and its validation set is not disjoint from training.

| | `main` | now |
|---|---|---|
| central head | predicts absolute HM | predicts Δ from t₀ through a zero-initialised skip |
| quantile heads | predict the absolute bound directly | cumulative softplus half-widths around a detached centre |
| ordering `lower ≤ central ≤ upper` | emergent; nothing enforces it | structural |
| spread vs lead time | unconstrained | non-decreasing by construction |
| gradient isolation | manual save/restore of `.grad` between two backward passes | structural (the centre is detached) |
| head input | 64 channels (trunk hidden state) | 72 (hidden state + 8 change-context) |
| long-range context | none — trunk radius is ~10 px | precomputed distance-to-past-change, radii to 100 px |
| trunk depth (as run) | 2 layers | 4 layers |
| held-out geography | one random 70/10/10/10 chip split | 5-fold CV on contiguous 512 px territories |
| validation set | **not disjoint from training** (see §2.2) | disjoint fold, and the chip jitter is disabled |
| normalisation | re-estimated from random windows every run | one `norm_stats.json` sidecar shared by every model |
| forward model | trains on the 70% split | `--train_all_splits`: every valid chip |
| prediction windows | two hard-coded options | any `--predict_input_years`, all four hindcast windows in one load |
| fold stitching | n/a | `holdout` (out-of-sample) and `mean` (seamless, in-sample) |
| output | plain GeoTIFF, `data/predictions/` | verified COGs with overviews |
| prediction accumulators | float64 (12 × 5.5 GB at global extent) | float32 (12 × 2.7 GB), and only for horizons that will be written |

---

## 1. The model

### 1.1 The central head predicts change, not level

`main`'s central head emits absolute HM. That sounds neutral and is not: 53–73% of pixels change
by less than 0.001 over a 5–20 year horizon, so for most of the map the correct answer is "emit
t₀ unchanged" — and a head predicting the level must first reproduce t₀ *through the trunk*
before it can add anything. That reproduction is not free. It leaks noise onto exactly the
pixels where the truth is that nothing happens.

The residual head adds its output to HM at t₀ through a skip connection, with the output
convolution zero-initialised. Training therefore starts at exact persistence, and saying
"nothing happens" costs nothing.

This is the single largest change. A k=5 run on argparse defaults — which is to say with this
flag off — scored h=5 skill **−0.50** against **+0.13** with it on, and the failure read exactly
like a fold-mask problem rather than an architecture one.

The yardstick throughout is **skill against persistence**, not against zero. The median 20-year
HM change is 0.0001, so a pooled RMSE can look unremarkable while the model is losing to
"nothing will change" — at one point by 2.2× in MSE at h=5.

### 1.2 The interval is built around the centre, not predicted beside it

`main`'s lower and upper heads each emit a bound directly, fitted to the absolute target with
pinball loss at 0.025 and 0.975. Nothing makes `lower ≤ central ≤ upper` true, and nothing stops
the interval narrowing as the forecast reaches further out.

Each head now emits a non-negative **half-width increment** through a softplus, and those
increments accumulate across horizons around the central forecast, which is **detached** before
being used as the anchor. Two properties follow from the parameterisation rather than from a
clip applied afterwards:

- `lower ≤ central ≤ upper` always holds;
- spread cannot shrink with lead time, because a cumulative sum of non-negative increments is
  non-decreasing.

One caveat, measured on the shipped products: the second property is a property of the model's
output and survives everything downstream, but predictions are clipped to [0, 1] before they are
written because HM is bounded there, and the clip is not linear. Where the upper bound has
already saturated at 1.0 and the centre keeps rising, the interval can narrow — 0.15% of pixels,
worst magnitude 0.02, **every one of them touching the clip**, and zero narrowing among the two
million pixels in the sample that never touch it. See `model_architecture.md` §4.

### 1.3 Gradient isolation became structural

`main` isolates the two objectives by hand (commit `c0d262d`): backward the central loss, clone
every non-quantile `.grad`, backward the pinball loss, write the saved gradients back, then step.
It works, and it is delicate — it depends on the parameter partition staying correct as heads
are added.

Detaching the centre before the quantile heads read it removes the gradient path instead of
undoing it afterwards. The manual scheme is still in the code and still runs whenever
`--monotone_quantile_width` is off, so `docs/pinball_loss_gradient_isolation.md` is superseded
for the shipped configuration rather than wrong.

The fingerprint of a correctly configured run is one startup line:

```
ConvLSTM grad norm: 0.000000 (from central loss only)
```

The trunk gets no gradient from the central loss under `--central_residual`. On defaults it
reads ~0.04 with an initial central loss 20× higher. That line is the cheapest way to tell "the
production architecture" from "a run that started and looked fine".

### 1.4 The model can now see past its own receptive radius

The trunk's receptive radius is roughly 10 px — 3×3 kernels over 4 layers. It therefore cannot
see whether change happened 30–100 px away, which is close to the best single predictor of
whether change is possible at a pixel at all.

`scripts/prepare_change_context.py` precomputes two bands per input window on the **full
raster**: signed past change, and Euclidean distance to the nearest pixel whose past change
exceeded 0.01. Both head families receive 8 channels derived from these — five occupancy
channels (`dist ≤ 1, 3, 10, 30, 100`), `log1p(dist)/10`, the signed change, and HM at t₀.

The "on the full raster" part is not an implementation detail. Derived inside a 128 px training
chip, the 100 px radius saturates into "is there any change anywhere in this chip", which is a
fact about the framing rather than about the geography.

---

## 2. The fitting

### 2.1 Five-fold cross-validation over geography

`main` has no fold concept at all. `create_validity_mask.py` writes one random 70/10/10/10
chip partition and that is the whole story; split value 4 (calibration) is generated and never
consumed.

The globe is now partitioned into five spatial folds of contiguous **512 px blocks**
(`--fold_block_chips 4`, giving `fold_mask_b4_1000.tif`). Five models train, each excluding one
fold from training entirely and validating on another; each predicts only the fold it never saw;
`src/prediction/stitch.py` joins them into one raster where every pixel comes from the model
that held it out.

Block size is a deliberate choice, not a default. The residual field's fitted practical range —
the distance over which model errors stay correlated — is about 300 px. Fold territories have to
exceed that, or every held-out pixel sits inside the correlation length of pixels its own model
trained on and held-out skill is optimistic by an unknown amount. At `--fold_block_chips 1` the
folds are a 128 px checkerboard, which is precisely that failure.

### 2.2 `main`'s validation set is not disjoint from training

This is worth stating on its own, because it is silent. `main`'s dataset honours `split_value`
only in `random` sampling mode. In `grid` mode it enumerates chip positions across the **entire
raster** and ignores the split mask completely:

```python
if self.mode == "grid":
    self.chip_positions = []
    for t in self.valid_time_idxs:
        for i in range(0, self.H - chip_size + 1, stride):
            for j in range(0, self.W - chip_size + 1, stride):
                self.chip_positions.append((t, i, j))
```

`--val_mode` defaults to `grid`. So on `main`'s shipped defaults, validation and test iterate the
whole world, training geography included, and `--val_chips` is ignored as well because `__len__`
returns the length of that list.

Grid mode now filters on the split mask like random mode does, and the ±32 px chip jitter is
disabled whenever an exclusion set is active, since a jittered chip could otherwise reach into
held-out territory.

### 2.3 One set of normalisation statistics

`main` estimates `hm_mean`/`hm_std`, the per-variable static stats and the per-component stats
from random windows at the start of every run. Two runs therefore normalise slightly
differently, and a checkpoint alone cannot be denormalised.

`--norm_stats_json` writes the statistics once and reloads them thereafter, so every fold and
the production model share one normalisation. It is **required at prediction time too**:
`hm_mean` and `hm_std` are plain attributes and are not in the `.ckpt`, so without the sidecar
inference builds the change context at a different scale than training did.

### 2.4 The forward model uses every chip

A fold model cannot forecast forward — each deliberately never saw a fifth of the world, which
is what makes its hindcast honest and what makes it unusable for production. The forward product
comes from a sixth model in the identical configuration, reading 2010/2015/2020 and predicting
2025/2030/2035/2040.

That model has no held-out geography to protect, so restricting it to split 1 discards ~30% of
the world for nothing. `--train_all_splits True` uses every valid chip. It is ignored under
`--exclude_fold`, so it can never pull a held-out fold back into training. On `main` there was no
way to ask for this at all.

### 2.5 The hyperparameters are passed, not defaulted

**No argparse default changed between `main` and now.** That cuts both ways: the port breaks
nothing that relied on a default, and *the defaults are still not the shipped model* — on either
axis. The production runs pass all of this explicitly:

| flag | default (both branches) | as run |
|---|---|---|
| `--num_layers` | 2 | **4** |
| `--ssim_weight` | 2.0 | **0.2** |
| `--laplacian_weight` | 1.0 | **0.3** |
| `--histogram_weight` | 0.67 | **1.0** |
| `--histogram_warmup_epochs` | 20 | **0** |
| `--central_residual` | False | **True** |
| `--central_context` | False | **True** |
| `--quantile_context` | False | **True** |
| `--monotone_quantile_width` | False | **True** |

Plus `--val_stride 1024`, `--train_chips 100`, `--batch_size 8`,
`--seed 42`, `--max_epochs 150`.

`--val_stride` is worth a note of its own. It is the grid stride for validation and test, and
what it draws depends on which mask is in play: at 1024, fold 2 of the b4 mask yields 131 chips,
while production split 2 yields 58. A fold-mask change therefore moves what the validation stride
samples, which matters because the selector should see pixels that actually change. 1024 was
chosen on that argument rather than on a measured benefit — the stride's effect in the correct
configuration has never been isolated.

`--checkpoint_monitor` is new and still defaults to `val_total_loss`, which **includes pinball**.
A quantile-only change therefore selects a different epoch and so a different *central* field;
a central-only A/B needs `--checkpoint_monitor val_central_loss`. The shipped models were trained
on the default.

---

## 3. Prediction and output

- **Any input window.** `main` offers `--predict_final_year` with exactly two choices, 2020 and
  2040. `--predict_input_years` takes any three years, and `--predict_all_windows` runs all four
  hindcast windows from one checkpoint load.
- **Restricted prediction.** `--predict_restrict_mask`/`--predict_restrict_values` skip tiles
  that do not overlap the requested fold — about 5× cheaper, and exact for kept pixels, because
  every tile overlapping a kept pixel is still processed.
- **Fold stitching.** `holdout` mode takes each pixel from the fold that held it out; `mean`
  averages all five. `holdout` is the only mode anything scored may use. `mean` is seamless and
  **in-sample at every pixel** — four of five folds trained on any given location — so it is a
  display product, and its interval is narrower than any single fold's because averaging discards
  the between-fold spread.
- **COGs.** `make_cogs.py` converts and then *proves* the conversion: `LAYOUT=COG`, overviews
  present, shape/transform/CRS/nodata identical, and 12 random 1024² windows compared for
  NaN-mask and exact value equality. `main` writes plain deflate GeoTIFFs with no overviews.
- **Memory.** Prediction accumulators went float64 → float32 and are allocated only for horizons
  that will actually be written. At the global extent that is 2.7 GB per accumulator rather than
  5.5 GB, across twelve of them.

---

## 4. Two defects fixed in passing

**Wrong nodata on every prediction raster.** The writer copied the source raster's profile and
overrode only `count`/`dtype`/`compress`, inheriting `nodata=3.4e38` while filling invalid pixels
with `NaN`. The header disagreed with the data, so anything honouring it renders ocean as valid.
This is on `main` and was still on the ensemble branch; it never surfaced there because the
published forecast COGs were built from recalibrated rasters, and the recalibration pass sets
`nodata=NaN`. The writer now sets `nodata=np.nan`, and `make_cogs.py --dst_nodata` corrects
rasters written before the fix.

**Shell command pasted into `.gitignore`** by commit `204553f`. Removed.

---

## 5. Consequences for anything trained on `main`

**`main` checkpoints will not load.** The central head's input width went 64 → 72 with the
context channels, so a `main`-architecture checkpoint fails with:

```
size mismatch for model.central_heads.0.0.weight: [64, 72, 3, 3] vs [64, 64, 3, 3]
```

That is not a bug to work around — it is the check working. `artifacts/model-khrpthgy:v0` is a
`main`-architecture checkpoint kept deliberately as the discriminating control for
`scripts/check_checkpoint_fingerprint.py`; it must be **REJECTED**, and without it the check has
not been shown to reject anything.

The architecture flags are required even under `--max_epochs 0`, because the model is built from
the CLI args before the state dict is loaded into it.

**Numbers do not carry across.** Anything measured on `main` was measured on a different model,
on a validation set that included training geography, under per-run normalisation. Two further
cautions apply to comparisons made from here:

- Training is **not deterministic at a fixed seed** on this box — cuDNN algorithm selection
  varies run to run. Invariance checks must be distributional, never bit-exact.
- The k=5 scorecard's own run-to-run spread was **7 rows out of 128**. Anything inside ±7 rows
  needs replicates before it means anything.

---

## 6. What did not change

The ConvLSTM cell itself (beyond an optional per-layer `dilation` that defaults to 1 and
reproduces the old cell exactly), the location encoder, the Laplacian pyramid loss, the
histogram loss's hard path, `config/config.yaml`, and the 11 dynamic + 7 static input channels.

The `ensemble` branch ran 22 model experiments and adopted **none** of them. Their flags came
across with the port and all default to off; `model_architecture.md` §9 lists them. Turning one
on is reopening a settled question, not picking up an unused improvement.

---

## 7. Verifying the change

The products in `data/global/` were built from the frozen `g1_foldb4` checkpoints rather than a
retrain, so the code carries the burden of proof. `fitting_running_model.md` §7 lists eight
gates; the ones that matter here:

| gate | result |
|---|---|
| fold prediction vs the frozen rasters | **bit-identical** (max abs diff 0.0 across all 12) |
| forecast prediction vs the frozen rasters | agrees to 8.9e-7, and the new raster carries `nodata=NaN` |
| the extracted stitcher | bit-identical in both modes, at 184,573,321 and 184,608,551 valid px |
| skill vs persistence, re-measured | reproduces the published +0.110 / +0.183 / +0.226 / +0.222 to within 0.003 |
| test suite | 73 passing, up from 21, with the same 7 stale legacy failures |

> One trap worth recording, because the first attempt at the prediction gate failed on it.
> Prediction tiles are enumerated as `range(r0, r1, stride)` from the **region bbox origin**, so
> a comparison region whose origin is not congruent to the reference run's modulo
> `--predict_stride` blends every pixel from a different set of tile offsets. An arbitrary
> southern-Africa box reads max |diff| up to 1.5e-2 against the frozen rasters with mean |diff|
> at 1e-6 — which looks like a real defect and is entirely tile phase.
> `config/region_smoke_aligned.geojson` exists to be phase-aligned, and gives exact equality.
> `--predict_restrict_mask` is exact for kept pixels *within one tile grid*; it does not make two
> different tile grids agree.
