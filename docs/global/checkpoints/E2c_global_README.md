# E2c — the production checkpoints, 2026-09-26

**THE PRODUCTION MODEL — chosen by the user 2026-09-27.** The five k=5 hindcast fold models and
the forward (`--train_all_splits`) model behind the delivered E2c products
(`/mnt/hdd1/spatio-temporal/data/conv_spline/products/E2c/`). They live in
`models/production/E2c_global/` (not in git), copied out of `models/checkpoints/` — Lightning's
shared `default_root_dir`, where a file is not evidence of which run wrote it. Each was
identified from its run's own log, by the `Prediction will use the end-of-training checkpoint`
line naming the file prediction actually ran on.

Configuration, identical for all six:

    --head_family pwl --free_scale True --mu_mse_weight 0.0 --kernel_size 5 --terrain_covariates True

plus `conv_spline_base.sh` BASE_ARGS with `--predict_row_chunk 512`, seed 42, the sidecar
`norm_stats_E2c.json` and (the five folds) `fold_mask_b4_land_1000.tif`. As built and as
verified out of all six logs: `Spline head: family pwl, knots default14 (n=15, bins=14), slopes
learned, free scale, 15 params/horizon`; `Trunk: ConvLSTM 4 layers x 64, kernel 5, ...,
receptive radius 12 px`; `Static channels: 10 (...); module 10`. All six passed all eight
fingerprints (loss weights, context wiring, head, trunk, static channels, sidecar, row banding,
weight averaging); the forward model also the production-mode banner and the absence of a
FOLD-CV banner. The first ConvLSTM layer's weight is `(256, 105, 5, 5)`.

| file | role | held out |
|---|---|---|
| final_fold1_3414639.ckpt | hindcast fold 1 | fold 1 |
| final_fold2_3414640.ckpt | hindcast fold 2 | fold 2 |
| final_fold3_3494620.ckpt | hindcast fold 3 | fold 3 |
| final_fold4_3495397.ckpt | hindcast fold 4 | fold 4 |
| final_fold5_3573153.ckpt | hindcast fold 5 | fold 5 |
| final_foldNone_3573819.ckpt | forward model, base 2020 | nothing (`--train_all_splits`) |

These are the averaged weights (mean of the last 20 epochs), which is what prediction ran on.

**The sidecar travels with them.** `norm_stats_E2c.json` in the same directory is a copy of
`data/conv_spline/norm_stats_E2c.json`: exact land-only statistics for HM, the ten covariates
and the ten static channels, and the `static_nodata` declaration that makes the reader treat
elevation's −32768 as sea level. A checkpoint without it predicts on different inputs.

**Loading one requires E2c's flags**, or the trunk and static channels are rebuilt from the
argparse defaults: `--kernel_size 5 --terrain_covariates True --norm_stats_json
data/conv_spline/norm_stats_E2c.json` (see the README; checked to load with 0 warm-started convs).

**The known limitation, which travels with the product:** beyond 100 px from past change at
+10 yr the 95% interval covers 86.1% of observations and the forecast loses to persistence on
CRPS (`docs/global/global_production_e2c.md`).

**Why they are kept.** Re-predicting the global product from these costs ~75 min per fold;
retraining adds ~20 min per fold plus the smoke. They are 41 MB each against ~435 GB of
run rasters.
