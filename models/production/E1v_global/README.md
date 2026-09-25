# E1v — the GLOBAL run's own checkpoints, 2026-09-17/18

The five k=5 hindcast fold models and the forward (`--train_all_splits`) model that produced
the delivered global products. Copied here out of `models/checkpoints/`, which is Lightning's
shared `default_root_dir` and holds every run this project has ever trained — a file there is
not evidence of which arm wrote it. These six are identified from each run's own log.

Configuration (E1v), identical for all six:

    --head_family pwl --isqf_tails True --isqf_space neglog --free_scale True
    --mu_mse_weight 0.0

plus `conv_spline_base.sh` BASE_ARGS with `--predict_row_chunk 512`, seed 42. Head as built:
`family pwl, knots default14 (n=15, bins=14), slopes learned, tails neglog, free scale,
17 params/horizon`. All five folds and the forward model were verified on all five
fingerprints (loss weights, 12 trunk context channels, head, row banding, weight averaging).

| file | role | held out |
|---|---|---|
| final_fold1_2938935.ckpt | hindcast fold 1 | fold 1 |
| final_fold2_2938936.ckpt | hindcast fold 2 | fold 2 |
| final_fold3_3024270.ckpt | hindcast fold 3 | fold 3 |
| final_fold4_3026908.ckpt | hindcast fold 4 | fold 4 |
| final_fold5_3111819.ckpt | hindcast fold 5 | fold 5 |
| final_foldNone_3115415.ckpt | forward model, base 2020 | nothing (`--train_all_splits`) |

These are the averaged weights (mean of the last 20 epochs), which is what prediction ran on.

**Why they are kept.** Re-predicting the global product from these costs ~2 h/fold; retraining
costs that plus ~50 min each. They are 97 MB against the ~586 GB of rasters that were deleted.
Predict-only re-run: `run_hindcast_folds.py --fold_checkpoints 1=...,2=... --max_epochs 0`.

Scored (global hindcast, out of sample, 184,573,321 land px): crps_skill
0.157 / 0.266 / 0.300 / 0.298 at +5/10/15/20 yr. Scorecard:
`data/conv_spline/scores/global/scorecard_detailed_g_E1v_hind.html`.

Distinct from `models/production/E1v/`, which holds the two AFRICA fold checkpoints the
promotion decision was made on.

**Not the production model since 2026-09-25.** E2a (`../E2a_global/`) was chosen after the
global A/B; these are kept as the measured alternative with calibrated far-field tails.
