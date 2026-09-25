# E2a — the GLOBAL run's own checkpoints, 2026-09-22/23

The five k=5 hindcast fold models and the forward (`--train_all_splits`) model that produced
the delivered global E2a products. Copied here out of `models/checkpoints/`, which is
Lightning's shared `default_root_dir` and holds every run this project has ever trained — a
file there is not evidence of which arm wrote it. These six are identified from each run's own
log, by the `Prediction will use ...` line that names the checkpoint prediction actually ran on.

Configuration (E2a), identical for all six:

    --head_family pwl --free_scale True --mu_mse_weight 0.0

plus `conv_spline_base.sh` BASE_ARGS with `--predict_row_chunk 512`, seed 42. Head as built and
as verified out of all six logs: `family pwl, knots default14 (n=15, bins=14), slopes learned,
free scale, 15 params/horizon` — **no tail channels**, which is the single difference from
`../E1v_global/` (17 params). All six were verified on all five fingerprints (loss weights, 12
trunk context channels, head family AND parameter count, row banding, weight averaging), and
the forward model additionally on the production-mode banner and the absence of a FOLD-CV
banner.

| file | role | held out |
|---|---|---|
| final_fold1_3782627.ckpt | hindcast fold 1 | fold 1 |
| final_fold2_3782629.ckpt | hindcast fold 2 | fold 2 |
| final_fold3_3865750.ckpt | hindcast fold 3 | fold 3 |
| final_fold4_3868240.ckpt | hindcast fold 4 | fold 4 |
| final_fold5_3948821.ckpt | hindcast fold 5 | fold 5 |
| final_foldNone_3951459.ckpt | forward model, base 2020 | nothing (`--train_all_splits`) |

These are the averaged weights (mean of the last 20 epochs), which is what prediction ran on.

**THE PRODUCTION MODEL — chosen 2026-09-25.** `final_foldNone_3951459.ckpt` is the forward
model behind the delivered forecast; the five fold models are behind the delivered hindcast.
The run began as a clean A/B on the learned tails against `../E1v_global/`
(`docs/global_production_e2a.md`): the tails cost nothing centrally, and beyond 100 px at
+20 yr this head's point forecast beats persistence where E1v's does not (RMSE skill +0.0310
against −0.0070). **The known cost, which travels with the product:** out there its 95% interval
covers only 70.9% and 12.7% of pixels exceed its 99.9th percentile, because with no tail
parameters the distribution is bounded by its outermost knots.

Loading one of these requires E2a's own flags, or the head is rebuilt wrong from the argparse
defaults: `$BASE_ARGS --head_family pwl --free_scale True --mu_mse_weight 0.0` (see README).

**Why they are kept.** Re-predicting the global product from these costs ~2 h/fold; retraining
costs that plus ~50 min each. They are 97 MB against the ~437 GB of run rasters. Those rasters are also
what any re-scoring reads, so clearing them is a decision, not housekeeping.
