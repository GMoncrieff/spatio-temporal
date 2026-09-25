# E1v — the production head, chosen 2026-09-17

Promoted at the close of the conv-spline phase. These are the two fold checkpoints the
Africa hindcast was scored on; the global run starts from this configuration.

    --head_family pwl
    --isqf_tails True
    --isqf_space neglog
    --free_scale True
    --mu_mse_weight 0.0

plus `conv_spline_base.sh`'s BASE_ARGS: `--central_residual True --checkpoint_monitor val_crps
--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean,max
--predict_row_chunk 2048 --predict_qf_dtype float32 --weight_avg_last 20
--ssim_weight 0.0 --laplacian_weight 0.0 --histogram_weight 0.0`, seed 42.

Head as built: `family pwl, knots default14 (n=15, bins=14), slopes learned, free scale,
17 params/horizon` — one location, fourteen increments, two learned tail rates. No scale
channel: `--free_scale` removes it as of 2026-09-16.

Note the run's own log banner does NOT say "tails neglog": the banner was gated on
`head_family == "isqf"` when this run started, and was widened the same day. The 17-param
count is what proves the tails are present (15 without them).

| file | fold |
|---|---|
| final_fold1_2609257.ckpt | 1 |
| final_fold2_2609258.ckpt | 2 |

Scored: CRPS skill 0.2152 / 0.2823 / 0.3033 / 0.3032 at +5/10/15/20 yr, cov95_20 0.9722,
far_tail_excess at h=5 1.361. Comparison against E2a: data/conv_spline/promotion/.
