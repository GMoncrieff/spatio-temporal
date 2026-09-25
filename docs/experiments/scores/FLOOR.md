# b1's floor — established 2026-09-15

Three seeds (42/43/44), Africa, folds 1+2, `fold_mask_b4`, 150 epochs, float32 qf export,
`--weight_avg_last 20`. All nine verifier checks passed (loss weights, context wiring, weight
averaging) on every seed.

    BASE_ARGS = --head_family spline --central_residual True --checkpoint_monitor val_crps
                --mu_mse_weight 1.0 --context_radii 3,30,100 --hm_context_radii 3,30,100
                --hm_context_stats mean,max --predict_row_chunk 2048
                --predict_qf_dtype float32 --weight_avg_last 20
                --ssim_weight 0.0 --laplacian_weight 0.0 --histogram_weight 0.0

## Rankable (24 of 31 metrics read `ok`)

| metric | min | max | width | rel |
|---|---|---|---|---|
| crps_skill5 | 0.20719 | 0.21083 | 0.00364 | 2% |
| crps_skill20 | 0.29326 | 0.30165 | 0.00839 | 3% |
| skill20 | 0.25756 | 0.27078 | 0.01322 | 5% |
| rmse20 | 0.02911 | 0.02938 | 0.00026 | 1% |
| cov95_20 | 0.94091 | 0.94859 | 0.00768 | 1% |
| **needle_mass_median_ref_5** | 0.83059 | 0.86573 | 0.03513 | **4%** |
| **needle_mass_p90_ref_5** | 0.89735 | 0.91542 | 0.01807 | **2%** |
| **needle_mass_median_export_5** | 0.67458 | 0.71800 | 0.04342 | **6%** |
| **over_f_max_frac_ref_5** | 0.86122 | 0.95114 | 0.08991 | **10%** |
| needle_mass_median_ref_20 | 0.48707 | 0.61851 | 0.13145 | 23% |
| pit_mean_5 | 0.53088 | 0.54128 | 0.01040 | 2% |
| pit_ks5 | 0.18822 | 0.21103 | 0.02281 | 12% |
| zero_leak_neg_ratio_5 | 1.53311 | 1.54647 | 0.01337 | 1% |
| exceedance_abs_log10 | 0.48135 | 0.51647 | 0.03512 | 7% |

The fence is rankable at h=5 on four separate statistics. That is what the re-baseline bought.

## NOT rankable — `WEAK`, adopt nothing on these

| metric | width | rel | why |
|---|---|---|---|
| max_density_p99_{ref,export}_5 | 5.9e7 | 211% | export-limited; reported, not ranked |
| max_density_p99_ref_20 | 2.2e7 | 298% | same |
| px_degenerate_frac_export_5 | 0.174 | 83% | still seed-dominated after averaging |
| gap_frac_zero_export_5 | 0.0077 | 125% | same |
| pit_ks20 | 0.0862 | 60% | |
| pit_rms_se_60_5 | 141 | 25% | |
| **tail_reach20** | 2.625 | 38% | got WIDER under averaging (was 2.29) |

`tail_reach20` is the one thing the re-baseline made worse. It correlates with `pit_ks5` at
r = 0.691 (rule 10), so it may not be an independent axis -- but it is a caveat, not a win.

## Why this floor replaced the first one

See `floor_argmin/README.md`. The argmin floor's gate bands were 25-115% of their own mean.
Averaging the plateau instead of picking its argmin cut `pit_mean_5` 12x, `pit_ks5` 4.7x and
`needle_mass_median_export_5` 4.4x -- **and improved accuracy** (skill20 0.258-0.271 against
0.252-0.253), by more than the whole argmin band.
