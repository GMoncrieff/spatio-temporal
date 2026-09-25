# The argmin floor — superseded 2026-09-14, kept as the evidence

Three seeds of b1 trained with `ModelCheckpoint`'s argmin on `val_crps`, i.e. **without**
`--weight_avg_last`. Not comparable to anything scored after that flag entered `BASE_ARGS`,
and moved out of the ranking directory for that reason.

Kept because it is the measurement that justified the change. `val_crps` plateaus after
~epoch 30 and then oscillates by +/- 0.001, so the argmin picked epochs **67, 124, 146, 67,
127, 111** across the six fold-models. Accuracy did not care; the gates did:

| metric | s42 | s43 | s44 | band width | as % of mean |
|---|---|---|---|---|---|
| `crps_skill20` | 0.2894 | 0.2891 | 0.2919 | 0.0027 | 0.9% |
| `skill20` | 0.2534 | 0.2534 | 0.2516 | 0.0018 | 0.7% |
| `rmse20` | 0.029459 | 0.029460 | 0.029494 | 3.5e-05 | 0.1% |
| `needle_mass_median_export_5` | 0.7392 | 0.5463 | 0.7033 | 0.193 | 29% |
| `needle_mass_median_export_20` | 0.3338 | 0.1379 | 0.2437 | 0.196 | 82% |
| `px_degenerate_frac_export_5` | 0.613 | 0.267 | 0.337 | 0.346 | 85% |
| `gap_frac_zero_export_5` | 0.0276 | 0.0087 | 0.0129 | 0.019 | 115% |
| `pit_mean_5` | 0.564 | 0.438 | 0.537 | 0.126 | 25% |
| `pit_ks5` | 0.291 | 0.228 | 0.183 | 0.108 | 46% |
| `tail_reach20` | 5.50 | 3.40 | 5.69 | 2.29 | 47% |

Accuracy reproducible to 0.1-1%; every gate the phase exists to move has a band of 25-115%
of its own mean, on three replicates of ONE configuration. Accuracy rows would have been
marked DEGEN and no gate movement below ~0.2 could have been read at all.

The seeds did not all land in one mode (rule 9's check): `pit_mean_5` straddles 0.5 at
0.438 / 0.537 / 0.564, so this is a real spread rather than a narrow sample.
