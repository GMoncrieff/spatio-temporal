"""Post-hoc spatiotemporal residual ensemble for the ConvLSTM HM forecast.

The frozen checkpoint's central forecast is preserved exactly; joint spatial/temporal
structure is added on top by sampling correlated error fields calibrated from the model's
own out-of-sample hindcast residuals and pushing them through per-pixel quantile marginals
via a Gaussian copula.

Modules
-------
residuals   Phase 0 — out-of-sample residual harness and rank-Gaussian transform
validate    Phase 1 — coverage-vs-scale and class-conditional coverage diagnostics
variogram   Phase 1c — stratified variogram estimation and multi-scale fitting
calibrate   Phase 1.5 — Mondrian (class-conditional) split-conformal recalibration
fields      Phase 2 — correlated Gaussian residual field generation
copula      Phase 3 — two-piece-normal marginals and copula sampling
aggregate   Phase 4 — region statistics and ensemble summaries
"""
