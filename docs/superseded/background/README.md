# Background — superseded development record

**Nothing in this folder describes the current product.**

The production system is documented by three standalone documents in `docs/`:

- `docs/global_ensemble_methodology.md` — the method
- `docs/global_scorecard.md` — every evaluation and what it scored
- `docs/fitting_running_model.md` — the runbook

Those three do not depend on anything here. This folder is kept for one purpose: recovering
*why* a design choice was made, and what was measured and rejected on the way to it. Much of
the value is in the negative results.

Read anything here with two cautions. **Numbers are not comparable to the current product** —
most were measured on regional extents, earlier fold masks, earlier calibration chains or
earlier scoring instruments, and several scorecards have different denominators from each
other. And **decisions recorded here have sometimes been reversed** by later measurement at
global scale; where they conflict with the three production documents, the production
documents are correct.

| file | what it holds |
|---|---|
| `global_product.md` | how the global product was built, and the four defects the global scale exposed |
| `global_final_phase.md` | the plan the global build executed |
| `ensemble_model_outline.md` / `.html` | the previous full method-and-scorecard write-up, on the regional lineage |
| `improvement_plan.md` | the last development round, stage by stage |
| `stage_a_instruments.md` | the scoring instruments and why several were rewritten |
| `model_phase.md` | 22 ConvLSTM experiments, none adopted |
| `validator_scaling.md` | the evaluation harness, its OOM and the budget-bounded rewrite |
| `central_field_baseline.md` | the central-field baseline measurements |
| `reference_card.md` | an earlier scorecard, two calibration layers |
| `current_progress.md` | an earlier status snapshot |
| `next_phase_marginals.md` | the marginal-fitting phase plan |
| `next_phase_model.md` | the model-experiment phase plan |
| `ensemble_uncertainty_plan.md` | the original ensemble design plan |
| `pinball_loss_gradient_isolation.md` | why the quantile loss is isolated from the trunk |
| `simple_model_architecture_and_training.md` | the original architecture note |
