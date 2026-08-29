#!/usr/bin/env python3
"""Generate docs/dist_model_phase2.md from the scored summaries.

Every number in the document comes from data/ensemble/exp/dist_scores/summary_*.json, so the
prose and the tables cannot drift apart. Run it again after any rescore.
"""
import glob
import json
import re

import pandas as pd

SCORES = "data/ensemble/exp/dist_scores"
OUT = "docs/dist_model_phase2.md"

H = [5, 10, 15, 20]
PER_H = ["crps", "crps_skill", "rmse", "skill", "pit_ks", "pit_gt_0999", "cov95", "tail_reach"]
SCALARS = ["exceedance_abs_log10", "central_outside_interval", "qf_vs_triple_max"]
# (metric, better direction) for the "seeds outside the floor" count.
DIRECTED = ([(f"crps_skill{h}", "up") for h in H] + [(f"skill{h}", "up") for h in H]
            + [(f"rmse{h}", "down") for h in H] + [(f"pit_ks{h}", "down") for h in H]
            + [("exceedance_abs_log10", "down")])

df = pd.DataFrame([json.load(open(p)) for p in sorted(glob.glob(f"{SCORES}/summary_*.json"))])
df = df.set_index("label")
num = df.select_dtypes("number")
FLOOR = num[[bool(re.match(r"^afw_s\d+$", str(i))) for i in num.index]]


def key(metric, h):
    """Column name for a metric at a horizon. Two of them carry an underscore."""
    return f"{metric}_{h}" if metric in ("cov95", "pit_gt_0999") else f"{metric}{h}"


def runs(pat):
    return num[[bool(re.match(pat, str(i))) for i in num.index]]


def fmt(v, metric):
    if metric.startswith("crps") and not metric.startswith("crps_skill"):
        return f"{v:.5f}"
    if metric.startswith("rmse"):
        return f"{v:.5f}"
    if metric.startswith("tail_reach"):
        return f"{v:.2f}"
    if metric.startswith("pit_gt"):
        return f"{v:.5f}"
    return f"{v:.4f}"


def metric_table(pat):
    """One row per horizon, median across the matching runs."""
    sub = runs(pat)
    head = "| h | CRPS | CRPS skill | RMSE | central skill | PIT KS | P(u>0.999) | cov95 | tail reach |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for h in H:
        cells = [fmt(sub[key(m, h)].median(), m) for m in PER_H]
        lines.append(f"| +{h} yr | " + " | ".join(cells) + " |")
    sc = sub[SCALARS].median()
    lines.append("")
    lines.append(f"`exceedance_abs_log10` {sc['exceedance_abs_log10']:.4f} · "
                 f"`central_outside_interval` {sc['central_outside_interval']:.5f} · "
                 f"`qf_vs_triple_max` {sc['qf_vs_triple_max']:.2e}")
    return "\n".join(lines)


def seed_lines(pat, metrics=("crps_skill20", "skill20", "pit_ks5", "tail_reach20")):
    sub = runs(pat)
    out = []
    for m in metrics:
        vals = " / ".join(fmt(v, m) for v in sub[m])
        out.append(f"`{m}` {vals}")
    return "; ".join(out)


def vs_floor(pat, unanimous_only=False):
    """Metrics where every seed (or any seed) sits outside the floor's whole range."""
    sub = runs(pat)
    n = len(sub)
    hits = []
    for m, d in DIRECTED:
        lo, hi = FLOOR[m].min(), FLOOR[m].max()
        w = hi - lo
        better = int((sub[m] > hi).sum()) if d == "up" else int((sub[m] < lo).sum())
        worse = int((sub[m] < lo).sum()) if d == "up" else int((sub[m] > hi).sum())
        med = sub[m].median()
        if better == n and n > 1:
            marg = ((med - hi) if d == "up" else (lo - med)) / w
            hits.append(f"**{m}** better {n}/{n} ({marg:+.2f} widths)")
        elif worse == n and n > 1:
            marg = ((lo - med) if d == "up" else (med - hi)) / w
            hits.append(f"**{m}** WORSE {n}/{n} (−{marg:.2f} widths)")
        elif not unanimous_only and (better or worse) and n > 1:
            side = "better" if better else "worse"
            hits.append(f"{m} {side} {max(better, worse)}/{n}")
    return hits


def reported_line(pat):
    """`tail_reach20` and `cov95` carry no "better" direction -- a target, not an extreme -- so
    the bar never judges them. They are the two rows this phase's headline results live in, so
    they are stated explicitly rather than left to be inferred from the table."""
    sub = runs(pat)
    n = len(sub)
    out = []
    for m in ("tail_reach20", "cov95_20"):
        lo, hi = FLOOR[m].min(), FLOOR[m].max()
        above, below = int((sub[m] > hi).sum()), int((sub[m] < lo).sum())
        where = (f"**all {n} seeds ABOVE** the floor" if above == n else
                 f"**all {n} seeds BELOW** the floor" if below == n else
                 f"{above} above / {below} below the floor" if (above or below) else
                 "inside the floor")
        w = hi - lo
        med = sub[m].median()
        marg = ""
        if above == n or below == n:
            marg = f", median {(med - hi) / w:+.2f} widths past the edge" if above == n \
                   else f", median {(lo - med) / w:.2f} widths past the edge"
        out.append(f"`{m}` median {fmt(med, m)} vs floor {fmt(lo, m)}–{fmt(hi, m)} — {where}{marg}")
    return "; ".join(out)


def floor_ranges():
    lines = ["| metric | floor min | floor max | width | width / level |", "|---|---|---|---|---|"]
    for m in ([f"crps_skill{h}" for h in H] + [f"skill{h}" for h in H]
              + [f"pit_ks{h}" for h in H] + [f"cov95_{h}" for h in H]
              + [f"rmse{h}" for h in H] + [f"tail_reach{h}" for h in H]
              + ["exceedance_abs_log10"]):
        lo, hi = FLOOR[m].min(), FLOOR[m].max()
        rel = (hi - lo) / abs(FLOOR[m].mean())
        lines.append(f"| `{m}` | {fmt(lo, m)} | {fmt(hi, m)} | {fmt(hi - lo, m)} | {rel:.0%} |")
    return "\n".join(lines)


def south_table():
    south = num[[not str(i).startswith(("af_", "afw")) for i in num.index]]
    cols = ["crps_skill5", "crps_skill20", "pit_ks5", "skill5", "skill20", "tail_reach20"]
    lines = ["| run | crps_skill5 | crps_skill20 | pit_ks5 | skill5 | skill20 | tail_reach20 |",
             "|---|---|---|---|---|---|---|"]
    for label in sorted(south.index):
        cells = [fmt(south.loc[label, c], c) for c in cols]
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


MODELS = [
    ("afw — the floor", r"^afw_s\d+$", "*(the comparator; six seeds)*",
     "The shipped distributional configuration: spline head, CRPS alone, `--central_residual`, "
     "`--central_context`, `--monotone_quantile_width`, `--quantile_context`, "
     "`--checkpoint_monitor val_crps`, `--weight_avg_last 20`, `--spline_knots default14`, "
     "learned slopes, cumulative width. Six seeds (42–47), identical in every other respect."),
    ("e1 — the neighbourhood covariate", r"^af_e1_s", "`--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean,max`",
     "Twelve context channels instead of eight. Before this the model knew where past *change* "
     "happened but never the *level* of development around a pixel. Note it changes two things: "
     "the covariate **and** the dropped fine radii — `rad` below separates them."),
    ("e1a — the covariate, mean only", r"^af_e1a_s", "`--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean`",
     "Isolates whether the `max` bands add anything over `mean`."),
    ("rad — the context radii alone", r"^af_rad_s", "`--context_radii 3,30,100`",
     "The control e1 was owed: the radii change without the covariate. `e1 − rad` is the "
     "covariate; `rad − floor` is the radii."),
    ("e2 — knots in the body", r"^af_e2_s", "`--spline_knots body_dense`",
     "Adds knots at u = 0.35/0.45/0.55/0.65. The default grid has three knots between u = 0.10 "
     "and u = 0.90 while 53% of pixels move by less than 0.001 in twenty years."),
    ("e4 — lower-tail weighted CRPS", r"^af_e4_s", "`--crps_tail_weight_lo 1.0`",
     "Weights the pinball terms below u = 0.05. `P(u<0.001)` read ~7x nominal at h=20, i.e. the "
     "model was blindsided by declines. A u-weighting moves where the optimiser spends effort "
     "and not the target, so it needs no importance correction."),
    ("e5 — horizon-weighted loss", r"^af_e5_s", "`--horizon_loss_weights 1,1.33,2,4`",
     "h=20 receives a quarter of h=5's gradient because exposure is 4:3:2:1 by horizon."),
    ("e6 — deeper shape head", r"^af_e6_s", "`--shape_head_hidden_layers 2`",
     "Tests whether the shape head is under-capacity."),
    ("e7 — wider shape head", r"^af_e7_s", "`--shape_head_width 64`",
     "The same question through width rather than depth."),
    ("nomono — no horizon-monotone width", r"^af_nomono_s", "`--spline_cumulative_width False`",
     "Each horizon gets an independent scale instead of a cumulative sum of non-negative "
     "increments, so the 95% width may **shrink** with lead time. Q(u) increasing in u is "
     "untouched — that is structural to the spline."),
    ("e9 — no slope freedom", r"^af_e9_s", "`--spline_slopes fritsch`",
     "Every knot slope is derived from the adjacent secants (monotonicity-preserving, zero "
     "parameters) instead of learned: 29 head parameters per horizon fall to 16."),
]

parts = []
parts.append(f"""# The distributional model, round 2: southern Africa, then Africa

Branch `dist-convlstm`. Written 2026-08-28, from
`data/ensemble/exp/dist_scores/summary_*.json` — every number below is generated from those
files by `scripts/gen_dist_phase2_doc.py`, so the prose and the tables cannot drift apart.
Regenerate it after any rescore; do not hand-edit the numbers.

**What is being tested.** A ConvLSTM whose head emits a full per-pixel quantile function
`Q_h(u|x)` — a monotone rational-quadratic spline on fixed tail-dense knots, anchored at
persistence, trained on CRPS. The quantile function *is* the product: no conformal scaling, no
width factors, no empirical marginal, no horizon-monotonicity pass. This document covers the
second round of experiments on it.

**What one run is.** Two folds (1 and 2) of the k=5 checkerboard mask, trained on global chips,
predicted and stitched over the region, scored on held-out fold territory only. A *variant* is
three such runs differing only in seed; the *floor* is six.

---

## Part 1 — Southern Africa: what the screen taught us, and why we left it

### 1.1 Two mechanical bugs explained round 1's "instability"

Round 1 concluded that the model was too unstable to rank anything: `tail_reach20` varied by
**4.3 within one configuration**. Both causes turned out to be mechanical.

**A `sqrt` derivative singularity silently killed 6 of 16 runs.** `quantile_spline.cdf` took
`sqrt(disc)` with `disc` floored at zero. `sqrt` is finite at zero and its derivative is not, and
a masked-out pixel supplies exactly-zero incoming gradient, so `0/0 = NaN`. **One pixel in
131,072** poisoned 38 of 94 parameter tensors in a single step — permanently, with a finite loss
and an ordinary gradient norm. It was invisible because `ModelCheckpoint` never selects a NaN
epoch: a dead run kept its last healthy checkpoint, finished, and **scored normally**. Fixed by
evaluating `sqrt` away from the singularity and selecting afterwards; bit-identical forward,
verified 38 → 0, and now guarded by `--abort_on_nonfinite` at three points.

Audited by grepping every fold log for `train_loss=nan`:

| run | flags | dead fold | verdict on record, now void |
|---|---|---|---|
| `d2` | `--chip_sampling stratified --chip_sampling_correct True` | fold 2, NaN 149/149 epochs | "tail destroyed, CRPS worse at every horizon" |
| `d8` | `--spline_slopes fritsch` | fold 2, NaN 84 (died ~epoch 65) | "clears pit_ks5 by 2.3x; tail worse by 1.4x the band" |
| `d1a` | `--crps_tail_weight 1.0` | fold 2, NaN 31 (died ~118) | the round's only apparent winner |
| `d1a_s43` | replicate of the above | fold 2, NaN from epoch 0 | supplied the 4.3 spread |
| `d3` | `--ssim_weight 0.2` | fold 2, NaN 30 | "SSIM does not earn its place" |
| `d6` | `--mu_mse_weight 0.0` | fold 2, NaN 41 | "pure CRPS costs nothing centrally" |
| `d7` | `--isolate_shape_grad True` | fold 1 NaN 108, fold 2 NaN 4 | "the trunk must hear the objective" |

Both folds are scored, so each of those verdicts was read off a half-dead raster. Clean:
`d0_s42–44`, `d1b`, `d2u`, `d4`, `n1`. **Round 1's two surviving findings — `d6` and `d7` — are
both on that void list** and are owed a redo.

**`val_crps` is flat to 1.0–2.4% across 150 epochs**, so checkpoint selection was a lottery: it
once picked epoch 19 of 150, and `r(selected epoch, tail_reach20) = +0.773` on clean runs. Round
1 had recorded that correlation as 0.150 and rejected the hypothesis; that measurement was made
on NaN-contaminated runs.

Two consequences for the record: round 2's design spec rests on artifacts of these bugs (the 4.3
spread is 1.52 without the epoch-4 death), and the `r = 0.691` correlation between `tail_reach20`
and `pit_ks5` that was written up as "two metrics are one axis" falls to +0.236 without the two
dead runs and is **−0.964** on six clean ones.

### 1.2 The stability gate could not work, and was never run

It proposed judging four configurations on the *spread* of three seeds. Measured on a six-seed
floor, a three-seed **range** varies by 17.5x across draws from one configuration (0.121 → 2.117)
while the median varies by 1.1x. A range is not a statistic. The gate was skipped and its lever
found directly:

- **`--weight_avg_last 20` — ADOPTED.** Beat the southern-Africa floor on all eight metrics and
  cut the `tail_reach20` spread 2.117 → 0.263. Carried by every Africa run since.
- **`--checkpoint_select final` — REJECTED.** Buys tail reach and drives `skill20` negative in
  2 of 3 seeds (`fin_s45` −0.0475, `fin_s46` −0.0077).
- `--lr_schedule cosine` — still untested.

### 1.3 Every southern-Africa run

Numbers here are **not comparable to Part 2**: southern Africa changes 2–4x less than Africa at
every HM level and its `[0,0.01)` stratum is 6% of the region against 40% of Africa, so the
skill denominators differ. Rows marked void above are included for completeness.

{south_table()}

`n1` is `--dist_loss nll`: the log score collapses the distribution to a point mass at the width
floor (`tail_reach20` exactly 1.0, `crps_skill5` −11.1). It is the mirror of the incumbent's
Gaussian-NLL failure, which inflated widths 7x, and it is why CRPS is the objective.

### 1.4 Why the region had to be abandoned

**Southern Africa's far-field band contains zero pixels.** Not a zero rate — an empty band. The
model exists to fix one named defect: the far field emits 3.4% of the observed rate of new
development beyond 100 px. That defect is *unmeasurable* at this extent, so every one of the 22
model experiments screened here was blind to the thing they were for. Africa's >100 px band holds
**175,478 px** at h=20, and the distributional model's first measurement there is
`P(ΔHM>0.05)` predicted 0.000010 against observed 0.000137 — **7.3% of observed**, against the
frozen product's 3.4%. Still 13.7x short, but measurable at last.

**Verdicts do not transfer between the regions.** Measured, not assumed:

| variant | southern Africa | Africa |
|---|---|---|
| `e1` / `e1a` (covariate) | worse on 7 of 8; monotone `floor > e1a > e1` | **inverts**: `e1 > e1a > floor` on every skill metric |
| `e4` (lower-tail weight) | the round's only winner, replicated at n=3 | **null at n=3**; the headline inverted at n=1 and then vanished |
| `e5` (horizon weights) | worse on everything | worse on everything — the one verdict that transferred |
| `e2` (body knots) | neutral | neutral |
| `e6` (deeper head) | the slate's worst, 0 better / 7 worse | null, one weak-row cost |

A regional screen ranked some of these backwards. That is the phase's most expensive lesson.

---

## Part 2 — The Africa programme

**The screen.** `--predict_subsample_blocks 200 --predict_subsample_seed 7` predicts 200 of 965
candidate 128 px blocks, intersected into the fold mask so every tile overlapping a kept pixel is
still processed. Blended values on kept pixels match a full-density run to float32 (measured
1.3e-6). Back-tested on existing rasters: at 41% of area, r = 0.990 on `tail_reach20` with 97%
ranking agreement. Every run in Part 3 uses the same blocks, so the comparisons are paired.

**The floor.** Six seeds of the identical configuration. Anything inside this range is seed
noise, not a result:

{floor_ranges()}

Two rows deserve care. `tail_reach20`'s width is 23% of its level and its spread is *model*
variance rather than sampling error (between-seed 0.860 against within-seed 0.369 across four
windows of one model), so it does not tighten with more pixels. The `pit_ks` rows at h ≥ 10 run
42–66% of their own level and are marked **under-powered**: a null there means "cannot see",
not "no effect".

**The bar.** A variant must beat the floor's range by **more than the range's own width** — a
single new draw falls outside a three-run range about half the time under the null, so "outside
the range" is barely a test. Variants are ranked on the **median** of their three seeds, never on
their range.

*One correction to the instrument, made during this phase:* the WORSE half of that test was
measured from the floor's **best** edge, so it fired the instant a value left the range while
"better" needed a full extra width — two halves of one test at bars one width apart. Seven of the
eight WORSE verdicts on this slate sat 0.04–0.78 widths past the floor and evaporate under the
symmetric rule. Any "N better / M worse" count recorded before 2026-08-27 was scored on the lax
half.

**How to read Part 3.** Each model gives the median across its seeds for every metric at every
horizon, then the list of metrics where **all** seeds fall outside the floor's whole range. That
unanimity count is the honest signal in this phase: the margins are mostly small, and three
independent draws landing on the same side of a six-seed range is roughly a 1-in-343 event under
the null.

---

## Part 3 — Every Africa model
""")

for title, pat, flags, blurb in MODELS:
    sub = runs(pat)
    seeds = ", ".join(str(i).split("_s")[-1] for i in sub.index)
    hits = vs_floor(pat, unanimous_only=False)
    unan = [h for h in hits if "/" in h and h.startswith("**")]
    other = [h for h in hits if not h.startswith("**")]
    parts.append(f"""
### {title}

**Changed from the floor:** {flags}
**Seeds:** {seeds} (n={len(sub)})

{blurb}

{metric_table(pat)}

**Per seed:** {seed_lines(pat)}
""")
    if not pat.startswith("^afw"):
        parts.append("**Reported, not judged:** " + reported_line(pat) + "\n")
    if pat.startswith("^afw"):
        parts.append("*This is the comparator, so it has no \"outside the floor\" line.*\n")
    else:
        parts.append("**Unanimous against the floor:** "
                     + ("; ".join(unan) if unan else "*none*") + "  \n")
        parts.append("**Split:** " + ("; ".join(other) if other else "*nothing outside the floor*")
                     + "\n")

parts.append("""
---

## Part 4 — What this phase established

**Nothing cleared the bar.** Ten variants, three seeds each, against a six-seed floor: not one
median beat the floor's range by more than the range's own width. Two results are nonetheless
solid, because they rest on unanimity across seeds rather than on margin.

**1. The horizon-monotone width constraint is load-bearing — keep it.** `nomono` puts all three
seeds below the floor's entire range on `tail_reach20` (3.99 / 4.09 / 4.33 against a floor of
5.23–6.67), 0.78 widths past the edge, at no measurable central cost. A **second, independent**
metric agrees: `exceedance_abs_log10` — the error in predicted exceedance rates, which is what
the far-field defect is actually about — is worse in all three seeds as well (median 0.4184
against a floor of 0.3361–0.3925, 0.46 widths past). Two unanimous rows pointing the same way is
a great deal harder to explain as noise than one. `tail_reach` is
`(Q(0.999) − Q(0.5)) / (Q(0.975) − Q(0.5))`; the Gaussian value is 1.577 and the far field needs
~12 to put +0.05 inside its interval, so a 35% lighter tail is movement directly away from the
defect the model exists to fix. An external design review recommended *not* forcing width to grow
with lead time; on this model that recommendation is measurably wrong.

*Mechanism, offered as inference rather than measurement:* `tail_reach` is scale-invariant, so
this is not the scale parameterisation showing through. Under the cumulative sum, h=20's spread
is built from non-negative increments over h=5's and part of it has to be expressed through
shape; freed of the constraint, the optimiser buys h=20 spread with the scale alone and leaves a
lighter shape.

**2. Parsimony is the most promising lever tested.** `e9` — slopes from secants instead of
learned, 29 head parameters per horizon down to 16 — puts all three seeds below the floor on
`pit_ks5` at **0.99 floor widths**, one hair short of the bar, on a row that is properly powered
(width 15% of level). It costs nothing: `skill5`, `skill20`, `rmse5`, `rmse20` are each better in
2 of 3 seeds, and `tail_reach20` is if anything heavier (median 6.59 against a floor median of
6.37). This is the design review's §2.1 hypothesis — that the spline is more flexible than the
effective sample size supports — surviving its first real test. Its round-1 predecessor `d8` was
one of the NaN-killed runs, so it had never been tested cleanly before.

**3. e1's gain is two separable effects, and the control found them.** Dropping the fine context
radii buys the +5 yr central field and nothing beyond it (`skill5` and `rmse5` unanimous, h ≥ 10
all noise); the covariate buys h ≥ 10 and nothing at h=5. Adding the covariate even gives a
little of the h=5 gain back and costs `pit_ks5`. Two levers, two jobs, no overlap.

**4. Capacity and loss-shape knobs are inert; one is harmful.** `e2`, `e6`, `e7` — body knots,
deeper head, wider head — move nothing. `e4` is a null at n=3 after being southern Africa's only
winner. `e5` (horizon weights) is the single clear loser, worse on four metrics, mostly at h=5:
it takes gradient from the horizon with four times the data and spends it on the least-powered
one, and both ends get worse.

The pattern across the whole slate: **the model is not short of capacity or of a cleverer
objective — it is short of an input it never had.** The only levers that moved anything were a
new covariate and *removing* head parameters.

### Not run, and why

- **e8** (`--chip_sampling stratified --chip_sampling_correct True`) — the redo `d2` is owed,
  since `d2`'s verdict came from a fold that was NaN for all 149 epochs. Machinery and weight
  table are built. Dropped from this phase by choice.
- **e10** (`--spline_knots lean9`, a strict subset of the default grid: 14 bins → 8) — implemented
  and tested, never trained. Dropped by choice.
- **e3** (`--spline_knots deep_lower`) — **structurally excluded.** Knots 1e-4 apart drive
  boundary derivatives, pinned to secants `h/width`, to ~1e4 against ~1e3, and the shape-head
  biases go non-finite on the first optimiser step. Not a null result.
- **`d6` and `d7` redone** — round 1's two surviving findings, both from runs with a dead fold.
- **`--lr_schedule cosine`** — the last untested lever of the weight-averaging family.

### Instrument changes made during this phase

- `--abort_on_nonfinite` (default on) at three points in training.
- `compare_dist_runs.py`: `--group_seeds` (rank on the per-metric median of a variant's
  replicates), `--include` (keep another region's runs out of a floor's table), and the symmetric
  WORSE bar.
- `_spline_head_banner` prints `Spline head: knots <preset> (n=, bins=), slopes <mode>, N
  params/horizon`, and `verify_spline_head` greps it, so a head-capacity flag that silently
  failed to engage cannot read as "the lever does nothing". The printer and the check are tested
  against each other, and the check is proven to fire on three controls.
- The block screen (`--predict_subsample_blocks`) and blend-on-write, which took the two-fold
  prediction peak from 196 GB to 53 GB.

### A note on process

Two failures in this phase were not model failures:

*A gate written for a three-seed spread.* The judging statistic was never checked against the
thing it would be applied to. One measurement on the six-seed floor showed it varies 17.5x.

*A wait loop keyed on process-table text.* Every chained stage waited with
`ps -eo cmd | grep -q "<previous script>.s[h]"`. An unrelated shell that merely *quoted* the
script name in a comment kept matching after the stage had exited, and the queue sat idle for
3.2 hours with both GPUs free. The `[h]` bracket trick stops a grep matching itself; nothing
stops a third process from containing the string. Every wait is now `while kill -0 <pid>`.
""")

with open(OUT, "w") as fh:
    fh.write("\n".join(parts))
print(f"wrote {OUT}")
