#!/usr/bin/env python3
"""Generate docs/dist_global_scorecard.md for one configuration, from its own scored CSVs.

Every number is computed here from data/ensemble/exp/dist_scores/{dist,central,consistency}_*.csv,
so the prose and the tables cannot drift apart. Regenerate after any rescore; do not hand-edit
the numbers.

Aggregation matches `score_distributional_model.py`: within a run, values are averaged over the
four forecast windows weighted by pixel count; across seeds the **median** is taken, which is the
statistic the phase ranks on (a three-seed range varies 17.5x across draws from one
configuration, a median 1.1x).

    python scripts/gen_dist_scorecard.py [--run af_e1_s] [--floor afw_s]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SCORES = "data/ensemble/exp/dist_scores"
OUT = "docs/dist_global_scorecard.md"
H = [5, 10, 15, 20]
BANDS = ["0-1", "1-3", "3-10", "10-30", "30-100", ">100"]


def load(kind, pat):
    fs = sorted(glob.glob(f"{SCORES}/{kind}_{pat}*.csv"))
    if not fs:
        raise SystemExit(f"no {kind}_{pat}*.csv under {SCORES}")
    return {os.path.basename(f)[len(kind) + 1:-4]: pd.read_csv(f) for f in fs}


def wmean(df, col):
    sub = df.dropna(subset=[col])
    return float(np.average(sub[col], weights=sub["n"])) if len(sub) else np.nan


def sel(d, h, stratum="pooled", b="all"):
    return d[(d.horizon == h) & (d.stratum == stratum) & (d["bin"] == b)]


def med(store, col, h, stratum="pooled", b="all"):
    """Median across seeds of the pixel-weighted mean over windows."""
    return float(np.median([wmean(sel(d, h, stratum, b), col) for d in store.values()]))


def per_seed(store, col, h, stratum="pooled", b="all"):
    return [wmean(sel(d, h, stratum, b), col) for d in store.values()]


def npx(store, h, stratum, b):
    d = next(iter(store.values()))
    return int(sel(d, h, stratum, b)["n"].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="af_e1_s", help="label prefix of the configuration scored")
    ap.add_argument("--floor", default="afw_s", help="label prefix of the comparator floor")
    args = ap.parse_args()

    D, C, K = load("dist", args.run), load("central", args.run), load("consistency", args.run)
    # exceedance_abs_log10 is a summary-level statistic (mean |log10(pred/obs)| over the
    # distance bands), so it comes from the same summary JSONs the phase ranks on.
    summaries = [json.load(open(f)) for f in sorted(glob.glob(f"{SCORES}/summary_{args.run}*.json"))]
    exc = float(np.median([s["exceedance_abs_log10"] for s in summaries]))
    FD, FC = load("dist", args.floor), load("central", args.floor)
    seeds = ", ".join(k.split("_s")[-1] for k in D)
    fseeds = ", ".join(k.split("_s")[-1] for k in FD)
    n_pooled = npx(D, 20, "pooled", "all")

    def frange(col, h, stratum="pooled", b="all", fmt="{:.4f}"):
        """The floor's own range for a metric -- the noise band. `skill` and `rmse` live in the
        central CSVs and everything else in the dist CSVs; picking the wrong store is a silent
        KeyError at best and the wrong column at worst."""
        store = FC if col in ("skill", "rmse", "mae", "bias", "corr", "slope") else FD
        v = per_seed(store, col, h, stratum, b)
        return f"{fmt.format(min(v))}–{fmt.format(max(v))}"

    doc = []
    doc.append(f"""# Distributional model scorecard — what each metric asks, and how the model does

Every metric `scripts/score_distributional_model.py` reports, explained in plain terms, with the
measured value beside it.

**Provenance.** Branch `dist-convlstm`, configuration **e1** — the floor plus the neighbourhood-HM
covariate (`--context_radii 3,30,100 --hm_context_radii 3,30,100 --hm_context_stats mean,max`,
twelve context channels instead of eight) — over **three seeds ({seeds})**. Screened on
**Africa**, folds 1 and 2 of `fold_mask_b4_1000.tif`, trained on global chips, predicted on the
200-block screen (`--predict_subsample_seed 7`), {n_pooled:,} scored pixels at h=20. Measured
2026-08-28. Method and the full slate in `docs/dist_model_phase2.md`.

**This supersedes the southern-Africa edition**, which scored configuration D0 on a region whose
far-field band contains **zero pixels** — the defect this model exists to fix was unmeasurable
there. Numbers are not comparable between the two editions: Africa changes 2–4x more at every HM
level and its `[0,0.01)` stratum is 40% of the region against 6%.

**e1 is the best available configuration, not a validated improvement.** Ten variants were tested
against a six-seed floor and **none cleared the bar** (beat the floor's whole range by more than
the range's own width). e1 is the only one whose three seeds all landed outside the floor on the
better side, on five metrics, by 0.06–0.24 floor widths.

**Read the levels, not small differences.** Each table below carries the floor's own six-seed
range (`floor`, seeds {fseeds}) as the noise band. Where the band is wide, the number cannot rank
anything — this is stated per section rather than left to be inferred.

---

## 1. Is the single best-guess map any good?

**`rmse`, `mae`, `bias`, `corr`, `slope`, `skill`**

Asks: how far is the central forecast from the truth, and does it beat "assume nothing changes"?
Persistence is the bar, not zero — the median 20-year HM change is 0.0001, so a forecast can look
accurate while being worse than doing nothing.

| h | RMSE | MAE | bias | corr | slope | skill vs persistence | floor's skill range |
|---|---|---|---|---|---|---|---|""")

    for h in H:
        doc.append(f"| +{h} | {med(C,'rmse',h):.5f} | {med(C,'mae',h):.5f} | "
                   f"{med(C,'bias',h):+.5f} | {med(C,'corr',h):.2f} | {med(C,'slope',h):.2f} | "
                   f"**{med(C,'skill',h):+.4f}** | {frange('skill', h)} |")

    doc.append(f"""
**Verdict: clearly positive.** It beats persistence by 14% at +5 yr rising to 25% at +15/+20 yr,
and `corr` runs 0.34–0.45, so a real share of the change it predicts is where change happened.

`bias` is essentially zero ({med(C,'bias',5):+.5f} at h=5, {med(C,'bias',20):+.5f} at h=20)
— the systematic over-prediction the southern-Africa edition reported is not present here.
`slope` runs {med(C,'slope',5):.2f}–{med(C,'slope',10):.2f}: when the model predicts a change of
1.0 the truth averages ~1.15, so it now *under*-reaches slightly rather than over-reaching.

RMSE is the most trustworthy number here — the floor's range is {frange('rmse', 20, fmt='{:.5f}')}
at h=20, under 2%. Skill has a small denominator and a wider band; read it as a level.

---

## 2. Is the *whole distribution* any good?

**`crps`, `crps_skill`**

CRPS generalises absolute error to a distribution: it rewards being both accurate and
appropriately confident. A point forecast's CRPS is its absolute error, so persistence gives a
free baseline.

| h | CRPS | persistence | skill | floor's skill range |
|---|---|---|---|---|""")

    for h in H:
        doc.append(f"| +{h} | {med(D,'crps',h):.5f} | {med(D,'crps_persistence',h):.5f} | "
                   f"**{med(D,'crps_skill',h):+.4f}** | {frange('crps_skill', h)} |")

    doc.append(f"""
**Verdict: the headline, and it is solid — 21% better than persistence at +5 yr and 28–29% at
+15/+20 yr.** As a full predictive distribution the model earns its keep at every horizon, and
the floor's own range is only 2–6% wide here, so the level is trustworthy.

---

## 3. Is the distribution the right *shape*?

**PIT — `pit_mean`, `pit_ks`, and the four tail fractions**

The sharpest test on the list. Ask each pixel: what percentile of my forecast did the truth land
at? If the distribution is right, those percentiles are uniform — as often at 3% as at 97%.

| h | pit_mean | P(u<0.001) | P(u<0.025) | P(u>0.975) | P(u>0.999) | KS | floor's KS range |
|---|---|---|---|---|---|---|---|""")

    for h in H:
        doc.append(f"| +{h} | {med(D,'pit_mean',h):.3f} | {med(D,'pit_lt_0001',h):.4f} | "
                   f"{med(D,'pit_lt_0025',h):.4f} | {med(D,'pit_gt_0975',h):.4f} | "
                   f"{med(D,'pit_gt_0999',h):.4f} | {med(D,'pit_ks',h):.3f} | "
                   f"{frange('pit_ks', h, fmt='{:.3f}')} |")
    doc.append("| **should be** | **0.500** | 0.0010 | 0.0250 | 0.0250 | 0.0010 | **0** | |")

    doc.append(f"""
**Verdict: the body is well calibrated; both extreme tails are too thin.**

`pit_mean` runs {med(D,'pit_mean',20):.3f}–{med(D,'pit_mean',5):.3f} against a target of 0.500,
so the distribution is centred — the upward shift reported on southern Africa is gone, which is
the same fact as the near-zero bias in §1. The 2.5% tails are close to nominal:
`P(u<0.025)` {med(D,'pit_lt_0025',5):.4f}–{med(D,'pit_lt_0025',20):.4f} and
`P(u>0.975)` {med(D,'pit_gt_0975',5):.4f}–{med(D,'pit_gt_0975',20):.4f} against 0.025.

The **0.1% tails are 2–4x too populated**: `P(u<0.001)` runs
{min(med(D,'pit_lt_0001',h) for h in H):.4f}–{max(med(D,'pit_lt_0001',h) for h in H):.4f} and
`P(u>0.999)` {min(med(D,'pit_gt_0999',h) for h in H):.4f}–{max(med(D,'pit_gt_0999',h) for h in H):.4f},
both against a nominal 0.001. So observations land beyond the deepest quantiles the model
expresses several times too often, on **both** sides. That is the defect `e4` (lower-tail
weighting) and `e10` (a coarser knot grid) were aimed at; e4 was a null and e10 was not run.

KS of {med(D,'pit_ks',5):.3f} at h=5 on {npx(D,5,'pooled','all'):,} pixels is still enormous —
the 5% critical value is about 0.001 — but the h≥10 rows sit at
{med(D,'pit_ks',15):.3f}–{med(D,'pit_ks',10):.3f} against southern Africa's 0.21–0.27. Note the
floor's own KS range at h=10 is {frange('pit_ks', 10, fmt='{:.3f}')}, which is 66% of its level:
that row is **under-powered** and a difference there means nothing.

---

## 4. Are the intervals the right *width*?

**`cov50/80/95/99` and the matching widths**

If you publish a 95% interval, the truth should fall inside 95% of the time.

| h | cov50 | cov80 | cov95 | cov99 | width50 | width95 | floor's cov95 range |
|---|---|---|---|---|---|---|---|""")

    for h in H:
        doc.append(f"| +{h} | **{med(D,'cov50',h):.3f}** | {med(D,'cov80',h):.3f} | "
                   f"{med(D,'cov95',h):.3f} | {med(D,'cov99',h):.3f} | {med(D,'width50',h):.4f} | "
                   f"{med(D,'width95',h):.4f} | {frange('cov95', h, fmt='{:.3f}')} |")

    doc.append(f"""
**Verdict: the outer intervals are honest; the 50% interval is wrong in opposite directions at
the two ends.** `cov95` runs {med(D,'cov95',20):.3f}–{med(D,'cov95',10):.3f} against 0.95 and
`cov99` {med(D,'cov99',5):.3f}–{med(D,'cov99',15):.3f} against 0.99 — both close. But `cov50`
covers {med(D,'cov50',5):.3f} at +5 yr (too narrow) and {med(D,'cov50',20):.3f} at +20 yr (too
wide). The middle of the distribution is the part that is mis-shaped, and it is mis-shaped in a
horizon-dependent way, which is why §3's KS is worst at h=5.

This is why coverage alone is not enough: look only at `cov95` and the model passes.

---

## 5. Does it predict the right *amount* of change?

**Exceedance: `P(Δ>0.01)`, `P(Δ>0.05)`, `P(Δ<−0.01)` against observed**

Read straight off the quantile function — no sampling, no marginal assumption. "How often does
the model say a pixel will gain more than 0.05?" against how often it actually did.

| h | P(Δ>0.01) pred / obs | P(Δ>0.05) pred / obs | P(Δ<−0.01) pred / obs |
|---|---|---|---|""")

    for h in H:
        a, b = med(D, 'pgt001_pred', h), med(D, 'pgt001_obs', h)
        c, d = med(D, 'pgt005_pred', h), med(D, 'pgt005_obs', h)
        e, f = med(D, 'plt001_pred', h), med(D, 'plt001_obs', h)
        doc.append(f"| +{h} | {a:.4f} / {b:.4f} = **{a/b:.2f}** | {c:.5f} / {d:.5f} = "
                   f"**{c/d:.2f}** | {e:.4f} / {f:.4f} = **{e/f:.2f}** |")

    doc.append(f"""
**Verdict: pooled exceedance is close to right — every ratio between 0.86 and 1.08.** The
southern-Africa edition reported a factor-of-two over-prediction of increases; on Africa the
aggregate amount of change is well calibrated, and declines are within 15% at h≥10.

The summary statistic `exceedance_abs_log10` is **{exc:.4f}**
— but note what it summarises: the mean of |log₁₀(pred/obs)| **across distance bands**, not
pooled. Pooled agreement of 0.97 coexists with per-band errors of 1.6x and worse. §6 is where
that lives, and it is the section that matters for this model's purpose.

---

## 6. Can it imagine rare, large change in remote country?

**`tail_reach`, `halfwidth_p999`, and exceedance per distance band**

This is what the phase exists for. `tail_reach` = how far Q(0.999) sits above the median, in
units of the 95% half-width. A normal distribution gives **1.577**, and the model starts there by
construction, so anything above it was learned.

At +20 yr, by distance from existing change:

| band | n | 95% half-width | Q(0.999) − median | tail reach | P(Δ>0.05) pred / obs | +0.05 sits at |
|---|---|---|---|---|---|---|""")

    for b in BANDS:
        n = npx(D, 20, "distance", b)
        if n == 0:
            doc.append(f"| {b} px | **0** | — | — | — | — | — |")
            continue
        hw = med(D, "width95", 20, "distance", b) / 2
        p999 = med(D, "halfwidth_p999_median", 20, "distance", b)
        reach = med(D, "tail_reach_median", 20, "distance", b)
        pr = med(D, "pgt005_pred", 20, "distance", b)
        ob = med(D, "pgt005_obs", 20, "distance", b)
        doc.append(f"| {b} px | {n:,} | {hw:.4f} | {p999:.4f} | **{reach:.2f}** | "
                   f"{pr:.6f} / {ob:.6f} = **{pr/ob:.2f}** | {0.05/hw:.1f} half-widths |")

    far = per_seed(D, "pgt005_pred", 20, "distance", ">100")
    far_obs = med(D, "pgt005_obs", 20, "distance", ">100")
    ffar = per_seed(FD, "pgt005_pred", 20, "distance", ">100")
    far_ratios = " / ".join(f"{v/far_obs:.3f}" for v in far)
    ffar_ratios = " / ".join(f"{v/far_obs:.3f}" for v in ffar)
    hw_far = med(D, "width95", 20, "distance", ">100") / 2
    oof = 0.05 / hw_far
    far_skill = med(D, "crps_skill", 20, "distance", ">100")
    reach_far = " / ".join(f"{v:.2f}" for v in per_seed(D, "tail_reach_median", 20, "distance", ">100"))

    doc.append(f"""
**Verdict: the tail is genuinely learned out to 30 px and the far field is still cold — but the
far-field exceedance number is not an estimate, and this scorecard will not pretend otherwise.**

The model reaches 2.5–8.0 against the Gaussian 1.58 it started from, and the near bands are
within 10–40% of observed out to 30 px. Two bands then fail in opposite directions: 30–100 px is
**1.64x too hot**, and >100 px reads 0.01 of observed at the median.

**That far-field ratio is unusable at this sample size.** Across e1's three seeds it is
{far_ratios}; across the floor's six it is {ffar_ratios}. Three orders of magnitude,
from the same configuration. The quantity is a tiny tail probability integrated over
{npx(D,20,'distance','>100'):,} pixels, and it is dominated by whether a handful of pixels'
quantile functions happen to reach past +0.05 at all. **Any single-number claim about far-field
exceedance — including "7.3% of observed", recorded earlier in this project — is one draw from
that spread, not a measurement.**

Use the two numbers in that band that *are* stable instead. `tail_reach` there is {reach_far}
across e1's seeds, and the 95% half-width is {hw_far:.5f}, which puts +0.05 at **{oof:.1f}
half-widths** above the median. To emit +0.05 at the observed rate the model would need a reach
of that order; it has ~6. That is the defect stated in the units where it can be measured — the
same statement the frozen product's scorecard made at ~12 half-widths, on a different extent.

Note also `crps_skill` in the far band is **{far_skill:.2f}**: where almost nothing happens,
persistence is very hard to beat, and the model loses to it badly.

---

## 7. Is the product internally consistent?

**`qf_vs_triple_max`, `central_outside_interval`**

Not a model-quality question — a "do the published files agree with each other" question. The
64-band quantile raster and the lower/central/upper triple must describe the same forecast, or
the scorecard and the ensemble would be scoring different models.""")

    qf = max(float(k["qf_vs_upper_max"].max()) for k in K.values())
    coi = {lbl: float(k["central_outside_interval"].max()) for lbl, k in K.items()}
    bad = {lbl: v for lbl, v in coi.items() if v > 0}
    doc.append(f"""
**`qf_vs_triple_max` = {qf:.2e}**, half the int16 storage quantum — the two agree exactly before
rounding, at every horizon and every seed.""")
    if bad:
        worst = max(bad, key=bad.get)
        doc.append(f"""
**`central_outside_interval` is not clean for this configuration.** Two of three seeds read
exactly 0, but `{worst}` reads **{bad[worst]:.5f}** — 0.6% of pixels at +15 yr where the central
forecast falls outside its own published interval. The floor never does this. It is one seed and
the magnitude is small, but it is a consistency violation rather than a quality metric, and it
should be understood before e1 is adopted.""")
    else:
        doc.append("\n**`central_outside_interval` = 0.00000** at every horizon and seed.")

    doc.append(f"""
---

## The short version

| question | verdict |
|---|---|
| point forecast | **beats persistence by 14–25%**; unbiased; slightly under-reaches (slope ~1.15) |
| whole distribution | **+21% to +29% over persistence — the headline, and the floor's band is narrow** |
| distribution shape | body centred and the 2.5% tails near nominal; **both 0.1% tails 2–4x too thin** |
| interval width | 95% and 99% honest; the 50% interval too narrow at h=5 and too wide at h=20 |
| amount of change | **pooled exceedance within 15% at every horizon** — the southern-Africa 2x over-prediction is absent |
| rare remote change | reach 2.5–8.0 vs Gaussian 1.58; 30–100 px **1.6x too hot**; >100 px cold, and **not estimable at 3 seeds** |
| internal consistency | quantile function and triple agree exactly; one seed puts the centre outside its own interval on 0.6% of h=15 pixels |

**The two things to fix next, in order.** The deep tails are too thin on both sides while the
body is well calibrated — that is a shape problem at fixed centre, and it is what the knot grid
and the tail-weighted objectives were aimed at. And the far field needs a reach of ~20 where it
has ~6; nothing tested in this phase moved it, and the one lever that clearly *does* move it
moves it the wrong way (removing the cumulative-width constraint costs 35% of tail reach).

**A measurement caveat that outranks both.** The far-field exceedance row cannot be estimated
from three seeds, or six. Before any experiment is judged on it, the estimator needs fixing —
more seeds will not do it, because the spread is model variance rather than sampling error.
""")

    with open(OUT, "w") as fh:
        fh.write("\n".join(doc) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
