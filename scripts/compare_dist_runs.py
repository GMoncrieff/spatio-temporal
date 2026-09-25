#!/usr/bin/env python3
"""Rank distributional runs against the baseline's own measured floor.

The bar is encoded here rather than applied by eye, because applying it by eye is how the
previous model phase produced two false positives. From ``docs/background/model_phase.md``
§6.1: a single new draw falls outside the range of three base runs roughly **half the time
under the null**, so "outside the base range" is barely a test. The bar used here is that

    the margin beyond the baseline range must exceed that range's own width,

and anything that clears it is replicated before it means anything.

Two further rules from the same record are enforced:

* **Judge on per-row values, never a total count.** Two settings once tied on a pass count
  while one had quietly pushed near-nominal rows below target, so every metric is printed.
* **A floor is a property of the configuration.** If fewer than two baseline replicates are
  present this refuses to rank rather than borrowing a number from somewhere else.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

# (metric, better direction). CRPS skill and central skill are up; every calibration error
# is down; coverage and tail reach are reported without a direction because "better" for them
# depends on a target rather than an extreme.
METRICS = [
    ("crps_skill5", "up"), ("crps_skill10", "up"),
    ("crps_skill15", "up"), ("crps_skill20", "up"),
    ("pit_ks5", "down"), ("pit_ks10", "down"),
    ("pit_ks15", "down"), ("pit_ks20", "down"),
    ("exceedance_abs_log10", "down"),
    ("skill5", "up"), ("skill20", "up"),
    ("rmse5", "down"), ("rmse20", "down"),
    ("cov95_5", "none"), ("cov95_20", "none"),
    ("tail_reach20", "none"),
]
# A band wider than this fraction of the metric's own level cannot resolve anything a single
# run is likely to do. Marked rather than hidden: an under-powered row that reports "nothing
# clears" is not evidence of no effect, and this project has already mistaken one for the
# other. CLAUDE.md rule 5.
UNDERPOWERED = 0.25
# The other end of the same problem. A baseline band narrower than this fraction of its own
# level has not measured the configuration's run-to-run floor; it has measured a metric the
# seeds did not happen to move. Differences on such a row are reported as "floor unmeasured"
# rather than as wins, because "the margin exceeded a band of width zero" is not a finding.
DEGENERATE_REL = 1e-4
REPORTED = ["cov95_5", "cov95_20", "pit_gt_0999_20", "tail_reach20",
            "qf_vs_triple_max", "central_outside_interval"]


def collapse_seeds(df: pd.DataFrame, pattern: str = r"_s\d+$"):
    """Collapse replicate rows to one row per configuration, metric by metric.

    A three-seed *range* is not a statistic: on the six-seed Africa floor a three-seed range
    swings 17.5x across draws from one configuration, against 1.1x for the median. So a
    variant is ranked on its median, and the median is taken independently per metric -- no
    single seed's row is selected, because a run can be the middle one on ``crps_skill5`` and
    the extreme one on ``tail_reach20``.
    """
    base = df.index.to_series().str.replace(pattern, "", regex=True)
    med = df.select_dtypes(include="number").groupby(base).median()
    return med, df.groupby(base).size()


def load(score_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(score_dir, "summary_*.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    if not rows:
        raise SystemExit(f"no summary_*.json under {score_dir}")
    return pd.DataFrame(rows).set_index("label")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score_dir", default="data/ensemble/exp/dist_scores")
    ap.add_argument("--baseline", default=r"^d0_s\d+$",
                    help="Regex selecting the baseline replicates whose spread is the floor.")
    ap.add_argument("--include", default=None,
                    help="Regex a variant label must match to be ranked. The floor is a "
                         "property of the configuration AND of the extent it was measured "
                         "on, so use this to keep another region's runs out of the table.")
    ap.add_argument("--group_seeds", action="store_true",
                    help="Rank each configuration on the MEDIAN of its replicates rather "
                         "than one row per seed.")
    args = ap.parse_args(argv)

    df = load(args.score_dir)
    base = df[[bool(re.match(args.baseline, str(i))) for i in df.index]]
    others = df[[not bool(re.match(args.baseline, str(i))) for i in df.index]]

    if args.include:
        others = others[[bool(re.search(args.include, str(i))) for i in others.index]]
        if others.empty:
            raise SystemExit(f"--include {args.include!r} matched no variant")
    per_seed = others
    n_seeds = pd.Series(1, index=others.index, dtype=int)
    if args.group_seeds:
        others, n_seeds = collapse_seeds(others)

    print(f"baseline replicates: {list(base.index)}")
    if args.group_seeds:
        print("variants ranked on the median of: "
              + ", ".join(f"{k} (n={v})" for k, v in n_seeds.items()))
    if len(base) < 2:
        raise SystemExit(
            "fewer than two baseline replicates: the floor is a property of THIS "
            "configuration and cannot be borrowed. Run scripts/run_dist_floor.sh first.")

    print("\n" + "=" * 104)
    print(f"{'metric':<22}{'base min':>11}{'base max':>11}{'width':>10}{'rel':>8}{'power':>7}"
          f"   variants beyond the bar")
    print("=" * 104)
    verdicts = {}
    for metric, direction in METRICS:
        if metric not in df.columns:
            continue
        vals = base[metric].dropna().to_numpy(dtype=float)
        if vals.size < 2:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        width = hi - lo
        level = float(np.mean(np.abs(vals))) or 1.0
        rel = width / level
        powered = rel <= UNDERPOWERED
        # A band that is *too narrow* is as unreadable as one that is too wide, and it fails
        # in the more dangerous direction: with width ~0 every variant clears "margin exceeds
        # the band's own width", so a metric the seeds happen not to move declares wins for
        # everything. That is not infinite sensitivity, it is an unmeasured floor. Added for
        # the conv-spline gates, several of which can sit at an identical value across three
        # seeds while still differing between configurations.
        degenerate = width <= 0 or rel < DEGENERATE_REL
        hits = []
        for label, v in others[metric].items():
            if not np.isfinite(v) or direction == "none":
                continue
            # The bar is two-sided and the same size on both sides: a value must sit more
            # than one floor-width past the floor's edge *in the direction being claimed*.
            # It was not always: WORSE used to be measured from the floor's BEST edge, which
            # made it fire one whole width earlier than "better" did. Of the eight WORSE
            # verdicts on the Africa slate under that rule, seven sat 0.04x-0.78x past the
            # floor -- inside the margin the other side of the same test has to exceed.
            better = (v - hi) if direction == "up" else (lo - v)
            worse = (lo - v) if direction == "up" else (v - hi)
            if degenerate:
                if v != lo or v != hi:
                    hits.append(f"{label} {v:.5f} (floor unmeasured)")
                    verdicts.setdefault(label, []).append(f"{metric} moved (floor unmeasured)")
            elif better > width:
                hits.append(f"{label} {v:.5f} (+{better / width:.1f}x)")
                if powered:
                    verdicts.setdefault(label, []).append(f"{metric} better")
                else:
                    verdicts.setdefault(label, []).append(f"{metric} better (weak row)")
            elif worse > width:
                tag = f"{metric} WORSE ({worse / width:.1f}x)"
                verdicts.setdefault(label, []).append(
                    tag if powered else tag + " (weak row)")
        note = "-" if direction == "none" else (", ".join(hits) if hits else "-")
        print(f"{metric:<22}{lo:>11.5f}{hi:>11.5f}{width:>10.5f}{rel:>7.0%}"
              f"{'DEGEN' if degenerate else ('  ok' if powered else ' WEAK'):>7}   {note}")

    print("\n" + "=" * 96)
    print("every run, every metric (read the rows, not a count)")
    print("=" * 96)
    cols = [m for m, _ in METRICS if m in df.columns] + [c for c in REPORTED if c in df.columns]
    table = pd.concat([base[cols], per_seed[cols]])
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(table.to_string(float_format=lambda v: f"{v:.5f}"))
        if args.group_seeds:
            print("\nthe same variants, collapsed to the median that was judged")
            print(others.reindex(columns=cols).to_string(float_format=lambda v: f"{v:.5f}"))

    print("\n" + "=" * 96)
    print("verdicts (a variant must clear the bar on a metric that matters, and replicate)")
    print("=" * 96)
    for label in others.index:
        notes = verdicts.get(label, [])
        print(f"  {label:<12} {'; '.join(notes) if notes else 'nothing clears the bar'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
