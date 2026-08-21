#!/usr/bin/env python3
"""Did the smoke run's scorecard come back complete, and is every number a number?

The smoke tier scores four members, so pass rates are meaningless and are deliberately
not checked. What is checked is everything that a scale-and-shape bug breaks:

* every stage emitted the rows it owns, rather than dying and leaving the rest to run;
* no stage recorded the ``stage '<name>' completed`` failure marker;
* no scored row carries a non-finite value -- a NaN reaching a hard gate is how the
  antimeridian seam scored as a discontinuity for a whole global run;
* every requested block scale produced its T2.1 rows, since the 1000 km path exists on no
  regional card and would otherwise first run inside a 17-hour stage.

Exits non-zero on any of those, so it can gate a long run from a shell script.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Stage -> the scorecard id prefixes it is responsible for. A stage that runs and emits
# nothing is as much a failure as one that raises.
STAGE_IDS = {
    "gates": ("T5.",),
    "percentiles": ("T1.",),
    "aggregate": ("T2.1", "T2.2", "T2.3", "T2.4"),
    "rank": ("T2.5",),
    "spatial": ("T3.",),
    "temporal": ("T4.",),
    "change": ("T6.",),
    "visual": ("T7.",),
    "clustering": ("T8.",),
}


# Word-boundary so "nanometre"-shaped words in a note cannot trip it; the columns scanned
# are value and target, which hold numbers and target expressions, not prose.
NONFINITE_RE = re.compile(r"(?<![A-Za-z0-9_])(nan|[+-]?inf(inity)?)(?![A-Za-z0-9_])", re.I)


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check(df, stages, block_sizes):
    problems = []

    # 1. Stage failure markers, written by validate_ensemble's _run_stage.
    for _, r in df.iterrows():
        if str(r["metric"]).startswith("stage '") and "completed" in str(r["metric"]):
            problems.append(f"stage failed: {r['id']} — {r['note']}")

    # 2. Every requested stage's rows are present.
    ids = df["id"].astype(str)
    for stage in stages:
        prefixes = STAGE_IDS.get(stage)
        if not prefixes:
            continue
        if not ids.str.startswith(tuple(prefixes)).any():
            problems.append(f"stage '{stage}' produced no rows (expected {prefixes})")

    # 3. Non-finite values on rows that carry a verdict. Reported-only rows are allowed to
    #    hold text; a row that was scored is not. The string scan is not belt-and-braces:
    #    the global card's T3.5 failed as a hard gate on the text
    #    "mean |Δ| across seam nan vs interior 0.02983", where the NaN never reaches the
    #    `value` column as a float and a numeric check alone reads the row as healthy.
    scored = df[df["pass"].notna()]
    for _, r in scored.iterrows():
        v = _as_float(r["value"])
        if v is not None and not np.isfinite(v):
            problems.append(f"non-finite value on a scored row: {r['id']} {r['metric']} "
                            f"= {r['value']}")
            continue
        for col in ("value", "target"):
            if NONFINITE_RE.search(str(r[col])):
                problems.append(f"non-finite text in '{col}' on a scored row: "
                                f"{r['id']} {r['metric']} = {r[col]!r}")

    # 4. Each requested block scale reached T2.1.
    t21 = ids[ids == "T2.1"]
    if len(t21):
        metrics = " ".join(df.loc[t21.index, "metric"].astype(str))
        for B in block_sizes:
            if f"{B}km" not in metrics:
                problems.append(f"block scale {B} km produced no T2.1 rows")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecard", required=True)
    ap.add_argument("--stages",
                    default="gates,percentiles,aggregate,rank,spatial,temporal,change,"
                            "visual,clustering")
    ap.add_argument("--block_sizes", default="1,10,100,1000")
    args = ap.parse_args(argv)

    path = Path(args.scorecard)
    if not path.exists():
        print(f"✗ no scorecard at {path}")
        return 1
    df = pd.read_csv(path)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    block_sizes = [int(b) for b in args.block_sizes.split(",") if b.strip()]

    problems = check(df, stages, block_sizes)
    n_scored = int(df["pass"].notna().sum())
    print(f"{len(df)} rows, {n_scored} scored, {len(df) - n_scored} reported-only")
    if problems:
        print(f"\n✗ smoke tier found {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("✓ every stage ran, every scored value is finite, every block scale reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
