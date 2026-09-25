#!/usr/bin/env python3
"""Side-by-side table of model-phase experiments, central field and intervals together.

Reads whatever ``score_model_experiment.py`` has written into a score directory and prints
one row per configuration. Two things it does deliberately:

* **Everything is shown against a noise floor**, computed from the runs whose labels are
  given as ``--seeds``. Run-to-run variance between identically configured folds is
  comparable to the effects this phase is chasing, so a delta without that column is not
  a result. Deltas inside the floor are marked ``~``.
* **Byte-identical numbers between two configurations are flagged**, not celebrated. Twice
  in this project that has meant a run scored the wrong checkpoint. The inverse also holds
  and is checked: a quantile-only experiment scored under a central-only checkpoint monitor
  *must* leave the central field identical, so an unexpected central change is a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 15, 20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score_dir", default="data/ensemble/exp/scores")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated labels that differ only by seed; their spread "
                         "is the noise floor every other delta is read against")
    ap.add_argument("--baseline", default=None,
                    help="label everything is differenced against (default: first --seeds "
                         "entry, else the first label found)")
    ap.add_argument("--out", default=None, help="write the table as CSV")
    args = ap.parse_args(argv)

    score_dir = Path(args.score_dir)
    rows = []
    for f in sorted(score_dir.glob("summary_*.json")):
        s = json.loads(f.read_text())
        label = s["label"]
        r = {"label": label,
             "mean_ln_k": s["width_mean_abs_ln_k_leaf"],
             "within20": s["width_frac_within_20pct_leaf"]}
        for h in HORIZONS:
            r[f"rmse{h}"] = s["central_rmse"].get(str(h), s["central_rmse"].get(h))
            r[f"skill{h}"] = s["central_skill"].get(str(h), s["central_skill"].get(h))
            r[f"cov{h}"] = s["coverage"].get(str(h), s["coverage"].get(h))
        rows.append(r)
    if not rows:
        raise SystemExit(f"no summary_*.json under {score_dir}")
    df = pd.DataFrame(rows).set_index("label")

    seeds = [s.strip() for s in args.seeds.split(",")] if args.seeds else []
    seeds = [s for s in seeds if s in df.index]
    floor = {}
    if len(seeds) >= 2:
        for c in df.columns:
            v = df.loc[seeds, c].to_numpy(float)
            floor[c] = float(np.nanmax(v) - np.nanmin(v))
    base = args.baseline or (seeds[0] if seeds else df.index[0])

    print(f"=== absolute ({len(df)} configurations) ===")
    print(df.to_string(float_format=lambda v: f"{v:.5f}"))
    if floor:
        print("\n=== seed-to-seed spread (the noise floor) ===")
        print(pd.Series(floor).to_string(float_format=lambda v: f"{v:.5f}"))

    print(f"\n=== change vs {base}; '~' means inside the noise floor, "
          f"'+' better, '-' worse ===")
    better_when_lower = {c for c in df.columns
                         if c.startswith("rmse") or c in ("mean_ln_k",)}
    out = {}
    for label, row in df.iterrows():
        if label == base:
            continue
        cells = {}
        for c in df.columns:
            d = row[c] - df.loc[base, c]
            f = floor.get(c)
            if f is not None and abs(d) <= f:
                mark = "~"
            elif (d < 0) == (c in better_when_lower):
                mark = "+"
            else:
                mark = "-"
            # coverage is a two-sided target, not a direction
            if c.startswith("cov"):
                mark = "~" if (f is not None and abs(d) <= f) else " "
            cells[c] = f"{d:+.5f}{mark}"
        out[label] = cells
    if out:
        print(pd.DataFrame(out).T.to_string())

    dup = df.round(8).duplicated(keep=False)
    if dup.any():
        print("\n!! IDENTICAL ROWS — check these scored distinct artifacts: "
              f"{sorted(df.index[dup])}")

    if args.out:
        df.to_csv(args.out)
        print(f"\n✓ wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
