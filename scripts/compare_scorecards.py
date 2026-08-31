#!/usr/bin/env python3
"""Row-by-row diff of ensemble scorecards, including the rows that did not move.

A pass count hides the thing worth knowing. Two settings have tied on count here before
while one had quietly pushed near-nominal rows below target, so this prints every row's
*value* alongside the baseline's and lets the eye do the ranking. Rows that did not move are
printed too: "T8 is unchanged" is a result about a correlation variant, not an omission.

Keyed on ``(id, metric)``, which is unique on this card -- ``id`` alone is not, since T2.8
carries twelve rows and T8.1 six.

    scripts/compare_scorecards.py \\
        --baseline base=data/ensemble/exp/dist_baseline_af_e1/scorecard_qf/scorecard.csv \\
        --patch data/ensemble/exp/dist_baseline_af_e1/scorecard_qf_t4fix/scorecard.csv \\
        --variant v1=data/ensemble/exp/dist_v1_long000/validation/scorecard.csv

``--patch`` overwrites baseline rows from a partial re-score. T4.1/T4.2 on the frozen card
were computed by inverting a two-piece normal the members were never drawn from
(``stage_temporal`` ignored ``--qf_dir``); the corrected rows live in their own directory and
this is how they get into the comparison without touching the frozen card.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    if df.groupby(["id", "metric"]).ngroups != len(df):
        raise SystemExit(f"{path}: (id, metric) is not unique")
    return df.set_index(["id", "metric"])


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def flag(p):
    if isinstance(p, str):
        p = p.strip().lower() in ("true", "1")
    elif pd.isna(p):
        return "  -"
    return " ok" if p else "FAIL"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="label=path/to/scorecard.csv")
    ap.add_argument("--patch", action="append", default=[],
                    help="scorecard.csv whose rows replace the baseline's (partial re-score)")
    ap.add_argument("--variant", action="append", default=[], help="label=path, repeatable")
    ap.add_argument("--moved_only", action="store_true",
                    help="Print only rows where some variant moved. Off by default: the "
                         "unmoved rows are half the finding.")
    ap.add_argument("--rel", type=float, default=0.02,
                    help="Relative change counted as movement (default 2%%)")
    args = ap.parse_args(argv)

    blabel, bpath = args.baseline.split("=", 1)
    base = load(bpath)
    for p in args.patch:
        patch = load(p)
        base.loc[patch.index] = patch
        print(f"# baseline rows patched from {p}: {', '.join(sorted({i for i, _ in patch.index}))}")

    variants = []
    for spec in args.variant:
        lab, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"# {lab}: {path} not present yet — skipped")
            continue
        variants.append((lab, load(path)))
    if not variants:
        raise SystemExit("no variant scorecards found")

    w = 58
    head = f"{'row':<{w}} {blabel:>12} {'':>4}"
    for lab, _ in variants:
        head += f" | {lab:>12} {'Δ':>10} {'':>4}"
    print("\n" + head)
    print("-" * len(head))

    moved_rows, seen = [], set()
    last_prefix = None
    for (rid, metric), row in base.iterrows():
        prefix = rid.split(".")[0]
        b = num(row["value"])
        cells, any_move = [], False
        for lab, v in variants:
            if (rid, metric) not in v.index:
                cells.append(f" | {'absent':>12} {'':>10} {'':>4}")
                continue
            vr = v.loc[(rid, metric)]
            x = num(vr["value"])
            d = x - b
            rel = abs(d) / max(abs(b), 1e-12) if np.isfinite(b) and np.isfinite(d) else np.nan
            if np.isfinite(rel) and rel >= args.rel:
                any_move = True
            ds = f"{d:+10.4f}" if np.isfinite(d) else " " * 10
            xs = f"{x:12.4f}" if np.isfinite(x) else f"{str(vr['value'])[:12]:>12}"
            cells.append(f" | {xs} {ds} {flag(vr['pass']):>4}")
        if args.moved_only and not any_move:
            continue
        if prefix != last_prefix:
            print()
            last_prefix = prefix
        bs = f"{b:12.4f}" if np.isfinite(b) else f"{str(row['value'])[:12]:>12}"
        label = f"{rid} {metric}"
        print(f"{label[:w]:<{w}} {bs} {flag(row['pass']):>4}" + "".join(cells))
        seen.add((rid, metric))
        if any_move:
            moved_rows.append((rid, metric))

    print("\n" + "=" * len(head))
    bp = base["pass"].map(lambda p: p if isinstance(p, bool) else
                          (str(p).strip().lower() == "true" if pd.notna(p) else None))
    nb = int(sum(1 for x in bp if x is True))
    nbf = int(sum(1 for x in bp if x is False))
    print(f"{blabel:>26}: {nb} pass / {nbf} fail / {len(base) - nb - nbf} unscored "
          f"of {len(base)}")
    for lab, v in variants:
        vp = v["pass"].map(lambda p: p if isinstance(p, bool) else
                           (str(p).strip().lower() == "true" if pd.notna(p) else None))
        np_, nf = int(sum(1 for x in vp if x is True)), int(sum(1 for x in vp if x is False))
        gained = [f"{i} {m}" for (i, m), r in v.iterrows()
                  if (i, m) in base.index
                  and str(r["pass"]).strip().lower() == "true"
                  and str(base.loc[(i, m), "pass"]).strip().lower() == "false"]
        lost = [f"{i} {m}" for (i, m), r in v.iterrows()
                if (i, m) in base.index
                and str(r["pass"]).strip().lower() == "false"
                and str(base.loc[(i, m), "pass"]).strip().lower() == "true"]
        print(f"{lab:>26}: {np_} pass / {nf} fail / {len(v) - np_ - nf} unscored of {len(v)}"
              f"   ({len(gained)} gained, {len(lost)} lost)")
        for g in gained:
            print(f"{'  + ':>28}{g}")
        for l in lost:
            print(f"{'  - ':>28}{l}")
    print(f"\n{len(moved_rows)} of {len(base)} rows moved by >= {100*args.rel:.0f}%; "
          f"{len(base) - len(moved_rows)} did not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
