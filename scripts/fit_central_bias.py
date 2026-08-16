#!/usr/bin/env python3
"""The central forecast's conditional bias, and a per-class correction for it.

The central head over-predicts change, and it does so in proportion to how much change it
predicts. Median standardized residual by predicted-change class:

    dhat class        h=5    h=10    h=15    h=20
    (-0.01,0.001]   -0.03   -0.02   -0.02   -0.03
    (0.001,0.01]    -0.09   -0.11   -0.21   -0.28
    (0.01,0.05]     -0.47   -0.29   -0.29   -0.30
    (0.05,0.15]         -       -   -0.79   -0.60

Where the model predicts 5-15% change it lands 0.6-0.8 half-widths high. **No interval width
can absorb a bias**, and trying is what makes the width factors trade against each other: a
factor fitted to restore coverage around a displaced centre has to be asymmetric (k_lo 1.07,
k_up 0.62), which fixes T1.2/T1.3 and chills the upper tail; a factor fitted symmetrically
ignores the displacement and breaks coverage. Correcting the centre lets the width factor do
the job it is actually for.

**This regenerates the central raster**, which the rest of the pipeline treats as inviolable
(`apply_recalibration.py` copies it byte for byte precisely so "the central forecast is
unchanged" needs no verifying). Doing it is a deliberate, flag-gated choice.

Two guards, because a post-hoc shift fitted on the residuals it is scored against is exactly
the kind of thing that flatters a number:

* the class is assigned from the **original** dhat, so the correction is a function of what
  the model predicted, not of what it predicts after being corrected — otherwise the
  assignment moves under its own output;
* a class whose bias does not agree in sign across the available windows is left alone. The
  h=5 (0.01,0.05] class reads -0.0099 / +0.0016 / -0.0037 / -0.0091 across windows; there is
  no stable bias there to correct, only noise to fit. Where the bias is real it is steady —
  h=15 (0.05,0.15] reads -0.0287 and -0.0255 on its two windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.validate import DHAT_BINS, DHAT_LABELS  # noqa: E402

N_DHAT = len(DHAT_LABELS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_count", type=int, default=2_000)
    ap.add_argument("--require_sign_agreement",
                    type=lambda x: (str(x).lower() == "true"), nargs="?",
                    const=True, default=True,
                    help="Zero the correction for classes whose windows disagree in sign.")
    args = ap.parse_args(argv)

    man = pd.read_csv(args.manifest)
    per_window: dict[tuple[int, int, int], float] = {}
    pooled: dict[tuple[int, int], list] = {}
    for _, row in man.iterrows():
        h, w = int(row["horizon"]), int(row["base_year"])
        with rasterio.open(row["path_res_native"]) as s:
            res = s.read(1).astype(np.float64)
        with rasterio.open(row["path_dhat"]) as s:
            dhat = s.read(1).astype(np.float64)
        ok = np.isfinite(res) & np.isfinite(dhat)
        d_idx = np.digitize(dhat, DHAT_BINS[1:-1])
        for d in range(N_DHAT):
            m = ok & (d_idx == d)
            if int(m.sum()) < args.min_count:
                continue
            per_window[(h, d, w)] = float(np.median(res[m]))
            pooled.setdefault((h, d), []).append(res[m])

    out: dict[str, dict] = {"by_horizon": {}}
    print(f"{'h':>3} {'dhat class':>15} {'n':>12} {'shift':>10} {'windows':>9}  status")
    n_applied = n_zeroed = 0
    for (h, d) in sorted(pooled):
        vals = np.concatenate(pooled[(h, d)])
        shift = float(np.median(vals))
        wins = [v for (hh, dd, _), v in per_window.items() if (hh, dd) == (h, d)]
        stable = len(wins) < 2 or (min(wins) < 0) == (max(wins) < 0)
        if args.require_sign_agreement and not stable:
            shift, status = 0.0, "windows disagree in sign — left alone"
            n_zeroed += 1
        else:
            status = "applied" if len(wins) > 1 else "applied (single window)"
            n_applied += 1
        out["by_horizon"].setdefault(str(h), {})[str(d)] = shift
        print(f"{h:>3} {DHAT_LABELS[d]:>15} {vals.size:>12,} {shift:>10.5f} "
              f"{len(wins):>9}  {status}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n✓ {args.out} — {n_applied} classes corrected, {n_zeroed} left alone. "
          f"The shift is the median of (observed - central) and is ADDED to the central "
          f"forecast, so a negative value moves an over-prediction down toward the "
          f"observation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
