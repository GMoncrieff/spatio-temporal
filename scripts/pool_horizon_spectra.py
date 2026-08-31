#!/usr/bin/env python3
"""Average a fitted spectrum across horizons into one shared spatial covariance.

A separable space-horizon model is ``Cov{Z_h(s), Z_h'(s')} = C_space(d) . R_hh'`` — one
spatial correlation function, shared by every horizon. Feeding it four *different* per-horizon
spectra would reintroduce exactly the cross-horizon spectral mixing it exists to remove, so
the shared ``C_space`` has to be built first.

The review's §5.4 asks for the assumption to be checked rather than assumed. On the e1 Africa
PIT fits the four horizons put 0.618 / 0.674 / 0.697 / 0.728 of variance on the 4-50 px band —
a spread of 0.110 against a realisation-noise floor of 0.044, so they are not identical, but
they are the same shape to within about twice that floor. This prints the spread it is
averaging over so the cost is visible rather than assumed away.

    scripts/pool_horizon_spectra.py --src pit_lw0.json --out pit_lw0_pooled.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from fit_field_spectra import RESOLVABLE_PX, weight_shares  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    blob = json.load(open(args.src))
    by_h = blob["by_horizon"]
    hs = sorted(by_h, key=int)
    ranges = by_h[hs[0]]["ranges_px"]
    for h in hs:
        if by_h[h]["ranges_px"] != ranges:
            raise SystemExit(f"horizon {h} uses a different basis; cannot pool")

    W = np.array([by_h[h]["weights"] for h in hs], dtype=float)
    N = np.array([by_h[h]["nugget"] for h in hs], dtype=float)

    lo, hi = RESOLVABLE_PX
    print(f"{'h':>4} {'nugget':>8} {'<%gpx' % lo:>8} {'%g-%gpx' % (lo, hi):>9} {'>%gpx' % hi:>8}")
    sh = np.array([weight_shares(ranges, W[i], N[i]) for i in range(len(hs))])
    for i, h in enumerate(hs):
        print(f"{h:>4} " + " ".join(f"{v:>8.4f}" for v in sh[i]))
    print(f"{'spread':>4} " + " ".join(f"{v:>8.4f}" for v in sh.max(axis=0) - sh.min(axis=0))
          + "   <- what pooling averages away (noise floor 0.044)")

    w, n = W.mean(axis=0), float(N.mean())
    tot = w.sum() + n
    w, n = w / tot, n / tot
    pooled = {"ranges_px": ranges, "weights": [float(x) for x in w], "nugget": n,
              "kernel": by_h[hs[0]].get("kernel", "matern"), "nu": by_h[hs[0]].get("nu", 0.5),
              "pooled_from": [int(h) for h in hs]}
    print(f"\n{'pooled':>4} " + " ".join(f"{v:>8.4f}" for v in weight_shares(ranges, w, n)))

    blob["by_horizon"] = {h: dict(pooled) for h in hs}
    blob["pooled_horizons"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(blob, open(args.out, "w"), indent=2)
    print(f"\n✓ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
