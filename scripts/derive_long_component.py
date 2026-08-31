#!/usr/bin/env python3
"""Re-derive a spectral_fits.json at a different --long_weight, without refitting.

`add_long_scale_component` is a pure post-hoc rescale of the NNLS solution: it multiplies every
fitted weight and the nugget by ``1 - long_weight`` and appends ``long_weight`` at
``long_range``. Nothing about the fit depends on it. So two variants that differ only in that
constant share one fit, and refitting the second is pure waste — which matters here because a
PIT-space fit reads a 8 GB quantile-function raster per window and costs the better part of an
hour on the Africa grid.

Input must be a fit written with ``--long_weight 0`` (the raw NNLS solution). Refuses anything
else rather than compounding two rescalings.

    scripts/derive_long_component.py --src fits_lw0.json --out fits_lw040.json --long_weight 0.40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fit_field_spectra import add_long_scale_component  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="spectral_fits.json fitted at --long_weight 0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--long_weight", type=float, required=True)
    ap.add_argument("--long_range", type=float, default=1500.0)
    args = ap.parse_args(argv)

    blob = json.load(open(args.src))
    if blob.get("long_weight", 0.0) != 0.0:
        raise SystemExit(f"{args.src} was written with long_weight="
                         f"{blob.get('long_weight')}; this needs the raw NNLS fit (0.0)")
    for h, fit in blob["by_horizon"].items():
        if "long_scale" in fit:
            raise SystemExit(f"{args.src} horizon {h} already carries a long_scale component")

    print(f"{'h':>4} {'nugget':>8} {'sum(w)':>8}  ->  {'nugget':>8} {'sum(w)':>8} {'long':>6}")
    for h in sorted(blob["by_horizon"], key=int):
        fit = blob["by_horizon"][h]
        before = (fit["nugget"], sum(fit["weights"]))
        new = add_long_scale_component(fit, args.long_range, args.long_weight)
        # Everything the fitter attached that is not part of the mixture rides along.
        for k in ("observed_band_shares", "band_labels", "kernel", "nu"):
            if k in fit:
                new[k] = fit[k]
        blob["by_horizon"][h] = new
        print(f"{h:>4} {before[0]:>8.4f} {before[1]:>8.4f}  ->  "
              f"{new['nugget']:>8.4f} {sum(new['weights']):>8.4f} "
              f"{new['long_scale']['weight']:>6.2f}")
        tot = sum(new["weights"]) + new["nugget"]
        if abs(tot - 1.0) > 1e-9:
            raise SystemExit(f"horizon {h}: weights + nugget = {tot}, not 1")

    blob["long_weight"] = args.long_weight
    blob["long_range"] = args.long_range
    blob["derived_from"] = str(args.src)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(blob, open(args.out, "w"), indent=2)
    print(f"\n✓ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
