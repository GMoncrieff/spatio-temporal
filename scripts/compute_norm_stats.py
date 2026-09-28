#!/usr/bin/env python3
"""Exact, land-only normalisation statistics: the JSON sidecar that training and prediction read.

    python scripts/compute_norm_stats.py --out data/conv_spline/norm_stats_E2c.json

E2a's sidecar (``data/conv_spline/norm_stats.json``) came from ``HumanFootprintChipDataset.
_estimate_norm_stats``, which samples random 128 px windows over the WHOLE grid. That has
two consequences:

* the ocean enters every static channel's moments -- climate values, zeros -- though the
  model never trains or predicts there. 73% of the grid is not land.
* ``hm_static_ele_1000.tiff`` declares no nodata and fills every row south of 56 S (9.09%
  of the grid) with -32768, so elevation came out at mean -4200 m, std 11184 m. Over land it
  is 675 m and 845 m, and land elevations were squeezed into a narrow standardised range.

This script replaces the sampling with one streaming pass over each raster, keeping a pixel
only where HM is land (``src.land.hm_land``, the same predicate the fold mask uses) and the
channel's own value is not its declared nodata, NaN, or an undeclared fill named in
``torchgeo_dataloader.STATIC_NODATA``. Moments are merged block by block (Chan et al.), so
the row band is an identity (tests/test_compute_norm_stats.py). The std is the population
std plus 1e-8, as the sampled estimator's ``np.nanstd(...) + 1e-8`` was.

The sidecar also DECLARES the fill table (``static_nodata``), and the readers apply whatever
the sidecar declares -- so the input treatment travels with the checkpoints. E2a's sidecar
declares nothing and reads exactly as it trained.

HM and each component are taken over every year in ``torchgeo_dataloader.years``, each year
masked by that year's own HM land; the statics are masked by ``--land_ref`` (HM 2020).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.land import hm_land  # noqa: E402


def _moments(job):
    """(value raster, land raster, undeclared fill, block rows) -> streamed land moments."""
    path, land_path, fill, block_rows = job
    n, mean, m2, n_land = 0, 0.0, 0.0, 0
    vmin, vmax = np.inf, -np.inf
    with rasterio.open(path) as src, rasterio.open(land_path) as ref:
        # almost_equals, not ==: the real gdp, population and static rasters carry a pixel
        # height 1.73e-18 off HM's 0.009 deg -- one ULP, the same grid.
        if ((src.height, src.width) != (ref.height, ref.width)
                or not src.transform.almost_equals(ref.transform, precision=1e-9)):
            raise ValueError(f"{path} is not on the grid of {land_path}")
        nd = src.nodata
        for r0 in range(0, src.height, int(block_rows)):
            win = Window(0, r0, src.width, min(int(block_rows), src.height - r0))
            land = hm_land(ref.read(1, window=win), ref.nodata)
            v = src.read(1, window=win)[land]
            n_land += int(land.sum())
            ok = np.isfinite(v)
            if nd is not None and np.isfinite(nd):
                ok &= v != np.asarray(nd, dtype=v.dtype)
            if fill is not None:
                ok &= v != np.asarray(fill, dtype=v.dtype)
            x = v[ok].astype(np.float64)
            if x.size == 0:
                continue
            bn, bmean = x.size, float(x.mean())
            bm2 = float(((x - bmean) ** 2).sum())
            tot = n + bn
            delta = bmean - mean
            mean += delta * bn / tot
            m2 += bm2 + delta * delta * n * bn / tot
            n = tot
            vmin, vmax = min(vmin, float(x.min())), max(vmax, float(x.max()))
    return dict(path=os.path.basename(path), n=n, mean=mean, m2=m2, n_land=n_land,
                n_excluded=n_land - n, min=vmin, max=vmax)


def _merge(parts):
    n, mean, m2 = 0, 0.0, 0.0
    for p in parts:
        if p["n"] == 0:
            continue
        tot = n + p["n"]
        delta = p["mean"] - mean
        mean += delta * p["n"] / tot
        m2 += p["m2"] + delta * delta * n * p["n"] / tot
        n = tot
    if n == 0:
        raise ValueError(f"no land pixels in {[p['path'] for p in parts]}")
    prov = dict(n=int(n), n_land=int(sum(p["n_land"] for p in parts)),
                n_excluded=int(sum(p["n_excluded"] for p in parts)),
                min=min(p["min"] for p in parts), max=max(p["max"] for p in parts),
                files=[p["path"] for p in parts])
    return float(mean), float(np.sqrt(m2 / n) + 1e-8), prov


def build_norm_stats(hm_files, component_files, years, hm_vars, static_files, land_ref,
                     static_nodata, block_rows=1024, workers=8):
    """The sidecar dict, in the keys ``HumanFootprintChipDataset._load_norm_stats`` reads."""
    years = list(years)
    if len(hm_files) != len(years):
        raise ValueError("hm_files must be aligned with years")
    declared = {os.path.basename(p): float(static_nodata[os.path.basename(p)])
                for p in static_files if os.path.basename(p) in (static_nodata or {})}
    groups = {("hm",): [(p, p, None, block_rows) for p in hm_files]}
    for vi, v in enumerate(hm_vars):
        groups[("comp", v)] = [(component_files[y][vi], hm_files[k], None, block_rows)
                               for k, y in enumerate(years)]
    for p in static_files:
        groups[("static", os.path.basename(p))] = [
            (p, land_ref, declared.get(os.path.basename(p)), block_rows)]
    jobs = [j for js in groups.values() for j in js]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            results = list(ex.map(_moments, jobs))
    else:
        results = [_moments(j) for j in jobs]
    it = iter(results)
    merged = {key: _merge([next(it) for _ in js]) for key, js in groups.items()}

    hm_mean, hm_std, hm_prov = merged[("hm",)]
    out = dict(hm_mean=hm_mean, hm_std=hm_std, static_means=[], static_stds=[],
               comp_means={}, comp_stds={})
    prov = dict(method="exact streaming moments over HM land pixels (src.land.hm_land); "
                       "population std + 1e-8",
                land_ref=os.path.basename(land_ref),
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                hm=hm_prov, comp={}, static={})
    for v in hm_vars:
        m, s, pr = merged[("comp", v)]
        out["comp_means"][v], out["comp_stds"][v] = m, s
        prov["comp"][v] = pr
    for p in static_files:
        b = os.path.basename(p)
        m, s, pr = merged[("static", b)]
        out["static_means"].append(m)
        out["static_stds"].append(s)
        prov["static"][b] = pr
    out.update(static_files=[os.path.basename(p) for p in static_files],
               include_components=bool(hm_vars), static_nodata=declared,
               stats_domain="land", provenance=prov)
    return out


def main(argv=None):
    import torchgeo_dataloader as tdl
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--land_ref", default=tdl.hm_files[-1],
                    help="HM raster whose land masks the static channels (default HM 2020)")
    ap.add_argument("--terrain_covariates", action="store_true",
                    help="Include the slope/aspect channels (E2c), in the order the dataset "
                         "reads them.")
    ap.add_argument("--block_rows", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing sidecar. Off by default: a model's checkpoints "
                         "are only meaningful beside the sidecar they trained on.")
    args = ap.parse_args(argv)
    if os.path.exists(args.out) and not args.overwrite:
        print(f"REFUSING: {args.out} exists; checkpoints may depend on it (--overwrite).",
              file=sys.stderr)
        return 1
    t0 = time.time()
    st = build_norm_stats(tdl.hm_files, tdl.component_files, tdl.years, tdl.HM_VARS,
                          tdl.static_file_list(args.terrain_covariates), args.land_ref,
                          tdl.STATIC_NODATA,
                          block_rows=args.block_rows, workers=args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(st, f, indent=2)
    print(f"hm: mean {st['hm_mean']:.6f} std {st['hm_std']:.6f} "
          f"(n {st['provenance']['hm']['n']:,})")
    for b, m, s in zip(st["static_files"], st["static_means"], st["static_stds"]):
        pr = st["provenance"]["static"][b]
        print(f"{b:36s} mean {m:14.6g} std {s:14.6g}  land {pr['n_land']:,} "
              f"excluded {pr['n_excluded']:,}  range [{pr['min']:.4g}, {pr['max']:.4g}]")
    for v in st["comp_means"]:
        pr = st["provenance"]["comp"][v]
        print(f"{v:36s} mean {st['comp_means'][v]:14.6g} std {st['comp_stds'][v]:14.6g}  "
              f"excluded {pr['n_excluded']:,}")
    print(f"wrote {args.out} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
