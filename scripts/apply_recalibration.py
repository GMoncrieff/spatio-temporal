#!/usr/bin/env python3
"""Phase 1.5c — apply the class-conditional rescale to published quantile rasters.

Reads ``scale_factors.csv`` plus the class covariates (predicted change, baseline HM level,
biome) and writes recalibrated lower/upper rasters. **Central rasters are copied byte for
byte** unless ``--central_bias`` is given, so "the central forecast is unchanged" is
structural rather than something to verify afterwards — and when it is not true, the
manifest records ``central_regenerated: true`` next to the outputs.

Applies to the hindcast rasters (so Phase 4 can re-score them) and to the production
2025–2040 rasters (so the ensemble is built on recalibrated marginals).

Note on the assumption being made: ŝ is fit on 2000–2020 hindcast residuals and applied to
2025–2040 forecasts, i.e. the error structure is assumed stationary in time. That cannot be
validated directly — there is no future data — and it weakens the further out the forecast
runs. It is recorded in ``recal_manifest.json`` next to the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.calibrate import ScaleFactorTable  # noqa: E402
from src.ensemble.validate import (  # noqa: E402
    DHAT_BINS, DIST_BINS, HM_BINS, biome_lut, distance_band,
)

REPO = Path(__file__).parent.parent
HM_DIR = REPO / "data" / "raw" / "hm_global"
ECO_RASTER = HM_DIR / "ecoregion_id_1000.tif"
ECO_LOOKUP = HM_DIR / "ecoregion_lookup.csv"


def recalibrate_one(
    central_path,
    lower_path,
    upper_path,
    baseline_hm_path,
    table: ScaleFactorTable,
    horizon: int,
    out_dir,
    out_stem,
    ecoregion_raster=None,
    lookup_csv=None,
    block_rows: int = 1024,
    dist_raster=None,
    width_factors=None,
    central_bias=None,
):
    """Write ``{stem}_lower_recal.tif`` / ``{stem}_upper_recal.tif`` and copy central.

    ``width_factors`` is an extra per-(horizon x distance band) multiplier on the half-width,
    applied on top of whatever the conformal table says.

    It has to act *here*, on the published bounds, rather than inside the ensemble. T5.2
    requires the ensemble's 2.5/97.5 percentiles to equal the published lower/upper, and the
    marginal shape is normalized to those same bounds — narrowing the members without
    narrowing the rasters would fail the gate by construction, and narrowing the rasters
    without refitting the shape would fit the shape to widths generation does not use.
    Rebuild the residuals against this output before refitting the shape.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The third class axis must be filled with whatever the factors were fit against.
    # When the fit used distance-to-past-change (the default now), reading a biome number
    # into that slot silently misses every cell and falls back to the coarse average.
    biome_map = None
    if dist_raster is None and ecoregion_raster is not None and lookup_csv is not None:
        biome_map, _, _ = biome_lut(lookup_csv)

    with rasterio.open(central_path) as c:
        profile = c.profile.copy()
        H, W = c.height, c.width
        p_t = c.transform
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", BIGTIFF="YES")

    central_out = out_dir / f"{out_stem}_central_recal.tif"
    if central_bias is None:
        shutil.copyfile(central_path, central_out)

    srcs = {
        "c": rasterio.open(central_path),
        "l": rasterio.open(lower_path),
        "u": rasterio.open(upper_path),
        "hm0": rasterio.open(baseline_hm_path),
    }
    eco_src = rasterio.open(ecoregion_raster) if biome_map is not None else None
    dist_src = rasterio.open(dist_raster) if dist_raster else None

    def _off(src):
        t = src.transform
        return int(round((p_t.f - t.f) / t.e)), int(round((p_t.c - t.c) / t.a))

    hm_off = _off(srcs["hm0"])
    e_off = _off(eco_src) if eco_src else None
    d_off = _off(dist_src) if dist_src else None

    stats = {"n_valid": 0, "sum_width_before": 0.0, "sum_width_after": 0.0,
             "n_monotonicity_fixed": 0, "n_clipped": 0}
    lo_out = out_dir / f"{out_stem}_lower_recal.tif"
    up_out = out_dir / f"{out_stem}_upper_recal.tif"
    dcen = rasterio.open(central_out, "w", **profile) if central_bias is not None else None
    try:
        with rasterio.open(lo_out, "w", **profile) as dlo, rasterio.open(up_out, "w", **profile) as dup:
            for r0 in range(0, H, block_rows):
                rr = min(block_rows, H - r0)
                win = Window(0, r0, W, rr)
                cen = srcs["c"].read(1, window=win).astype(np.float64)
                low = srcs["l"].read(1, window=win).astype(np.float64)
                upp = srcs["u"].read(1, window=win).astype(np.float64)
                hm0 = srcs["hm0"].read(1, window=Window(hm_off[1], hm_off[0] + r0, W, rr),
                                       boundless=True, fill_value=np.nan).astype(np.float64)
                hm0 = np.where(hm0 < 0, np.nan, hm0)
                valid = np.isfinite(cen) & np.isfinite(low) & np.isfinite(upp)

                if dist_src is not None:
                    dd = dist_src.read(1, window=Window(d_off[1], d_off[0] + r0, W, rr),
                                       boundless=True, fill_value=1e4).astype(np.float64)
                    biome = distance_band(dd).astype(np.int64)
                elif eco_src is not None:
                    eco = eco_src.read(1, window=Window(e_off[1], e_off[0] + r0, W, rr),
                                       boundless=True, fill_value=0)
                    biome = biome_map[np.clip(eco, 0, len(biome_map) - 1)].astype(np.int64)
                else:
                    biome = np.zeros((rr, W), dtype=np.int64)

                dhat = np.where(np.isfinite(hm0), cen - hm0, 0.0)
                d_idx = np.digitize(dhat, DHAT_BINS[1:-1])

                if central_bias is not None:
                    # Shift the whole interval, so the width factors below still act on the
                    # width and not on the displacement. The class index is deliberately the
                    # one computed above, from the *uncorrected* central: the correction is
                    # a function of what the model predicted, and letting the assignment
                    # move under its own output would make it circular.
                    by_d = central_bias.get(str(int(horizon)), {})
                    shift = np.zeros_like(cen)
                    for d_str, v in by_d.items():
                        shift[d_idx == int(d_str)] = v
                    cen = cen + shift
                    low = low + shift
                    upp = upp + shift
                h_idx = np.digitize(np.nan_to_num(hm0, nan=0.0), HM_BINS[1:-1])
                s_up, s_lo = table.lookup(horizon, d_idx, h_idx, biome)

                if width_factors is not None:
                    # Keyed horizon -> distance band -> dhat bin -> HM bin, and each level
                    # is optional, so older two- and one-level artifacts still read. All
                    # three axes were measured to carry signal the others do not: the band
                    # is where the marginal fails, the dhat bin is where *coverage* fails
                    # (a band-only factor over-narrows the high-change class by ~24%, which
                    # is what T1.2/T1.3 measure), and within a fixed (band x dhat) cell the
                    # factor still varies 1.4-81x across HM level.
                    k_up = np.ones_like(s_up, dtype=np.float64)
                    k_lo = np.ones_like(s_lo, dtype=np.float64)

                    def _walk(entry, axes, sel):
                        if not isinstance(entry, dict):
                            ku, kl = entry
                            k_up[sel] = ku
                            k_lo[sel] = kl
                            return
                        if not axes:
                            return
                        for key, sub in entry.items():
                            s = sel & (axes[0] == int(key))
                            if s.any():
                                _walk(sub, axes[1:], s)

                    _walk(width_factors.get(str(int(horizon)), {}),
                          [biome, d_idx, h_idx], np.ones_like(k_up, dtype=bool))
                    s_up = s_up * k_up
                    s_lo = s_lo * k_lo

                w_up = np.maximum(upp - cen, 0.0)
                w_lo = np.maximum(cen - low, 0.0)
                new_up = cen + s_up * w_up
                new_lo = cen - s_lo * w_lo

                # Guard 4: monotonicity and bounds are enforced here, not assumed.
                bad = new_lo > cen
                stats["n_monotonicity_fixed"] += int((bad & valid).sum())
                new_lo = np.minimum(new_lo, cen)
                new_up = np.maximum(new_up, cen)
                pre_clip = (new_lo < 0) | (new_up > 1)
                stats["n_clipped"] += int((pre_clip & valid).sum())
                new_lo = np.clip(new_lo, 0.0, 1.0)
                new_up = np.clip(new_up, 0.0, 1.0)

                if dcen is not None:
                    dcen.write(np.where(valid, np.clip(cen, 0.0, 1.0), np.nan
                                        ).astype(np.float32), 1, window=win)

                new_lo = np.where(valid, new_lo, np.nan)
                new_up = np.where(valid, new_up, np.nan)
                stats["n_valid"] += int(valid.sum())
                stats["sum_width_before"] += float(np.nansum(np.where(valid, upp - low, 0.0)))
                stats["sum_width_after"] += float(np.nansum(np.where(valid, new_up - new_lo, 0.0)))

                dlo.write(new_lo.astype(np.float32), 1, window=win)
                dup.write(new_up.astype(np.float32), 1, window=win)
    finally:
        for s in srcs.values():
            s.close()
        if eco_src is not None:
            eco_src.close()
        if dist_src is not None:
            dist_src.close()

    width_ratio = (stats["sum_width_after"] / stats["sum_width_before"]
                   if stats["sum_width_before"] > 0 else np.nan)
    return {
        "central": str(central_out), "lower": str(lo_out), "upper": str(up_out),
        "horizon": horizon, "width_ratio": width_ratio, **stats,
    }


def enforce_horizon_monotonicity(out_dir, suffix="_recal", block_rows: int = 512):
    """Make each pixel's half-widths non-decreasing across horizons, in place.

    **Why this cannot live in the fit.** Constraining the factor table stops any *class*
    claiming more certainty at longer lead time, but a pixel is not in one class: predicted
    change grows with horizon, so 2.6% of pixels move between predicted-change bins along the
    sequence and pick up a different factor at each. Only a per-pixel pass closes that.

    **Why a cumulative maximum and not a smoothing.** The gate (T4.2) asks that spread never
    *decrease*. A cummax is the smallest change that guarantees it: it only ever widens, only
    at the horizons that dipped, and it leaves h=5 untouched. Measured cost on Africa:
    +1.2% / +1.5% / +2.5% of mean width at h=10/15/20.

    Applied to the half-widths about the central forecast, separately per side, so the
    central raster is untouched and T5.1's median identity is unaffected.

    **Streamed by row block.** The cummax runs along the horizon axis and is independent
    between pixels, so a block carries exactly the same result as a whole-raster pass. It is
    written that way because the whole-raster spelling holds ``3 x H`` full-grid float64
    arrays plus their ``np.stack`` copies plus four accumulators -- over 200 GB on the
    17111 x 40000 grid, which OOM-killed a global run. On a regional extent it was ~20 GB and
    invisible. Equivalence against the whole-raster version is asserted in
    ``tests/test_horizon_monotonicity_streaming.py``.
    """
    out_dir = Path(out_dir)
    groups = {}
    # Two naming conventions reach this function. Hindcast rasters carry the input window,
    # "w2000_prediction_2005"; production rasters do not, "prediction_2025". The glob used to
    # require the prefix, so it matched no production file at all and this pass was a silent
    # no-op on the forward product -- which would have shipped without the per-pixel cumulative
    # max that makes T4.2 structural. It was never noticed because no production model existed.
    for cen in sorted(out_dir.glob(f"*prediction_*_central{suffix}.tif")):
        stem = cen.name.replace(f"_central{suffix}.tif", "")
        head = stem.split("_")[0]
        base = int(head[1:]) if (head.startswith("w") and head[1:].isdigit()) else None
        year = int(stem.split("_")[-1])
        # Production years all belong to one forecast window, so they form one group; the sort
        # key only has to order the horizons, and target year does that either way.
        groups.setdefault(base, []).append((year, stem))
    total_lifted = n_px = 0
    for base, items in sorted(groups.items()):
        items.sort()
        if len(items) < 2:
            continue
        stems = [s for _, s in items]
        with rasterio.open(out_dir / f"{stems[0]}_central{suffix}.tif") as s0:
            H, W = s0.height, s0.width
            prof = s0.profile.copy()
        srcs_c = [rasterio.open(out_dir / f"{s}_central{suffix}.tif") for s in stems]
        srcs_l = [rasterio.open(out_dir / f"{s}_lower{suffix}.tif") for s in stems]
        srcs_u = [rasterio.open(out_dir / f"{s}_upper{suffix}.tif") for s in stems]
        # Write to siblings and swap at the end: the sources stay readable while streaming.
        tmp_l = [out_dir / f"{s}_lower{suffix}.mono.tmp.tif" for s in stems]
        tmp_u = [out_dir / f"{s}_upper{suffix}.mono.tmp.tif" for s in stems]
        dst_l = [rasterio.open(p_, "w", **prof) for p_ in tmp_l]
        dst_u = [rasterio.open(p_, "w", **prof) for p_ in tmp_u]
        g_lifted = g_valid = 0
        try:
            for r0 in range(0, H, block_rows):
                rr = min(block_rows, H - r0)
                win = Window(0, r0, W, rr)
                C = np.stack([s.read(1, window=win).astype(np.float64) for s in srcs_c])
                L = np.stack([s.read(1, window=win).astype(np.float64) for s in srcs_l])
                U = np.stack([s.read(1, window=win).astype(np.float64) for s in srcs_u])
                valid = (np.isfinite(C).all(0) & np.isfinite(L).all(0)
                         & np.isfinite(U).all(0) & (C >= 0).all(0))
                hu = np.maximum(U - C, 0.0)
                hl = np.maximum(C - L, 0.0)
                hu2 = np.maximum.accumulate(hu, axis=0)
                hl2 = np.maximum.accumulate(hl, axis=0)
                lifted = ((hu2 > hu + 1e-12) | (hl2 > hl + 1e-12))
                g_lifted += int(lifted[:, valid].any(axis=0).sum())
                g_valid += int(valid.sum())
                for i in range(len(stems)):
                    new_up = np.where(valid, np.clip(C[i] + hu2[i], 0.0, 1.0), U[i])
                    new_lo = np.where(valid, np.clip(C[i] - hl2[i], 0.0, 1.0), L[i])
                    dst_l[i].write(new_lo.astype(prof["dtype"]), 1, window=win)
                    dst_u[i].write(new_up.astype(prof["dtype"]), 1, window=win)
        finally:
            for s in srcs_c + srcs_l + srcs_u:
                s.close()
            for d in dst_l + dst_u:
                d.close()
        for i, s in enumerate(stems):
            os.replace(tmp_l[i], out_dir / f"{s}_lower{suffix}.tif")
            os.replace(tmp_u[i], out_dir / f"{s}_upper{suffix}.tif")
        total_lifted += g_lifted
        n_px += g_valid
        label = f"w{base}" if base is not None else "production"
        print(f"  {label}: {len(items)} horizons, "
              f"{100 * g_lifted / max(g_valid, 1):.1f}% of pixels widened somewhere")
    if n_px:
        print(f"  horizon monotonicity enforced on {total_lifted:,} of {n_px:,} pixels "
              f"({100 * total_lifted / n_px:.1f}%)")
    return total_lifted


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factors", default="data/ensemble/calibration/scale_factors.csv")
    ap.add_argument("--targets", default="both", choices=["hindcast", "production", "both"])
    ap.add_argument("--hindcast_dir", default="data/ensemble/hindcast/stitched")
    ap.add_argument("--hindcast_suffix", default="",
                    help="Suffix before .tif on the hindcast rasters (e.g. '_blended')")
    ap.add_argument("--hindcast_out", default="data/ensemble/hindcast/recal")
    ap.add_argument("--production_dir", default="data/predictions")
    ap.add_argument("--production_out", default="data/predictions/recal")
    ap.add_argument("--production_years", default="2025,2030,2035,2040")
    ap.add_argument("--production_base_year", type=int, default=2020)
    ap.add_argument("--ecoregion_raster", default=str(ECO_RASTER))
    ap.add_argument("--lookup_csv", default=str(ECO_LOOKUP))
    ap.add_argument("--no_biome", action="store_true", help="Ignore the biome stratum")
    ap.add_argument("--central_bias", default=None,
                    help="JSON of per-(horizon x predicted-change class) shifts "
                         "(scripts/fit_central_bias.py). REGENERATES the central raster "
                         "instead of copying it, which the rest of the pipeline assumes it "
                         "never does — pass it only deliberately. The whole interval is "
                         "shifted, so a width factor still acts on width alone.")
    ap.add_argument("--monotone_horizons", type=lambda x: (str(x).lower() == "true"),
                    nargs="?", const=True, default=False,
                    help="After all factors are applied, force each pixel's half-widths to be "
                         "non-decreasing across horizons (cumulative max). This is what makes "
                         "T4.2 structural rather than hoped-for; see "
                         "enforce_horizon_monotonicity for why it cannot live in the fit.")
    ap.add_argument("--width_factors", default=None,
                    help="JSON of per-(horizon x distance band) half-width multipliers "
                         "(scripts/predict_change_rates.py --out_width_json), applied on "
                         "top of the conformal table. Requires --dist_raster, since the "
                         "band is the class axis. Rebuild residuals against the output "
                         "before refitting the marginal shape.")
    ap.add_argument("--dist_raster", default=None,
                    help="Distance-to-past-change raster; fills the third class axis when "
                         "the factors were fit against distance rather than biome")
    args = ap.parse_args(argv)

    if str(args.factors).lower() in ("none", ""):
        # Unified mode: the width-factor table is the sole source of the rescaling, so the
        # conformal table is the identity. The two used to be fitted separately on different
        # class axes and multiplied here, which meant no single fit ever saw the product it
        # was contributing to — and the isotonic horizon constraint on one half was defeated
        # by the other. One layer, one class definition, one constraint.
        table = ScaleFactorTable(pd.DataFrame([
            {"horizon": h, "dhat_bin_idx": d, "hm_bin_idx": j, "biome": b,
             "s_up": 1.0, "s_lo": 1.0, "n_eff": 1}
            for h in (5, 10, 15, 20) for d in range(8) for j in range(8) for b in range(64)
        ]))
        print("  conformal table: IDENTITY (unified mode — width factors carry everything)")
    else:
        table = ScaleFactorTable.from_csv(args.factors)
    central_bias = None
    if args.central_bias:
        central_bias = json.load(open(args.central_bias)).get("by_horizon", {})
        n = sum(1 for d in central_bias.values() for v in d.values() if v)
        print(f"  ⚠ central bias from {args.central_bias}: {n} classes corrected — "
              f"the central raster is REGENERATED, not copied")
    width_factors = None
    if args.width_factors:
        _wf = json.load(open(args.width_factors))
        width_factors = _wf.get("by_horizon", {})
        axis = _wf.get("class_axis", "band")
        # The third class axis is keyed off whichever raster is supplied — dist wins over
        # ecoregion inside recalibrate_one — so a mismatch here looks up the wrong classes
        # and silently applies the wrong factors. Checked rather than trusted.
        if axis in ("band", "both") and not args.dist_raster:
            raise SystemExit(f"width factors were fitted on the '{axis}' axis, which needs "
                             f"--dist_raster")
        if axis == "biome":
            if args.dist_raster:
                raise SystemExit("width factors were fitted on the 'biome' axis but "
                                 "--dist_raster is set; the distance raster takes priority "
                                 "when both are given, so this would key the lookup off the "
                                 "wrong axis. Drop --dist_raster.")
            if args.no_biome or not args.ecoregion_raster:
                raise SystemExit("width factors were fitted on the 'biome' axis and need "
                                 "--ecoregion_raster")
        print(f"  width factors keyed on '{axis}'")
        n = sum(len(v) if isinstance(v, dict) else 1
                for d in width_factors.values() for v in d.values())
        print(f"  width factors from {args.width_factors}: "
              f"{len(width_factors)} horizons, {n} classes")
    eco = None if args.no_biome else args.ecoregion_raster
    lut = None if args.no_biome else args.lookup_csv
    results = []

    if args.targets in ("hindcast", "both"):
        hd = Path(args.hindcast_dir)
        suf = args.hindcast_suffix
        for cen in sorted(hd.glob(f"*_prediction_*_central{suf}.tif")):
            stem = cen.name.replace(f"_central{suf}.tif", "")
            base = int(stem.split("_")[0][1:])          # "w2000_prediction_2020" -> 2000
            target_year = int(stem.split("_")[-1])
            horizon = target_year - base
            low = hd / f"{stem}_lower{suf}.tif"
            upp = hd / f"{stem}_upper{suf}.tif"
            if not (low.exists() and upp.exists()):
                continue
            print(f"Recalibrating hindcast {stem} (h={horizon}) ...")
            results.append(recalibrate_one(
                cen, low, upp, HM_DIR / f"HM_{base}_AA_1000.tiff", table, horizon,
                args.hindcast_out, stem, ecoregion_raster=eco, lookup_csv=lut,
                dist_raster=args.dist_raster, width_factors=width_factors,
                central_bias=central_bias,
            ))

    if args.targets in ("production", "both"):
        pd_dir = Path(args.production_dir)
        base = args.production_base_year
        for year in [int(y) for y in args.production_years.split(",")]:
            cen = pd_dir / f"prediction_{year}_central_blended.tif"
            low = pd_dir / f"prediction_{year}_lower_blended.tif"
            upp = pd_dir / f"prediction_{year}_upper_blended.tif"
            if not (cen.exists() and low.exists() and upp.exists()):
                print(f"  ⚠ missing production rasters for {year}; skipping")
                continue
            horizon = year - base
            print(f"Recalibrating production {year} (h={horizon}) ...")
            results.append(recalibrate_one(
                cen, low, upp, HM_DIR / f"HM_{base}_AA_1000.tiff", table, horizon,
                args.production_out, f"prediction_{year}", ecoregion_raster=eco,
                lookup_csv=lut, dist_raster=args.dist_raster, width_factors=width_factors,
                central_bias=central_bias,
            ))

    if args.monotone_horizons:
        print("\nEnforcing non-decreasing half-widths across horizons ...")
        if args.targets in ("hindcast", "both"):
            enforce_horizon_monotonicity(args.hindcast_out, "_recal")
        if args.targets in ("production", "both"):
            enforce_horizon_monotonicity(args.production_out, "_recal")

    out_manifest = Path(args.production_out if args.targets != "hindcast" else args.hindcast_out)
    out_manifest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "factors": str(args.factors),
        "width_factors": str(args.width_factors) if args.width_factors else None,
        "central_bias": str(args.central_bias) if args.central_bias else None,
        "central_regenerated": bool(args.central_bias),
        "width_factors_applied": width_factors,
        "stationarity_assumption": (
            "scale factors fit on 2000-2020 hindcast residuals and applied to later "
            "forecasts; assumes the error structure is stationary in time and cannot be "
            "validated directly"
        ),
        "outputs": results,
    }
    with open(out_manifest / "recal_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    print("\n" + "=" * 70)
    for r in results:
        print(f"  {Path(r['upper']).name}: width x{r['width_ratio']:.3f}, "
              f"{r['n_monotonicity_fixed']:,} monotonicity fixes, {r['n_clipped']:,} clipped")
    print(f"Manifest: {out_manifest / 'recal_manifest.json'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
