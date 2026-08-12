#!/usr/bin/env python3
"""Phase 3 — generate the correlated ensemble and write it to zarr.

Marginals come from the **recalibrated** production rasters (Phase 1.5) plus the unmodified
central forecast; correlation structure comes from the Phase 1c variogram fit and the
hindcast horizon autocorrelation. Members are AR(1)-coupled across horizons so that
*change* statistics between two horizons inherit the right correlation.

Members are split across GPUs in blocks aligned to the zarr member-chunk, so two workers
never write the same chunk.

Storage: ``(M, n_horizons, H, W)`` int16, chunks ``(10, 1, 1024, 1024)``, scale 1/32767
(HM is bounded on [0,1] and the observed maximum reaches 1.0), sentinel -32768 for invalid
pixels. ``manifest.json`` records each member's seed and AR(1) chain, so a single member can
be regenerated deterministically and the ensemble can be extended from 50 to 200 members
later without redoing anything.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ensemble.copula import DEFAULT_SCALE, INT16_SENTINEL, Z975  # noqa: E402

REPO = Path(__file__).parent.parent
MEMBER_CHUNK = 10


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--central_dir", default="data/predictions")
    ap.add_argument("--recal_dir", default="data/predictions/recal")
    ap.add_argument("--central_pattern", default="prediction_{year}_central_blended.tif")
    ap.add_argument("--recal_pattern", default="prediction_{year}_{q}_recal.tif")
    ap.add_argument("--fallback_pattern", default="prediction_{year}_{q}_blended.tif")
    ap.add_argument("--years", default="2025,2030,2035,2040")
    ap.add_argument("--base_year", type=int, default=2020)
    ap.add_argument("--members", type=int, default=50)
    ap.add_argument("--out", default="data/ensemble/members.zarr")
    ap.add_argument("--variogram_fits", default="data/ensemble/diagnostics/variogram_fits.csv")
    ap.add_argument("--rho_json", default="data/ensemble/residuals/horizon_autocorrelation.json")
    ap.add_argument("--ranges_px", default=None, help="Override, e.g. '8,240'")
    ap.add_argument("--weights", default=None, help="Override, e.g. '0.5,0.4'")
    ap.add_argument("--nugget", type=float, default=None)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--wrap_lon", type=lambda x: str(x).lower() == "true", default=None,
                    help="Default: auto (True only for the full-globe grid)")
    ap.add_argument("--independent", action="store_true",
                    help="Build the independent-pixel null ensemble (same marginals, no "
                         "spatial structure) — the honest null for T3.2/T3.3")
    ap.add_argument("--disable_wandb", action="store_true")
    ap.add_argument("--wandb_group", default=None)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------
def resolve_paths(args, years):
    """(central, lower, upper) per year, preferring recalibrated bounds."""
    out = {}
    used_recal = {}
    for y in years:
        cen = Path(args.central_dir) / args.central_pattern.format(year=y)
        trio = {"central": cen}
        for q in ("lower", "upper"):
            r = Path(args.recal_dir) / args.recal_pattern.format(year=y, q=q)
            if r.exists():
                trio[q] = r
                used_recal[y] = True
            else:
                trio[q] = Path(args.central_dir) / args.fallback_pattern.format(year=y, q=q)
                used_recal[y] = False
        missing = [str(p) for p in trio.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing rasters for {y}: {missing}")
        out[y] = trio
    return out, used_recal


def load_marginals(paths, years):
    """Compact per-valid-pixel marginal parameters, plus the shared valid mask."""
    loc, sl, sr = {}, {}, {}
    valid = None
    profile = None
    for y in years:
        with rasterio.open(paths[y]["central"]) as c:
            cen = c.read(1)
            profile = c.profile.copy()
        with rasterio.open(paths[y]["lower"]) as s:
            low = s.read(1)
        with rasterio.open(paths[y]["upper"]) as s:
            upp = s.read(1)
        v = np.isfinite(cen) & np.isfinite(low) & np.isfinite(upp)
        valid = v if valid is None else (valid & v)
        loc[y], sl[y], sr[y] = cen, low, upp

    idx = np.flatnonzero(valid.ravel()).astype(np.int64)
    compact = {}
    for y in years:
        cen = loc[y].ravel()[idx].astype(np.float32)
        low = sl[y].ravel()[idx].astype(np.float32)
        upp = sr[y].ravel()[idx].astype(np.float32)
        compact[y] = {
            "loc": cen,
            "scale_left": np.maximum((cen - low) / Z975, 1e-6).astype(np.float32),
            "scale_right": np.maximum((upp - cen) / Z975, 1e-6).astype(np.float32),
        }
    return compact, valid, idx, profile


def field_params(args, horizons):
    """Correlation structure per horizon: from the fitted variogram unless overridden."""
    if args.ranges_px and args.weights:
        ranges = [float(x) for x in args.ranges_px.split(",")]
        weights = [float(x) for x in args.weights.split(",")]
        nug = float(args.nugget if args.nugget is not None else max(0.0, 1 - sum(weights)))
        return {h: {"ranges_px": ranges, "weights": weights, "nugget": nug} for h in horizons}

    fits = pd.read_csv(args.variogram_fits)
    glob = fits[fits["stratum"].astype(str).isin(["ALL", "nan", "None"])]
    if glob.empty:
        glob = fits
    out = {}
    for h in horizons:
        sub = glob[glob["horizon"] == h] if "horizon" in glob else glob
        if sub.empty:
            sub = glob
        row = sub.iloc[0]
        total = row["nugget"] + row["var_short"] + row["var_long"]
        out[h] = {
            "ranges_px": [float(row["range_short_px"]), float(row["range_long_px"])],
            "weights": [float(row["var_short"] / total), float(row["var_long"] / total)],
            "nugget": float(row["nugget"] / total),
        }
    return out


def load_rho(args, horizons):
    rho = {}
    p = Path(args.rho_json)
    if p.exists():
        raw = json.load(open(p))
        rho = {int(k): float(v) for k, v in raw.items() if np.isfinite(float(v))}
    return {h: rho.get(h, 0.9) for h in horizons[1:]}


def assign_members(n_members, n_workers, block=MEMBER_CHUNK):
    """Partition members into chunk-aligned blocks so workers never share a zarr chunk."""
    blocks = [list(range(s, min(s + block, n_members))) for s in range(0, n_members, block)]
    out = [[] for _ in range(n_workers)]
    for i, b in enumerate(blocks):
        out[i % n_workers].extend(b)
    return out


# --------------------------------------------------------------------------------------
def worker(worker_id, gpu, member_ids, cfg):
    import torch

    from src.ensemble.copula import marginal_from_z_torch
    from src.ensemble.fields import generate_correlated_field

    import zarr

    years = cfg["years"]
    horizons = cfg["horizons"]
    H, W = cfg["shape"]
    device = f"cuda:{gpu}" if (gpu is not None and torch.cuda.is_available()) else "cpu"

    compact = cfg["compact"]
    idx_np = cfg["idx"]
    z_store = zarr.open(cfg["out"], mode="r+")

    idx_t = torch.as_tensor(idx_np, device=device)
    marg = {
        y: {k: torch.as_tensor(v, device=device) for k, v in compact[y].items()}
        for y in years
    }
    scatter = np.full(H * W, INT16_SENTINEL, dtype=np.int16)

    for m in member_ids:
        t0 = time.time()
        prev = None
        for hi, (h, y) in enumerate(zip(horizons, years)):
            seed = cfg["seed"] + 1_000_003 * m + 101 * h
            if cfg["independent"]:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                eps = torch.randn(idx_t.shape[0], generator=gen, device=device)
            else:
                fp = cfg["field_params"][h]
                field = generate_correlated_field(
                    H, W, fp["ranges_px"], fp["weights"], fp["nugget"],
                    wrap_lon=cfg["wrap_lon"], device=device, seed=seed, return_torch=True,
                )
                eps = field.reshape(-1)[idx_t].clone()
                del field
                torch.cuda.empty_cache() if device.startswith("cuda") else None

            if prev is None:
                z = eps
            else:
                rho = float(cfg["rho"].get(h, 0.9))
                z = rho * prev + np.sqrt(max(0.0, 1 - rho ** 2)) * eps
            prev = z

            vals = marginal_from_z_torch(
                z, marg[y]["loc"], marg[y]["scale_left"], marg[y]["scale_right"]
            )
            q = torch.clamp(torch.round(vals / cfg["scale"]), INT16_SENTINEL + 1, 32767)
            q = q.to(torch.int16).cpu().numpy()
            scatter[:] = INT16_SENTINEL
            scatter[idx_np] = q
            z_store[m, hi] = scatter.reshape(H, W)
        print(f"[worker {worker_id} gpu {gpu}] member {m} done in {time.time() - t0:.1f}s", flush=True)


def main(argv=None):
    args = parse_args(argv)
    years = [int(y) for y in args.years.split(",")]
    horizons = [y - args.base_year for y in years]

    paths, used_recal = resolve_paths(args, years)
    if not all(used_recal.values()):
        print("⚠ Recalibrated bounds not found for every year; falling back to the raw heads "
              "for those years. Phase 1.5 must be final before the production run.")

    print("Loading marginals ...")
    compact, valid, idx, profile = load_marginals(paths, years)
    H, W = valid.shape
    print(f"  grid {H} x {W}, {idx.size:,} valid pixels ({100 * idx.size / valid.size:.1f}%)")

    wrap = args.wrap_lon
    if wrap is None:
        wrap = bool(W >= 39_000 and abs(profile["transform"].c + 180.0) < 0.05)
    fps = field_params(args, horizons)
    rho = load_rho(args, horizons)
    print(f"  wrap_lon={wrap}  rho={rho}")
    for h in horizons:
        print(f"  h={h}: ranges={[round(r, 1) for r in fps[h]['ranges_px']]} "
              f"weights={[round(w, 3) for w in fps[h]['weights']]} nugget={fps[h]['nugget']:.3f}")

    import zarr

    out_path = Path(args.out)
    est_gb = args.members * len(years) * H * W * 2 / 1e9
    print(f"  writing {out_path} — {est_gb:.1f} GB uncompressed "
          f"({100 * idx.size / valid.size:.0f}% populated, expect far less on disk)")
    store = zarr.open(
        str(out_path), mode="w",
        shape=(args.members, len(years), H, W),
        chunks=(MEMBER_CHUNK, 1, 1024, 1024),
        dtype="i2", fill_value=INT16_SENTINEL,
    )
    store.attrs.update({
        "scale": DEFAULT_SCALE, "offset": 0.0, "sentinel": INT16_SENTINEL,
        "years": years, "horizons": horizons, "base_year": args.base_year,
        "members": args.members, "independent_null": bool(args.independent),
        "wrap_lon": bool(wrap),
        "field_params": {str(k): v for k, v in fps.items()},
        "rho": {str(k): v for k, v in rho.items()},
        "seed": args.seed,
        "marginal": "median-spliced two-piece normal; ppf(0.5)=central by construction",
        "sources": {str(y): {k: str(v) for k, v in paths[y].items()} for y in years},
    })

    cfg = {
        "years": years, "horizons": horizons, "shape": (H, W), "compact": compact,
        "idx": idx, "out": str(out_path), "field_params": fps, "rho": rho,
        "seed": args.seed, "scale": DEFAULT_SCALE, "wrap_lon": bool(wrap),
        "independent": bool(args.independent),
    }

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""] if args.gpus else [None]
    assignment = assign_members(args.members, len(gpus))
    t0 = time.time()
    if len(gpus) == 1:
        worker(0, gpus[0], assignment[0], cfg)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for wid, (gpu, members) in enumerate(zip(gpus, assignment)):
            p = ctx.Process(target=worker, args=(wid, gpu, members, cfg))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Ensemble worker failed with exit code {p.exitcode}")
    elapsed = time.time() - t0

    manifest = {
        "members": [
            {"member": m, "seeds": {str(h): args.seed + 1_000_003 * m + 101 * h for h in horizons},
             "ar1_chain": [f"z_{horizons[0]}"] + [f"z_{h} = rho_{h} z_{p} + sqrt(1-rho^2) eps_{h}"
                                                  for p, h in zip(horizons[:-1], horizons[1:])]}
            for m in range(args.members)
        ],
        "rho": rho, "field_params": {str(k): v for k, v in fps.items()},
        "years": years, "wrap_lon": bool(wrap), "independent_null": bool(args.independent),
        "minutes": elapsed / 60.0,
    }
    with open(out_path.parent / (out_path.stem + "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    du = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e9
    print(f"\n✓ {args.members} members x {len(years)} horizons in {elapsed / 60:.1f} min "
          f"— {du:.1f} GB on disk")
    print("  NOTE: with M=50 the ensemble's own 2.5/97.5 percentiles carry real Monte-Carlo "
          "noise (they fall between the 1st and 2nd order statistics). T5.2 must be scored "
          "with an MC-scaled tolerance; tail-sensitive per-pixel products want M >= 200.")

    if not args.disable_wandb:
        try:
            import wandb
            run = wandb.init(project="spatio-temporal-convlstm",
                             group=args.wandb_group or "ensemble-generation",
                             job_type="ensemble-generation", tags=["ensemble", "phase3"],
                             config={**vars(args), "n_valid_px": int(idx.size)})
            run.log({"minutes": elapsed / 60.0, "disk_gb": du, "members": args.members})
            run.finish()
        except Exception as e:
            print(f"⚠ W&B unavailable ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
