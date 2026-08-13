"""Phase 1.5 — class-conditional recalibration of the existing quantile heads.

Mechanism: Mondrian (class-conditional) split conformal on a *normalized* nonconformity
score, so the model's own spatial pattern of interval width is retained and only its level
is corrected:

    E_i = (Y_i − central_i) / w_up_i     if Y_i > central_i
          (central_i − Y_i) / w_lo_i     otherwise

Per class and tail, ŝ is the conformal quantile of E; the published bounds become
``upper' = central + ŝ_up · w_up`` and ``lower' = central − ŝ_lo · w_lo``. The central
forecast is never touched, ``PinballLoss`` is never touched, and if the audit says the
heads are already fine the whole phase collapses to the identity.

Tail level. The plan's rule takes the (1−α) conformal quantile within each tail's subset.
That equals the intended α/2 *marginal* miscoverage only when exactly half the pixels sit
above the central forecast. The residuals here are skewed, so the level actually used is

    level_up = 1 − (α/2) · n_total / n_above

which reduces to the plan's rule at a 50/50 split and hits T1.1 when it is not. Both the
naive and marginal levels are recorded in the factors table.

Four guards, all necessary (see plan §1.5b): effective sample size from 128 px chips,
shrinkage toward the per-horizon global factor, smoothness across ordered bins, and
bounds/monotonicity (including ŝ non-decreasing in horizon, so spread still grows with
lead time).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from .validate import (
    CHIP_SIZE,
    DHAT_BINS,
    DHAT_LABELS,
    DIST_BINS,
    DIST_LABELS,
    HM_BINS,
    HM_LABELS,
    biome_lut,
    wilson_interval,
)

CLASS_KEYS = ("horizon", "dhat_bin_idx", "hm_bin_idx", "biome")
S_FLOOR = 0.2
S_CEIL = 20.0
# Half widths at or below this are treated as degenerate (crossed or collapsed heads).
MIN_HALF_WIDTH = 1e-5


# --------------------------------------------------------------------------------------
# Score collection
# --------------------------------------------------------------------------------------
@dataclass
class _Scores:
    up: list = field(default_factory=list)
    lo: list = field(default_factory=list)
    w_up: list = field(default_factory=list)
    w_lo: list = field(default_factory=list)
    n_up: int = 0
    n_lo: int = 0
    n_degenerate: int = 0
    n_total: int = 0
    chips: set = field(default_factory=set)


class ScoreStore:
    """Per-(class, fold) reservoirs of conformal scores.

    Keeping the fold split lets Phase 1.5d refit leave-one-fold-out for free: fitting ŝ and
    reporting coverage on the same residuals is circular and always looks good.
    """

    def __init__(self, cap: int = 8000, random_seed: int = 42, class_keys=CLASS_KEYS,
                 third_axis: str = "biome"):
        self.cap = cap
        self.rng = np.random.default_rng(random_seed)
        self.class_keys = tuple(class_keys)
        # What the third class axis holds: "dist" (6 ordered distance-to-past-change bands)
        # or "biome" (14 unordered categories). It decides whether that axis belongs in the
        # primary stratum: ordered distance bands carry the signal that determines whether
        # change is possible at all, while biome would merely re-fragment thin classes.
        self.third_axis = third_axis
        self.data: dict = defaultdict(_Scores)

    def add(self, key, fold, up=None, lo=None, chips=None, n_degenerate=0, n_total=0,
            w_up=None, w_lo=None):
        s = self.data[(key, fold)]
        # The half widths themselves are kept (subsampled) because the monotone-spread
        # guard has to act on s * w, not on s alone.
        if w_up is not None and w_up.size:
            s.w_up.append(self._subsample(w_up))
        if w_lo is not None and w_lo.size:
            s.w_lo.append(self._subsample(w_lo))
        if up is not None and up.size:
            s.n_up += int(up.size)
            s.up.append(self._subsample(up))
        if lo is not None and lo.size:
            s.n_lo += int(lo.size)
            s.lo.append(self._subsample(lo))
        if chips is not None:
            s.chips.update(chips.tolist())
        s.n_degenerate += int(n_degenerate)
        s.n_total += int(n_total)

    def _subsample(self, a):
        if a.size <= self.cap:
            return a.astype(np.float32)
        idx = self.rng.choice(a.size, size=self.cap, replace=False)
        return a[idx].astype(np.float32)

    def pooled(self, exclude_fold=None, keys=None):
        """Merge folds (optionally leaving one out) into per-class score arrays."""
        out = defaultdict(lambda: {"up": [], "lo": [], "n_up": 0, "n_lo": 0, "chips": set()})
        for (key, fold), s in self.data.items():
            if exclude_fold is not None and fold == exclude_fold:
                continue
            if keys is not None and key not in keys:
                continue
            o = out[key]
            o["up"].extend(s.up)
            o["lo"].extend(s.lo)
            o["n_up"] += s.n_up
            o["n_lo"] += s.n_lo
            o["n_degenerate"] = o.get("n_degenerate", 0) + s.n_degenerate
            o["n_total"] = o.get("n_total", 0) + s.n_total
            o.setdefault("w_up", []).extend(s.w_up)
            o.setdefault("w_lo", []).extend(s.w_lo)
            o["chips"].update(s.chips)
        merged = {}
        for key, o in out.items():
            wu = np.concatenate(o["w_up"]) if o.get("w_up") else np.empty(0, np.float32)
            wl = np.concatenate(o["w_lo"]) if o.get("w_lo") else np.empty(0, np.float32)
            merged[key] = {
                "up": np.concatenate(o["up"]) if o["up"] else np.empty(0, np.float32),
                "lo": np.concatenate(o["lo"]) if o["lo"] else np.empty(0, np.float32),
                "n_up": o["n_up"], "n_lo": o["n_lo"], "n_eff": len(o["chips"]),
                "n_degenerate": o.get("n_degenerate", 0), "n_total": o.get("n_total", 0),
                "w_up_med": float(np.median(wu)) if wu.size else np.nan,
                "w_lo_med": float(np.median(wl)) if wl.size else np.nan,
            }
        return merged

    def folds(self):
        return sorted({f for (_, f) in self.data if f is not None})

    def pooled_by(self, key_fn, exclude_fold=None):
        """Merge reservoirs under a coarser key, e.g. (horizon, dhat_bin) only.

        The primary stratum is horizon x predicted-change; HM level and biome are
        secondary. Fitting only at the full 4-way product fragments exactly the class the
        exercise is about — the >0.15 change bin holds ~10k chips in total but only a few
        dozen per (HM bin, biome) cell, so every cell gets shrunk back to the global
        factor and the miscoverage survives.
        """
        out = defaultdict(lambda: {"up": [], "lo": [], "n_up": 0, "n_lo": 0, "chips": set()})
        for (key, fold), s in self.data.items():
            if exclude_fold is not None and fold == exclude_fold:
                continue
            o = out[key_fn(key)]
            o["up"].extend(s.up)
            o["lo"].extend(s.lo)
            o["n_up"] += s.n_up
            o["n_lo"] += s.n_lo
            o["chips"].update(s.chips)
        merged = {}
        for key, o in out.items():
            merged[key] = {
                "up": np.concatenate(o["up"]) if o["up"] else np.empty(0, np.float32),
                "lo": np.concatenate(o["lo"]) if o["lo"] else np.empty(0, np.float32),
                "n_up": o["n_up"], "n_lo": o["n_lo"], "n_eff": len(o["chips"]),
            }
        return merged

    def save(self, path):
        """Persist the reservoirs (npz) so refits do not require rereading rasters.

        Collecting the scores streams every residual and covariate raster; refitting with
        different bins or shrinkage is milliseconds. Keeping the two apart is the
        difference between iterating on the calibration in seconds and in an hour.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        meta = []
        for i, ((key, fold), s) in enumerate(self.data.items()):
            payload[f"up_{i}"] = np.concatenate(s.up) if s.up else np.empty(0, np.float32)
            payload[f"lo_{i}"] = np.concatenate(s.lo) if s.lo else np.empty(0, np.float32)
            payload[f"chips_{i}"] = np.fromiter(s.chips, dtype=np.int64, count=len(s.chips))
            meta.append({"i": i, "key": list(key), "fold": fold, "n_up": s.n_up,
                         "n_lo": s.n_lo, "n_degenerate": s.n_degenerate, "n_total": s.n_total})
        np.savez(path, meta=np.array(json.dumps(meta)),
                 third_axis=np.array(self.third_axis), **payload)
        return path

    @classmethod
    def load(cls, path, cap: int = 8000, random_seed: int = 42):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        third = str(z["third_axis"]) if "third_axis" in z else "biome"
        store = cls(cap=cap, random_seed=random_seed, third_axis=third)
        for m in meta:
            i = m["i"]
            s = store.data[(tuple(m["key"]), m["fold"])]
            up, lo = z[f"up_{i}"], z[f"lo_{i}"]
            if up.size:
                s.up.append(up)
            if lo.size:
                s.lo.append(lo)
            s.n_up, s.n_lo = m["n_up"], m["n_lo"]
            s.n_degenerate, s.n_total = m.get("n_degenerate", 0), m.get("n_total", 0)
            s.chips.update(z[f"chips_{i}"].tolist())
        return store


def collect_conformal_scores(
    manifest,
    fold_mask_path=None,
    ecoregion_raster=None,
    lookup_csv=None,
    cap: int = 8000,
    block_rows: int = 1024,
    random_seed: int = 42,
):
    """Stream the hindcast residual rasters into a :class:`ScoreStore`."""
    df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
    has_dist = any(
        isinstance(r.get("path_dist_past_change"), str)
        and Path(r["path_dist_past_change"]).exists()
        for _, r in (df.iterrows() if hasattr(df, "iterrows") else [])
    )
    store = ScoreStore(cap=cap, random_seed=random_seed,
                       third_axis="dist" if has_dist else "biome")
    biome_map = None
    if ecoregion_raster is not None and lookup_csv is not None:
        biome_map, _, _ = biome_lut(lookup_csv)

    for _, row in df.iterrows():
        horizon = int(row["horizon"])
        srcs = {
            "res": rasterio.open(row["path_res_native"]),
            "dhat": rasterio.open(row["path_dhat"]),
            "hm0": rasterio.open(row["path_hm_t0"]),
            "w_up": rasterio.open(row["path_w_up"]),
            "w_lo": rasterio.open(row["path_w_lo"]),
        }
        eco_src = rasterio.open(ecoregion_raster) if biome_map is not None else None
        fold_src = rasterio.open(fold_mask_path) if fold_mask_path else None
        dist_col = row.get("path_dist_past_change")
        dist_src = rasterio.open(dist_col) if isinstance(dist_col, str) and Path(dist_col).exists() else None
        H, W = srcs["res"].height, srcs["res"].width
        p_t = srcs["res"].transform
        glob_row = int(round((p_t.f - 84.0) / p_t.e))
        glob_col = int(round((p_t.c + 180.0) / p_t.a))

        def _off(src):
            t = src.transform
            return int(round((p_t.f - t.f) / t.e)), int(round((p_t.c - t.c) / t.a))

        e_off = _off(eco_src) if eco_src else None
        f_off = _off(fold_src) if fold_src else None
        try:
            for r0 in range(0, H, block_rows):
                rr = min(block_rows, H - r0)
                win = Window(0, r0, W, rr)
                res = srcs["res"].read(1, window=win).astype(np.float64)
                dhat = srcs["dhat"].read(1, window=win).astype(np.float64)
                hm0 = srcs["hm0"].read(1, window=win).astype(np.float64)
                w_up = srcs["w_up"].read(1, window=win).astype(np.float64)
                w_lo = srcs["w_lo"].read(1, window=win).astype(np.float64)
                ok = (np.isfinite(res) & np.isfinite(dhat) & np.isfinite(hm0)
                      & np.isfinite(w_up) & np.isfinite(w_lo))
                if not ok.any():
                    continue
                if dist_src is not None:
                    # Proximity to past change dominates where change can happen at all;
                    # it replaces biome as the third class axis when available.
                    dd = dist_src.read(1, window=Window(0, r0, W, rr)).astype(np.float64)
                    biome = np.digitize(dd, DIST_BINS[1:-1]).astype(np.int64)
                elif eco_src is not None:
                    eco = eco_src.read(1, window=Window(e_off[1], e_off[0] + r0, W, rr),
                                       boundless=True, fill_value=0)
                    biome = biome_map[np.clip(eco, 0, len(biome_map) - 1)].astype(np.int64)
                else:
                    biome = np.zeros((rr, W), dtype=np.int64)
                if fold_src is not None:
                    folds = fold_src.read(1, window=Window(f_off[1], f_off[0] + r0, W, rr),
                                          boundless=True, fill_value=0).astype(np.int64)
                else:
                    folds = np.zeros((rr, W), dtype=np.int64)

                rows_idx, cols_idx = np.nonzero(ok)
                chip_id = (((glob_row + r0 + rows_idx) // CHIP_SIZE).astype(np.int64) * 400000
                           + ((glob_col + cols_idx) // CHIP_SIZE))
                r = res[ok]
                wu = w_up[ok]
                wl = w_lo[ok]
                # Degenerate/crossed heads (half width <= 0) cannot be rescaled: any finite
                # factor leaves a zero-width interval. Such pixels are dropped from the
                # score set and counted, rather than clamped — clamping would inject
                # enormous scores that wreck the global factor and the decision statistic.
                ok_up = wu > MIN_HALF_WIDTH
                ok_lo = wl > MIN_HALF_WIDTH
                d_idx = np.digitize(dhat[ok], DHAT_BINS[1:-1])
                h_idx = np.digitize(hm0[ok], HM_BINS[1:-1])
                b = biome[ok]
                fo = folds[ok]

                # Packed scalar key (see the note in validate.compute_class_conditional_coverage):
                # dhat bin | HM bin | biome | fold.
                keys = (d_idx.astype(np.int64) * 1_000_000
                        + h_idx.astype(np.int64) * 100_000
                        + b * 10
                        + fo)
                uniq, inv = np.unique(keys, return_inverse=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    e_up = np.where(ok_up, r / np.maximum(wu, MIN_HALF_WIDTH), np.nan)
                    e_lo = np.where(ok_lo, -r / np.maximum(wl, MIN_HALF_WIDTH), np.nan)
                take_up = (r > 0) & ok_up
                take_lo = (r <= 0) & ok_lo
                for u_i, packed in enumerate(uniq):
                    sel = inv == u_i
                    packed = int(packed)
                    key = (horizon, packed // 1_000_000,
                           (packed // 100_000) % 10, (packed // 10) % 10_000)
                    fold = (packed % 10) or None
                    su = sel & take_up
                    sl = sel & take_lo
                    store.add(
                        key, fold,
                        up=e_up[su] if su.any() else None,
                        lo=e_lo[sl] if sl.any() else None,
                        chips=np.unique(chip_id[sel]),
                        n_degenerate=int((sel & (~ok_up | ~ok_lo)).sum()),
                        n_total=int(sel.sum()),
                        w_up=wu[sel & ok_up], w_lo=wl[sel & ok_lo],
                    )
        finally:
            for s in srcs.values():
                s.close()
            if eco_src is not None:
                eco_src.close()
            if fold_src is not None:
                fold_src.close()
            if dist_src is not None:
                dist_src.close()
    return store


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------
def _weighted_median(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def _conformal_quantile(scores, n_eff, level):
    """Empirical quantile with the finite-sample conformal correction on n_eff blocks."""
    if scores.size == 0 or n_eff < 1:
        return np.nan
    level = float(np.clip(level, 0.0, 1.0))
    corrected = min(1.0, np.ceil((n_eff + 1) * level) / max(n_eff, 1))
    return float(np.quantile(scores, corrected))


def fit_scale_factors(
    store: ScoreStore,
    alpha: float = 0.05,
    n0: float = 200.0,
    exclude_fold=None,
    smooth: bool = True,
    tail_level: str = "marginal",
    primary_includes_dist=None,
):
    """Per-class ŝ_up / ŝ_lo with shrinkage, smoothing, and monotonicity guards."""
    if primary_includes_dist is None:
        primary_includes_dist = getattr(store, "third_axis", "biome") == "dist"
    pooled = store.pooled(exclude_fold=exclude_fold)
    rows = []
    for key, d in pooled.items():
        horizon, d_idx, h_idx, biome = key
        n_up, n_lo, n_eff = d["n_up"], d["n_lo"], d["n_eff"]
        n_tot = n_up + n_lo
        if n_tot == 0:
            continue
        if tail_level == "marginal":
            lvl_up = 1.0 - (alpha / 2.0) * n_tot / max(n_up, 1)
            lvl_lo = 1.0 - (alpha / 2.0) * n_tot / max(n_lo, 1)
        else:  # plan's literal rule
            lvl_up = lvl_lo = 1.0 - alpha
        rows.append({
            "horizon": horizon, "dhat_bin_idx": d_idx, "hm_bin_idx": h_idx, "biome": biome,
            "dhat_bin": DHAT_LABELS[d_idx], "hm_bin": HM_LABELS[h_idx],
            "n_px": n_tot, "n_up": n_up, "n_lo": n_lo, "n_eff": n_eff,
            "n_degenerate": d.get("n_degenerate", 0),
            "frac_degenerate": d.get("n_degenerate", 0) / max(d.get("n_total", 0), 1),
            "level_up": lvl_up, "level_lo": lvl_lo,
            "s_up_raw": _conformal_quantile(d["up"], n_eff, lvl_up),
            "s_lo_raw": _conformal_quantile(d["lo"], n_eff, lvl_lo),
            "median_e_up": float(np.median(d["up"])) if d["up"].size else np.nan,
            "median_e_lo": float(np.median(d["lo"])) if d["lo"].size else np.nan,
            "w_up_med": d.get("w_up_med", np.nan), "w_lo_med": d.get("w_lo_med", np.nan),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Raw factors are clipped before any aggregate is taken: a single class with collapsed
    # widths can otherwise produce a factor in the thousands and drag the global target and
    # the decision statistic with it.
    for tail in ("up", "lo"):
        df[f"s_{tail}_raw"] = df[f"s_{tail}_raw"].clip(0.0, S_CEIL)

    # --- primary-stratum factors, fitted on pooled scores ---------------------------------
    # The primary stratum includes the distance-to-past-change band when one is present.
    # It has to: the frozen upper head barely responds to proximity (w_up decays 1.9x from
    # adjacent to >100 px while the observed rate of change falls to exactly zero), so if
    # the far field is pooled with near-field low-change pixels its factor is dragged up by
    # them and the interval never collapses where change is impossible.
    primary = (lambda k: (k[0], k[1], k[3])) if primary_includes_dist else (lambda k: (k[0], k[1]))
    group = {}
    for key, d in store.pooled_by(primary, exclude_fold=exclude_fold).items():
        n_up, n_lo, n_eff = d["n_up"], d["n_lo"], d["n_eff"]
        n_tot = n_up + n_lo
        if n_tot == 0:
            continue
        if tail_level == "marginal":
            lvl_up = 1.0 - (alpha / 2.0) * n_tot / max(n_up, 1)
            lvl_lo = 1.0 - (alpha / 2.0) * n_tot / max(n_lo, 1)
        else:
            lvl_up = lvl_lo = 1.0 - alpha
        group[key] = {
            "s_up": float(np.clip(_conformal_quantile(d["up"], n_eff, lvl_up), 0.0, S_CEIL)),
            "s_lo": float(np.clip(_conformal_quantile(d["lo"], n_eff, lvl_lo), 0.0, S_CEIL)),
            "n_eff": n_eff,
        }

    # --- global (per-horizon) factor: the outermost shrinkage target ----------------------
    # Weighted *median*, not mean: the target must not be movable by one wild thin class.
    glob = {}
    for horizon, grp in df.groupby("horizon"):
        for tail in ("up", "lo"):
            w = grp["n_eff"].to_numpy(dtype=float)
            v = grp[f"s_{tail}_raw"].to_numpy(dtype=float)
            m = np.isfinite(v) & (w > 0)
            glob[(horizon, tail)] = _weighted_median(v[m], w[m]) if m.any() else 1.0
    df["s_up_global"] = [glob[(h, "up")] for h in df["horizon"]]
    df["s_lo_global"] = [glob[(h, "lo")] for h in df["horizon"]]

    # --- guard 2: hierarchical shrinkage cell -> primary stratum -> horizon ---------------
    keys = (list(zip(df["horizon"], df["dhat_bin_idx"], df["biome"]))
            if primary_includes_dist else list(zip(df["horizon"], df["dhat_bin_idx"])))
    for tail in ("up", "lo"):
        g_val = np.array([group.get(k, {}).get(f"s_{tail}", np.nan) for k in keys], dtype=float)
        g_n = np.array([group.get(k, {}).get("n_eff", 0.0) for k in keys], dtype=float)
        g_lam = g_n / (g_n + n0)
        g_val = np.where(np.isfinite(g_val), g_val, df[f"s_{tail}_global"])
        # The primary stratum itself is shrunk toward the per-horizon factor.
        df[f"s_{tail}_group"] = g_lam * g_val + (1 - g_lam) * df[f"s_{tail}_global"]
    df["lambda_group"] = np.array([group.get(k, {}).get("n_eff", 0.0) for k in keys]) / (
        np.array([group.get(k, {}).get("n_eff", 0.0) for k in keys]) + n0)

    df["lambda"] = df["n_eff"] / (df["n_eff"] + n0)
    for tail in ("up", "lo"):
        raw = df[f"s_{tail}_raw"].fillna(df[f"s_{tail}_group"])
        df[f"s_{tail}_shrunk"] = df["lambda"] * raw + (1 - df["lambda"]) * df[f"s_{tail}_group"]

    # --- guard 3: smoothness across ordered Delta-hat bins -------------------------------
    if smooth:
        df = _isotonic_smooth(df)
    else:
        for tail in ("up", "lo"):
            df[f"s_{tail}_smooth"] = df[f"s_{tail}_shrunk"]

    # --- guard 4: bounds + non-decreasing in horizon --------------------------------------
    for tail in ("up", "lo"):
        df[f"s_{tail}"] = df[f"s_{tail}_smooth"].clip(S_FLOOR, S_CEIL)
    df = _enforce_horizon_monotonicity(df)
    return df.sort_values(["horizon", "dhat_bin_idx", "hm_bin_idx", "biome"]).reset_index(drop=True)


def _isotonic_smooth(df):
    """Monotone (isotonic) fit of ŝ across ordered Delta-hat bins, per horizon and tail.

    A step function in ŝ produces visible discontinuities at bin boundaries in the output
    map — a real artifact, and T3.5 is a hard gate against it. The direction is chosen from
    the data rather than assumed.
    """
    from sklearn.isotonic import IsotonicRegression

    out = df.copy()
    for tail in ("up", "lo"):
        col = f"s_{tail}_shrunk"
        out[f"s_{tail}_smooth"] = out[col]
        for (horizon, h_idx, biome), grp in out.groupby(["horizon", "hm_bin_idx", "biome"]):
            if len(grp) < 3:
                continue
            x = grp["dhat_bin_idx"].to_numpy(float)
            y = grp[col].to_numpy(float)
            w = grp["n_eff"].to_numpy(float) + 1.0
            good = np.isfinite(y)
            if good.sum() < 3:
                continue
            # Weighted Pearson correlation picks the direction.
            xm = np.average(x[good], weights=w[good])
            ym = np.average(y[good], weights=w[good])
            cov = np.average((x[good] - xm) * (y[good] - ym), weights=w[good])
            increasing = bool(cov >= 0)
            ir = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            fitted = ir.fit_transform(x[good], y[good], sample_weight=w[good])
            out.loc[grp.index[good], f"s_{tail}_smooth"] = fitted
    return out


def _enforce_horizon_monotonicity(df):
    """Non-decreasing *spread* in horizon within a class — s * w, not s alone.

    T4.2 requires the interval to widen with lead time. The published half width is
    ``s * w``, and w already grows with horizon, so constraining s to grow as well is
    doubly conservative: measured on southern Africa it more than doubled the far-field
    factor at h=20 (conformal asked for 0.705, the s-only guard imposed 1.756) in exactly
    the classes where the data says the interval should collapse.
    """
    out = df.copy()
    keys = ["dhat_bin_idx", "hm_bin_idx", "biome"]
    for tail in ("up", "lo"):
        col, wcol = f"s_{tail}", f"w_{tail}_med"
        if wcol not in out:
            for _, grp in out.groupby(keys):
                g = grp.sort_values("horizon")
                out.loc[g.index, col] = np.maximum.accumulate(g[col].to_numpy(float))
            continue
        for _, grp in out.groupby(keys):
            g = grp.sort_values("horizon")
            s_vals = g[col].to_numpy(float)
            w_vals = g[wcol].to_numpy(float)
            good = np.isfinite(w_vals) & (w_vals > 0)
            if good.sum() < 2:
                out.loc[g.index, col] = np.maximum.accumulate(s_vals)
                continue
            spread = np.where(good, s_vals * w_vals, np.nan)
            filled = pd.Series(spread).ffill().bfill().to_numpy()
            mono = np.maximum.accumulate(filled)
            out.loc[g.index, col] = np.where(good, mono / np.where(good, w_vals, 1.0), s_vals)
    return out


def decide_recalibration(factors: pd.DataFrame, audit: pd.DataFrame | None = None,
                         s_tol=(0.9, 1.1), cov_tol=(0.93, 0.97), spread_tol: float = 0.1):
    """keep / global rescale / stratified rescale, per the plan's decision table."""
    if factors.empty:
        return {"decision": "keep", "reason": "no calibration data"}
    w = factors["n_eff"].to_numpy(float) + 1.0
    s_all = np.concatenate([factors["s_up_raw"].to_numpy(float), factors["s_lo_raw"].to_numpy(float)])
    w_all = np.concatenate([w, w])
    m = np.isfinite(s_all)
    s_all, w_all = s_all[m], w_all[m]
    in_band = bool(np.all((s_all >= s_tol[0]) & (s_all <= s_tol[1])))

    cov_ok = True
    if audit is not None and not audit.empty:
        sub = audit[audit["n_eff"] >= 100]
        if not sub.empty:
            cov_ok = bool(sub["coverage"].between(*cov_tol).all())

    mean_s = float(np.average(s_all, weights=w_all))
    median_s = _weighted_median(s_all, w_all)
    rel_spread = float(np.sqrt(np.average((s_all - mean_s) ** 2, weights=w_all)) / max(median_s, 1e-9))

    if in_band and cov_ok:
        decision, reason = "keep", "all class factors within [0.9,1.1] and coverage within [0.93,0.97]"
    elif rel_spread < spread_tol:
        decision, reason = "global", f"factors roughly constant across classes (relative spread {rel_spread:.3f})"
    else:
        decision, reason = "stratified", f"factors vary systematically across classes (relative spread {rel_spread:.3f})"
    frac_degen = (float(np.average(factors["frac_degenerate"], weights=w))
                  if "frac_degenerate" in factors else 0.0)
    return {
        "decision": decision, "reason": reason,
        "mean_s": mean_s, "median_s": median_s, "relative_spread": rel_spread,
        "min_s": float(np.min(s_all)), "max_s": float(np.max(s_all)),
        "coverage_in_band": cov_ok,
        "frac_degenerate_width": frac_degen,
    }


def as_identity(factors: pd.DataFrame):
    """Force ŝ = 1 everywhere (the 'keep' decision as an explicit no-op transform)."""
    out = factors.copy()
    out["s_up"] = 1.0
    out["s_lo"] = 1.0
    return out


def collapse_to_global(factors: pd.DataFrame):
    """One (s_up, s_lo) per horizon — 8 numbers total."""
    out = factors.copy()
    out["s_up"] = out["s_up_global"].clip(S_FLOOR, S_CEIL)
    out["s_lo"] = out["s_lo_global"].clip(S_FLOOR, S_CEIL)
    return _enforce_horizon_monotonicity(out)


# --------------------------------------------------------------------------------------
# Lookup table used by apply_recalibration / the audit
# --------------------------------------------------------------------------------------
class ScaleFactorTable:
    """Vectorized (horizon, dhat_bin, hm_bin, biome) -> (s_up, s_lo) lookup."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._dense_cache = {}
        self._map = {
            (int(r.horizon), int(r.dhat_bin_idx), int(r.hm_bin_idx), int(r.biome)):
                (float(r.s_up), float(r.s_lo))
            for r in df.itertuples()
        }
        # Fallbacks in decreasing specificity.
        self._by_hd = df.groupby(["horizon", "dhat_bin_idx"]).apply(
            lambda g: (np.average(g["s_up"], weights=g["n_eff"] + 1),
                       np.average(g["s_lo"], weights=g["n_eff"] + 1)),
            include_groups=False,
        ).to_dict()
        self._by_h = df.groupby("horizon").apply(
            lambda g: (np.average(g["s_up"], weights=g["n_eff"] + 1),
                       np.average(g["s_lo"], weights=g["n_eff"] + 1)),
            include_groups=False,
        ).to_dict()

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(path))

    def to_csv(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        return path

    def _dense(self, horizon, n_d, n_h, n_b):
        """Dense (d_idx, hm_idx, biome) -> factor arrays, built once per horizon.

        Applying the table means evaluating it at every pixel of a global raster; a Python
        loop over the ~450 class combinations with a full-array mask each is what made
        recalibration take minutes per block instead of seconds.
        """
        cached = self._dense_cache.get((horizon, n_d, n_h, n_b))
        if cached is not None:
            return cached
        h = int(horizon)
        fill_up, fill_lo = self._by_h.get(h, (1.0, 1.0))
        up = np.full((n_d, n_h, n_b), float(fill_up))
        lo = np.full((n_d, n_h, n_b), float(fill_lo))
        # Primary stratum first, then the finer cells override where they exist.
        for (hh, d), (u, l) in self._by_hd.items():
            if int(hh) == h and 0 <= int(d) < n_d:
                up[int(d)] = u
                lo[int(d)] = l
        for (hh, d, hm, b), (u, l) in self._map.items():
            if int(hh) == h and 0 <= int(d) < n_d and 0 <= int(hm) < n_h and 0 <= int(b) < n_b:
                up[int(d), int(hm), int(b)] = u
                lo[int(d), int(hm), int(b)] = l
        self._dense_cache[(horizon, n_d, n_h, n_b)] = (up, lo)
        return up, lo

    def lookup(self, horizon, d_idx, h_idx, biome):
        d_idx = np.asarray(d_idx)
        h_idx = np.asarray(h_idx)
        biome = np.asarray(biome)
        n_d = max(int(self.df["dhat_bin_idx"].max()) + 1, int(d_idx.max()) + 1, 1)
        n_h = max(int(self.df["hm_bin_idx"].max()) + 1, int(h_idx.max()) + 1, 1)
        n_b = max(int(self.df["biome"].max()) + 1, int(biome.max()) + 1, 1)
        up, lo = self._dense(horizon, n_d, n_h, n_b)
        di = np.clip(d_idx, 0, n_d - 1)
        hi = np.clip(h_idx, 0, n_h - 1)
        bi = np.clip(biome, 0, n_b - 1)
        return up[di, hi, bi], lo[di, hi, bi]

    def as_scale_fn(self):
        """Adapter for ``compute_class_conditional_coverage(scale_factors=...)``."""
        return lambda horizon, d_idx, h_idx, biome: self.lookup(horizon, d_idx, h_idx, biome)


# --------------------------------------------------------------------------------------
# 1.5d — leave-one-fold-out evaluation
# --------------------------------------------------------------------------------------
def lofo_coverage(store: ScoreStore, alpha: float = 0.05, n0: float = 200.0, **fit_kwargs):
    """Fit ŝ on k−1 folds, score coverage on the held-out fold, rotate, pool.

    The in-sample table always looks good; the gap between the two is the honest measure of
    how much the stratification is overfitting.
    """
    folds = store.folds()
    rows = []
    for f in folds:
        factors = fit_scale_factors(store, alpha=alpha, n0=n0, exclude_fold=f, **fit_kwargs)
        if factors.empty:
            continue
        table = ScaleFactorTable(factors)
        held = store.pooled(keys=None)
        held = {k: v for k, v in store.pooled().items()}  # class keys present overall
        for key in held:
            horizon, d_idx, h_idx, biome = key
            s = store.data.get((key, f))
            if s is None:
                continue
            up = np.concatenate(s.up) if s.up else np.empty(0, np.float32)
            lo = np.concatenate(s.lo) if s.lo else np.empty(0, np.float32)
            if up.size + lo.size == 0:
                continue
            s_up, s_lo = table.lookup(horizon, np.array([d_idx]), np.array([h_idx]), np.array([biome]))
            n_above = int((up > s_up[0]).sum())
            n_below = int((lo > s_lo[0]).sum())
            n_tot = int(up.size + lo.size)
            rows.append({
                "held_out_fold": f, "horizon": horizon,
                "dhat_bin_idx": d_idx, "dhat_bin": DHAT_LABELS[d_idx],
                "hm_bin_idx": h_idx, "hm_bin": HM_LABELS[h_idx], "biome": biome,
                "n_px": n_tot, "n_eff": len(s.chips),
                "n_covered": n_tot - n_above - n_below,
                "n_above_upper": n_above, "n_below_lower": n_below,
                "s_up": float(s_up[0]), "s_lo": float(s_lo[0]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["coverage"] = df["n_covered"] / df["n_px"]
    return df


def pool_lofo(df: pd.DataFrame, by=("horizon", "dhat_bin")):
    if df.empty:
        return df
    g = df.groupby(list(by), observed=True).agg(
        n_px=("n_px", "sum"), n_covered=("n_covered", "sum"),
        n_above_upper=("n_above_upper", "sum"), n_below_lower=("n_below_lower", "sum"),
        n_eff=("n_eff", "sum"),
    ).reset_index()
    g["coverage"] = g["n_covered"] / g["n_px"]
    lo, hi = wilson_interval(g["coverage"] * g["n_eff"], g["n_eff"])
    g["wilson_lo"], g["wilson_hi"] = lo, hi
    return g
