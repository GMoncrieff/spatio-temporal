"""The scorecard's two figures, and the HTML page that carries them.

``score_distributional_model.py`` reported the two gates as numbers and nothing else. A
needle-mass of 0.045 and a max density of 677 are the *summary* of the picket fence; the
fence itself is a shape, and the phase's opening evidence (``fitted_densities.png``,
``marginal_change.png``) was a picture. Every experiment therefore writes the same two
pictures beside its numbers:

* the implied density of nine sampled per-pixel forecasts, on a 3x3 grid;
* the PIT histogram in twenty bins, one panel per horizon.

Both are read off the exported quantile-function raster, so they are properties of the
published product rather than of a re-derivation -- the same raster the gates are computed
from, and the same piecewise-linear reading (``dp/dq`` per segment) that ``max_density_p99``
summarises. A figure drawn from a different object than the metric would be the fourth way
this project has found to make a measurement disagree with itself.

The pixels are chosen by a **seeded walk over the fold mask**, not over the run's own finite
pixels, so two experiments scored on the same folds draw the same nine pixels and their
panels can be laid side by side. A pixel the run failed to predict is skipped and the walk
continues, which is the only way the choice can differ between runs -- and worth seeing.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np

from .qf_diagnostics import F_MAX_DENSITY, implied_density, needle_mass

#: Bins for the PIT histogram. Twenty, per docs/conv_spline_phase.md section 3.
PIT_BINS = 20


def _mpl():
    """Import matplotlib with a headless backend, at call time.

    At module scope this would make the scorer depend on matplotlib to compute a CRPS, and
    the scorer runs on machines that have no display and sometimes no matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fitted_densities_figure(u, qf, hm_t0, observed, labels, path, title=""):
    """The 3x3 grid of per-pixel implied densities. ``qf`` is ``[n_levels, 9]``.

    The x axis is *change* (``Q(u) - HM_t0``), because that is the quantity the product
    forecasts and the axis the fence's needles sit on -- against absolute HM the whole core
    collapses into one pixel of the plot. The y axis is logarithmic: the density spans the
    persistence core at several hundred and the tails at well under one, and a linear axis
    shows only the spike.

    Drawn as a step function, which is what the distribution *is*: the raster is a
    piecewise-linear quantile function, so its density is piecewise constant, and smoothing
    it would hide exactly the discontinuities the gate counts.
    """
    plt = _mpl()
    dens = implied_density(u, qf)                       # [n_seg, 9]
    nm = needle_mass(u, qf)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5))
    for k, ax in enumerate(axes.ravel()):
        if k >= qf.shape[1]:
            ax.axis("off")
            continue
        chg = qf[:, k] - hm_t0[k]
        d = dens[:, k]
        # stairs(values, edges), not step(): the density is one value PER SEGMENT and the
        # quantile values are its EDGES, which is exactly what stairs takes. step() would
        # drop the first segment's left edge, and the first segment is the lower tail.
        ax.stairs(d, chg, lw=1.0, color="#1f4e79")
        ax.axhline(F_MAX_DENSITY, color="#c0392b", ls="--", lw=0.9)
        ax.axvline(0.0, color="#888888", ls=":", lw=0.9)
        ax.axvline(observed[k] - hm_t0[k], color="#e67e22", lw=1.4)
        ax.set_yscale("log")
        # The core is ~0.0036 wide against a range above 1.2; drawn full-width the fence is
        # invisible. 2%-98% of the represented range keeps both the core and a real tail.
        lo, hi = np.percentile(chg, [2.0, 98.0])
        pad = max(hi - lo, 1e-4) * 0.15
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(f"{labels[k]}   needle mass {nm[k]:.3f}", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("change in HM", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("implied density", fontsize=8)
    fig.suptitle(title or "fitted per-pixel distributions", fontsize=11)
    fig.text(0.5, 0.005,
             f"orange = observed change   dotted = zero   dashed red = f_max "
             f"({F_MAX_DENSITY:g}), the sharpest the observation noise supports",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def pit_histogram_figure(counts_by_horizon, path, title=""):
    """One 20-bin PIT histogram per horizon. ``counts_by_horizon`` is ``{h: counts[20]}``.

    Plotted as a density so the calibrated reference is the line at 1.0 at every horizon,
    whatever the pixel count -- a raw count axis makes four panels with four different
    reference heights, which is the same reason ``pit_rms_se`` is reported instead of raw
    RMS.
    """
    plt = _mpl()
    hs = sorted(counts_by_horizon)
    fig, axes = plt.subplots(1, max(len(hs), 1), figsize=(3.4 * max(len(hs), 1), 3.2),
                             squeeze=False)
    edges = np.linspace(0.0, 1.0, PIT_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for ax, h in zip(axes[0], hs):
        c = np.asarray(counts_by_horizon[h], dtype=np.float64)
        tot = c.sum()
        dens = c / tot * PIT_BINS if tot else c
        # The PIT mean straight off the histogram: the bin centres are what the picture
        # shows, so the number printed on it is the number the picture supports.
        mean = float((dens * centres).sum() / PIT_BINS) if tot else np.nan
        ax.bar(centres, dens, width=1.0 / PIT_BINS, color="#1f4e79", edgecolor="white",
               linewidth=0.4)
        ax.axhline(1.0, color="#c0392b", ls="--", lw=1.0)
        ax.set_xlim(0, 1)
        ax.set_title(f"h = {h}   PIT mean {mean:.4f}   n = {int(tot):,}", fontsize=8.5)
        ax.set_xlabel("PIT  u* = Q$^{-1}$(y)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("density (uniform = 1)", fontsize=8)
    fig.suptitle(title or "PIT, 20 bins", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _table(rows, header):
    out = ["<table><thead><tr>"]
    out += [f"<th>{html.escape(str(c))}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


CSS = """
body{font:13px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
     margin:0;padding:28px 32px;color:#1a1a1a;background:#fbfbfa;max-width:1200px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:14px;margin:26px 0 8px;
   text-transform:uppercase;letter-spacing:.06em;color:#555}
.sub{color:#666;margin:0 0 18px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12px}
th,td{padding:4px 10px;text-align:right;border-bottom:1px solid #e4e4e0}
th{text-align:right;color:#555;font-weight:600;border-bottom:1px solid #bbb}
td:first-child,th:first-child{text-align:left}
img{max-width:100%;height:auto;border:1px solid #e4e4e0;background:#fff}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}
p.note{color:#666;font-size:12px;max-width:80ch}
code{background:#f0f0ec;padding:1px 4px;border-radius:3px}
"""


def write_scorecard(path, label, meta, tables, images, notes=()):
    """A self-contained HTML scorecard: the tables the scorer prints, and both figures.

    Self-contained (images inlined as data URIs) because these get copied between machines
    and attached to write-ups, and a scorecard whose figures are missing is worse than one
    with none -- it reads as a run where the gates were not measured.
    """
    parts = [f"<title>{html.escape(label)}</title>",
             f"<style>{CSS}</style>",
             f"<h1>{html.escape(label)}</h1>",
             "<p class='sub'>" + " &middot; ".join(
                 f"{html.escape(str(k))}: <code>{html.escape(str(v))}</code>"
                 for k, v in meta.items()) + "</p>"]
    for heading, header, rows in tables:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        # Wrapped: these tables run to twelve numeric columns and are read on phones as well
        # as on a desktop. Only the table scrolls; the page itself must not.
        parts.append(f"<div class='tw'>{_table(rows, header)}</div>")
    for heading, img in images:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.append(f"<img alt='{html.escape(heading)}' "
                     f"src='data:image/png;base64,{_b64(img)}'>")
    for n in notes:
        parts.append(f"<p class='note'>{n}</p>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
    return Path(path)
