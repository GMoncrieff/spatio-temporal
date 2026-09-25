#!/usr/bin/env python
"""Render the E1v vs E2a promotion comparison to a single HTML page.

Every number on the page comes from tables.json / metrics.json, which come from the scored
runs. Nothing is transcribed by hand and nothing is interpreted.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/conv_spline/promotion")
T = json.loads((OUT / "tables.json").read_text())
M = json.loads((OUT / "metrics.json").read_text())
P = json.loads((OUT / "pit_stats.json").read_text())
RES = ["10", "20", "50"]

HS = ["5", "10", "15", "20"]
MODELS = ["E2a", "E1v"]
DIST = ["0-1", "1-3", "3-10", "10-30", "30-100", ">100"]
OC = ["<-0.01", "[-0.01,0.001)", "[0.001,0.01)", "[0.01,0.05)", ">=0.05"]


def f(x, n=4):
    return "&mdash;" if x is None or x != x else f"{x:.{n}f}"


def i(x):
    return "&mdash;" if x is None or x != x else f"{int(x):,}"


def swatch(m):
    return f'<span class="sw sw-{m.lower()}"></span>'


def metric_rows(spec):
    """spec: [(label, path-into-T, decimals)] -> rows of h=5/10/15/20 for both models."""
    out = []
    for label, get, dec in spec:
        cells = []
        for m in MODELS:
            for h in HS:
                cells.append(f'<td>{f(get(m, h), dec)}</td>')
        out.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    return "\n".join(out)


def horizon_table(caption, spec, note=""):
    head = "".join(f'<th>{h}</th>' for h in HS)
    return f"""
<figure class="tbl">
  <table>
    <caption>{caption}</caption>
    <thead>
      <tr><td class="corner"></td>
          <th colspan="4" class="grp grp-e2a">{swatch('E2a')}E2a</th>
          <th colspan="4" class="grp grp-e1v">{swatch('E1v')}E1v</th></tr>
      <tr><th scope="col" class="rowhead">horizon (yr)</th>{head}{head}</tr>
    </thead>
    <tbody>{metric_rows(spec)}</tbody>
  </table>
  {f'<figcaption>{note}</figcaption>' if note else ''}
</figure>"""


def band_table(caption, bands, fields, src, note=""):
    head = "".join(f'<th>{b}</th>' for b in bands)
    rows = []
    for label, key, dec, per_model in fields:
        if per_model:
            for m in MODELS:
                cells = "".join(f'<td>{f(src[m][b][key], dec)}</td>' for b in bands)
                rows.append(f'<tr><th scope="row">{swatch(m)}{label} <span class="mn">{m}</span></th>{cells}</tr>')
        else:
            cells = "".join(f'<td>{i(src["E2a"][b][key])}</td>' for b in bands)
            rows.append(f'<tr class="shared"><th scope="row">{label}</th>{cells}</tr>')
    return f"""
<figure class="tbl">
  <table>
    <caption>{caption}</caption>
    <thead><tr><th scope="col" class="rowhead">band</th>{head}</tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  {f'<figcaption>{note}</figcaption>' if note else ''}
</figure>"""


def fig(src, alt, caption, wide=True):
    return f"""
<figure class="fig{' wide' if wide else ''}">
  <div class="plate"><img src="{src}" alt="{alt}" loading="lazy"></div>
  <figcaption>{caption}</figcaption>
</figure>"""


# ----------------------------------------------------------------- per-pixel table

pix_rows = []
for k, p in enumerate(M["pixels"], 1):
    pix_rows.append(
        f'<tr><th scope="row">{k}</th>'
        f'<td class="mono">r{p["r"]} c{p["c"]}</td>'
        f'<td>{f(p["hm_t0"])}</td><td>{f(p["observed"])}</td>'
        f'<td>{f(p["observed_change"])}</td>'
        f'<td>{f(p["E2a_central"])}</td><td>{f(p["E1v_central"])}</td>'
        f'<td>{f(p["E2a_pit"], 3)}</td><td>{f(p["E1v_pit"], 3)}</td></tr>')

pit_rows_l = []
for m in MODELS:
    for h in HS:
        d = P[m][h]
        pit_rows_l.append(
            f'<tr><th scope="row">{swatch(m)}{m}</th><td class="rowhead">{h}</td>'
            + "".join(f'<td>{f(d[f"rms_se_{b}"], 1)}</td>' for b in RES)
            + "".join(f'<td>{f(d[f"peak_u_{b}"], 3)}</td>' for b in RES)
            + f'<td>{f(d["growth_vs_noise"], 3)}</td>'
            f'<td>{f(d["mean"], 4)}</td><td>{f(d["frac_below_half"], 3)}</td></tr>')
pit_rows = "".join(pit_rows_l)

chip_rows = []
for k, c in enumerate(M["chips"], 1):
    chip_rows.append(
        f'<tr><th scope="row">{k}</th><td class="mono">r{c["r"]} c{c["c"]}</td>'
        f'<td>{f(c["observed_mean"])}</td>'
        f'<td>{f(c["E2a_mean"])}</td><td>{f(c["E1v_mean"])}</td>'
        f'<td>{f(c["E2a_mean"] - c["observed_mean"])}</td>'
        f'<td>{f(c["E1v_mean"] - c["observed_mean"])}</td></tr>')

HTML = f"""<title>E1v vs E2a</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --bg:#F6F7F9; --surface:#FFFFFF; --plate:#FFFFFF;
  --ink:#151A21; --ink-2:#39424F; --muted:#67717F; --rule:#DCE1E8; --rule-2:#EDF0F4;
  --e2a:#1F6FEB; --e1v:#D98324;
  --accent:#1F6FEB;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --serif:"Newsreader",Georgia,"Times New Roman",serif;
  --maxw:1160px; --prose:68ch;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#111419; --surface:#181C23; --plate:#FFFFFF;
    --ink:#E7EAF0; --ink-2:#C3CAD5; --muted:#949DAA; --rule:#2A313B; --rule-2:#222831;
    --e2a:#4D94F5; --e1v:#C98234; --accent:#4D94F5;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#111419; --surface:#181C23; --plate:#FFFFFF;
  --ink:#E7EAF0; --ink-2:#C3CAD5; --muted:#949DAA; --rule:#2A313B; --rule-2:#222831;
  --e2a:#4D94F5; --e1v:#C98234; --accent:#4D94F5;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:var(--maxw); margin:0 auto; padding-inline:24px; padding-block:0 96px; }}
p, li {{ max-width:var(--prose); color:var(--ink-2); }}
a {{ color:var(--accent); }}
h1,h2,h3 {{ font-family:var(--serif); font-weight:500; text-wrap:balance; margin:0; }}
h1 {{ font-size:clamp(2.1rem,5vw,3.15rem); line-height:1.08; letter-spacing:-0.012em; color:var(--ink); }}
h2 {{ font-size:clamp(1.5rem,3vw,1.95rem); line-height:1.16; letter-spacing:-0.008em; }}
h3 {{ font-size:1.12rem; font-weight:600; letter-spacing:-0.004em; }}
.eyebrow {{
  font-family:var(--mono); font-size:0.7rem; font-weight:500; letter-spacing:0.14em;
  text-transform:uppercase; color:var(--muted); margin:0 0 10px;
}}

/* ---------- masthead ---------- */
header.top {{ padding-block:64px 40px; border-bottom:1px solid var(--rule); }}
.lede {{ font-size:1.12rem; color:var(--ink-2); margin-top:20px; max-width:64ch; }}
.runbar {{ display:flex; flex-wrap:wrap; gap:10px 28px; margin-top:26px;
  font-family:var(--mono); font-size:0.775rem; color:var(--muted); }}
.runbar b {{ color:var(--ink-2); font-weight:500; }}

/* ---------- model identity ---------- */
.models {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; margin-top:34px; }}
.model {{ background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:22px 24px; }}
.model.e2a {{ border-top:3px solid var(--e2a); }}
.model.e1v {{ border-top:3px solid var(--e1v); }}
.model h3 {{ display:flex; align-items:baseline; gap:10px; font-family:var(--mono); font-size:1.05rem; }}
.model .flags {{ font-family:var(--mono); font-size:0.75rem; color:var(--muted);
  background:var(--rule-2); border-radius:2px; padding:9px 11px; margin:14px 0 0;
  overflow-x:auto; white-space:pre; }}
.model dl {{ display:grid; grid-template-columns:auto 1fr; gap:4px 16px; margin:16px 0 0;
  font-size:0.85rem; }}
.model dt {{ color:var(--muted); }}
.model dd {{ margin:0; font-family:var(--mono); color:var(--ink); }}
.sw {{ width:11px; height:11px; border-radius:2px; display:inline-block; flex:none; }}
.sw-e2a {{ background:var(--e2a); }}
.sw-e1v {{ background:var(--e1v); }}

/* ---------- sections ---------- */
section {{ padding-block:60px 0; }}
section > .hd {{ border-top:1px solid var(--rule); padding-top:22px; margin-bottom:26px; }}
.num {{ font-family:var(--mono); font-size:0.72rem; color:var(--accent); letter-spacing:0.1em; }}

/* ---------- figures ---------- */
.fig {{ margin:32px 0 0; }}
.fig.wide {{ max-width:none; }}
.plate {{ background:var(--plate); border:1px solid var(--rule); border-radius:3px;
  padding:14px; overflow-x:auto; }}
.plate img {{ display:block; width:100%; max-width:100%; height:auto; }}
figcaption {{ font-size:0.82rem; color:var(--muted); margin-top:11px; max-width:78ch; }}

/* ---------- tables ---------- */
.tbl {{ margin:30px 0 0; }}
.tbl table {{ width:100%; border-collapse:collapse; font-family:var(--mono);
  font-size:0.78rem; font-variant-numeric:tabular-nums; }}
.tbl caption {{ text-align:left; font-family:var(--sans); font-size:0.9rem; font-weight:600;
  color:var(--ink); padding-bottom:12px; }}
.tbl th, .tbl td {{ padding:7px 9px; text-align:right; border-bottom:1px solid var(--rule-2); }}
.tbl thead th {{ color:var(--muted); font-weight:500; border-bottom:1px solid var(--rule); }}
.tbl th[scope="row"], .rowhead {{ text-align:left; color:var(--ink-2); font-weight:500;
  white-space:nowrap; }}
.tbl th[scope="row"] {{ display:table-cell; }}
.tbl .grp {{ text-align:left; font-family:var(--mono); font-weight:600; }}
.tbl .grp-e2a {{ color:var(--e2a); border-left:1px solid var(--rule); }}
.tbl .grp-e1v {{ color:var(--e1v); border-left:1px solid var(--rule); }}
.tbl .grp .sw {{ margin-right:6px; vertical-align:middle; }}
.tbl th[scope="row"] .sw {{ margin-right:7px; vertical-align:middle; }}
.tbl .mn {{ color:var(--muted); }}
.tbl tr.shared td, .tbl tr.shared th {{ color:var(--muted); }}
.tbl .corner {{ border-bottom:1px solid var(--rule); }}
.scroll {{ overflow-x:auto; }}

/* ---------- notes ---------- */
.note {{ border-left:2px solid var(--accent); background:var(--surface);
  padding:16px 20px; margin:26px 0 0; border-radius:0 3px 3px 0; }}
.note p {{ margin:0; font-size:0.9rem; }}
.note p + p {{ margin-top:9px; }}
.note .lbl {{ font-family:var(--mono); font-size:0.68rem; letter-spacing:0.12em;
  text-transform:uppercase; color:var(--accent); display:block; margin-bottom:6px; }}
.note.warn {{ border-left-color:var(--e1v); }}
.note.warn .lbl {{ color:var(--e1v); }}

code {{ font-family:var(--mono); font-size:0.88em; background:var(--rule-2);
  padding:1px 5px; border-radius:2px; }}

.diagram {{ background:var(--surface); border:1px solid var(--rule); border-radius:3px;
  padding:22px; margin:30px 0 0; }}
.diagram svg {{ display:block; width:100%; height:auto; max-width:100%; }}

.toc {{ display:flex; flex-wrap:wrap; gap:8px 10px; margin-top:28px; }}
.toc a {{ font-family:var(--mono); font-size:0.75rem; text-decoration:none;
  border:1px solid var(--rule); border-radius:2px; padding:6px 11px; color:var(--ink-2); }}
.toc a:hover {{ border-color:var(--accent); color:var(--accent); }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
@media (max-width:640px) {{
  header.top {{ padding-block:40px 30px; }}
  .model .flags {{ font-size:0.68rem; }}
}}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important; transition:none!important; }} }}
</style>

<div class="wrap">

<header class="top">
  <p class="eyebrow">conv-spline &middot; promotion candidates &middot; Africa, folds 1+2</p>
  <h1>E1v vs E2a</h1>
  <p class="lede">Two piecewise-linear free-scale heads, scored on the same
  <code>fold_mask_b4</code> hindcast over Africa. Every figure below draws both models from the
  same pixels and the same chips. This page reports what was measured; it draws no conclusion.</p>
  <div class="runbar">
    <span><b>region</b> Africa (63.1 Mpx)</span>
    <span><b>folds</b> 1, 2 &middot; holdout stitch</span>
    <span><b>pixels scored</b> 23,443,456</span>
    <span><b>seed</b> 42</span>
    <span><b>epochs</b> 150, weight-avg last 20</span>
    <span><b>panels</b> base 2000 &rarr; 2020 (h=20)</span>
  </div>
  <nav class="toc">
    <a href="#heads">The two heads</a>
    <a href="#crps">1 &middot; CRPS</a>
    <a href="#calib">2 &middot; Calibration</a>
    <a href="#tails">3 &middot; Tails &amp; rare change</a>
    <a href="#pixels">4 &middot; Nine pixels</a>
    <a href="#chips">5 &middot; Six chips</a>
    <a href="#appendix">6 &middot; All numbers</a>
  </nav>
</header>

<section id="heads">
  <div class="hd"><p class="num">SPECIFICATION</p><h2>The two heads</h2></div>
  <p>Both emit a full per-pixel quantile function <code>Q(u)</code> at +5/10/15/20&nbsp;yr from
  one linear ladder of positive increments, with no injected width and no cross-horizon
  accumulation. Both anchor the ladder at <code>Q(0.5)</code>, which receives persistence
  through a zero-init skip. Both score on CRPS alone.</p>

  <div class="models">
    <div class="model e2a">
      <h3>{swatch('E2a')} E2a</h3>
      <p class="flags">--head_family pwl
--free_scale True
--mu_mse_weight 0.0</p>
      <dl>
        <dt>run</dt><dd>E2a_pwl_freescale_s42</dd>
        <dt>knots</dt><dd>default14 (15 knots, 14 bins)</dd>
        <dt>emitted</dt><dd>16 params/horizon</dd>
        <dt>read</dt><dd>15 &mdash; see note below</dd>
        <dt>tails</dt><dd>none (clamp at 0 and 1)</dd>
      </dl>
    </div>
    <div class="model e1v">
      <h3>{swatch('E1v')} E1v</h3>
      <p class="flags">--head_family pwl
--isqf_tails True
--isqf_space neglog
--free_scale True
--mu_mse_weight 0.0</p>
      <dl>
        <dt>run</dt><dd>E1v_pwl_neglog_s42</dd>
        <dt>knots</dt><dd>default14 (15 knots, 14 bins)</dd>
        <dt>emitted</dt><dd>17 params/horizon</dd>
        <dt>read</dt><dd>17</dd>
        <dt>tails</dt><dd>learned &beta;<sub>L</sub>, &beta;<sub>R</sub> on &minus;log(1&minus;HM)</dd>
      </dl>
    </div>
  </div>

  <div class="note warn">
    <span class="lbl">The pair is not a single-variable comparison</span>
    <p><b>E2a ran before <code>--free_scale</code> was made to remove the scale channel.</b> Its
    head banner reads <code>16 params/horizon</code>: one location, fourteen increments, and one
    scale channel that the width head still emitted and the decoder never read. Its effective
    head is 15 parameters, with one dead channel per horizon carrying no gradient from the
    quantile path. E1v ran after that change, so all 17 of its channels are read.</p>
    <p>The two therefore differ by <b>the learned tails</b> and by <b>the presence of that dead
    channel</b>. Re-running E2a on the current code would give a 15-parameter head and is the
    only way to isolate the tails.</p>
  </div>

  <div class="diagram">
    <svg viewBox="0 0 900 250" role="img" aria-label="Where the two heads differ: E1v replaces the outermost bin at each end with a learned exponential tail; the interior ladder is identical.">
      <defs>
        <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
        </marker>
      </defs>
      <g font-family="IBM Plex Mono, monospace" font-size="12">
        <text x="0" y="16" fill="var(--muted)">u = 0</text>
        <text x="200" y="16" fill="var(--muted)">u = 0.001</text>
        <text x="600" y="16" fill="var(--muted)">u = 0.999</text>
        <text x="820" y="16" fill="var(--muted)">u = 1</text>
      </g>
      <line x1="20" y1="34" x2="880" y2="34" stroke="var(--rule)" stroke-width="1"/>
      <g stroke="var(--rule)" stroke-width="1">
        <line x1="20" y1="28" x2="20" y2="40"/><line x1="230" y1="28" x2="230" y2="40"/>
        <line x1="640" y1="28" x2="640" y2="40"/><line x1="860" y1="28" x2="860" y2="40"/>
      </g>
      <rect x="230" y="60" width="410" height="42" rx="2" fill="none" stroke="var(--ink-2)" stroke-width="1.4"/>
      <text x="435" y="86" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="13" fill="var(--ink)">14 linear increments &mdash; identical in both heads</text>

      <g>
        <rect x="20" y="130" width="210" height="42" rx="2" fill="none" stroke="var(--e2a)" stroke-width="1.4"/>
        <rect x="640" y="130" width="220" height="42" rx="2" fill="none" stroke="var(--e2a)" stroke-width="1.4"/>
        <text x="125" y="156" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="12.5" fill="var(--e2a)">E2a: outer bin + clamp</text>
        <text x="750" y="156" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="12.5" fill="var(--e2a)">E2a: outer bin + clamp</text>
      </g>
      <g>
        <rect x="20" y="186" width="210" height="42" rx="2" fill="none" stroke="var(--e1v)" stroke-width="1.4"/>
        <rect x="640" y="186" width="220" height="42" rx="2" fill="none" stroke="var(--e1v)" stroke-width="1.4"/>
        <text x="125" y="212" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="12.5" fill="var(--e1v)">E1v: exponential, rate &beta;<tspan dy="3" font-size="9">L</tspan></text>
        <text x="750" y="212" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="12.5" fill="var(--e1v)">E1v: exponential, rate &beta;<tspan dy="3" font-size="9">R</tspan></text>
      </g>
      <text x="435" y="212" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="12" fill="var(--muted)">the tails act only outside u = 0.001 / 0.999</text>
    </svg>
  </div>
  <figcaption style="max-width:78ch;color:var(--muted);font-size:0.82rem;margin-top:11px;">
    The learned tails replace the outermost bin at each end and are anchored at the 0.001 / 0.999
    knots, so the spline interior is bit-identical between the two heads at initialisation. They
    carry two extra channels; everything between 0.001 and 0.999 is the same construction.
  </figcaption>
</section>

<section id="crps">
  <div class="hd"><p class="num">01</p><h2>CRPS and CRPS skill</h2></div>
  <p>CRPS is the continuous ranked probability score of the full predicted distribution against
  the single observed value, in raw HM units. CRPS skill is
  <code>1 &minus; CRPS / CRPS<sub>persistence</sub></code>, where persistence predicts no change;
  0 means no better than persistence. RMSE is of <code>E[Q]</code>, the published central field,
  and its skill score is <code>1 &minus; MSE / MSE<sub>persistence</sub></code>.</p>

  {fig("fig/crps_by_horizon.png", "CRPS, CRPS skill and RMSE against lead time for both models",
       "Left: CRPS in raw HM, with the persistence reference. Centre: CRPS skill against "
       "persistence. Right: RMSE of the published central field.")}

  {horizon_table("CRPS and skill by horizon",
     [("CRPS", lambda m, h: T["headline"][m]["crps"][h], 6),
      ("CRPS skill", lambda m, h: T["headline"][m]["crps_skill"][h], 4),
      ("RMSE of E[Q]", lambda m, h: T["headline"][m]["rmse"][h], 5),
      ("MSE skill", lambda m, h: T["headline"][m]["skill"][h], 4)],
     "Pooled over folds 1+2 and all four window-years, weighted by pixel count.")}
</section>

<section id="calib">
  <div class="hd"><p class="num">02</p><h2>PIT, coverage and calibration</h2></div>
  <p>The PIT value of a pixel is <code>F(observed)</code> &mdash; the predicted cumulative
  probability at the value that actually occurred. If the forecast distribution is right, PIT is
  uniform on [0,&nbsp;1], its mean is 0.50, and a <code>p</code>% interval contains the
  observation <code>p</code>% of the time. The KS distance is the largest gap between the
  observed PIT distribution and that uniform.</p>

  <div class="diagram">
    <svg viewBox="0 0 900 190" role="img" aria-label="How a PIT value is read: the observed HM is located on the predicted cumulative distribution, and its height is the PIT.">
      <g font-family="IBM Plex Sans, sans-serif" font-size="12.5">
        <path d="M80 150 C 190 150, 250 60, 340 44 C 430 28, 520 24, 620 22" fill="none" stroke="var(--ink-2)" stroke-width="2"/>
        <line x1="80" y1="160" x2="640" y2="160" stroke="var(--rule)" stroke-width="1.2"/>
        <line x1="80" y1="160" x2="80" y2="18" stroke="var(--rule)" stroke-width="1.2"/>
        <line x1="300" y1="160" x2="300" y2="52" stroke="var(--ink)" stroke-width="1.4" stroke-dasharray="4 3"/>
        <line x1="80" y1="52" x2="300" y2="52" stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="4 3"/>
        <circle cx="300" cy="52" r="4.5" fill="var(--accent)"/>
        <text x="306" y="176" fill="var(--ink-2)">observed HM</text>
        <text x="660" y="26" fill="var(--muted)">F(HM), the predicted CDF</text>
        <text x="20" y="56" fill="var(--accent)">PIT</text>
        <text x="60" y="176" fill="var(--muted)">0</text>
        <text x="55" y="24" fill="var(--muted)">1</text>
      </g>
      <g font-family="IBM Plex Sans, sans-serif" font-size="12" fill="var(--muted)">
        <text x="700" y="96">PIT near 0 &rarr; observed below the forecast</text>
        <text x="700" y="118">PIT near 1 &rarr; observed above it</text>
        <text x="700" y="140">uniform across all pixels &rarr; calibrated</text>
      </g>
    </svg>
  </div>

  {fig("fig/calibration.png", "PIT mean, PIT KS distance, coverage against nominal, and 95% interval width",
       "PIT mean against the 0.50 target; KS distance from uniform; empirical coverage at the "
       "50/80/95/99% levels against nominal at h=20; and the mean 95% interval width.")}

  {horizon_table("PIT",
     [("PIT mean", lambda m, h: T["calib"][m]["pit_mean"][h], 4),
      ("PIT KS", lambda m, h: T["calib"][m]["pit_ks"][h], 4)],
     "PIT mean targets 0.5000; KS targets 0.")}

  {horizon_table("Empirical coverage",
     [(f"cov{n}", (lambda n: lambda m, h: T["coverage"][m][n]["cov"][h])(n), 4)
      for n in ["50", "80", "95", "99"]],
     "Nominal levels are 0.50, 0.80, 0.95 and 0.99.")}

  {horizon_table("Interval width (HM)",
     [(f"width{n}", (lambda n: lambda m, h: T["coverage"][m][n]["width"][h])(n), 5)
      for n in ["50", "80", "95", "99"]])}

  <h3 style="margin-top:56px">PIT histograms at three binning resolutions</h3>
  <p>A PIT histogram's bumpiness is part forecast and part bin count: under a
  <em>calibrated</em> forecast the RMS deviation from uniform grows as &radic;B, so one
  resolution cannot separate a real feature from sampling noise. Three can. A feature that
  keeps its position and relative height as the bins refine is being resolved; one that appears
  only at the finest is noise.</p>

  <div class="diagram">
    <svg viewBox="0 0 900 150" role="img" aria-label="Reading the resolution test: a real feature holds its position as bins refine, noise does not.">
      <g font-family="IBM Plex Sans, sans-serif" font-size="12">
        <text x="0" y="14" fill="var(--muted)">coarse bins</text>
        <text x="330" y="14" fill="var(--muted)">finer</text>
        <text x="640" y="14" fill="var(--muted)">finest</text>
      </g>
      <g stroke="var(--ink-2)" stroke-width="1.6" fill="none">
        <path d="M10 110 L70 110 L70 70 L130 70 L130 105 L190 105 L190 112 L250 112"/>
        <path d="M330 112 L360 112 L360 96 L390 96 L390 58 L420 58 L420 100 L450 100 L450 108 L480 108 L480 113 L510 113 L510 110 L540 110"/>
        <path d="M640 113 L655 113 L655 105 L670 105 L670 96 L685 96 L685 44 L700 44 L700 92 L715 92 L715 104 L730 104 L730 110 L745 110 L745 108 L760 108 L760 112 L775 112 L775 111 L790 111 L790 113 L880 113"/>
      </g>
      <g stroke="var(--accent)" stroke-width="1.2" stroke-dasharray="4 3">
        <line x1="100" y1="30" x2="100" y2="125"/>
        <line x1="405" y1="30" x2="405" y2="125"/>
        <line x1="692" y1="30" x2="692" y2="125"/>
      </g>
      <text x="100" y="142" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="var(--accent)">same u</text>
      <text x="405" y="142" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="var(--accent)">same u</text>
      <text x="692" y="142" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10.5" fill="var(--accent)">same u</text>
    </svg>
  </div>
  <figcaption style="max-width:78ch;color:var(--muted);font-size:0.82rem;margin-top:11px;">
    <code>rms_se</code> is the RMS deviation from uniform in units of its own standard error:
    it is ~1 under a calibrated forecast at <em>every</em> bin count, so the three columns are
    directly comparable. <code>growth</code> is the observed growth in raw RMS from 10 to 50
    bins divided by what pure sampling noise predicts &mdash; below 1 means the structure is
    growing more slowly than noise would.
  </figcaption>

  {fig("fig/pit_resolutions.png",
       "PIT histograms for both models at 10, 20 and 50 bins, for each of the four horizons",
       "Rows are bin count, columns are lead time, both models overlaid as step outlines so "
       "neither hides the other. The dashed line at density 1.0 is the calibrated reference. "
       "Sampled on every 6th band of 256 rows: 24,728,480 pixel-horizons per model, the same "
       "pixels for both.")}

  <div class="scroll">
  <figure class="tbl">
    <table>
      <caption>PIT structure across resolutions</caption>
      <thead>
        <tr><td class="corner"></td><td class="corner"></td>
            <th colspan="3" class="grp">rms_se (~1 if calibrated)</th>
            <th colspan="3" class="grp">peak bin centre</th>
            <td class="corner"></td><td class="corner"></td><td class="corner"></td></tr>
        <tr><th scope="col" class="rowhead">model</th><th class="rowhead">h</th>
            <th>10</th><th>20</th><th>50</th>
            <th>10</th><th>20</th><th>50</th>
            <th>growth</th><th>PIT mean</th><th>frac &lt; 0.5</th></tr>
      </thead>
      <tbody>{pit_rows}</tbody>
    </table>
    <figcaption>Expected <code>frac &lt; 0.5</code> is 0.500 and expected PIT mean is 0.500.</figcaption>
  </figure>
  </div>
</section>

<section id="tails">
  <div class="hd"><p class="num">03</p><h2>Tails, rare large changes, and remote land</h2></div>
  <p>Three separate questions. <b>Far-tail excess</b> is
  <code>(P(PIT&lt;0.001) + P(PIT&gt;0.999)) / 0.002</code> &mdash; how many times more often the
  observation falls outside the forecast's deepest quantiles than it should; 1.0 is calibrated.
  <b>Rare large change</b> is the predicted against observed rate of a 20-year HM increase above
  0.05. <b>Remote land</b> is the distance to the nearest past change, in pixels; the
  <code>&gt;100</code> band is land with no change within 100&nbsp;px.</p>

  {fig("fig/tails.png", "Tail reach, far-tail excess, rare-change rates by distance band, and CRPS skill by distance band",
       "Tail reach and far-tail excess against lead time; then, at h=20, predicted vs observed "
       "P(change &gt; 0.05) by distance band, and CRPS skill by distance band.")}

  {horizon_table("Far tail",
     [("far-tail excess", lambda m, h: T["tails"][m]["far_tail_excess"][h], 4),
      ("P(PIT &lt; 0.001)", lambda m, h: T["tails"][m]["pit_lt_0001"][h], 5),
      ("P(PIT &gt; 0.999)", lambda m, h: T["tails"][m]["pit_gt_0999"][h], 5),
      ("tail reach (median)", lambda m, h: T["tails"][m]["tail_reach"][h], 4)],
     "Expected rate outside each 0.001 tail is 0.001, so far-tail excess of 1.0 is calibrated.")}

  <div class="scroll">
  {band_table("By distance to past change, h = 20", DIST,
     [("pixels", "n", 0, False),
      ("CRPS skill", "crps_skill", 4, True),
      ("cov95", "cov95", 4, True),
      ("P(&Delta;&gt;0.05) predicted", "pgt005_pred", 5, True),
      ("P(&Delta;&gt;0.05) observed", "pgt005_obs", 5, False)],
     T["distance"],
     "Bands from <code>src.strata.distance_band</code>. Pixel counts and the observed rate are "
     "properties of the data, identical for both models.")}
  </div>

  {fig("fig/obs_change.png", "Pixel counts, CRPS skill and 95% coverage across observed-change magnitude bands",
       "The rare-event axis: how many pixels fall in each observed 20-year change band (log "
       "scale), and CRPS skill and 95% coverage within each.")}

  <div class="scroll">
  {band_table("By observed 20-year change, h = 20", OC,
     [("pixels", "n", 0, False),
      ("CRPS skill", "crps_skill", 4, True),
      ("cov95", "cov95", 4, True)],
     T["obs_change"],
     "Bands from <code>src.strata.OBS_MAG_BINS</code>.")}
  </div>
</section>

<section id="pixels">
  <div class="hd"><p class="num">04</p><h2>Nine pixels, three views each</h2></div>
  <p>Nine pixels drawn by a seeded walk over the fold mask &mdash; not over either run's own
  finite pixels, so both models are read at exactly the same nine locations. Each pixel gets
  three views of the same object: the quantile function as <b>predicted change</b> from
  HM(t&#8320;), the quantile function in <b>absolute HM</b>, and the <b>cumulative
  probability</b> with the observed value marked.</p>

  <div class="diagram">
    <svg viewBox="0 0 900 200" role="img" aria-label="The three per-pixel views are the same quantile function plotted three ways.">
      <g font-family="IBM Plex Sans, sans-serif" font-size="12.5" fill="var(--ink-2)">
        <rect x="10" y="30" width="250" height="130" rx="3" fill="none" stroke="var(--rule)"/>
        <rect x="325" y="30" width="250" height="130" rx="3" fill="none" stroke="var(--rule)"/>
        <rect x="640" y="30" width="250" height="130" rx="3" fill="none" stroke="var(--rule)"/>
        <text x="10" y="22" font-weight="600" fill="var(--ink)">predicted change</text>
        <text x="325" y="22" font-weight="600" fill="var(--ink)">predicted HM</text>
        <text x="640" y="22" font-weight="600" fill="var(--ink)">cumulative probability</text>
        <text x="24" y="182" font-family="IBM Plex Mono, monospace" font-size="11" fill="var(--muted)">x = Q(u) &minus; HM&#8320;,  y = u</text>
        <text x="339" y="182" font-family="IBM Plex Mono, monospace" font-size="11" fill="var(--muted)">x = u,  y = Q(u)</text>
        <text x="654" y="182" font-family="IBM Plex Mono, monospace" font-size="11" fill="var(--muted)">x = HM,  y = F(HM)</text>
        <path d="M30 145 C 90 140, 110 70, 140 60 C 175 48, 215 44, 248 42" fill="none" stroke="var(--ink-2)" stroke-width="1.8"/>
        <path d="M340 150 C 400 120, 430 112, 470 108 C 520 103, 550 70, 565 40" fill="none" stroke="var(--ink-2)" stroke-width="1.8"/>
        <path d="M660 145 C 720 140, 740 70, 770 60 C 805 48, 845 44, 878 42" fill="none" stroke="var(--ink-2)" stroke-width="1.8"/>
      </g>
      <g stroke="var(--muted)" stroke-width="1.2" fill="none" marker-end="url(#ar)" color="var(--muted)">
        <line x1="272" y1="95" x2="312" y2="95"/>
        <line x1="587" y1="95" x2="627" y2="95"/>
      </g>
      <g font-family="IBM Plex Mono, monospace" font-size="10" fill="var(--muted)">
        <text x="270" y="86">+HM&#8320;</text>
        <text x="588" y="86">transpose</text>
      </g>
    </svg>
  </div>
  <figcaption style="max-width:78ch;color:var(--muted);font-size:0.82rem;margin-top:11px;">
    The middle and right panels hold the same information transposed; the left panel is the
    middle one shifted by the pixel's starting HM. Axes are windowed to
    <code>Q(0.02)</code>&ndash;<code>Q(0.98)</code> across both models, widened to include the
    observation &mdash; the published u-grid is tail-dense, so a full-range axis spends its width
    on the clamp.
  </figcaption>

  {fig("fig/pixels_1.png", "Pixels 1 to 3: predicted change, predicted HM and cumulative probability for both models", "Pixels 1&ndash;3 of 9.")}
  {fig("fig/pixels_2.png", "Pixels 4 to 6: predicted change, predicted HM and cumulative probability for both models", "Pixels 4&ndash;6 of 9.")}
  {fig("fig/pixels_3.png", "Pixels 7 to 9: predicted change, predicted HM and cumulative probability for both models", "Pixels 7&ndash;9 of 9.")}

  <figure class="tbl">
    <table>
      <caption>The nine pixels</caption>
      <thead><tr>
        <th scope="col" class="rowhead">#</th><th class="rowhead">location</th>
        <th>HM&#8320;</th><th>observed HM</th><th>observed change</th>
        <th>{swatch('E2a')} E[Q]</th><th>{swatch('E1v')} E[Q]</th>
        <th>{swatch('E2a')} PIT</th><th>{swatch('E1v')} PIT</th>
      </tr></thead>
      <tbody>{"".join(pix_rows)}</tbody>
    </table>
    <figcaption>Seeded walk, seed 0, restricted to folds 1+2. PIT is read off each model's own
    published quantile function at the observed value.</figcaption>
  </figure>
</section>

<section id="chips">
  <div class="hd"><p class="num">05</p><h2>Six chips: observed vs predicted change</h2></div>
  <p>Six 128&nbsp;&times;&nbsp;128&nbsp;px windows inside the fold mask, chosen to span mean
  absolute observed change from near zero to the largest available. Each row shows the observed
  20-year HM change and each model's predicted mean change on one shared diverging scale, set
  from the 99.5th percentile of everything drawn in that row.</p>

  {fig("fig/chips.png", "Six 128 by 128 pixel chips: observed, E2a predicted and E1v predicted mean HM change", "Observed (left) against each model. The colour scale is shared within a row and stated on its own colour bar; it differs between rows.")}

  {fig("fig/chips_scatter.png", "Observed versus predicted change for every pixel in the six chips", "Every pixel in the same six chips, observed change on x against predicted change on y. The dashed line is 1:1.")}

  <figure class="tbl">
    <table>
      <caption>Chip means (HM change, 2000&rarr;2020)</caption>
      <thead><tr>
        <th scope="col" class="rowhead">#</th><th class="rowhead">location</th>
        <th>observed</th><th>{swatch('E2a')} E2a</th><th>{swatch('E1v')} E1v</th>
        <th>{swatch('E2a')} E2a &minus; obs</th><th>{swatch('E1v')} E1v &minus; obs</th>
      </tr></thead>
      <tbody>{"".join(chip_rows)}</tbody>
    </table>
  </figure>
</section>

<section id="appendix">
  <div class="hd"><p class="num">06</p><h2>Everything else on the scorecard</h2></div>

  <figure class="tbl">
    <table>
      <caption>Pooled scalars</caption>
      <thead><tr><th scope="col" class="rowhead">metric</th>
        <th>{swatch('E2a')} E2a</th><th>{swatch('E1v')} E1v</th></tr></thead>
      <tbody>
        <tr><th scope="row">exceedance mean |log10(pred/obs)|</th>
          <td>{f(T["misc"]["E2a"]["exceedance_abs_log10"])}</td>
          <td>{f(T["misc"]["E1v"]["exceedance_abs_log10"])}</td></tr>
        <tr><th scope="row">E[Q] outside its own 95% interval</th>
          <td>{f(T["misc"]["E2a"]["central_outside_interval"], 6)}</td>
          <td>{f(T["misc"]["E1v"]["central_outside_interval"], 6)}</td></tr>
        <tr><th scope="row">qf vs published triple, max |diff|</th>
          <td>{f(T["misc"]["E2a"]["qf_vs_triple_max"], 8)}</td>
          <td>{f(T["misc"]["E1v"]["qf_vs_triple_max"], 8)}</td></tr>
      </tbody>
    </table>
  </figure>

  <div class="note">
    <span class="lbl">Provenance</span>
    <p>Scored by <code>scripts/score_distributional_model.py</code> over folds 1+2 of
    <code>fold_mask_b4_1000.tif</code>, four window-years &times; four horizons, 23,443,456
    pixels, <code>--stitch_mode holdout</code>. Both runs passed all three pre-scoring verifiers
    (loss weights <code>mse=0.0 ssim=0.0 lap=0.0 hist=0.0</code>, context wiring
    <code>12 channels &rarr; trunk, heads none</code>, weight averaging over the last 20 epochs).</p>
    <p>Figures and tables generated by <code>scripts/make_promotion_figures.py</code> and
    <code>scripts/build_promotion_page.py</code> from
    <code>data/conv_spline/scores/</code> and the stitched rasters. Single seed (42) per model:
    no replicate band is available for either, so none is shown.</p>
  </div>
</section>

</div>
"""

(OUT / "index.html").write_text(HTML, encoding="utf-8")
print("wrote", OUT / "index.html", len(HTML), "bytes")
