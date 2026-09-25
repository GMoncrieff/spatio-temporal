#!/usr/bin/env python3
"""Assert that an arm's quantile-function raster is a usable forecast, off disk.

The smoke runner's teeth. A training step that does not crash proves very little: the defect
that would have cost this phase six arms was a prediction writer that decoded every head with
the *rational-quadratic* decoder, so `pwl` and `isqf` raised at the first prediction batch --
after training. Everything here is therefore read back from the written file, with the
scorer's own reader, rather than from anything still in memory.

Checks, in the order they would fail:
  1. a quantile raster exists at all (the E1/E2/E4 failure)
  2. its u grid is strictly increasing and carries the gate levels (0.025 / 0.5 / 0.975)
  3. its storage scale is tagged and honoured (float32 must carry 1.0)
  4. Q(u) is finite and non-decreasing on every scored pixel
  5. the head emitted the channel count the arm implies (E1a's 18 against E1's 16)
  6. the raster's OWN head_family/head_params tags say which head wrote it
  7. the run's own log carries the fingerprints the verifiers grep for

Check 6 is metadata on a delivered product, not a diagnostic. It was the literal "spline"
for every head until 2026-09-17, so E1v's Africa rasters -- the promoted model's own output
-- carry head_family="spline" while the head is "pwl". That raster is the control this check
was proved against before it was trusted.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from score_distributional_model import qf_levels, read_qf  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--expect_params", type=int, default=0)
    ap.add_argument("--expect_family", default="",
                    help="Head family the arm asked for. Checked against the "
                         "run's own banner, which until 2026-09-15 said "
                         "'Spline head' whatever family was running.")
    ap.add_argument("--expect_tag_family", default="",
                    help="Head family the RASTER's own head_family tag must carry. "
                         "Distinct from --expect_family, which reads the log: a run can "
                         "log the right head and tag the raster with the wrong one, and "
                         "did, for every head, until 2026-09-17.")
    ap.add_argument("--expect_tag_params", type=int, default=0,
                    help="params/horizon the raster's head_params tag must carry.")
    ap.add_argument("--log", default="")
    ap.add_argument("--flags", default="")
    ap.add_argument("--row_chunk", type=int, default=512)
    ap.add_argument("--min_px", type=int, default=200_000,
                    help="Stop once this many finite pixels are checked. A "
                         "monotonicity defect is per-pixel and structural, so "
                         "it shows in the first band that carries data.")
    a = ap.parse_args(argv)

    bad = []
    qfs = sorted(Path(a.pred_dir).glob("*_qf_blended.tif"))
    if not qfs:
        print(f"FATAL: no *_qf_blended.tif in {a.pred_dir} -- the head wrote no quantile "
              f"function. This is exactly how a head that cannot decode fails: it trains, "
              f"then writes nothing.", file=sys.stderr)
        return 1

    for p in qfs:
        try:
            u, nod, scale = qf_levels(str(p))
        except SystemExit as e:
            print(f"FATAL: {p.name}: {e}", file=sys.stderr)
            bad.append(p.name)
            continue
        if not (np.diff(u) > 0).all():
            bad.append(f"{p.name}: u not strictly increasing")
        for gate in (0.025, 0.5, 0.975):
            if not np.isclose(u, gate).any():
                bad.append(f"{p.name}: u grid lost the {gate} gate level")
        # Banded, and it stops as soon as it has seen enough finite pixels to judge. Reading
        # a 64-band Africa raster whole is 16 GB, and a smoke that costs more than the thing
        # it guards does not get run.
        with rasterio.open(str(p)) as src:
            H = src.height
        n, n_back, lo_v, hi_v = 0, 0, np.inf, -np.inf
        for r0 in range(0, H, a.row_chunk):
            _, qb = read_qf(str(p), r0, min(a.row_chunk, H - r0))
            qb = qb.reshape(qb.shape[0], -1)
            ok = np.isfinite(qb).all(axis=0)
            if not ok.any():
                continue
            qq = qb[:, ok]
            n += int(ok.sum())
            n_back += int((np.diff(qq, axis=0) < -1e-6).sum())
            lo_v, hi_v = min(lo_v, float(qq.min())), max(hi_v, float(qq.max()))
            if n >= a.min_px:
                break
        if n == 0:
            bad.append(f"{p.name}: no finite pixel in any row band")
            continue
        if n_back:
            bad.append(f"{p.name}: Q(u) decreases at {n_back} (segment, pixel) pairs")
        if a.expect_tag_family or a.expect_tag_params:
            with rasterio.open(str(p)) as src:
                tags = src.tags()
            if a.expect_tag_family:
                got = tags.get("head_family")
                if got != a.expect_tag_family:
                    bad.append(f"{p.name}: head_family tag is {got!r}, expected "
                               f"{a.expect_tag_family!r}")
            if a.expect_tag_params:
                got = tags.get("head_params")
                if got is None or int(got) != a.expect_tag_params:
                    bad.append(f"{p.name}: head_params tag is {got!r}, expected "
                               f"{a.expect_tag_params}")
        print(f"  ✓ {p.name}: {len(u)} levels, {n:,} finite px checked, "
              f"scale={scale:g}, range [{lo_v:.5f}, {hi_v:.5f}]")

    if a.log and Path(a.log).is_file():
        txt = Path(a.log).read_text(errors="ignore")
        for pat, why in (
            (r"^Context into trunk: (\d+) channels; heads: none", "context wiring"),
            (r"^MSE weight:\s+([0-9.]+)", "loss-weight banner"),
            (r"\[weight averaging\] wrote the mean of the last (\d+) epochs", "weight averaging"),
        ):
            if not re.search(pat, txt, re.M):
                bad.append(f"log carries no {why} fingerprint")
        m = re.search(r"^Context into trunk: (\d+)", txt, re.M)
        if m and int(m.group(1)) != 12:
            bad.append(f"trunk built with {m.group(1)} context channels, expected 12")
        if "--mu_mse_weight 0.0" in a.flags:
            m = re.search(r"^MSE weight:\s+([0-9.]+)", txt, re.M)
            if m and float(m.group(1)) != 0.0:
                bad.append(f"arm asked for --mu_mse_weight 0.0, log says {m.group(1)}")
        if a.expect_family:
            m = re.search(r"^Spline head:\s+family (\w+)", txt, re.M)
            if not m:
                bad.append("log carries no head-family fingerprint")
            elif m.group(1) != a.expect_family:
                bad.append(f"arm asked for head_family={a.expect_family}, log says "
                           f"{m.group(1)}")
        if a.expect_params:
            # The count PRECEDES the phrase: "... slopes learned, 29 params/horizon".
            # Searching forward from the phrase found a stray 0 further down the log and
            # failed the known-good b1 control -- which is precisely why b1 is smoked.
            m = re.search(r"(\d+)\s*params/horizon", txt)
            if m and int(m.group(1)) != a.expect_params:
                bad.append(f"head emitted {m.group(1)} params/horizon, expected "
                           f"{a.expect_params}")
    elif a.log:
        bad.append(f"no readable log at {a.log}")

    if bad:
        print("FATAL: " + "\n       ".join(bad), file=sys.stderr)
        return 1
    print(f"  ✓ smoke OK: {len(qfs)} quantile rasters readable and monotone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
