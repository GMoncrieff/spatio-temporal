#!/usr/bin/env python3
"""Rank conv-spline runs against b1's own measured floor.

Reuses the ranking machinery of ``compare_dist_runs.py`` -- the floor, the bar, the
under-powered marking -- and changes only what is being ranked. The bar is worth restating
because it is the one thing the previous phase got wrong: a single new draw falls outside the
range of three baseline runs roughly **half the time under the null**, so "outside the base
range" is barely a test at all. What counts is

    the margin beyond the baseline range exceeds that range's own width,

and anything clearing it is replicated before it means anything.

The metric list is the difference. Two gates now sit alongside CRPS and are not tie-breakers:

``needle_mass_*`` / ``max_density_*`` / ``over_f_max_*``
    the picket fence, read twice. ``_export`` is what a consumer of the product sees and E0
    moves it by construction; ``_ref`` is a property of the forecast distribution and only a
    model change should move it. **An experiment that improves ``_export`` alone changed the
    rendering, not the model** -- the row exists so that cannot be recorded as a fix.

``pit_rms_se_*`` / ``pit_growth_vs_noise`` / ``pit_mean``
    PIT structure. ``rms_se`` is ~1 under a calibrated forecast at every bin count, which is
    what makes 20 and 60 bins comparable; ``growth`` distinguishes structure with resolvable
    width from sampling noise, and misses features narrower than the finest bin (measured:
    0.97 at sigma 0.004), so it is read only after ``rms_se`` says there is structure at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import compare_dist_runs as base  # noqa: E402

# (metric, better direction). "none" means the target is a value rather than an extreme, so
# it is printed without a verdict -- coverage at 0.95, PIT mean at 0.50, a leak ratio at 1.00.
METRICS = [
    # what the model is for
    ("crps_skill5", "up"), ("crps_skill20", "up"),
    ("crps5", "down"), ("crps20", "down"),
    ("rmse5", "down"), ("rmse20", "down"),
    ("skill5", "up"), ("skill20", "up"),
    ("cov95_5", "none"), ("cov95_20", "none"),
    # gate 1: the picket fence. Both readings, always together.
    ("needle_mass_median_ref_5", "down"), ("needle_mass_p90_ref_5", "down"),
    ("max_density_p99_ref_5", "down"), ("over_f_max_frac_ref_5", "down"),
    ("needle_mass_median_export_5", "down"), ("max_density_p99_export_5", "down"),
    ("needle_mass_median_ref_20", "down"), ("max_density_p99_ref_20", "down"),
    # gate 2: PIT structure
    ("pit_rms_se_20_5", "down"), ("pit_rms_se_60_5", "down"),
    ("pit_growth_vs_noise_5", "none"),
    ("pit_mean_5", "none"), ("pit_mean_20", "none"),
    ("pit_ks5", "down"), ("pit_ks20", "down"),
    ("zero_leak_neg_ratio_5", "none"),
    # placement, kept from the previous phase so the two are comparable where they overlap
    ("exceedance_abs_log10", "down"),
    ("tail_reach20", "none"),
]

TARGETS = {
    "cov95_5": 0.95, "cov95_20": 0.95,
    "pit_mean_5": 0.50, "pit_mean_20": 0.50,
    "zero_leak_neg_ratio_5": 1.00,
    "pit_growth_vs_noise_5": 1.00,
}


def main(argv=None):
    """Same CLI as ``compare_dist_runs``, plus ``--floor_prefix`` as a readable alias.

    The underlying flag is a regex (``--baseline '^b1_s\d+$'``). Spelling that out at every
    call site is how a typo silently selects zero baseline runs, at which point the tool
    refuses to rank -- a safe failure, but an obscure one.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--floor_prefix" in argv:
        i = argv.index("--floor_prefix")
        prefix = argv[i + 1]
        del argv[i:i + 2]
        argv += ["--baseline", rf"^{prefix}_s\d+$"]
    base.METRICS = METRICS
    if hasattr(base, "TARGETS"):
        base.TARGETS = TARGETS
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
