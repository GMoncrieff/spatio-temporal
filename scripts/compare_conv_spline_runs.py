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
    # RANKED: bounded in [0, 1], cannot saturate, and measured on b1_s42 they discriminate.
    ("needle_mass_median_ref_5", "down"), ("needle_mass_p90_ref_5", "down"),
    ("over_f_max_frac_ref_5", "down"),
    ("needle_mass_median_export_5", "down"),
    # REPORTED, NOT RANKED. max_density is export-limited: measured on b1_s42's float32
    # raster, 47.9% of pixels have their sharpest segment within TWO float32 ULPs and the
    # median is 3.0 ULP, so for about half the pixels this number is the storage format
    # rather than the forecast. It was worse before -- on int16 it read 1531.8 at all four
    # horizons, exactly dp_max/quantum -- and float32 moved the ceiling to ~1e7 without
    # removing it. A p99 that is byte-identical across four window-years is the tell.
    # Read it as a LOWER BOUND beside needle_mass and px_degenerate_frac, never as a value.
    ("max_density_p99_ref_5", "none"), ("max_density_p99_export_5", "none"),
    ("max_density_p50_export_5", "none"),
    # Read the density beside these two or not at all: a max density is only a density while
    # the export can represent the gap. b1_s42 pinned max_density_p99 at dp_max/quantum with
    # 34% of gaps at exactly zero, and the degenerate pixels were being dropped before the
    # statistic was taken -- so the number was a ceiling computed over the survivors.
    ("px_degenerate_frac_export_5", "down"), ("gap_frac_zero_export_5", "down"),
    # The support boundary, reported and NOT ranked: HM piling up at 0 is correct behaviour,
    # it was the whole of b1_s42's apparent degeneracy (0.3% of needle mass against the
    # core's 73.3%), and a column that is a large constant on every arm cannot discriminate.
    ("px_clamp_frac_export_5", "none"),
    ("needle_mass_median_ref_20", "down"), ("max_density_p99_ref_20", "none"),
    # gate 2: PIT structure
    ("pit_rms_se_20_5", "down"), ("pit_rms_se_60_5", "down"),
    ("pit_growth_vs_noise_5", "none"),
    ("pit_mean_5", "none"), ("pit_mean_20", "none"),
    ("pit_ks5", "down"), ("pit_ks20", "down"),
    ("zero_leak_neg_ratio_5", "none"),
    # The far tails. E1a's learned tail rates act only beyond u = 0.001 / 0.999, so without
    # these rows that arm could work perfectly and still read as a null -- the same failure as
    # a flag that is accepted, logged and inert, one level up. Measured on b1's floor these
    # carry a 44% band and will print WEAK; that is reported honestly rather than hidden,
    # and it means E1a must roughly HALVE the excess (2.5x -> ~1.3x) to be readable at all.
    ("far_tail_excess_5", "down"), ("far_tail_excess_20", "down"),
    ("pit_gt_0999_5", "down"), ("pit_lt_0001_5", "down"),
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
