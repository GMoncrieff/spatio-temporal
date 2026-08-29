"""`compare_dist_runs.py --group_seeds` must judge a variant on its median, not its range.

The Africa floor measured a three-seed *range* swinging 17.5x across draws from one
configuration, so a range is not a statistic that can rank anything. The median is (its
sampling spread over the same six-seed floor was 1.1x). These tests pin three things a
mistake here would silently change:

* the median is taken **per metric**, not by picking one seed's row -- a variant can be the
  middle run on `crps_skill5` and the extreme one on `tail_reach20`;
* the bar is applied to the collapsed median, so per-seed values straddling the floor no
  longer produce a hit; and
* `--include` keeps a region's variants apart from another region's, because the floor is a
  property of the configuration *and* the extent it was measured on.
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compare_dist_runs import collapse_seeds, main  # noqa: E402


def test_collapse_takes_the_median_per_metric_not_a_whole_row():
    df = pd.DataFrame(
        {"crps_skill5": [0.10, 0.20, 0.30], "tail_reach20": [9.0, 1.0, 5.0]},
        index=["e1_s45", "e1_s46", "e1_s47"],
    )
    med, n = collapse_seeds(df)
    assert list(med.index) == ["e1"]
    assert n["e1"] == 3
    # 0.20 comes from s46 and 5.0 from s47: no single run supplies both.
    assert med.loc["e1", "crps_skill5"] == pytest.approx(0.20)
    assert med.loc["e1", "tail_reach20"] == pytest.approx(5.0)


def test_collapse_leaves_a_seedless_label_alone():
    df = pd.DataFrame({"crps_skill5": [0.1, 0.2, 0.3]}, index=["d2", "e1_s45", "e1_s46"])
    med, n = collapse_seeds(df)
    assert sorted(med.index) == ["d2", "e1"]
    assert n["d2"] == 1 and n["e1"] == 2


def _write(tmp, label, **metrics):
    with open(os.path.join(tmp, f"summary_{label}.json"), "w") as fh:
        json.dump({"label": label, **metrics}, fh)


def _bar_line(out, metric):
    """The one line of the bar table for this metric (table rows start with a label)."""
    hits = [ln for ln in out.splitlines() if ln.startswith(metric + " ")]
    assert len(hits) == 1, hits
    return hits[0]


def _floor(tmp):
    # width 0.010, so the bar is "beat 0.120 by more than 0.010".
    for seed, v in zip(range(42, 48), [0.110, 0.112, 0.115, 0.117, 0.119, 0.120]):
        _write(tmp, f"afw_s{seed}", crps_skill5=v, tail_reach20=6.0)


def test_the_bar_applies_to_the_median_not_to_any_one_seed(tmp_path, capsys):
    tmp = str(tmp_path)
    _floor(tmp)
    # Two seeds clear the bar outright; the third is inside the floor. Median 0.128 does not
    # clear (needs > 0.130), so ungrouped this variant is a hit and grouped it is not.
    for seed, v in zip([45, 46, 47], [0.145, 0.128, 0.118]):
        _write(tmp, f"af_x_s{seed}", crps_skill5=v, tail_reach20=6.0)

    main(["--score_dir", tmp, "--baseline", r"^afw_s\d+$"])
    bar = _bar_line(capsys.readouterr().out, "crps_skill5")
    assert "af_x_s45 0.14500" in bar

    main(["--score_dir", tmp, "--baseline", r"^afw_s\d+$", "--group_seeds"])
    out = capsys.readouterr().out
    assert "af_x" not in _bar_line(out, "crps_skill5")
    assert "nothing clears the bar" in out
    assert "af_x" in out   # still reported, just not as a hit


def test_include_keeps_another_regions_runs_out_of_the_ranking(tmp_path, capsys):
    tmp = str(tmp_path)
    _floor(tmp)
    _write(tmp, "e9_s45", crps_skill5=0.400, tail_reach20=6.0)   # other region, huge value
    _write(tmp, "af_e9_s45", crps_skill5=0.116, tail_reach20=6.0)

    main(["--score_dir", tmp, "--baseline", r"^afw_s\d+$", "--include", r"^af_"])
    out = capsys.readouterr().out
    assert "e9_s45" not in out.replace("af_e9_s45", "")
    assert "af_e9" in out


def test_worse_uses_the_same_width_bar_as_better(tmp_path, capsys):
    """WORSE must be one floor-width past the floor's WORST edge, not its best.

    It used to be measured from the best edge, so it fired the instant a run left the floor
    range while "better" needed a full extra width -- the two halves of one test at bars a
    width apart. Seven of the Africa slate's eight WORSE verdicts sat inside that gap.
    """
    tmp = str(tmp_path)
    _floor(tmp)                                    # crps_skill5 floor [0.110, 0.120], w 0.010
    _write(tmp, "af_edge_s45", crps_skill5=0.105, tail_reach20=6.0)   # 0.5x below: not WORSE
    _write(tmp, "af_bad_s45", crps_skill5=0.098, tail_reach20=6.0)    # 1.2x below: WORSE

    main(["--score_dir", tmp, "--baseline", r"^afw_s\d+$", "--group_seeds"])
    out = capsys.readouterr().out
    edge = [ln for ln in out.splitlines() if ln.strip().startswith("af_edge")][-1]
    bad = [ln for ln in out.splitlines() if ln.strip().startswith("af_bad")][-1]
    assert "nothing clears the bar" in edge
    assert "crps_skill5 WORSE (1.2x)" in bad
