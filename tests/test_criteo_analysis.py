import numpy as np
import pytest

from src.criteo_analysis import segment_effect, tie_summary


def test_tie_summary_counts_zeros_modes_and_blocks():
    s = np.array([0, 0, 0, 0.1, 0.1, 0.2, 0.3, 0.0])
    r = tie_summary(s)
    assert r["share_zero"] == pytest.approx(4 / 8)
    assert r["mode_value"] == 0 and r["mode_share"] == pytest.approx(0.5)
    assert r["largest_tie_block"] == 4 and r["n_tie_blocks"] == 2
    assert r["share_rows_in_tie_blocks"] == pytest.approx(6 / 8)
    assert r["n_unique"] == 4


def test_segment_effect_hand_computed():
    # lowest 50% = rows with scores 1..4: treated (y=1,0), control (y=0,0)
    score = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    t = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    r = segment_effect(score, t, y, 0.5, lowest=True)
    assert (r["k"], r["n_treated"], r["n_control"], r["conv_treated"], r["conv_control"]) == (4, 2, 2, 1, 0)
    assert r["rate_diff"] == pytest.approx(0.5)
    assert r["rate_diff_se"] == pytest.approx(np.sqrt(0.5 * 0.5 / 2))
    assert r["rate_diff_ci_low"] < 0.5 < r["rate_diff_ci_high"]
    hi = segment_effect(score, t, y, 0.5, lowest=False)
    assert (hi["n_treated"], hi["conv_treated"], hi["conv_control"]) == (2, 1, 1)


def test_segment_effect_reports_boundary_ties_and_rejects_empty_group():
    score = np.array([0, 0, 0, 0, 1, 2], dtype=float)
    t = np.array([1, 0, 1, 0, 1, 0])
    y = np.zeros(6, dtype=int)
    assert segment_effect(score, t, y, 0.5)["boundary_tie_rows"] == 4
    with pytest.raises(ValueError):
        segment_effect(score, np.ones(6, dtype=int), y, 0.5)
