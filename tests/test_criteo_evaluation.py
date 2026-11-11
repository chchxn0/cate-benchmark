import warnings

import numpy as np
import pytest

from src.criteo_evaluation import (
    evaluate_uplift,
    paired_bootstrap,
    stratified_bootstrap_weights,
)


def _blocks(spec):
    """spec: list of (score, n_treated, conv_treated, n_control, conv_control) -> score, t, y arrays."""
    s, t, y = [], [], []
    for score, nt, yt, nc, yc in spec:
        s += [score] * (nt + nc)
        t += [1] * nt + [0] * nc
        y += [1] * yt + [0] * (nt - yt) + [1] * yc + [0] * (nc - yc)
    return np.array(s, dtype=float), np.array(t), np.array(y)


def _no_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


def test_hand_computed_curve_points():
    # ranked rows (t, y): (1,1) (0,0) | (1,0) (0,1) | (1,1) (0,0)
    score = np.array([5, 4, 3, 2, 1, 0], dtype=float)
    t = np.array([1, 0, 1, 0, 1, 0])
    y = np.array([1, 0, 0, 1, 1, 0])
    r = evaluate_uplift(score, t, y, n_grid=3)
    c = r["curve"]
    np.testing.assert_allclose(c["k"], [0, 2, 4, 6])
    np.testing.assert_allclose(c["uplift"], [0, 2, 0, 2])   # d(k) * k
    np.testing.assert_allclose(c["qini"], [0, 1, 0, 1])     # Y_t - Y_c * N_t / N_c
    assert r["auuc"] == pytest.approx(1.0)
    assert r["auuc_random"] == pytest.approx(1.0)
    assert r["qini_area"] == pytest.approx(0.5)
    assert r["qini_coefficient"] == pytest.approx(0.0)


def test_all_equal_scores_equal_random_baseline():
    rng = np.random.default_rng(0)
    n = 5000
    t = (rng.random(n) < 0.85).astype(int)
    y = (rng.random(n) < np.where(t == 1, 0.05, 0.03)).astype(int)
    r = evaluate_uplift(np.zeros(n), t, y)
    c = r["curve"]
    np.testing.assert_allclose(c["uplift"], c["phi"] * r["total_uplift"], atol=1e-9)
    np.testing.assert_allclose(c["qini"], c["phi"] * r["total_qini"], atol=1e-9)
    assert r["auuc_minus_random"] == pytest.approx(0.0, abs=1e-9)
    assert r["qini_coefficient"] == pytest.approx(0.0, abs=1e-9)
    assert r["total_uplift"] > 0
    assert r["auuc_normalized"] == pytest.approx(0.0, abs=1e-12)


def test_constant_effect_gives_linear_curves():
    # 10 score blocks with identical composition: every targeting depth has the same rate difference
    r = evaluate_uplift(*_blocks([(s, 17, 2, 3, 0) for s in range(10)]), n_grid=10)
    np.testing.assert_allclose(r["curve"]["rate_diff"][1:], 2 / 17)
    assert r["qini_coefficient"] == pytest.approx(0.0, abs=1e-9)
    assert r["auuc_minus_random"] == pytest.approx(0.0, abs=1e-9)
    assert r["top10_rate_diff"] == pytest.approx(2 / 17)


def test_imbalanced_groups_use_group_denominators():
    score, t, y = _blocks([(0, 17, 4, 3, 1)])   # 85% treated
    r = evaluate_uplift(score, t, y)
    naive = 4 - 1
    assert r["total_qini"] == pytest.approx(4 - 1 * 17 / 3)
    assert r["total_uplift"] == pytest.approx((4 / 17 - 1 / 3) * 20)
    assert r["total_qini"] != pytest.approx(naive)
    assert r["total_rate_diff"] == pytest.approx(4 / 17 - 1 / 3)


def test_high_effect_customers_on_top():
    spec = [(1, 85, 20, 15, 1), (0, 85, 5, 15, 1)]   # high-uplift block scored first
    good = evaluate_uplift(*_blocks(spec))
    score, t, y = _blocks(spec)
    bad = evaluate_uplift(-score, t, y)
    assert good["qini_coefficient"] > 0 > bad["qini_coefficient"]
    assert good["auuc_minus_random"] > 0 > bad["auuc_minus_random"]
    assert 0 < good["auuc_normalized"] <= 1
    # top 10% lies inside the first block: its rate difference is that block's difference
    assert good["top10_k"] == pytest.approx(20)
    assert good["top10_rate_diff"] == pytest.approx(20 / 85 - 1 / 15)
    assert good["top10_incremental_conversions"] == pytest.approx((20 / 85 - 1 / 15) * 20)
    assert good["auuc"] == pytest.approx(good["auuc_random"] + good["auuc_minus_random"])


def test_rare_conversions_stay_finite():
    score, t, y = _blocks([(1, 500, 1, 90, 0), (0, 350, 0, 60, 0)])   # 1 conversion in 1000 rows
    r = _no_warnings(evaluate_uplift, score, t, y)
    assert r["n_undefined_points"] == 0
    assert np.isfinite([r["auuc"], r["qini_coefficient"], r["top10_rate_diff"]]).all()
    assert r["total_rate_diff"] == pytest.approx(1 / 850)
    assert r["top30_rate_control"] == 0


def test_zero_denominator_gives_nan_without_warnings():
    # top 10% of rows are all treated -> N_c(k) = 0 at phi = 0.1
    score, t, y = _blocks([(2, 10, 3, 0, 0), (1, 75, 10, 15, 1)])
    r = _no_warnings(evaluate_uplift, score, t, y, n_grid=10)
    assert r["n_undefined_points"] >= 1
    assert np.isnan(r["curve"]["uplift"][1]) and np.isnan(r["curve"]["qini"][1])
    assert np.isnan(r["auuc"]) and np.isnan(r["qini_coefficient"])
    assert np.isnan(r["top10_rate_diff"]) and np.isnan(r["top10_incremental_conversions"])
    assert np.isfinite(r["total_uplift"])
    ipw = _no_warnings(evaluate_uplift, score, t, y, n_grid=10, estimator="ipw")
    assert np.isfinite(ipw["auuc"])


def test_zero_or_negative_total_effect():
    zero = evaluate_uplift(*_blocks([(1, 85, 17, 15, 3), (0, 85, 17, 15, 3)]))
    assert zero["total_uplift"] == pytest.approx(0)
    assert np.isnan(zero["auuc_normalized"]) and np.isnan(zero["qini_normalized"])
    neg = evaluate_uplift(*_blocks([(1, 85, 2, 15, 3), (0, 85, 4, 15, 2)]))
    assert neg["total_uplift"] < 0
    assert np.isnan(neg["auuc_normalized"])
    assert np.isfinite(neg["auuc"]) and np.isfinite(neg["auuc_minus_random"])


def test_many_ties_are_order_invariant_and_interpolated():
    rng = np.random.default_rng(1)
    n = 3000
    score = rng.integers(0, 3, n).astype(float)
    t = (rng.random(n) < 0.7).astype(int)
    y = (rng.random(n) < 0.1 + 0.05 * score * t).astype(int)
    base = evaluate_uplift(score, t, y)
    perm = rng.permutation(n)
    shuffled = evaluate_uplift(score[perm], t[perm], y[perm])
    for key in ("auuc", "qini_coefficient", "top10_rate_diff", "top20_incremental_conversions"):
        assert shuffled[key] == pytest.approx(base[key], rel=1e-12)
    # top 10% sits inside the top score block: counts are the block's counts scaled proportionally
    top = score == 2
    frac = 0.1 * n / top.sum()
    assert base["top10_n_treated"] == pytest.approx(frac * (t[top] == 1).sum())
    assert base["top10_conv_control"] == pytest.approx(frac * ((t[top] == 0) & (y[top] == 1)).sum())


def test_ipw_matches_ratio_on_full_population():
    score, t, y = _blocks([(1, 70, 9, 30, 2), (0, 100, 6, 20, 1)])
    ratio = evaluate_uplift(score, t, y)
    ipw = evaluate_uplift(score, t, y, estimator="ipw")
    assert ipw["total_uplift"] == pytest.approx(ratio["total_uplift"])
    assert ipw["curve"]["qini"] == pytest.approx(ratio["curve"]["qini"])


def test_integer_weights_equal_duplicated_rows():
    score, t, y = _blocks([(2, 30, 5, 10, 1), (1, 40, 3, 12, 2), (0, 20, 1, 8, 0)])
    w = np.random.default_rng(2).integers(0, 4, len(score))
    weighted = evaluate_uplift(score, t, y, weights=w)
    dup = evaluate_uplift(np.repeat(score, w), np.repeat(t, w), np.repeat(y, w))
    for key in ("auuc", "qini_coefficient", "top20_rate_diff", "total_uplift"):
        assert weighted[key] == pytest.approx(dup[key])
    ones = evaluate_uplift(score, t, y, weights=np.ones(len(score), dtype=int))
    assert ones["auuc"] == pytest.approx(evaluate_uplift(score, t, y)["auuc"])


def test_stratified_bootstrap_keeps_group_sizes_and_is_reproducible():
    t = np.r_[np.ones(850, dtype=int), np.zeros(150, dtype=int)]
    w1 = stratified_bootstrap_weights(t, np.random.default_rng(7))
    w2 = stratified_bootstrap_weights(t, np.random.default_rng(7))
    assert w1[t == 1].sum() == 850 and w1[t == 0].sum() == 150
    np.testing.assert_array_equal(w1, w2)


def test_paired_bootstrap_uses_identical_resamples():
    rng = np.random.default_rng(3)
    n = 2000
    t = (rng.random(n) < 0.85).astype(int)
    y = (rng.random(n) < 0.05).astype(int)
    s = rng.random(n)
    out = paired_bootstrap({"a": s, "b": s.copy(), "c": -s}, t, y, n_boot=5, seed=11)
    assert len(out["a:auuc"]) == 5
    np.testing.assert_array_equal(out["a:auuc"], out["b:auuc"])
    again = paired_bootstrap({"a": s}, t, y, n_boot=5, seed=11)
    np.testing.assert_array_equal(again["a:auuc"], out["a:auuc"])


@pytest.mark.parametrize("bad", [
    dict(score=[1.0, np.nan], treatment=[1, 0], y=[0, 1]),
    dict(score=[1.0, 2.0], treatment=[1, 2], y=[0, 1]),
    dict(score=[1.0, 2.0], treatment=[1, 0], y=[0]),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        evaluate_uplift(bad["score"], bad["treatment"], bad["y"])
