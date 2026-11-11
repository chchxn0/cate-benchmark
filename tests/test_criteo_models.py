import joblib
import numpy as np
import pytest

from src.criteo_models import (
    build_criteo_model,
    fixed_subset_positions,
    predict_in_batches,
    prediction_summary,
)

FAST_CFG = {
    "seed": 42,
    "base_learner": {"n_estimators": 10, "min_samples_leaf": 5, "max_features": 1.0, "n_jobs": 1},
    "x_learner": {"propensity": None},
    "causal_forest": {"n_estimators": 20, "min_samples_leaf": 5, "cv": 2, "n_jobs": 1},
}


@pytest.fixture(scope="module")
def binary_data():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 12)).astype(np.float32)
    T = (rng.random(n) < 0.85).astype(int)
    p = 0.02 + 0.03 * T * (X[:, 0] > 0)
    Y = (rng.random(n) < p).astype(int)
    return X, T, Y


@pytest.mark.parametrize("key", ["T", "X", "CF"])
def test_models_fit_binary_outcome_and_batches_match(binary_data, key, tmp_path):
    X, T, Y = binary_data
    model = build_criteo_model(key, FAST_CFG, treated_share=T.mean()).fit(X, T, Y)
    full = model.effect(X[:500])
    batched = predict_in_batches(model, X[:500], batch_size=64)
    np.testing.assert_allclose(batched, full, rtol=0, atol=1e-12)
    assert np.isfinite(batched).all()
    assert np.abs(batched).max() <= 1.0 + 1e-9          # difference of probabilities for a 0/1 outcome
    path = tmp_path / f"{key}.joblib"
    joblib.dump(model, path, compress=3)
    np.testing.assert_allclose(joblib.load(path).effect(X[:500]), full, rtol=0, atol=1e-12)


def test_x_learner_uses_training_treated_share(binary_data):
    X, T, Y = binary_data
    model = build_criteo_model("X", FAST_CFG, treated_share=0.85)
    assert model.propensity == 0.85
    fixed = dict(FAST_CFG, x_learner={"propensity": 0.8})
    assert build_criteo_model("X", fixed, treated_share=0.85).propensity == 0.8


def test_unknown_model_key():
    with pytest.raises(ValueError):
        build_criteo_model("S", FAST_CFG, 0.85)


def test_fixed_subset_is_deterministic_sorted_and_unique():
    a = fixed_subset_positions(1000, 100, 42)
    np.testing.assert_array_equal(a, fixed_subset_positions(1000, 100, 42))
    assert len(np.unique(a)) == 100 and np.all(np.diff(a) > 0) and a.max() < 1000
    with pytest.raises(ValueError):
        fixed_subset_positions(10, 11, 0)


def test_prediction_summary_flags():
    s = prediction_summary(np.array([0.1, -0.2, np.nan, np.inf, 0.3]))
    assert s["n_nan"] == 1 and s["n_inf"] == 1 and s["n"] == 5
    assert s["min"] == pytest.approx(-0.2) and s["max"] == pytest.approx(0.3)
    assert prediction_summary(np.full(100, 0.01))["near_constant"]
    assert not prediction_summary(np.random.default_rng(0).normal(size=1000))["near_constant"]


def test_predict_in_batches_rejects_bad_batch(binary_data):
    with pytest.raises(ValueError):
        predict_in_batches(None, binary_data[0], 0)
