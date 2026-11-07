import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data_generation import SimulationConfig, generate_data
from src.evaluation import (
    evaluate_models,
    extract_model_inputs,
    mae,
    rmse,
    split_train_test,
)
from src.models import (
    CausalForest,
    TLearner,
    XLearner,
    build_models,
    combine_effects,
    compute_pseudo_effects,
)

FAST_RF = {"n_estimators": 20, "min_samples_leaf": 5, "max_features": 1.0, "n_jobs": 1}
FAST_CFG = {
    "base_learner": FAST_RF,
    "x_learner": {"propensity": None},
    "causal_forest": {"n_estimators": 100, "min_samples_leaf": 5, "cv": 2, "n_jobs": 1},
}


@pytest.fixture(scope="module")
def split():
    df = generate_data(SimulationConfig(n=600, treat_prob=0.5, effect="linear", seed=7))
    return split_train_test(df, test_size=0.2, seed=7)


@pytest.fixture(scope="module")
def fitted(split):
    train, test = split
    models = build_models(FAST_CFG, seed=0, treat_prob=0.5)
    metrics, preds, fit_info = evaluate_models(models, train, test)
    return models, metrics, preds, fit_info


# T-Learner --------------------------------------------------------------------------
def test_tlearner_fits_one_model_per_arm(split):
    train, _ = split
    X, T, Y = extract_model_inputs(train)
    m = TLearner(LinearRegression()).fit(X, T, Y)
    assert m.n_train_ == {"control": int((T == 0).sum()), "treated": int((T == 1).sum())}
    # each arm model must equal an OLS fit on that arm only
    for arm, model in ((0, m.model0_), (1, m.model1_)):
        ref = LinearRegression().fit(X[T == arm], Y[T == arm])
        np.testing.assert_allclose(model.coef_, ref.coef_)
    np.testing.assert_allclose(m.effect(X), m.model1_.predict(X) - m.model0_.predict(X))


def test_tlearner_recovers_linear_effect_without_noise():
    df = generate_data(SimulationConfig(n=2000, effect="linear", noise_sd=0.0, seed=1))
    X, T, Y = extract_model_inputs(df)
    pred = TLearner(LinearRegression()).fit(X, T, Y).effect(X)
    np.testing.assert_allclose(pred, df["true_cate"], atol=1e-8)


# X-Learner --------------------------------------------------------------------------
def test_pseudo_effects_sign_and_values():
    Y = np.array([3.0, 1.0, 5.0, 2.0])
    T = np.array([1, 0, 1, 0])
    mu0_hat = np.array([1.0, 0.5, 4.0, 2.5])
    mu1_hat = np.array([2.5, 2.0, 6.0, 3.0])
    # treated: Y - mu0_hat ; control: mu1_hat - Y
    np.testing.assert_allclose(compute_pseudo_effects(Y, T, mu0_hat, mu1_hat), [2.0, 1.0, 1.0, 1.0])


def test_pseudo_effects_positive_for_positive_true_effect():
    # with perfect outcome models, D equals the true effect for both arms
    rng = np.random.default_rng(0)
    mu0_true = rng.normal(size=100)
    tau = np.full(100, 0.7)
    T = rng.integers(0, 2, 100)
    Y = mu0_true + T * tau
    D = compute_pseudo_effects(Y, T, mu0_true, mu0_true + tau)
    np.testing.assert_allclose(D, tau)


def test_combine_effects_weight_direction():
    tau0, tau1 = np.array([10.0]), np.array([0.0])
    # e = 0.1 (few treated): tau1, built from the well-estimated mu0, dominates
    assert combine_effects(tau0, tau1, 0.1)[0] == pytest.approx(1.0)
    assert combine_effects(tau0, tau1, 0.9)[0] == pytest.approx(9.0)
    assert combine_effects(tau0, tau1, 0.5)[0] == pytest.approx(5.0)
    assert combine_effects(tau0, tau1, 0.0)[0] == pytest.approx(0.0)
    assert combine_effects(tau0, tau1, 1.0)[0] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        combine_effects(tau0, tau1, 1.5)


def test_xlearner_effect_uses_stage2_models(split):
    train, test = split
    X, T, Y = extract_model_inputs(train)
    X_te, _, _ = extract_model_inputs(test)
    m = XLearner(RandomForestRegressor(n_estimators=20, random_state=0), propensity=0.3).fit(X, T, Y)

    D = compute_pseudo_effects(Y, T, m.model0_.predict(X), m.model1_.predict(X))
    ref_tau1 = RandomForestRegressor(n_estimators=20, random_state=0).fit(X[T == 1], D[T == 1])
    np.testing.assert_allclose(m.tau1_model_.predict(X_te), ref_tau1.predict(X_te))

    expected = 0.3 * m.tau0_model_.predict(X_te) + 0.7 * m.tau1_model_.predict(X_te)
    np.testing.assert_allclose(m.effect(X_te), expected)


def test_xlearner_recovers_linear_effect_without_noise():
    df = generate_data(SimulationConfig(n=2000, effect="linear", noise_sd=0.0, seed=1))
    X, T, Y = extract_model_inputs(df)
    pred = XLearner(LinearRegression(), propensity=0.5).fit(X, T, Y).effect(X)
    np.testing.assert_allclose(pred, df["true_cate"], atol=1e-8)


def test_build_models_uses_design_propensity():
    models = build_models(FAST_CFG, seed=0, treat_prob=0.3)
    assert models["X-Learner"].propensity == 0.3


# Causal Forest ----------------------------------------------------------------------
def test_causal_forest_valid_predictions(fitted, split):
    models, _, preds, _ = fitted
    _, test = split
    pred = preds["Causal Forest"].to_numpy()
    assert isinstance(models["Causal Forest"], CausalForest)
    assert models["Causal Forest"].estimator_.__class__.__name__ == "CausalForestDML"
    assert np.isfinite(pred).all()
    assert pred.std() > 0  # heterogeneous, not a single constant


# All models -------------------------------------------------------------------------
def test_prediction_length_matches_test(fitted, split):
    _, metrics, preds, fit_info = fitted
    train, test = split
    assert len(preds) == len(test)
    for name in ["T-Learner", "X-Learner", "Causal Forest"]:
        assert preds[name].notna().all()
        assert sum(fit_info[name].values()) == len(train)
    assert (metrics["Pred Unique"] > 1).all()


def test_split_is_stratified_and_disjoint(split):
    train, test = split
    assert len(test) == 120 and len(train) == 480
    assert set(train.index).isdisjoint(test.index)
    assert abs(train["T"].mean() - test["T"].mean()) < 0.01


# Metrics ----------------------------------------------------------------------------
def test_rmse_and_mae_exact():
    y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 1.0, 4.0])
    assert rmse(y, p) == pytest.approx(np.sqrt((0 + 1 + 4 + 0) / 4))
    assert mae(y, p) == pytest.approx(3 / 4)
    assert rmse(y, y) == 0.0


# Leakage ----------------------------------------------------------------------------
def test_model_inputs_exclude_ground_truth(split):
    train, _ = split
    X, T, Y = extract_model_inputs(train)
    assert X.shape[1] == 10
    for col in ("true_cate", "mu0"):
        assert not any(np.allclose(X[:, j], train[col]) for j in range(X.shape[1]))
        assert not np.allclose(Y, train[col])


def test_predictions_invariant_to_ground_truth_columns(split):
    train, test = split
    corrupted = train.copy()
    corrupted["true_cate"] = np.random.default_rng(1).normal(size=len(train)) * 100
    corrupted["mu0"] = -999.0
    _, p_clean, _ = evaluate_models(build_models(FAST_CFG, seed=0, treat_prob=0.5), train, test)
    _, p_bad, _ = evaluate_models(build_models(FAST_CFG, seed=0, treat_prob=0.5), corrupted, test)
    pd.testing.assert_frame_equal(p_clean, p_bad)


# Reproducibility --------------------------------------------------------------------
def test_same_seed_reproducible(split, fitted):
    train, test = split
    _, _, preds, _ = fitted
    _, preds_again, _ = evaluate_models(build_models(FAST_CFG, seed=0, treat_prob=0.5), train, test)
    pd.testing.assert_frame_equal(preds, preds_again)
