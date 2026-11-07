"""CATE estimators: T-Learner, X-Learner and a CausalForestDML wrapper.

Every estimator exposes the same minimal interface so that experiment code can
treat them interchangeably:

    model.fit(X, T, Y)  -> self      X: (n, p) covariates, T: (n,) in {0, 1}, Y: (n,)
    model.effect(X)     -> (n,)      estimated CATE tau(x)

Only X, T and Y are ever passed in; ground-truth columns (true_cate, mu0) are
never visible to the estimators.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _check_inputs(X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    if not (len(X) == len(T) == len(Y)):
        raise ValueError("X, T and Y must have the same number of rows")
    if not set(np.unique(T)) <= {0, 1}:
        raise ValueError("T must be binary (0/1)")
    if T.sum() == 0 or T.sum() == len(T):
        raise ValueError("both treated and control units are required")
    return X, T, Y


def make_base_learner(params: dict, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=seed, **params)


# ---------------------------------------------------------------------------
# T-Learner
# ---------------------------------------------------------------------------
class TLearner:
    """tau(x) = mu1(x) - mu0(x), with mu_t fitted only on units with T = t."""

    name = "T-Learner"

    def __init__(self, base_learner: RegressorMixin):
        self.base_learner = base_learner

    def fit(self, X, T, Y) -> "TLearner":
        X, T, Y = _check_inputs(X, T, Y)
        self.model0_ = clone(self.base_learner).fit(X[T == 0], Y[T == 0])
        self.model1_ = clone(self.base_learner).fit(X[T == 1], Y[T == 1])
        self.n_train_ = {"control": int((T == 0).sum()), "treated": int((T == 1).sum())}
        return self

    def effect(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.model1_.predict(X) - self.model0_.predict(X)


# ---------------------------------------------------------------------------
# X-Learner (Kunzel et al., 2019)
# ---------------------------------------------------------------------------
def compute_pseudo_effects(Y, T, mu0_hat, mu1_hat) -> np.ndarray:
    """Imputed individual effects, both oriented as (treated - untreated).

    treated  (T=1): D1 = Y - mu0_hat(X)
    control  (T=0): D0 = mu1_hat(X) - Y
    """
    Y, T = np.asarray(Y, dtype=float), np.asarray(T).astype(int)
    return np.where(T == 1, Y - mu0_hat, mu1_hat - Y)


def combine_effects(tau0, tau1, propensity) -> np.ndarray:
    """tau(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x) with g = e(x).

    tau1 is trained on treated units with mu0_hat imputed from the controls;
    when treated units are scarce (small e), mu0_hat is the more reliable
    model, so tau1 receives the larger weight (1 - e).
    """
    g = np.asarray(propensity, dtype=float)
    if np.any((g < 0) | (g > 1)):
        raise ValueError("propensity must lie in [0, 1]")
    return g * np.asarray(tau0) + (1.0 - g) * np.asarray(tau1)


class XLearner:
    """Two-stage X-Learner with a known (design) propensity score."""

    name = "X-Learner"

    def __init__(self, base_learner: RegressorMixin, propensity: float):
        if not 0.0 < propensity < 1.0:
            raise ValueError(f"propensity must be in (0, 1), got {propensity}")
        self.base_learner = base_learner
        self.propensity = propensity

    def fit(self, X, T, Y) -> "XLearner":
        X, T, Y = _check_inputs(X, T, Y)
        treated, control = T == 1, T == 0

        # Stage 1: outcome models per arm
        self.model0_ = clone(self.base_learner).fit(X[control], Y[control])
        self.model1_ = clone(self.base_learner).fit(X[treated], Y[treated])

        # Stage 2: pseudo-effects imputed with the opposite arm's model
        D = compute_pseudo_effects(Y, T, self.model0_.predict(X), self.model1_.predict(X))
        self.tau0_model_ = clone(self.base_learner).fit(X[control], D[control])
        self.tau1_model_ = clone(self.base_learner).fit(X[treated], D[treated])

        self.n_train_ = {"control": int(control.sum()), "treated": int(treated.sum())}
        return self

    def effect(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        g = np.full(len(X), self.propensity)
        return combine_effects(self.tau0_model_.predict(X), self.tau1_model_.predict(X), g)


# ---------------------------------------------------------------------------
# Causal Forest (econml.dml.CausalForestDML)
# ---------------------------------------------------------------------------
class CausalForest:
    """Thin wrapper around econml's CausalForestDML.

    Nuisance models (cross-fitted with `cv` folds):
        model_y : E[Y | X]   -> RandomForestRegressor (shared base-learner params)
        model_t : P(T=1 | X) -> constant (DummyClassifier 'prior', i.e. the fold's
                                 treated share) or RandomForestClassifier
    Final stage: generalized random forest on the residualised outcome/treatment.
    """

    name = "Causal Forest"
    estimator_name = "econml.dml.CausalForestDML"

    def __init__(
        self,
        base_params: dict,
        seed: int,
        n_estimators: int = 1000,
        min_samples_leaf: int = 5,
        max_features: float | str = 1.0,
        honest: bool = True,
        cv: int = 5,
        propensity_model: str = "constant",
        n_jobs: int = 1,
    ):
        if propensity_model not in ("constant", "random_forest"):
            raise ValueError("propensity_model must be 'constant' or 'random_forest'")
        self.base_params = base_params
        self.seed = seed
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.honest = honest
        self.cv = cv
        self.propensity_model = propensity_model
        self.n_jobs = n_jobs

    def _build(self):
        from econml.dml import CausalForestDML

        if self.propensity_model == "constant":
            model_t = DummyClassifier(strategy="prior")
        else:
            model_t = RandomForestClassifier(random_state=self.seed, **self.base_params)
        return CausalForestDML(
            model_y=make_base_learner(self.base_params, self.seed),
            model_t=model_t,
            discrete_treatment=True,
            cv=self.cv,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            honest=self.honest,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )

    def fit(self, X, T, Y) -> "CausalForest":
        X, T, Y = _check_inputs(X, T, Y)
        self.estimator_ = self._build()
        self.estimator_.fit(Y, T, X=X)
        self.n_train_ = {"control": int((T == 0).sum()), "treated": int((T == 1).sum())}
        return self

    def effect(self, X) -> np.ndarray:
        return np.asarray(self.estimator_.effect(np.asarray(X, dtype=float))).ravel()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_models(model_cfg: dict, seed: int, treat_prob: float) -> dict:
    """Instantiate all estimators from the pre-registered config (configs/models.yaml)."""
    base_params = dict(model_cfg["base_learner"])
    propensity = model_cfg.get("x_learner", {}).get("propensity") or treat_prob
    base = make_base_learner(base_params, seed)
    return {
        TLearner.name: TLearner(base),
        XLearner.name: XLearner(base, propensity=propensity),
        CausalForest.name: CausalForest(base_params, seed, **model_cfg["causal_forest"]),
    }


def describe_model(model) -> dict:
    """JSON-serialisable description of an (unfitted or fitted) estimator."""
    if isinstance(model, CausalForest):
        return {
            "estimator": CausalForest.estimator_name,
            "model_y": f"RandomForestRegressor({model.base_params}, random_state={model.seed})",
            "model_t": (
                "DummyClassifier(strategy='prior')"
                if model.propensity_model == "constant"
                else f"RandomForestClassifier({model.base_params}, random_state={model.seed})"
            ),
            "discrete_treatment": True,
            "cv": model.cv,
            "n_estimators": model.n_estimators,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features,
            "honest": model.honest,
            "n_jobs": model.n_jobs,
            "random_state": model.seed,
        }
    info = {
        "estimator": f"src.models.{type(model).__name__}",
        "base_learner": type(model.base_learner).__name__,
        "base_learner_params": {k: v for k, v in model.base_learner.get_params().items()},
    }
    if isinstance(model, XLearner):
        info["propensity"] = model.propensity
        info["weighting"] = "tau = e*tau0 + (1-e)*tau1"
    return info
