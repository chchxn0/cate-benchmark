"""Criteo (STEP 5) helpers around the STEP 4 estimators in src/models.py.

The estimators themselves are reused unchanged; this module only adds what real, large, binary-outcome
data needs: building the three models from configs/criteo_models.yaml, batched prediction, a fixed
validation subset and prediction diagnostics.
"""

from __future__ import annotations

import numpy as np

from src.models import CausalForest, TLearner, XLearner, make_base_learner

MODEL_KEYS = {"T": TLearner.name, "X": XLearner.name, "CF": CausalForest.name}


def build_criteo_model(key: str, cfg: dict, treated_share: float):
    """One estimator ('T', 'X' or 'CF') with the Criteo config; X-Learner propensity = training treated share."""
    seed = cfg["seed"]
    base_params = dict(cfg["base_learner"])
    if key == "T":
        return TLearner(make_base_learner(base_params, seed))
    if key == "X":
        propensity = cfg.get("x_learner", {}).get("propensity") or treated_share
        return XLearner(make_base_learner(base_params, seed), propensity=propensity)
    if key == "CF":
        return CausalForest(base_params, seed, **cfg["causal_forest"])
    raise ValueError(f"unknown model key {key!r}; expected one of {sorted(MODEL_KEYS)}")


def predict_in_batches(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """model.effect over row batches (bounds the float64 copy each estimator makes)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    out = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        out[start:start + batch_size] = model.effect(X[start:start + batch_size])
    return out


def fixed_subset_positions(n: int, size: int, seed: int) -> np.ndarray:
    """Sorted positions of a fixed simple random subset (without replacement)."""
    if size > n:
        raise ValueError("subset larger than population")
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def prediction_summary(pred: np.ndarray, near_constant_sd: float = 1e-6) -> dict:
    """Distribution and sanity flags of predicted CATE scores."""
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(pred)
    p = pred[finite]
    q = np.quantile(p, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]) if len(p) else [np.nan] * 7
    uniq, cnt = np.unique(p, return_counts=True) if len(p) else (np.array([]), np.array([0]))
    sd = float(p.std()) if len(p) else np.nan
    return {
        "n": int(len(pred)), "n_nan": int(np.isnan(pred).sum()), "n_inf": int(np.isinf(pred).sum()),
        "mean": float(p.mean()) if len(p) else np.nan, "sd": sd,
        "min": float(p.min()) if len(p) else np.nan, "max": float(p.max()) if len(p) else np.nan,
        **{f"p{int(k * 100)}": float(v) for k, v in zip([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], q)},
        "share_negative": float((p < 0).mean()) if len(p) else np.nan,
        "n_unique": int(len(uniq)), "mode_share": float(cnt.max() / len(p)) if len(p) else np.nan,
        "near_constant": bool(len(p) == 0 or sd < near_constant_sd or len(uniq) < 10),
    }
