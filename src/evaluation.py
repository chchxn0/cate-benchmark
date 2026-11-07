"""Train/test splitting, model fitting and CATE evaluation.

Leakage guard: `extract_model_inputs` is the only path from a dataframe to an
estimator and returns X1..Xp, T and Y only. Ground-truth columns
(GROUND_TRUTH_COLUMNS) are read exclusively inside `evaluate_models` when
scoring test-set predictions.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_generation import covariate_columns

GROUND_TRUTH_COLUMNS = ("true_cate", "mu0")


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def split_train_test(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split stratified on T so both sets keep the treated share."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["T"])
    return train, test


def group_counts(df: pd.DataFrame) -> dict:
    n_treated = int(df["T"].sum())
    return {"total": int(len(df)), "treated": n_treated, "control": int(len(df) - n_treated)}


def extract_model_inputs(df: pd.DataFrame, n_covariates: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, T, Y) — the only information estimators are allowed to see."""
    X = df[covariate_columns(n_covariates)].to_numpy(dtype=float)
    return X, df["T"].to_numpy(dtype=int), df["Y"].to_numpy(dtype=float)


def evaluate_models(models: dict, train: pd.DataFrame, test: pd.DataFrame, n_covariates: int = 10):
    """Fit every model on `train`, predict CATE on `test`, score against true_cate.

    Returns
    -------
    metrics     : DataFrame, one row per model
    predictions : DataFrame, one row per test unit (true_cate + one column per model)
    fit_info    : dict, per-model training-set group counts
    """
    X_tr, T_tr, Y_tr = extract_model_inputs(train, n_covariates)
    X_te, _, _ = extract_model_inputs(test, n_covariates)
    true_cate = test["true_cate"].to_numpy(dtype=float)

    rows, fit_info = [], {}
    predictions = pd.DataFrame({"row_id": test.index.to_numpy(), "T": test["T"].to_numpy(), "true_cate": true_cate})
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_tr, T_tr, Y_tr)
        train_time = time.perf_counter() - start

        pred = model.effect(X_te)
        if pred.shape != (len(test),):
            raise RuntimeError(f"{name}: expected {len(test)} predictions, got shape {pred.shape}")
        predictions[name] = pred
        fit_info[name] = model.n_train_
        rows.append(
            {
                "Model": name,
                "RMSE": rmse(true_cate, pred),
                "MAE": mae(true_cate, pred),
                "Train Time (sec)": train_time,
                "Pred Mean": float(pred.mean()),
                "Pred SD": float(pred.std(ddof=1)),
                "Pred Unique": int(np.unique(np.round(pred, 10)).size),
                "Corr(true, pred)": float(np.corrcoef(true_cate, pred)[0, 1]),
            }
        )
    return pd.DataFrame(rows), predictions, fit_info


def constant_ate_baseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Reference: predict the training difference-in-means ATE for every test unit."""
    ate = float(train.loc[train["T"] == 1, "Y"].mean() - train.loc[train["T"] == 0, "Y"].mean())
    pred = np.full(len(test), ate)
    return {"ate_estimate": ate, "RMSE": rmse(test["true_cate"], pred), "MAE": mae(test["true_cate"], pred)}
