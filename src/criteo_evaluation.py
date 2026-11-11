"""Uplift evaluation for real randomized data without a ground-truth CATE (Criteo, STEP 5).

Units are ranked by a model's uplift score in descending order. For the top k targeted units
(fraction phi = k / N) let N_t(k), N_c(k) be the treated / control counts and Y_t(k), Y_c(k)
their conversions. Definitions (fixed for every model):

    rate difference   d(k) = Y_t(k) / N_t(k) - Y_c(k) / N_c(k)            [conversion-rate units]
    uplift curve      U(k) = d(k) * k                                      [conversions]
        estimated incremental conversions if all k targeted units were treated vs. none
    qini curve        Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k) = d(k) * N_t(k)  [conversions]
        incremental conversions among the treated units in the top k (control rescaled to N_t)

Group-specific denominators make both curves valid under the 85:15 treated:control split; the
raw difference Y_t(k) - Y_c(k) is never used. With estimator="ipw" the uplift curve instead uses
the known assignment probability e: U_ipw(k) = Y_t(k) / e - Y_c(k) / (1 - e) (defined even when a
group is empty, but noisier).

Areas use the trapezoid rule on phi in {0, 1/G, ..., 1} (G = n_grid) with U(0) = Q(0) = 0:

    AUUC                 = integral_0^1 U(phi N) dphi          (raw area, conversions)
    AUUC_random          = U(N) / 2                             (straight line (0,0)-(1,U(N)))
    AUUC_minus_random    = AUUC - AUUC_random
    AUUC_normalized      = AUUC_minus_random / U(N)             (only if U(N) > 0, else NaN)
    Qini area / random / coefficient / normalized: same with Q and Q(N).

Ties: units with equal scores form one block. Inside a block counts are interpolated linearly,
i.e. the expected counts under a uniformly random order of the tied units, so results do not
depend on input row order. If N_t(k) or N_c(k) is 0 at a grid point (k > 0), d, U and Q are NaN
there and the areas are NaN (`n_undefined_points` reports how many).

Causal reading of d(k) as the average effect in the targeted group requires: randomized T
independent of X, SUTVA (no interference), a score that is a function of X only and fitted
without the evaluation data, and consistency of the conversion outcome.
"""

from __future__ import annotations

import numpy as np

TOP_FRACTIONS = (0.1, 0.2, 0.3)


def _check_inputs(score, treatment, y, weights=None):
    score = np.asarray(score, dtype=float)
    treatment = np.asarray(treatment)
    y = np.asarray(y)
    if not (score.ndim == treatment.ndim == y.ndim == 1) or not (len(score) == len(treatment) == len(y)):
        raise ValueError("score, treatment and y must be 1-D arrays of equal length")
    if len(score) == 0:
        raise ValueError("empty input")
    if not np.isfinite(score).all():
        raise ValueError("score contains NaN or inf")
    for name, arr in (("treatment", treatment), ("y", y)):
        if not np.isin(arr, (0, 1)).all():
            raise ValueError(f"{name} must be binary 0/1")
    if weights is None:
        weights = np.ones(len(score), dtype=np.int64)
    else:
        weights = np.asarray(weights)
        if weights.shape != score.shape or (weights < 0).any():
            raise ValueError("weights must be non-negative and match score")
    return score, treatment.astype(np.int8), y.astype(np.int8), weights


def rank_order(score) -> tuple[np.ndarray, np.ndarray]:
    """Descending order of the scores and the end position (exclusive) of each tie block."""
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    s = score[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True]) + 1
    return order, ends


def cumulative_counts(order, ends, treatment, y, weights=None) -> dict[str, np.ndarray]:
    """Weighted cumulative (n, n_t, n_c, y_t, y_c) at tie-block boundaries, starting with 0."""
    t = np.asarray(treatment)[order].astype(np.int64)
    yy = np.asarray(y)[order].astype(np.int64)
    w = np.ones(len(order), dtype=np.int64) if weights is None else np.asarray(weights)[order]
    out = {}
    for key, v in (("n", w), ("n_t", w * t), ("n_c", w * (1 - t)), ("y_t", w * t * yy), ("y_c", w * (1 - t) * yy)):
        out[key] = np.r_[0, np.cumsum(v)[ends - 1]].astype(float)
    return out


def _at(cum: dict, k: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.interp(k, cum["n"], cum[key]) for key in ("n_t", "n_c", "y_t", "y_c")}


def _safe_div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def _area(phi, values):
    if np.isnan(values).any():
        return np.nan
    return float(np.sum((values[1:] + values[:-1]) / 2 * np.diff(phi)))


def evaluate_from_counts(cum: dict, n_grid: int = 100, top_fractions=TOP_FRACTIONS, estimator: str = "ratio",
                         propensity: float | None = None) -> dict:
    """Curves, areas and top-q metrics from cumulative counts (see module docstring)."""
    if estimator not in ("ratio", "ipw"):
        raise ValueError("estimator must be 'ratio' or 'ipw'")
    N = cum["n"][-1]
    phi = np.linspace(0.0, 1.0, n_grid + 1)
    c = _at(cum, phi * N)
    rate_t, rate_c = _safe_div(c["y_t"], c["n_t"]), _safe_div(c["y_c"], c["n_c"])
    d = rate_t - rate_c
    k = phi * N
    if estimator == "ipw":
        e = propensity if propensity is not None else cum["n_t"][-1] / N
        if not 0 < e < 1:
            raise ValueError("propensity must be in (0, 1)")
        uplift = c["y_t"] / e - c["y_c"] / (1 - e)
    else:
        uplift = d * k
    qini = c["y_t"] - c["y_c"] * _safe_div(c["n_t"], c["n_c"])
    uplift[0], qini[0], d[0] = 0.0, 0.0, np.nan
    defined = ~np.isnan(uplift[1:]) & ~np.isnan(qini[1:])
    u_total, q_total = uplift[-1], qini[-1]

    auuc = _area(phi, uplift)
    qini_area = _area(phi, qini)
    res = {
        "estimator": estimator, "n": N, "n_treated": cum["n_t"][-1], "n_control": cum["n_c"][-1],
        "n_grid": n_grid, "n_undefined_points": int((~defined).sum()),
        "total_rate_diff": d[-1], "total_uplift": u_total, "total_qini": q_total,
        "auuc": auuc, "auuc_random": u_total / 2, "auuc_minus_random": auuc - u_total / 2,
        "auuc_normalized": (auuc - u_total / 2) / u_total if u_total > 0 else np.nan,
        "qini_area": qini_area, "qini_random": q_total / 2, "qini_coefficient": qini_area - q_total / 2,
        "qini_normalized": (qini_area - q_total / 2) / q_total if q_total > 0 else np.nan,
        "curve": {"phi": phi, "k": k, "n_t": c["n_t"], "n_c": c["n_c"], "y_t": c["y_t"], "y_c": c["y_c"],
                  "rate_t": rate_t, "rate_c": rate_c, "rate_diff": d, "uplift": uplift, "qini": qini},
    }
    for q in top_fractions:
        kq = q * N
        cq = _at(cum, np.array([kq]))
        rt, rc = _safe_div(cq["y_t"], cq["n_t"])[0], _safe_div(cq["y_c"], cq["n_c"])[0]
        tag = f"top{round(q * 100)}"
        res[f"{tag}_k"] = kq
        res[f"{tag}_n_treated"], res[f"{tag}_n_control"] = cq["n_t"][0], cq["n_c"][0]
        res[f"{tag}_conv_treated"], res[f"{tag}_conv_control"] = cq["y_t"][0], cq["y_c"][0]
        res[f"{tag}_rate_treated"], res[f"{tag}_rate_control"] = rt, rc
        res[f"{tag}_rate_diff"] = rt - rc
        res[f"{tag}_incremental_conversions"] = (rt - rc) * kq
    return res


def evaluate_uplift(score, treatment, y, n_grid: int = 100, top_fractions=TOP_FRACTIONS, estimator: str = "ratio",
                    propensity: float | None = None, weights=None) -> dict:
    """Evaluate one model's uplift scores on a test set (higher score = target first)."""
    score, treatment, y, weights = _check_inputs(score, treatment, y, weights)
    order, ends = rank_order(score)
    cum = cumulative_counts(order, ends, treatment, y, weights)
    return evaluate_from_counts(cum, n_grid, top_fractions, estimator, propensity)


def stratified_bootstrap_weights(treatment, rng: np.random.Generator) -> np.ndarray:
    """Resample treated and control units separately with replacement (group sizes fixed).

    Returned as per-unit counts so that the same draw can be applied to every model's ranking
    (paired comparison) without copying the data.
    """
    treatment = np.asarray(treatment)
    w = np.zeros(len(treatment), dtype=np.int32)
    for g in (0, 1):
        idx = np.flatnonzero(treatment == g)
        if len(idx):
            w[idx] = np.bincount(rng.integers(0, len(idx), size=len(idx)), minlength=len(idx))
    return w


def paired_bootstrap(scores: dict[str, np.ndarray], treatment, y, n_boot: int, seed: int, n_grid: int = 100,
                     top_fractions=TOP_FRACTIONS, metrics=("auuc", "qini_coefficient")) -> dict[str, np.ndarray]:
    """Bootstrap metric draws for several models on identical resamples.

    Each model is ranked once; every replicate draws one set of stratified weights and evaluates
    all models with it. Returns {f"{model}:{metric}": array of n_boot values}.
    """
    first = next(iter(scores.values()))
    _check_inputs(first, treatment, y)
    ranks = {m: rank_order(s) for m, s in scores.items()}
    rng = np.random.default_rng(seed)
    out = {f"{m}:{k}": np.empty(n_boot) for m in scores for k in metrics}
    for b in range(n_boot):
        w = stratified_bootstrap_weights(treatment, rng)
        for m, (order, ends) in ranks.items():
            res = evaluate_from_counts(cumulative_counts(order, ends, treatment, y, w), n_grid, top_fractions)
            for k in metrics:
                out[f"{m}:{k}"][b] = res[k]
    return out
