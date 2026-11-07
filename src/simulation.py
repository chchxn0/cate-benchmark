"""Repeated simulation experiments: condition grid, single replication, summary.

Reuses the data generator (src.data_generation), the estimators (src.models)
and the evaluation helpers (src.evaluation) without modification.

Seeding
-------
replication r -> data_seed  = data_seed_base + r   (generate_data + train/test split)
              -> model_seed = model_seed_base + r  (every estimator)
All models in one replication are fitted on the same train set and scored on
the same test set. Because data seeds differ across replications, the datasets
are independent across replications. Within one replication, conditions that
share n reuse the same covariate/noise draws (common random numbers, see
src.data_generation).
"""

from __future__ import annotations

import gc
import itertools
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.data_generation import SimulationConfig, generate_data
from src.evaluation import constant_ate_baseline, evaluate_models, group_counts, split_train_test
from src.models import build_models

FACTORS = ("n", "treat_prob", "effect")
EXPERIMENT_NAMES = {"n": "sample_size", "treat_prob": "treat_ratio", "effect": "complexity"}
BASELINE_MODEL = "Constant ATE"
CONDITION_KEYS = ["n", "treat_prob", "effect", "noise_sd"]


def build_conditions(baseline: dict, factors: dict, design: str = "ofat") -> pd.DataFrame:
    """Return one row per unique condition.

    `experiments` lists every OFAT experiment the condition belongs to
    (e.g. the baseline belongs to all three), separated by '|'.
    """
    if design == "ofat":
        membership: dict[tuple, list[str]] = {}
        for factor in FACTORS:
            for level in factors[factor]:
                cond = {**baseline, factor: level}
                key = tuple(cond[k] for k in CONDITION_KEYS)
                membership.setdefault(key, []).append(EXPERIMENT_NAMES[factor])
        rows = [dict(zip(CONDITION_KEYS, key), experiments="|".join(exps)) for key, exps in membership.items()]
    elif design == "full":
        rows = [
            {"n": n, "treat_prob": p, "effect": e, "noise_sd": baseline["noise_sd"], "experiments": "full_factorial"}
            for n, p, e in itertools.product(factors["n"], factors["treat_prob"], factors["effect"])
        ]
    else:
        raise ValueError(f"design must be 'ofat' or 'full', got {design!r}")
    conditions = pd.DataFrame(rows)
    conditions.insert(0, "condition_id", [f"c{i:02d}" for i in range(len(conditions))])
    return conditions


def select_conditions(
    conditions: pd.DataFrame,
    condition_ids: list[str] | None = None,
    n: int | None = None,
    baseline: dict | None = None,
) -> pd.DataFrame:
    """Filter the condition table; all given filters must hold (intersection).

    condition_ids : keep only these ids (unknown ids raise ValueError)
    n             : keep only this sample size
    baseline      : keep only the baseline condition (n, treat_prob, effect)
    """
    selected = conditions
    if condition_ids:
        unknown = sorted(set(condition_ids) - set(conditions["condition_id"]))
        if unknown:
            raise ValueError(f"unknown condition_id(s) {unknown}; valid: {conditions['condition_id'].tolist()}")
        selected = selected[selected["condition_id"].isin(condition_ids)]
    if n is not None:
        selected = selected[selected["n"] == n]
    if baseline is not None:
        selected = selected[
            (selected["n"] == baseline["n"])
            & (selected["treat_prob"] == baseline["treat_prob"])
            & (selected["effect"] == baseline["effect"])
        ]
    if selected.empty:
        raise ValueError("no condition matches the given filters")
    return selected.reset_index(drop=True)


def condition_status(conditions: pd.DataFrame, runs: pd.DataFrame | None, n_reps: int) -> pd.DataFrame:
    """Add completed / planned replication counts per condition from an existing runs table."""
    status = conditions.copy()
    completed = {}
    if runs is not None and not runs.empty:
        keys = [replication_key(*r) for r in runs[[*CONDITION_KEYS, "rep"]].itertuples(index=False)]
        reps = pd.Series([k[-1] for k in keys], index=[k[:4] for k in keys])
        completed = reps.groupby(level=0).nunique().to_dict()
    status["completed_reps"] = [
        completed.get(replication_key(*r, 0)[:4], 0) for r in status[CONDITION_KEYS].itertuples(index=False)
    ]
    status["planned_reps"] = n_reps
    return status


def check_condition_ids(runs: pd.DataFrame, conditions: pd.DataFrame) -> None:
    """Raise if a stored condition_id disagrees with the id the current design assigns to that condition."""
    current = {replication_key(*r, 0)[:4]: cid for cid, *r in
               conditions[["condition_id", *CONDITION_KEYS]].itertuples(index=False)}
    for cid, *r in runs[["condition_id", *CONDITION_KEYS]].drop_duplicates().itertuples(index=False):
        key = replication_key(*r, 0)[:4]
        if key in current and current[key] != cid:
            raise RuntimeError(f"condition_id mismatch for {key}: stored {cid} vs design {current[key]}")


def pending_replications(
    conditions: pd.DataFrame, n_reps: int, done: set[tuple], max_new: int | None = None
) -> list[tuple[dict, int]]:
    """(condition, rep) pairs not yet in `done`, in condition order then ascending rep.

    `max_new` caps how many are returned, so a long condition can be run in batches;
    rerunning with the same cap picks up the next unfinished replications.
    """
    if max_new is not None and max_new < 1:
        raise ValueError(f"max_new must be >= 1, got {max_new}")
    pending = [
        (cond, rep)
        for cond in conditions.to_dict(orient="records")
        for rep in range(n_reps)
        if replication_key(*(cond[k] for k in CONDITION_KEYS), rep) not in done
    ]
    return pending if max_new is None else pending[:max_new]


def replication_seeds(rep: int, seeds_cfg: dict) -> tuple[int, int]:
    return seeds_cfg["data_seed_base"] + rep, seeds_cfg["model_seed_base"] + rep


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def run_replication(condition: dict, rep: int, model_cfg: dict, seeds_cfg: dict) -> list[dict]:
    """Generate one dataset, fit all models on the same split, return one row per model."""
    data_seed, model_seed = replication_seeds(rep, seeds_cfg)
    sim = SimulationConfig(
        n=int(condition["n"]),
        treat_prob=float(condition["treat_prob"]),
        effect=condition["effect"],
        seed=data_seed,
        noise_sd=float(condition["noise_sd"]),
    )
    df = generate_data(sim)
    train, test = split_train_test(df, model_cfg["split"]["test_size"], data_seed)
    tr, te = group_counts(train), group_counts(test)

    models = build_models(model_cfg, seed=model_seed, treat_prob=sim.treat_prob)
    metrics, _, _ = evaluate_models(models, train, test, sim.n_covariates)
    baseline = constant_ate_baseline(train, test)
    del models
    gc.collect()

    common = {
        "condition_id": condition["condition_id"],
        "experiments": condition["experiments"],
        **{k: condition[k] for k in CONDITION_KEYS},
        "rep": rep,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "n_train_treated": tr["treated"],
        "n_train_control": tr["control"],
        "n_test_treated": te["treated"],
        "n_test_control": te["control"],
    }
    rows = [
        {
            **common,
            "model": m["Model"],
            "rmse": m["RMSE"],
            "mae": m["MAE"],
            "train_time_sec": m["Train Time (sec)"],
            "pred_mean": m["Pred Mean"],
            "pred_sd": m["Pred SD"],
        }
        for m in metrics.to_dict(orient="records")
    ]
    rows.append(
        {
            **common,
            "model": BASELINE_MODEL,
            "rmse": baseline["RMSE"],
            "mae": baseline["MAE"],
            "train_time_sec": np.nan,
            "pred_mean": baseline["ate_estimate"],
            "pred_sd": 0.0,
        }
    )
    peak = peak_rss_mb()
    for row in rows:
        row["peak_rss_mb"] = peak
    return rows


WIDE_CI_REL = 0.10  # CI half-width above 10% of the mean is flagged as imprecise


def _t_halfwidth(sd: float, k: int, confidence: float) -> float:
    return stats.t.ppf(0.5 + confidence / 2, k - 1) * sd / np.sqrt(k) if k > 1 else np.nan


def summarize_runs(runs: pd.DataFrame, confidence: float = 0.95, n_reps_planned: int | None = None) -> pd.DataFrame:
    """Mean, SD and t-based CI of RMSE (plus MAE / time) per condition x model.

    The CI is for the mean RMSE over independent replications of one condition.
    `precision_note` flags fewer replications than planned and CIs whose
    half-width exceeds WIDE_CI_REL of the mean.
    """
    group_cols = ["condition_id", "experiments", *CONDITION_KEYS, "model"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        k = len(g)
        mean, sd = g["rmse"].mean(), g["rmse"].std(ddof=1)
        half = _t_halfwidth(sd, k, confidence)
        notes = []
        if n_reps_planned is not None and k < n_reps_planned:
            notes.append(f"n_reps {k} < planned {n_reps_planned}")
        if k < 2:
            notes.append("CI undefined (k<2)")
        elif half > WIDE_CI_REL * mean:
            notes.append(f"wide CI (half-width {half / mean:.0%} of mean)")
        return pd.Series(
            {
                "n_reps": k,
                "rmse_mean": mean,
                "rmse_sd": sd,
                "rmse_ci_low": mean - half,
                "rmse_ci_high": mean + half,
                "rmse_ci_halfwidth": half,
                "mae_mean": g["mae"].mean(),
                "mae_sd": g["mae"].std(ddof=1),
                "train_time_mean": g["train_time_sec"].mean(),
                "train_time_sd": g["train_time_sec"].std(ddof=1),
                "precision_note": "; ".join(notes),
            }
        )

    summary = runs.groupby(group_cols, sort=False).apply(_agg, include_groups=False).reset_index()
    summary["n_reps"] = summary["n_reps"].astype(int)
    return summary


def _holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def paired_differences(runs: pd.DataFrame, confidence: float = 0.95, n_reps_planned: int | None = None) -> pd.DataFrame:
    """Paired RMSE differences between every pair of models, matched on (condition, rep).

    diff = RMSE(model_a) - RMSE(model_b); negative means model_a is more accurate.
    CI: paired t interval over replications. p_value: paired t-test; p_holm:
    Holm-adjusted within each condition across all model pairs.
    """
    models = list(dict.fromkeys(runs["model"]))
    out = []
    for cond_vals, g in runs.groupby(["condition_id", "experiments", *CONDITION_KEYS], sort=False):
        wide = g.pivot(index="rep", columns="model", values="rmse")
        rows = []
        for a, b in itertools.combinations(models, 2):
            pair = wide[[a, b]].dropna()
            d = (pair[a] - pair[b]).to_numpy()
            k = len(d)
            mean, sd = d.mean(), d.std(ddof=1) if k > 1 else np.nan
            half = _t_halfwidth(sd, k, confidence)
            if k > 1 and sd > 0:
                t_stat = mean / (sd / np.sqrt(k))
                p = 2 * stats.t.sf(abs(t_stat), k - 1)
            else:
                t_stat, p = np.nan, np.nan
            rows.append(
                {
                    **dict(zip(["condition_id", "experiments", *CONDITION_KEYS], cond_vals)),
                    "model_a": a,
                    "model_b": b,
                    "n_pairs": k,
                    "diff_mean": mean,
                    "diff_sd": sd,
                    "diff_ci_low": mean - half,
                    "diff_ci_high": mean + half,
                    "t_stat": t_stat,
                    "p_value": p,
                    "a_better_reps": int((d < 0).sum()),
                }
            )
        block = pd.DataFrame(rows)
        valid = block["p_value"].notna()
        block["p_holm"] = np.nan
        block.loc[valid, "p_holm"] = _holm(block.loc[valid, "p_value"].to_numpy())

        def _note(r) -> str:
            notes = []
            if n_reps_planned is not None and r["n_pairs"] < n_reps_planned:
                notes.append(f"n_pairs {r['n_pairs']} < planned {n_reps_planned}")
            if r["n_pairs"] < 2:
                notes.append("CI undefined (k<2)")
            elif r["diff_ci_low"] <= 0 <= r["diff_ci_high"]:
                notes.append("CI includes 0")
            return "; ".join(notes)

        block["note"] = block.apply(_note, axis=1)
        out.append(block)
    return pd.concat(out, ignore_index=True)


def replication_key(n, treat_prob, effect, noise_sd, rep) -> tuple:
    """Identity of one replication, independent of condition_id numbering."""
    return int(n), round(float(treat_prob), 10), str(effect), round(float(noise_sd), 10), int(rep)


def check_runs_integrity(runs: pd.DataFrame, seeds_cfg: dict, expected_models: set[str]) -> set[tuple]:
    """Validate an existing runs table and return the keys of completed replications.

    Raises if any (replication, model) row is duplicated, if a replication is missing
    models, if seeds break the seeding rule, or if one condition_id maps to
    several conditions.
    """
    if runs.empty:
        return set()
    keys = [replication_key(*r) for r in runs[[*CONDITION_KEYS, "rep"]].itertuples(index=False)]
    tagged = runs.assign(_key=keys)
    dup = tagged.duplicated(["_key", "model"])
    if dup.any():
        raise RuntimeError(f"duplicated (replication, model) rows: {tagged.loc[dup, ['_key', 'model']].values.tolist()[:5]}")
    per_rep = tagged.groupby("_key")["model"].agg(set)
    incomplete = per_rep[per_rep.apply(lambda s: s != expected_models)]
    if len(incomplete):
        raise RuntimeError(f"replications with missing/extra models: {list(incomplete.index)[:5]}")
    bad_seed = (runs["data_seed"] != seeds_cfg["data_seed_base"] + runs["rep"]) | (
        runs["model_seed"] != seeds_cfg["model_seed_base"] + runs["rep"]
    )
    if bad_seed.any():
        raise RuntimeError("seeds in existing runs do not follow the configured seeding rule")
    ids = runs.groupby("condition_id")[CONDITION_KEYS].nunique()
    if (ids > 1).any().any():
        raise RuntimeError("a condition_id maps to more than one condition")
    return set(per_rep.index)


def model_fit_counts(model_cfg: dict) -> dict:
    """Number of fitted learners per replication (estimator level and base-learner level)."""
    cv = model_cfg["causal_forest"]["cv"]
    return {
        "estimators": 3,
        "T-Learner": 2,
        "X-Learner": 4,
        # cross-fitted model_y and model_t per fold + final causal forest
        "Causal Forest": 2 * cv + 1,
    }


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start
