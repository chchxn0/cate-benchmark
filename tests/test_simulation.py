import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.simulation import (
    BASELINE_MODEL,
    build_conditions,
    replication_seeds,
    run_replication,
    summarize_runs,
)

BASELINE = {"n": 1000, "treat_prob": 0.5, "effect": "linear", "noise_sd": 1.0}
FACTORS = {"n": [1000, 10000, 100000], "treat_prob": [0.5, 0.3, 0.1], "effect": ["linear", "nonlinear", "interaction"]}
SEEDS = {"data_seed_base": 42, "model_seed_base": 1000}
FAST_CFG = {
    "base_learner": {"n_estimators": 10, "min_samples_leaf": 5, "max_features": 1.0, "n_jobs": 1},
    "x_learner": {"propensity": None},
    "causal_forest": {"n_estimators": 40, "min_samples_leaf": 5, "cv": 2, "n_jobs": 1},
    "split": {"test_size": 0.2, "stratify": "T"},
}
SMALL_COND = {"condition_id": "c00", "experiments": "test", "n": 300, "treat_prob": 0.5, "effect": "linear", "noise_sd": 1.0}


def test_ofat_conditions_vary_one_factor_and_share_baseline():
    conds = build_conditions(BASELINE, FACTORS, "ofat")
    assert len(conds) == 7
    assert conds[["n", "treat_prob", "effect"]].duplicated().sum() == 0
    for row in conds.itertuples():
        changed = sum([row.n != 1000, row.treat_prob != 0.5, row.effect != "linear"])
        assert changed <= 1
    base = conds[(conds.n == 1000) & (conds.treat_prob == 0.5) & (conds.effect == "linear")]
    assert base["experiments"].item() == "sample_size|treat_ratio|complexity"


def test_full_factorial_has_27_conditions():
    conds = build_conditions(BASELINE, FACTORS, "full")
    assert len(conds) == 27
    assert conds[["n", "treat_prob", "effect"]].duplicated().sum() == 0


def test_data_and_model_seeds_are_distinct_streams():
    seeds = [replication_seeds(r, SEEDS) for r in range(30)]
    data_seeds, model_seeds = zip(*seeds)
    assert len(set(data_seeds)) == 30 and len(set(model_seeds)) == 30
    assert set(data_seeds).isdisjoint(model_seeds)


@pytest.fixture(scope="module")
def two_reps():
    return [pd.DataFrame(run_replication(SMALL_COND, rep, FAST_CFG, SEEDS)) for rep in (0, 1)]


def test_replication_shares_data_across_models(two_reps):
    rows = two_reps[0]
    assert set(rows["model"]) == {"T-Learner", "X-Learner", "Causal Forest", BASELINE_MODEL}
    # one data seed / split per replication -> identical counts for every model
    for col in ["data_seed", "model_seed", "n_train_treated", "n_train_control", "n_test_treated", "n_test_control"]:
        assert rows[col].nunique() == 1
    assert rows["n_train_treated"].iloc[0] + rows["n_train_control"].iloc[0] == 240


def test_replications_use_independent_data(two_reps):
    r0, r1 = two_reps
    assert r0["data_seed"].iloc[0] != r1["data_seed"].iloc[0]
    assert r0["model_seed"].iloc[0] != r1["model_seed"].iloc[0]
    assert not np.allclose(r0["rmse"], r1["rmse"])


def test_replication_is_reproducible(two_reps):
    again = pd.DataFrame(run_replication(SMALL_COND, 0, FAST_CFG, SEEDS))
    cols = ["model", "rmse", "mae", "pred_mean", "pred_sd"]
    pd.testing.assert_frame_equal(two_reps[0][cols], again[cols])


def test_summary_mean_sd_ci():
    rmse = np.array([0.2, 0.3, 0.4, 0.25])
    runs = pd.DataFrame(
        {
            "condition_id": "c00", "experiments": "x", "n": 1000, "treat_prob": 0.5, "effect": "linear",
            "noise_sd": 1.0, "model": "M", "rep": range(4), "rmse": rmse, "mae": rmse / 2, "train_time_sec": 1.0,
        }
    )
    s = summarize_runs(runs, 0.95).iloc[0]
    half = stats.t.ppf(0.975, 3) * rmse.std(ddof=1) / 2
    assert s["n_reps"] == 4
    assert s["rmse_mean"] == pytest.approx(rmse.mean())
    assert s["rmse_sd"] == pytest.approx(rmse.std(ddof=1))
    assert s["rmse_ci_low"] == pytest.approx(rmse.mean() - half)
    assert s["rmse_ci_high"] == pytest.approx(rmse.mean() + half)


# STEP 4: paired comparisons, precision flags, resume integrity -----------------------
from src.simulation import _holm, check_runs_integrity, paired_differences, replication_key  # noqa: E402

MODELS = ["T-Learner", "X-Learner", "Causal Forest", BASELINE_MODEL]


def _fake_runs(rmse_by_model: dict, n_reps: int) -> pd.DataFrame:
    rows = []
    for rep in range(n_reps):
        for m, values in rmse_by_model.items():
            rows.append({
                "condition_id": "c00", "experiments": "x", "n": 1000, "treat_prob": 0.5, "effect": "linear",
                "noise_sd": 1.0, "rep": rep, "data_seed": 42 + rep, "model_seed": 1000 + rep, "model": m,
                "rmse": values[rep], "mae": values[rep] / 2, "train_time_sec": 1.0,
            })
    return pd.DataFrame(rows)


def test_paired_differences_match_manual_computation():
    rng = np.random.default_rng(0)
    vals = {m: rng.uniform(0.1, 0.4, 10) for m in MODELS}
    out = paired_differences(_fake_runs(vals, 10), 0.95)
    assert len(out) == 6  # all model pairs
    row = out[(out.model_a == "T-Learner") & (out.model_b == "Causal Forest")].iloc[0]
    d = vals["T-Learner"] - vals["Causal Forest"]
    half = stats.t.ppf(0.975, 9) * d.std(ddof=1) / np.sqrt(10)
    assert row["diff_mean"] == pytest.approx(d.mean())
    assert row["diff_ci_low"] == pytest.approx(d.mean() - half)
    assert row["p_value"] == pytest.approx(stats.ttest_rel(vals["T-Learner"], vals["Causal Forest"]).pvalue)
    assert row["a_better_reps"] == int((d < 0).sum())


def test_paired_uses_same_rep_not_sorted_values():
    # A is better than B in every replication even though their marginal ranges overlap
    a = np.array([0.10, 0.30, 0.50, 0.20])
    vals = {"T-Learner": a, "X-Learner": a + 0.01, "Causal Forest": a + 0.02, BASELINE_MODEL: a + 0.03}
    out = paired_differences(_fake_runs(vals, 4))
    row = out[(out.model_a == "T-Learner") & (out.model_b == "X-Learner")].iloc[0]
    assert row["diff_mean"] == pytest.approx(-0.01)
    assert row["a_better_reps"] == 4


def test_holm_adjustment():
    np.testing.assert_allclose(_holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_precision_notes():
    rng = np.random.default_rng(1)
    noisy = {m: rng.uniform(0.05, 0.5, 3) for m in MODELS}
    s = summarize_runs(_fake_runs(noisy, 3), 0.95, n_reps_planned=30)
    assert s["precision_note"].str.contains("planned 30").all()
    assert s["precision_note"].str.contains("wide CI").any()
    p = paired_differences(_fake_runs(noisy, 3), 0.95, n_reps_planned=30)
    assert p["note"].str.contains("planned 30").all()


def test_integrity_returns_completed_keys_and_rejects_bad_runs():
    vals = {m: np.full(3, 0.2) for m in MODELS}
    runs = _fake_runs(vals, 3)
    done = check_runs_integrity(runs, SEEDS, set(MODELS))
    assert done == {replication_key(1000, 0.5, "linear", 1.0, r) for r in range(3)}
    with pytest.raises(RuntimeError, match="duplicated"):
        check_runs_integrity(pd.concat([runs, runs.iloc[[0]]]), SEEDS, set(MODELS))
    with pytest.raises(RuntimeError, match="missing"):
        check_runs_integrity(runs.iloc[1:], SEEDS, set(MODELS))
    bad = runs.copy()
    bad.loc[0, "data_seed"] = 7
    with pytest.raises(RuntimeError, match="seed"):
        check_runs_integrity(bad, SEEDS, set(MODELS))


def test_resume_does_not_rerun_completed_replications(tmp_path):
    import subprocess
    import sys

    import yaml

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    exp = {
        "baseline": {"n": 300, "treat_prob": 0.5, "effect": "linear", "noise_sd": 1.0},
        "factors": {"n": [300], "treat_prob": [0.5], "effect": ["linear"]},
        "design": "ofat", "n_reps": 3, "seeds": SEEDS, "confidence_level": 0.95, "output_dir": str(tmp_path),
    }
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(exp))
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(FAST_CFG))
    cmd = [sys.executable, str(root / "scripts" / "run_simulation.py"), "--config", str(tmp_path / "exp.yaml"),
           "--models", str(tmp_path / "models.yaml"), "--tag", "t"]

    subprocess.run(cmd + ["--reps", "2"], check=True, capture_output=True)
    first = pd.read_csv(tmp_path / "t_runs.csv")
    # re-running without --resume must refuse and leave the file untouched
    assert subprocess.run(cmd, capture_output=True).returncode != 0
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "t_runs.csv"), first)

    out = subprocess.run(cmd + ["--resume"], check=True, capture_output=True, text=True).stdout
    runs = pd.read_csv(tmp_path / "t_runs.csv")
    assert sorted(runs["rep"].unique()) == [0, 1, 2]
    assert not runs.duplicated(["rep", "model"]).any()
    pd.testing.assert_frame_equal(runs.iloc[: len(first)].reset_index(drop=True), first)
    assert out.count("rep=") == 1  # only rep 2 was executed

    snapshot = {f: (tmp_path / f).read_bytes() for f in ("t_runs.csv", "t_summary.csv", "t_paired.csv", "t_metadata.json")}
    out = subprocess.run(cmd + ["--resume"], check=True, capture_output=True, text=True).stdout  # nothing left to do
    assert "Nothing to run" in out
    assert {f: (tmp_path / f).read_bytes() for f in snapshot} == snapshot  # no file touched
    meta = __import__("json").loads((tmp_path / "t_metadata.json").read_text())
    assert [inv["replications_run"] for inv in meta["invocations"]] == [2, 1]
    assert (tmp_path / "t_paired.csv").exists()


# Condition filter (--condition-id) ---------------------------------------------------
from src.simulation import check_condition_ids, condition_status, select_conditions  # noqa: E402

FORMAL_BASELINE = {"n": 10000, "treat_prob": 0.5, "effect": "linear", "noise_sd": 1.0}


@pytest.fixture
def formal_conditions():
    return build_conditions(FORMAL_BASELINE, FACTORS, "ofat")


@pytest.mark.parametrize(
    "cid, expected",
    [
        ("c00", (1000, 0.5, "linear")),
        ("c01", (10000, 0.5, "linear")),
        ("c02", (100000, 0.5, "linear")),
        ("c03", (10000, 0.3, "linear")),
        ("c04", (10000, 0.1, "linear")),
        ("c05", (10000, 0.5, "nonlinear")),
        ("c06", (10000, 0.5, "interaction")),
    ],
)
def test_condition_ids_match_formal_plan(formal_conditions, cid, expected):
    sel = select_conditions(formal_conditions, condition_ids=[cid])
    assert len(sel) == 1
    assert (sel.loc[0, "n"], sel.loc[0, "treat_prob"], sel.loc[0, "effect"]) == expected
    assert sel.loc[0, "condition_id"] == cid


def test_select_multiple_ids_and_intersection(formal_conditions):
    assert select_conditions(formal_conditions, condition_ids=["c03", "c05"])["condition_id"].tolist() == ["c03", "c05"]
    assert select_conditions(formal_conditions, n=10000)["condition_id"].tolist() == ["c01", "c03", "c04", "c05", "c06"]
    assert select_conditions(formal_conditions, baseline=FORMAL_BASELINE)["condition_id"].tolist() == ["c01"]
    with pytest.raises(ValueError, match="no condition"):
        select_conditions(formal_conditions, condition_ids=["c00"], n=10000)


def test_unknown_condition_id_rejected(formal_conditions):
    with pytest.raises(ValueError, match="unknown condition_id"):
        select_conditions(formal_conditions, condition_ids=["c99"])


def test_condition_status_counts_completed_reps(formal_conditions):
    vals = {m: np.full(3, 0.2) for m in MODELS}
    runs = _fake_runs(vals, 3)  # c00: n=1000, p=0.5, linear, 3 reps
    status = condition_status(formal_conditions, runs, 30).set_index("condition_id")
    assert status.loc["c00", "completed_reps"] == 3
    assert status.loc[["c01", "c02", "c03", "c04", "c05", "c06"], "completed_reps"].eq(0).all()
    assert (status["planned_reps"] == 30).all()


def test_condition_id_mismatch_detected(formal_conditions):
    runs = _fake_runs({m: np.full(1, 0.2) for m in MODELS}, 1)
    check_condition_ids(runs, formal_conditions)  # c00 is consistent
    runs["condition_id"] = "c03"
    with pytest.raises(RuntimeError, match="mismatch"):
        check_condition_ids(runs, formal_conditions)


def test_condition_id_cli_preserves_other_conditions(tmp_path):
    import json
    import subprocess
    import sys

    import yaml

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    exp = {
        "baseline": {"n": 300, "treat_prob": 0.5, "effect": "linear", "noise_sd": 1.0},
        "factors": {"n": [200, 300], "treat_prob": [0.5, 0.3], "effect": ["linear"]},
        "design": "ofat", "n_reps": 2, "seeds": SEEDS, "confidence_level": 0.95, "output_dir": str(tmp_path),
    }
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(exp))
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(FAST_CFG))
    cmd = [sys.executable, str(root / "scripts" / "run_simulation.py"), "--config", str(tmp_path / "exp.yaml"),
           "--models", str(tmp_path / "models.yaml"), "--tag", "t"]
    runs_path = tmp_path / "t_runs.csv"
    # conditions: c00 n=200, c01 n=300 (baseline), c02 n=300 p=0.3

    listing = subprocess.run(cmd + ["--list-conditions"], check=True, capture_output=True, text=True).stdout
    assert "c00" in listing and "c02" in listing and not runs_path.exists()

    subprocess.run(cmd + ["--condition-id", "c00"], check=True, capture_output=True)
    c00_rows = pd.read_csv(runs_path)
    assert set(c00_rows["condition_id"]) == {"c00"}

    out = subprocess.run(cmd + ["--condition-id", "c02", "--resume"], check=True, capture_output=True, text=True).stdout
    runs = pd.read_csv(runs_path)
    assert set(runs["condition_id"]) == {"c00", "c02"}
    assert out.count("rep=") == 2 and "[c02" in out and "[c00" not in out
    pd.testing.assert_frame_equal(runs[runs.condition_id == "c00"].reset_index(drop=True), c00_rows)
    c02 = runs[runs.condition_id == "c02"]
    assert (c02["n"] == 300).all() and (c02["treat_prob"] == 0.3).all()

    # re-running a finished condition does nothing; status reflects both conditions
    out = subprocess.run(cmd + ["--condition-id", "c02", "--resume"], check=True, capture_output=True, text=True).stdout
    assert out.count("rep=") == 0 and len(pd.read_csv(runs_path)) == len(runs)
    listing = subprocess.run(cmd + ["--list-conditions"], check=True, capture_output=True, text=True).stdout
    assert {line.split()[0]: line.split()[-2] for line in listing.splitlines()[2:]} == {"c00": "2", "c01": "0", "c02": "2"}

    # destructive or invalid invocations are refused and leave the file untouched
    before = runs_path.read_bytes()
    for bad in (["--condition-id", "c02"], ["--condition-id", "c02", "--overwrite"], ["--condition-id", "c99", "--resume"]):
        assert subprocess.run(cmd + bad, capture_output=True).returncode != 0
        assert runs_path.read_bytes() == before

    summary = pd.read_csv(tmp_path / "t_summary.csv")
    assert set(summary["condition_id"]) == {"c00", "c02"}
    meta = json.loads((tmp_path / "t_metadata.json").read_text())
    assert [inv["conditions"] for inv in meta["invocations"]] == [["c00"], ["c02"]]


# Batched execution (--max-new-reps) --------------------------------------------------
from src.simulation import pending_replications  # noqa: E402


def test_pending_replications_order_and_cap(formal_conditions):
    c02 = select_conditions(formal_conditions, condition_ids=["c02"])
    done = {replication_key(100000, 0.5, "linear", 1.0, r) for r in (0, 1, 3)}
    pending = pending_replications(c02, 10, done)
    assert [r for _, r in pending] == [2, 4, 5, 6, 7, 8, 9]  # gaps filled first, ascending
    assert [r for _, r in pending_replications(c02, 10, done, max_new=3)] == [2, 4, 5]
    assert pending_replications(c02, 10, done, max_new=100) == pending
    full = {replication_key(100000, 0.5, "linear", 1.0, r) for r in range(10)}
    assert pending_replications(c02, 10, full, max_new=5) == []
    with pytest.raises(ValueError):
        pending_replications(c02, 10, done, max_new=0)


def test_max_new_reps_batches_resume_and_stop(tmp_path):
    import json
    import subprocess
    import sys

    import yaml

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    exp = {
        "baseline": {"n": 300, "treat_prob": 0.5, "effect": "linear", "noise_sd": 1.0},
        "factors": {"n": [200, 300], "treat_prob": [0.5], "effect": ["linear"]},
        "design": "ofat", "n_reps": 5, "seeds": SEEDS, "confidence_level": 0.95, "output_dir": str(tmp_path),
    }
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(exp))
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(FAST_CFG))
    base = [sys.executable, str(root / "scripts" / "run_simulation.py"), "--config", str(tmp_path / "exp.yaml"),
            "--models", str(tmp_path / "models.yaml"), "--tag", "t"]
    runs_path = tmp_path / "t_runs.csv"
    # conditions: c00 n=200, c01 n=300 (baseline). c00 plays the role of an already finished condition.
    subprocess.run(base + ["--condition-id", "c00"], check=True, capture_output=True)
    finished = pd.read_csv(runs_path)
    finished_bytes = runs_path.read_bytes()

    batch = base + ["--condition-id", "c01", "--resume", "--max-new-reps", "2"]
    expected_batches = [[0, 1], [2, 3], [4]]
    for i, reps in enumerate(expected_batches):
        out = subprocess.run(batch, check=True, capture_output=True, text=True).stdout
        assert [int(x.split("rep=")[1].split("]")[0]) for x in out.splitlines() if x.startswith("[c01")] == reps
        runs = pd.read_csv(runs_path)
        c01 = runs[runs.condition_id == "c01"]
        assert sorted(c01["rep"].unique()) == list(range(sum(len(b) for b in expected_batches[: i + 1])))
        assert runs_path.read_bytes().startswith(finished_bytes)  # finished condition untouched
        pd.testing.assert_frame_equal(runs[runs.condition_id == "c00"].reset_index(drop=True), finished)

    # seeds follow the rule regardless of batching, and no duplicates exist
    assert (c01["data_seed"] == 42 + c01["rep"]).all() and (c01["model_seed"] == 1000 + c01["rep"]).all()
    assert not runs.duplicated(["condition_id", "rep", "model"]).any()

    # batched results equal a single uninterrupted run with the same seeds
    single = tmp_path / "single"
    single.mkdir()
    exp_single = {**exp, "output_dir": str(single)}
    (single / "exp.yaml").write_text(yaml.safe_dump(exp_single))
    subprocess.run([*base[:2], "--config", str(single / "exp.yaml"), "--models", str(tmp_path / "models.yaml"),
                    "--tag", "t", "--condition-id", "c01"], check=True, capture_output=True)
    ref = pd.read_csv(single / "t_runs.csv")
    cols = ["rep", "model", "data_seed", "model_seed", "rmse", "mae", "pred_mean", "pred_sd"]
    pd.testing.assert_frame_equal(c01[cols].reset_index(drop=True), ref[cols])

    # once complete, the same command trains nothing and modifies no file
    files = ("t_runs.csv", "t_summary.csv", "t_paired.csv", "t_metadata.json")
    snapshot = {f: (tmp_path / f).read_bytes() for f in files}
    out = subprocess.run(batch, check=True, capture_output=True, text=True).stdout
    assert "Nothing to run" in out and "[c01" not in out
    assert {f: (tmp_path / f).read_bytes() for f in files} == snapshot

    meta = json.loads((tmp_path / "t_metadata.json").read_text())
    batches = [inv for inv in meta["invocations"] if inv["max_new_reps"] == 2]
    assert [b["replications_run"] for b in batches] == [2, 2, 1]
    assert [b["replications_pending_after"] for b in batches] == [3, 1, 0]
    summary = pd.read_csv(tmp_path / "t_summary.csv")
    assert (summary.loc[summary.condition_id == "c01", "n_reps"] == 5).all()

    assert subprocess.run(base + ["--condition-id", "c01", "--resume", "--max-new-reps", "0"],
                          capture_output=True).returncode != 0
