"""STEP 6-1/6-2: verify and integrate the simulation (STEP 4) and Criteo (STEP 5) results. No training, no test re-evaluation.

Reads only existing result files (results/tables/*.csv|json, final_plot_data_*.csv) and writes
    results/tables/step6_final_simulation_summary.csv   condition x model RMSE / MAE / rank / time
    results/tables/step6_final_criteo_summary.csv       model x stage (100k valid, 300k valid, final test)
    results/tables/step6_simulation_key_tests.csv       key paired / cross-condition tests (+ supplementary MAE)
    results/tables/step6_criteo_key_tests.csv           validation and test paired comparisons
    results/tables/step6_verification_checks.csv        every consistency check and claim check with its result
    results/figures/step6_final/                        presentation copies of six existing figures (PNG 300 dpi + PDF)

    .venv/bin/python scripts/step6_summarize.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
T = PROJECT_ROOT / "results" / "tables"
F = PROJECT_ROOT / "results" / "figures"
OUT_FIG = F / "step6_final"
MODELS = ["T-Learner", "X-Learner", "Causal Forest"]
CONDS = ["c00", "c01", "c02", "c03", "c04", "c05", "c06"]
checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        print(f"[FAIL] {name}: {detail}", flush=True)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def simulation() -> None:
    runs, summ = pd.read_csv(T / "formal_runs.csv"), pd.read_csv(T / "formal_summary.csv")
    paired = pd.read_csv(T / "formal_paired.csv")
    g = runs.groupby(["condition_id", "model"])
    rec = pd.DataFrame({"mean": g.rmse.mean(), "sd": g.rmse.std(), "mae": g.mae.mean(), "reps": g.rep.nunique()})
    s = summ.set_index(["condition_id", "model"])
    check("sim: 840 rows, 7 conditions x 4 models x 30 reps", len(runs) == 840 and set(rec.reps) == {30})
    check("sim: summary RMSE mean/SD equal recomputation from runs",
          np.allclose(rec["mean"], s.loc[rec.index, "rmse_mean"], atol=1e-6) and np.allclose(rec["sd"], s.loc[rec.index, "rmse_sd"], atol=1e-6))
    check("sim: summary MAE equal recomputation from runs", np.allclose(rec["mae"], s.loc[rec.index, "mae_mean"], atol=1e-6))
    check("sim: seed rule data_seed=42+rep, model_seed=1000+rep",
          bool(((runs.data_seed == 42 + runs.rep) & (runs.model_seed == 1000 + runs.rep)).all()))

    rows = []
    for c in CONDS:
        sub = summ[summ.condition_id == c].copy()
        sub["rank_rmse_all4"] = sub.rmse_mean.rank().astype(int)
        sub["rank_mae_all4"] = sub.mae_mean.rank().astype(int)
        for r in sub.itertuples():
            rows.append({"condition_id": c, "n": int(r.n), "treat_prob": r.treat_prob, "effect": r.effect, "model": r.model,
                         "n_reps": int(r.n_reps), "rmse_mean": r.rmse_mean, "rmse_sd": r.rmse_sd, "rmse_ci_low": r.rmse_ci_low,
                         "rmse_ci_high": r.rmse_ci_high, "mae_mean": r.mae_mean, "mae_sd": r.mae_sd,
                         "train_time_mean_sec": r.train_time_mean, "rank_rmse": r.rank_rmse_all4, "rank_mae": r.rank_mae_all4,
                         "replication_wall_mean_sec": float(runs[runs.condition_id == c].groupby("rep").replication_wall_sec.first().mean())})
    pd.DataFrame(rows).to_csv(T / "step6_final_simulation_summary.csv", index=False, float_format="%.6g")

    tests = []
    for c in CONDS:
        for a, b in [("X-Learner", "Causal Forest"), ("T-Learner", "X-Learner"), ("T-Learner", "Constant ATE")]:
            r = paired[(paired.condition_id == c) & (paired.model_a == a) & (paired.model_b == b)].iloc[0]
            tests.append({"family": "within-condition RMSE (paired t, Holm over 6 pairs)", "condition_id": c,
                          "comparison": f"{a} - {b}", "metric": "rmse", "diff": r.diff_mean, "ci_low": r.diff_ci_low,
                          "ci_high": r.diff_ci_high, "p_adj": r.p_holm, "a_better_reps": int(r.a_better_reps),
                          "significant_0.05": bool(r.p_holm < 0.05), "source": "formal_paired.csv"})
        xa = runs[(runs.condition_id == c) & (runs.model == "X-Learner")].set_index("rep").mae.sort_index()
        cb = runs[(runs.condition_id == c) & (runs.model == "Causal Forest")].set_index("rep").mae.sort_index()
        d = xa - cb
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        tests.append({"family": "supplementary (STEP 6 re-analysis of formal_runs): MAE paired t, unadjusted",
                      "condition_id": c, "comparison": "X-Learner - Causal Forest", "metric": "mae", "diff": d.mean(),
                      "ci_low": d.mean() - h, "ci_high": d.mean() + h, "p_adj": float(stats.ttest_1samp(d, 0).pvalue),
                      "a_better_reps": int((d < 0).sum()), "significant_0.05": bool(stats.ttest_1samp(d, 0).pvalue < 0.05),
                      "source": "formal_runs.csv"})
    for f, pcol, fam in (("formal_sample_size_comparison", "p_holm_12", "sample size (Holm over 12)"),
                         ("formal_treat_ratio_comparison", "p_holm_8", "treatment ratio (Holm over 8)"),
                         ("formal_complexity_comparison", "p_holm_8", "effect structure (Holm over 8)")):
        d = pd.read_csv(T / f"{f}.csv")
        for r in d.itertuples():
            tests.append({"family": fam, "condition_id": "", "comparison": f"{r.model}: {r.comparison}", "metric": "rmse",
                          "diff": r.diff_mean, "ci_low": r.diff_ci_low, "ci_high": r.diff_ci_high, "p_adj": getattr(r, pcol),
                          "pct_change": r._asdict()["pct_change"] if "pct_change" in r._fields else np.nan, "significant_0.05": bool(getattr(r, pcol) < 0.05), "source": f"{f}.csv"})
    pd.DataFrame(tests).to_csv(T / "step6_simulation_key_tests.csv", index=False, float_format="%.6g")

    # claims made in earlier reports / README
    m = s["rmse_mean"]
    best = {c: summ[summ.condition_id == c].sort_values("rmse_mean").model.iloc[0] for c in CONDS}
    check("claim: CF lowest mean RMSE in linear conditions c00-c04", all(best[c] == "Causal Forest" for c in CONDS[:5]), str(best))
    check("claim: X lowest mean RMSE in c05 and c06", best["c05"] == best["c06"] == "X-Learner", str(best))
    p = paired.set_index(["condition_id", "model_a", "model_b"])
    check("claim: c05 X-CF RMSE difference not significant (p_holm 0.1059)", abs(p.loc[("c05", "X-Learner", "Causal Forest"), "p_holm"] - 0.1059) < 5e-4)
    check("claim: c06 X better than CF in 30/30 reps", p.loc[("c06", "X-Learner", "Causal Forest"), "a_better_reps"] == 30)
    check("claim: T-Learner worst of the three learners in all 7 conditions",
          all(m[(c, "T-Learner")] > max(m[(c, "X-Learner")], m[(c, "Causal Forest")]) for c in CONDS))
    t_worse_const = [c for c in CONDS if p.loc[(c, "T-Learner", "Constant ATE"), "diff_mean"] > 0 and p.loc[(c, "T-Learner", "Constant ATE"), "p_holm"] < 0.05]
    check("finding: T-Learner significantly worse than Constant ATE in c00 and c04", t_worse_const == ["c00", "c04"], str(t_worse_const))
    check("claim: CF 1k->100k RMSE -70.3%", abs(pd.read_csv(T / "formal_sample_size_comparison.csv").query(
        "model=='Causal Forest' and comparison=='n=100,000 - n=1,000'")["pct_change"].item() + 70.3) < 0.05)

    # figure data
    for f, ids in (("final_plot_data_sample_size", ["c00", "c01", "c02"]), ("final_plot_data_treatment_ratio", ["c01", "c03", "c04"]),
                   ("final_plot_data_complexity", ["c01", "c05", "c06"])):
        d = pd.read_csv(F / "final_step4" / f"{f}.csv").set_index(["condition_id", "model"])
        check(f"figure data {f} equals formal_summary",
              np.allclose(d.rmse_mean, s.loc[d.index, "rmse_mean"], atol=1e-6) and np.allclose(d.rmse_ci_halfwidth, s.loc[d.index, "rmse_ci_halfwidth"], atol=1e-6))


def criteo() -> None:
    rows = []
    stages = [("100k pilot", "validation", "pilot_100k"), ("300k", "validation", "300k")]
    vb = pd.read_csv(T / "criteo_300k_bootstrap_summary.csv")
    for label, split, tag in stages:
        met = pd.read_csv(T / f"criteo_{tag}_metrics.csv").set_index("model")
        rt = pd.read_csv(T / f"criteo_{tag}_runtime.csv").set_index("model")
        ps = pd.read_csv(T / f"criteo_{tag}_predictions_summary.csv").query("data=='valid'").set_index("model")
        for mod in MODELS:
            ci = vb[(vb.model == mod) & (vb.train == ("300k" if tag == "300k" else "100k")) & (vb.metric == "qini_coefficient")].iloc[0]
            rows.append({"stage": label, "eval_split": split, "model": mod, "n_train": int(rt.loc[mod, "n_train"]),
                         "train_conv_control": int(rt.loc[mod, "train_conv_control"]),
                         "qini_coefficient": met.loc[mod, "qini_coefficient"], "qini_ci95_low": ci.ci95_low, "qini_ci95_high": ci.ci95_high,
                         "auuc": met.loc[mod, "auuc"], "auuc_normalized": met.loc[mod, "auuc_normalized"],
                         **{f"top{q}_incremental_conversions": met.loc[mod, f"top{q}_incremental_conversions"] for q in (10, 20, 30)},
                         "share_pred_zero": ps.loc[mod, "mode_share"] if ps.loc[mod, "mode_share"] else np.nan,
                         "fit_sec": rt.loc[mod, "fit_sec"], "valid_predict_sec": rt.loc[mod, "full_predict_sec"],
                         "peak_rss_mb": rt.loc[mod, "peak_rss_mb"]})
    fm = pd.read_csv(T / "criteo_final_test_metrics.csv").set_index("model")
    fb = pd.read_csv(T / "criteo_final_test_bootstrap.csv")
    meta = json.loads((T / "criteo_final_test_metadata.json").read_text())
    rt = pd.read_csv(T / "criteo_300k_runtime.csv").set_index("model")
    for mod in MODELS:
        ci = fb[(fb.model == mod) & (fb.metric == "qini_coefficient")].iloc[0]
        rows.append({"stage": "300k final test", "eval_split": "test", "model": mod, "n_train": int(rt.loc[mod, "n_train"]),
                     "train_conv_control": int(rt.loc[mod, "train_conv_control"]),
                     "qini_coefficient": fm.loc[mod, "qini_coefficient"], "qini_ci95_low": ci.ci95_low, "qini_ci95_high": ci.ci95_high,
                     "auuc": fm.loc[mod, "auuc"], "auuc_normalized": fm.loc[mod, "auuc_normalized"],
                     **{f"top{q}_incremental_conversions": fm.loc[mod, f"top{q}_incremental_conversions"] for q in (10, 20, 30)},
                     "share_pred_zero": meta["predictions"][mod]["share_zero"], "fit_sec": rt.loc[mod, "fit_sec"],
                     "test_predict_sec": meta["predictions"][mod]["predict_sec"], "peak_rss_mb": rt.loc[mod, "peak_rss_mb"]})
    rr = fm.loc["Random (all-equal score)"]
    rows.append({"stage": "300k final test", "eval_split": "test", "model": "Random targeting", "qini_coefficient": 0.0,
                 "auuc": rr.auuc, "auuc_normalized": 0.0, **{f"top{q}_incremental_conversions": rr[f"top{q}_incremental_conversions"] for q in (10, 20, 30)}})
    pd.DataFrame(rows).to_csv(T / "step6_final_criteo_summary.csv", index=False, float_format="%.6g")

    tests = []
    for split, f in (("validation", "criteo_300k_paired_comparison.csv"), ("test", "criteo_final_test_paired.csv")):
        d = pd.read_csv(T / f)
        d = d[d.metric.isin(["qini_coefficient", "auuc", "top10_incremental_conversions", "top20_incremental_conversions",
                             "top30_incremental_conversions"])]
        if split == "validation":
            d = d[d.train == "300k"]
        for r in d.itertuples():
            tests.append({"eval_split": split, "comparison": r.comparison, "metric": r.metric, "diff": r.point_diff,
                          "ci95_low": r.ci95_low, "ci95_high": r.ci95_high, "ci_bonf3_low": r.ci_bonf3_low,
                          "ci_bonf3_high": r.ci_bonf3_high, "p_bonf3": r.p_bonf3, "significant_bonf3": bool(r.ci_bonf3_excludes_0),
                          "source": f})
    pv = pd.read_csv(T / "criteo_300k_paired_comparison.csv").query("train == '300k vs 100k' and metric == 'qini_coefficient'")
    for r in pv.itertuples():
        tests.append({"eval_split": "validation", "comparison": r.comparison, "metric": r.metric, "diff": r.point_diff,
                      "ci95_low": r.ci95_low, "ci95_high": r.ci95_high, "significant_bonf3": None,
                      "significant_95_unadjusted": bool(r.ci95_excludes_0), "source": "criteo_300k_paired_comparison.csv"})
    pd.DataFrame(tests).to_csv(T / "step6_criteo_key_tests.csv", index=False, float_format="%.6g")

    tg = pd.read_csv(T / "criteo_final_test_targeting.csv")
    check("criteo: final test SHA-256 of models/config/splits verified", all(meta["sha256_verified"].values()))
    check("criteo: test shares no rows with train/valid", not any(meta["index_overlap"].values()), str(meta["index_overlap"]))
    check("criteo: test n=2,795,918 (treated 2,376,530 / control 419,388)",
          meta["n"] == 2_795_918 and meta["n_treated"] == 2_376_530 and meta["n_control"] == 419_388)
    check("criteo: final metrics == bootstrap points == targeting points", all(
        np.isclose(fm.loc[mo, "qini_coefficient"], fb[(fb.model == mo) & (fb.metric == "qini_coefficient")].point.item(), rtol=1e-6)
        and all(np.isclose(fm.loc[mo, f"top{q}_incremental_conversions"],
                           tg[(tg.model == mo) & (tg.top_pct == q)].incremental_conversions.item(), rtol=1e-6) for q in (10, 20, 30))
        for mo in MODELS))
    check("claim: test Qini T 399.2 / X 691.7 / CF 685.4", np.allclose([fm.loc[m_, "qini_coefficient"] for m_ in MODELS], [399.2, 691.7, 685.4], atol=0.05))
    check("claim: all three Qini 95% CI lower bounds > 0 (beats random)",
          all(fb[(fb.model == m_) & (fb.metric == "qini_coefficient")].ci95_low.item() > 0 for m_ in MODELS))
    tp = pd.read_csv(T / "criteo_final_test_paired.csv").query("metric == 'qini_coefficient'").set_index("comparison")
    check("claim: test CF-T and X-T significant after Bonferroni, CF-X not",
          bool(tp.loc["Causal Forest - T-Learner", "ci_bonf3_excludes_0"] and tp.loc["X-Learner - T-Learner", "ci_bonf3_excludes_0"]
               and not tp.loc["Causal Forest - X-Learner", "ci_bonf3_excludes_0"]))
    vp = pd.read_csv(T / "criteo_300k_paired_comparison.csv").query("metric == 'qini_coefficient' and train == '300k'").set_index("comparison")
    check("claim: validation CF-X not significant after Bonferroni, CF-T significant",
          bool((not vp.loc["Causal Forest - X-Learner", "ci_bonf3_excludes_0"]) and vp.loc["Causal Forest - T-Learner", "ci_bonf3_excludes_0"]))
    check("claim: test T-Learner predictions 83.6% exactly zero", abs(meta["predictions"]["T-Learner"]["share_zero"] - 0.836) < 0.0005)
    check("claim: top 10% incremental conversions are 5-6x random",
          all(5 <= fm.loc[m_, "top10_incremental_conversions"] / rr.top10_incremental_conversions <= 6.05 for m_ in MODELS))


def figures() -> dict:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    plan = [("fig1_sim_sample_size_rmse", F / "final_step4" / "sample_size_rmse"),
            ("fig2_sim_treatment_ratio_rmse", F / "final_step4" / "treatment_ratio_rmse"),
            ("fig3_sim_effect_structure_rmse", F / "final_step4" / "effect_complexity_rmse"),
            ("fig4_criteo_test_qini_curve", F / "criteo_final_test" / "qini_curve"),
            ("fig5_criteo_test_qini_coefficient_ci", F / "criteo_final_test" / "qini_coefficient_ci"),
            ("fig6_criteo_test_targeting", F / "criteo_final_test" / "targeting_incremental_conversions")]
    copied = {}
    for new, src in plan:
        for ext in ("png", "pdf"):
            s, d = src.with_suffix(f".{ext}"), OUT_FIG / f"{new}.{ext}"
            if d.exists() and md5(d) != md5(s):
                raise SystemExit(f"{d} exists and differs from its source; refusing to overwrite")
            shutil.copy2(s, d)
            check(f"figure copy {d.name} identical to {s.relative_to(PROJECT_ROOT)}", md5(s) == md5(d))
            copied[d.name] = str(s.relative_to(PROJECT_ROOT))
    return copied


def main() -> None:
    simulation()
    criteo()
    copied = figures()
    out = pd.DataFrame(checks)
    out.to_csv(T / "step6_verification_checks.csv", index=False)
    (OUT_FIG / "SOURCES.json").write_text(json.dumps(copied, indent=2) + "\n", encoding="utf-8")
    print(out.to_string(index=False))
    print(f"{out.ok.sum()}/{len(out)} checks passed")
    if not out.ok.all():
        sys.exit(1)


if __name__ == "__main__":
    main()
