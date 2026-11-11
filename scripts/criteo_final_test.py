"""STEP 5-6: one-time final test evaluation of the three fixed 300k Criteo models (no training, no tuning).

Steps
1. Verify SHA-256 of the saved models, config and split files against the values fixed in STEP 5-2 / 5-5,
   and that the evaluation split shares no row with train_300000 / valid / train_pool.
2. Predict the evaluation split in batches with the saved models (one model in memory at a time);
   check row order against the split index and a separately predicted fixed subset.
3. Evaluate with src/criteo_evaluation.py unchanged (STEP 5-2 definitions) and run a B-replicate
   stratified paired bootstrap (same draws for all models and the random reference).
4. Write tables, metadata and figures.

    # dry run on the validation split (outputs to a scratch directory; must reproduce STEP 5-5 numbers)
    .venv/bin/python scripts/criteo_final_test.py --split valid --out-dir <scratch>
    # the single final test run
    caffeinate -i .venv/bin/python scripts/criteo_final_test.py --split test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.criteo_analysis import tie_summary  # noqa: E402
from src.criteo_data import FEATURES, OUTCOME, SPLIT_DIR, TREATMENT, load_split  # noqa: E402
from src.criteo_evaluation import evaluate_uplift, paired_bootstrap  # noqa: E402
from src.criteo_models import fixed_subset_positions, predict_in_batches  # noqa: E402

MODELS = {"T": "T-Learner", "X": "X-Learner", "CF": "Causal Forest"}
RANDOM = "Random"
PAIRS = [("CF", "T"), ("CF", "X"), ("X", "T")]
EXPECTED_SHA256 = {   # fixed in docs/criteo_step5_5_report.md (models, config) and criteo_split_metadata.json (splits)
    "data/criteo/models/300k/T.joblib": "47c114a5a062024f1dca5aca4b863452197280ba0e120be850f6bc83d66e8305",
    "data/criteo/models/300k/X.joblib": "34c27a4a6e0bf52e975f9818b580f00a54ca15d41a36a453756625117962be4b",
    "data/criteo/models/300k/CF.joblib": "73bf1f9d9a2cbd97c8f43e4d7880a536be3b455523ee2da1689c90b16e93f150",
    "configs/criteo_models.yaml": "2b221b7e449dbc1fac101984552c10e9a64449a1cc6d75bb8b4f852c7b926479",
    "data/criteo/splits/train_300000.npy": "0c80d8c8c343aa016542ad7e26e95fc4e8f6ff7c27c6b990eed7d2631308c584",
    "data/criteo/splits/test.npy": "46af8d990616e757a70ffabf5ea5ae26149b3d6b0d71ececcd116bb2ef4902e9",
    "data/criteo/splits/valid.npy": "e703db7dbd21e7c0067256ad2656079a183562516fc6d585bc2b7593e15baba0",
    "data/criteo/splits/train_pool.npy": "56570fa165cece6409342fd2861f2e6ac55e7354e95105618993c1ed71816380",
}
BATCH = 200_000
SUBSET = 50_000
METRICS = ["auuc", "auuc_minus_random", "auuc_normalized", "qini_coefficient", "qini_normalized", "total_uplift",
           "n_undefined_points",
           *[f"top{q}_{m}" for q in (10, 20, 30) for m in ("rate_diff", "incremental_conversions", "n_control", "conv_control")]]
PAIR_METRICS = ["qini_coefficient", "auuc", "auuc_normalized",
                *[f"top{q}_{m}" for q in (10, 20, 30) for m in ("incremental_conversions", "rate_diff")]]
CRITERION = ("pre-specified (STEP 5-5): a model beats random targeting if the 95% percentile CI of its Qini "
             "coefficient (area above the random line, random = 0) lies above 0; model differences use the "
             "Bonferroni-adjusted (3 pairs) 98.33% CI of the paired Qini difference")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def pct_ci(x, level):
    a = (1 - level) / 2
    return float(np.nanquantile(x, a)), float(np.nanquantile(x, 1 - a))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["test", "valid"], required=True)
    ap.add_argument("--out-dir", type=Path, help="table/figure root (default: results/); required for --split valid")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw figures from the saved predictions and tables (no re-evaluation is reported)")
    args = ap.parse_args()
    if args.split == "valid" and args.out_dir is None:
        raise SystemExit("--split valid is a dry run and needs --out-dir (outside results/)")
    out_root = args.out_dir or PROJECT_ROOT / "results"
    tables, figs = out_root / "tables", out_root / "figures" / "criteo_final_test"
    stem = "criteo_final_test" if args.split == "test" else "criteo_dryrun_valid"
    pred_dir = PROJECT_ROOT / "data" / "criteo" / "predictions" / ("final_test" if args.split == "test" else "dryrun_valid")
    if args.figures_only:
        redraw(args.split, tables, figs, stem, pred_dir)
        return
    outputs = [tables / f"{stem}_{s}" for s in ("metrics.csv", "bootstrap.csv", "paired.csv", "targeting.csv", "metadata.json")]
    if any(p.exists() for p in outputs) or (pred_dir / "T_scores.npz").exists():
        raise SystemExit("final outputs already exist; refusing to rerun (the test set is evaluated once)")
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    timing, t_all = {}, time.perf_counter()

    # 1. integrity
    hashes = {p: sha256(PROJECT_ROOT / p) for p in EXPECTED_SHA256}
    hash_ok = {p: hashes[p] == EXPECTED_SHA256[p] for p in EXPECTED_SHA256}
    if not all(hash_ok.values()):
        raise SystemExit(f"SHA-256 mismatch: {[p for p, ok in hash_ok.items() if not ok]}")
    idx = {n: np.load(SPLIT_DIR / f"{n}.npy") for n in ("test", "valid", "train_pool", "train_300000")}
    other = [n for n in ("test", "valid", "train_pool", "train_300000") if n != args.split]
    overlap = {f"{args.split}&{n}": int(len(np.intersect1d(idx[args.split], idx[n], assume_unique=True)))
               for n in other if not (args.split == "valid" and n == "train_pool") or True}
    if args.split == "test" and any(overlap.values()):
        raise SystemExit(f"index overlap: {overlap}")
    print(f"[final] hashes OK ({len(hashes)} files); overlaps {overlap}", flush=True)

    # 2. predictions
    t0 = time.perf_counter()
    data = load_split(args.split)
    timing["load_split_sec"] = time.perf_counter() - t0
    ids = data["row_id"].to_numpy()
    if not np.array_equal(ids, idx[args.split]):
        raise SystemExit("loaded row_id order differs from the split index")
    X, T, Y = data[FEATURES].to_numpy(), data[TREATMENT].to_numpy(), data[OUTCOME].to_numpy()
    del data
    sub = fixed_subset_positions(len(X), SUBSET, args.seed)
    scores, pred_info = {}, {}
    for key, name in MODELS.items():
        t0 = time.perf_counter()
        model = joblib.load(PROJECT_ROOT / f"data/criteo/models/300k/{key}.joblib")
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        s = predict_in_batches(model, X, BATCH)
        t_pred = time.perf_counter() - t0
        s_sub = predict_in_batches(model, X[sub], BATCH)
        del model
        if not np.isfinite(s).all():
            raise SystemExit(f"{key}: non-finite predictions")
        if not np.allclose(s[sub], s_sub, rtol=0, atol=1e-12):
            raise SystemExit(f"{key}: subset re-prediction differs from the full batch prediction")
        np.savez(pred_dir / f"{key}_scores.npz", row_id=ids.astype(np.int32), score=s)
        z = np.load(pred_dir / f"{key}_scores.npz")
        if not (np.array_equal(z["row_id"], ids) and np.array_equal(z["score"], s)):
            raise SystemExit(f"{key}: saved predictions do not round-trip")
        scores[key] = s
        pred_info[name] = {"load_model_sec": t_load, "predict_sec": t_pred, "n": len(s), "n_nan": 0, "n_inf": 0,
                           "mean": float(s.mean()), "sd": float(s.std()), "min": float(s.min()), "max": float(s.max()),
                           "share_negative": float((s < 0).mean()), **tie_summary(s),
                           "peak_rss_mb_so_far": peak_rss_mb()}
        print(f"[final] {name}: predicted {len(s):,} rows in {t_pred:.1f}s (load {t_load:.1f}s), "
              f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
    del X
    scores[RANDOM] = np.zeros(len(T))

    # 3. evaluation and bootstrap
    t0 = time.perf_counter()
    point = {k: evaluate_uplift(s, T, Y) for k, s in scores.items()}
    timing["point_eval_sec"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    draws = paired_bootstrap(scores, T, Y, n_boot=args.n_boot, seed=args.seed, metrics=METRICS)
    timing["bootstrap_sec"] = time.perf_counter() - t0
    print(f"[final] bootstrap B={args.n_boot}: {timing['bootstrap_sec']:.0f}s", flush=True)
    pd.DataFrame(draws).to_csv(pred_dir / f"bootstrap_draws_B{args.n_boot}_seed{args.seed}.csv.gz", index=False)
    lbl = {**MODELS, RANDOM: "Random (all-equal score)"}

    metric_rows = []
    for k, r in point.items():
        metric_rows.append({"model": lbl[k], "eval_split": args.split, **{m: v for m, v in r.items() if m != "curve"}})
    pd.DataFrame(metric_rows).to_csv(tables / f"{stem}_metrics.csv", index=False, float_format="%.6g")

    brow = []
    for k in scores:
        for m in METRICS:
            d = draws[f"{k}:{m}"]
            lo, hi = pct_ci(d, 0.95)
            brow.append({"model": lbl[k], "metric": m, "point": point[k][m], "boot_mean": float(np.nanmean(d)),
                         "boot_se": float(np.nanstd(d, ddof=1)), "ci95_low": lo, "ci95_high": hi,
                         "n_nan_draws": int(np.isnan(d).sum()),
                         "beats_random_by_criterion": bool(lo > 0) if m == "qini_coefficient" and k != RANDOM else None})
    boot = pd.DataFrame(brow)
    boot.to_csv(tables / f"{stem}_bootstrap.csv", index=False, float_format="%.6g")

    prow = []
    for a, b in PAIRS:
        for m in PAIR_METRICS:
            d = draws[f"{a}:{m}"] - draws[f"{b}:{m}"]
            lo, hi = pct_ci(d, 0.95)
            blo, bhi = pct_ci(d, 1 - 0.05 / len(PAIRS))
            p_two = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))
            prow.append({"comparison": f"{MODELS[a]} - {MODELS[b]}", "metric": m,
                         "role": "primary" if m == "qini_coefficient" else "secondary",
                         "point_diff": point[a][m] - point[b][m], "boot_se": float(np.nanstd(d, ddof=1)),
                         "ci95_low": lo, "ci95_high": hi, "ci_bonf3_low": blo, "ci_bonf3_high": bhi,
                         "boot_p_two_sided": p_two, "p_bonf3": min(1.0, 3 * p_two),
                         "ci95_excludes_0": bool(lo > 0 or hi < 0), "ci_bonf3_excludes_0": bool(blo > 0 or bhi < 0),
                         "n_nan_draws": int(np.isnan(d).sum())})
    paired = pd.DataFrame(prow)
    paired.to_csv(tables / f"{stem}_paired.csv", index=False, float_format="%.6g")

    trow = []
    for k in [*MODELS, RANDOM]:
        for q in (10, 20, 30):
            r = {"model": lbl[k], "top_pct": q, **{f: point[k][f"top{q}_{f}"] for f in
                                                   ("k", "n_treated", "n_control", "conv_treated", "conv_control",
                                                    "rate_treated", "rate_control")}}
            for m in ("rate_diff", "incremental_conversions"):
                d = draws[f"{k}:top{q}_{m}"]
                lo, hi = pct_ci(d, 0.95)
                r.update({m: point[k][f"top{q}_{m}"], f"{m}_se": float(np.nanstd(d, ddof=1)),
                          f"{m}_ci95_low": lo, f"{m}_ci95_high": hi})
            r["boot_conv_control_min"] = float(np.min(draws[f"{k}:top{q}_conv_control"]))
            trow.append(r)
    targeting = pd.DataFrame(trow)
    targeting.to_csv(tables / f"{stem}_targeting.csv", index=False, float_format="%.6g")

    timing["total_sec"] = time.perf_counter() - t_all
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": "STEP 5-6 final test evaluation" if args.split == "test" else "STEP 5-6 dry run on validation (not a result)",
        "eval_split": args.split, "n": int(len(T)), "n_treated": int(T.sum()), "n_control": int((1 - T).sum()),
        "conv_treated": int((T * Y).sum()), "conv_control": int(((1 - T) * Y).sum()),
        "models": {MODELS[k]: f"data/criteo/models/300k/{k}.joblib" for k in MODELS},
        "training": "train_300000, configs/criteo_models.yaml, seed 42 (fitted in STEP 5-4; not refitted)",
        "sha256": hashes, "sha256_verified": hash_ok, "index_overlap": overlap,
        "row_order_checks": "row_id == split index; subset re-prediction == full prediction; saved npz round-trip",
        "evaluation": {"code": "src/criteo_evaluation.py (unchanged since STEP 5-2)", "estimator": "ratio",
                       "grid": "phi in {0, 0.01, ..., 1}, trapezoid rule", "ties": "tie-block linear interpolation",
                       "units": "AUUC, Qini area/coefficient and incremental conversions in conversions; normalized "
                                "AUUC = (AUUC - U(N)/2)/U(N), normalized Qini = (Qini area - Q(N)/2)/Q(N)"},
        "bootstrap": {"B": args.n_boot, "seed": args.seed, "scheme": "treated/control resampled separately with "
                      "replacement (sizes fixed); identical draws for all models and the random reference",
                      "ci": "percentile 95%; paired differences also Bonferroni over 3 pairs (98.33%)",
                      "scope": "test-sample variation of fixed fitted models; training-seed variation not included"},
        "decision_rule": CRITERION,
        "nan_draws": {k: int(np.isnan(v).sum()) for k, v in draws.items() if np.isnan(v).any()},
        "max_undefined_grid_points": {k: int(np.max(v)) for k, v in draws.items() if k.endswith("n_undefined_points")},
        "predictions": pred_info, "prediction_files": str(pred_dir.relative_to(PROJECT_ROOT)),
        "timing_sec": timing, "peak_rss_mb": peak_rss_mb(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "econml", "joblib"]}},
    }
    (tables / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float) + "\n",
                                                  encoding="utf-8")
    make_figures(point, boot, paired, targeting, figs, args.split)
    print(f"[final] done in {timing['total_sec']:.0f}s, peak RSS {peak_rss_mb():.0f} MB", flush=True)


def redraw(split: str, tables: Path, figs: Path, stem: str, pred_dir: Path) -> None:
    """Rebuild figures only; the curves are recomputed from the saved scores and must match the saved metrics."""
    d = load_split(split, columns=[TREATMENT, OUTCOME])
    ids, T, Y = d["row_id"].to_numpy(), d[TREATMENT].to_numpy(), d[OUTCOME].to_numpy()
    saved = pd.read_csv(tables / f"{stem}_metrics.csv").set_index("model")
    point = {}
    for key, name in MODELS.items():
        z = np.load(pred_dir / f"{key}_scores.npz")
        if not np.array_equal(z["row_id"], ids):
            raise SystemExit(f"{key}: row_id mismatch")
        point[key] = evaluate_uplift(z["score"], T, Y)
        for m in ("auuc", "qini_coefficient", "top10_incremental_conversions"):
            if not np.isclose(point[key][m], saved.loc[name, m], rtol=1e-5):
                raise SystemExit(f"{key}:{m} recomputed value differs from the saved metrics")
    point[RANDOM] = evaluate_uplift(np.zeros(len(T)), T, Y)
    make_figures(point, pd.read_csv(tables / f"{stem}_bootstrap.csv"), pd.read_csv(tables / f"{stem}_paired.csv"),
                 pd.read_csv(tables / f"{stem}_targeting.csv"), figs, split)
    print("[final] figures redrawn; recomputed metrics match the saved results")


def make_figures(point, boot, paired, targeting, figs: Path, split: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next(f for f in ["AppleSDGothicNeo", "Apple SD Gothic Neo", "AppleGothic"]
                                             if f in available), "axes.unicode_minus": False, "pdf.fonttype": 42})
    colors = {"T-Learner": "#2a78d6", "X-Learner": "#eb6834", "Causal Forest": "#1baf7a"}
    styles = {"T-Learner": "-", "X-Learner": "--", "Causal Forest": "-."}
    ink, muted, grid = "#1f1f1e", "#5f5e58", "#e4e3df"
    name = "최종 테스트" if split == "test" else "검증(드라이런)"
    n = int(point[RANDOM]["n"])

    def style(ax, title, sub, xlabel, ylabel):
        ax.set_title(title, loc="left", fontsize=12.5, color=ink, pad=22)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=muted)
        ax.set_xlabel(xlabel, color=muted)
        ax.set_ylabel(ylabel, color=muted)
        ax.grid(color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted)

    def save(fig, stem):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figs / f"{stem}.{ext}", dpi=300, facecolor="white")
        plt.close(fig)

    for kind, ylabel, title in (("qini", "Qini Q(k) (처치군 규모 환산 증분 전환, 건)", "Qini curve"),
                                ("uplift", "Uplift U(k) (추정 증분 전환, 건)", "Uplift curve")):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for k, m in MODELS.items():
            c = point[k]["curve"]
            ax.plot(c["phi"] * 100, c[kind], color=colors[m], ls=styles[m], lw=2, label=m)
        end = point[RANDOM]["total_qini" if kind == "qini" else "total_uplift"]
        ax.plot([0, 100], [0, end], color="#6b6a64", ls=":", lw=1.5, label="무작위 타기팅")
        ax.legend(frameon=False, fontsize=9.5, loc="lower right")
        style(ax, f"{title} — {name} ({n:,}명)",
              "300k 학습 모델 · 예측 uplift 내림차순 · 101점 격자 · 동점은 블록 보간",
              "타기팅 비율 (예측 uplift 상위 %)", ylabel)
        save(fig, f"{kind}_curve")

    q = boot[(boot.metric == "qini_coefficient") & (boot.model != "Random (all-equal score)")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, r in q.iterrows():
        ax.plot([r.ci95_low, r.ci95_high], [i, i], color=colors[r.model], lw=3, solid_capstyle="round")
        ax.plot(r.point, i, "o", color=colors[r.model], ms=9, mec="white", mew=1.5)
        ax.annotate(f"{r.point:.0f}  [{r.ci95_low:.0f}, {r.ci95_high:.0f}]", (r.ci95_high, i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=ink)
    ax.axvline(0, color="#6b6a64", ls=":", lw=1.2)
    ax.set_yticks(range(len(q)), q.model)
    ax.set_ylim(len(q) - 0.4, -0.6)
    ax.text(0, len(q) - 0.45, " 무작위 = 0", fontsize=8.5, color=muted, va="bottom")
    ax.set_xlim(min(0, q.ci95_low.min()) - 50, q.ci95_high.max() * 1.35)
    style(ax, f"모델별 Qini coefficient와 95% 신뢰구간 — {name}",
          "층화 짝 부트스트랩 B=1,000 (처치·대조 별도 복원추출) · 학습 seed 변동 미반영", "Qini coefficient (전환 건수)", "")
    ax.grid(axis="y", visible=False)
    save(fig, "qini_coefficient_ci")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    order = list(MODELS.values()) + ["Random (all-equal score)"]
    cols = {**colors, "Random (all-equal score)": "#b0afa8"}
    width = 0.2
    for j, m in enumerate(order):
        t = targeting[targeting.model == m].sort_values("top_pct")
        x = np.arange(3) + (j - 1.5) * width
        ax.bar(x, t.incremental_conversions, width * 0.92, color=cols[m], label="무작위 타기팅" if j == 3 else m, zorder=3,
               yerr=[t.incremental_conversions - t.incremental_conversions_ci95_low,
                     t.incremental_conversions_ci95_high - t.incremental_conversions],
               capsize=3, error_kw=dict(ecolor=ink, elinewidth=1))
    ax.set_xticks(range(3), ["상위 10%", "상위 20%", "상위 30%"])
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    ax.set_ylim(0, targeting.incremental_conversions_ci95_high.max() * 1.18)
    style(ax, f"타기팅 비율별 추정 증분 전환 건수 — {name}",
          "증분 전환 = (처치군 전환율 - 대조군 전환율) × 타기팅 인원 · 오차막대: 부트스트랩 95% CI", "", "추정 증분 전환 (건)")
    ax.grid(axis="x", visible=False)
    save(fig, "targeting_incremental_conversions")


if __name__ == "__main__":
    main()
