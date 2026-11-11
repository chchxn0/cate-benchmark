"""STEP 5-5: paired stratified bootstrap of the saved validation predictions (no training, no test data).

1. Verification on the real validation set (stops on any mismatch):
   a. evaluate_uplift with all-one weights == unweighted evaluate_uplift (same sorting / tie blocks)
   b. one stratified bootstrap draw applied as integer weights == evaluating explicitly duplicated rows
   c. paired_bootstrap == a manual loop of stratified_bootstrap_weights + evaluate_uplift(weights=...)
2. B replicates with src/criteo_evaluation.paired_bootstrap: treated and control rows are resampled
   separately (group sizes fixed) and every score vector is evaluated on the same draw.
   Score vectors: the three 300k models (primary), the three 100k models (supplementary, for the
   300k - 100k change) and an all-equal score (random-targeting reference).

    caffeinate -i .venv/bin/python scripts/criteo_bootstrap.py --n-boot 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.criteo_data import OUTCOME, TREATMENT, load_split  # noqa: E402
from src.criteo_evaluation import (  # noqa: E402
    evaluate_uplift,
    paired_bootstrap,
    stratified_bootstrap_weights,
)

TABLES = PROJECT_ROOT / "results" / "tables"
FIGS = PROJECT_ROOT / "results" / "figures" / "criteo_300k_bootstrap"
PRED_ROOT = PROJECT_ROOT / "data" / "criteo" / "predictions"
DRAWS_DIR = PROJECT_ROOT / "data" / "criteo" / "bootstrap"
MODELS = {"T": "T-Learner", "X": "X-Learner", "CF": "Causal Forest"}
SOURCES = {"300k": "300k", "100k": "pilot_100k"}
RANDOM = "Random"
PAIRS = [("CF", "X"), ("CF", "T"), ("X", "T")]
METRICS = ["auuc", "auuc_minus_random", "auuc_normalized", "qini_coefficient", "qini_normalized",
           "total_uplift", "n_undefined_points",
           *[f"top{q}_{m}" for q in (10, 20, 30) for m in ("rate_diff", "incremental_conversions", "n_control", "conv_control")]]
PAIR_METRICS = ["qini_coefficient", "auuc", "auuc_normalized", "qini_normalized",
                *[f"top{q}_{m}" for q in (10, 20, 30) for m in ("incremental_conversions", "rate_diff")]]
COMPARE_KEYS = ["auuc", "auuc_normalized", "qini_coefficient", "qini_normalized", "total_uplift",
                "top10_rate_diff", "top20_incremental_conversions", "top30_incremental_conversions"]


def same(a: dict, b: dict, keys, rtol=1e-9) -> bool:
    return all(np.isclose(a[k], b[k], rtol=rtol, atol=1e-9, equal_nan=True) for k in keys)


def verify(scores: dict, T, Y, seed: int) -> dict:
    out = {}
    keys = COMPARE_KEYS
    for name in ("300k:CF", "300k:T"):          # T has the largest tie block (exact zeros)
        s = scores[name]
        base = evaluate_uplift(s, T, Y)
        ones = evaluate_uplift(s, T, Y, weights=np.ones(len(s), dtype=np.int32))
        out[f"{name} all-one weights == unweighted"] = same(base, ones, keys)
        w = stratified_bootstrap_weights(T, np.random.default_rng(seed + 999))
        weighted = evaluate_uplift(s, T, Y, weights=w)
        dup = evaluate_uplift(np.repeat(s, w), np.repeat(T, w), np.repeat(Y, w))
        out[f"{name} weighted draw == duplicated rows"] = same(weighted, dup, keys, rtol=1e-7)
        out[f"{name} draw keeps group sizes"] = bool(w[T == 1].sum() == (T == 1).sum() and w[T == 0].sum() == (T == 0).sum())
    sub = {k: scores[k] for k in ("300k:CF", "300k:X")}
    pb = paired_bootstrap(sub, T, Y, n_boot=2, seed=seed + 7, metrics=keys)
    rng = np.random.default_rng(seed + 7)
    manual_ok = True
    for b in range(2):
        w = stratified_bootstrap_weights(T, rng)
        for k, s in sub.items():
            r = evaluate_uplift(s, T, Y, weights=w)
            manual_ok &= all(np.isclose(pb[f"{k}:{m}"][b], r[m], rtol=1e-12, equal_nan=True) for m in keys)
    out["paired_bootstrap == manual loop (B=2)"] = bool(manual_ok)
    return out


def pct_ci(x: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.nanquantile(x, a)), float(np.nanquantile(x, 1 - a))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t_start = time.perf_counter()
    for p in (TABLES / "criteo_300k_bootstrap_summary.csv", TABLES / "criteo_300k_paired_comparison.csv",
              TABLES / "criteo_300k_targeting_ci.csv"):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")

    valid = load_split("valid", columns=[TREATMENT, OUTCOME])
    ids, T, Y = valid["row_id"].to_numpy(), valid[TREATMENT].to_numpy(), valid[OUTCOME].to_numpy()
    scores = {}
    for size, tag in SOURCES.items():
        for key in MODELS:
            z = np.load(PRED_ROOT / tag / f"{key}_valid.npz")
            if not np.array_equal(z["row_id"], ids):
                raise SystemExit(f"{tag}/{key}: row_id mismatch")
            scores[f"{size}:{key}"] = z["score"]
    scores[RANDOM] = np.zeros(len(ids))

    t0 = time.perf_counter()
    checks = verify(scores, T, Y, args.seed)
    t_verify = time.perf_counter() - t0
    print(json.dumps(checks, indent=1), f"\nverification {t_verify:.1f}s", flush=True)
    if not all(checks.values()):
        raise SystemExit("bootstrap verification failed; not running the bootstrap")

    point = {k: evaluate_uplift(s, T, Y) for k, s in scores.items()}
    t0 = time.perf_counter()
    draws = paired_bootstrap(scores, T, Y, n_boot=args.n_boot, seed=args.seed, metrics=METRICS)
    t_boot = time.perf_counter() - t0
    print(f"bootstrap B={args.n_boot}: {t_boot:.0f}s", flush=True)
    DRAWS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(draws).to_csv(DRAWS_DIR / f"validation_draws_B{args.n_boot}_seed{args.seed}.csv.gz", index=False)

    def label(k):
        return (RANDOM, "-") if k == RANDOM else (MODELS[k.split(":")[1]], k.split(":")[0])

    # 1. per-model summary
    rows = []
    for k in scores:
        model, train = label(k)
        for m in METRICS:
            d = draws[f"{k}:{m}"]
            lo, hi = pct_ci(d, 0.95)
            rows.append({"model": model, "train": train, "metric": m, "point": point[k][m],
                         "boot_mean": float(np.nanmean(d)), "boot_se": float(np.nanstd(d, ddof=1)),
                         "ci95_low": lo, "ci95_high": hi, "n_nan_draws": int(np.isnan(d).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "criteo_300k_bootstrap_summary.csv", index=False, float_format="%.6g")

    # 2. paired differences (300k models; Bonferroni over the 3 pairs within each metric)
    level_adj = 1 - 0.05 / len(PAIRS)
    prow = []
    for a, b in PAIRS:
        for m in PAIR_METRICS:
            d = draws[f"300k:{a}:{m}"] - draws[f"300k:{b}:{m}"]
            lo, hi = pct_ci(d, 0.95)
            blo, bhi = pct_ci(d, level_adj)
            p_le0 = float(np.mean(d <= 0))
            p_two = min(1.0, 2 * min(p_le0, float(np.mean(d >= 0))))
            prow.append({"comparison": f"{MODELS[a]} - {MODELS[b]}", "train": "300k", "metric": m,
                         "point_diff": point[f"300k:{a}"][m] - point[f"300k:{b}"][m],
                         "boot_se": float(np.nanstd(d, ddof=1)), "ci95_low": lo, "ci95_high": hi,
                         "ci_bonf3_low": blo, "ci_bonf3_high": bhi, "share_draws_diff_le_0": p_le0,
                         "boot_p_two_sided": p_two, "p_bonf3": min(1.0, p_two * len(PAIRS)),
                         "ci95_excludes_0": bool(lo > 0 or hi < 0), "ci_bonf3_excludes_0": bool(blo > 0 or bhi < 0),
                         "n_nan_draws": int(np.isnan(d).sum())})
    for key in MODELS:   # supplementary: same model, 300k - 100k (validation-sampling variation only)
        for m in ("qini_coefficient", "auuc", "auuc_normalized", "top10_incremental_conversions"):
            d = draws[f"300k:{key}:{m}"] - draws[f"100k:{key}:{m}"]
            lo, hi = pct_ci(d, 0.95)
            p_two = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))
            prow.append({"comparison": f"{MODELS[key]} 300k - 100k", "train": "300k vs 100k", "metric": m,
                         "point_diff": point[f"300k:{key}"][m] - point[f"100k:{key}"][m],
                         "boot_se": float(np.nanstd(d, ddof=1)), "ci95_low": lo, "ci95_high": hi,
                         "share_draws_diff_le_0": float(np.mean(d <= 0)), "boot_p_two_sided": p_two,
                         "ci95_excludes_0": bool(lo > 0 or hi < 0), "n_nan_draws": int(np.isnan(d).sum())})
    paired = pd.DataFrame(prow)
    paired.to_csv(TABLES / "criteo_300k_paired_comparison.csv", index=False, float_format="%.6g")

    # 3. targeting table (300k models + random reference)
    trow = []
    for k in [f"300k:{m}" for m in MODELS] + [RANDOM]:
        model, train = label(k)
        for q in (10, 20, 30):
            r = {"model": model, "train": train, "top_pct": q, "k": point[k][f"top{q}_k"],
                 "n_treated": point[k][f"top{q}_n_treated"], "n_control": point[k][f"top{q}_n_control"],
                 "conv_treated": point[k][f"top{q}_conv_treated"], "conv_control": point[k][f"top{q}_conv_control"]}
            for m in ("rate_diff", "incremental_conversions"):
                d = draws[f"{k}:top{q}_{m}"]
                lo, hi = pct_ci(d, 0.95)
                r.update({m: point[k][f"top{q}_{m}"], f"{m}_se": float(np.nanstd(d, ddof=1)),
                          f"{m}_ci95_low": lo, f"{m}_ci95_high": hi})
            r["boot_conv_control_min"] = float(np.min(draws[f"{k}:top{q}_conv_control"]))
            trow.append(r)
    targeting = pd.DataFrame(trow)
    targeting.to_csv(TABLES / "criteo_300k_targeting_ci.csv", index=False, float_format="%.6g")

    nan_counts = {k: int(np.isnan(v).sum()) for k, v in draws.items() if np.isnan(v).any()}
    undefined = {k: int(np.max(v)) for k, v in draws.items() if k.endswith("n_undefined_points")}
    meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "step": "STEP 5-5 validation bootstrap (saved predictions; no training; test split never loaded)",
            "n_boot": args.n_boot, "seed": args.seed, "n_valid": len(ids),
            "resampling": "stratified by treatment, with replacement, group sizes fixed; identical draws for all score vectors",
            "evaluation": "src/criteo_evaluation.py (STEP 5-2 definitions: ratio estimator, 101-point grid, tie-block interpolation)",
            "ci": "percentile; paired differences also with Bonferroni over 3 model pairs per metric (98.33% percentile)",
            "score_vectors": list(scores), "verification": checks,
            "nan_draws": nan_counts, "max_undefined_grid_points_per_draw": undefined,
            "draws_file": str((DRAWS_DIR / f"validation_draws_B{args.n_boot}_seed{args.seed}.csv.gz").relative_to(PROJECT_ROOT)),
            "timing_sec": {"verification": t_verify, "bootstrap": t_boot, "total": time.perf_counter() - t_start}}
    (TABLES / "criteo_300k_bootstrap_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                                                encoding="utf-8")
    make_figures(summary, paired, draws)
    print(f"done in {time.perf_counter() - t_start:.0f}s")


def make_figures(summary: pd.DataFrame, paired: pd.DataFrame, draws: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    FIGS.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next(f for f in ["AppleSDGothicNeo", "Apple SD Gothic Neo", "AppleGothic"]
                                             if f in available), "axes.unicode_minus": False, "pdf.fonttype": 42})
    colors = {"T-Learner": "#2a78d6", "X-Learner": "#eb6834", "Causal Forest": "#1baf7a"}
    muted, grid = "#5f5e58", "#e4e3df"

    def finish(fig, ax, name, title, sub):
        ax.set_title(title, loc="left", fontsize=12, pad=20)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=muted)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"{name}.{ext}", dpi=200, facecolor="white")
        plt.close(fig)

    # model CIs (300k and 100k) for Qini coefficient and normalized AUUC
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, lab in ((axes[0], "qini_coefficient", "Qini coefficient (전환 건수)"),
                            (axes[1], "auuc_normalized", "정규화 AUUC")):
        sub = summary[(summary.metric == metric) & (summary.model != RANDOM)]
        ylabels = []
        for i, r in enumerate(sub.sort_values(["train", "model"], ascending=[False, True]).itertuples()):
            filled = r.train == "300k"
            ax.plot([r.ci95_low, r.ci95_high], [i, i], color=colors[r.model], lw=2, zorder=1)
            ax.plot(r.point, i, "o" if filled else "s", color=colors[r.model], mfc=colors[r.model] if filled else "white",
                    ms=7, zorder=2)
            ylabels.append(f"{r.model} · {r.train}")
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="#b0afa8", lw=0.8)
        ax.set_xlabel(lab, color=muted)
        ax.grid(axis="x", color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("검증 세트 성능과 부트스트랩 95% 신뢰구간 (B=1,000, 처치·대조 층화)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.9, "학습 변동(seed) 미반영 · 검증 세트는 모델 선택용 개발 데이터", fontsize=8.5, color=muted)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"model_ci.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    # paired differences (300k)
    p = paired[(paired.train == "300k") & paired.metric.isin(["qini_coefficient", "auuc", "top10_incremental_conversions",
                                                              "top20_incremental_conversions", "top30_incremental_conversions"])]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = []
    for i, r in enumerate(p.itertuples()):
        ax.plot([r.ci_bonf3_low, r.ci_bonf3_high], [i, i], color="#b0afa8", lw=6, solid_capstyle="butt", zorder=1)
        ax.plot([r.ci95_low, r.ci95_high], [i, i], color="#1f1f1e", lw=1.6, zorder=2)
        ax.plot(r.point_diff, i, "o", color="#1f1f1e", ms=6, zorder=3)
        labels.append(f"{r.comparison} · {r.metric}")
    ax.axvline(0, color="#c0392b", lw=1, ls="--")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("짝비교 차이 (전환 건수)", color=muted)
    ax.grid(axis="x", color=grid, lw=0.8)
    finish(fig, ax, "paired_differences_300k", "300k 모델 짝비교: 차이와 신뢰구간",
           "검은 막대: 95% 백분위 구간 · 회색 띠: Bonferroni(3쌍) 98.3% 구간 · 같은 부트스트랩 표본")

    # bootstrap distributions of the Qini coefficient
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, name in MODELS.items():
        ax.hist(draws[f"300k:{key}:qini_coefficient"], bins=50, alpha=0.55, color=colors[name], label=name)
    ax.set_xlabel("Qini coefficient (부트스트랩 반복값)", color=muted)
    ax.set_ylabel("빈도", color=muted)
    ax.legend(frameon=False, fontsize=9)
    finish(fig, ax, "qini_bootstrap_distribution_300k", "300k 모델 Qini coefficient 부트스트랩 분포",
           "B=1,000 · 처치·대조 층화 복원추출 · 세 모델 동일 표본")


if __name__ == "__main__":
    main()
