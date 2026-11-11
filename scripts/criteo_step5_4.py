"""STEP 5-4: train selected models on a larger training split and compare with the 100k pilot.

Reuses scripts/criteo_pilot.py unchanged (`fit` per model, `evaluate` per tag) so the model settings,
seed and evaluation code are identical to STEP 5-3. The final test split is never loaded.

    # models one at a time, each under memory_guard; stops on failure / guard kill / swap growth
    caffeinate -i .venv/bin/python scripts/criteo_step5_4.py run --models T X --train train_300000 --tag 300k
    caffeinate -i .venv/bin/python scripts/criteo_step5_4.py run --models CF --train train_300000 --tag 300k

    # 100k vs 300k on the same validation rows (no training)
    .venv/bin/python scripts/criteo_step5_4.py compare --base pilot_100k --new 300k
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.criteo_analysis import segment_effect, tie_summary  # noqa: E402
from src.criteo_data import OUTCOME, TREATMENT, load_split  # noqa: E402
from src.criteo_evaluation import evaluate_uplift  # noqa: E402
from src.criteo_models import MODEL_KEYS  # noqa: E402

PILOT = PROJECT_ROOT / "scripts" / "criteo_pilot.py"
GUARD = PROJECT_ROOT / "scripts" / "memory_guard.py"
TABLES = PROJECT_ROOT / "results" / "tables"
PRED_ROOT = PROJECT_ROOT / "data" / "criteo" / "predictions"
MEMORY_LIMIT_MB = 7000
SWAP_JUMP_MB = 1024
BOTTOM = 0.03
COLORS = {"T-Learner": "#2a78d6", "X-Learner": "#eb6834", "Causal Forest": "#1baf7a"}


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(out.split("used = ")[1].split()[0].rstrip("M"))


def cmd_run(args) -> None:
    if args.train == "test":
        raise SystemExit("the final test split must not be used")
    preds = PRED_ROOT / args.tag
    preds.mkdir(parents=True, exist_ok=True)
    driver_path = preds / "driver.json"
    driver = json.loads(driver_path.read_text()) if driver_path.exists() else []
    for key in args.models:
        if any(d["model"] == key for d in driver):
            raise SystemExit(f"{key} already recorded in {driver_path}; refusing to rerun")
        swap0 = swap_used_mb()
        log = preds / f"{key}_memory.csv"
        cmd = [sys.executable, str(GUARD), "--limit-mb", str(MEMORY_LIMIT_MB), "--interval", "2", "--log", str(log), "--",
               sys.executable, str(PILOT), "fit", "--model", key, "--train", args.train, "--tag", args.tag]
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode
        wall = time.perf_counter() - t0
        swap1 = swap_used_mb()
        sampled = pd.read_csv(log)["rss_mb"].max() if log.exists() else float("nan")
        driver.append({"model": key, "exit_code": rc, "process_wall_sec": wall, "guard_sampled_peak_rss_mb": sampled,
                       "swap_used_before_mb": swap0, "swap_used_after_mb": swap1})
        driver_path.write_text(json.dumps(driver, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"[step5-4] {key}: exit={rc} wall={wall:.0f}s sampled peak={sampled:.0f} MB swap {swap0:.0f}->{swap1:.0f} MB",
              flush=True)
        if rc != 0:
            raise SystemExit(f"[step5-4] {key} failed (exit {rc}; 3 = memory limit); stopping without further models")
        if swap1 - swap0 > SWAP_JUMP_MB:
            raise SystemExit(f"[step5-4] swap grew by {swap1 - swap0:.0f} MB during {key}; stopping")
    rc = subprocess.run([sys.executable, str(PILOT), "evaluate", "--tag", args.tag]).returncode
    if rc != 0:
        raise SystemExit(f"evaluate failed (exit {rc})")


def cmd_compare(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    valid = load_split("valid", columns=[TREATMENT, OUTCOME])
    ids, T, Y = valid["row_id"].to_numpy(), valid[TREATMENT].to_numpy(), valid[OUTCOME].to_numpy()
    size = {args.base: "100k", args.new: "300k"} if args.base == "pilot_100k" else {args.base: args.base, args.new: args.new}
    rows, curves, bottoms = [], {}, []
    for tag in (args.base, args.new):
        runtime = pd.read_csv(TABLES / f"criteo_{tag}_runtime.csv").set_index("model")
        for key, name in MODEL_KEYS.items():
            npz = PRED_ROOT / tag / f"{key}_valid.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            if not np.array_equal(z["row_id"], ids):
                raise SystemExit(f"{tag}/{key}: row_id mismatch with validation split")
            s = z["score"]
            res = evaluate_uplift(s, T, Y)
            curves[(name, tag)] = res["curve"]
            bot = segment_effect(s, T, Y, BOTTOM, lowest=True)
            top = segment_effect(s, T, Y, 0.10, lowest=False)
            bottoms.append({"model": name, "train": size[tag], **bot})
            rt = runtime.loc[name]
            rows.append({
                "model": name, "train": size[tag], "train_split": rt.train_split, "n_train": int(rt.n_train),
                "train_conv_control": int(rt.train_conv_control), "train_conv_treated": int(rt.train_conv_treated),
                **{k: res[k] for k in ("auuc", "auuc_minus_random", "auuc_normalized", "qini_coefficient",
                                       "qini_normalized", "n_undefined_points")},
                **{f"top{q}_{m}": res[f"top{q}_{m}"] for q in (10, 20, 30) for m in ("rate_diff", "incremental_conversions")},
                "pred_mean": float(s.mean()), "pred_sd": float(s.std()), "pred_min": float(s.min()), "pred_max": float(s.max()),
                "share_negative": float((s < 0).mean()), **tie_summary(s),
                "top10_observed_rate_diff_sorted": top["rate_diff"],
                "bottom3_rate_diff": bot["rate_diff"], "bottom3_rate_diff_ci_low": bot["rate_diff_ci_low"],
                "bottom3_rate_diff_ci_high": bot["rate_diff_ci_high"],
                "fit_sec": rt.fit_sec, "full_predict_sec": rt.full_predict_sec, "peak_rss_mb": rt.peak_rss_mb,
                "guard_sampled_peak_rss_mb": rt.guard_sampled_peak_rss_mb, "model_file_mb": rt.model_file_mb,
            })
    table = pd.DataFrame(rows)
    deltas = []
    for name in table.model.unique():
        a = table[(table.model == name) & (table.train == size[args.base])]
        b = table[(table.model == name) & (table.train == size[args.new])]
        if len(a) and len(b):
            num = [c for c in table.columns if c not in ("model", "train", "train_split") and np.issubdtype(table[c].dtype, np.number)]
            d = {"model": name, "train": f"{size[args.new]} - {size[args.base]}", "train_split": ""}
            d.update({c: b[c].iloc[0] - a[c].iloc[0] for c in num})
            deltas.append(d)
    out = pd.concat([table, pd.DataFrame(deltas)], ignore_index=True)
    out.to_csv(TABLES / f"criteo_{size[args.base]}_vs_{size[args.new]}.csv", index=False, float_format="%.6g")
    bottom_df = pd.DataFrame(bottoms)
    overall = segment_effect(np.zeros(len(T)), T, Y, 1.0)
    bottom_df = pd.concat([bottom_df, pd.DataFrame([{"model": "Validation overall", "train": "-", **overall}])],
                          ignore_index=True)
    bottom_df.to_csv(TABLES / f"criteo_{size[args.new]}_bottom3_segment.csv", index=False, float_format="%.6g")

    figs = PROJECT_ROOT / "results" / "figures" / f"criteo_{size[args.new]}"
    figs.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next(f for f in ["AppleSDGothicNeo", "Apple SD Gothic Neo", "AppleGothic"]
                                             if f in available), "axes.unicode_minus": False, "pdf.fonttype": 42})
    base_curve = evaluate_uplift(np.zeros(len(T)), T, Y)
    names = [n for n in MODEL_KEYS.values() if (n, args.new) in curves or (n, args.base) in curves]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.0), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        for tag, ls, lw in ((args.base, ":", 1.8), (args.new, "-", 2.0)):
            if (name, tag) in curves:
                c = curves[(name, tag)]
                ax.plot(c["phi"] * 100, c["uplift"], color=COLORS[name], ls=ls, lw=lw, label=f"학습 {size[tag]}")
        ax.plot([0, 100], [0, base_curve["total_uplift"]], color="#6b6a64", lw=1, ls="--", label="무작위")
        ax.set_title(name, loc="left", fontsize=11)
        ax.set_xlabel("타기팅 비율 (%)", fontsize=9, color="#5f5e58")
        ax.grid(color="#e4e3df", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("추정 증분 전환 U(k)", color="#5f5e58")
    fig.suptitle(f"Uplift curve: 학습 {size[args.base]} vs {size[args.new]} (같은 검증 세트 {len(T):,}행)", x=0.01,
                 ha="left", fontsize=12)
    fig.text(0.01, 0.905, "검증 세트는 개발용 데이터 · 단일 학습 · 부트스트랩 구간 없음", fontsize=8.5, color="#5f5e58")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"uplift_curve_{size[args.base]}_vs_{size[args.new]}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    b = bottom_df[bottom_df.model != "Validation overall"].reset_index(drop=True)
    xs = np.arange(len(b))
    for i, r in b.iterrows():
        ax.errorbar(i, r.rate_diff, yerr=[[r.rate_diff - r.rate_diff_ci_low], [r.rate_diff_ci_high - r.rate_diff]],
                    fmt="o" if r.train == size[args.new] else "s", color=COLORS[r.model], capsize=4, ms=7,
                    mfc=COLORS[r.model] if r.train == size[args.new] else "white")
    ax.axhline(overall["rate_diff"], color="#6b6a64", ls="--", lw=1)
    ax.text(len(b) - 0.5, overall["rate_diff"], f"검증 전체 {overall['rate_diff']:.4f}", va="bottom", ha="right",
            fontsize=8.5, color="#5f5e58")
    ax.axhline(0, color="#b0afa8", lw=0.8)
    ax.set_xticks(xs, [f"{r.model}\n학습 {r.train}" for r in b.itertuples()], fontsize=8.5)
    ax.set_ylabel("관측 전환율 차 (처치 - 대조)", color="#5f5e58")
    ax.set_title(f"예측 하위 {BOTTOM:.0%} 고객의 관측 전환율 차", loc="left", fontsize=12, pad=20)
    ax.text(0, 1.02, "오차막대: 정규근사 95% 신뢰구간 · 검증 세트 · 모델 선택 과정의 다중 비교 미보정", transform=ax.transAxes,
            fontsize=8.5, color="#5f5e58")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"bottom3_segment_{size[args.base]}_vs_{size[args.new]}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    show = ["model", "train", "auuc", "auuc_normalized", "qini_coefficient", "top10_incremental_conversions",
            "top20_incremental_conversions", "top30_incremental_conversions", "share_zero", "mode_share",
            "largest_tie_block", "fit_sec", "peak_rss_mb"]
    print(out[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(bottom_df[["model", "train", "k", "n_treated", "n_control", "conv_treated", "conv_control", "rate_treated",
                     "rate_control", "rate_diff", "rate_diff_ci_low", "rate_diff_ci_high", "boundary_tie_rows"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5g}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", choices=list(MODEL_KEYS), required=True)
    r.add_argument("--train", default="train_300000")
    r.add_argument("--tag", default="300k")
    c = sub.add_parser("compare")
    c.add_argument("--base", default="pilot_100k")
    c.add_argument("--new", default="300k")
    args = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
