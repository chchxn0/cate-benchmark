"""STEP 5-3: Criteo pilot - fit T-Learner, X-Learner and Causal Forest once on a training split, predict the
validation split and evaluate uplift metrics on validation only. The final test split is never loaded.

Subcommands (from the project root):
    # all three models sequentially, each in its own process under memory_guard, then evaluation
    caffeinate -i .venv/bin/python scripts/criteo_pilot.py all --train train_100000 --tag pilot_100k

    .venv/bin/python scripts/criteo_pilot.py fit --model T --train train_100000 --tag pilot_100k
    .venv/bin/python scripts/criteo_pilot.py evaluate --tag pilot_100k

Outputs:
    data/criteo/models/<tag>/<model>.joblib            fitted estimator (Git-ignored)
    data/criteo/predictions/<tag>/<model>_valid.npz    row_id + score for every validation row (Git-ignored)
    data/criteo/predictions/<tag>/<model>_run.json     per-model timing / memory / diagnostics
    results/tables/criteo_<tag>_{metrics,runtime,predictions_summary}.csv, criteo_<tag>_metadata.json
    results/figures/criteo_<tag>/
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.criteo_data import FEATURES, OUTCOME, SPLIT_DIR, TREATMENT, load_split  # noqa: E402
from src.criteo_evaluation import evaluate_uplift  # noqa: E402
from src.criteo_models import (  # noqa: E402
    MODEL_KEYS,
    build_criteo_model,
    fixed_subset_positions,
    predict_in_batches,
    prediction_summary,
)
from src.models import describe_model  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "criteo_models.yaml"
TABLES = PROJECT_ROOT / "results" / "tables"
MODEL_ORDER = ["T", "X", "CF"]
VALID_SPLIT = "valid"
SWAP_JUMP_MB = 1024           # stop the driver if swap grows more than this during one model
MEMORY_LIMIT_MB = 7000


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # macOS: bytes


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = out.split("used = ")[1].split()[0]
    return float(used.rstrip("M"))


def paths(tag: str) -> dict:
    return {"models": PROJECT_ROOT / "data" / "criteo" / "models" / tag,
            "preds": PROJECT_ROOT / "data" / "criteo" / "predictions" / tag,
            "figs": PROJECT_ROOT / "results" / "figures" / f"criteo_{tag}"}


def guard_split(name: str) -> None:
    if name == "test":
        raise SystemExit("the final test split must not be used in STEP 5-3")


def xty(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return df[FEATURES].to_numpy(), df[TREATMENT].to_numpy(), df[OUTCOME].to_numpy()


def cmd_fit(args) -> None:
    guard_split(args.train)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    p = paths(args.tag)
    for d in (p["models"], p["preds"]):
        d.mkdir(parents=True, exist_ok=True)
    run_path = p["preds"] / f"{args.model}_run.json"
    if run_path.exists():
        raise SystemExit(f"{run_path} exists; refusing to overwrite (use a new --tag)")
    run = {"model": MODEL_KEYS[args.model], "key": args.model, "train_split": args.train, "valid_split": VALID_SPLIT,
           "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    t0 = time.perf_counter()
    train = load_split(args.train)
    X, T, Y = xty(train)
    run["load_train_sec"] = time.perf_counter() - t0
    run["n_train"] = {"total": len(T), "treated": int(T.sum()), "control": int((1 - T).sum()),
                      "conv_treated": int((T * Y).sum()), "conv_control": int(((1 - T) * Y).sum())}
    treated_share = float(T.mean())
    model = build_criteo_model(args.model, cfg, treated_share)
    run["config"] = describe_model(model)
    run["seed"] = cfg["seed"]
    print(f"[{args.model}] config: {json.dumps(run['config'], default=str)}", flush=True)
    print(f"[{args.model}] seed={cfg['seed']} train={args.train} n={len(T):,} treated_share={treated_share:.4f}", flush=True)

    t0 = time.perf_counter()
    model.fit(X, T, Y)
    run["fit_sec"] = time.perf_counter() - t0
    run["peak_rss_after_fit_mb"] = peak_rss_mb()
    train_pred = predict_in_batches(model, X, cfg["prediction"]["batch_size"])
    run["train_prediction_summary"] = prediction_summary(train_pred)
    print(f"[{args.model}] fit {run['fit_sec']:.1f}s, peak RSS {run['peak_rss_after_fit_mb']:.0f} MB", flush=True)
    del train, X, T, Y, train_pred

    t0 = time.perf_counter()
    joblib.dump(model, p["models"] / f"{args.model}.joblib", compress=3)
    run["save_model_sec"] = time.perf_counter() - t0
    run["model_file_mb"] = (p["models"] / f"{args.model}.joblib").stat().st_size / 1024**2

    t0 = time.perf_counter()
    valid = load_split(VALID_SPLIT)
    run["load_valid_sec"] = time.perf_counter() - t0
    expected_ids = np.load(SPLIT_DIR / f"{VALID_SPLIT}.npy")
    if not np.array_equal(valid["row_id"].to_numpy(), expected_ids):
        raise SystemExit("validation row_id order does not match the saved split")
    Xv = valid[FEATURES].to_numpy()
    pc = cfg["prediction"]
    sub = fixed_subset_positions(len(Xv), pc["subset_size"], pc["subset_seed"])
    t0 = time.perf_counter()
    sub_pred = predict_in_batches(model, Xv[sub], pc["batch_size"])
    run["subset_predict_sec"] = time.perf_counter() - t0
    run["subset_size"] = len(sub)
    run["subset_prediction_summary"] = prediction_summary(sub_pred)
    est_min = run["subset_predict_sec"] * len(Xv) / len(sub) / 60
    run["full_predict_estimate_min"] = est_min
    print(f"[{args.model}] subset {len(sub):,} rows predicted in {run['subset_predict_sec']:.1f}s "
          f"-> full estimate {est_min:.1f} min", flush=True)
    if est_min > pc["max_full_predict_minutes"]:
        run["status"] = "fit_ok_full_prediction_skipped"
        run["skip_reason"] = f"estimated {est_min:.1f} min > limit {pc['max_full_predict_minutes']} min"
    else:
        t0 = time.perf_counter()
        pred = predict_in_batches(model, Xv, pc["batch_size"])
        run["full_predict_sec"] = time.perf_counter() - t0
        if not np.allclose(pred[sub], sub_pred, rtol=0, atol=1e-12):
            raise SystemExit("subset and full predictions disagree on the same rows")
        np.savez(p["preds"] / f"{args.model}_valid.npz", row_id=valid["row_id"].to_numpy().astype(np.int32), score=pred)
        run["valid_prediction_summary"] = prediction_summary(pred)
        run["status"] = "ok"
        print(f"[{args.model}] full validation ({len(pred):,} rows) predicted in {run['full_predict_sec']:.1f}s", flush=True)
    run["peak_rss_mb"] = peak_rss_mb()
    run["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    print(f"[{args.model}] status={run['status']} peak RSS {run['peak_rss_mb']:.0f} MB", flush=True)


def cmd_all(args) -> None:
    guard_split(args.train)
    p = paths(args.tag)
    p["preds"].mkdir(parents=True, exist_ok=True)
    driver = []
    for key in MODEL_ORDER:
        swap0 = swap_used_mb()
        log = p["preds"] / f"{key}_memory.csv"
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "memory_guard.py"), "--limit-mb", str(MEMORY_LIMIT_MB),
               "--interval", "2", "--log", str(log), "--",
               sys.executable, str(Path(__file__).resolve()), "fit", "--model", key, "--train", args.train, "--tag", args.tag, "--config", str(args.config)]
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode
        wall = time.perf_counter() - t0
        swap1 = swap_used_mb()
        sampled = pd.read_csv(log)["rss_mb"].max() if log.exists() else float("nan")
        entry = {"model": key, "exit_code": rc, "process_wall_sec": wall, "guard_sampled_peak_rss_mb": sampled,
                 "swap_used_before_mb": swap0, "swap_used_after_mb": swap1}
        driver.append(entry)
        print(f"[driver] {key}: exit={rc} wall={wall:.0f}s sampled peak={sampled:.0f} MB swap {swap0:.0f}->{swap1:.0f} MB",
              flush=True)
        (p["preds"] / "driver.json").write_text(json.dumps(driver, indent=2, default=float) + "\n", encoding="utf-8")
        if rc != 0:
            raise SystemExit(f"[driver] {key} failed (exit {rc}); not starting the next model")
        if swap1 - swap0 > SWAP_JUMP_MB:
            raise SystemExit(f"[driver] swap grew by {swap1 - swap0:.0f} MB during {key}; stopping")
    cmd_evaluate(args)


def cmd_evaluate(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    p = paths(args.tag)
    p["figs"].mkdir(parents=True, exist_ok=True)
    driver = {d["model"]: d for d in json.loads((p["preds"] / "driver.json").read_text())} \
        if (p["preds"] / "driver.json").exists() else {}
    runs = {k: json.loads((p["preds"] / f"{k}_run.json").read_text()) for k in MODEL_ORDER
            if (p["preds"] / f"{k}_run.json").exists()}
    valid = load_split(VALID_SPLIT, columns=[TREATMENT, OUTCOME])
    ids, T, Y = valid["row_id"].to_numpy(), valid[TREATMENT].to_numpy(), valid[OUTCOME].to_numpy()

    runtime, pred_rows, metric_rows, curves = [], [], [], {}
    for key, run in runs.items():
        d = driver.get(key, {})
        runtime.append({"model": run["model"], "status": run["status"], "train_split": run["train_split"],
                        "n_train": run["n_train"]["total"], "train_conv_treated": run["n_train"]["conv_treated"],
                        "train_conv_control": run["n_train"]["conv_control"], "fit_sec": run["fit_sec"],
                        "peak_rss_after_fit_mb": run["peak_rss_after_fit_mb"], "peak_rss_mb": run["peak_rss_mb"],
                        "guard_sampled_peak_rss_mb": d.get("guard_sampled_peak_rss_mb"),
                        "subset_predict_sec": run["subset_predict_sec"], "subset_size": run["subset_size"],
                        "full_predict_sec": run.get("full_predict_sec"), "n_valid": len(ids),
                        "model_file_mb": run["model_file_mb"], "save_model_sec": run["save_model_sec"],
                        "load_train_sec": run["load_train_sec"], "load_valid_sec": run["load_valid_sec"],
                        "process_wall_sec": d.get("process_wall_sec"),
                        "swap_used_before_mb": d.get("swap_used_before_mb"), "swap_used_after_mb": d.get("swap_used_after_mb")})
        for where in ("train", "valid"):
            s = run.get(f"{where}_prediction_summary")
            if s:
                pred_rows.append({"model": run["model"], "data": where, **s})
        npz = p["preds"] / f"{key}_valid.npz"
        if run["status"] != "ok" or not npz.exists():
            continue
        z = np.load(npz)
        if not np.array_equal(z["row_id"], ids):
            raise SystemExit(f"{key}: prediction row_id does not match validation rows")
        res = evaluate_uplift(z["score"], T, Y)
        ipw = evaluate_uplift(z["score"], T, Y, estimator="ipw")
        curves[run["model"]] = res["curve"]
        metric_rows.append({"model": run["model"], **{k: v for k, v in res.items() if k != "curve"},
                            "auuc_ipw": ipw["auuc"], "auuc_minus_random_ipw": ipw["auuc_minus_random"]})
    base = evaluate_uplift(np.zeros(len(ids)), T, Y)
    metric_rows.append({"model": "Random (all-equal score)", **{k: v for k, v in base.items() if k != "curve"}})
    metrics = pd.DataFrame(metric_rows)
    metrics.insert(1, "eval_split", VALID_SPLIT)
    stem = f"criteo_{args.tag}"
    pd.DataFrame(runtime).to_csv(TABLES / f"{stem}_runtime.csv", index=False, float_format="%.6g")
    pd.DataFrame(pred_rows).to_csv(TABLES / f"{stem}_predictions_summary.csv", index=False, float_format="%.6g")
    metrics.to_csv(TABLES / f"{stem}_metrics.csv", index=False, float_format="%.6g")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": args.tag,
            "step": "STEP 5-3 pilot (validation only; final test never loaded)", "config_file": str(Path(args.config).resolve().relative_to(PROJECT_ROOT)) if Path(args.config).resolve().is_relative_to(PROJECT_ROOT) else str(args.config),
            "config": cfg, "model_descriptions": {r["model"]: r["config"] for r in runs.values()},
            "evaluation": {"split": VALID_SPLIT, "n": len(ids), "n_grid": 100, "estimator": "ratio (ipw reported as check)",
                           "definitions": "see src/criteo_evaluation.py and docs/criteo_step5_2_report.md"},
            "artifacts": {"models": str(p["models"].relative_to(PROJECT_ROOT)), "predictions": str(p["preds"].relative_to(PROJECT_ROOT))},
            "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                            **{k: version(k) for k in ["numpy", "pandas", "scikit-learn", "econml", "joblib"]}}}
    (TABLES / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float) + "\n",
                                                  encoding="utf-8")

    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next(f for f in ["AppleSDGothicNeo", "Apple SD Gothic Neo", "AppleGothic"]
                                             if f in available), "axes.unicode_minus": False, "pdf.fonttype": 42})
    colors = {"T-Learner": "#2a78d6", "X-Learner": "#eb6834", "Causal Forest": "#1baf7a"}
    styles = {"T-Learner": "-", "X-Learner": "--", "Causal Forest": "-."}
    for kind, ylab, title in (("uplift", "추정 증분 전환 건수 U(k)", "Uplift curve"),
                              ("qini", "Qini Q(k) (처치군 규모 환산 증분 전환)", "Qini curve")):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for m, c in curves.items():
            ax.plot(c["phi"] * 100, c[kind], color=colors[m], ls=styles[m], lw=2, label=m)
        end = base["total_uplift"] if kind == "uplift" else base["total_qini"]
        ax.plot([0, 100], [0, end], color="#6b6a64", lw=1.4, ls=":", label="무작위 타기팅")
        ax.set_xlabel("타기팅 비율 (예측 uplift 상위 %)", color="#5f5e58")
        ax.set_ylabel(ylab, color="#5f5e58")
        ax.set_title(f"{title} — 검증 세트 {len(ids):,}행 (학습 {args.tag.split('_')[-1]} 파일럿)", loc="left", fontsize=12, pad=20)
        ax.text(0, 1.01, "파일럿 참고용 · 단일 학습 · 부트스트랩 구간 없음", transform=ax.transAxes, fontsize=8.5,
                color="#5f5e58", va="bottom")
        ax.grid(color="#e4e3df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(p["figs"] / f"{kind}_curve_valid.{ext}", dpi=200, facecolor="white")
        plt.close(fig)
    if curves:
        fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.6))
        for ax, (m, _) in zip(np.atleast_1d(axes), curves.items()):
            key = next(k for k, v in MODEL_KEYS.items() if v == m)
            s = np.load(p["preds"] / f"{key}_valid.npz")["score"]
            lo, hi = np.quantile(s, [0.001, 0.999])
            ax.hist(np.clip(s, lo, hi), bins=80, color=colors[m])
            ax.axvline(0, color="#6b6a64", lw=1, ls=":")
            ax.set_title(f"{m}", loc="left", fontsize=11)
            ax.set_xlabel("예측 CATE (0.1~99.9% 분위 범위)", fontsize=8.5, color="#5f5e58")
            ax.set_yscale("log")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle("검증 세트 예측 CATE 분포 (로그 빈도)", x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(p["figs"] / f"prediction_distribution_valid.{ext}", dpi=200, facecolor="white")
        plt.close(fig)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(pd.DataFrame(runtime).to_string(index=False))
    print(metrics[["model", "auuc", "auuc_random", "auuc_minus_random", "auuc_normalized", "qini_coefficient",
                   "qini_normalized", "top10_rate_diff", "top20_rate_diff", "top30_rate_diff",
                   "top10_incremental_conversions", "n_undefined_points"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", choices=MODEL_ORDER, required=True)
    for s in (f, sub.add_parser("all")):
        s.add_argument("--train", default="train_100000")
        s.add_argument("--tag", default="pilot_100k")
    e = sub.add_parser("evaluate")
    e.add_argument("--tag", default="pilot_100k")
    for s in (f, e, sub.choices["all"]):
        s.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args()
    {"fit": cmd_fit, "all": cmd_all, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
