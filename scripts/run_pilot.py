"""Pilot experiment: fit T-Learner, X-Learner and Causal Forest once on the default dataset.

Usage (from the project root, after `python scripts/generate_data.py`):
    python scripts/run_pilot.py
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib import font_manager  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation import SimulationConfig, _sha256, dataset_stem  # noqa: E402
from src.evaluation import constant_ate_baseline, evaluate_models, group_counts, split_train_test  # noqa: E402
from src.models import build_models, describe_model  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
KOREAN_FONTS = ["AppleSDGothicNeo", "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]


def load_dataset(config: SimulationConfig, data_dir: Path) -> tuple[pd.DataFrame, Path, dict]:
    """Load the saved CSV and verify it against its metadata sidecar (hash + config)."""
    stem = dataset_stem(config)
    csv_path, meta_path = data_dir / f"{stem}.csv", data_dir / f"{stem}.meta.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `python scripts/generate_data.py` first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if _sha256(csv_path) != meta["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {csv_path.name}: file differs from its metadata")
    if meta["config"] != asdict(config):
        raise RuntimeError(f"Config mismatch: metadata {meta['config']} vs requested {asdict(config)}")
    return pd.read_csv(csv_path), csv_path, meta


def set_korean_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise RuntimeError(f"No Korean font found among {KOREAN_FONTS}")


def plot_rmse(metrics: pd.DataFrame, config: SimulationConfig, path: Path) -> str:
    font = set_korean_font()
    ink, muted, grid, bar = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    bars = ax.bar(metrics["Model"], metrics["RMSE"], width=0.5, color=bar, zorder=3)
    for rect, value in zip(bars, metrics["RMSE"]):
        ax.annotate(f"{value:.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, color=ink)

    ax.set_title("예비 실험: 모델별 CATE 추정 오차 (RMSE)", fontsize=13, color=ink, loc="left", pad=22)
    ax.text(0, 1.02, f"테스트 데이터 기준 · n={config.n}, 처치군 비율={config.treat_prob}, "
            f"효과={config.effect}, seed={config.seed} · 1회 실행",
            transform=ax.transAxes, fontsize=8.5, color=muted)
    ax.set_xlabel("모델", color=muted)
    ax.set_ylabel("CATE RMSE (낮을수록 좋음)", color=muted)
    ax.set_ylim(0, metrics["RMSE"].max() * 1.2)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(colors=muted, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return font


def main() -> None:
    sim_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "simulation.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    config = SimulationConfig(**sim_cfg["default"])
    seed = config.seed

    df, csv_path, data_meta = load_dataset(config, PROJECT_ROOT / sim_cfg["output"]["synthetic_dir"])
    split_cfg = model_cfg["split"]
    train, test = split_train_test(df, split_cfg["test_size"], seed)
    counts = {"full": group_counts(df), "train": group_counts(train), "test": group_counts(test)}

    print(f"Data: {csv_path.relative_to(PROJECT_ROOT)} (sha256 verified)")
    print("\n[Train/test split, stratified on T]")
    print(pd.DataFrame(counts).T[["total", "treated", "control"]])

    models = build_models(model_cfg, seed=seed, treat_prob=config.treat_prob)
    metrics, predictions, fit_info = evaluate_models(models, train, test, config.n_covariates)
    baseline = constant_ate_baseline(train, test)

    print("\n[Training rows seen by each model]")
    print(pd.DataFrame(fit_info).T)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n[Test-set CATE metrics]")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nReference — constant ATE (diff-in-means = {baseline['ate_estimate']:.4f}): "
          f"RMSE {baseline['RMSE']:.4f}, MAE {baseline['MAE']:.4f}")
    print(f"Causal Forest estimator: {models['Causal Forest'].estimator_name}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLES_DIR / "pilot_model_comparison.csv"
    pred_path = TABLES_DIR / "pilot_predictions.csv"
    fig_path = FIGURES_DIR / "pilot_rmse_comparison.png"
    meta_path = TABLES_DIR / "pilot_experiment_metadata.json"

    metrics[["Model", "RMSE", "MAE", "Train Time (sec)"]].to_csv(table_path, index=False, float_format="%.6f")
    predictions.to_csv(pred_path, index=False, float_format="%.10g")
    font = plot_rmse(metrics, config, fig_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "pilot (single run)",
        "seed": seed,
        "data": {
            "file": str(csv_path.relative_to(PROJECT_ROOT)),
            "sha256": data_meta["sha256"],
            "config": data_meta["config"],
        },
        "split": {
            "method": "sklearn.model_selection.train_test_split",
            "test_size": split_cfg["test_size"],
            "stratify": split_cfg["stratify"],
            "random_state": seed,
            "counts": counts,
        },
        "model_inputs": ["X1..X10", "T", "Y"],
        "excluded_from_training": ["true_cate", "mu0"],
        "models": {name: {**describe_model(m), "n_train": fit_info[name]} for name, m in models.items()},
        "metrics": metrics.to_dict(orient="records"),
        "reference_constant_ate": baseline,
        "outputs": {
            "table": str(table_path.relative_to(PROJECT_ROOT)),
            "predictions": str(pred_path.relative_to(PROJECT_ROOT)),
            "figure": str(fig_path.relative_to(PROJECT_ROOT)),
            "figure_font": font,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **{pkg: version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "econml", "scipy", "matplotlib", "PyYAML"]},
        },
        "notes": "Single pilot run; train times are indicative only. No statistical comparison is implied.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nSaved: {table_path.relative_to(PROJECT_ROOT)}, {pred_path.relative_to(PROJECT_ROOT)}, "
          f"{fig_path.relative_to(PROJECT_ROOT)}, {meta_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
