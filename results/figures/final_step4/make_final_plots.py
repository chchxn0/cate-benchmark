"""STEP 4-1: final figures from the formal OFAT simulation results.

Reads only results/tables/formal_summary.csv (no training, no new experiments)
and writes PNG (dpi 300) + PDF figures, plot-data CSVs and captions next to this file.

Usage (from the project root):
    .venv/bin/python results/figures/final_step4/make_final_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUT_DIR.parents[2]
SUMMARY_CSV = PROJECT_ROOT / "results" / "tables" / "formal_summary.csv"

KOREAN_FONTS = ["AppleSDGothicNeo", "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
MODELS = ["T-Learner", "X-Learner", "Causal Forest", "Constant ATE"]
MODEL_SHORT = {"T-Learner": "T", "X-Learner": "X", "Causal Forest": "CF", "Constant ATE": "Constant"}
# Fixed per-model encoding: color + marker (+ dashed line for the non-heterogeneous reference model).
COLORS = {"T-Learner": "#2a78d6", "X-Learner": "#eb6834", "Causal Forest": "#1baf7a", "Constant ATE": "#6b6a64"}
MARKERS = {"T-Learner": "o", "X-Learner": "s", "Causal Forest": "^", "Constant ATE": "D"}
LINESTYLES = {"T-Learner": "-", "X-Learner": "-", "Causal Forest": "-", "Constant ATE": "--"}
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3df"
CONDITIONS = ["c00", "c01", "c02", "c03", "c04", "c05", "c06"]
ERRBAR_NOTE = "오차막대: 30회 반복 평균 RMSE의 95% 신뢰구간 반폭 (t 분포)"


def set_korean_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise RuntimeError(f"No Korean font found among {KOREAN_FONTS}")


def condition_label(row: pd.Series, sep: str = " / ") -> str:
    return sep.join([f"n={int(row.n):,}", f"p={row.treat_prob:g}", row.effect])


def load_summary() -> pd.DataFrame:
    s = pd.read_csv(SUMMARY_CSV)
    if set(s.condition_id) != set(CONDITIONS) or (s.n_reps != 30).any() or set(s.model) != set(MODELS):
        raise ValueError("formal_summary.csv is not the complete 7-condition x 4-model x 30-rep result")
    s["condition_label"] = s.apply(condition_label, axis=1)
    return s


def plot_data(s: pd.DataFrame, condition_ids: list[str], factor: str) -> pd.DataFrame:
    cols = ["condition_id", "condition_label", "n", "treat_prob", "effect", "model", "n_reps",
            "rmse_mean", "rmse_sd", "rmse_ci_low", "rmse_ci_high", "rmse_ci_halfwidth", "mae_mean", "mae_sd"]
    d = s[s.condition_id.isin(condition_ids)][cols].copy()
    d.insert(1, "factor", factor)
    d["condition_id"] = pd.Categorical(d.condition_id, condition_ids, ordered=True)
    d["model"] = pd.Categorical(d.model, MODELS, ordered=True)
    return d.sort_values(["condition_id", "model"]).reset_index(drop=True)


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, color=INK, loc="left", pad=26)
    ax.set_xlabel(xlabel, fontsize=11, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=11, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")


def save(fig, stem: str) -> list[str]:
    names = []
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        names.append(path.name)
    plt.close(fig)
    return names


def line_plot(d: pd.DataFrame, x_values, x_labels, stem: str, title: str, xlabel: str, log_x: bool = False):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for m in MODELS:
        g = d[d.model == m]
        ax.errorbar(x_values, g.rmse_mean, yerr=g.rmse_ci_halfwidth, color=COLORS[m], marker=MARKERS[m],
                    linestyle=LINESTYLES[m], linewidth=2, markersize=7, capsize=4, elinewidth=1.4,
                    markeredgecolor="white", markeredgewidth=1.2, label=m, zorder=3)
        ax.annotate(f"{g.rmse_mean.iloc[-1]:.3f}", (x_values[-1], g.rmse_mean.iloc[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(x_values[0] / 1.6, x_values[-1] * 2.2)
    else:
        ax.set_xlim(x_values[0] - 0.3, x_values[-1] + 0.5)
    ax.set_xticks(x_values, x_labels)
    ax.minorticks_off()
    ax.set_ylim(0, None)
    style_axes(ax, title, xlabel, "평균 RMSE")
    subtitle(ax, f"30회 반복 · {ERRBAR_NOTE.replace('오차막대: 30회 반복 ', '오차막대: ')} · 낮을수록 좋음")
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", ncol=2)
    return save(fig, stem)


def grouped_bars(ax, d: pd.DataFrame, condition_ids: list[str], value: str, err: str | None) -> None:
    x = np.arange(len(condition_ids))
    width = 0.8 / len(MODELS)
    for i, m in enumerate(MODELS):
        g = d[d.model == m].set_index("condition_id").loc[condition_ids]
        ax.bar(x + (i - 1.5) * width, g[value], width=width * 0.92, color=COLORS[m], label=m, zorder=3,
               yerr=g[err] if err else None, capsize=3 if err else 0,
               error_kw=dict(ecolor=INK, elinewidth=1, capthick=1))


def main() -> None:
    font = set_korean_font()
    s = load_summary()
    written: list[str] = []

    # (1) sample size
    ids = ["c00", "c01", "c02"]
    d = plot_data(s, ids, "sample_size")
    d.to_csv(OUT_DIR / "final_plot_data_sample_size.csv", index=False, float_format="%.6f")
    written.append("final_plot_data_sample_size.csv")
    written += line_plot(d, [1000, 10000, 100000], ["1,000", "10,000", "100,000"], "sample_size_rmse",
                         "표본 크기에 따른 CATE RMSE", "표본 크기 (로그 축)", log_x=True)

    # (2) treatment ratio (p=0.5 -> 0.3 -> 0.1, n=10,000, linear)
    ids = ["c01", "c03", "c04"]
    d = plot_data(s, ids, "treatment_ratio")
    d.to_csv(OUT_DIR / "final_plot_data_treatment_ratio.csv", index=False, float_format="%.6f")
    written.append("final_plot_data_treatment_ratio.csv")
    written += line_plot(d, [0, 1, 2], ["0.5", "0.3", "0.1"], "treatment_ratio_rmse",
                         "처치군 비율에 따른 CATE RMSE", "처치군 비율")

    # (3) effect complexity (n=10,000, p=0.5): grouped bars, categories are unordered
    ids = ["c01", "c05", "c06"]
    d = plot_data(s, ids, "effect_complexity")
    d.to_csv(OUT_DIR / "final_plot_data_complexity.csv", index=False, float_format="%.6f")
    written.append("final_plot_data_complexity.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    grouped_bars(ax, d, ids, "rmse_mean", "rmse_ci_halfwidth")
    ax.set_xticks(np.arange(3), ["linear", "nonlinear", "interaction"])
    ax.set_ylim(0, 0.36)
    style_axes(ax, "처치효과 복잡성에 따른 CATE RMSE", "효과 구조", "평균 RMSE")
    subtitle(ax, "n=10,000 · p=0.5 · 30회 반복 · 오차막대: 평균 RMSE의 95% 신뢰구간 반폭 (t 분포) · 낮을수록 좋음")
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    written += save(fig, "effect_complexity_rmse")

    # (4) stability: RMSE SD across replications, all conditions
    d = plot_data(s, CONDITIONS, "all_conditions")
    stab = d[["condition_id", "condition_label", "n", "treat_prob", "effect", "model", "n_reps", "rmse_mean", "rmse_sd"]].copy()
    stab["rmse_cv"] = stab.rmse_sd / stab.rmse_mean
    stab.to_csv(OUT_DIR / "final_plot_data_stability.csv", index=False, float_format="%.6f")
    written.append("final_plot_data_stability.csv")
    labels = s.drop_duplicates("condition_id").set_index("condition_id").loc[CONDITIONS]
    tick_labels = [condition_label(r, "\n") for _, r in labels.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped_bars(ax, stab, CONDITIONS, "rmse_sd", None)
    ax.set_xticks(np.arange(len(CONDITIONS)), tick_labels, fontsize=9)
    style_axes(ax, "조건별 모델 성능 안정성 비교", "실험 조건", "RMSE 표준편차")
    subtitle(ax, "막대: 30회 반복 RMSE의 표본 표준편차 (오차막대 없음) · 낮을수록 반복 간 변동이 작음")
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper right")
    written += save(fig, "model_stability_sd")

    # (5) heatmap: single-hue sequential, light = low RMSE (better)
    mat = s.pivot(index="condition_id", columns="model", values="rmse_mean").loc[CONDITIONS, MODELS]
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8, 5.6))
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=float(mat.values.max()), aspect="auto")
    best = mat.values.argmin(axis=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            dark = v / mat.values.max() > 0.55
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=10.5,
                    color="white" if dark else INK, fontweight="bold" if j == best[i] else "normal")
    ax.set_xticks(range(len(MODELS)), [MODEL_SHORT[m] if m == "Constant ATE" else m for m in MODELS], fontsize=10)
    ax.set_yticks(range(len(CONDITIONS)), [condition_label(r) for _, r in labels.iterrows()], fontsize=9.5)
    ax.tick_params(length=0, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("조건별 모델 평균 RMSE 히트맵", fontsize=14, color=INK, loc="left", pad=26)
    ax.text(0, 1.02, "셀 값: 30회 반복 평균 RMSE · 밝을수록 오차가 작음(좋음) · 굵은 글씨: 조건별 최저값",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("평균 RMSE (낮을수록 좋음)", color=MUTED, fontsize=9.5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, labelsize=8.5)
    written += save(fig, "rmse_heatmap")

    write_captions(s)
    print(f"font: {font}")
    print("written:", ", ".join(written))


def write_captions(s: pd.DataFrame) -> None:
    r = s.set_index(["condition_id", "model"]).rmse_mean
    sd = s.set_index(["condition_id", "model"]).rmse_sd
    caps = {
        "sample_size_rmse": (
            "그림. 표본 크기(n=1,000/10,000/100,000; p=0.5, 선형 효과)에 따른 모델별 평균 CATE RMSE(30회 반복). "
            f"{ERRBAR_NOTE}이며, x축은 로그 척도이다. "
            f"표본이 커질수록 T·X·Causal Forest의 RMSE는 모두 감소했고 Causal Forest의 감소율이 가장 컸으나"
            f"({r['c00', 'Causal Forest']:.4f}→{r['c02', 'Causal Forest']:.4f}, "
            f"{(r['c02', 'Causal Forest'] / r['c00', 'Causal Forest'] - 1) * 100:.1f}%), "
            f"이질성을 반영하지 않는 Constant ATE는 약 {r['c02', 'Constant ATE']:.2f} 수준에 머물렀다."
        ),
        "treatment_ratio_rmse": (
            "그림. 처치군 비율(p=0.5/0.3/0.1; n=10,000, 선형 효과)에 따른 모델별 평균 CATE RMSE(30회 반복). "
            f"{ERRBAR_NOTE}이다. "
            f"처치군 비율이 0.1로 줄면 세 학습기 모두 RMSE가 증가했으며, T-Learner는 p=0.1에서 "
            f"Constant ATE보다 평균 RMSE가 높았다({r['c04', 'T-Learner']:.4f} vs {r['c04', 'Constant ATE']:.4f})."
        ),
        "effect_complexity_rmse": (
            "그림. 처치효과 구조(linear/nonlinear/interaction; n=10,000, p=0.5)에 따른 모델별 평균 CATE RMSE(30회 반복). "
            f"{ERRBAR_NOTE}이다. "
            "선형 효과에서는 Causal Forest의 RMSE가 가장 낮았지만, 상호작용 효과에서는 X-Learner가 더 낮았고 "
            "비선형 효과에서는 두 모델의 차이가 다중비교 보정 후 통계적으로 확인되지 않았다(대응 차이 95% CI가 0 포함)."
        ),
        "model_stability_sd": (
            "그림. 7개 실험 조건에서 30회 반복 RMSE의 표준편차(막대 높이; 오차막대 없음). 값이 작을수록 반복 간 성능 변동이 작다. "
            f"모든 모델에서 표본 크기가 커질수록 표준편차가 크게 줄었으며(예: Causal Forest {sd['c00', 'Causal Forest']:.4f}→"
            f"{sd['c02', 'Causal Forest']:.4f}), n=1,000 조건의 변동이 가장 컸다."
        ),
        "rmse_heatmap": (
            "그림. 7개 실험 조건 × 4개 모델의 평균 CATE RMSE(30회 반복) 히트맵. 밝은 셀일수록 오차가 작고, 굵은 숫자는 조건별 최저값이다. "
            "선형 효과 5개 조건에서는 Causal Forest가, 비선형·상호작용 조건에서는 X-Learner가 가장 낮은 평균 RMSE를 보였다"
            "(단, 비선형 조건의 X–CF 차이는 통계적으로 확인되지 않음)."
        ),
    }
    for stem, text in caps.items():
        (OUT_DIR / f"{stem}_caption.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
