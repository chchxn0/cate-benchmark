"""STEP 4-2: publication tables (PNG 300 dpi + vector PDF + editable CSV) from the formal results.

Reads only results/tables/formal_*.csv (no training, no new experiments). Before drawing, every
number is cross-checked: summary means/SDs are recomputed from formal_runs.csv, rates of change are
recomputed from the summary and compared with the comparison CSVs, and significance labels are
checked against both the Holm p-value and the confidence interval.

Usage (from the project root):
    .venv/bin/python results/figures/final_step4/tables/make_final_tables.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUT_DIR.parents[3]
TABLES = PROJECT_ROOT / "results" / "tables"

KOREAN_FONTS = ["AppleSDGothicNeo", "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
MODELS = ["T-Learner", "X-Learner", "Causal Forest", "Constant ATE"]
LEARNERS = MODELS[:3]
SHORT = {"T-Learner": "T", "X-Learner": "X", "Causal Forest": "CF", "Constant ATE": "Constant ATE"}
CONDITIONS = ["c00", "c01", "c02", "c03", "c04", "c05", "c06"]
N_REPS = 30
ALPHA = 0.05
INK, MUTED = "#111111", "#555555"

METRIC_NOTE = ("RMSE: 테스트 세트(전체의 20%)에서 CATE 추정값과 참값 차이의 제곱평균제곱근(낮을수록 좋음). "
               f"각 조건 {N_REPS}회 반복(data_seed = 42 + rep, model_seed = 1000 + rep).")
SD_NOTE = "평균 ± SD: 30회 반복 RMSE의 평균과 표본 표준편차(반복 간 변동이며 평균의 신뢰구간이 아님)."


# ---------------------------------------------------------------- data + verification
def load() -> dict[str, pd.DataFrame]:
    d = {name: pd.read_csv(TABLES / f"formal_{name}.csv") for name in
         ("runs", "summary", "paired", "sample_size_comparison", "treat_ratio_comparison", "complexity_comparison")}
    runs, summary = d["runs"], d["summary"]
    counts = runs.groupby(["condition_id", "model"]).rep.nunique()
    assert len(runs) == len(CONDITIONS) * len(MODELS) * N_REPS, f"unexpected row count {len(runs)}"
    assert (counts == N_REPS).all() and set(runs.condition_id) == set(CONDITIONS), "not 7 conditions x 30 reps"
    assert not runs.duplicated(["condition_id", "rep", "model"]).any()
    recomputed = runs.groupby(["condition_id", "model"]).rmse.agg(["mean", "std"])
    s = summary.set_index(["condition_id", "model"])
    assert np.allclose(recomputed.loc[s.index, "mean"], s.rmse_mean, atol=1e-6), "summary mean != runs"
    assert np.allclose(recomputed.loc[s.index, "std"], s.rmse_sd, atol=1e-6), "summary SD != runs"
    assert (s.n_reps == N_REPS).all()
    return d


def stat(summary: pd.DataFrame, cid: str, model: str, col: str) -> float:
    return float(summary.loc[(summary.condition_id == cid) & (summary.model == model), col].iloc[0])


def pm(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def pct(v: float) -> str:
    return f"{v:+.2f}%" if abs(v) < 0.1 else f"{v:+.1f}%"


def pfmt(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def condition_label(row) -> str:
    return f"n={int(row.n):,} / p={row.treat_prob:g} / {row.effect}"


def check_pct(cmp_row: pd.Series, before: float, after: float) -> float:
    expected = (after / before - 1) * 100
    assert abs(expected - cmp_row["pct_change"]) < 0.01, f"pct mismatch {cmp_row.to_dict()}"
    return float(cmp_row["pct_change"])


# ---------------------------------------------------------------- rendering (booktabs style)
class Measurer:
    def __init__(self):
        self.fig = plt.figure()
        self.r = self.fig.canvas.get_renderer()

    def width(self, text: str, size: float, weight: str = "normal") -> float:
        t = self.fig.text(0, 0, text, fontsize=size, fontweight=weight)
        w = t.get_window_extent(self.r).width / self.fig.dpi
        t.remove()
        return w

    def wrap(self, text: str, size: float, max_w: float) -> list[str]:
        lines, cur = [], ""
        for word in text.split(" "):
            trial = f"{cur} {word}".strip()
            if cur and self.width(trial, size) > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        return lines + [cur]


def render(stem: str, title: str, subtitle: str, header: list[str], rows: list, col_w: list[float],
           notes: list[str], align: list[str] | None = None, size: float = 8.5) -> list[str]:
    """rows: list of lists of cells; a cell is str or (str, "bold"); a row may be ("panel", text)."""
    m = Measurer()
    W, pad = sum(col_w), 0.08
    align = align or ["left"] + ["center"] * (len(header) - 1)
    line_h = size / 72 * 1.55
    note_size = size - 1.3
    ops, y = [], 0.0

    def add(kind, **kw):
        ops.append((kind, y, kw))

    for ln in m.wrap(title, size + 2.5, W):
        y += (size + 2.5) / 72 * 1.45
        add("text", x=0, s=ln, size=size + 2.5, weight="bold", ha="left")
    if subtitle:
        for ln in m.wrap(subtitle, size, W):
            y += line_h
            add("text", x=0, s=ln, size=size, color=MUTED, ha="left")
    y += 0.10
    add("rule", lw=1.2)
    n_head = max(h.count("\n") + 1 for h in header)
    y += 0.06 + n_head * line_h
    xs = np.cumsum([0] + col_w)
    for i, h in enumerate(header):
        x = xs[i] + pad if align[i] == "left" else (xs[i] + xs[i + 1]) / 2
        add("text", x=x, s=h, size=size, weight="bold", ha=align[i], va="baseline", ma=align[i])
    y += 0.08
    add("rule", lw=0.6)
    for row in rows:
        if isinstance(row, tuple) and row[0] == "panel":
            if ops[-1][0] != "rule":
                y += 0.04
                add("rule", lw=0.4, color="#999999")
            y += line_h + 0.03
            add("text", x=pad, s=row[1], size=size, weight="bold", ha="left", color=INK)
            y += 0.03
            add("rule", lw=0.4, color="#999999")
            continue
        y += line_h + 0.03
        for i, cell in enumerate(row):
            text, weight = cell if isinstance(cell, tuple) else (cell, "normal")
            x = xs[i] + pad if align[i] == "left" else (xs[i] + xs[i + 1]) / 2
            add("text", x=x, s=text, size=size, weight=weight, ha=align[i])
    y += 0.09
    add("rule", lw=1.2)
    y += 0.04
    for note in notes:
        for ln in m.wrap(note, note_size, W):
            y += note_size / 72 * 1.5
            add("text", x=0, s=ln, size=note_size, color=MUTED, ha="left")
    plt.close(m.fig)

    M = 0.15  # page margin (inches) on every side
    H = y + 0.05
    fig = plt.figure(figsize=(W + 2 * M, H + 2 * M))
    tr = matplotlib.transforms.Affine2D().translate(M, M) + fig.dpi_scale_trans
    for kind, yy, kw in ops:
        if kind == "rule":
            fig.add_artist(plt.Line2D([0, W], [H - yy, H - yy], transform=tr, lw=kw["lw"], color=kw.get("color", INK)))
        else:
            fig.text(kw["x"], H - yy, kw["s"], transform=tr, fontsize=kw["size"], fontweight=kw.get("weight", "normal"),
                     color=kw.get("color", INK), ha=kw["ha"], va="baseline", multialignment=kw.get("ma", kw["ha"]))
    names = []
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        names.append(path.name)
    plt.close(fig)
    return names


# ---------------------------------------------------------------- tables
def bold_min(values: dict[str, float]) -> str:
    return min(values, key=values.get)


def table01(d) -> list[str]:
    s = d["summary"]
    labels = s.drop_duplicates("condition_id").set_index("condition_id").loc[CONDITIONS]
    rows, recs = [], []
    for cid in CONDITIONS:
        means = {m: stat(s, cid, m, "rmse_mean") for m in MODELS}
        best = bold_min(means)
        r = labels.loc[cid]
        cells = [f"{cid}{'†' if cid == 'c05' else ''}", f"{int(r.n):,}", f"{r.treat_prob:g}", r.effect]
        for m in MODELS:
            txt = pm(means[m], stat(s, cid, m, "rmse_sd"))
            cells.append((txt, "bold") if m == best else txt)
        rows.append(cells)
        recs.append({"condition_id": cid, "n": int(r.n), "treat_prob": r.treat_prob, "effect": r.effect, "best_model": best,
                     **{f"{m}_{k}": stat(s, cid, m, f"rmse_{k}") for m in MODELS for k in ("mean", "sd")}})
    pd.DataFrame(recs).to_csv(OUT_DIR / "table01_overall_rmse.csv", index=False, float_format="%.6f")
    p = paired_row(d, "c05", "X-Learner", "Causal Forest")
    return render(
        "table01_overall_rmse", "Table 1. 전체 실험 조건별 평균 RMSE (Overall CATE RMSE by condition)",
        "OFAT 설계 7개 조건 × 4개 모델, 평균 ± 표준편차",
        ["조건", "n", "처치 비율 p", "효과 구조", "T-Learner", "X-Learner", "Causal Forest", "Constant ATE"],
        rows, [0.55, 0.7, 0.75, 0.85, 1.15, 1.15, 1.15, 1.15],
        [METRIC_NOTE, SD_NOTE, "굵은 글씨: 해당 조건에서 평균 RMSE가 가장 낮은 모델(평균값 기준 표시이며 통계적 우위를 뜻하지 않음).",
         f"† c05(비선형): X-Learner와 Causal Forest의 평균 RMSE 차이는 통계적으로 유의하지 않음 "
         f"(대응 차이 {p.diff_mean:+.4f}, 95% CI [{p.diff_ci_low:.4f}, {p.diff_ci_high:.4f}], Holm 보정 p = {p.p_holm:.4f}; "
         "보정 범위: 조건 내 6개 모델 쌍)."])


def change_table(d, stem, title, subtitle, ids, col_names, cmp_df, cmp_specs, p_col, scope, extra_notes, chg_w=0.85):
    """cmp_specs: list of (header, comparison label in CSV, from_cid, to_cid)."""
    s = d["summary"]
    rows, recs = [], []
    best = {cid: bold_min({m: stat(s, cid, m, "rmse_mean") for m in MODELS}) for cid in ids}
    for m in MODELS:
        cells, rec = [m], {"model": m}
        for cid, name in zip(ids, col_names):
            mean, sd = stat(s, cid, m, "rmse_mean"), stat(s, cid, m, "rmse_sd")
            cells.append((pm(mean, sd), "bold") if best[cid] == m else pm(mean, sd))
            rec[f"{name}_mean"], rec[f"{name}_sd"] = mean, sd
        for head, label, a, b in cmp_specs:
            row = cmp_df[(cmp_df.model == m) & (cmp_df.comparison == label)].iloc[0]
            v = check_pct(row, stat(s, a, m, "rmse_mean"), stat(s, b, m, "rmse_mean"))
            sig = row[p_col] < ALPHA
            assert sig == (row.diff_ci_high < 0 or row.diff_ci_low > 0) or not sig, f"CI/p disagree {label} {m}"
            cells.append(pct(v) + ("" if sig else " ns"))
            key = head.replace("\n", " ")
            rec[f"pct_change {key}"], rec[f"{p_col} {key}"] = v, row[p_col]
        rows.append(cells)
        recs.append(rec)
    pd.DataFrame(recs).to_csv(OUT_DIR / f"{stem}.csv", index=False, float_format="%.6g")
    header = ["모델"] + col_names + [h for h, *_ in cmp_specs]
    col_w = [1.05] + [1.15] * len(ids) + [chg_w] * len(cmp_specs)
    notes = [METRIC_NOTE, SD_NOTE,
             "변화율 = (비교 조건 평균 RMSE / 기준 조건 평균 RMSE - 1) × 100. 양수는 오차 증가(악화), 음수는 오차 감소(개선).",
             f"ns: 같은 반복 번호끼리 짝지은 대응 t검정에서 Holm 보정 p ≥ 0.05 (보정 범위: {scope}). 표시가 없으면 p < 0.05.",
             "굵은 글씨: 해당 열에서 평균 RMSE가 가장 낮은 모델."] + extra_notes
    return render(stem, title, subtitle, header, rows, col_w, notes)


def table02(d):
    c = d["sample_size_comparison"]
    return change_table(
        d, "table02_sample_size", "Table 2. 표본 크기별 성능 비교 (CATE RMSE by sample size)",
        "고정 조건: 처치 비율 p = 0.5, 선형(linear) 효과 · 조건 c00 / c01 / c02",
        ["c00", "c01", "c02"], ["n = 1,000", "n = 10,000", "n = 100,000"], c,
        [("변화율\n1k→10k", "n=10,000 - n=1,000", "c00", "c01"),
         ("변화율\n10k→100k", "n=100,000 - n=10,000", "c01", "c02"),
         ("변화율\n1k→100k", "n=100,000 - n=1,000", "c00", "c02")],
        "p_holm_12", "표본 크기 비교 12개(4개 모델 × 3개 쌍)", [])


def table03(d):
    c = d["treat_ratio_comparison"]
    return change_table(
        d, "table03_treatment_ratio", "Table 3. 처치군 불균형 비교 (CATE RMSE by treatment ratio)",
        "고정 조건: n = 10,000, 선형(linear) 효과 · 조건 c01 / c03 / c04",
        ["c01", "c03", "c04"], ["p = 0.5 (50%)", "p = 0.3 (30%)", "p = 0.1 (10%)"], c,
        [("변화율\n0.5→0.3", "c03(p=0.3) - c01(p=0.5)", "c01", "c03"),
         ("변화율\n0.5→0.1", "c04(p=0.1) - c01(p=0.5)", "c01", "c04")],
        "p_holm_8", "처치 비율 비교 8개(4개 모델 × 2개 쌍)",
        ["p = 0.1에서 T-Learner의 평균 RMSE(0.3195)는 Constant ATE(0.3018)보다 높았다."
         if stat(d["summary"], "c04", "T-Learner", "rmse_mean") > stat(d["summary"], "c04", "Constant ATE", "rmse_mean") else ""])


def table04(d):
    c = d["complexity_comparison"]
    p = paired_row(d, "c05", "X-Learner", "Causal Forest")
    assert p.p_holm >= ALPHA
    return change_table(
        d, "table04_effect_complexity", "Table 4. 처치효과 복잡성 비교 (CATE RMSE by effect complexity)",
        "고정 조건: n = 10,000, 처치 비율 p = 0.5 · 조건 c01 / c05 / c06",
        ["c01", "c05", "c06"], ["linear", "nonlinear†", "interaction"], c,
        [("변화율\nlinear→nonlinear", "c05(nonlinear) - c01(linear)", "c01", "c05"),
         ("변화율\nlinear→interaction", "c06(interaction) - c01(linear)", "c01", "c06")],
        "p_holm_8", "효과 구조 비교 8개(4개 모델 × 2개 쌍)",
        [f"† nonlinear 조건에서 X-Learner와 Causal Forest의 평균 RMSE 차이는 통계적으로 유의하지 않음 "
         f"(X - CF = {p.diff_mean:+.4f}, 95% CI [{p.diff_ci_low:.4f}, {p.diff_ci_high:.4f}], Holm 보정 p = {p.p_holm:.4f}; "
         "보정 범위: 조건 내 6개 모델 쌍). 굵은 표시는 평균값 기준일 뿐이다."], chg_w=1.3)


def table05(d):
    s = d["summary"]
    labels = s.drop_duplicates("condition_id").set_index("condition_id").loc[CONDITIONS]
    rows, recs = [("panel", "Panel A. 반복 간 RMSE 표준편차 (SD across 30 replications; 낮을수록 안정적)")], []
    for cid in CONDITIONS:
        rows.append([f"{cid}  {condition_label(labels.loc[cid])}"] + [f"{stat(s, cid, m, 'rmse_sd'):.4f}" for m in LEARNERS])
    rows.append(("panel", "Panel B. 평균 RMSE의 95% 신뢰구간 (95% CI of the mean RMSE)"))
    for cid in CONDITIONS:
        rows.append([f"{cid}  {condition_label(labels.loc[cid])}"]
                    + [f"[{stat(s, cid, m, 'rmse_ci_low'):.4f}, {stat(s, cid, m, 'rmse_ci_high'):.4f}]" for m in LEARNERS])
    for cid in CONDITIONS:
        for m in LEARNERS:
            recs.append({"condition_id": cid, "condition": condition_label(labels.loc[cid]), "model": m,
                         **{k: stat(s, cid, m, k) for k in ("rmse_mean", "rmse_sd", "rmse_ci_low", "rmse_ci_high", "rmse_ci_halfwidth")}})
    pd.DataFrame(recs).to_csv(OUT_DIR / "table05_model_stability.csv", index=False, float_format="%.6f")
    return render(
        "table05_model_stability", "Table 5. 모델 안정성 비교 (Model stability across replications)",
        "Panel A는 개별 반복 성능의 흩어짐(SD), Panel B는 평균 RMSE 추정의 불확실성(95% CI)으로 서로 다른 양이다.",
        ["조건", "T-Learner", "X-Learner", "Causal Forest"], rows, [2.6, 1.45, 1.45, 1.45],
        [METRIC_NOTE,
         "Panel A: 30회 반복 RMSE의 표본 표준편차(ddof = 1). 한 번의 실험 결과가 반복마다 얼마나 달라지는지를 나타낸다.",
         "Panel B: 평균 RMSE ± t(0.975, 29) × SD / √30. 30회 평균값 자체의 불확실성이며 개별 반복의 범위가 아니다.",
         "n = 100,000 조건의 테스트 세트(20,000개)는 n = 1,000 조건(200개)보다 커서, SD 감소의 일부는 RMSE 측정 잡음 감소에서 온다."])


def paired_row(d, cid, a, b) -> pd.Series:
    p = d["paired"]
    r = p[(p.condition_id == cid) & (p.model_a == a) & (p.model_b == b)]
    assert len(r) == 1, (cid, a, b)
    return r.iloc[0]


def table06(d):
    s = d["summary"]
    labels = s.drop_duplicates("condition_id").set_index("condition_id")
    specs = [(cid, "X-Learner", "Causal Forest") for cid in CONDITIONS]
    specs += [("c04", "T-Learner", "Constant ATE"), ("c05", "T-Learner", "Constant ATE")]
    rows, recs = [], []
    for cid, a, b in specs:
        r = paired_row(d, cid, a, b)
        ci_excl = r.diff_ci_high < 0 or r.diff_ci_low > 0
        sig = r.p_holm < ALPHA
        if sig:
            assert ci_excl, f"Holm-significant but CI includes 0: {cid} {a} {b}"
            verdict = f"{SHORT[a] if r.diff_mean < 0 else SHORT[b]} 우세"
        else:
            verdict = "유의하지 않음"
        rows.append([f"{cid}  {condition_label(labels.loc[cid])}", f"{SHORT[a]} - {SHORT[b]}", f"{r.diff_mean:+.4f}",
                     f"[{r.diff_ci_low:+.4f}, {r.diff_ci_high:+.4f}]", pfmt(r.p_holm), f"{int(r.a_better_reps)}/{int(r.n_pairs)}",
                     verdict if sig else (verdict, "bold")])
        recs.append({"condition_id": cid, "condition": condition_label(labels.loc[cid]), "model_a": a, "model_b": b,
                     "diff_mean": r.diff_mean, "diff_ci_low": r.diff_ci_low, "diff_ci_high": r.diff_ci_high,
                     "p_value": r.p_value, "p_holm": r.p_holm, "a_better_reps": int(r.a_better_reps), "n_pairs": int(r.n_pairs),
                     "significant_holm_0.05": sig, "ci_excludes_0": ci_excl, "verdict": verdict})
    pd.DataFrame(recs).to_csv(OUT_DIR / "table06_statistical_comparison.csv", index=False, float_format="%.6g")
    return render(
        "table06_statistical_comparison", "Table 6. 모델 간 통계적 비교 (Paired comparison of RMSE)",
        "핵심 비교 선별: 모든 조건의 X-Learner vs Causal Forest, Constant ATE 대비 T-Learner가 우위를 보이지 못한 두 조건",
        ["조건", "비교 (A - B)", "평균 차이", "95% CI", "Holm p", "A 우세 반복", "판정"], rows,
        [2.15, 1.25, 0.8, 1.45, 0.75, 0.8, 1.05],
        ["평균 차이 = RMSE(A) - RMSE(B)를 같은 반복 번호(동일 데이터·분할)끼리 짝지어 30쌍에 대해 평균한 값. 음수면 A의 오차가 더 작다.",
         "95% CI: 대응 차이의 t 신뢰구간(다중비교 미보정). Holm p: 대응 t검정 p값을 조건 내 6개 모델 쌍(4개 모델의 모든 쌍)에 대해 Holm 보정한 값.",
         "판정: Holm p < 0.05이면 평균 차이 방향의 모델 우세로, 그렇지 않으면 '유의하지 않음'(굵게)으로 표시. A 우세 반복: 30회 중 A의 RMSE가 더 낮았던 횟수.",
         "Holm 보정은 각 조건 내 비교에만 적용되며, 7개 조건 전체나 조건 간 비교를 아우르는 보정은 아니다. "
         "결과는 고정된 하이퍼파라미터와 본 연구의 합성 데이터 생성 과정에 한정된다.",
         METRIC_NOTE])


def main() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in KOREAN_FONTS if f in available), None)
    if font is None:
        raise RuntimeError(f"No Korean font found among {KOREAN_FONTS}")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "pdf.fonttype": 42})
    d = load()
    written = []
    for fn in (table01, table02, table03, table04, table05, table06):
        written += fn(d)
    print(f"font: {font}")
    print("verified: 840 rows, 7 conditions x 4 models x 30 reps; summary matches runs; pct and significance cross-checked")
    print("written:", ", ".join(written))


if __name__ == "__main__":
    main()
